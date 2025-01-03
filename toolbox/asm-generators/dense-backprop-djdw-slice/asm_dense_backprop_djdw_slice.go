package main

import (
	"github.com/ahmedtd/ml/toolbox/asm-generators/genlib"
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
)

func main() {
	TEXT("denseBackpropDjdwSliceKernel", NOSPLIT,
		"func(batchSize int, inputSize int, djdaT []float32, dadzT []float32, xT []float32, djdw []float32, iStart int, jStart int, elementsToCompute int)")

	batchSize := Load(Param("batchSize"), GP64())

	inputSize := Load(Param("inputSize"), GP64())

	djdaTPtr := Load(Param("djdaT").Base(), GP64())
	dadzTPtr := Load(Param("dadzT").Base(), GP64())
	xTPtr := Load(Param("xT").Base(), GP64())

	elementsLeft := Load(Param("elementsToCompute"), GP64())

	iReg := Load(Param("iStart"), GP64())
	jReg := Load(Param("jStart"), GP64())

	// TODO: This doesn't account for jStart.
	djdwPtr := Load(Param("djdw").Base(), GP64())
	djdwBase := GP64()
	MOVQ(U32(4), djdwBase)
	IMULQ(inputSize, djdwBase)
	IMULQ(iReg, djdwBase)
	ADDQ(djdwBase, djdwPtr)

	Comment("The outer loop, running over elements of djdw in-order")

	Label("outerLoop")
	CMPQ(elementsLeft, U32(0))
	JE(LabelRef("outerLoopExit"))

	Comment("Make copies of djdaPtr, dadzPtr, and xTPtr")

	djdaTDotPtr := GP64()
	MOVQ(U32(4), djdaTDotPtr)
	IMULQ(batchSize, djdaTDotPtr)
	IMULQ(iReg, djdaTDotPtr)
	ADDQ(djdaTPtr, djdaTDotPtr)

	dadzTDotPtr := GP64()
	MOVQ(U32(4), dadzTDotPtr)
	IMULQ(batchSize, dadzTDotPtr)
	IMULQ(iReg, dadzTDotPtr)
	ADDQ(dadzTPtr, dadzTDotPtr)

	xTDotPtr := GP64()
	MOVQ(U32(4), xTDotPtr)
	IMULQ(batchSize, xTDotPtr)
	IMULQ(jReg, xTDotPtr)
	ADDQ(xTPtr, xTDotPtr)

	n := GP64()
	MOVQ(batchSize, n)

	resultReg := genlib.GenSIMDDot3(n, djdaTDotPtr, dadzTDotPtr, xTDotPtr, 6)
	VEXTRACTPS(U8(0), resultReg, Mem{Base: djdwPtr})

	ADDQ(U32(4), djdwPtr)

	INCQ(jReg)
	CMPQ(jReg, inputSize)
	JNE(LabelRef("afterIJ"))
	MOVQ(U32(0), jReg)
	INCQ(iReg)
	Label("afterIJ")

	DECQ(elementsLeft)

	JMP(LabelRef("outerLoop"))
	Label("outerLoopExit")

	RET()

	Generate()
}
