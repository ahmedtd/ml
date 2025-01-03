package main

import (
	"github.com/ahmedtd/ml/toolbox/asm-generators/genlib"
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
)

func main() {
	TEXT("denseApplyLinearSliceKernel", NOSPLIT,
		"func(outputSize int, inputSize int, x []float32, w []float32, b []float32, z []float32, kStart int, iStart int, elementsToCompute int)")

	outputSize := Load(Param("outputSize"), GP64())
	inputSize := Load(Param("inputSize"), GP64())

	xPtr := Load(Param("x").Base(), GP64())
	wPtr := Load(Param("w").Base(), GP64())
	bPtr := Load(Param("b").Base(), GP64())

	kReg := Load(Param("kStart"), GP64())
	iReg := Load(Param("iStart"), GP64())

	elementsLeft := Load(Param("elementsToCompute"), GP64())

	Comment("Set up the output pointer according to jStart and kStart")
	zPtr := Load(Param("z").Base(), GP64())
	Comment("The offset to get to the correct row of z")
	kStartBytesReg := GP64()
	MOVQ(U32(4), kStartBytesReg)
	IMULQ(inputSize, kStartBytesReg)
	IMULQ(kReg, kStartBytesReg)
	Comment("The offset to get to the correct column of z")
	iStartBytesReg := GP64()
	MOVQ(U32(4), iStartBytesReg)
	IMULQ(iReg, iStartBytesReg)
	ADDQ(kStartBytesReg, zPtr)
	ADDQ(iStartBytesReg, zPtr)

	Comment("The outer loop, running over elements of z in-order")

	Label("outerLoop")
	CMPQ(elementsLeft, U32(0))
	JE(LabelRef("outerLoopExit"))

	Comment("Make copies of xPtr, wPtr pointing at the correct rows for use by the dot product")
	xDotPtr := GP64()
	MOVQ(U32(4), xDotPtr)
	IMULQ(inputSize, xDotPtr)
	IMULQ(kReg, xDotPtr)
	ADDQ(xPtr, xDotPtr)

	wDotPtr := GP64()
	MOVQ(U32(4), wDotPtr)
	IMULQ(inputSize, wDotPtr)
	IMULQ(iReg, wDotPtr)
	ADDQ(wPtr, wDotPtr)

	Comment("Compute dot product of w and x")
	n := GP64()
	MOVQ(inputSize, n)
	zReg := genlib.GenSIMDDot2(n, xDotPtr, wDotPtr, 6)

	Comment("Add b")
	bEltPtr := GP64()
	MOVQ(iReg, bEltPtr)
	SHLQ(U8(2), bEltPtr)
	ADDQ(bPtr, bEltPtr)
	VADDSS(Mem{Base: bEltPtr}, zReg, zReg)

	Comment("Write linear activation to z")
	VEXTRACTPS(U8(0), zReg, Mem{Base: zPtr})

	ADDQ(U32(4), zPtr)

	Comment("Increment k and i")
	INCQ(iReg)
	CMPQ(iReg, outputSize)
	JNE(LabelRef("afterKI"))
	MOVQ(U32(0), iReg)
	INCQ(kReg)
	Label("afterKI")

	DECQ(elementsLeft)

	JMP(LabelRef("outerLoop"))
	Label("outerLoopExit")

	RET()

	Generate()
}
