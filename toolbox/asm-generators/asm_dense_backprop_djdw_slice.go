package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

var unroll = 6

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

	// Allocate accumulation registers.
	acc := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		acc[i] = YMM()
	}

	// Zero initialization.
	for i := 0; i < unroll; i++ {
		VXORPS(acc[i], acc[i], acc[i])
	}

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

	Comment("The inner loop, running over batchSize elements of djdaT, dadzT, and xT in-order")

	n := GP64()
	MOVQ(batchSize, n)

	// Loop over blocks and process them with vector instructions.
	blockitems := 8 * unroll
	blocksize := 4 * blockitems

	Label("dotproductblockloop")
	CMPQ(n, U32(blockitems))
	JL(LabelRef("dotproducttail"))

	xs := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		xs[i] = YMM()
	}

	// The gradient computation --- put the output into acc[i]
	for i := 0; i < unroll; i++ {
		VMOVUPS(Mem{Base: djdaTDotPtr}.Offset(32*i), xs[i])
	}
	for i := 0; i < unroll; i++ {
		VMULPS(Mem{Base: dadzTDotPtr}.Offset(32*i), xs[i], xs[i])
	}
	for i := 0; i < unroll; i++ {
		VFMADD231PS(Mem{Base: xTDotPtr}.Offset(32*i), xs[i], acc[i])
	}

	ADDQ(U32(blocksize), djdaTDotPtr)
	ADDQ(U32(blocksize), dadzTDotPtr)
	ADDQ(U32(blocksize), xTDotPtr)

	SUBQ(U32(blockitems), n)

	JMP(LabelRef("dotproductblockloop"))

	// Process any trailing entries.
	Label("dotproducttail")
	tailAccumulator := XMM()
	VXORPS(tailAccumulator, tailAccumulator, tailAccumulator)

	Label("dotproducttailloop")
	CMPQ(n, U32(0))
	JE(LabelRef("dotproductreduce"))

	tailElement := XMM()
	VMOVSS(Mem{Base: dadzTDotPtr}, tailElement)
	VMULSS(Mem{Base: djdaTDotPtr}, tailElement, tailElement)
	VFMADD231SS(Mem{Base: xTDotPtr}, tailElement, tailAccumulator)

	ADDQ(U32(4), dadzTDotPtr)
	ADDQ(U32(4), djdaTDotPtr)
	ADDQ(U32(4), xTDotPtr)
	DECQ(n)
	JMP(LabelRef("dotproducttailloop"))

	// Reduce the lanes to one.
	Label("dotproductreduce")
	for i := 1; i < unroll; i++ {
		VADDPS(acc[0], acc[i], acc[0])
	}

	result := acc[0].AsX()
	top := XMM()
	VEXTRACTF128(U8(1), acc[0], top)
	VADDPS(result, top, result)
	VADDPS(result, tailAccumulator, result)
	VHADDPS(result, result, result)
	VHADDPS(result, result, result)
	VEXTRACTPS(U8(0), result, Mem{Base: djdwPtr})

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

// // genSIMDDot3 emits a 3-way dot-product.
// func genSIMDDot3(n Register, v0, v1, v2 Register) Register {

// }
