package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

const unroll = 6

func main() {
	TEXT("denseApplyReluActivateSliceKernel", NOSPLIT,
		"func(a []float32, dadz []float32, iStart int, elementsToCompute int)")

	bytes := GLOBL("bytes", RODATA|NOPTR)
	DATA(16, F32(1.0))

	iReg := Load(Param("iStart"), GP64())
	elementsLeft := Load(Param("elementsToCompute"), GP64())

	aPtr := Load(Param("a").Base(), GP64())
	aStartBytesReg := GP64()
	MOVQ(iReg, aStartBytesReg)
	SHLQ(U8(2), aStartBytesReg)
	ADDQ(aStartBytesReg, aPtr)

	dadzPtr := Load(Param("a").Base(), GP64())
	dadzStartBytesReg := GP64()
	MOVQ(iReg, dadzStartBytesReg)
	SHLQ(U8(2), dadzStartBytesReg)
	ADDQ(dadzStartBytesReg, dadzPtr)

	// Loop over blocks and process them with vector instructions.
	blockitems := 8 * unroll
	blocksize := 4 * blockitems

	Comment("Load an immediate 1.0 into all 8 slots of a YMM register")
	onesReg := YMM()
	VBROADCASTSS(bytes, onesReg)

	Label("dotproductblockloop")
	CMPQ(elementsLeft, U32(blockitems))
	JL(LabelRef("dotproducttail"))

	zeros := YMM()
	VXORPS(zeros, zeros, zeros)

	Comment("The relu op")
	xs := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		xs[i] = YMM()
	}
	grads := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		grads[i] = YMM()
	}
	for i := 0; i < unroll; i++ {
		// Get the linear activation.
		VMOVUPS(Mem{Base: aPtr}.Offset(32*i), xs[i])
	}
	for i := 0; i < unroll; i++ {
		// xs[i] >= 0.0.  grads[i] contains a 32-bit all-one mask where this is
		// true.
		//
		// 0x0d -> greater than / equal
		VCMPPS(U8(0x0d), xs[i], zeros, grads[i])
	}
	for i := 0; i < unroll; i++ {
		// Keep only entries that are greater than 0
		VANDPS(xs[i], grads[i], xs[i])
	}
	for i := 0; i < unroll; i++ {
		// Write back activation values
		VMOVUPS(xs[i], Mem{Base: aPtr}.Offset(32*i))
	}
	for i := 0; i < unroll; i++ {
		// Gradient is one where the linear input was >= 0.
		VANDPS(grads[i], onesReg, grads[i])
	}
	for i := 0; i < unroll; i++ {
		// Write back gradient values.
		VMOVUPS(grads[i], Mem{Base: dadzPtr}.Offset(32*i))
	}

	ADDQ(U32(blocksize), aPtr)
	ADDQ(U32(blocksize), dadzPtr)

	SUBQ(U32(blockitems), elementsLeft)

	JMP(LabelRef("dotproductblockloop"))

	// TODO: Need to implement the trailing entries.

	// Process any trailing entries.
	Label("dotproducttail")
	tailAccumulator := XMM()
	VXORPS(tailAccumulator, tailAccumulator, tailAccumulator)

	Label("dotproducttailloop")
	CMPQ(n, U32(0))
	JE(LabelRef("dotproductreduce"))

	tailElement := XMM()
	VMOVSS(Mem{Base: v0Ptr}, tailElement)
	VFMADD231SS(Mem{Base: v1Ptr}, tailElement, tailAccumulator)

	ADDQ(U32(4), v0Ptr)
	ADDQ(U32(4), v1Ptr)
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

	RET()

	Generate()
}
