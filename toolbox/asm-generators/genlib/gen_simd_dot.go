package genlib

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

// GenSIMDDot3 emits a 3-way dot-product.
func GenSIMDDot3(n Register, v0Ptr, v1Ptr, v2Ptr Register, unroll int) Register {
	// Allocate accumulation registers.
	acc := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		acc[i] = YMM()
	}

	// Zero initialization.
	for i := 0; i < unroll; i++ {
		VXORPS(acc[i], acc[i], acc[i])
	}

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
		VMOVUPS(Mem{Base: v0Ptr}.Offset(32*i), xs[i])
	}
	for i := 0; i < unroll; i++ {
		VMULPS(Mem{Base: v1Ptr}.Offset(32*i), xs[i], xs[i])
	}
	for i := 0; i < unroll; i++ {
		VFMADD231PS(Mem{Base: v2Ptr}.Offset(32*i), xs[i], acc[i])
	}

	ADDQ(U32(blocksize), v0Ptr)
	ADDQ(U32(blocksize), v1Ptr)
	ADDQ(U32(blocksize), v2Ptr)

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
	VMOVSS(Mem{Base: v0Ptr}, tailElement)
	VMULSS(Mem{Base: v1Ptr}, tailElement, tailElement)
	VFMADD231SS(Mem{Base: v2Ptr}, tailElement, tailAccumulator)

	ADDQ(U32(4), v0Ptr)
	ADDQ(U32(4), v1Ptr)
	ADDQ(U32(4), v2Ptr)
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

	return result
}

// GenSIMDDot3 emits a 3-way dot-product.
func GenSIMDDot2(n Register, v0Ptr, v1Ptr Register, unroll int) Register {
	// Allocate accumulation registers.
	acc := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		acc[i] = YMM()
	}

	// Zero initialization.
	for i := 0; i < unroll; i++ {
		VXORPS(acc[i], acc[i], acc[i])
	}

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
		VMOVUPS(Mem{Base: v0Ptr}.Offset(32*i), xs[i])
	}
	for i := 0; i < unroll; i++ {
		VFMADD231PS(Mem{Base: v1Ptr}.Offset(32*i), xs[i], acc[i])
	}

	ADDQ(U32(blocksize), v0Ptr)
	ADDQ(U32(blocksize), v1Ptr)

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

	return result
}
