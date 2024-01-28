//go:build ignore

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

var unroll = 6

func main() {
	TEXT("denseDot3", NOSPLIT,
		"func(n int, v0 []float32, v0Base int, v1 []float32, v1Base int, v2 []float32, v2Base int) float32")

	n := Load(Param("n"), GP64())

	v0 := Mem{Base: Load(Param("v0").Base(), GP64())}
	v0Base := Load(Param("v0Base"), GP64())
	ADDQ(v0Base, v0.Base)

	v1 := Mem{Base: Load(Param("v1").Base(), GP64())}
	v1Base := Load(Param("v1Base"), GP64())
	ADDQ(v1Base, v1.Base)

	v2 := Mem{Base: Load(Param("v2").Base(), GP64())}
	v2Base := Load(Param("v2Base"), GP64())
	ADDQ(v2Base, v2.Base)

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
	Label("blockloop")
	CMPQ(n, U32(blockitems))
	JL(LabelRef("tail"))

	xs := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		xs[i] = YMM()
	}

	// The gradient computation --- put the output into acc[i]
	for i := 0; i < unroll; i++ {
		VMOVUPS(v0.Offset(32*i), xs[i])
	}
	for i := 0; i < unroll; i++ {
		VMULPS(v1.Offset(32*i), xs[i], xs[i])
	}
	for i := 0; i < unroll; i++ {
		VFMADD231PS(v2.Offset(32*i), xs[i], acc[i])
	}

	ADDQ(U32(blocksize), v0.Base)
	ADDQ(U32(blocksize), v1.Base)
	ADDQ(U32(blocksize), v2.Base)

	SUBQ(U32(blockitems), n)

	JMP(LabelRef("blockloop"))

	// Process any trailing entries.
	Label("tail")
	tailAccumulator := XMM()
	VXORPS(tailAccumulator, tailAccumulator, tailAccumulator)

	Label("tailloop")
	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	tailElement := XMM()
	VMOVSS(v0, tailElement)
	VMULSS(v1, tailElement, tailElement)
	VFMADD231SS(v2, tailElement, tailAccumulator)

	ADDQ(U32(4), v0.Base)
	ADDQ(U32(4), v1.Base)
	ADDQ(U32(4), v2.Base)
	DECQ(n)
	JMP(LabelRef("tailloop"))

	// Reduce the lanes to one.
	Label("reduce")
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
	Store(result, ReturnIndex(0))

	RET()

	Generate()
}
