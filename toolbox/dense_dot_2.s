// Code generated by command: go run asm_dense_dot_2.go -out dense_dot_2.s -stubs stub_dense_dot_2.go. DO NOT EDIT.

#include "textflag.h"

// func denseDot2(n int, v0 []float32, v0Base int, v1 []float32, v1Base int) float32
// Requires: AVX, FMA3, SSE
TEXT ·denseDot2(SB), NOSPLIT, $0-76
	MOVQ   n+0(FP), AX
	MOVQ   v0_base+8(FP), CX
	MOVQ   v0Base+32(FP), DX
	ADDQ   DX, CX
	MOVQ   v1_base+40(FP), DX
	MOVQ   v1Base+64(FP), BX
	ADDQ   BX, DX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5

blockloop:
	CMPQ        AX, $0x00000030
	JL          tail
	VMOVUPS     (CX), Y6
	VMOVUPS     32(CX), Y7
	VMOVUPS     64(CX), Y8
	VMOVUPS     96(CX), Y9
	VMOVUPS     128(CX), Y10
	VMOVUPS     160(CX), Y11
	VFMADD231PS (DX), Y6, Y0
	VFMADD231PS 32(DX), Y7, Y1
	VFMADD231PS 64(DX), Y8, Y2
	VFMADD231PS 96(DX), Y9, Y3
	VFMADD231PS 128(DX), Y10, Y4
	VFMADD231PS 160(DX), Y11, Y5
	ADDQ        $0x000000c0, CX
	ADDQ        $0x000000c0, DX
	SUBQ        $0x00000030, AX
	JMP         blockloop

tail:
	VXORPS X6, X6, X6

tailloop:
	CMPQ        AX, $0x00000000
	JE          reduce
	VMOVSS      (CX), X7
	VFMADD231SS (DX), X7, X6
	ADDQ        $0x00000004, CX
	ADDQ        $0x00000004, DX
	DECQ        AX
	JMP         tailloop

reduce:
	VADDPS       Y0, Y1, Y0
	VADDPS       Y0, Y2, Y0
	VADDPS       Y0, Y3, Y0
	VADDPS       Y0, Y4, Y0
	VADDPS       Y0, Y5, Y0
	VEXTRACTF128 $0x01, Y0, X1
	VADDPS       X0, X1, X0
	VADDPS       X0, X6, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS        X0, ret+72(FP)
	RET