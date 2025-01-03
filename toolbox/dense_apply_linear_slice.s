// Code generated by command: go run asm_dense_apply_linear_slice.go -out dense_apply_linear_slice.s -stubs stub_dense_apply_linear_slice.go. DO NOT EDIT.

#include "textflag.h"

// func denseApplyLinearSliceKernel(outputSize int, inputSize int, x []float32, w []float32, b []float32, z []float32, kStart int, iStart int, elementsToCompute int)
// Requires: AVX, FMA3
TEXT ·denseApplyLinearSliceKernel(SB), NOSPLIT, $0-136
	MOVQ outputSize+0(FP), AX
	MOVQ inputSize+8(FP), CX
	MOVQ x_base+16(FP), DX
	MOVQ w_base+40(FP), BX
	MOVQ b_base+64(FP), SI
	MOVQ kStart+112(FP), DI
	MOVQ iStart+120(FP), R8
	MOVQ elementsToCompute+128(FP), R9

	// Set up the output pointer according to jStart and kStart
	MOVQ z_base+88(FP), R10

	// The offset to get to the correct row of z
	MOVQ  $0x00000004, R11
	IMULQ CX, R11
	IMULQ DI, R11

	// The offset to get to the correct column of z
	MOVQ  $0x00000004, R12
	IMULQ R8, R12
	ADDQ  R11, R10
	ADDQ  R12, R10

	// The outer loop, running over elements of z in-order
outerLoop:
	CMPQ R9, $0x00000000
	JE   outerLoopExit

	// Make copies of xPtr, wPtr pointing at the correct rows for use by the dot product
	MOVQ  $0x00000004, R11
	IMULQ CX, R11
	IMULQ DI, R11
	ADDQ  DX, R11
	MOVQ  $0x00000004, R12
	IMULQ CX, R12
	IMULQ R8, R12
	ADDQ  BX, R12

	// Compute dot product of w and x
	MOVQ   CX, R13
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5

dotproductblockloop:
	CMPQ        R13, $0x00000030
	JL          dotproducttail
	VMOVUPS     (R11), Y6
	VMOVUPS     32(R11), Y7
	VMOVUPS     64(R11), Y8
	VMOVUPS     96(R11), Y9
	VMOVUPS     128(R11), Y10
	VMOVUPS     160(R11), Y11
	VFMADD231PS (R12), Y6, Y0
	VFMADD231PS 32(R12), Y7, Y1
	VFMADD231PS 64(R12), Y8, Y2
	VFMADD231PS 96(R12), Y9, Y3
	VFMADD231PS 128(R12), Y10, Y4
	VFMADD231PS 160(R12), Y11, Y5
	ADDQ        $0x000000c0, R11
	ADDQ        $0x000000c0, R12
	SUBQ        $0x00000030, R13
	JMP         dotproductblockloop

dotproducttail:
	VXORPS X6, X6, X6

dotproducttailloop:
	CMPQ        R13, $0x00000000
	JE          dotproductreduce
	VMOVSS      (R11), X7
	VFMADD231SS (R12), X7, X6
	ADDQ        $0x00000004, R11
	ADDQ        $0x00000004, R12
	DECQ        R13
	JMP         dotproducttailloop

dotproductreduce:
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

	// Add b
	MOVQ   R8, R11
	SHLQ   $0x02, R11
	ADDQ   SI, R11
	VADDSS (R11), X0, X0

	// Write linear activation to z
	VEXTRACTPS $0x00, X0, (R10)
	ADDQ       $0x00000004, R10

	// Increment k and i
	INCQ R8
	CMPQ R8, AX
	JNE  afterKI
	MOVQ $0x00000000, R8
	INCQ DI

afterKI:
	DECQ R9
	JMP  outerLoop

outerLoopExit:
	RET
