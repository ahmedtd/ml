// Code generated by command: go run asm_dense_apply_linear_slice.go -out dense_apply_linear_slice.s -stubs stub_dense_apply_linear_slice.go. DO NOT EDIT.

package toolbox

func denseApplyLinearSliceKernel(outputSize int, inputSize int, x []float32, w []float32, b []float32, z []float32, kStart int, iStart int, elementsToCompute int)
