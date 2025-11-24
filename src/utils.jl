@kernel function test_arg_kernel(x, arg)
    I = @index(Global)
    x[I] += 1
end

"""
    This function helps assessing if a structure (passed as `arg`) is GPU-compatible.
    The function will error if it's not the case
"""
function test_arg(backend, arg)
    x = KernelAbstractions.zeros(backend, Float32, 10)
    test_arg_kernel(backend, WORKGROUP_SIZE)(x, arg; ndrange = size(x))
end

function test_arg(backend::Bcube.BcubeBackendCPUSerial, arg)
    return true
end

function test_arg_AK(i, x, arg)
    x[i] += 1
end

"""
    This function helps assessing if a structure (passed as `arg`) is GPU-compatible
    by passing it as an argument to the function `AcceleratedKernels.foreachindex`.
    The function will error if it's not the case.
"""
function test_arg_AK_foreachindex(backend, arg)
    x = KernelAbstractions.zeros(backend, Float32, 10)
    f(i) = test_arg_AK(i, x, arg)
    AK.foreachindex(f, x)
end
