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
    test_arg_kernel(backend, WORKGROUP_SIZE)(x, arg; ndrange=size(x))
end