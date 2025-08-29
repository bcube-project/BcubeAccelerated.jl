@kernel function test_arg_kernel(x, arg)
    I = @index(Global)
    x[I] += 1
end

function test_arg(backend, arg)
    x = KernelAbstractions.zeros(backend, Float32, 10)
    test_arg_kernel(backend, WORKGROUP_SIZE)(x, arg; ndrange=size(x))
end