"""
    Bcube.dirichlet_face_offsets(counts, backend::AbstractBcubeBackendAcc{K, B}) where {K, B <: KernelAbstractions.GPU}

Scan the device counts and return their exclusive offsets and scalar total.
"""
function Bcube.dirichlet_face_offsets(
    counts,
    backend::AbstractBcubeBackendAcc{K, B},
) where {K, B <: KernelAbstractions.GPU}
    isempty(counts) && return similar(counts, 0), 0
    execution_backend = get_backend(backend)
    offsets = AK.accumulate(+, counts, execution_backend; init = 0, inclusive = false)
    occurrence_count = AK.reduce(+, counts, execution_backend; init = 0)
    return offsets, occurrence_count
end

"""
    Bcube.unique_dirichlet_indices(occurrences, ndofs, backend::AbstractBcubeBackendAcc{K, B}) where {K, B <: KernelAbstractions.GPU}

Keep the first occurrence of each device dof in encounter order.
"""
function Bcube.unique_dirichlet_indices(
    occurrences,
    ndofs,
    backend::AbstractBcubeBackendAcc{K, B},
) where {K, B <: KernelAbstractions.GPU}
    isempty(occurrences) && return occurrences
    execution_backend = get_backend(backend)

    # Concurrent occurrences retain the lowest position for their global dof.
    first_occurrence = similar(occurrences, Int, ndofs)
    fill!(first_occurrence, typemax(Int))
    AK.foreachindex(occurrences, execution_backend) do occurrence
        @inbounds dof = occurrences[occurrence]
        @inbounds Atomix.@atomic :monotonic first_occurrence[dof] min occurrence
    end
    KernelAbstractions.synchronize(backend)

    # Scan in occurrence order so the compact indices follow CPU encounter order.
    retained = similar(occurrences, Int)
    AK.foreachindex(occurrences, execution_backend) do occurrence
        @inbounds dof = occurrences[occurrence]
        @inbounds retained[occurrence] = first_occurrence[dof] == occurrence
    end
    KernelAbstractions.synchronize(backend)
    offsets, unique_count = Bcube.dirichlet_face_offsets(retained, backend)

    indices = similar(occurrences, unique_count)
    AK.foreachindex(occurrences, execution_backend) do occurrence
        @inbounds if retained[occurrence] != 0
            dof = occurrences[occurrence]
            position = offsets[occurrence] + 1
            indices[position] = dof
        end
    end
    KernelAbstractions.synchronize(backend)
    return indices
end

"""
    Bcube.allocate_dirichlet_sparse_vector(U, I_diri::AbstractGPUArray, T)

Allocate native sparse storage with explicit zero values and sorted device indices.
The backend supplies the `sparsevec` constructor for its device buffers.
"""
function Bcube.allocate_dirichlet_sparse_vector(U, I_diri::AbstractGPUArray, T)
    sorted_indices = isempty(I_diri) ? I_diri : AK.sort(I_diri)
    values = fill!(similar(I_diri, T), zero(T))
    return sparsevec(sorted_indices, values, Bcube.get_ndofs(U))
end

"""
    Bcube.prepare_dirichlet_vector!(b::AbstractSparseVector, I_diri, position_by_dof, backend::AbstractBcubeBackendAcc{K, B}) where {K, B <: KernelAbstractions.GPU}

Return sparse storage positions in compact Dirichlet order. Insert absent entries
into `b` while preserving its other stored values, including explicit zeros.
"""
function Bcube.prepare_dirichlet_vector!(
    b::AbstractSparseVector,
    I_diri,
    position_by_dof,
    backend::AbstractBcubeBackendAcc{K, B},
) where {K, B <: KernelAbstractions.GPU}
    isempty(I_diri) && return similar(I_diri, 0)
    execution_backend = get_backend(backend)
    stored_indices = SparseArrays.nonzeroinds(b)
    stored_values = nonzeros(b)
    positions = KernelAbstractions.zeros(backend, Int, length(I_diri))
    AK.foreachindex(stored_indices, execution_backend) do stored_position
        @inbounds position = position_by_dof[stored_indices[stored_position]]
        @inbounds if position != 0
            positions[position] = stored_position
        end
    end
    KernelAbstractions.synchronize(backend)
    absent_count = AK.mapreduce(iszero, +, positions, execution_backend; init = 0)
    absent_count == 0 && return positions

    # Mark the union of stored and constrained dofs in global index order.
    stored_by_dof = KernelAbstractions.zeros(backend, Int, length(b))
    AK.foreachindex(stored_indices, execution_backend) do stored_position
        @inbounds stored_by_dof[stored_indices[stored_position]] = stored_position
    end
    KernelAbstractions.synchronize(backend)
    retained = similar(stored_by_dof)
    AK.foreachindex(retained, execution_backend) do dof
        @inbounds retained[dof] = stored_by_dof[dof] != 0 || position_by_dof[dof] != 0
    end
    offsets = AK.accumulate(+, retained, execution_backend; init = 0, inclusive = false)
    new_count = length(stored_indices) + absent_count
    new_indices = similar(stored_indices, new_count)
    new_values = similar(stored_values, new_count)
    AK.foreachindex(retained, execution_backend) do dof
        @inbounds if retained[dof] != 0
            new_position = offsets[dof] + 1
            old_position = stored_by_dof[dof]
            new_indices[new_position] = dof
            new_values[new_position] =
                old_position == 0 ? zero(eltype(stored_values)) : stored_values[old_position]
            compact_position = position_by_dof[dof]
            if compact_position != 0
                positions[compact_position] = new_position
            end
        end
    end
    KernelAbstractions.synchronize(backend)
    copyto!(b, typeof(b)(new_indices, new_values, length(b)))
    KernelAbstractions.synchronize(backend)
    return positions
end

"""
    Bcube.dense_dirichlet_vector(sparse_values::AbstractSparseVector, backend::AbstractBcubeBackendAcc)

Materialize the sparse Dirichlet values in a zero-initialized vector on `backend`.

The stored indices and values must be accessible from the execution backend.
"""
function Bcube.dense_dirichlet_vector(
    sparse_values::AbstractSparseVector,
    backend::AbstractBcubeBackendAcc,
)
    dense_values = KernelAbstractions.zeros(backend, eltype(sparse_values), length(sparse_values))
    indices = SparseArrays.nonzeroinds(sparse_values)
    values = nonzeros(sparse_values)
    Bcube.set_dirichlet_values!(dense_values, indices, values, backend)
    return dense_values
end

"""
    Bcube.set_dirichlet_value!(values, index, value, backend::AbstractBcubeBackendAcc)

Resolve concurrent writes from faces sharing a dof using an atomic assignment.

Concurrent occurrences of a shared dof must provide the same value to obtain a
deterministic result.
"""
function Bcube.set_dirichlet_value!(values, index, value, backend::AbstractBcubeBackendAcc)
    Atomix.@atomic values[index] = value
end

"""
    Bcube.set_dirichlet_values!( b, positions, V_diri, backend::AbstractBcubeBackendAcc, )

Assign `V_diri[k]` to the prepared storage position `positions[k]` in `b`.

The positions must follow the compact Dirichlet order and be unique. Only stored
values are written; no structure changes or atomic operations are needed in this step.
The accelerated method synchronizes before returning.
"""
function Bcube.set_dirichlet_values!(b, positions, V_diri, backend::AbstractBcubeBackendAcc)
    storage = Bcube.dirichlet_vector_storage(b)
    AK.foreachindex(V_diri, get_backend(backend)) do k
        storage[positions[k]] = V_diri[k]
    end
    KernelAbstractions.synchronize(backend)
end

"""
    fill_device_csc_diagonal_positions!(positions, I_diri, rows, pointers, backend)

Find the diagonal of each constrained CSC column. Store zero for an absent
diagonal and wait for the positions before returning.
"""
function fill_device_csc_diagonal_positions!(positions, I_diri, rows, pointers, backend)
    AK.foreachindex(I_diri, get_backend(backend)) do k
        @inbounds begin
            column = I_diri[k]
            positions[k] = 0
            for position in pointers[column]:(pointers[column + 1] - 1)
                row = rows[position]
                if row == column
                    positions[k] = position
                    break
                elseif row > column
                    break
                end
            end
        end
    end
    KernelAbstractions.synchronize(backend)
end

"""
    Bcube.prepare_dirichlet_matrix!(matrix::GPUArrays.AbstractGPUSparseMatrixCSC, I_diri, backend::AbstractBcubeBackendAcc)

Insert only absent diagonals on the device, preserving the CSC matrix object.
Return the diagonal positions in compact Dirichlet order.
"""
function Bcube.prepare_dirichlet_matrix!(
    matrix::GPUArrays.AbstractGPUSparseMatrixCSC,
    I_diri,
    backend::AbstractBcubeBackendAcc,
)
    positions = similar(I_diri, Int)
    isempty(I_diri) && return positions
    execution_backend = get_backend(backend)
    rows = rowvals(matrix)
    pointers = SparseArrays.getcolptr(matrix)
    fill_device_csc_diagonal_positions!(positions, I_diri, rows, pointers, backend)
    absent_count = AK.mapreduce(iszero, +, positions, execution_backend; init = 0)
    absent_count == 0 && return positions

    column_count = size(matrix, 2)
    insert_by_column = KernelAbstractions.zeros(backend, Int, column_count)
    AK.foreachindex(I_diri, execution_backend) do k
        @inbounds insert_by_column[I_diri[k]] = iszero(positions[k])
    end
    offsets = AK.accumulate(+, insert_by_column, execution_backend; init = 0, inclusive = false)

    values = nonzeros(matrix)
    new_count = length(values) + absent_count
    new_pointers = similar(pointers)
    new_rows = similar(rows, new_count)
    new_values = similar(values, new_count)
    AK.foreachindex(insert_by_column, execution_backend) do column
        @inbounds begin
            new_position = Int(pointers[column]) + offsets[column]
            new_pointers[column] = new_position
            insert_diagonal = !iszero(insert_by_column[column])

            # Insert the new diagonal before the first row greater than the column.
            for old_position in pointers[column]:(pointers[column + 1] - 1)
                row = rows[old_position]
                if insert_diagonal && row > column
                    new_rows[new_position] = column
                    new_values[new_position] = zero(eltype(values))
                    new_position += 1
                    insert_diagonal = false
                end
                new_rows[new_position] = row
                new_values[new_position] = values[old_position]
                new_position += 1
            end

            if insert_diagonal
                new_rows[new_position] = column
                new_values[new_position] = zero(eltype(values))
            end
            if column == column_count
                new_pointers[column + 1] = new_count + 1
            end
        end
    end
    KernelAbstractions.synchronize(backend)
    copyto!(matrix, typeof(matrix)(new_pointers, new_rows, new_values, size(matrix)))
    fill_device_csc_diagonal_positions!(
        positions, I_diri, rowvals(matrix), SparseArrays.getcolptr(matrix), backend,
    )
    return positions
end

"""
    zero_dirichlet_csc_kernel!( values, @Const(rows), @Const(pointers), @Const(mask), cancel_columns, )

Cancel CSC entries with one kernel thread per column; no thread changes sparse structure.
"""
@kernel function zero_dirichlet_csc_kernel!(
    values,
    @Const(rows),
    @Const(pointers),
    @Const(mask),
    cancel_columns::Bool,
)
    column = @index(Global)
    for position in pointers[column]:(pointers[column + 1] - 1)
        if mask[rows[position]] || (cancel_columns && mask[column])
            values[position] = zero(eltype(values))
        end
    end
end

"""
    Bcube.zero_dirichlet_entries!(
        matrix::Union{SparseArrays.AbstractSparseMatrixCSC, GPUArrays.AbstractGPUSparseMatrixCSC},
        mask,
        cancel_columns::Bool,
        backend::AbstractBcubeBackendAcc,
    )

Launch CSC cancellation using the stored row, column pointer and value buffers.
Wait for cancellation before diagonal restoration.
"""
function Bcube.zero_dirichlet_entries!(
    matrix::Union{SparseArrays.AbstractSparseMatrixCSC, GPUArrays.AbstractGPUSparseMatrixCSC},
    mask,
    cancel_columns::Bool,
    backend::AbstractBcubeBackendAcc,
)
    zero_dirichlet_csc_kernel!(get_backend(backend), WORKGROUP_SIZE)(
        nonzeros(matrix),
        rowvals(matrix),
        SparseArrays.getcolptr(matrix),
        mask,
        cancel_columns;
        ndrange = size(matrix, 2),
    )
    KernelAbstractions.synchronize(backend)
end

"""
    set_dirichlet_diagonal_kernel!(values, @Const(positions), value)

Restore one diagonal per thread at prepared unique storage positions.
"""
@kernel function set_dirichlet_diagonal_kernel!(values, @Const(positions), value)
    k = @index(Global)
    values[positions[k]] = value
end

"""
    Bcube.set_dirichlet_diagonal!(
        matrix::Union{SparseArrays.AbstractSparseMatrixCSC, GPUArrays.AbstractGPUSparseMatrixCSC},
        positions,
        value,
        backend::AbstractBcubeBackendAcc,
    )

Restore CSC diagonals after cancellation using the prepared positions in the
values buffer. Wait for the writes before returning.
"""
function Bcube.set_dirichlet_diagonal!(
    matrix::Union{SparseArrays.AbstractSparseMatrixCSC, GPUArrays.AbstractGPUSparseMatrixCSC},
    positions,
    value,
    backend::AbstractBcubeBackendAcc,
)
    set_dirichlet_diagonal_kernel!(get_backend(backend), WORKGROUP_SIZE)(
        nonzeros(matrix),
        positions,
        value;
        ndrange = length(positions),
    )
    KernelAbstractions.synchronize(backend)
end
