# Revised Simplex Method Implementation
# Author: Shobhit Bhatnagar

using LinearAlgebra

function get_min(array, filter)
    minValue = Inf
    minIndex = 0
    for i in 1:length(array)
        if (array[i] < minValue) && (filter[i])
            minValue = array[i]
            minIndex = i
        end
    end
    return (minValue, minIndex)
end

function update_basis_inverse!(A::Matrix{Float64}, B::Matrix{Float64}, i::Int64, j::Int64)
    # i is the index of the leaving variable in the basis
    # j is the index of the entering variable
    d = B * A[:, j]
    tmp = d[i]
    d .= -d ./ tmp
    d[i] /= -tmp


    n = length(d)
    E = Matrix{Float64}(1.0I, n, n)
    E[:, i] = d
    B .= E * B
end

function revised_primal_simplex(c::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, basis::Vector{Int64})
    # Solves the problem:
    # min c' x
    # subject to A x = b
    # x >= 0

    n = length(c)
    m = length(b)
    (m1, n1) = size(A)
    if (n != n1) || (m != m1)
        println("Error: size mismatch")
        exit(1)
    end

    not_basis = Vector{Int64}()
    for i in 1:n
        if !(i in basis)
            append!(not_basis, i)
        end
    end
    iter = 0
    x = zeros(n)
    B = inv(A[:, basis])
    x[basis] = B * b

    while true
        println("Iter: ", iter)
        println("-> basis: ", basis)
        println("-> solution: ", x)

        y = transpose(B) * c[basis]
        z = transpose(A[:, not_basis]) * y - c[not_basis]
        # Find the entering variable:
        (k, entering_var_index) = get_min(not_basis, z .> 0)

        if (entering_var_index > 0)
            d_basis = -B * A[:, k]
            d = zeros(n)
            d[basis] = d_basis
            d[k] = 1.0

            if all(d_basis .>= 0)
                println("Error: problem is unbounded; returning direction")
                return d
            end

            # Find the leaving variable using the ratio test:
            ratios = x ./ (-d)
            (lambda, _) = get_min(ratios, d .< 0)
            (r, leaving_var_index) = get_min(basis, ratios .== lambda)

            # update the solution:
            x .+= lambda * d

            # Update the basis:
            update_basis_inverse!(A, B, leaving_var_index, k)

            tmp = not_basis[entering_var_index]
            not_basis[entering_var_index] = basis[leaving_var_index]
            basis[leaving_var_index] = tmp

        else
            break
        end

        iter += 1
    end
    return x
end

function revised_dual_simplex(c::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, basis::Vector{Int64})
    # Solves the problem:
    # max y' b
    # subject to y' A <= c

    n = length(c)
    m = length(b)
    (m1, n1) = size(A)
    if (n != n1) || (m != m1)
        println("Error: size mismatch")
        exit(1)
    end

    not_basis = Vector{Int64}()
    for i in 1:n
        if !(i in basis)
            append!(not_basis, i)
        end
    end

    iter = 0
    B = inv(A[:, basis])
    y = transpose(B) * c[basis]

    while true
        println("Iter: ", iter)
        println("-> basis: ", basis)
        println("-> solution: ", y)

        x = B * b
        # Find the leaving variable:
        (r, leaving_var_index) = get_min(basis, x .< 0)

        if (leaving_var_index > 0)
            q = zeros(m)
            q[leaving_var_index] = -1.0
            d = transpose(B) * q
            u = transpose(A[:, not_basis]) * d

            if all(u .<= 0)
                # Dual unbounded => Primal infeasible
                println("Error: problem is unbounded; returning direction")
                return d
            else
                z = c[not_basis] - transpose(A[:, not_basis]) * y
                ratios = z ./ u

                # Find the entering variable:
                (lambda, _) = get_min(ratios, u .> 0)
                (k, entering_var_index) = get_min(not_basis, ratios .== lambda)

                y .+= lambda * d

                # Update the basis:
                update_basis_inverse!(A, B, leaving_var_index, k)

                tmp = not_basis[entering_var_index]
                not_basis[entering_var_index] = basis[leaving_var_index]
                basis[leaving_var_index] = tmp
            end

        else
            break
        end

        iter += 1
    end
    return y
end

# Primal simplex examples:
# Example problem 1
println("Solving problem 1: ")
A = [[1.0, 1] [1, -1] [1, 2] [1, -1]]
b = [4.0, 2]
c = [1.0, 3, -1, 3]
basis = [1,2]
x = revised_primal_simplex(c, A, b, basis)

# Example problem 2
println("Solving problem 2: ")
A = [[1.0, 1] [-2, 1] [1, 2] [-1, -1]]
b = [0.0, 1]
c = [3.0, 2, -1, 0]
basis = [1,2]
x = revised_primal_simplex(c, A, b, basis)

# Example problem 3
println("Solving problem 3: ")
A = [[1.0, 1] [1, -1] [-2, 2] [2, 0]]
b = [3.0, 1]
c = [2.0, 0, -1, 2]
basis = [1,2]
x = revised_primal_simplex(c, A, b, basis)

# Dual simplex examples:
# Example problem 1
println("Solving problem 1: ")
A = [[1.0, 1] [1, -1] [-1, 2]]
b = [2.0, 3]
c = [0.0, 0, 0]
basis = [1,2]
y = revised_dual_simplex(c, A, b, basis)

# Example problem 2
println("Solving problem 2: ")
A = [[3.0 ,-1] [-1, 2] [-1, 1] [0, -1]]
b = [1, 3.0]
c = [4.0, 3, 0 , -1]
basis = [3,4]
y = revised_dual_simplex(c, A, b, basis)

# Example problem 3
println("Solving problem 3: ")
A = [[1.0, 2, -1, 1] [1, 1, 1, 1] [0, 1, 0, 0] [1, 0, 0, 0] [0, 0, 1, 0] [0, 0, 0, 1]]
b = [6.0, 10, 4, 5]
c = [-3.0, -4, 0, 0, 0, 0]
basis = [1, 2, 3, 6]
y = revised_dual_simplex(c, A, b, basis)
