# Revised simplex method
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
    x[basis] = A[:, basis] \ b
    while true
        println("Iter: ", iter)
        println("- basis: ", basis)
        println("- solution: ", x)

        y = transpose(A[:, basis]) \ c[basis]
        z = transpose(A[:, not_basis]) * y - c[not_basis]
        (zmax, k) = findmax(z)

        # not_basis[k] is the entering variable
        if zmax > 0
            d_basis = -(A[:, basis] \ A[:, not_basis[k]])
            d = zeros(n)
            d[basis] = d_basis
            d[not_basis[k]] = 1.0

            if all(d .>= 0)
                println("Error: problem is unbounded")
                return zeros(n)
            end

            # find the leaving variable using the ratio test
            lambda = Inf
            r = -1
            for i in 1:length(basis)
                if d[basis[i]] < 0
                    tmp = x[basis[i]]/(-d[basis[i]])
                    if lambda > tmp
                        lambda = tmp
                        r = i
                    end
                end
            end

            # update the solution:
            x = x + lambda * d

            # basis[r] is the leaving variable
            # update the basis:
            tmp = not_basis[k]
            not_basis[k] = basis[r]
            basis[r] = tmp
        else
            break
        end

        iter += 1
    end
    return x
end

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
