# Revised simplex method
function revised_primal_simplex(c::Vector{Float64}
                                , A::Matrix{Float64}
                                , b::Vector{Float64}
                                , basis::Vector{Int64})
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

        # find the entering variable:
        k = n + 1
        enter_loc = -1
        cost = -1
        for s in 1:length(not_basis)
            if (not_basis[s] < k) && (z[s] > 0)
                k = not_basis[s]
                enter_loc = s
                cost = z[s]
            end
        end
        # k is the entering variable
        if cost > 0
            d_basis = -(A[:, basis] \ A[:, k])
            d = zeros(n)
            d[basis] = d_basis
            d[k] = 1.0

            candidates = d_basis .< 0
            if !any(candidates)
                println("Error: problem is unbounded; returning direction")
                return d
            end

            # find the leaving variable using the ratio test
            lambda = Inf
            r = n + 1
            leave_loc = -1
            for i in 1:length(basis)
                if d[basis[i]] < 0
                    tmp = x[basis[i]]/(-d[basis[i]])
                    if (lambda >= tmp) && (r > basis[i])
                        lambda = tmp
                        r = basis[i]
                        leave_loc = i
                    end
                end
            end

            # update the solution:
            x += lambda * d

            # r is the leaving variable
            # update the basis:
            tmp = not_basis[enter_loc]
            not_basis[enter_loc] = basis[leave_loc]
            basis[leave_loc] = tmp
        else
            break
        end

        iter += 1
    end
    return x
end

function revised_dual_simplex(c::Vector{Float64}
                             , A::Matrix{Float64}
                             , b::Vector{Float64}
                             , basis::Vector{Int64})
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
    y = transpose(A[:, basis]) \ c[basis]

    while true
        println("Iter: ", iter)
        println("- basis: ", basis)
        println("- solution: ", y)

        x = A[:, basis] \ b

        # find the leaving variable
        r = n+1
        leave_loc = -1
        for i in 1:length(x)
            if (x[i] < 0) && (basis[i] < r)
                leave_loc = i
                r = basis[i]
            end
        end

        # r is the leaving variable
        if r < n+1
            q = zeros(m)
            q[leave_loc] = -1.0
            d = transpose(A[:, basis]) \ q
            u = transpose(A[:, not_basis]) * d

            if all(u .<= 0)
                # unbounded
                println("Error: problem is unbounded")
                return d
            else
                z = c[not_basis] - transpose(A[:, not_basis]) * y
                # find the entering variable
                lambda = Inf
                k = n + 1
                enter_loc = -1
                for j in 1:length(not_basis)
                    if (u[j] > 0)
                        tmp = z[j] / u[j]
                        if (lambda >= tmp) && (not_basis[j] < k)
                            lambda = tmp
                            k = not_basis[j]
                            enter_loc = j
                        end
                    end
                end

                y += lambda * d

                tmp = not_basis[enter_loc]
                not_basis[enter_loc] = basis[leave_loc]
                basis[leave_loc] = tmp
            end

        else
            break
        end
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
