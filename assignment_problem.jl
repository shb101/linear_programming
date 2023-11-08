# Assignment Problem
# Author: Shobhit Bhatnagar

include("revised_simplex.jl")
using Main.RevisedSimplex, LinearAlgebra, Random

Random.seed!(0)

# Lets create a random assignment problem:
function generate_assignment_problem(size::Int64)
    if size <= 1
        throw(DomainError(size, "Problem size should be >= 2"))
    end

    cost = round.(20*rand(size, size))
    n = size^2
    m = 2*size
    varIndex(i, j) = i + (j-1)*size
    c = zeros(n)
    b = ones(m-1)
    A = zeros(m-1,n)
    for i in 1:size
        for j in 1:size
            c[varIndex(i,j)] = cost[i,j]
            A[i, varIndex(i,j)] = 1.0
        end
    end

    for j in 1:(size-1)
        for i in 1:size
            A[j+size, varIndex(i,j+1)] = 1.0
        end
    end

    return (A, b, c)
end

function indexVar(k, n)
    i = 1 + (k-1) % n
    j = 1 + div(k-1, n)
    return (i, j)
end

function solve_assignment_problem(c::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    (m, n) = size(A)
    n1 = length(c)
    m1 = length(b)

    if (m != m1) || (n != n1)
        throw(DimensionMismatch("Please check the dimensions of matrices c, A and b."))
    end

    A1 = hcat(Matrix{Float64}(1.0I, m, m), A)
    c1 = zeros(n+m)
    c1[1:m] .= 1e+3
    c1[(m+1):(m+n)] .= c

    x = RevisedSimplex.revised_primal_simplex(c1, A1, b, Vector{Int64}(1:m))

    s = Int64(sqrt(n))
    r = 1
    assignments = zeros(s, 2)
    for k in (m+1):(n+m)
        if x[k] == 1
            (i, j) = indexVar(k-m, s)
            assignments[r, 1] = i
            assignments[r, 2] = j
            r += 1
        end
    end
    optimal_cost = sum(c1 .* x)
    return (assignments, optimal_cost)
end

# Assignment problem of size 50:
(A, b, c) = generate_assignment_problem(50)
(assignments, optimal_cost) = solve_assignment_problem(c, A, b)

println("Objective function: ")
println(c)

println("Optimal cost: ")
println(optimal_cost)

println("Assignments: ")
println(assignments)
