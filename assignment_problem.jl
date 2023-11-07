# Assignment Problem
# Author: Shobhit Bhatnagar
include("revised_simplex.jl")
using Main.RevisedSimplex, LinearAlgebra

# Lets create a random assignment problem
function generate_assignment_problem(size::Int64)
    if size <= 1
        throw(DomainError(size, "Problem size should be >= 2"))
    end

    cost = round.(20*rand(size, size))
    n = size^2
    m = 2*size
    varIndex(i, j) = i + (j-1)*size
    c = zeros(n)
    b = ones(m)
    A = zeros(m,n)
    for i in 1:size
        for j in 1:size
            c[varIndex(i,j)] = cost[i,j]
            A[i, varIndex(i,j)] = 1.0
        end
    end

    for j in 1:size
        for i in 1:size
            A[j+size, varIndex(i,j)] = 1.0
        end
    end

    return (A, b, c)
end
# Assignment problem of size 5:
(A, b, c) = generate_assignment_problem(5)

println("Objective function: ", c)
println("Constraint Matrix: ")
for i in 1:10
    println(A[i, :])
end

identity = Matrix{Float64}(1.0I, 10, 10)
A1 = hcat(identity, A)
c1 = zeros(35)
c1[1:10] .= 1.0
x = RevisedSimplex.revised_primal_simplex(c1, A1, b, Vector{Int64}(1:10))
println(A1 * x)
