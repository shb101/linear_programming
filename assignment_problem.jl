# Assignment Problem
# Author: Shobhit Bhatnagar

include("revised_simplex.jl")
using Main.RevisedSimplex, LinearAlgebra, Random, JuMP, Cbc

Random.seed!(0)

# Lets create a random assignment problem:
function generate_assignment_problem(size::Int64)
    if size <= 1
        throw(DomainError(size, "Problem size should be >= 2"))
    end

    cost = round.(20*rand(size, size))
    return cost
end

function indexVar(k, n)
    i = 1 + (k-1) % n
    j = 1 + div(k-1, n)
    return (i, j)
end

function solve_assignment_problem(cost::Matrix{Float64})
    (dim1, dim2) = size(cost)
    if (dim1 != dim2)
        throw(DimensionMismatch("The cost matrix should be a square matrix!"))
    end

    n = dim1^2
    m = 2*dim1
    varIndex(i, j) = i + (j-1)*dim1

    c = zeros(n)
    b = ones(m-1)
    A = zeros(m-1,n)
    for i in 1:dim1
        for j in 1:dim1
            c[varIndex(i,j)] = cost[i,j]
            A[i, varIndex(i,j)] = 1.0
        end
    end

    for j in 1:(dim1-1)
        for i in 1:dim1
            A[j+dim1, varIndex(i,j+1)] = 1.0
        end
    end

    A = hcat(Matrix{Float64}(1.0I, m-1, m-1), A)
    c1 = zeros(n+m-1)
    c1[1:(m-1)] .= findmax(c)[1] * 10
    c1[m:(m+n-1)] .= c

    x = RevisedSimplex.revised_primal_simplex(c1, A, b, Vector{Int64}(1:(m-1)))
    
    r = 1
    assignments = zeros(Int64, dim1, 2)
    for k in m:(n+m-1)
        if x[k] == 1
            (i, j) = indexVar(k-m+1, dim1)
            assignments[r, 1] = i
            assignments[r, 2] = j
            r += 1
        end
    end
    optimal_cost = sum(c1 .* x)
    
    # print("Shobhit's solution: ", optimal_cost)
    # # Verification
    # model = Model(Cbc.Optimizer)
    # @variable(model, x_var[1:length(c1)] >= 0)
    # @objective(model, Min, sum(c1 .* x_var))
    # @constraint(model, A * x_var .== b)
    # optimize!(model)
    # sol_cbc = Vector{Int64}(round.(value.(model[:x_var])))
    # obj_cbc = sum(c1 .* sol_cbc)
    # println("CBC solution: ", obj_cbc)

    return (assignments, optimal_cost)
end

cost_matrix = generate_assignment_problem(100)
println("Cost matrix: ")
(n, _) = size(cost_matrix)
for i in 1:n
    println(cost_matrix[i, :])
end

(assignments, optimal_cost) = solve_assignment_problem(cost_matrix)

println("Optimal cost: ")
println(optimal_cost)

println("Optimal Assignments: ")
for i in 1:n
    println(assignments[i, :])
end
