using Pkg
Pkg.activate("../Quantum_Neural_Network_Classifiers/amplitude_encode")

using Random
using MAT
include("train.jl")

vars = matread("MNIST_1_9_wk.mat")
x_train = vars["x_train"]
y_train = vars["y_train"]
x_test = vars["x_test"]
y_test = vars["y_test"]

num_train = 1000
num_test = 200
x_train = x_train[:,1:num_train]
y_train = y_train[1:num_train,:]
x_test = x_test[:,1:num_test]
y_test = y_test[1:num_test,:]

function randtrain(chain::AbstractVector, struct_list::AbstractMatrix,  x_train, x_test, y_train, y_test)
    return rand()
end

function evaluate(chainset::AbstractMatrix, struct_list::AbstractMatrix, nbest::Int)
    merits = zeros(Float64, size(chainset,1))
    for i in 1:size(chainset,1)
        chain = chainset[i,:]
        merits[i] = fitness(chain, struct_list, x_train, x_test, y_train, y_test)
    end
    indices = partialsortperm(merits, 1:nbest, rev=true)
    println(merits)
    return chainset[indices,:], merits
end

function evolve_chain(chain::AbstractVector, depth_evolve::Int, dirG::AbstractMatrix)
    chain = reshape(chain, 1, :)
    for _ in 1:depth_evolve
        tail = chain[end]
        available_indices = findall(x -> x == 1, dirG[tail,:]) 
        chain = hcat(chain, rand(available_indices))
    end
    return chain
end

function initialize(n::Int, depth::Int, dirG::AbstractMatrix)
    maxindex = size(dirG,1)
    chainset = zeros(Int, 0, depth)
    for _ in 1:n
        chain = evolve_chain([rand(1:maxindex)], depth-1, dirG) # Get the head of the chain randomly and evolve depth-1 further
        chainset = vcat(chainset,
                        chain)
    end
    return chainset
end

function evolve_set(chainset::AbstractMatrix, 
                    num_split::Int, nbest::Int, depth_evolve::Int, 
                    struct_list::AbstractMatrix, dirG::AbstractMatrix)
    newchainset = zeros(Int,0,size(chainset,2)+depth_evolve)
    for chain in eachrow(chainset)
        for _ in 1:num_split
            newchain = evolve_chain(chain, depth_evolve, dirG)
            newchainset = vcat(newchainset,
                               newchain)
        end
    end
    println(newchainset)
    newchainset, merits = evaluate(newchainset, struct_list, nbest)
    println(newchainset)
    return newchainset, merits
end

function main()
    Random.seed!(42)
    graphs = matread("graphs8.mat")
    struct_list = graphs["struct_list"]
    dirG = graphs["dirG"]

    num_chain_init = 10
    depth_init = 5

    num_split = 10
    nbest = 1
    depth_evolve = 1

    num_generation = 6

    println("Initializing ... ")
    chainset = initialize(num_chain_init, depth_init, dirG)
    merits_history = []
    chainset, merits = evaluate(chainset, struct_list, nbest)
    push!(merits_history, copy(merits))
    println(chainset)
    for _ in 1:num_generation
        chainset, merits = evolve_set(chainset, 
                              num_split, nbest, depth_evolve,
                              struct_list, dirG)
        push!(merits_history, copy(merits))
    end
end

main()


