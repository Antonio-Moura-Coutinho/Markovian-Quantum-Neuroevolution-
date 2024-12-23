using Pkg
Pkg.activate("../Quantum_Neural_Network_Classifiers/amplitude_encode")

using Random
using MAT

function train(chain::AbstractVector, struct_list::AbstractMatrix)
    return rand()
end

function evaluate(chainset::AbstractMatrix, struct_list::AbstractMatrix, nbest::Int)
    merits = zeros(Float64, size(chainset,1))
    for i in 1:size(chainset,1)
        chain = chainset[i,:]
        merits[i] = train(chain, struct_list)
    end
    indices = partialsortperm(merits, 1:nbest, rev=true)
    return chainset[indices,:]
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
    newchainset = evaluate(newchainset, struct_list, nbest)
    return newchainset
end

function main()
    Random.seed!(42)
    graphs = matread("graph_generate/graphs.mat")
    struct_list = graphs["struct_list"]
    dirG = graphs["dirG"]

    num_chain_init = 10
    depth_init = 5

    num_split = 10
    nbest = 1
    depth_evolve = 1

    num_generation = 5

    chainset = initialize(num_chain_init, depth_init, dirG)
    chainset = evaluate(chainset, struct_list, nbest)
    println(chainset)
    for _ in 1:num_generation
        chainset = evolve_set(chainset, 
                              num_split, nbest, depth_evolve,
                              struct_list, dirG)
        println(chainset)
    end
end

main()


