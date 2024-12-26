using Pkg
Pkg.activate("../Quantum_Neural_Network_Classifiers/amplitude_encode")

using MAT

function build_struct_list(nqubit::Int)
    # 
    # This function returns a list of vectors representing a collection of 
    # gateblock structures.
    # We don't use the conventional k+2*div(k,2) vector in the paper.
    # We just use a nqubit-length vector representing the gateblock structure,
    # which is more convenient when comparing different gateblocks.
    # RULES:
    # 0 --- nothing
    # 1 --- R gate
    # 2 3 --- 2 control 3
    # 4 5 --- 5 control 4
    #
    struct_list = zeros(Int,0,0)
    pre1list = zeros(Int,0,0)
    pre2list = zeros(Int,0,0)
    for iqubit in 1:nqubit
        if iqubit==1
            struct_list = Int[0;1]
            pre1list = zeros(Int,1,0)
        else
            pre2list = pre1list
            pre1list = struct_list
            pre1row = size(pre1list,1)
            pre2row = size(pre2list,1)

            cat1 = vcat(hcat(zeros(Int,pre1row,1), pre1list), 
                        hcat( ones(Int,pre1row,1), pre1list))
            cat2 = vcat(hcat(repeat(Int[2 3], pre2row, 1), pre2list), 
                        hcat(repeat(Int[4 5], pre2row, 1), pre2list))
            struct_list = vcat(cat1, cat2)
        end
    end
    return struct_list
end

function allowCAT(base::AbstractVector, seq::AbstractVector)
    len = size(base, 1)
    isallow = 1
    for i in 1:len
        # check for single R gate 
        isFree1 = ((base[i]==0) && (seq[i]==1))
        isRepeated1 = ((base[i]==1) && (seq[i]==1))
        if i != len
            # check for CRx gate
            isFree2 = (base[i]==0 && base[i+1]==0) && ((seq[i]==2 && seq[i+1]==3) || (seq[i]==4 && seq[i+1]==5))
            isRepeated2 = (base[i]==2 && base[i+1]==3 && seq[i]==2 && seq[i+1]==3) || (base[i]==4 && base[i+1]==5 && seq[i]==4 && seq[i+1]==5)
            if isFree2 || isRepeated2
                isallow = 0
                break
            end
        end
        if isFree1 || isRepeated1
            isallow = 0
            break
        end
    end
    return Bool(isallow)
end

function build_dirGraph(n::Int, reflist::AbstractMatrix)
    G = trues(n, n)
    for i in 1:n
        for j in 1:n
            base = reflist[i,:]
            seq = reflist[j,:]
            if !allowCAT(base, seq)
                G[i,j] = 0
            end
        end
    end
    return G
end


function main()
    num_qubit = 8
    struct_list = build_struct_list(num_qubit)
    num_nodes = size(struct_list, 1)
    dirG = build_dirGraph(num_nodes, struct_list)

    outfile_name = "graphs8.mat" 
    matwrite(outfile_name, Dict("struct_list" => struct_list,
                                "dirG" => dirG))
end

main()

