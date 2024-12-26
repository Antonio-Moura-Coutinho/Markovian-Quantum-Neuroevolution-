using Yao, BitBasis, CuYao, YaoExtensions, Yao.AD, Zygote
using Flux:batch, Flux
using LinearAlgebra, Statistics, Random, StatsBase
using NPZ, FileIO, Printf, BenchmarkTools, MAT
using YaoPlots, Plots, CUDA, ArgParse, Zygote
using Combinatorics, Compose, Distributions, Dates, Test

# number of qubits
const nbit = 8 ;
const niters = 200;
nqubits = 9 ;

# the gate-block library
library = matread("packages/gate-block-library.mat")
Column_set = library["Column_set"]  ;

# 邻接矩阵已经跑过一次，矩阵已经单独保存下来，之后使用直接调用即可。
# 获得邻接矩阵的程序，跑一次时间大概10分钟。
matrix = matread("packages/direct_matrix.mat")
direct_matrix = matrix["direct_matrix"]  ;

const1 = size(direct_matrix)[2];  
const2 = size(direct_matrix)[1] ;  

# BlockGene
function BlockGene( number )
    index = Column_set[:, number]
    BlockGene = chain(nbit+1, 
        chain(nbit+1, BlockGene1(index[2i-1:2i]) for i in 1:Int(floor(nqubits/2))),
        chain(nbit+1, BlockGene2(index[j]) for j in (2*Int(floor(nqubits/2))+1):(2*Int(floor(nqubits/2))+nqubits)  )
    )
     
end

function BlockGene1( index1 ) 
    
    if index1[1] == 0 && index1[2] == 0
        BlockGene1 = chain(nbit+1)
    else
        BlockGene1 = chain(
        nbit+1,
        control(index1[1], index1[2]=>Rx(0))
            )
    end
    return BlockGene1
end

function BlockGene2( index2 )    
    if index2 == 0
        BlockGene2 = chain(nbit+1)
    else
        BlockGene2 = chain(nbit+1,put(index2=>Rz(0)),put(index2=>Rx(0)),put(index2=>Rz(0)))
    end
    return BlockGene2
end 


# transfer the encode_number into its corresponding circuit.
function quant_circuit(encode_number)  
    return chain(    
            BlockGene(encode_number[i])  
            for i = 1:length(encode_number)
            )
end


# add a path of length of k to the original path.
function add_number(encode_number, k)
    encode_number_copy = deepcopy(encode_number) ;
    for i in 1:k
        temp = encode_number_copy[length(encode_number_copy)] ;
        temp1 = direct_matrix[ :, temp ];
        temp2 = temp1[rand(1:const2)]
        while temp2 == 0 
            temp2 = temp1[rand(1:const2)]
        end
        encode_number_copy = [encode_number_copy  temp2]
    end
    return encode_number_copy
end

# randomly generate path of length--(path_length).
function generate_path(path_length)
    initial_code = const1;
    return  add_number(initial_code, path_length-1) 
end


# define the function for obtaining the accuracy and the loss.
op0 = put(nbit+1, nbit+1=>0.5*(I2+Z))
op1 = put(nbit+1, nbit+1=>0.5*(I2-Z))

function acc_loss_rdm_cu(circuit,reg,y_batch,batch_size)   
    reg_ = focus!(addbits!(copy(reg),1) |> circuit, (nbit+1)) |> cpu
#     rdm = density_matrix(reg_).state;
    rdm_ = density_matrix.(reg_) ; rdm = zeros(ComplexF64, 2,2,batch_size) ; [rdm[:,:,i] = rdm_[i].state for i = 1:batch_size ];
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        q_[i,:] = diag(rdm[:,:,i]) |> real
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end


#  fitness
function fitness(encode_number)
    circuit = quant_circuit(encode_number) 
    
    test_loss = 1
    train_loss = 1
    f = 0
    while (test_loss > 0.7 || train_loss > 0.7)  && f < 20
        dbs2 = collect_blocks(ControlBlock, circuit);   dispatch!.(dbs2, pi);
        dbs1 = collect_blocks(PutBlock, circuit);       dispatch!.(dbs1, :random);
        train_acc, train_loss = acc_loss_rdm_cu(circuit,cu_x_train_yao,y_train_m,num_train)
        test_acc, test_loss = acc_loss_rdm_cu(circuit,cu_x_test_yao,y_test_m,num_test)     
        f = f + 1
    end
    
    optim = Flux.ADAM(lr)
    params = parameters(circuit) ;
    params_history = [] ;
    push!(params_history, params)
    
    acc_test_history = Float64[] 
    test_acc, test_loss = acc_loss_rdm_cu(circuit,cu_x_test_yao,y_test_m,num_test)
    push!(acc_test_history, test_acc)   
    
    for k in 0:niters
        batch_index = randperm(size(x_train_m)[2])[1:batch_size]
        x_batch = x_train_m[:,batch_index]
        y_batch = y_train_m[batch_index,:];

        x_batch_1 = copy(x_batch)
        x_batch_yao = zero_state(nbit,nbatch=batch_size)
        x_batch_yao.state  = x_batch_1;

        batc = [zero_state(nbit) for i in 1:batch_size]
        for i in 1:batch_size
            batc[i].state = x_batch_1[:,i:i]
        end

        reg_ = focus!(addbits!(copy(x_batch_yao),1) |> circuit, (nbit+1)) 
#         rdm = density_matrix(reg_).state;
        rdm_ = density_matrix.(reg_) ; rdm = zeros(ComplexF64, 2,2,batch_size) ; [rdm[:,:,i] = rdm_[i].state for i=1:batch_size];
        q_ = zeros(batch_size,2);
        for i=1:batch_size
            q_[i,:] = diag(rdm[:,:,i]) |> real
        end

        Arr = Array{Float64}(zeros(batch_size,nparameters(circuit)))
        for i in 1:batch_size
            Arr[i,:] = expect'(op0, addbits!(copy(batc[i]),1)=>circuit)[2]
        end

        C = [Arr, -Arr]
        
        grads = collect(mean([-sum([y_batch[i,j]*((1 ./ q_)[i,j])*batch(C)[i,:,j] for j in 1:2]) for i=1:batch_size]))

        updates = Flux.Optimise.update!(optim, params, grads);
        push!(params_history, updates)
        
        dispatch!(circuit, updates)
        #train_acc, train_loss = acc_loss_rdm_cu(circuit,cu_x_train_yao,y_train_m,num_train)
        test_acc, test_loss = acc_loss_rdm_cu(circuit,cu_x_test_yao,y_test_m,num_test)
        push!(acc_test_history, test_acc)
    end
    
    index1 = findmax(acc_test_history)[2]
    parma_record = params_history[index1] ;
    return [maximum(acc_test_history), parma_record]
end