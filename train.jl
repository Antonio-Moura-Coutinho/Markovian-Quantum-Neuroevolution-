using Yao, Zygote, YaoPlots, CuYao, Yao.EasyBuild
using LinearAlgebra, Statistics, Random, StatsBase, ArgParse, Distributions
using Printf, BenchmarkTools, MAT, Plots
using Flux: batch, Flux
using Quantum_Neural_Network_Classifiers: ent_cx, params_layer, acc_loss_evaluation

nqubits = 8

# BlockGene
function BlockGene(blockID::Int, struct_list::AbstractMatrix)
    blockcode = struct_list[blockID,:]
    blockchain = chain(nqubits)
    for i in 1:nqubits
        if blockcode[i]==0 || blockcode[i]==3 || blockcode[i]==5
            continue
        elseif blockcode[i]==1
            blockchain = chain(nqubits, blockchain, put(i=>Rz(0)),put(i=>Rx(0)),put(i=>Rz(0)))
        elseif blockcode[i]==2
            blockchain = chain(nqubits, blockchain, control(i, (i+1)=>Rx(0)))
        elseif blockcode[i]==4
            blockchain = chain(nqubits, blockchain, control(i+1, i=>Rx(0)))
        end
    end
    return blockchain
end

# transfer the encode_number into its corresponding circuit.
function quant_circuit(encode_number::AbstractVector, struct_list::AbstractMatrix)  
    return chain(    
            BlockGene(encode_number[i], struct_list)  
            for i = 1:length(encode_number)
            )
end

batch_size = 100 # batch size
lr = 0.01        # learning rate
niters = 150    # number of iterations
optim = Flux.ADAM(lr) # Adam optimizer

# index of qubit that will be measured
pos_ = 8  
op0 = put(nqubits, pos_=>0.5*(I2+Z))
op1 = put(nqubits, pos_=>0.5*(I2-Z))

num_train = 1000
num_test = 200

# fitness
function fitness(encode_number::AbstractVector, struct_list::AbstractMatrix, x_train, x_test, y_train, y_test)
    circuit = quant_circuit(encode_number, struct_list) 
    # ent_layer(nbit::Int64) = ent_cx(nbit)
    # parameterized_layer(nbit::Int64) = params_layer(nbit)
    # composite_block(nbit::Int64) = chain(nbit, parameterized_layer(nbit::Int64), ent_layer(nbit::Int64))
    # circuit = chain(composite_block(nqubits) for _ in 1:depth)
    # assign random initial parameters to the circuit
    dispatch!(circuit, :random)

    params = parameters(circuit)

    # if GPU resources are available, you can make modifications including 
    # replacing  "|> cpu" by "|> cu"
    x_train_yao = zero_state(nqubits, nbatch=num_train)
    x_train_yao.state = x_train
    cu_x_train_yao = copy(x_train_yao) |> cpu
    
    x_test_yao = zero_state(nqubits, nbatch=num_test)
    x_test_yao.state  = x_test
    cu_x_test_yao = copy(x_test_yao) |> cpu

    # record the training history
    loss_train_history = Float64[]
    acc_train_history = Float64[]
    loss_test_history = Float64[]
    acc_test_history = Float64[]

    for k in 0:niters
        # calculate the accuracy & loss for the training & test set
        train_acc, train_loss = acc_loss_evaluation(circuit, cu_x_train_yao, y_train, num_train, pos_)
        test_acc, test_loss = acc_loss_evaluation(circuit, cu_x_test_yao, y_test, num_test, pos_)
        push!(loss_train_history, train_loss)
        push!(loss_test_history, test_loss)
        push!(acc_train_history, train_acc)
        push!(acc_test_history, test_acc)
        # if k % 5 == 0
        #     @printf("\nStep=%d, loss=%.3f, acc=%.3f, test_loss=%.3f,test_acc=%.3f\n",k,train_loss,train_acc,test_loss,test_acc)
        # end
        
        # at each training epoch, randomly choose a batch of samples from the training set
        batch_index = randperm(size(x_train)[2])[1:batch_size]
        x_batch = x_train[:,batch_index]
        y_batch = y_train[batch_index,:]

        # prepare these samples into quantum states
        x_batch_1 = copy(x_batch)
        x_batch_yao = zero_state(nqubits,nbatch=batch_size)
        x_batch_yao.state = x_batch_1;
        cu_x_batch_yao = copy(x_batch_yao) |> cpu;
        batc = [zero_state(nqubits) for i in 1:batch_size]
        for i in 1:batch_size
            batc[i].state = x_batch_1[:,i:i]
        end
        
        # for all samples in the batch, repeatly measure their qubits at position pos_ 
        # on the computational basis
        q_ = zeros(batch_size,2)
        res = copy(cu_x_batch_yao) |> circuit
        for i=1:batch_size
            rdm = density_matrix(viewbatch(res, i), (pos_,))
            q_[i,:] = Yao.probs(rdm)
        end
        
        # calculate the gradients w.r.t. the cross-entropy loss function
        Arr = Array{Float64}(zeros(batch_size,nparameters(circuit)))
        for i in 1:batch_size
            Arr[i,:] = expect'(op0, copy(batc[i])=>circuit)[2]
        end
        C = [Arr, -Arr]
        grads = collect(mean([-sum([y_batch[i,j]*((1 ./ q_)[i,j])*batch(C)[i,:,j] for j in 1:2]) for i=1:batch_size]))
        
        # update the parameters
        updates = Flux.Optimise.update!(optim, params, grads);
        dispatch!(circuit, updates) 
    end

    return maximum(acc_test_history), loss_train_history, acc_train_history, loss_test_history, acc_test_history
end