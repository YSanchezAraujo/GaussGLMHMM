
include("/utils.jl")
include("/glmhmm.jl")


"""
inputs: \n
data: 1-d array of continuous values \n
dists: 1-d array of k Distribution objects w/o any parameters
      example: [Normal, Normal]
      if the number of latent variables being fit to the data is 2
"""
function fit_hmm_em(y, X, dists; tol=1e-4, max_iter=250)
    if size(y, 2) > 1
        throw(DomainError("n-dimensional samples not supported"))
    end
    
    T, K = size(y, 1), length(dists)
            
    init_mu = repeat([mean(y)], T) .+ rand(1, K) .+ rand(T)
            
    init_sd = std(y) .+ rand(K)
            
    init_pi = ones(K) .* 0.5
            
    A = ones(K, K) .* 0.5
    
    init_W = (X'X \ X'y) .+ rand(size(X, 2), K)
    
    # construct poseterior object       
    pos = posterior_object(init_W, init_mu, init_sd, zeros(T, K), init_pi, A)
                
    ll, ll_change, ll_iter = 1e7, 1, fill(NaN, max_iter)
    
    dm = data_models(X, y, dists, zeros(T, K))
      
    compute_likelihoods!(dm, pos.μ, pos.σ)
    
    f_msg = forward_object(zeros(T, K), init_pi, A, zeros(T))
      
    b_msg = backward_object(zeros(T, K)) 
    
    for m in 1:max_iter
        if ll_change < tol
            println("converged in: ", m, " iterations")
            break
        end
      
        # run messages forward
        forward_message!(f_msg, dm)
      
        # run messages backwards
        backward_message!(b_msg, dm, f_msg.Ψ)
      
        # compute posterior quantities
        pos = compute_posteriors(f_msg, b_msg, dm)
      
        # get current log-likelihood
        ll_iter[m] = sum(log.(f_msg.Z))
        
        ll_change = abs.(ll - ll_iter[m])
      
        ll = ll_iter[m]
            
        compute_likelihoods!(dm, pos.μ, pos.σ)
      
        update_forward_message!(f_msg, pos)
      
    end
    return pos, f_msg, b_msg, dm, ll_iter
end
