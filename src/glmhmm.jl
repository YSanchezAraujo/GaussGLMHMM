using Distributions
using LinearAlgebra
using LogExpFunctions

include("utils.jl")

struct forward_object
    α::Matrix
    π::Vector
    Ψ::Matrix
    Z::Vector
end

function forward_message!(fobj, dm_obj)
    T = size(dm_obj.L, 1)
    αt, Zt = normalize_z(dm_obj.L[1, :] .* fobj.π)
    fobj.α[1, :] = αt
    fobj.Z[1] = Zt
    for t in 2:T
        αt, Zt = normalize_z(dm_obj.L[t, :] .* (fobj.Ψ' * fobj.α[t-1, :]))
        fobj.α[t, :] = αt
        fobj.Z[t] = Zt
    end
end

function update_forward_message!(fobj, pos)
    fobj.π[:] = pos.π
    fobj.Ψ[:, :] = pos.Ψ
    fobj.Z[:] = zeros(size(fobj.α, 1))
end

struct backward_object
    β::Matrix
end

function backward_message!(bobj, dm_obj, Ψ)
    T = size(dm_obj.L, 1)
    bobj.β[T, :] .= 1.
    for t in T:-1:2
        βt, _ = normalize_z(Ψ * (dm_obj.L[t, :] .* bobj.β[t, :]))
        bobj.β[t-1, :] = βt
    end
end


struct data_models
    X::Matrix
    y::Vector
    ϕ::Vector
    L::Matrix
end


# same here
function compute_likelihoods!(dm_obj, μ, σ)
    T = size(dm_obj.y, 1)
    K = length(dm_obj.ϕ
        
    for k in 1:K
        pdist = dm_obj.ϕ[k].(μ[:, k], σ[k])
        dm_obj.L[:, k] = pdf.(pdist, dm_obj.y)
    end
end

struct posterior_object
    W::Matrix
    μ::Matrix 
    σ::Vector
    γ::Matrix
    π::Vector
    Ψ::Matrix
end

function compute_posteriors(fobj, bobj, dm_obj)
    T = size(dm_obj.y, 1)
    K = length(dm_obj.ϕ)
    γ = fobj.α .* bobj.β
    W = zeros(size(dm_obj.X, 2), K)
    
    for k in 1:K
        p_z = diagm(γ[:, k] ./ sum(γ[:, k]))
        X_proj = dm_obj.X' * p_z * dm_obj.X
        y_proj = dm_obj.X' * p_z * dm_obj.y
        W[:, k] = X_proj \ y_proj
    end
    
    μ = dm_obj.X * W # we know this is a matrix
        
    y_minus_μ = dm_obj.y .- μ
        
    σ2, ξ = zeros(K), zeros(K, K, T)
        
    # lets look into fixing this
    for k in 1:K
        σ2[k] = γ[:, k]'y_minus_μ[:, k].^2 / sum(γ[:, k])
    end
    
    π = γ[1, :] ./ sum(γ[1, :])
    
    @inbounds for t in 1:T-1
        lik_beta = dm_obj.L[t+1, :] .* bobj.β[t+1, :]
        
        alpha = fobj.α[t, :]
        
        ξ[:, :, t] = (fobj.Ψ * (alpha * lik_beta')) ./ fobj.Z[t+1]
            
    end
    
    ξ_NT = drop_dim(sum(ξ; dims=(3, 2)))
    
    ξ = drop_dim(sum(ξ; dims=3)) ./ ξ_NT
    
    return posterior_object(W, μ, sqrt.(σ2), γ, π, ξ)
end
