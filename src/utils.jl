
function init_estimates_kmeans(data, K, max_iter)
    R = kmeans(add_dim(data)', K; maxiter=max_iter)
    unique_assignments = sort(unique(R.assignments))
    init_sds = [std(data[R.assignments .== j]) for j in unique_assignments]
    init_pi = [mean(R.assignments .== j) for j in unique_assignments]
    dist_params = [(R.centers[k], init_sds[k]) for k in 1:K]
    RT = collect(zip(R.assignments[1:2:end], R.assignments[2:2:end]))
    pairs = sort(unique(RT))
    pair_counts = [count(t -> (t == pair), RT) for pair in pairs]
    Tmat = zeros(K*K)
    for k in 1:K*K
        for usgn in unique_assignments
            if pairs[k][1] == usgn
                Tmat[k] = pair_counts[k] / sum(R.assignments .== usgn)
            end
        end
    end
    Tmat = mapslices(normalize_1o, reshape([Tmat[1:2:end]; Tmat[2:2:end]], (2, 2)), dims=2)
    return init_pi, dist_params, Tmat, vec(R.centers), init_sds
end
