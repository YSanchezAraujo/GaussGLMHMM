
function normalize_z(x)
    Z = sum(x)
    return x ./ Z,  Z
end

normalize_1o(x) = x ./ sum(x)
add_dim(x) = reshape(x, (size(x)..., 1))
drop_dim(a) = dropdims(a, dims = (findall(size(a) .== 1)...,))


function state_occupancy(prob_states)
    T, K = size(prob_states)

    occu = zeros(K)

    for t in 1:T
        v = argmax(prob_states[t, :])
        occu[v] += 1
    end

    return occu ./ T
end