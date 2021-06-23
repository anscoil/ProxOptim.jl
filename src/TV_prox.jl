using CUDA

function D!(DA::AbstractArray{T,3}, A::AbstractArray{T,2},
            dx::Real, dz::Real) where T
    @assert size(DA,1) == size(A,1)
    @assert size(DA,2) == size(A,2)
    @assert size(DA,3) == 2

    @views DA[1:end-1,:,1] .= (A[2:end,:] .- A[1:end-1,:])./dx
    @views DA[:,1:end-1,2] .= (A[:,2:end] .- A[:,1:end-1])./dz
    # Periodic boundary conditions
    @views DA[end,:,1] .= (A[1,:] .- A[end,:])./dx
    @views DA[:,end,2] .= (A[:,1] .- A[:,end])./dz

    DA
end

function D!(DA::AbstractArray{T}, A::AbstractArray{T}) where T
    D!(DA, A, 1, 1)
end

function Dᵀ!(A::AbstractArray{T,2}, DA::AbstractArray{T,3},
             dx::Real, dz::Real) where T
    @assert size(DA,1) == size(A,1)
    @assert size(DA,2) == size(A,2)
    @assert size(DA,3) == 2

    @views A[2:end,:] .= (DA[1:end-1,:,1] .- DA[2:end,:,1])./dx
    @views A[1,:] .= (DA[end,:,1] .- DA[1,:,1])./dx
    @views A[:,2:end] .+= (DA[:,1:end-1,2] .- DA[:,2:end,2])./dz
    @views A[:,1] .+= (DA[:,end,2] .- DA[:,1,2])./dx
    A
end

function Dᵀ!(A::AbstractArray{T}, DA::AbstractArray{T}) where T
    Dᵀ!(A, DA, 1, 1)
end

function proj_G1!(G::AbstractArray{T,3}) where T
    map!(x -> x / max(1, abs(x)), G, G)
    G
end

function proj_G2!(G::AbstractArray{T,3}) where T
    n, m, d = size(G)
    for j in 1:m
        for i in 1:n
            @views nl2 = sqrt.(sum(abs.(G[i,j,:]).^2))
            for k in 1:d
                G[i,j,k] /= max(1, nl2)
            end
        end
    end
    G
end

function proj_G2_kernel(G::CuDeviceArray{T,3}) where T
    n, m, d = size(G)
    ix = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    jx = (blockIdx().y - 1)*blockDim().y + threadIdx().y
    
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    @inbounds for j in jx:stride_y:m
        for i in ix:stride_x:n
            nl2 = real(T)(0)
            for k in 1:d
                nl2 += abs(G[i,j,k])^2
            end
            nl2 = sqrt(nl2)
            if nl2 > 1
                for k in 1:d
                    G[i,j,k] /= nl2
                end
            end
        end
    end
    nothing
end

function proj_G2!(G::CuArray{T,3}) where T
    n, m, d = size(G)
    @cuda threads=(16,16) blocks=(Int(ceil(n/16)),Int(ceil(m/16))) (
        proj_G2_kernel(G))
    G
end

struct TV_FGP{T}
    Z
    Xt
    Dt
    Gt
    Gt_tmp
    qt :: Ref{T}
    τ :: T
    γ :: T
    projG! :: Function
    prox! :: Function

    function TV_FGP(Z::AbstractArray{T}, τ::Real,
                    q0::Real=1, projG!::Function=proj_G2!;
                    prox!::Function=identity) where T
        RT = real(T)
        @assert τ > 0
        γ = 1/(12τ)
        Xt = similar(Z)
        Dt = similar(Z, (size(Z)..., ndims(Z)))
        Gt = similar(Dt)
        Gt_tmp = similar(Dt)
        Xt .= Z
        prox!(Xt)
        D!(Dt, Xt)
        Dt .*= γ
        projG!(Dt)
        new{RT}(copy(Z), Xt, Dt, Gt, Gt_tmp, Ref(RT(q0)), RT(τ), RT(γ), projG!, prox!)
    end
end

function TV1_FGP(Z::AbstractArray{T}, τ::Real, q0::Real=1;
                 prox!::Function=identity) where T
    TV_FGP(Z, τ, q0, proj_G1!; prox! = prox!)
end

struct TV_GP{T}
    Z
    Xt
    Gt
    Gt_tmp
    τ :: T
    γ :: T
    projG! :: Function
    prox! :: Function

    function TV_GP(Z::AbstractArray{T}, τ::Real,
                   projG!::Function=proj_G2!;
                   prox!::Function=identity) where T
        RT = real(T)
        @assert τ > 0
        γ = 1/(12τ)
        Xt = similar(Z)
        Gt = similar(Z, (size(Z)..., ndims(Z)))
        Gt_tmp = similar(Gt)
        Xt .= Z
        prox!(Xt)
        D!(Gt_tmp, Xt)
        Gt_tmp .*= γ
        projG!(Gt_tmp)
        new{RT}(copy(Z), Xt, Gt, Gt_tmp, RT(τ), RT(γ), projG!, prox!)
    end
end

function TV1_GP(Z::AbstractArray{T}, τ::Real;
                 prox!::Function=identity) where T
    TV_GP(Z, τ, proj_G1!; prox! = prox!)
end

function TV_step(tv::TV_FGP)
    Dᵀ!(tv.Xt, tv.Dt)
    tv.Xt .= tv.Z .- tv.τ.*tv.Xt
    tv.prox!(tv.Xt)
    D!(tv.Gt, tv.Xt)
    tv.Gt .= tv.Dt .+ tv.γ.*tv.Gt
    tv.projG!(tv.Gt)
    
    Dᵀ!(tv.Xt, tv.Gt)
    tv.Xt .= tv.Z .- tv.τ.*tv.Xt
    tv.prox!(tv.Xt)
    
    qt_tmp = tv.qt[]
    tv.qt[] = (1 + sqrt(1+4*tv.qt[]^2))/2
    β = (qt_tmp - 1)/tv.qt[]
    tv.Dt .= tv.Gt .+ β.*(tv.Gt .- tv.Gt_tmp)
    tv.Gt_tmp .= tv.Gt
    tv.Xt
end

function TV_step(tv::TV_GP)
    Dᵀ!(tv.Xt, tv.Gt_tmp)
    tv.Xt .= tv.Z .- tv.τ.*tv.Xt
    tv.prox!(tv.Xt)
    D!(tv.Gt, tv.Xt)
    tv.Gt .= tv.Gt_tmp .+ tv.γ.*tv.Gt
    tv.projG!(tv.Gt)
    
    Dᵀ!(tv.Xt, tv.Gt)
    tv.Xt .= tv.Z .- tv.τ.*tv.Xt
    tv.prox!(tv.Xt)
    
    tv.Gt_tmp .= tv.Gt
    tv.Xt
end

function initialize!(tv::TV_FGP, Z::AbstractArray{T}) where T
    tv.qt[] = 1
    tv.Z .= Z
    tv.Xt .= Z
    tv.prox!(tv.Xt)
    D!(tv.Dt, tv.Xt)
    tv.Dt .*= tv.γ
    tv.projG!(tv.Dt)
end

function initialize!(tv::TV_GP, Z::AbstractArray{T}) where T
    tv.Z .= Z
    tv.Xt .= Z
    tv.prox!(tv.Xt)
    D!(tv.Gt_tmp, tv.Xt)
    tv.Gt_tmp .*= tv.γ
    tv.projG!(tv.Gt_tmp)
end

function iter_TV_prox!(tv::Union{TV_GP,TV_FGP},
                       X::AbstractArray{T}, n_iter::Int) where T
    initialize!(tv, X)
    for i in 1:n_iter
        TV_step(tv)
    end
    X .= tv.Xt
end

function get_FTV_prox!(Z::AbstractArray{T}, τ::Real;
                       n_iter::Int=50,
                       prox!::Function=identity) where T
    tv = TV_FGP(Z, τ, prox! = prox!)
    X -> iter_TV_prox!(tv, X, n_iter)
end

function get_FTV1_prox!(Z::AbstractArray{T}, τ::Real;
                       n_iter::Int=50,
                       prox!::Function=identity) where T
    tv = TV1_FGP(Z, τ, prox! = prox!)
    X -> iter_TV_prox!(tv, X, n_iter)
end

function get_TV_prox!(Z::AbstractArray{T}, τ::Real;
                      n_iter::Int=50,
                      prox!::Function=identity) where T
    tv = TV_GP(Z, τ, prox! = prox!)
    X -> iter_TV_prox!(tv, X, n_iter)
end

function get_TV1_prox!(Z::AbstractArray{T}, τ::Real;
                       n_iter::Int=50,
                       prox!::Function=identity) where T
    tv = TV1_GP(Z, τ, prox! = prox!)
    X -> iter_TV_prox!(tv, X, n_iter)
end
