mutable struct Gradient{T}
    L :: T
end

mutable struct Fista{T <: Real, AType <: AbstractArray{T}}
    xk :: AType
    xk_tmp :: AType
    tk :: T
    L :: T
    function Fista(A::AbstractArray{T}, L::Real, tk::Real=1) where {T <: Real}
        new{T, typeof(A)}(similar(A), copy(A), T(tk), T(L))
    end
end

mutable struct MFista{T <: Real, AType <: AbstractArray{T}}
    zk :: AType
    xk :: AType
    xk_tmp :: AType
    β :: T
    L :: T
    F :: Function
    F_tmp :: T
    function MFista(A::AbstractArray{T}, L::Real,
                    F::Function, β::Real=1) where {T <: Real}
        new{T, typeof(A)}(similar(A), similar(A), copy(A),
                          T(β), T(L), F, T(Inf))
    end
end

function Base.similar(t::NTuple{N,M where M}) where N
    ([ similar(t[i]) for i in 1:N]...,)
end

function optim_restart(method)
    nothing
end

function optim_restart(method::Fista{T,AType}) where {T <: Real,
                                                AType <: AbstractArray{T}}
    method.tk = 1
    nothing
end

function optim_restart(method::NTuple{N,M where M}) where N
    for i in 1:N
        optim_restart(method[i])
    end
    nothing
end

function gradient_step!(A, ∇A, prox!, method::Nothing)
    A
end

function gradient_step!(A, ∇A::Nothing, prox!, method)
    A
end

function gradient_step!(A::AbstractArray{T},
                        ∇A::AbstractArray{T},
                        prox!::Function,
                        method::Gradient) where {T <: Real}
    A .-= (1/method.L).*∇A
    prox!(A)
end

function gradient_step!(A::AbstractArray{T},
                        ∇A::AbstractArray{T},
                        prox!::Function,
                        method::Fista{T,AType}) where {T <: Real,
                                                       AType <: AbstractArray{T}}
    method.xk .= A .- (1/method.L).*∇A
    prox!(method.xk)
    tk = (1+sqrt(1+4*method.tk^2))/2
    β = (method.tk-1)/tk
    method.tk = tk
    A .= method.xk .+ β.*(method.xk .- method.xk_tmp)
    method.xk_tmp .= method.xk
end

function gradient_step!(A::AbstractArray{T},
                        ∇A::AbstractArray{T},
                        prox!::Function,
                        method::MFista{T,AType}) where {T <: Real,
                                                        AType <: AbstractArray{T}}
    # Different than original Monotone Fista
    method.zk .= A .- (1/method.L).*∇A
    prox!(method.zk)
    method.xk .= method.zk
    Fk = method.F(method.xk)
    if Fk <= method.F_tmp
        method.F_tmp = Fk
        A .= method.zk .+ method.β.*(method.zk .- method.xk_tmp)
        method.xk_tmp .= method.zk
    else
        A .= method.xk_tmp
    end
end

function proximal_step!(A::AbstractArray{T},
                        ∇A::AbstractArray{T},
                        env,
                        uf,
                        compute_fwd!::Function,
                        compute_grad!::Function,
                        method,
                        prox!::Union{Nothing,Function}=nothing) where {T <: Real}
    compute_grad!(env, A, ∇A, uf)
    proxop! = if prox! == nothing; identity else prox! end
    gradient_step!(A, ∇A, proxop!, method)
    uf = compute_fwd!(env, A)
    return uf
end

function proximal_step!(A::NTuple{N,AbstractArray{T}},
                        ∇A::NTuple{N,Union{Nothing,AbstractArray{T}}},
                        env,
                        uf,
                        compute_fwd!::Function,
                        compute_grad!::Function,
                        method::NTuple{N,M where M},
                        prox!::Union{Nothing,
                                     NTuple{N,Function}}=nothing) where {N, T <: Real}
    compute_grad!(env, A, ∇A, uf)
    for i in 1:N
        proxop! = if prox! == nothing; identity else prox![i] end
        gradient_step!(A[i], ∇A[i], proxop!, method[i])
    end
    uf = compute_fwd!(env, A)
    return uf
end

function proximal_iter!(A, ∇A, env,
                        compute_fwd!::Function,
                        compute_err!::Function,
                        compute_grad!::Function,
                        method, n::Int;
                        prox! = nothing, errors=[], print_errors=true,
                        restart=false)
    uf = compute_fwd!(env, A)
    err = compute_err!(env, uf)
    push!(errors, err)
    if print_errors
        println(err)
    end
    for k in 1:n
        err_tmp = err
        uf = proximal_step!(A, ∇A, env, uf,
                            compute_fwd!, compute_grad!, method, prox!)
        err = compute_err!(env, uf)
        push!(errors, err)
        if restart && err > err_tmp
            optim_restart(method)
        end
        if print_errors
            println(err)
        end
    end
    errors
end
