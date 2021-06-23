module ProxOptim

export Gradient, Fista, MFista
export proximal_iter!

export TV_FGP, TV1_FGP, TV_GP, TV1_GP
export iter_TV_prox!,
    get_FTV_prox!, get_FTV1_prox!,
    get_TV_prox!, get_TV1_prox!

include("proximal_optim.jl")
include("TV_prox.jl")

end
