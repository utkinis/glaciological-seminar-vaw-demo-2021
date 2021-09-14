const USE_GPU = true

const outdir = "sia_hershel_bulkley_bn_0.5"

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using Printf
using Plots
using MAT

@views function run()
    # physics
    npow    = 3.0 # Glen's power law exponent
    lz      = 1.0 # m
    a_ρg_n  = 1.0 # m^(-npow)*s^(-1)
    # scales
    tsc     = lz^(-npow) / a_ρg_n # s
    dsc     = lz^2 / tsc          # m^2/s
    # non-dimensional
    lx_lz   = 100.0
    ly_lz   = 150.0
    w_lz    = 10.0
    theta   = π / 18
    bn      = 0.5  # bn = τY_ρg/Lz - Bingham number
    H0_lz   = 25
    # dependent physics
    lx      = lx_lz * lz
    ly      = ly_lz * lz
    w       = w_lz * lz
    ttot    = 2e2 * tsc
    τY_ρg   = bn * lz
    # numerics
    nx      = 512
    ny      = ceil(Int, ly_lz / lx_lz * nx)
    niter   = 2 * max(nx, ny)
    eiter   = 1e-12
    nout    = 10
    ncheck  = ceil(Int, 0.2*nx)
    CFL     = 0.5 / sqrt(2)
    Resc    = 1.2
    # preprocessing
    dx, dy  = lx / (nx - 1), ly / (ny - 1)
    x, y    = LinRange(-lx / 2, lx / 2, nx), LinRange(-ly / 2, ly / 2, ny)
    Vpdt    = CFL * min(dx, dy)
    dtmin   = 1e-8 * tsc
    dtmax   = 1e0 * tsc
    dt0     = min(2*dtmin, dtmax)
    max_lxy = max(lx, ly)
    # init
    H0      = Data.Array(H0_lz * lz * exp.(-(x ./ w).^2 .- ((y .- ly/4)' ./ w).^2))
    B0      = Data.Array(y' .* tan(theta) .- 5lz .* cos.(2π .* x ./ lx) .+ 2lz .* sin.(7π .* y' ./ ly))
    H       = @ones(nx,ny) .* H0
    H_old   = @zeros(nx,ny)
    dH      = @zeros(nx,ny)
    B       = @ones(nx,ny) .* B0
    S       = B .+ H
    dSdx    = @zeros(nx - 1,ny)
    dSdy    = @zeros(nx,ny - 1)
    Deff    = @zeros(nx - 1,ny - 1)
    Deffm   = @zeros(nx - 1,ny - 1)
    Yi      = @zeros(nx - 1,ny - 1)
    Qx      = @zeros(nx - 1,ny - 2)
    Qy      = @zeros(nx - 2,ny - 1)
    Qx_old  = @zeros(nx - 1,ny - 2)
    Qy_old  = @zeros(nx - 2,ny - 1)
    # static data
    if isdir(outdir)
        rm(outdir, recursive=true, force=true)
    end
    mkdir(outdir)
    matwrite("$outdir/static.mat", Dict(
        "npow" => npow, "ArhogN" => a_ρg_n, "theta" => theta, "Bn" => bn,
        "Ly_Lz" => lx_lz, "Ly_Lz" => ly_lz, "w_Lz"  => w_lz, "H0_Lz" => w_lz,
        "ttot" => ttot,
        "Lx" => lx, "Ly"=> ly, "Lz" => lz,
        "tauY_rhog" =>τY_ρg,
        "nx" => nx, "ny" => ny,
        "dx" => dx, "dy" => dy,
        "niter" => niter, "eiter" => eiter,
        "x" => Array(x), "y" => Array(y),
        "B" => Array(B), "H0" => Array(H0)
    ))
    # action
    t      = 0.0
    it     = 0
    dt     = dt0
    while t < ttot
        @printf "  # it = %d\n" it
        iter = 0; err = Inf
        H_old  .= H
        Qx_old .= Qx
        Qy_old .= Qy
        while iter < niter && err > eiter
            if iter % ncheck == 0 dH .= H end
            @parallel compute_slope!(dSdx, dSdy, S, dx, dy)
            @parallel compute_Deff!(Deff, H, S, Yi, dSdx, dSdy, npow, a_ρg_n, τY_ρg, dx, dy)
            @parallel compute_Deffm!(Deffm, Deff)
            @parallel update_fluxes!(Qx, Qy, dSdx, dSdy, Deff, Deffm, Vpdt, Resc, max_lxy, dsc, dt, dx, dy)
            @parallel update_surface!(H, H_old, S, B, Qx, Qy, Deff, Deffm, Vpdt, Resc, max_lxy, dsc, dt, dx, dy)
            S .= B .+ H
            if iter % ncheck == 0
                dH .= dH .- H
                errsc = maximum(H)
                err = maximum(abs.(dH))/errsc
                if !isfinite(err) || !isfinite(errsc) || isapprox(err, 0.0; atol = 1e-18); err = NaN; break end
                @printf "    # iter/ndof = %.3f, err = %g\n" iter/max(nx,ny) err 
            end
            iter += 1
        end
        if err > eiter || isnan(err)
            @printf "    Recalculating, didn't converge...\n"
            H  .= H_old
            Qx .= Qx_old
            Qy .= Qy_old
            S .= B .+ H
            dt /= 2.0
            if dt < dtmin error("sim failed") end
            continue
        end
        t  += dt; it += 1
        dt = min(dt*1.1, dtmax)
        if it % nout == 0
            display(heatmap(x, y, Array(H'),
                        aspect_ratio=1, framestyle=:box,
                        xlims=(x[1],x[end]), ylims=(y[1],y[end]),
                        xlabel="x", ylabel="y", color=:jet,
                        title = @sprintf "# it = %d, t = %g/%g, dt = %g" it t ttot dt))
            matwrite("$outdir/step_$it.mat", Dict(
                "H" => Array(H), "S" => Array(S), "Yi" => Array(Yi),
                "dt" => dt, "time" => t, "iter" => iter
            ); compress=true)
        end
        end
end

macro gam()      esc(:( sqrt(@av_ya(dSdx)^2 + @av_xa(dSdy)^2)                                     )) end
macro Re_ax()    esc(:( π + sqrt(π * π + max_lxy * max_lxy / max(@av_ya(Deffm), 1e-5 * dsc) / dt) )) end
macro Re_ay()    esc(:( π + sqrt(π * π + max_lxy * max_lxy / max(@av_xa(Deffm), 1e-5 * dsc) / dt) )) end
macro Re_av()    esc(:( π + sqrt(π * π + max_lxy * max_lxy / max(@av(Deffm), 1e-5 * dsc) / dt)    )) end
macro τr_dτ_ax() esc(:( max_lxy / Vpdt / @Re_ax() / Resc                                          )) end
macro τr_dτ_ay() esc(:( max_lxy / Vpdt / @Re_ay() / Resc                                          )) end
macro dτ_ρ()     esc(:( Vpdt * max_lxy / max(@av(Deffm), 1e-5 * dsc) / @Re_av() / Resc            )) end
macro divQ()     esc(:( @d_xa(Qx) / dx + @d_ya(Qy) / dy                                           )) end

@parallel function compute_slope!(dSdx, dSdy, S, dx, dy)
    @all(dSdx) = @d_xa(S) / dx
    @all(dSdy) = @d_ya(S) / dy
    return
end

@parallel function compute_Deff!(Deff, H, S, Yi, dSdx, dSdy, npow, a_ρg_n, τY_ρg, dx, dy)
    @all(Yi)   = max(@av(H) - τY_ρg / @gam(), 0.0)
    @all(Deff) = a_ρg_n * @all(Yi)^(npow + 1.0) * (@av(H) * (npow + 2.0) - @all(Yi)) * @gam()^(npow - 1.0) / (npow + 1.0)
    return
end

@parallel function compute_Deffm!(Deffm, Deff)
    @inn(Deffm) = @maxloc(Deff)
    return
end

@parallel function update_fluxes!(Qx, Qy, dSdx, dSdy, Deff, Deffm, Vpdt, Resc, max_lxy, dsc, dt, dx, dy)
    @all(Qx) = (@all(Qx) * @τr_dτ_ax() - @av_ya(Deff) * @inn_y(dSdx)) / (1.0 + @τr_dτ_ax())
    @all(Qy) = (@all(Qy) * @τr_dτ_ay() - @av_xa(Deff) * @inn_x(dSdy)) / (1.0 + @τr_dτ_ay())
    return
end

@parallel function update_surface!(H, Hold, S, B, Qx, Qy, Deff, Deffm, Vpdt, Resc, max_lxy, dsc, dt, dx, dy)
    @inn(H) = (@inn(H) + @dτ_ρ() * (@inn(Hold) / dt - @divQ())) / (1.0 + @dτ_ρ() / dt)
    return
end

run()