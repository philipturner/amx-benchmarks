Credit: https://github.com/JuliaLang/julia/issues/42312#issuecomment-1490792020

Here's the script for mul!. Nothing profound here, but should be easy to modify for other functions.

You need the Random, BenchmarkTools, and Printf packages for this thing.

```julia
"""
testmm()
Script for testing AMX mul!

You have to restart Julia after running this if you want to return to
Open BLAS
"""
function testmm()
    Random.seed!(46071)
    nd = 6
    low = 8
    high = low + nd - 1
    dcol = 7
    topen = zeros(nd, dcol)
    tapple = zeros(nd)
#
# Make a place to put the data and put it therer
#
    MA = Vector(undef, nd)
    MB = Vector(undef, nd)
    MC = Vector(undef, nd)
    MA32 = Vector(undef, nd)
    MB32 = Vector(undef, nd)
    MC32 = Vector(undef, nd)
    for ip = 1:nd
        p = low + ip - 1
        N = 2^p
        topen[ip, 1] = N
        MA[ip] = rand(N, N)
        MB[ip] = rand(N, N)
        MC[ip] = zeros(N, N)
        MA32[ip] = rand(Float32, N, N)
        MB32[ip] = rand(Float32, N, N)
        MC32[ip] = zeros(Float32, N, N)
    end
#
# Open BLAS
#
    for ip = 1:nd
        p = low + ip - 1
        N = 2^p
        A = MA[ip]
        B = MB[ip]
        C = MC[ip]
        A32 = MA32[ip]
        B32 = MB32[ip]
        C32 = MC32[ip]
        topen[ip, 2] = @belapsed mul!($C, $A, $B)
        topen[ip, 5] = @belapsed mul!($C32, $A32, $B32)
    end
#
# Switch to AMX with LBT
#
    AddAcc(false)
#
# Accelerate
#
    for ip = 1:nd
        A = MA[ip]
        B = MB[ip]
        C = MC[ip]
        A32 = MA32[ip]
        B32 = MB32[ip]
        C32 = MC32[ip]
        topen[ip, 3] = @belapsed mul!($C, $A, $B)
        topen[ip, 6] = @belapsed mul!($C32, $A32, $B32)
    end
    topen[:, 4] = topen[:, 3] ./ topen[:, 2]
    topen[:, 7] = topen[:, 6] ./ topen[:, 5]
#
# Tabulate
#
    printf(fmt::String, args...) = @eval @printf($fmt, $(args...))
    sprintf(fmt::String, args...) = @eval @sprintf($fmt, $(args...))
    headers = ["N", "O-64", "A-64", "R-64", "O-32", "A-32", "R-32"]
    println("Test of mul!(C, A, B)")
    for i = 1:dcol
        @printf("%9s ", headers[i])
    end
    printf("\n")
    dformat = "%9d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n"
    for i = 1:nd
        printf(dformat, topen[i, :]...)
    end
    return topen
end
```
