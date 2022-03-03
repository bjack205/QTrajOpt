real2complex(a::AbstractMatrix) = a[1] + a[2]*1im
complex2real(c::Complex) = SA[real(c) -imag(c); imag(c) real(c)]

@generated function real2complex(x::StaticVector{n}) where n
    @assert iseven(n)
    m = n ÷ 2
    c = [:(Complex(x[$i], x[$(m+i)])) for i = 1:m]
    :(SVector{$m}($(c...)))
end

@generated function complex2real(c::StaticVector{m}) where m
    n = 2m
    re = [:(real(c[$i])) for i = 1:m]
    im = [:(imag(c[$(i)])) for i = 1:m]
    :(SVector{$n}($(re...), $(im...)))
end

@generated function real2complex(A::StaticMatrix{n,n}) where n
    @assert iseven(n)
    m = n ÷ 2
    cij = vec([:(Complex(A[$i,$(j)], A[$(i+m),$j])) for i=1:m, j=1:m])
    :(SMatrix{$m,$m}($(cij...)))
end
   
@generated function complex2real(C::StaticMatrix{m,m}) where m
    n = 2m
    re = [:(real(C[$i,$j])) for i=1:m, j=1:m]
    im = [:(imag(C[$i,$j])) for i=1:m, j=1:m]
    ne = [:(-imag(C[$i,$j])) for i=1:m, j=1:m]
    A = vec([re ne; im re])
    :(SMatrix{$n,$n}($(A...)))
end