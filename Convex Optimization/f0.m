function value=f0(Q,p,v)
    value=transpose(v)*Q*v+transpose(p)*v;
end