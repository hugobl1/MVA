function value=f(Q,p,A,b,t,v)
    value=t*transpose(v)*Q*v+t*transpose(p)*v-sum(log(-A*v+b));
end