function grad=gradient(Q,p,A,b,t,v)
    N=size(A,1);
    grad=2*t*Q*v+t*p-transpose(sum(A./(A*v-b),1));
end