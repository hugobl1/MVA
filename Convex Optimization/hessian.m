function hes=hessian(t,Q,A,b,v)
    N=size(A,1);
    hes=2*t*Q+transpose(A)*(A./((A*v-b).^2));
end