function [i,v_seq]=centering_step(Q,p,A,b,t,v0,eps)
    v_seq=[v0];
    alpha=0.25;
    beta=0.5;
    %%Calcul du hessien et du gradient initiaux
    grad=gradient(Q,p,A,b,t,v0);
    hes=hessian(t,Q,A,b,v0);
    invhes=inv(hes);
    xnew=-invhes*grad;
    lambda2=transpose(grad)*invhes*grad;
    test=eps-(lambda2/2);
    v=v0;
    %%Test d'arrêt
    i=1;
    while(test<0)
        %%Choix du pas à utiliser
        s=1;
        value=f(Q,p,A,b,t,v);
        nextvalue=f(Q,p,A,b,t,v+s*xnew);
        while(nextvalue>=(value+alpha*s*transpose(grad)*xnew))
            s=beta*s;
            nextvalue=f(Q,p,A,b,t,v+s*xnew);      
        end
        v=v+s*xnew;
        v_seq=[v_seq v];
        %%Calcul du hessien et du gradient actuel
        grad=gradient(Q,p,A,b,t,v);
        hes=hessian(t,Q,A,b,v);
        invhes=inv(hes);
        xnew=-invhes*grad;
        lambda2=transpose(grad)*invhes*grad;
        test=eps-(lambda2/2);
        i=i+1;
    end
end