function [v_seq,n_itermoyen]=barr_method(Q,p,A,b,v0,eps,mu)
    v_seq=[v0];
    m=size(A,1);
    t0=1;
    t=t0;
    v=v0;
    %Possibilité de prendre une autre précision pour 
    % le centering step
    [i,seq]=centering_step(Q,p,A,b,t,v,eps);
    lN=[];
    lN=[lN i];
    v=seq(:,size(seq,2));
    v_seq=[v_seq v];
    test=m/t-eps;
    while(test>=0)
        t=mu*t;
        [i,seq]=centering_step(Q,p,A,b,t,v,eps);
        lN=[lN i];
        v=seq(:,size(seq,2));
        v_seq=[v_seq v];
        test=m/t-eps;
    end
    n_itermoyen=mean(lN);
end