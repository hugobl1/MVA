%Variables initiales
n=3;
d=20;
lambda=10;
X=rand(n,d);
y=rand(n,1);

%Variables intermédiaires
Q=0.5*eye(n);
A=[transpose(X) ;-transpose(X)];
b=lambda*ones(2*d,1);
p=y;
v0=A\b-0.5;
eps=1e-6;

mu=10;
%Barrier Method
[v_seq,n_itermoyen]=barr_method(Q,p,A,b,v0,eps,mu);

bestv=v_seq(:,size(v_seq,2));
criteria=[];
bestf=f0(Q,p,bestv);
for i=1:size(v_seq,2)
    criteria=[criteria (f0(Q,p,v_seq(:,i))-bestf)];
end

x=[0:0.1:size(v_seq,2)];
semilogy(x,eps+x*0,"r");
hold all
%semilogy(1:size(criteria,2), log(criteria));
stairs(1:size(criteria,2),criteria);
set(gca,'YScale','log');

xlabel('Itérations') 
ylabel('f(v_t)-f^{*}') 
legend('precision criterion','\mu=10'); 
grid on

bestw=X\(bestv-y);
disp("La valeur de la solution optimal w_* est: ")
disp(bestw)

nbzeros=size(find(bestw==0));


fprintf('Nombre de coefficients nuls dans w: %d \n',nbzeros(1));


Mu=[2,7,15,50,100,200,500];
ln_itermoyen=[];
for i = 1:length(Mu)
	%Barrier Method
    [v_seq,n_itermoyen]=barr_method(Q,p,A,b,v0,eps,Mu(i));
    ln_itermoyen=[ln_itermoyen  n_itermoyen];
    bestv=v_seq(:,size(v_seq,2));
    criteria=[];
    bestf=f0(Q,p,bestv);
    for i=1:size(v_seq,2)
        criteria=[criteria (f0(Q,p,v_seq(:,i))-bestf)];
    end

    %semilogy(1:size(criteria,2), log(criteria));
    stairs(1:size(criteria,2),criteria);
    set(gca,'YScale','log');
    hold on
    bestw=X\(bestv-y);
    disp("La valeur de la solution optimal w_* est: ")
    disp(bestw)
    nbzeros=size(find(bestw==0));
    fprintf('Nombre de coefficients nuls dans w: %d \n',nbzeros(1));
end
x=[0:0.1:26];
semilogy(x,eps+x*0,"r");
xlabel('Itérations') 
ylabel('f(v_t)-f^{*}') 
legend('\mu=2','\mu=7','\mu=15','\mu=50','\mu=100','\mu=200','\mu=500','precision criterion');
grid on
hold off

plot(Mu,ln_itermoyen)
xlabel('Valeur de \mu') 
ylabel('Nombre moyen d''itérations') 
