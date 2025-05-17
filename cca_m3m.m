function [A,B,r] = cca_m3m(X,Y)
[n,~] = size(X);    
X = X - mean(X,1); 
Y = Y - mean(Y,1);
[Q1,R1,perm1] = qr(X,0);   
[Q2,R2,perm2] = qr(Y,0);  
[U,D,V] = svd(Q1' * Q2,0);   
r = diag(D);                  
A = R1\U*sqrt(n-1);     
B = R2\V*sqrt(n-1);
A(perm1,:) = A;                
B(perm2,:) = B;
end
