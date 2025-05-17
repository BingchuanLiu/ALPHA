function [result, rm] = alpha_test(rawdata,N,freq,model,latency)

num_r_use = 3;
nharmonics = 5;
n_cond = length(freq);
X = rawdata(:,1+latency:N+latency)';
X = X - mean(X,1);

template = model.template;
A_bar = model.A_bar;
Cs = model.Cs;
condY = model.condY;
Wtrca = model.Wtrca;
Wlda = model.Wlda;
w_source = model.w_source;

v = ones(N,1);
Center_t = eye(N) - v*v'/N;

weight = 2;

Ct = X'*Center_t*X/(size(X,1)-1);
n_k = 8;
rm = zeros(5,n_cond,n_k);

for cond = 1:n_cond
    X_bar = template(:,:,cond)';
    Y = condY{cond}(:,1:2*nharmonics);
    [Wx,~,r1] = cca_m3m(X,Y);
    A_test = inv(Wx)';
    P0 = A_bar(:,:,cond)'*A_test;
    [u,~,v] = svd(P0);
    P = u*v';
    W_adapt = Wx*P';
    Q(:,:,cond) = Cs(:,:,cond)^(-1/2)*Ct^(1/2);
    rm(1,cond,:) = r1(1:n_k);
    
    Xw = X*Wx(:,1:n_k);
    Xbar_w = X_bar*Q(:,:,cond)*Wx(:,1:n_k);
    rm(2,cond,:) = diag(corr(Xw,Xbar_w));

    [W_bar,~,~] = cca_m3m(X_bar,Y);
    Xbar_wbar = X_bar*W_bar(:,1:n_k); 
    X_wbar = X*W_adapt(:,1:n_k);% (W_bar no longer applies to X due to domain gap) Appromixate W_bar by A_test*P'
    rm(3,cond,:) = diag(corr(Xbar_wbar,X_wbar));  

    Xbar_trca = X_bar*Wlda(:,1:n_k);

    X_trca = X/Q(:,:,cond)*Wlda(:,1:n_k);
    rm(5,cond,:) = diag(corr(Xbar_trca,X_trca));   
    
    [A,B,~] = cca_m3m(X, X_bar);
    A_test = inv(A)';
    P0 = model.A_trca(:,:,cond)'*A_test;
    [u,~,v] = svd(P0);
    P = u*v';  
    W_adapt1(:,:,cond) = A*P';
end
for cond_query = 1:n_cond
    X_bar = template(:,:,cond_query)';
    for k = 1 : n_k 
        Xw = X*squeeze(W_adapt1(:,k,:));
        Xbar_w = X_bar*squeeze(Wtrca(:,k,:));
        rm(4,cond_query,k) = corr2_new(Xw,Xbar_w);
    end
end
rm = sign(rm).*(abs(rm).^weight);         
rrm = squeeze(sum(rm,1));
logits = rrm(:,1:num_r_use);
rr = diag(logits*w_source) - model.nt_rm_source;
result = find(rr==max(rr)); 
end