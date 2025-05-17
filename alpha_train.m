function [model, rrm] = alpha_train(data,N,freq,latency,rfs)

num_r_use = 3;
nharmonics = 5;
n_cond = length(freq);
condY = ccaReference(N, rfs, freq);
datum = data(:,1+latency:N+latency,:,:);
datum0 = datum;
datum = datum - mean(datum,2);
n_block = size(datum,3);
weight = 2;
n_k = 8;
rrm = zeros(n_block,n_cond,5,n_cond,n_k);

v = ones(N,1);
Center_t = eye(N) - v*v'/N;
v = ones(N*(n_block-1),1);
Center_s = eye(N*(n_block-1)) - v*v'/(N*(n_block-1));
% Train for cv
for cv = 1 : n_block
    train_idx = setdiff([1:n_block],cv);
    target = squeeze(datum(:,:,cv,:));
    Xbart_all = datum0(:,:,train_idx,:);
    template = squeeze(mean(datum(:,:,train_idx,:),3));
    
    trca_Xm = template(:,:,:);
    trca_Xma = mean(trca_Xm,3);
    trca_Xmb = trca_Xm - trca_Xma;
    Hb = trca_Xmb(:,:)/sqrt(n_cond);
    Xmc = Xbart_all;
    Xmc = Xmc - mean(Xmc,2);
    Xmca = Xmc - mean(Xmc,3);
    Hw = Xmca(:,:)/sqrt((n_block-1)*n_cond);    
    Sw = Hw*Hw';
    Sb = Hb*Hb';
    [V,D] = eig(Sw\Sb);
    [~,index] = sort(diag(D),'descend');
    Wlda = V(:,index);
        
    for cond = 1 : n_cond

        X = datum0(:,:,cv,cond)';
        Ct = X'*Center_t*X/(size(X,1)-1);
        for cond_query = 1:n_cond
            X_bar = template(:,:,cond_query);
            X_barm(:,:,cond_query) = repmat(X_bar, [1,n_block-1])';
            Xbart_cond = Xbart_all(:,:,:,cond_query);
            Xbart(:,:,cond_query) = Xbart_cond(:,:)';
            Y = condY{cond_query}(:,1:2*nharmonics);
            [W_xbar,~,~] = cca_m3m(X_bar',Y);
            [Wx,~,~] = cca_m3m(X,Y);
            W(:,:,cond_query) = Wx;
            A_bar = inv(W_xbar)';
            A_test = inv(W(:,:,cond_query))';
            P0 = A_bar'*A_test;
            [u,~,v] = svd(P0);
            P = u*v';

            W_adapt(:,:,cond_query) = W(:,:,cond_query)*P';
            X1 = Xbart(:,:,cond_query);
            Cs = X1'*Center_s*X1/(size(X1,1)-1);
            Q(:,:,cond_query) = Cs^(-1/2)*Ct^(1/2);
            [A,~,~] = cca_m3m(X_barm(:,:,cond_query), Xbart(:,:,cond_query));
            Wtrca(:,:,cond_query) = A;
            A_trca(:,:,cond_query) = inv(Wtrca(:,:,cond_query))';
        end

        rm = zeros(5,n_cond,n_k);
        for cond_query = 1:n_cond
            Y = condY{cond_query}(:,1:2*nharmonics);
            X_bar = template(:,:,cond_query)';
            [Wx,~,r1] = cca_m3m(X,Y);
            rm(1,cond_query,:) = r1(1:n_k);
            
            Xw = X*Wx(:,1:n_k);
            Xbar_w = X_bar*Q(:,:,cond_query)*W(:,1:n_k,cond_query);
            rm(2,cond_query,:) = diag(corr(Xw,Xbar_w));
            
            [W_bar,~,~] = cca_m3m(X_bar,Y);
            Xbar_wbar = X_bar*W_bar(:,1:n_k); 
            X_wbar = X*W_adapt(:,1:n_k,cond_query);% (W_bar no longer applies to X due to domain gap) Appromixate W_bar by A_test*P'
            rm(3,cond_query,:) = diag(corr(Xbar_wbar,X_wbar));  

            Xbar_trca = X_bar*Wlda(:,1:n_k);
            X_trca = X/Q(:,:,cond_query)*Wlda(:,1:n_k);
            rm(5,cond_query,:) = diag(corr(Xbar_trca,X_trca));     
            
            [A,B,~] = cca_m3m(X, X_bar);
            A_test = inv(A)';
            P0 = A_trca(:,:,cond_query)'*A_test;
            [u,~,v] = svd(P0);
            P = u*v';  
            W_adapt1(:,:,cond_query) = A*P';
        end
        for cond_query = 1:n_cond
            X_bar = template(:,:,cond_query)';
            for k = 1 : n_k 
                Xw = X*squeeze(W_adapt1(:,k,:));
                Xbar_w = X_bar*squeeze(Wtrca(:,k,:));
                rm(4,cond_query,k) = corr2_new(Xw,Xbar_w);
            end
        end
        
        rm = sign(rm).*abs(rm).^weight;      
        rrm(cv,cond,:,:,:) = rm;
    end
end

v = ones(N*n_block,1);
Center_s = eye(N*n_block) - v*v'/(N*n_block);
% Train for target
template = squeeze(mean(datum,3));
Xbart_all = datum0;
Xbart = [];
for cond = 1:n_cond
    X_bar = template(:,:,cond);
    X_barm = repmat(X_bar, [1,n_block])';
    Xbart_cond = Xbart_all(:,:,:,cond);
    Xbart(:,:,cond) = Xbart_cond(:,:)';
    Y = condY{cond}(:,1:2*nharmonics);
    [W_bar,~,~] = cca_m3m(X_bar',Y);
    W_xbar(:,:,cond) = W_bar;
    A_bar(:,:,cond) = inv(W_xbar(:,:,cond))';
    X1 = Xbart(:,:,cond);
    Cs(:,:,cond) = X1'*Center_s*X1/(size(X1,1)-1);
    [A,B,r] = cca_m3m(X_barm, Xbart(:,:,cond));
    Wtrca(:,:,cond) = A;
    A_trca(:,:,cond) = inv(Wtrca(:,:,cond))';
end
trca_Xm = template;% 9x50x40
trca_Xm = trca_Xm - mean(trca_Xm,2);
trca_Xma = mean(trca_Xm,3);% 9x50
trca_Xmb = trca_Xm - trca_Xma;
Hb = trca_Xmb(:,:)/sqrt(n_cond);
Xmc = Xbart_all;
Xmc = Xmc - mean(Xmc,2);
Xmca = Xmc - mean(Xmc,3);
Hw = Xmca(:,:)/sqrt(n_block*n_cond);    
Sw = Hw*Hw';
Sb = Hb*Hb';
[V,D] = eig(Sw\Sb);
[~,index] = sort(diag(D),'descend');
Wlda = V(:,index);

% 10x12x5x12x8
for cond = 1 : n_cond
    ym = eye(n_cond);
    rrrm = squeeze(sum(rrm(:,cond,:,:,:),3));% 10x12x12x8
    H = zeros(n_block,n_block*n_cond);
    k = 1;
    for block = 1 : n_block
        H(k,n_cond*(k-1)+1:n_cond*k) = ym(cond,:);% 120X1440
        k = k + 1;
    end
    rrrm1 = permute(rrrm,[2,1,3]);
    logitsm = reshape(rrrm1(:,:,1:num_r_use),[n_block*n_cond,num_r_use]);
    Hx = H*logitsm;%120x8, rho value at the location of ground truth label
    S = Hx'*Hx;
    [V, D] = eig(S);
    [~,index] = sort(diag(D),'descend');
    V = V(:,index);
    w = V(:,1);
    [~,idx] = max(abs(w));
    signs = sign(w(idx));
    w = w * signs;
    wm(:,cond) = w;
end
nt_rm = [];
for block = 1 : n_block
    for cond = 1 : n_cond
        logits = squeeze(sum(rrm(block,cond,:,:,1:num_r_use),3))*wm(:,cond); 
        logits(cond) = 0;
        nt_rm(:,block,cond) = logits;
    end
end
nt_rm_source = squeeze(mean(nt_rm,[2,3]));
model.w_source = wm;
model.template = template;
model.condY = condY;
model.A_bar = A_bar;
model.Cs = Cs;
model.Wtrca = Wtrca;
model.Wlda = Wlda;
model.Xbart = Xbart;
model.A_trca = A_trca;
model.nt_rm_source = nt_rm_source;
end