

function x = ANM(r)

r = r(:);
N = length(r);
t = 100; % 100; % 10; % 1e3;

cvx_solver sdpt3
cvx_begin sdp quiet
    variable G(N+1, N+1) hermitian;
    G>=0;
    G(N+1, N+1) == 1;
    trace(G) <= 1+t;
    for idx = 1:N-1
        sum(diag(G(1:N, 1:N), idx)) == 0;
    end 
    minimize(norm(r-G(1:N, 1+N)))
cvx_end
x = G(1:N, N+1);
x = x(:).';
