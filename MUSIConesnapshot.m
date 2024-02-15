
function x = MUSIConesnapshot(r, target_num, doa_grid)
vecH = @(MAT) MAT(:).';
vec = @(MAT) MAT(:);
d = 0.5;
steer = @(ang, antnum) exp(1j*2*pi*[0:antnum-1].'*d*sind(ang(:).'));
r = r(:);
N = length(r);
ant_num_new = 4;

L = 8;
hankel_mat = zeros(L, N-L+1);
for idx = 1:L
    hankel_mat(idx,:) = vecH(r(idx:idx+N-L));
end

[U,D,V] = svd(hankel_mat);

D = abs(diag(D));
[~,sort_idx] = sort(D,'descend');

U1 = U(:, sort_idx(1:target_num));
U2 = U(:, sort_idx(1+target_num:end));

ang_range = doa_grid(:);
dic_mat = steer(ang_range, L);

sp = vec(sum(abs(dic_mat).^2,1))./vec(sum(abs(dic_mat'*U2).^2, 2));
sp = sp/max(sp);
x = vecH(sp);
% figure; plot(ang_range, 10*log10(x)); hold on; stem(theta, zeros(3,1), 'BaseValue', -100);
