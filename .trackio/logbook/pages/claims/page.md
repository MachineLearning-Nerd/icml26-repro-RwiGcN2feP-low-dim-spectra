# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6bb886aaa3db", "created_at": "2026-07-22T08:04:04+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Theorem 4.1 proves the layer-wise Hessian in the L-deep linear unconstrained features model (UFM) has rank K^2 with all non-zero eigenvalues equal, analytically reproducing the bulk-outlier Hessian spectrum reported in prior empirical studies (Theorem 4.1).
2. Theorem 4.2 decomposes the Gauss-Newton/Fisher Information component into three terms: G_within (rank 0), G_cross (rank K(K-1), producing the mini-bulk of eigenvalues), and G_class (rank K, producing the main outliers), mirroring Papyan's empirical knockout experiments (Theorem 4.2).
3. Theorem 4.3 shows the aggregated gradient update is a sum over only K of the K^2 possible eigenvector directions, each with equal coefficient beta^(l+1)/K, explaining the observed gradient alignment with a low-dimensional subspace (Theorem 4.3).
4. Theorem 4.4 proves the Gram matrix of the optimal weights, W_l*^T W_l*, has rank K with eigenvalues proportional to the squared norms of the class-mean features, giving a closed-form account of the low-rank weight structure (Theorem 4.4).
5. For a deep linear UFM with K=3 classes, numerical experiments show K^2=9 Hessian outliers separating from the bulk and converging to equal eigenvalues over training, with eigenvector alignment metric f_cc' rising from about 0.2 to 1.0 (Figures 3 and 4).
6. In the non-linear (ReLU) Deep UFM, K^2=9 Hessian outliers separate but do not fully converge to equal values, and the gradient has K non-zero coefficients that remain unequal, unlike the linear case (Figure 9, Table 2).
