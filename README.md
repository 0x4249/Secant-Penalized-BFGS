# The Secant Penalized BFGS (SP-BFGS) Paper
This Git repository contains the Julia code for running the numerical experiments discussed in the paper 
"Secant Penalized BFGS: A Noise Robust Quasi-Newton Method Via Penalizing The Secant Condition"
by Brian Irwin and Eldad Haber. The published paper can be found at [https://doi.org/10.1007/s10589-022-00448-x](https://doi.org/10.1007/s10589-022-00448-x) and a preprint can be found at [https://arxiv.org/abs/2010.01275](https://arxiv.org/abs/2010.01275). 


# The SP-BFGS Update
The secant penalized BFGS update generalizes the well-known BFGS update. Recall that if one defines $\rho_{k} = 1/\mathbf{s}_{k}^{\rm T} \mathbf{y}_{k}$, the BFGS update is
$$\mathbf{H}_{k+1} = \bigg ( \mathbf{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^{\rm T} \bigg ) \mathbf{H}_k \bigg ( \mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^{\rm T} \bigg ) + \rho_k \mathbf{s}_k \mathbf{s}_k^{\rm T}$$.



# Running The Code
To run the quadratic function experiments from Section 6.1 of the published paper, execute the files:
```
julia aggregate_spbfgs_optimize_noisy_quadratic.jl 
julia aggregate_bfgs_optimize_noisy_quadratic.jl
julia aggregate_gd_optimize_noisy_quadratic.jl 
```

To run the CUTEst experiments from Section 6.2 of the published paper, execute the files:
```
julia SPBFGS_test_table.jl
julia BFGS_test_table.jl
julia GD_test_table.jl
```

Do not forget to configure the relevant settings in each file before running it.

The numerical experiments code contained in this Git repository was originally tested using Julia 1.5.4 on a computer with the Ubuntu 20.04 LTS operating system installed.


# Citation
If you use this code, please cite the paper:

Irwin, B., Haber, E. Secant penalized BFGS: a noise robust quasi-Newton method via penalizing the secant condition. 
*Comput Optim Appl* **84**, 651-702 (2023). [https://doi.org/10.1007/s10589-022-00448-x](https://doi.org/10.1007/s10589-022-00448-x)

BibTeX: 
```
@article{irwin-spbfgs-2023,
    Author = {Brian Irwin and Eldad Haber},
    Title = {Secant penalized {BFGS}: a noise robust quasi-{N}ewton method via penalizing the secant condition},
    Journal = {Computational Optimization and Applications},
    Volume = {84},
    Pages = {651--702},
    Year = {2023},
    DOI = {10.1007/s10589-022-00448-x}
}
```


