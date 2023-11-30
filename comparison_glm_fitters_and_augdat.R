#_______________________________________________________________________________
# Comparison of augmented-data and traditional projection for Bernoulli family
#_______________________________________________________________________________

# Preparations ------------------------------------------------------------

pkg_fit <- "rstanarm" # May be replaced by "brms".
options(mc.cores = parallel::detectCores(logical = FALSE))
seed_dat <- 8541351
seed_fit <- 1140350788
seed_prj <- 457324
seed_vsel <- 583946653
seed_brms <- 2989346

# Data --------------------------------------------------------------------
# Based on: `test_glm_ridge.R`

set.seed(seed_dat)
nobsv <- 41
nterms_tru <- 10
x_tru <- matrix(rnorm(nobsv * nterms_tru), nobsv, nterms_tru)
b_tru <- runif(nterms_tru, min = -0.5, max = 0.5)
prb_tru <- binomial()$linkinv(x_tru %*% b_tru)
y <- rbinom(nobsv, size = 1, prob = prb_tru)
dat <- data.frame(y, x_tru)

# Reference model fit -----------------------------------------------------

if (pkg_fit == "rstanarm") {
  library(rstanarm)
  # Use rstanarm's default priors:
  fit <- stan_glm(y ~ .,
                  data = dat,
                  family = binomial(),
                  seed = seed_fit,
                  refresh = 0)
} else if (pkg_fit == "brms") {
  library(brms)
  # Use brms's default priors:
  fit <- brm(y ~ .,
             data = dat,
             family = bernoulli(),
             seed = seed_fit,
             refresh = 0)
} else {
  stop("Unrecognized `pkg_fit`.")
}

# Code using projpred -----------------------------------------------------

stopifnot(packageVersion("projpred") >= "2.4.0")
library(projpred)
options(projpred.verbose_project = FALSE)

# Project onto the submodel containing the 3 (truly) most relevant predictors:
idx_soltrms <- head(order(abs(b_tru), decreasing = TRUE), 3)
nms_soltrms <- paste0("X", idx_soltrms)

## Traditional projection -------------------------------------------------

if (pkg_fit == "rstanarm") {
  refm <- get_refmodel(fit)
} else {
  refm <- get_refmodel(fit, brms_seed = seed_brms)
}
expr_prj <- expression(assign(
  out_nm,
  project(refm,
          predictor_terms = nms_soltrms,
          ndraws = nprjdraws,
          seed = seed_prj,
          regul = 0)
))
expr_vs <- expression(assign(
  out_nm,
  varsel(refm,
         method = "L1",
         nclusters = nclusters_vsel,
         nclusters_pred = nclusters_pred_vsel,
         nterms_max = nterms_max_vsel,
         seed = seed_vsel,
         regul = 0,
         verbose = FALSE)
))
expr_vs_fw <- expression(assign(
  out_nm,
  varsel(refm,
         nclusters = nclusters_vsel,
         nclusters_pred = nclusters_pred_vsel,
         nterms_max = nterms_max_vsel,
         seed = seed_vsel,
         regul = 0,
         verbose = FALSE)
))
expr_cvvs_loo <- expression(assign(
  out_nm,
  cv_varsel(refm,
            method = "L1",
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_loo_fw <- expression(assign(
  out_nm,
  cv_varsel(refm,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_loo_noval <- expression(assign(
  out_nm,
  cv_varsel(refm,
            method = "L1",
            validate_search = FALSE,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_loo_fw_noval <- expression(assign(
  out_nm,
  cv_varsel(refm,
            validate_search = FALSE,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_kfold <- expression(assign(
  out_nm,
  cv_varsel(refm,
            method = "L1",
            cv_method = "kfold",
            K = 2,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_kfold_fw <- expression(assign(
  out_nm,
  cv_varsel(refm,
            cv_method = "kfold",
            K = 2,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
# TODO: K-fold CV with `validate_search = FALSE`.

## Augmented-data projection ----------------------------------------------

if (pkg_fit == "rstanarm") {
  refm_aug <- get_refmodel(fit,
                           augdat_y_unqs = c("0", "1"),
                           augdat_link = augdat_link_binom,
                           augdat_ilink = augdat_ilink_binom)
} else {
  refm_aug <- get_refmodel(fit,
                           brms_seed = seed_brms,
                           augdat_y_unqs = c("0", "1"),
                           augdat_link = augdat_link_binom,
                           augdat_ilink = augdat_ilink_binom)
}
expr_prj_aug <- expression(assign(
  out_nm,
  project(refm_aug,
          predictor_terms = nms_soltrms,
          ndraws = nprjdraws,
          seed = seed_prj,
          regul = 0)
))
expr_vs_aug <- expression(assign(
  out_nm,
  varsel(refm_aug,
         nclusters = nclusters_vsel,
         nclusters_pred = nclusters_pred_vsel,
         nterms_max = nterms_max_vsel,
         seed = seed_vsel,
         regul = 0,
         verbose = FALSE)
))
expr_cvvs_aug_loo <- expression(assign(
  out_nm,
  cv_varsel(refm_aug,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_aug_loo_noval <- expression(assign(
  out_nm,
  cv_varsel(refm_aug,
            validate_search = FALSE,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
expr_cvvs_aug_kfold <- expression(assign(
  out_nm,
  cv_varsel(refm_aug,
            cv_method = "kfold",
            K = 2,
            nclusters = nclusters_vsel,
            nclusters_pred = nclusters_pred_vsel,
            nterms_max = nterms_max_vsel,
            seed = seed_vsel,
            regul = 0,
            verbose = FALSE)
))
# TODO: K-fold CV with `validate_search = FALSE`.

# Benchmark ---------------------------------------------------------------

library(microbenchmark)

## project() --------------------------------------------------------------

neval <- 10

nprjdraws <- 20
out_nm <- "prj20"
microbenchmark(eval(expr_prj), times = neval)
## --> Gives on my machine:
# Unit: milliseconds
#           expr      min       lq     mean   median      uq      max neval
# eval(expr_prj) 33.96374 36.15072 38.02773 37.77602 39.2816 42.47076    10
##
out_nm <- "prj20_aug"
microbenchmark(eval(expr_prj_aug), times = neval)
## --> Gives on my machine:
# Unit: milliseconds
#               expr      min       lq     mean   median       uq      max neval
# eval(expr_prj_aug) 42.15957 43.41935 46.80305 47.79589 49.57314 52.31946    10
##

nprjdraws <- nrow(as.matrix(fit)) # 4000 here
out_nm <- "prj"
microbenchmark(eval(expr_prj), times = neval)
## --> Gives on my machine:
# Unit: seconds
#           expr     min       lq     mean   median       uq      max neval
# eval(expr_prj) 9.04169 9.218647 9.456081 9.321489 9.647979 10.38877    10
##
out_nm <- "prj_aug"
microbenchmark(eval(expr_prj_aug), times = neval)
## --> Gives on my machine:
# Unit: seconds
#               expr     min       lq     mean  median       uq      max neval
# eval(expr_prj_aug) 6.84817 6.918832 7.201971 7.16795 7.403043 7.783208    10
##

## varsel() ---------------------------------------------------------------

neval <- 10

### Defaults (preferable for investigating possible speed improvements):
nclusters_vsel <- 20
nclusters_pred_vsel <- 400
nterms_max_vsel <- NULL
###
### For reduced runtime:
# nclusters_vsel <- 3
# nclusters_pred_vsel <- 5
# nterms_max_vsel <- 4
###

out_nm <- "vs"
microbenchmark(eval(expr_vs), times = neval)
## --> Gives on my machine:
# Unit: seconds
#          expr      min       lq     mean  median      uq     max neval
# eval(expr_vs) 11.10757 11.35867 11.78405 11.8121 12.2827 12.4041    10
###
out_nm <- "vs_fw"
microbenchmark(eval(expr_vs_fw), times = neval)
## --> Gives on my machine:
# Unit: seconds
#             expr      min       lq     mean   median       uq      max neval
# eval(expr_vs_fw) 13.90288 14.66552 14.99932 14.92758 15.37122 15.88203    10
###
out_nm <- "vs_aug"
microbenchmark(eval(expr_vs_aug), times = neval)
## --> Gives on my machine:
# Unit: seconds
#              expr      min       lq    mean   median       uq      max neval
# eval(expr_vs_aug) 16.45619 16.59917 17.7791 17.24364 19.03249 20.87832    10
###

## cv_varsel() ------------------------------------------------------------

neval <- 3

### Defaults (preferable for investigating possible speed improvements):
# nclusters_vsel <- 20
# nclusters_pred_vsel <- 400
# nterms_max_vsel <- NULL
###
### For reduced runtime:
nclusters_vsel <- 3
nclusters_pred_vsel <- 5
nterms_max_vsel <- 4
###

### TODO:
out_nm <- "cvvs_loo"
microbenchmark(eval(expr_cvvs_loo), times = neval)

out_nm <- "cvvs_loo_fw"
microbenchmark(eval(expr_cvvs_loo_fw), times = neval)

out_nm <- "cvvs_loo_noval"
microbenchmark(eval(expr_cvvs_loo_noval), times = neval)

out_nm <- "cvvs_loo_fw_noval"
microbenchmark(eval(expr_cvvs_loo_fw_noval), times = neval)

out_nm <- "cvvs_kfold"
microbenchmark(eval(expr_cvvs_kfold), times = neval)

out_nm <- "cvvs_kfold_fw"
microbenchmark(eval(expr_cvvs_kfold_fw), times = neval)

out_nm <- "cvvs_aug_loo"
microbenchmark(eval(expr_cvvs_aug_loo), times = neval)

out_nm <- "cvvs_aug_loo_noval"
microbenchmark(eval(expr_cvvs_aug_loo_noval), times = neval)

out_nm <- "cvvs_aug_kfold"
microbenchmark(eval(expr_cvvs_aug_kfold), times = neval)

###

# Comparison of results ---------------------------------------------------

## project() --------------------------------------------------------------

prjmat <- as.matrix(prj)
prjmat_aug <- as.matrix(prj_aug)
quantile(abs(prjmat - prjmat_aug))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 0.000000e+00 8.326673e-17 1.665335e-16 3.330669e-16 2.886580e-15
##

## varsel() ---------------------------------------------------------------

smmry <- summary(vs)
smmry_fw <- summary(vs_fw)
smmry_aug <- summary(vs_aug)

stopifnot(isTRUE(
  all.equal(smmry_fw[setdiff(names(smmry_fw), c("method", "nclusters",
                                                "suggested_size", "perf_sub"))],
            smmry[setdiff(names(smmry), c("method", "nclusters",
                                          "suggested_size", "perf_sub"))])
))
## --> Errors, so the results from L1 search and forward search differ
## remarkably. Thus, focus only on forward search (because the augmented-data
## projection currently requires forward search).
stopifnot(isTRUE(
  all.equal(smmry_fw[setdiff(names(smmry_fw), c("family", "perf_sub"))],
            smmry_aug[setdiff(names(smmry_aug), c("family", "perf_sub"))])
))
stopifnot(identical(
  smmry_fw$perf_sub[, c("size", "ranking_fulldata")],
  smmry_aug$perf_sub[, c("size", "ranking_fulldata")]
))
## --> We may focus on element `perf_sub` and there on the columns related to
## the performance measures.

quantile(abs(smmry_fw$perf_sub$elpd - smmry_aug$perf_sub$elpd))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 2.433609e-12 4.381562e-11 1.738840e-10 3.116476e-10 3.740084e-10
##
stopifnot(isTRUE(
  all.equal(smmry_fw$perf_sub, smmry_aug$perf_sub, tolerance = 1e-9)
))

## cv_varsel() ------------------------------------------------------------

### TODO:
smmry_cv_loo_fw <- summary(cvvs_loo_fw)
smmry_cv_loo_fw_noval <- summary(cvvs_loo_fw_noval)
smmry_cv_kfold_fw <- summary(cvvs_kfold_fw)
smmry_cv_aug_loo <- summary(cvvs_aug_loo)
smmry_cv_aug_loo_noval <- summary(cvvs_aug_loo_noval)
smmry_cv_aug_kfold <- summary(cvvs_aug_kfold)

quantile(abs(
  smmry_cv_loo_fw$perf_sub$elpd - smmry_cv_aug_loo$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_loo_fw$perf_sub, smmry_cv_aug_loo$perf_sub,
            tolerance = 1e-9)
))

quantile(abs(
  smmry_cv_loo_fw_noval$perf_sub$elpd - smmry_cv_aug_loo_noval$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_loo_fw_noval$perf_sub, smmry_cv_aug_loo_noval$perf_sub,
            tolerance = 1e-9)
))

quantile(abs(
  smmry_cv_kfold_fw$perf_sub$elpd - smmry_cv_aug_kfold$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_kfold_fw$perf_sub, smmry_cv_aug_kfold$perf_sub,
            tolerance = 1e-9)
))
###

# With glm() --------------------------------------------------------------

## Setup ------------------------------------------------------------------

glmfittr_orig <- options(projpred.glm_fitter = "fit_glm_callback")

## Benchmark --------------------------------------------------------------

### project() -------------------------------------------------------------

neval <- 10

nprjdraws <- 20
out_nm <- "prj20_glm"
microbenchmark(eval(expr_prj), times = neval)
## --> Gives on my machine:
# Unit: milliseconds
#           expr      min       lq     mean   median       uq      max neval
# eval(expr_prj) 39.80089 40.90956 43.20427 43.13515 45.15778 47.05077    10
##
out_nm <- "prj20_aug_glm"
microbenchmark(eval(expr_prj_aug), times = neval)
## --> Gives on my machine:
# Unit: milliseconds
#               expr      min       lq     mean   median      uq      max neval
# eval(expr_prj_aug) 44.32463 49.16584 55.47541 51.40357 63.5681 70.51769    10
##

nprjdraws <- nrow(as.matrix(fit)) # 4000 here
out_nm <- "prj_glm"
microbenchmark(eval(expr_prj), times = neval)
## --> Gives on my machine:
# Unit: seconds
#           expr     min      lq     mean median      uq     max neval
# eval(expr_prj) 12.1421 12.5953 13.03945 12.992 13.2824 14.5758    10
##
out_nm <- "prj_aug_glm"
microbenchmark(eval(expr_prj_aug), times = neval)
## --> Gives on my machine:
# Unit: seconds
#               expr      min       lq     mean   median       uq      max neval
# eval(expr_prj_aug) 8.835068 9.573083 9.845413 9.921406 10.12194 10.62507    10
##

### varsel() --------------------------------------------------------------

neval <- 10

### Defaults (preferable for investigating possible speed improvements):
nclusters_vsel <- 20
nclusters_pred_vsel <- 400
nterms_max_vsel <- NULL
###
### For reduced runtime:
# nclusters_vsel <- 3
# nclusters_pred_vsel <- 5
# nterms_max_vsel <- 4
###

out_nm <- "vs_glm"
microbenchmark(eval(expr_vs), times = neval)
## --> Gives on my machine:
# Unit: seconds
#          expr      min       lq     mean   median       uq      max neval
# eval(expr_vs) 13.39199 14.23155 15.22399 15.17233 16.10023 17.66308    10
##
out_nm <- "vs_fw_glm"
microbenchmark(eval(expr_vs_fw), times = neval)
## --> Gives on my machine:
# Unit: seconds
#             expr      min      lq     mean   median       uq      max neval
# eval(expr_vs_fw) 17.48052 18.2164 19.26419 19.26699 20.35536 21.34945    10
##
out_nm <- "vs_aug_glm"
microbenchmark(eval(expr_vs_aug), times = neval)
## --> Gives on my machine:
# Unit: seconds
#              expr      min       lq     mean   median       uq      max neval
# eval(expr_vs_aug) 20.22887 20.50453 22.42577 22.60337 23.72735 24.78615    10
##

### cv_varsel() -----------------------------------------------------------

neval <- 3

### Defaults (preferable for investigating possible speed improvements):
# nclusters_vsel <- 20
# nclusters_pred_vsel <- 400
# nterms_max_vsel <- NULL
###
### For reduced runtime:
nclusters_vsel <- 3
nclusters_pred_vsel <- 5
nterms_max_vsel <- 4
###

### TODO:
out_nm <- "cvvs_loo_glm"
microbenchmark(eval(expr_cvvs_loo), times = neval)

out_nm <- "cvvs_loo_fw_glm"
microbenchmark(eval(expr_cvvs_loo_fw), times = neval)

out_nm <- "cvvs_loo_noval_glm"
microbenchmark(eval(expr_cvvs_loo_noval), times = neval)

out_nm <- "cvvs_loo_fw_noval_glm"
microbenchmark(eval(expr_cvvs_loo_fw_noval), times = neval)

out_nm <- "cvvs_kfold_glm"
microbenchmark(eval(expr_cvvs_kfold), times = neval)

out_nm <- "cvvs_kfold_fw_glm"
microbenchmark(eval(expr_cvvs_kfold_fw), times = neval)

out_nm <- "cvvs_aug_loo_glm"
microbenchmark(eval(expr_cvvs_aug_loo), times = neval)

out_nm <- "cvvs_aug_loo_noval_glm"
microbenchmark(eval(expr_cvvs_aug_loo_noval), times = neval)

out_nm <- "cvvs_aug_kfold_glm"
microbenchmark(eval(expr_cvvs_aug_kfold), times = neval)

###

## Comparison of results --------------------------------------------------

### project() -------------------------------------------------------------

prjmat_glm <- as.matrix(prj_glm)
prjmat_aug_glm <- as.matrix(prj_aug_glm)

# Augmented-data vs. traditional projection, both with glm():
quantile(abs(prjmat_glm - prjmat_aug_glm))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 0.000000e+00 6.241854e-12 2.183249e-10 1.663843e-09 4.305462e-08
##

# Traditional projection, glm() vs. glm_ridge():
quantile(abs(prjmat_glm - prjmat))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 2.195109e-12 7.731982e-07 6.717184e-06 4.405347e-05 8.684574e-04
##
stopifnot(isTRUE(
  all.equal(prjmat, prjmat_glm, tolerance = 1e-3)
))

# Augmented-data projection, glm() vs. glm_ridge():
quantile(abs(prjmat_aug_glm - prjmat_aug))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 1.052269e-10 7.702390e-07 6.714387e-06 4.405347e-05 8.684591e-04
##
stopifnot(isTRUE(
  all.equal(prjmat_aug, prjmat_aug_glm, tolerance = 1e-3)
))

### varsel() --------------------------------------------------------------

smmry_fw_glm <- summary(vs_fw_glm)
smmry_aug_glm <- summary(vs_aug_glm)

# Augmented-data vs. traditional projection, both with glm():
quantile(abs(smmry_fw_glm$perf_sub$elpd - smmry_aug_glm$perf_sub$elpd))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 5.871925e-11 7.762768e-10 2.252385e-09 4.440825e-09 5.158540e-09
##

# Traditional projection, glm() vs. glm_ridge():
quantile(abs(smmry_fw_glm$perf_sub$elpd - smmry_fw$perf_sub$elpd))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 5.870929e-06 4.583964e-05 9.580811e-05 1.131271e-04 1.191820e-04
##

# Augmented-data projection, glm() vs. glm_ridge():
quantile(abs(smmry_aug_glm$perf_sub$elpd - smmry_aug$perf_sub$elpd))
## --> Gives on my machine:
#           0%          25%          50%          75%         100%
# 5.870896e-06 4.583905e-05 9.580568e-05 1.131220e-04 1.191775e-04
##

### cv_varsel() -----------------------------------------------------------

### TODO:
smmry_cv_loo_fw_glm <- summary(cvvs_loo_fw_glm)
smmry_cv_loo_fw_noval_glm <- summary(cvvs_loo_fw_noval_glm)
smmry_cv_kfold_fw_glm <- summary(cvvs_kfold_fw_glm)
smmry_cv_aug_loo_glm <- summary(cvvs_aug_loo_glm)
smmry_cv_aug_loo_noval_glm <- summary(cvvs_aug_loo_noval_glm)
smmry_cv_aug_kfold_glm <- summary(cvvs_aug_kfold_glm)

# Augmented-data vs. traditional projection, both with glm():
# LOO:
quantile(abs(
  smmry_cv_loo_fw_glm$perf_sub$elpd - smmry_cv_aug_loo_glm$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_loo_fw_glm$perf_sub, smmry_cv_aug_loo_glm$perf_sub,
            tolerance = 1e-8)
))
# LOO with `validate_search = FALSE`:
quantile(abs(
  smmry_cv_loo_fw_noval_glm$perf_sub$elpd -
    smmry_cv_aug_loo_noval_glm$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_loo_fw_noval_glm$perf_sub,
            smmry_cv_aug_loo_noval_glm$perf_sub,
            tolerance = 1e-8)
))
# K-fold:
quantile(abs(
  smmry_cv_kfold_fw_glm$perf_sub$elpd - smmry_cv_aug_kfold_glm$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_kfold_fw_glm$perf_sub, smmry_cv_aug_kfold_glm$perf_sub,
            tolerance = 1e-8)
))

# Traditional projection, glm() vs. glm_ridge():
# LOO:
quantile(abs(
  smmry_cv_loo_fw_glm$perf_sub$elpd - smmry_cv_loo_fw$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_loo_fw$perf_sub, smmry_cv_loo_fw_glm$perf_sub,
            tolerance = 1e-3)
))
# LOO with `validate_search = FALSE`:
quantile(abs(
  smmry_cv_loo_fw_noval_glm$perf_sub$elpd -
    smmry_cv_loo_fw_noval$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_loo_fw_noval$perf_sub,
            smmry_cv_loo_fw_noval_glm$perf_sub,
            tolerance = 1e-3)
))
# K-fold:
quantile(abs(
  smmry_cv_kfold_fw_glm$perf_sub$elpd - smmry_cv_kfold_fw$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_kfold_fw$perf_sub, smmry_cv_kfold_fw_glm$perf_sub,
            tolerance = 1e-3)
))

# Augmented-data projection, glm() vs. glm_ridge():
# LOO:
quantile(abs(
  smmry_cv_aug_loo_glm$perf_sub$elpd - smmry_cv_aug_loo$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_aug_loo_glm$perf_sub, smmry_cv_aug_loo$perf_sub,
            tolerance = 1e-3)
))
# LOO with `validate_search = FALSE`:
quantile(abs(
  smmry_cv_aug_loo_noval_glm$perf_sub$elpd -
    smmry_cv_aug_loo_noval$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_aug_loo_noval_glm$perf_sub,
            smmry_cv_aug_loo_noval$perf_sub,
            tolerance = 1e-3)
))
# K-fold:
quantile(abs(
  smmry_cv_aug_kfold_glm$perf_sub$elpd - smmry_cv_aug_kfold$perf_sub$elpd
))

stopifnot(isTRUE(
  all.equal(smmry_cv_aug_kfold$perf_sub, smmry_cv_aug_kfold_glm$perf_sub,
            tolerance = 1e-3)
))
###

## Teardown ---------------------------------------------------------------

options(glmfittr_orig)
