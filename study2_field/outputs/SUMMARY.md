# Study 2 results summary (auto-generated)

N = 109; >100% retention artifacts: 21; baseline ceiling: 16.

## Primary test (TOTAL strategy use) per specification:
                       spec   n pearson_r beta_strategy      p
              B_capped_full 109    -0.108               0.2638
            A_exclude_gt100  88    -0.193               0.0711
C_exclude_gt100_and_ceiling  72    -0.193               0.1035
      D_residualized_change 109                  -0.971 0.4371

## Exploratory subscales surviving BH-FDR (q<.05): 0 of 36 tests
(none)

## Predictive (5-fold CV + permutation):
      target  auc_logreg  auc_rf  best_auc  perm_null_mean  perm_null_p95  p_permutation
Retention_90       0.476   0.517     0.517           0.499          0.634          0.438
Retention_80       0.467   0.413     0.467           0.500          0.617          0.692
Retention_70       0.459   0.493     0.493           0.493          0.625          0.522

## Compensatory pattern:
predictor       outcome  pearson_r      p
    TOTAL Initial_score      0.084 0.3871
      MEM Initial_score     -0.137 0.1550
     META Initial_score      0.068 0.4802
      COG Initial_score     -0.011 0.9104

## Power: {'min_detectable_r_n109': 0.266, 'min_detectable_r_n88': 0.295, 'min_detectable_r_n72': 0.325}