# SING Learned Structure Attention

TODO попробовать сделать нормировку в лоссе (потому что ssm_error обычно сильно больше)

```
alpha = 0.20859
total_loss = bce_loss + alpha * ssm_error
```

в lca alpha = 0.43632

n = 20
=== AVERAGED METRICS ===
diversity: ORIGINAL=0.4493, LSA=0.7051
unique_steps: ORIGINAL=28.0500, LSA=84.0500
mean_ssm: ORIGINAL=0.5507, LSA=0.2949
std_ssm: ORIGINAL=0.1804, LSA=0.1186
diag_drop: ORIGINAL=0.1402, LSA=0.0490
iou: ORIGINAL=0.1518, LSA=0.1524
mse: ORIGINAL=0.1468, LSA=0.0382