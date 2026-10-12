# Released results

These machine-readable tables mirror the aggregate values shown in the project page and manuscript working copy. Success rates use the shared 1,800-episode dynamic population unless a table states otherwise.

- `cross_model_async.csv`: six pretrained policies at each model's best measured asynchronous horizon.
- `execution_strategy_best_points.csv`: selected best measured points for the execution-strategy comparison. VLASH uses a separately trained delay-aware checkpoint and is a system reference, not a same-checkpoint controller ablation.
- `dynamic_finetuning_ksweep.csv`: measured asynchronous points for static, dynamic-only, and mixed training recipes.
- `real_robot_results_by_k.csv`: physical experiments at K=4 and K=8. The first three tasks report successes out of 50; whack-a-mole reports total successful presses over 50 sessions of 30 seconds.

Curves connect measured points only as a visual guide. No interpolated point is included in these CSV files.
