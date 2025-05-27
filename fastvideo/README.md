## More Discussion on FLUX
1. We set the inference batch size to 1 because we observed differences in probability outputs when it exceeds the training batch size. This inconsistency may stem from PyTorch or other underlying mechanisms, though the exact cause remains unclear.
2. A stronger supervised fine-tuning (SFT) stage can suppress exploration during the GRPO phase. For those developing custom models, we recommend using fewer SFT iterations to maintain exploratory diversity.
3. The provided reward curves show FLUX's performance with extended training. However, prolonged training does not improve visualization quality, likely due to limitations in the reward model design. Since HPS-v2.1 was not optimized for FLUX, its effectiveness is constrained. Enabling EMA (Exponential Moving Average) may enhance visualization—a feature we plan to support in future updates.

<div align="center">
<img src=../assets/rewards/opensource_flux_more_steps.png width="49%">
<div>

