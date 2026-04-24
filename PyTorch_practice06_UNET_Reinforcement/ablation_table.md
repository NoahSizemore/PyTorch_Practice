# This is an *ablation table* for the model using HPC training

- The goal of this table is to evaluate changes in average train loss, average testing loss, and average PSNR across several different numbers of epochs.

## Things to know

- The information being recorded is after the LAST epoch and is the average of that category during the entire cycle of epochs.
- I noticed signs of overfitting past epoch 10: The training loss would improve, but the testing loss stayed the same. Epoch 100 exemplifies this overfitting.
- PSNR is in an acceptable range; however, there is room for improvement. AdamW, different MSE, and improved decoding could provide better scores.

| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0329   | 0.0326   | 26.1427  |
| 10       | 0.0303   | 0.0318   | 26.3487  |
| 100      | 0.0115   | 0.0319   | 27.4832  |
