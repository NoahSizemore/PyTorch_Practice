# This is an *ablation table* for the model using HPC training

- The goal of this table is to evaluate changes in average train loss, average testing loss, and average PSNR across several different numbers of epochs.
- This is the non-offical version, with the purpose to record notes, trainings, and other meterics needed for future reports.

## Things to know

- The information being recorded is after the LAST epoch and is the average of that category during the entire cycle of epochs.
- I noticed signs of overfitting past epoch 10: The training loss would improve, but the testing loss stayed the same. Epoch 100 exemplifies this overfitting.
- PSNR is in an acceptable range; however, there is room for improvement. AdamW, different MSE, and improved decoding could provide better scores.

April 24, 2026
| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0329   | 0.0326   | 26.1427  |
| 10       | 0.0303   | 0.0318   | 26.3487  |
| 100      | 0.0115   | 0.0319   | 27.4832  |

---

- The goal for this table is to determine which hyperparameters should change to increase the scores to be the best possible
- Changes that have been made: Added LR scheduler cosine annealing.
- After continual testing, the epoch amount of "6" was chosen as it is the last spot in previous training where there wasn't overfitting.

- Base model: LR = 1e-4, batch size = 256, seed = 42, epoch = 6

April 27, 2026
| Hyperparm| Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| Base     | 0.0312   | 0.0315   | 26.4920  |
| LR = 1e-5| 0.0330   | 0.0332   | 26.0301  |
| **LR = 1e-3**| **0.0313**   | **0.0314**   | **26.5537**  |
| LR = 1e-2| 0.0320   | 0.0325   | 26.2481  |

Testing base model with LR of 1e-3 and adjusting batch size.
| Hyperparm| Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| **Batch = 512**| **0.0316**   | **0.0316**   | **26.4718**  |
| Batch = 128| 0.0313   | 0.0317   | 26.3772  |
| Batch = 1024| 0.0320   | 0.0321   | 26.3105  |

Testing base model with LR of 1e-3 and batch size of 512 for 100 epochs (Noting important epochs only)
| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0330   | 0.0329   | 26.1500  |
| 4        | 0.0321   | 0.0325   | 26.2990  |
| 6        | 0.0316   | 0.0321   | 26.3884  |
| 8        | 0.0311   | 0.0312   | 26.5786  |
| 10       | 0.0308   | 0.0310   | 26.5990  |
| **11**       | **0.0307**   | **0.0310**   | **26.6076**  |
| 15       | 0.0310   | 0.0314   | 26.4585  |
| 17       | 0.0310   | 0.0314   | 26.5424  |
| 100      | 0.0174   | 0.0324   | 26.3727  |

After running for 100 epochs, signs of overfitting began at epoch 20. By epoch 22, the training loss had dropped 0.10, making the loss 0.0302, while the testing grew to 0.0320. The highest PSNR value was 26.7136 recorded during epoch 29. PSNR decpreiated and platued after epoch 29.

- New changes made: Added SSIM and LPIPS meterics to model, changed PSNR, as well as the other meterics, to use torchmetrics.image instead of manually doing the math.
- **NOTABLE CHANGE**: *PSNR value was using inflated meterics during testing*, now PSNR is adjusted, along with the other meterics, **to be updated after every evaluation**, making the scores more accurate (at the cost of the higher score).

*Note: SSIM and LPIPS are from the scale 0 to 1.*
| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0333   | 0.0342   | 22.4915  | 0.9336   | 0.2186   |
| 5        | 0.0319   | 0.0318   | 22.7617  | 0.9352   | 0.2054   |
| 10       | 0.0308   | 0.0311   | 22.9189  | 0.9380   | 0.1987   |
| 11       | 0.0306   | 0.0309   | 22.9538  | 0.9382   | 0.1977   |
| 24       | 0.0271   | **0.0307**   | **23.0250**  | 0.9399   | 0.1926   |
| 31       | 0.0236   | 0.0312   | 22.9468  | 0.9398   | **0.1914**   |
| 50       | **0.0198**   | 0.0321   | 22.7741  | 0.9388   | 0.1940   |

After training for 50 epochs, results are fairly similar even after the changes. Epoch 11 remains to be the best overall result that does not show overfitting. After epoch 11, signs of overfitting remain the same throughout the rest of the epochs. Some changes that were observed: Epoch 24 is showing signs of being potentially the best option, however the model is still overfitting, the best LPIPS score was during epoch 31, and, of course, the best training value was recorded during the last epoch.

---

