# This is an *ablation table* for the model using HPC training

- The goal of this table is to evaluate changes in average train loss, average testing loss, and average PSNR across several different numbers of epochs.

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

Testing base model with LR of 1e-3 and batch size of 512 for 15 epochs
| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0330   | 0.0329   | 26.1500  |
| 4        | 0.0321   | 0.0325   | 26.2990  |
| 6        | 0.0316   | 0.0321   | 26.3884  |
| 8        | 0.0311   | 0.0312   | 26.5786  |
| 10       | 0.0308   | 0.0310   | 26.5990  |
| 11       | **0.0307**   | **0.0310**   | **26.6076**  |
| 15       | 0.0310   | 0.0314   | 26.4585  |

Testing base model with LR of 1e-3 and batch size of 512 for 100 epochs (Noting important epochs only)
| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0330   | 0.0329   | 26.1500  |
| 4        | 0.0321   | 0.0325   | 26.2990  |
| 10       | 0.0308   | 0.0310   | 26.5990  |
| 11       | **0.0307**   | **0.0310**   | **26.6076**  |
| #        | 0.0330   | 0.0329   | 26.1500  |
