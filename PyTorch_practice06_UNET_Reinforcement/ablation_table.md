# This is an *ablation table* for the model using HPC training

- The goal of this table is to evaluate changes in average train loss, average testing loss, and average PSNR across several different numbers of epochs.
- This is the non-official version, intended to record notes, training runs, and other metrics needed for future reports.

## Things to know

- The information recorded is taken after the LAST epoch and represents the average for that category over the entire cycle of epochs.
- I noticed signs of overfitting past epoch 10: the training loss would improve, but the testing loss stayed the same. Epoch 100 exemplifies this overfitting.
- PSNR is in an acceptable range; however, there is room for improvement. AdamW, different MSE formulations, and improved decoding could provide better scores.

April 24, 2026
| Epoch #  | Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| 2        | 0.0329   | 0.0326   | 26.1427  |
| 10       | 0.0303   | 0.0318   | 26.3487  |
| 100      | 0.0115   | 0.0319   | 27.4832  |

---

- The goal of this table is to determine which hyperparameters should be changed to achieve the best possible scores.
- Changes made: Added a cosine annealing LR scheduler.
- After continual testing, an epoch count of 6 was chosen, as it was the last point in previous training runs where overfitting did not occur.

- Base model: LR = 1e-4, batch size = 256, seed = 42, epoch = 6

April 27, 2026
| Hyperparam| Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| Base     | 0.0312   | 0.0315   | 26.4920  |
| LR = 1e-5| 0.0330   | 0.0332   | 26.0301  |
| **LR = 1e-3**| **0.0313** | **0.0314** | **26.5537** |
| LR = 1e-2| 0.0320   | 0.0325   | 26.2481  |

Testing base model with LR of 1e-3 and adjusting batch size.
| Hyperparam| Training | Testing  | PSNR     |
| -------- | -------- | -------- | -------- |
| **Batch = 512**| **0.0316** | **0.0316** | **26.4718** |
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
| **11** | **0.0307** | **0.0310** | **26.6076** |
| 15       | 0.0310   | 0.0314   | 26.4585  |
| 17       | 0.0310   | 0.0314   | 26.5424  |
| 100      | 0.0174   | 0.0324   | 26.3727  |

After running for 100 epochs, signs of overfitting began at epoch 20. By epoch 22, the training loss had dropped by 0.10, making the loss 0.0302, while the testing loss grew to 0.0320. The highest PSNR value was 26.7136, recorded during epoch 29. PSNR depreciated and plateaued after epoch 29.

- New changes made: Added SSIM and LPIPS metrics to the model, and changed PSNR (along with the other metrics) to use torchmetrics.image instead of calculating them manually.
- **NOTABLE CHANGE**: *The PSNR value was using inflated metrics during testing*. Now, PSNR and the other metrics are **updated after every evaluation**, making the scores more accurate (albeit at the cost of yielding a lower score).

*Note: SSIM and LPIPS are on a scale of 0 to 1.*
| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0333   | 0.0342   | 22.4915  | 0.9336   | 0.2186   |
| 5        | 0.0319   | 0.0318   | 22.7617  | 0.9352   | 0.2054   |
| 10       | 0.0308   | 0.0311   | 22.9189  | 0.9380   | 0.1987   |
| 11       | 0.0306   | 0.0309   | 22.9538  | 0.9382   | 0.1977   |
| 24       | 0.0271   | **0.0307** | **23.0250** | 0.9399   | 0.1926   |
| 31       | 0.0236   | 0.0312   | 22.9468  | 0.9398   | **0.1914** |
| 50       | **0.0198** | 0.0321   | 22.7741  | 0.9388   | 0.1940   |

After training for 50 epochs, the results remain fairly similar even after the changes. Epoch 11 remains the best overall result that does not show overfitting. After epoch 11, signs of overfitting remain consistent throughout the rest of the epochs. Some observed changes: Epoch 24 shows signs of potentially being the best option; however, the model is still overfitting. The best LPIPS score occurred during epoch 31, and, as expected, the best training value was recorded during the final epoch.

---

April 28, 2026

Trying a different learning rate with a smaller batch size: LR = 2e-4, Batch size = 128
| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0324   | 0.0324   | 22.5763  | 0.9344   | 0.2126   |
| 4        | 0.0317   | 0.0317   | 22.7054  | 0.9362   | 0.2036   |
| 6        | 0.0311   | 0.0313   | 22.8783  | 0.9371   | 0.1979   |
| **9** | **0.0301** | **0.0308** | **23.0207** | **0.9392** | **0.1940** |
| 11       | 0.0292   | 0.0307   | 23.0571  | 0.9392   | 0.1908   |

The new hyperparameters performed better, and did so earlier, than the previous parameters, showing greater improvements. Epochs 10 and 11 were the first to show signs of overfitting.

New table comparing parameters
| Hyperparameters | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| LR = 2e-4, BS = 128, seed = 42| 0.0311   | 0.0313   | 22.8783  | 0.9371   | 0.1979   |
| LR = 1e-3, BS = 512, seed = 42| 0.0317   | 0.0322   | 22.6270  | 0.9358   | 0.2085   |
| LR = 1e-3, BS = 256, seed = 42| 0.0314   | 0.0314   | 22.8367  | 0.9374   | 0.2002   |
| LR = 2e-4, BS = 128, seed = 60| 0.0308   | 0.0310   | 22.9205  | 0.9386   | 0.1958   |
| LR = 1e-3, BS = 512, seed = 60| 0.0317   | 0.0321   | 22.6479  | 0.9372   | 0.2063   |
| LR = 1e-3, BS = 256, seed = 60| 0.0314   | 0.0315   | 22.7782  | 0.9379   | 0.2029   |

Ironically, the previous test with the increased batch size underperformed compared to the lower batch size. Fortunately, the new hyperparameters were still better than both previous tests across multiple seeds. This is an overall positive outcome for the testing phase.

Things I potentially want to add in the future to improve the score: a perceptual loss term, switching BatchNorm to GroupNorm (or InstanceNorm), and adding data augmentation in ColorizationDataset.

---

April 30, 2026

Testing U-Net using GroupNorm in place of BatchNorm
- This test uses LR = 2e-4, BS = 128, seed = 42, epoch = 11
  
| Num of groups | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| BatchNorm| 0.0292   | 0.0307   | 23.0571  | 0.9392   | 0.1908   |
| 8        | 0.0309   | 0.0311   | 22.9018  | 0.9381   | 0.1994   |
| 16       | 0.0308   | 0.0310   | 22.9320  | 0.9383   | 0.1982   |
| 32       | 0.0306   | 0.0310   | 22.9528  | 0.9382   | 0.1982   |

Testing GroupNorm 8 and 16 for longer epochs
- This test uses LR = 2e-4, BS = 128, seed = 42, epoch = 24
  
| Num of groups | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 16 @ 11       | 0.0308   | 0.0310   | 22.9320  | 0.9383   | 0.1982   |
| 16 @ 15       | 0.0304   | 0.0310   | 22.9367  | 0.9388   | 0.1959   |
| **16 @ 17** | **0.0300** | **0.0308** | **22.9822** | **0.9388** | **0.1937** |
| 16 @ 20       | 0.0293   | 0.0308   | 22.9445  | 0.9393   | 0.1981   |
| 16 @ 24       | 0.0288   | 0.0309   | 22.9807  | 0.9387   | 0.1938   |
| 32 @ 11       | 0.0306   | 0.0310   | 22.9528  | 0.9382   | 0.1982   |
| 32 @ 15       | 0.0301   | 0.0310   | 22.9653  | 0.9388   | 0.1968   |
| 32 @ 17       | 0.0296   | 0.0309   | 22.9853  | 0.9384   | 0.1939   |
| 32 @ 20       | 0.0288   | 0.0309   | 22.9793  | 0.9388   | 0.1963   |
| 32 @ 24       | 0.0282   | 0.0310   | 22.9815  | 0.9398   | 0.1946   |

This testing has shown that the switch to GroupNorm with 16 groups performed almost identically to BatchNorm after further training. This presents a two-sided situation. On one hand, the longer training suggests the model is learning more, meaning GroupNorm learns deeper representations than BatchNorm. On the other hand, BatchNorm achieved similar results much more quickly (eight epochs faster), meaning the model uses less time and computational energy to produce similar outcomes.

Based on these findings, I will initially move forward with BatchNorm for testing and adapt GroupNorm after the finalized BatchNorm model is complete.

Adding perceptual loss
Changes: Changed the sigmoid function in U-Net to Tanh for better output matching for LPIPS, added perceptual loss to the training, and adapted the ab values to use Tanh instead of sigmoid where needed.

| Epoch #  | &darr; Training |  &darr; Testing  | &uarr; PSNR     | &uarr; SSIM     | &darr; LPIPS    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 2        | 0.0858   | 0.0653   | 22.5883  | 0.9346   | 0.1945   |
| 10       | 0.0782   | 0.0628   | 22.9586  | 0.9377   | 0.1821   |
| 20       | 0.0662   | 0.0621   | 22.9943  | 0.9372   | 0.1793   |
| 80       | 0.0301   | 0.0653   | 22.7600  | 0.9352   | 0.1893   |

Adding perceptual loss did not improve the model's performance. The pixel-level metrics did not improve much relatively, not to mention the increased loss in both training and testing. Overall, the model performed better before introducing perceptual loss.

**After all testing, the best hyperparameters for the model were LR = 2e-4, BS = 128, and GroupNormalization using 8 groups, with the model performing best at epochs 16-18.**
