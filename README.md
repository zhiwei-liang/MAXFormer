# MAXFormer

The codes for the work "MAXFormer: Enhanced Transformer for Medical Image Segmentation with Multi-Attention and Multi-Scale Features Fusion". A U-shaped hierarchical Transformer. Our paper has been accepted by Knowledge-Based Systems. We updated the Reproducibility. I hope this will help you to reproduce the results. We have provided the source code as well as our model weights file, which we hope will help you to replicate and improve.

### Download the dataset and model weights

| Task                     | Dataset                                                      | Model Weights                                                | Prediction result file                                       |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Multi-organ segmentation | [Synapse](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi) | [MAXFormer](https://drive.google.com/file/d/1u8yYT4VLzmOsJ0VfJo5ZwnOpVtWaJFca/view?usp=sharing) | [Synapse test dataset prediction(ours)](https://drive.google.com/file/d/1JecWCd2HeqhmaVPJf7GZx9TOWtOkCpo9/view?usp=share_link) |
| Cardiac segmentation     | [ACDC](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) | [MAXFormer](https://drive.google.com/file/d/1kQZKuDY8axM_QLbIYBUenTzPD7DvVLc1/view?usp=sharing) | [ACDC test dataset prediction(ours)](https://drive.google.com/file/d/10HBCZ6uQ7mv6JmajARVYrHEd_sLrRR4E/view?usp=share_link) |

### Environment

To better replicate our experiment, please prepare an environment with python=3.8, and then run the following command to install the dependencies.

```shell
pip install -r requirements.txt
```

### Train/Test

- When you want to train, you first need to fill in the train.py file with the necessary information, i.e. `root_path` and `test_path`. Other information, such as `batch_size` and `base_lr`, can be modified according to your needs. Note: If you want to use our provided model for initialization, set pre_trained params to True and place the downloaded model `synapse_8366.pth` in the `output_dir` directory.

- Train

  ```shell
  python train.py
  ```

- Test

  Similarly, you need to first set up some necessary parameters, i.e., test set path `volume_path` and output_dir path `output_dir` for the model.

  ```shell
  python test.py
  ```



### Acknowledgments

This project has benefited from the following resources, and I would like to express my gratitude:

- [TransUNet](https://github.com/Beckschen/TransUNet)

- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet.git)
- [DAE_Former](https://github.com/mindflow-institue/DAEFormer.git)



### Citation

Coming soon!