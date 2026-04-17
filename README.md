# PyTorch_Practice
- This is a comprehensive repository of projects I created during the process of learning PyTorch and understanding neural networks.
- This was a self-taught endeavor supported by [this YouTube video](https://www.youtube.com/watch?v=Z_ikDlimN6A&themeRefresh=1).
- Sources are included in the code for items used outside the scope of the video.
  
## PyTorch_practice01
- This was my introduction, learning what each section of code does.
- Covered the basics of tensors, scalars, devices, etc.
- This code is not designed to be run; rather, it is designed to show that I spent the time to understand the PyTorch framework.
## PyTorch_practice02
- This is the first project where I turned learning into something tangible.
- This is a linear regression NN testing the code based on custom-generated data.
- A random seed of 42 is applied to the code.
- The main goal of this project was to build an understanding of how a model is constructed.
## PyTorch_practice03
- This project was designed to build upon the linear model.
- Using the "circles" dataset from the scikit-learn library, I created data that visualizes two circles in a scatter plot.
- Used a multi-layer, non-linear NN to separate the data to find how accurately the model determines if data points should be "blue" or "red".
- The main goal of this project was to understand nn.ReLU and implement multiple layers into a model.
## PyTorch_practice04
- This project is designed to understand and building a CNN model.
- Using torchvision datasets, the code uses and downloads FashionMNIST.
- The project also encompasses a linear model to compare with the CNN for prefromance.
- The main goal of this project was to understand CNN and image-to-tensor processing.
## PyTorch_practice05
- This project is designed to be a larger model to understand diffusion and U-Nets.
- This project is very large and contains sevearl different models using custom and commonly used modules.
- The project is meant to generate images from text prompts (using CLIP) and/or images to create.
- This project needs the files from [this Hugging Face library](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer) to run the code, assign it to a "data" folder.
- This project is my own code following the guide from [this YouTube video](https://www.youtube.com/watch?v=ZBKpAp_6TGI). The creator also used other repositiories for their code, see [here](https://github.com/hkproj/pytorch-stable-diffusion) for more.
- The main goal of this project was to test my limits and create something beyond the scope of my knowledge as well as take the time to learn about diffusion and U-Nets.
- Personal video presentation on delving into some aspects present in the model (more of a general overview), looking at the overall process of the model, and overviewing some of the code as well as comparing its outputs can be found [here.](https://youtu.be/hDsl1UvYNRU)
- *Note: The video is likely to be replaced in the future with one containg a better overall presentation. When replaced this text will disappear as well* 
## PyTorch_practice06
- This project is designed to be a reinforcement of knowledge and implementation for U-Net models.
- The project is in two parts: The U-Net and the training.
- The results from training the model are logged in the "color_unet_train.log"
- The goal of this poject was to ultimately understand U-Nets further and train a model on a HPC.
