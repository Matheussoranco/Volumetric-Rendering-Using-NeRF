# Volumetric-Rendering-Using-NeRF
This is  minimal implementation of the research paper NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis by Ben Mildenhall et. al. The authors have proposed an ingenious way to synthesize novel views of a scene by modelling the volumetric scene function through a neural network.
The quetion is: would it be possible to give to a neural network the position of a pixel in an image, and ask the network to predict the color at that position?

The neural network would hypothetically memorize (overfit on) the image. This means that our neural network would have encoded the entire image in its weights. We could query the neural network with each position, and it would eventually reconstruct the entire image.
The authors of the paper propose a minimal and elegant way to learn a 3D scene using a few images of the scene. They discard the use of voxels for training. The network learns to model the volumetric scene, thus generating novel views (images) of the 3D scene that the model was not shown at training time.

## Sources and References
- [Keras: NeRF Example](https://keras.io/examples/vision/nerf/)  
- [Official NeRF GitHub Repository](https://github.com/bmild/nerf)  
- [MathWorks: Camera Calibration](https://www.mathworks.com/help/vision/ug/camera-calibration.html)  
- [Original Paper on ArXiv](https://arxiv.org/abs/2003.08934)  

