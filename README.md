## Experimental U-Net for RGB hologram reconstruction
<p align="center"><img src="saved_images/Unet.png" align="center" width="400">

Takes in 4-channel RGBD image and output 3-channel RGB reconstructed image\
Outputs under `/saved_images`\
Inputs folders are specified under `/data`\
Run train.py to reconstruct input holograms\
Adjust hyperparameters in train.py
#### Sample Hologram
<p align="center"><img src="saved_images/random1.png" align="center" width="400">

- `/train_depth` training depth images in grey scale 
- `/train_images` RGB input images, composited with depth as RGBD 
- `/train_masks` training groundtruth holograms
- `/val_depth` validation depth images
- `/val_images` validation RGB images
- `/val_masks` validation groundtruth

 