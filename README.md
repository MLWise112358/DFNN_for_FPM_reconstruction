# DFNN_for_FPM_reconstruction
The trained networks for open-sourced USAF and biological samples and the test file
## The dataset
The datasets we use to test the network are open-sourced by other papers. 

The USAF dataset we use is open-sourced by Cao Zuo *et al* [1]. \
"https://scilaboratory.com/code.html". \
We take every other one of the 441(21x21) images, and use a total of 121(11x11) images to test the resolution enhancement ability of our network which are also provided in the *data* folder. \
The U2OS and Hela dataset are open-sourced by Lei Tian *et al* [2, 3].\
"http://www.laurawaller.com/opensource/". \
We use 49（7x7）and 121 (11x11) images out of the 293 images in the dataset to test the performance of our network on experimental samples respectively. The images we use are provided in the *data* folder.\


We are grateful to Zuo *et al* and Tian *et al* for providing the open-sourced dataset which support our work.

## Reference
1. C. Zuo, J. Sun, and Q. Chen, “Adaptive step-size strategy for noise-robust Fourier ptychographic microscopy,” Opt. Express 24(18), 20724-20744 (2016). 
2. L. Tian, X. Li, K. Ramchandran, and L. Waller, “Multiplexed coded illumination for Fourier Ptychography with an LED array microscope,” Biomed. Opt. Express 5(7), 2376–2389 (2014).
3. L. Tian, Z. Liu, L.-H. Yeh, M. Chen, J. Zhong, and L. Waller, “Computational illumination for high-speed in vitro Fourier ptychographic microscopy,” Optica 2, 904–911 (2015).


