# Nucleui segmentation-tensorflow-slim (in progress)
  
This source is tensorflow version of [1].   
We used tensorflow-slim and implemented on ubuntu 16.04, python3.5, and tensorflow1.13.0rc1.  
   



## NOTE

1) To extract nuclei and non-nuclei patch, we used original matlab patch extraction code[3]   
because converted python code[1] decreased segmentation accuracy significantly.   
However, we are still struggling to correct step1_patch_extraction.py python code.   

2) Andrew[3] used the modified Alexnet but we use cifarnet included in tensorflow slim instead.  
For this reason, our result of segmentation(middle) are not so good as caffe version(right) like below.  
 

<div>
<img src="https://user-images.githubusercontent.com/46143444/53543054-8e333600-3b64-11e9-89c5-910007a497a1.png" width="90%"></img>
</div>




## Requirments
python3.5  
tensorflow

Current version ran on CPU. Install tensorflow-gpu version and chagne below code if we want to run on GPU.

```bash
   cd DEEP_TUTORIAL_ROOT
   gedit step4_train_image_classifier.py

   ------------------------------------------------------
   tf.app.flags.DEFINE_boolean('clone_on_cpu', True,
                            'Use CPUs to deploy clones.')
   change to 

   tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
   -----------------------------------------------------

```

## Overview  
### Quick start

User can run test step(step5 and step6) with a little modfication in below script using a pre-trained model.   
For instance, change CHECKPOINT_DIR variable to the directory where a pre-trained or user-generated model exists.    

```bash
   cd DEEP_TUTORIAL_ROOT
   ./train_test_nuclei.sh
```

### Data  
Download [dataset(train and valiation tfRecord)](http://naver.me/Fyamxy1v)
URL_PASSWORD: 1234

```bash
   cd DEEP_TUTORIAL_ROOT/data/1-nuclei/images
   mv DOWNLOAD_DIR/nuclei* ./
```


For step6 segmentation, original image is on [here](http://andrewjanowczyk.com/wp-static/nuclei.tgz)
 

### Training/Evalution  
```bash
   cd DEEP_TUTORIAL_ROOT/
   step1_patch_extraction.py (not recommended!. use original patch extraction matlab code)
   step2_cross_validation_creation.py 
   step3_generate_datasets (make tensorflow-slim data format 
   step4_train_image_classifier.py 
   step5_eval_image_classifier.py 
```

### Segmentation
```bash
   cd DEEP_TUTORIAL_ROOT/
   step6_segment_test_images.py (segmentation for test image)
```

### Output generation time
It taks to segment an image very long time (almost ~65 minutes / orginal caffe ~75 minutes on one 1080ti GPU)   
Fortunately, Andrew[3] reduced processing time considerably in [4].   

## Acknowledgements  
 We would like to thank the authors of DLtutorialCode[2], which we use in this work.

## References  
[1][python version of [3]](https://github.com/ai-lab-circle/deep_tutorial_python)   
[2]https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim   
[3][original source](https://github.com/choosehappy/public/tree/master/DL%20tutorial%20Code)   
[4]http://www.andrewjanowczyk.com/efficient-pixel-wise-deep-learning-on-large-images/   


