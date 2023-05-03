
## Part 3 First Solution + Validation Performance

### Source codes

* The source code to run the validation sample is in: ../models/validate_single_image.py.

* To run this source code: python3 validate_single_image.py -t "test_img_path" -s "saved_model_path".

* I have done the validation only for the linear model, and in the final solution I will also perform the validation on the other models (Convolutional and ResNet). 

There is requirements in the requirements.txt file.

### A report

Our Face Emotion Recognition Project has three models, ResNet, Convolutional, and a simple MLP, to check the result of the face emotion recognition. 

* Simple feedforward neural network has 6 fully connected (or dense) layers with ReLU activation function in between. 

* LeNet model consists of two sets of convolutional layers followed by max pooling layers, and a fully connected layer with ReLU activation function. 

* Optimizer - Adam optimizer with a learning rate of 0.0001, 
* loss functioncross-entropy loss function.  
* Epochs= 50 
* Batch-size = 90 
* Logged the training process using WandB.

### A classification accuracy achieved on the training and validation sets:

* Developed a face emotion recognition model to recognize multiple emotions, such as happy, sad, neutral, angry, surprise, fear, and disgust. Since the training and the validation accuracy of my model is very low, it couldn’t perform well on the given sample images. I tested one picture for each emotions (angry, disgust, fear and happy), where the linear model could only recognize the happy faces. The result is shown below

* (NNproject) [psharma3@qa-1080ti-001 models]$ python3 validate_single_image.py -t ../images/sample_image.jpg -p ../saved_models/SimpleNet.pt Loading the pre-trained weights...... Loading the image and resizing torch.Size([3, 32, 32]) model prediction! The predicted label is tensor([3], device='cuda:0') 
#################### The model predicted the image to be happy. ########## 
FOR HAPPY IMAGE RECOGNIZED CORRECTLY 
#################### 

* (NNproject) [psharma3@qa-1080ti-001 models]$ python3 validate_single_image.py -t ../images/fear.jpg -p ../saved_models/SimpleNet.pt Loading the pre-trained weights...... Loading the image and resizing torch.Size([3, 32, 32]) model prediction! The predicted label is tensor([3], device='cuda:0') 
#################### 
 ** The model predicted the image to be happy ########## 
FOR FEAR IMAGE RECOGNIZED INCORRECTLY 
#################### 

* (NNproject) [psharma3@qa-1080ti-001 models]$ python3 validate_single_image.py -t ../images/angry.jpg -p ../saved_models/SimpleNet.pt Loading the pre-trained weights...... Loading the image and resizing torch.Size([3, 32, 32]) model prediction! The predicted label is tensor([6], device='cuda:0') 
#################### The model predicted the image to be surprise #################### 

* (NNproject) [psharma3@qa-1080ti-001 models]$ python3 validate_single_image.py -t ../images/disgust.jpg -p ../saved_models/SimpleNet.pt Loading the pre-trained weights...... Loading the image and resizing torch.Size([3, 32, 32]) model prediction! The predicted label is tensor([3], device='cuda:0') #################### The model predicted the image to be happy #################### ############****************************************************####################

### I have selected accuracy as my performance metric.

### A short commentary related to the observed accuracy and ideas for improvements

* Both the training and validation accuracy of the models are low. I can think of few areas I can work on to improve the accuracy of both the training and validation sets which are mainly by adjusting the hyperparameters of the model. 
* I realized that the batch-size while I trained the model was quite high, however, I couldn't re-train and re-validate the model because of the unavailability of the GPU server at crc and can do it for final solution. Moreover, I can also improve the accuracy by adjusting other hyperparameters or allowing the model to train for more epochs.

#### Training accuracy
* Resnet - 38.068 %, 
* LeNet - 36.767 %, 
* MLP - 32.456 %

#### Validation accuract
* Resnet - 36.878 %, 
* Lenet - 35.82 %, 
* MLP - 32.298 %

Plots are attaached in the folder: ../Plots_FER


## Part 4: Final Solution

1. Source Codes:

* The source code to run final solution on a single test sample: 
> ../models/test_single_image.py

* To run this source code: python3 test_single_image.py.

* There is all the package requirements in the requirements.txt file.

2. A report:

* The database I collected for the test purpose are one of the folder that I had not used for the training and testing  purpose which I obtained from the kaggle (../images/new_test_samples), and another datasets are randomly picked from the google (../images/new_google_test_images) for each of the emotion.

* Regarding the size of the dataset, new_test_samples contains 35 images, 5 images for each of the emotion and new_google_test_images contains one image for each of the emotion.

* Since I have used the unknown images for the testing purpose which was never used during the training or the validation purpose, I believe, this difference is sufficient to know the generalization capabalities of the models.

* ## accuracy achieved on the test set

Previously the training and the testing accuracy was very low therefore the models performed very bad on the unknown samples. 

However, I tuned some of the hyperparameters of the models from which I obtained the quite good accuracy of the models. I performed the test on same samples for both the old trained models and on newly trained  models.

The hyperparameters I tuned are:

* learning rate =  0.001 
    * previously, it was 0.0001
* Epochs =  60 
    * previously, it was 50
* Batch-size =  50
    * previously, it was 90

Comparison of the accuracies:

* Training accuracy 
    * MLP - 32.456 
      * newMLP - 32.117
    * LeNet - 36. 767 
      * newLeNet- 47.658
    * Resnet -38.068
      * newResnet - 50.707

* Validation accuracy
    * MLP - 32.298
      * newMLP - 31.813
    * LeNet - 35.82
      * newLeNet - 45.291
    * Resnet - 36.878
      * newResnet - 47.806

Hence, we can clearly observe that both the training and the validation accuracies increased significantly for the leNet and Resnet. Therefore, we perforemed test for the LeNet and the Resnet only.

These results can be clearly seen on the file FER_newplots.

* ## accuracy achieved on the test set
The label we have provided for different emotions are:
angry = 0
disgust = 1
fear = 2
happy = 3
neutral = 4
sad = 5
surprise = 6

* Based on these labels we took new five test image samples for each of the emotions which was not used during training and validation, so there were 35 new images together.

   
* The accuracy obtained on the testset with the LeNet model is  = 20% which is very low as compared to the training and the validation accuracy which is almost around 50%.


* Similarly the accuracy obtained on the testsets with the Resnet model is 25%.

* to check the test accuracy:
  > python test_accuracy.py

* to perform test on single image:
  > python3 test_single_image.py

* [Copied from Terminal

   > (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for surprise, newLenet is 6]


Results:

* Training accuracy in %
    * MLP - 32.456 
      * newMLP - 32.117
    * LeNet - 36. 767 
      * newLeNet- 47.658
    * Resnet -38.068
      * newResnet - 50.707

* Validation accuracy in %
    * MLP - 32.298
      * newMLP - 31.813
    * LeNet - 35.82
      * newLeNet - 45.291
    * Resnet - 36.878
      * newResnet - 47.806

* Testing accuracy
   * MLP: 8% (sometimes less than 8%)
   * LeNet: 20% (sometimes less than 20%)
   * ResNet: 20% (sometimes less than 20%)

    * The result of the testsets are attached in folder "test_results"

Hence, we can clearly expect to see the worst result on the acuuracy of test samples performed on three different architectures.


For instance: 
* The path for the test image is: " ../images/new_test_samples"

* For Lenet

* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for surprise, newLenet is 3
(NNproject) [psharma3@crcfe01 models]$ 

    * In this case, newLenet model has predicted the wrong label for the suprise which must be 6 instead of 3.

* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for fear, newLenet is 2
(NNproject) [psharma3@crcfe01 models]$ 

    * for fear it has predicted true which is 2

* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for angry, newLenet is 6
    * for angry it has predicted label surprise iwhich is false.

* For Resnet
* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for surprise, Resnet is 6
(NNproject) [psharma3@crcfe01 models]$ 

    * for suprise it has predicted true.

* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for fear, Resnet is 3
    * for fear image it has predicted as happy which is fasle prefiction.

* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for disgust, Resnet is 3
* for disgust, it has prediced as happy which is false prediction.

* For SimpleNet,
* (NNproject) [psharma3@crcfe01 models]$ python3 test_single_image.py
(48, 48) RGB
The predicted class for neutral, Resnet is 6

    * For neutral, it has predicted as surprise which is false prediction.



* The test results are  attached in the folder "test_results" inside NNproject



There can be several reasons why the model might have performed poor: 

> Poor quality of training data: The training data might have errors, noise, or bias that makes it difficult for the model to learn the true underlying patterns.

> Insufficient training data: The model might not have enough training data to learn the underlying patterns in the data.

> Incorrect hyperparameters: The hyperparameters of the model, such as learning rate, regularization strength, and batch size, might not have still been tuned properly.

> Overfitting: The model might be overfitting to the training data, which means it is memorizing the training set instead of learning general patterns that can be applied to new data.

> Underfitting: The model might be underfitting, which means it is too simple to capture the complexity of the data.

> Data imbalance: The distribution of data across classes might be imbalanced, with some classes having much fewer samples than others. This can lead to bias in the model towards the majority classes.



    















 







