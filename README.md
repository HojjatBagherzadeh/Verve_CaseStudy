# Verve Group data science case study

## Gender prediction (Male/Female) task:

In this task, a dataset of 3700 training samples was given to predict the gender of a device user so that we could help the advertisers to target within the apps. The features of the dataset are given below:
* device_name: the name given to the device, e.g., "Maria's iPhone"
* app_category: the classification of the app content, e.g., "fashion"
* interaction_with_app: how long the user interacted with the app during this session (in minutes)
* ad_category: the classification of the ad that was shown during the event (e.g., "clothing")
* click: whether the user clicked the ad (yes/no)

Challenges of the task and proposed solutions are discussed below:
## Challenges:
1.	Unbalanced dataset: The first challenge is the unbalance in the dataset, as the frequency of male targets is twice the other class. Thus, we have to define the proper loss function in order to compensate for the unbalance of the dataset. Another solution for such problems is to augment the data to balance the target classes, but since almost all the features are categorical, fabricating and augmenting the dataset would make false-positive cases. Therefore, it has to be done with much care. 
2.	Missing values: One of the most valuable features in this dataset is the device_name feature. With help of NLP techniques, one can find out the gender of the sample with accepted accuracy. But this feature has almost 60% missing values and the point is we can’t fabricate this feature based on other factors of the dataset. Therefore, we can only use this feature when the value is not missing.
3.	Data normalization/standardization: In order to use features in any machine learner, we have to standardize the features, and since these features are not the same type, this normalization process could be challenging. For example, the device_name feature can’t be used with others in the same manner. The interaction_with_app feature is also given as time and has to convert to a categorical feature. 
4.	Few training samples: Another challenge for this task based on its specification is having very few training samples. As mentioned before, data augmentation is a challenge for this task. Therefore, we have to design models in a way that works with very few samples or to use pre-trained models and only fine-tune the model. 

## Features: 
1.	*Device_name:* This feature can be used with the help of NLP techniques. First, we try to extract the name text from the device_name feature and then we try to classify the gender. Because of the 60% missing value in the dataset, we can only use this feature and its NLP model when the data is available as a peripheral model. 
2.	*App_category:* This is a categorical feature and its missing value is less than 2%. But since more than half of all sample belongs to a certain category, we have to combine this feature with other features like ad_category and interaction_with_app as a compound feature to be able to use it in a model.
3.	*Ad_category:* This is also a categorical feature and the diversity of classes is high enough that this feature can be used individually.
4.	*Interaction_with_app:* This is a numerical feature and to cope with other features of the dataset, we have to convert it. As shown in the time interacting with the app plot, based on the frequency values, we can define unequal time slots (with borrowing the histogram equalization technique from image processing). In this manner, the categorical interaction_with_app feature would have enough diversity and can be used as an individual feature. 
5.	*Click:* Even though this is a two-class feature because it has very low entropy, we can’t use this feature in our approach. 


## Proposed models:
1.	**A boosting algorithm (Adaboost/XGboost/Gradient boost):** Our first approach would be using a boosting algorithm. Based on our features (device_name, app_category,…) we can design some weak learners and combine them with a boosting algorithm to have a strong classifier. Our weak classifiers can be a combination of feature categories. 
One of the advantages of this approach is the ability to define and combine weak classifiers to build a strong model. However, because of having few samples, the model might overfit and doesn't have great performance on the real test data.
2.	**A hybrid method:** Another method would be a combination of an NLP technique for the device_name feature and a kernel-SVM for other categorical features. 
The main advantage of this method is to have an NLP machine for the device_name feature which can give the correct answer with great confidence for the available values. But the import drawback of using hybrid methods is how to combine(fusion) individual model results and give the final answer as the confidence score of each model can mislead the final result. 
3.	**Other hybrid methods** like Naïve Bayes, Decision Trees, and … can also be used.
4.	**Out of the box thinking:** Based on the sample data given in the task, we have multiple records of the same person in the dataset. If we can use more than one data row for the final classification, we can reexamine our results with help of other entries.    
