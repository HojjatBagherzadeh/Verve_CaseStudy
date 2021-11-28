#Verve Group data science case study

Gender prediction (Male/Female) task:

In this task, a dataset of 3700 training sample was given in order to predict the gender of a device user so that we could help the advertisers to target within the apps. The features of dataset are given below:
•	device_name: the name given to the device, e.g., "Maria's iPhone"
•	app_category: the classification of the app content, e.g., "fashion"
•	interaction_with_app: how long the user interacted with the app during this session (in minutes)
•	ad_category: the classification of the ad that was shown during the event (e.g., "clothing")
•	click: whether the user clicked the ad (yes/no)

Challenges of the task and proposed solutions are discussed below:
Challenges:
1.	Unbalanced dataset: The first challenge is the unbalancy in the dataset, as the frequency of male targets are twice the other class. Thus, we have to define the proper loss function in order to compensate for the unbalancy of dataset. Another solutions for such problems is to augment the data to balance the target classes, but since almost all the features are categorical, fabricating and augmenting the dataset would make false positive cases. Therefore, it has to done with much care. 
2.	Missing values: One of the most valueable feature in this dataset is the device_name feature. With help of NLP techniques, one can find out the gender of the sample with accepted accuracy. But this feature has almost 60% missing values and the poinst is we can’t fabricate this feature based on other factors of the dataset. Therefore, we can only use this feature when the value is not missing.
3.	Data normalization/standardization: In order to use features in any machine leariner, we have to standarzie the features, and since these features are not the same type, this normalization process could be challenging. For example, the device_name feature can’t be used with others in the same manner. The interaction_with_app feature is also given as time and has to converted to a categorical feature. 
4.	Few training samples: Another challenge for this task based on its specification is having very few training samples. As mentioned before, data augmention is a challenge for this task. Therefore, we have to design models in a way that work with very few samples or to use pretrained models and only fine-tune the model. 

Features:
•	Device_name: This feature can be used with the help of NLP techniques. First we trye to extract the name text from device_name feature and then we try to classify the gender. Because of 60% missing value in the dataset, we can only use this feature and its NLP model when the data is available as a peripheral model . 
•	App_category: This is a categorical feature and its missing value is less than 2%. But since more that half of all sample belongs to a certain category, we have to combine this feature with other features like ad_category and interaction_with_app as a compound feature in order to be able to use it in a model.
•	Ad_category: This is also a categorical feature and the diversity of classes is high enough that this feature can be used individually.
•	Interaction_with_app: This is a numerical feature and in order to cope with other features of the dataset, we have to convert it. As shown in the time interacting with the app, based on the frequency values, we can define unequal time slots (with borrowing the histogram equalization technique from image processing). In this manner, the categorical interaction_with_app feature would have enough diversity and can be used as an individual feature. 
•	Click: Even though this is a two class feature, because of having very low entropy, we can’t use this feature in our approach. 

Proposed models:
•	A boosting algorithm (Adaboost/XGboost/Gradient boost): Our first approach would be using a boosting algorithm. Based on our features (device_name, app_category,…) we can design some weak learners and combine them with a boosting algorithm to have a strong classifier. Our weak classifiers can be combination of feature categories. 
One of the advantages of this approach is the ability to define and combine weak classifiers to build the strong model. However, because of having few samples, the model might overfit and doesn't have great performance on the real test data.
•	A hybrid method: Another method would be combination of an NLP technique for device_name feature and a kernel-SVM for other categorical features. 
The main advantage of this method is to have an NLP machine for the device_name feature which can give the correct answer with greate confidence for the available values. But the import drawback of using hybrid methods is how to combine(fussion) individual model results and give the final answer as the confidence score of each model can mislead the final result. 
•	Other hybrid method like Naïve Bayes, Decision Trees and … can also be used.
•	Out of the box thinking: Based on the sample data given in the task, we have multiple records of the same person in the dataset. If we are able to use more that one data row for the final classification, we can reexamine our results with help of other entries. 
