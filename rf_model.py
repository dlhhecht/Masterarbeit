import numpy as np 
import os
from keras.applications.vgg16 import VGG16, preprocess_input
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.ensemble import RandomForestClassifier

###########################################
#pre-settings, loading VGG16
###########################################

SIZE = 256
#Load model without classifier/fully connected layers and make loaded layers as non-trainable.
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
for layer in VGG_model.layers:
	layer.trainable = False
    
#VGG_model.summary() #trainable params should be 0

###########################################
#functions
###########################################

def preprocess_with_VGG16(img):
	
	# 
	# Pre: 
	#	img: 		4 dimensional numpy array with samples, rows, columns and channels
	# Post:
	#	features: 	1 dimensional numpy array with features
	#
	# Description:
	# 	Feature extraction with VGG16 and loaded weights.
	#
	
    features = VGG_model.predict(img)
    features = features.reshape(-1)   #features.reshape(features.shape[0], -1)
	
    return features

def load_images_and_labels(path,size):

	#
	# Pre:
	#	path:		string with folder path
	#	size: 		integer with requested image size (size x size)
	# Post:
	#	df: 		pandas dataframe with columns ['object_id','image','label','feature'] 
	#
	# Description:
	#	Function loads Data from folder in path
	#
	
    df = pd.DataFrame(columns=['object_id','image','label','feature'])
    img_list = os.listdir(path)
    
    for i in range(0,len(img_list)):        
        img = load_img(os.path.join(path, img_list[i]), target_size=(size, size))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        obj_id = img_list[i].split("_")[0]
        label = img_list[i].split("_")[2][:-4] 
        features = preprocess_with_VGG16(img_array)
        new_data = {
            'object_id': [obj_id],
            'image': [img_array],
            'label': [label],
            'feature' : [features]
            }

        new_row = pd.DataFrame(new_data)
        df = pd.concat([df, new_row], ignore_index=True)
        
    return df

def elementwise_max(df_input):

	# 
	# Pre:
	#	df_input:	pandas dataframe with columns ['object_id','image','label','feature'] 
	# Post:
	#	df:		pandas dataframe with columns ['object_id','label','feature'] 
	#
	# Description:
	#	From here we start with multi view. Function takes the extracted features of every image in the data set
	#	as input and does elementwise max operation for every image with the same object id (Multi View).
	#

    df = pd.DataFrame(columns=['object_id','label','feature'])
    for name, group in df_input.groupby('object_id'):
        group_with_same_obj_id = np.asarray(group['feature'])
        curr_feature = group_with_same_obj_id[0]
        for x in range(0,group_with_same_obj_id.shape[0]):
            for y in range(0,len(group_with_same_obj_id[0])):
                if group_with_same_obj_id[x][y]>curr_feature[y]:
                    curr_feature[y] = group_with_same_obj_id[x,y]
        
        new_data = {
            'object_id': [np.asarray(group['object_id'])[0]],
            'label': [np.asarray(group['label'])[0]],
            'feature' : [curr_feature]
            }

        new_row = pd.DataFrame(new_data)
        df = pd.concat([df, new_row], ignore_index=True)
	    
    return df

###########################################
#main code
###########################################

df_train_set = load_images_and_labels('images/train', SIZE)
df_test_set = load_images_and_labels('images/validation', SIZE)

df_train_set_grouped_and_maxed = elementwise_max(df_train_set)
df_test_set_grouped_and_maxed = elementwise_max(df_test_set)

X_train = df_train_set_grouped_and_maxed['feature'].apply(pd.Series)
y_train = df_train_set_grouped_and_maxed['label']

X_test = df_test_set_grouped_and_maxed['feature'].apply(pd.Series)
y_test = df_test_set_grouped_and_maxed['label']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)








