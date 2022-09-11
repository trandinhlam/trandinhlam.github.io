---
layout: post
title:  "Thực hành Tensorflow: Mô hình phân loại bệnh trên lá xoài"
date:   2022-09-11
categories: neuron-network
---

+ Dataset: kaggle: Images of Mango Leaves
+ Model: Keras + Tensorflow
+ Author: tdlam123@gmail.com

# Giới thiệu:
Mình sinh ra và lớn lên ở vùng đất Cam Lâm, nơi có rất nhiều giống xoài ngon và và mang lại giá trị kinh tế cao cho người dân địa phương quê mình.
Hôm nay trong lúc họp hành với nhóm học trên trường mình có ý tưởng xây dựng một hệ thống nông nghiệp thông minh hỗ trợ người dân chăm sóc xoài.
Lục lọi các paper và một số mô hình sẵn có, mình đã thử thực hành vài bước cơ bản để đưa mô hình Machine Learning ra ứng dụng vào ý tưởng hệ thống nông nghiệp thông minh này.

# Thực hành:

Dữ liệu bao gồm các hình ảnh thu được từ các trang trại Xoài bị ảnh hưởng bởi 15 loại sâu bệnh có thể xác định được thông qua biến dạng cấu trúc và hình ảnh biểu hiện trên lá Xoài. Việc thu thập dữ liệu liên quan đến việc sử dụng thiết bị cảm biến giá rẻ thường được nông dân sử dụng để thu thập hình ảnh từ trang trại. Cụ thể:

+ mictis longicornis: bọ xít gây héo ngọn non, trái non
+ apoderus javanicus: bọ ăn lá
+ valanga nigricornis: châu chấu chim ăn lá
+ dappula tertia: sâu bướm ăn lá
+ normal: bình thường
+ neomelicharia sparsa: Rệp phấn trắng
+ dialeuropora Decempuncta: Bọ phấn trắng vân đen
+ icerya_seychellarum: Rệp sáp
+ procontarinia_matteiana: ruồi ghẻ
+ procontarinia_rubus
+ orthaga_euadrusalis
+ cisaberoptus_kenyae
+ erosomyia_sp
+ aulacaspis_tubercularis
+ ceroplastes_rubens
+ ischnaspis_longirostris


![Image pest classification](https://lh3.googleusercontent.com/fife/AAbDypDH1dDrowEU5oKl7LC-RN82g_TqAPI8gY9zp_tNgYontDs8mUW9G0ofm6jF4Y-kqEoaAjTsL-Vj_KQqB9y-DlPubxztYA5NMoLlAj0ro1AsAsRfAcgJDyGNoakuAZqMMvRdCOfkl8bPYsMZVxntqbsVHeTqJA18SVkg2CETL_1fwnGxaO_ns93VY0l7KpK4E0o8Y_wuJufHp7rjDQgMRCcVO9n2tfQ28rfepDIul7A4HuEn01NdEzt-x1VM7FGEa4B0phcwl_q5045NDALzSw6v8OmJ8JjVXexgtikVfQGYSXwxfjCNkB2tBHZVvTfh8_sJO7KLJDdRDD9Q_mXguDhcxPHL7H320QGw-x2Tw1WvenslmCwgJZqckcOdkRONCZ75zXtyrDXtroD5f55plUNf5EX5GWfVQiIOwDRjVScwzZxNY9AE2HTTzxg4iR1WlUgVE04A-_FRQiHeuDtx1WmRXix0SldekKNFSnUulZekKfCWXo9NuPjeinrEfKpv5nYq_1Ev6e8g6T-1llFSEjRcGVW-1ORy0ETFILN31C8TR3XVZzAMiW-wgHTRvJLYaaVz3o1n_W6TMw7oU2UhC8watbB7MDMfndIon5uTHbeNbAxJApyJ0YY0KF1Ycaol_GPtkMuTigoyCHTZUuQIBKl8sB4jwggiQkIuqd0TNa6UxMpvDPe5-VoYDMM07dkP_SLr8J1DQ21w_APMX_180GZ7POgUP2BfP0lFfGI1r3w7tuR9eCGAfDLMyIZF8Nl9BJDZtXKQS2cTM1t5VqcvgUkON1wvthSH9k15nyEdARWKoNBPTPGXi3kA8EuDWxcrwZKwJLCddOs9lUkesSsKP05WN0OO9_XSjXw3ounxKBuxtJzw0-yt1nTLTZUTXxZ0m4-zhjN0gt2Tryb6NwsM8CZeyi9v-R1nzkhkubOOVsxSxF_ECufzF5-11pnMbBHsTVku5d9hOwNEWNy_Ct0Dx9TgnoaVFDp2j1TfBYmzkZG7to5W38QDh_MbDp5prG3sptb5oOw5EN19tKDtWnGngar8wBPVRuZyKdVWQIE3asEFFxbXJrOG5XLBf_STZlAFkgbTPaQdgzVkkCwZc1RJm90Kvm22UhG4TCTWdF3SMcVIzz5hZ3Ukr7JWxbNgk-UHk2EYk5Pl1LZjMvA23RPrjHOBF2NQg14tNecD8tEOwGodd-T5evjeQF6ex_HkxhDERDBGuIzE1B8tDvVXTZuIF7Mu7ep72g1qD6I4_DHeBEfVVz7sbLP1ts3hadT23qazpZ9YhDYTmzqLT7zoKdgeVtUwbBAya4ZabBf3KY-SeULy=w1852-h887)


# Ý tưởng: Dùng VGG trước, và RepVGG sau để tăng tốc độ. 
Ta bắt đầu Download dataset vào thư mục

``` {.python}
!wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/94jf97jzc8-1.zip
!unzip /content/94jf97jzc8-1.zip
!unzip /content/MangoPestClassification_Original.zip
!unzip /content/MangoPestClassification_Version0.zip
!unzip /content/MangoPestClassification_Version1.zip
```

# Import các thư viện cần thiết & một số hằng số {#import-các-thư-viện-cần-thiết--một-số-hằng-số}

``` {.python}
import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras import callbacks
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt

# INPUT_SIZE = (500, 333)
INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
PATCH_SIZE = 64
DATASET_FOLDER = '/content/versi1'
LEARNING_RATE = 0.01
EPOCHS = 100
MOMENTUM=0.1
MODEL_FOLDER = '/content/drive/MyDrive/Colab/Mango_pest_classification/version1/'
MODEL_NAME='EfficientNetB2'

base_model = EfficientNetB2(include_top=False,weights='imagenet')
```

# Load dữ liệu sẵn sàng cho các tập train/validate/test
``` {.python}
gen = ImageDataGenerator(
                        #   rescale=1. / 255,
                        #  width_shift_range=[-75, 75],
                        #  height_shift_range=0.25,
                        #  horizontal_flip=True,
                        #  vertical_flip=True,
                        #  rotation_range=90,
                        #  brightness_range=[0.2, 1.0]
                         )

train = gen.flow_from_directory(DATASET_FOLDER+'/train/',
                                target_size=INPUT_SIZE,
                                batch_size=PATCH_SIZE,
                                shuffle=True,
                                subset='training')

valid = gen.flow_from_directory(DATASET_FOLDER+'/valid/',
                               target_size=INPUT_SIZE,
                               batch_size=PATCH_SIZE,
                               shuffle=True)

test = gen.flow_from_directory(DATASET_FOLDER+'/test/',
                               target_size=INPUT_SIZE,
                               batch_size=PATCH_SIZE,
                               shuffle=False)
```

# Xây dựng model dựa trên các backbone thông dụng (VGG,Resnet,Xception,Inception\...) {#xây-dựng-model-dựa-trên-các-backbone-thông-dụng-vggresnetxceptioninception}

``` {.python}
def getModel():
  # add a global spatial average pooling layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)

  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)

  # first: train only the top layers (which were randomly initialized)
  for layer in base_model.layers:
      layer.trainable = False
  return model

def getCustomModel():
  # add a global spatial average pooling layer
  x = base_model.output
  x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
  x = GlobalAveragePooling2D()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)

  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)

  # first: train only the top layers (which were randomly initialized)
  for layer in base_model.layers:
      layer.trainable = False
  return model
```

# Tiến hành train model và vẽ được loss, validate_acc:

``` {.python}
sgd = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
model = getModel()
acc = CategoricalAccuracy(name='acc')

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=[acc])
model.summary()

now = str(time.time())
cb_checkpointer = callbacks.ModelCheckpoint(filepath = MODEL_FOLDER + MODEL_NAME+'/'+MODEL_NAME+'_best_'+now+'_{epoch:02d}.hdf5', 
                                            monitor = 'val_loss',
                                            # save_best_only = True, 
                                            mode = 'auto')

history = model.fit(train, epochs=EPOCHS, validation_data=valid, shuffle=True, verbose=1, 
                    callbacks=[cb_checkpointer]
                    , workers=6, use_multiprocessing=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```

# Chạy trên tập test

# Kết quả chạy của mình trên một số backbone có sẵn:

| STT | Model          | version0 | version0 | version1 | version1 |
|-----|----------------|----------|----------|----------|----------|
|     |                |  val_acc | test_acc |  val_acc | test_acc |
| 1   | Resnet50       |  78.64%  |  69.70%  |  74.50%  |  72.10%  |
| 2   | VGG16          |  69.90%  |  60.61%  |  63.50%  |  67.01%  |
| 3   | VGG19          |  79.69%  |  66.67%  |  76.56%  |  66.67%  |
| 4   | EfficientNetB2 |  71.79%  |  67.19%  |  82.00%  |  67.19%  |


# Tạm kết: 
Trên đây là kết quả chạy train và test một số model nhất định để giải bài toán phân loại bệnh trên xoài.
Các phần khác của kiến trúc hệ thống nông nghiệp thông minh mình sẽ phác thảo và tiếp tục cập nhật ở những bài thực hành tiếp theo.
