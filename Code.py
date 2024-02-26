from teachableMachine import predict
from keras.models import load_model
from PIL import Image

model = load_model('keras_model.h5')
image = Image.open('test-photo-2.jpg')

x=predict(image,model)

if x[0][0]>0.75:
    print('metal can with confidence :', x[0][0])

elif x[0][1]>0.75:
    print('plastic bottle with confidence :', x[0][1])

else:
    print('unknown')