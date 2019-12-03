from functions import *


model=get_model("new chest model1.pth")
transform=get_PIL_transform()

image=image_to_tensor("\photos\pnemonia\person515_bacteria_2186.jpeg",transform)
image=image[None, :, :]
result=get_result(image,model)
print(result)