import cv2
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
def num_classifer(imgs,model,crop=True):

    model.eval()

    candidate = {}
    for img in imgs:
        img_ = Image.fromarray(img)

        size = img_.size

        if crop == True:
            '''only get the upper half of the player.'''
            region = (0,0,size[0],int(0.5*size[1]))
            img_ = img_.crop(tuple(region[0:4]))

        # IMG = image
        transform = transforms.Compose([
            transforms.Resize([64,64]),
            transforms.CenterCrop([54,54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img_ = transform(img_).unsqueeze(0)
        img_tensor = Variable(img_.cuda())

        length_logits, digits_logits = model(img_tensor)
        '''This max function return two column, the first row is value, and the second row is index '''
        length_predictions = length_logits.data.max(1)[1]
        digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

        length_score = length_logits.data.max(1)[0]
        digits_score = [digit_logits.data.max(1)[0] for digit_logits in digits_logits]

        '''add coefficient in front of the scores'''
        alpha = 1
        belta = 1
        gamma = 1
        if length_predictions == 0:
            num = -1
            scores = float(length_score)
        elif length_predictions == 1:
            num = int( alpha * digits_predictions[0])
            scores = float(alpha  * length_score + belta * digits_score[0]) / 2
        elif length_predictions == 2:
            num = int(10 * digits_predictions[0] + digits_predictions[1])
            scores = float(alpha * length_score + belta * digits_score[0] + gamma * digits_score[1]) / 3


        if num in candidate:
            candidate[num] += scores
        else:
            candidate[num] = scores

    return max(candidate)

