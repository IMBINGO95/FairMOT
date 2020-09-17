import torch.utils.data
from torch.autograd import Variable
from SVHN_data_preprocess.SVHN_class import DatasetV1,SJNDataset,SWDataset


class SVHN_Evaluator(object):
    def __init__(self, path_to_data_dir,mode,crop = False):
        self._loader = torch.utils.data.DataLoader(DatasetV1(path_to_data_dir,mode,crop), batch_size=128, shuffle=False)

    def evaluate(self, model):
        model.eval()
        num_correct = 0
        needs_include_length = True

        for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
            images, length_labels, digits_labels = (Variable(images.cuda(), volatile=True),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model(images)
            '''This max function return two column, the first row is value, and the second row is index '''
            length_predictions = length_logits.data.max(1)[1] 
            digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

            if needs_include_length:
                num_correct += (length_predictions.eq(length_labels.data) &
                                digits_predictions[0].eq(digits_labels[0].data) &
                                digits_predictions[1].eq(digits_labels[1].data) &
                                digits_predictions[2].eq(digits_labels[2].data) &
                                digits_predictions[3].eq(digits_labels[3].data) &
                                digits_predictions[4].eq(digits_labels[4].data)).cpu().sum()
            else:
                num_correct += (digits_predictions[0].eq(digits_labels[0].data) &
                                digits_predictions[1].eq(digits_labels[1].data) &
                                digits_predictions[2].eq(digits_labels[2].data) &
                                digits_predictions[3].eq(digits_labels[3].data) &
                                digits_predictions[4].eq(digits_labels[4].data)).cpu().sum()

        accuracy = float(num_correct.item() / (len(self._loader.dataset)))
        return accuracy, num_correct , len(self._loader.dataset)

class SJN_Evaluator(object):
    def __init__(self, path_to_data_dir,mode,label_type,crop = False):
        self.path_to_data_dir = path_to_data_dir
        self.mode = mode
        self.label_type = label_type
        self.crop = crop

    def evaluate(self, model):
        _loader = torch.utils.data.DataLoader(SJNDataset(self.path_to_data_dir,self.mode,self.label_type,self.crop), batch_size=128, shuffle=False)
        model.eval()
        num_correct = 0
        needs_include_length = True

        for batch_idx, (images, length_labels, digits_labels) in enumerate(_loader):
            images, length_labels, digits_labels = (Variable(images.cuda(), volatile=True),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model(images)
            '''This max function return two column, the first row is value, and the second row is index '''
            length_predictions = length_logits.data.max(1)[1]
            digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

            if needs_include_length:
                num_correct += (length_predictions.eq(length_labels.data) &
                                digits_predictions[0].eq(digits_labels[0].data) &
                                digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()
            else:
                num_correct += (digits_predictions[0].eq(digits_labels[0].data) &
                                digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()

        accuracy = float(num_correct.item() / (len(_loader.dataset)))

        return accuracy, num_correct , len(_loader.dataset)
    def calculate_scores(self,model):
        _loader = torch.utils.data.DataLoader(SJNDataset(self.path_to_data_dir,self.mode,self.label_type,self.crop), batch_size=1, shuffle=False)
        model.eval()
        num_correct = 0
        digit_score_correct = []
        length_score_correct = []
        digit_score_wrong = []
        length_score_wrong = []

        for batch_idx, (images, length_labels, digits_labels) in enumerate(_loader):
            images, length_labels, digits_labels = (Variable(images.cuda(), volatile=True),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])

            length_logits, digits_logits = model(images)
            '''This max function return two column, the first row is value, and the second row is index '''
            length_predictions = length_logits.data.max(1)[1]
            length_score = length_logits.data.max(1)[0]
            digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]
            digits_score = [digit_logits.data.max(1)[0] for digit_logits in digits_logits]

            equal = (length_predictions.eq(length_labels.data) &
                            digits_predictions[0].eq(digits_labels[0].data) &
                            digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()

            if equal:
                num_correct += 1
                length_score_correct.append(length_score)
                digit_score_correct.append(digits_score)
            else:
                length_score_wrong.append(length_score)
                digit_score_wrong.append(digits_score)

        digit_score_correct = torch.tensor(digit_score_correct)
        length_score_correct = torch.tensor(length_score_correct)
        digit_score_wrong = torch.tensor(digit_score_wrong)
        length_score_wrong = torch.tensor(length_score_wrong)


        accuracy = float(num_correct / (len(_loader.dataset)))
        return accuracy, num_correct, len(_loader.dataset), (digit_score_correct, length_score_correct,digit_score_wrong,length_score_wrong)

class SW_Evaluator(object):
    def __init__(self, path_to_data_dir,mode,crop = False):
        self._loader = torch.utils.data.DataLoader(SWDataset(path_to_data_dir,mode,crop), batch_size=1, shuffle=False)

    def evaluate(self, model):
        model.eval()
        num_correct = 0
        needs_include_length = True
        valid_num = 0
        invalid_num = 0
        valid_data = []

        for batch_idx, (images, length_labels, digits_labels,sub_data,file_name) in enumerate(self._loader):
            images, length_labels, digits_labels = (Variable(images.cuda(), volatile=True),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model(images)
            '''This max function return two column, the first row is value, and the second row is index '''
            '''if num exits in the img'''
            length_predictions = length_logits.data.max(1)[1]
            length_pre = int(length_predictions.data)


            if length_pre == 0 :
                invalid_num += 1
                continue
            else:
                valid_num += 1

                digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

                if needs_include_length:
                    if_equal = (length_predictions.eq(length_labels.data) &
                                    digits_predictions[0].eq(digits_labels[0].data) &
                                    digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()
                else:
                    if_equal = (digits_predictions[0].eq(digits_labels[0].data) &
                                    digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()

                sub_data = sub_data.tolist()
                sub_data.append(file_name[0])
                sub_data.append(int(if_equal))

                valid_data.append(sub_data)
                num_correct += if_equal

        accuracy = float(num_correct.item() / valid_num)

        return accuracy, num_correct , valid_num, invalid_num, valid_data