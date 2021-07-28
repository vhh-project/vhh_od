import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import random
import sklearn.metrics

used_classes = ["others", "soldier", "corpse", "person_with_kz_uniform"]
n_classes = len(used_classes)

test_transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        # Normalization from our data:
        transforms.Normalize(mean=[76.031337/255, 76.031337/255, 76.031337/255], std=[57.919809/255, 57.919809/255, 57.919809/255]),
    ])

def class_to_idx(class_name):
    return used_classes.index(class_name)

def idx_to_class(idx):
    return used_classes[idx]

def get_class_distribution(dataset):
    samples = [0 for i in used_classes]
    for i, sample in enumerate(dataset):
        label = sample["class"]
        samples[label] += 1

    print([float(i) for i in samples])
    return [float(i)/len(dataset) for i in samples]

def output_to_prediction(outputs):
    return [np.argmax(outputs[i]) for i in range(outputs.shape[0])]

def get_metrics(outputs, labels, do_class_metrics = False):
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, outputs, average="macro")
    metrics = {"macro_prec": precision, "macro_rec": recall, "macro_f1": f1}

    if do_class_metrics:
        for label in range(n_classes):
            precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, outputs, average="macro", labels = [label])
            metrics["prec ({0})".format( idx_to_class(label))] = precision
            metrics["rec ({0})".format(idx_to_class(label))] = recall
            metrics["f1 ({0})".format(idx_to_class(label))] = f1

    nr_correct_predictions = sum([1 for i in range(len(outputs)) if outputs[i] == labels[i]])
    accuracy = float(nr_correct_predictions) / len(outputs)
    metrics["accuracy"] = accuracy
    return metrics

class ClassifierModel(nn.Module):
    def __init__(self, model_name):
        super(ClassifierModel, self).__init__()

        if model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, 4)
        elif model_name == "vgg11_bn":
            self.model = models.vgg11_bn(pretrained=True)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,4)
        elif model_name == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(pretrained=True)
            self.model.fc = nn.Linear(2048, 4)
        self.input_size = 224

    def forward(self, x):
        return self.model(x)

class PersonsDataset(Dataset):
    def __init__(self, img_folder, shuffle=True, transforms = None, dropFilesFrom = None):
        """
        Args:
            img_folder (string): The folder containing the class folders
            dropFilesFrom (list of tuples (string, int)): 
                Can be used to downsample or upsample data. 
                [("corpse", 5)] would mean that only every fifth image from the corpse folder would be used.
                [("corpse", 5), ("soldier", -3)] means the same as above and that every image from the soldier folder will be used three times. 
        """
        self.img_folder = img_folder
        self.transform = transforms

        self.data = []


        do_drop_classes = dropFilesFrom is not None
        if do_drop_classes:
            dropped_classes, acceptance_mod = list(zip(*dropFilesFrom))
            print(dropped_classes, acceptance_mod)


        # Load data
        for directory in os.listdir(img_folder):
            if directory in used_classes:
                for i, path in enumerate(Path(os.path.join(img_folder, directory)).rglob('*.png')):
                    # Up and downsampling
                    if do_drop_classes and directory in dropped_classes:
                        acceptance = acceptance_mod[dropped_classes.index(directory)] 
                        # Drop samples if acceptance > 0 (downsampling)
                        if acceptance > 0 and i % acceptance != 0:
                            continue
                        # Import samples multiple times if acceptance < 0 (upsampling)
                        elif acceptance < 0:
                            for j in range(int(-acceptance)):
                                self.data.append((str(path), class_to_idx(directory)))

                    self.data.append((str(path), class_to_idx(directory)))

        # Shuffle
        if shuffle:
            np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        img_class = self.data[idx][1]
        img =  cv2.imread(img_path)
        np_img = np.asarray(img)

        if self.transform:
            np_img = self.transform(np_img)

        return {"image": np_img, "class": img_class}

def evaluate(dataloader, name, epoch, device, model, criterion, do_class_metrics = False):
    model.eval()
    with torch.no_grad():
        curr_loss = 0
        predictions_train = []
        labels_train = []
        for batch in dataloader:
            inputs = batch["image"].to(device)
            labels = batch["class"].to(device)

            outputs = model(inputs)

            labels_train += labels.cpu().detach().numpy().tolist()
            predictions_train += output_to_prediction(outputs.cpu().detach().numpy())

            loss = criterion(outputs, labels)
            curr_loss += loss.cpu().item()

        metrics = get_metrics(predictions_train, labels_train, do_class_metrics=do_class_metrics)
        curr_loss = curr_loss / len(dataloader)
        metrics["total_loss"] = curr_loss
        print("\t{0} loss: {1}\n\t{0} accuracy: {2}".format(name, curr_loss, metrics["accuracy"]))
        return metrics

