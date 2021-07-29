import vhh_od.Classifier as Classifier
import torch.optim as optim
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import random, os
"""
Trains the person classifier.
"""


data_path_train = "/data/ext/VHH/datasets/Classifier_data/train_val"
data_path_test = "/data/ext/VHH/datasets/Classifier_data/test"
model_folder = "/data/share/fjogl/Classifier_models"

do_early_stopping = True
max_epochs_without_improvement = 20

batchsize = 32
epochs = 1000
lr = 0.0001
# Momentum for SGD
momentum = 0.9
train_size_percentage = 0.8

optimizer_name ="AdamW"
model_name = "wide_resnet50_2"

class DatasetWrapper(Dataset):
    """"
    A wrapper around a dataset, so we can apply transforms to it.
    The reason we need this is because if we split a dataset into subsets then normally we can only assign a transform to BOTH of them. 
    This class fixes this.
    from ptrblck @ pytorch forum in "Torch.utils.data.dataset.random_split"
    """
    def __init__(self, subset, transform = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        data = self.subset[idx]
        img = data["image"]
        _class = data["class"]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "class": _class}

    def __len__(self):
        return len(self.subset)

def log_metrics(metrics, epoch, name):
    log = {}
    for item in metrics.items():
        log[name + "/" + item[0]] = item[1]
    log["epochs"] = epoch
    wandb.log(log)

def main():
    # Set seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model = Classifier.ClassifierModel(model_name)
    model.to(device)

    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters()))


    transform_train =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)), 
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomVerticalFlip(p=0.2),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # Normalization from our data:
        transforms.Normalize(mean=[82.0899/255, 82.0899/255, 82.0899/255], std=[58.8442/255, 58.8442/255, 58.8442/255]),
    ])

    transform_test = Classifier.test_transform
    
    train_val_set = Classifier.PersonsDataset(data_path_train)
    testset = Classifier.PersonsDataset(data_path_test, dropFilesFrom=None)
    trainset, valset = torch.utils.data.random_split(train_val_set, ((int(len(train_val_set)*train_size_percentage), len(train_val_set) - int(len(train_val_set)*train_size_percentage))))

    trainset_size = len(trainset)
    valset_size = len(valset)
    testset_size = len(testset)

    print("Trainset size: {0} (class distr: {1})\nValset size: {2} (class distr: {3}\nTestset size: {4} (class distr. {5}".format(
        trainset_size, Classifier.get_class_distribution(trainset), valset_size, Classifier.get_class_distribution(valset), testset_size, Classifier.get_class_distribution(testset)))

    trainset = DatasetWrapper(trainset, transform_train)
    valset = DatasetWrapper(valset, transform_test)
    testset = DatasetWrapper(testset, transform_test)

    train_dataloader = DataLoader(trainset, batch_size=batchsize, shuffle = True)
    val_dataloader = DataLoader(valset, batch_size=batchsize, shuffle = True)
    test_dataloader = DataLoader(testset, batch_size=batchsize, shuffle = True)

    criterion = nn.CrossEntropyLoss()
    
    config = {
        "model_name": model_name,
        "learning_rate": lr, 
        "max_epochs_without_improvement": max_epochs_without_improvement, 
        "train_set_percentage": train_size_percentage,
        "optimizer": optimizer_name,
        "batchsize": batchsize}

    # Todo: try sgd
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else: 
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        config["momentum"] = momentum

    wandb.init(
    entity="cvl-vhh-research",
    project="Train Persons Classifier",
    notes="",
    tags=[],
    config=config
    )

    print("Config: ", config)

    epochs_without_improvement = 0
    previous_best_f1 = -1
    for epoch in range(epochs):
        # Train
        model.train()
        curr_loss = 0
        predictions_train = []
        labels_train = []
        for i, batch in enumerate(train_dataloader):
            # for j in range(batch["image"].shape[0]):
                # cv2.imshow("hey", batch["image"][j].numpy().transpose(1, 2, 0))
                # cv2.waitKey(0)


            optimizer.zero_grad()

            inputs = batch["image"].to(device)
            labels = batch["class"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()


            labels_train += labels.cpu().detach().numpy().tolist()
            predictions_train += Classifier.output_to_prediction(outputs.cpu().detach().numpy())

            # Logging
            wandb.log({
            "train/loss":loss.cpu().item(),
            "epochs": epoch + float(i) / len(train_dataloader)
            })

        # Logging
        metrics = Classifier.get_metrics(predictions_train, labels_train)
        curr_loss = curr_loss / len(train_dataloader)
        metrics["total_loss"] = curr_loss
        log_metrics(metrics, epoch + 1, "train")

        print("Epoch {0}:\n\tTraining loss: {1}\n\tTraining accuracy: {2}".format(epoch, curr_loss, metrics["accuracy"]))

        # Validate
        val_metrics = Classifier.evaluate(val_dataloader, "validate", epoch + 1, device, model, criterion, do_class_metrics=False)
        log_metrics(val_metrics, epoch + 1, "validate")
        
        # Check if this is our best model yet
        if val_metrics["macro_f1"] > previous_best_f1:
            previous_best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(model_folder, "best_" + model_name + ".weights"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1            

        # Early stopping
        if do_early_stopping and epochs_without_improvement > max_epochs_without_improvement:
            print("Training will be stopped since validation macro_f1 has not improved for {0} epochs".format(epochs_without_improvement))
            break
        
    # Test model
    test_metrics = Classifier.evaluate(test_dataloader, "test", epoch + 1, device, model, criterion, do_class_metrics=True)
    log_metrics(test_metrics, epoch + 1, "test")
    print("Test metrics: ", test_metrics)



if __name__ == "__main__":
    main()