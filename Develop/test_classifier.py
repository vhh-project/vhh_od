import vhh_od.Classifier as Classifier
import torch
from torch.utils.data import DataLoader
from torch import nn


"""
Test a pretrained model on the test set.
Ensure that you set both the path to the model and the name of the used architecture.
"""

data_path_val = "/data/ext/VHH/datasets/Classifier_data_final/val"
data_path_test = "/data/ext/VHH/datasets/Classifier_data_final/test"

model_path = "/data/ext/VHH/datasets/Classifier_models/wd_0.05_best_wide_resnet50_2.weights"
model_name = "wide_resnet50_2"

batchsize = 32

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model = Classifier.ClassifierModel(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    testset = Classifier.PersonsDataset(data_path_test, transforms=Classifier.test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle = True)
    valset = Classifier.PersonsDataset(data_path_val, transforms=Classifier.test_transform)
    val_dataloader = DataLoader(valset, batch_size=32, shuffle = True)
    criterion = nn.CrossEntropyLoss()

    print("------VALIDATION SET------")
    metrics = Classifier.evaluate(val_dataloader, "validate", 0, device, model, criterion, do_class_metrics=False)
    print(metrics)
    print("------TEST SET------")
    metrics = Classifier.evaluate(test_dataloader, "test", 0, device, model, criterion, do_class_metrics=True)
    print(metrics)

if __name__ == "__main__":
    main()