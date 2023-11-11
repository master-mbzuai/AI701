import re
import os
from matplotlib import pyplot as plt

def extract_metrics(input_string):
    pattern = r"Epoch (\d+): train_accuracy: (\d+\.\d+) - train_loss: (\d+\.\d+); val_accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+)."
    matches = re.findall(pattern, input_string)

    extracted_data = {
        "epoch": [],
        "train_accuracy": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": []
    }

    for match in matches:
        extracted_data["epoch"].append(int(match[0]))
        extracted_data["train_accuracy"].append(float(match[1]))
        extracted_data["train_loss"].append(float(match[2]))
        extracted_data["val_accuracy"].append(float(match[3]))
        extracted_data["val_loss"].append(float(match[4]))

    return extracted_data


#path = "../results/augment_01_scheduler/0/augment_01_scheduler/"
path = "../results/adaptive_50_epochs_01/"
#open text file in read mode


def create_image(path, x):
    #text_file = open("../logs/1000_log.txt", "r")
    text_file = open(path + str(x) + "/train_log.txt", "r")
    #text_file = open("../results/adaptive_exp_0/90/exp/train_log.txt", "r")

    # Your input string
    #read whole file to a string
    input_string = text_file.read()

    # Extract metrics
    extracted_data = extract_metrics(input_string)

    # Create the scatter plot
    plt.plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
    plt.plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')

    plt.title('Train loss (blue) VS Validation loss - d: ' + x)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(path + str(x) + '/trainloss.jpg')
    plt.show()


for x in os.listdir(path):
    print(x)
    if x in ["0", "10","25","50","75","90"]:
        create_image(path, x)