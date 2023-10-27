import re
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

#open text file in read mode
text_file = open("../logs/1000_log.txt", "r")
 

# Your input string
#read whole file to a string
input_string = text_file.read()

# Extract metrics
extracted_data = extract_metrics(input_string)
print(extracted_data)

# Create the scatter plot
plt.plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
plt.plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')

plt.title('Accuracy vs Compression - 200 epochs training')
plt.xlabel('Number')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()