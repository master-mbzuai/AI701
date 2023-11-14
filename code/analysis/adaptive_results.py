import os
import re
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

exp = 0
alpha = 0.9

# read folder 

def read_architecture(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    section = None
    result = {}

    for line in lines:
        if 'BACKBONE' in line:
            section = 'BACKBONE'
        elif 'CLASSIFIER' in line:
            section = 'CLASSIFIER'

        if section:
            macs_match = re.search(r'MACs (\d+\.\d+)', line)
            params_match = re.search(r'learnable parameters (\d+\.\d+ k|\d+)', line)

            if macs_match:
                result[section + '_MACs'] = macs_match.group(1)
            if params_match:
                result[section + '_Params'] = params_match.group(1)

    return result
    

def read_results(file_path):
    with open(file_path, 'r') as f:
        print("ciaooo")
        line = f.readlines()[-1]
        #accuracy_match = r'\d+\.\d+'
        #Epoch 99: train_accuracy: 0.62 - train_loss: 2.19 - lr: 0.00; val_accuracy: 0.7174 - val_loss: 1.1699.

        match = r'Epoch (\d+): train_accuracy: (\d+\.\d+) - train_loss: (\d+\.\d+) - lr: 0.00; val_accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+).'

        matches = re.findall(match, line)        
        return matches[0][3:]
    

if __name__ == "__main__":

    results = {}
    path = "../results/adaptive_01/"

    for folder in os.listdir(path):      
        if folder in ["0", "10","25","50","75","90"]:
            print(folder)
            #folder = [x for x in folder if "." not in x]  
            results[folder] = {}
            for exp in os.listdir(path + folder):                        
                if("architecture.txt" == exp):                
                    meta = read_architecture(path + folder + "/" + exp)                
                    results[folder]["mac_classifier"] = meta["CLASSIFIER_MACs"]                
                    quantity = meta["CLASSIFIER_MACs"]                

                    quantity2 = meta["BACKBONE_MACs"]                

                    results[folder]["mac_all"] = str(float(quantity) + float(quantity2))

                elif("train_log.txt" == exp):
                    accuracy, loss = read_results(path + folder + "/" + exp)
                    results[folder]["accuracy"] = accuracy
                    results[folder]["loss"] = loss

    data = {k: v for k, v in sorted(results.items(), key=lambda item: item[0], reverse=False)}    

    print(data)

    # Extract numbers, accuracies, and parameters from the data
    numbers = [int(key) for key in data.keys()]
    accuracies = [float(data[key]['accuracy']) for key in data.keys()]
    params = [float(data[key]['mac_classifier'].split()[0])*0.01 for key in data.keys()]  # Assuming KMac, so multiplied by 1000

    # Choose the colormap
    colormap = plt.cm.viridis

    # Normalize y-values between 0 and 1
    norm = mcolors.Normalize(vmin=min(params), vmax=max(params))

    dot_colors = [colormap(norm(value)) for value in params]


    # Sort by numbers for better plotting
    sorted_indices = sorted(range(len(numbers)), key=lambda k: numbers[k])
    numbers = [numbers[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    params = [params[i] for i in sorted_indices]

    # Create the scatter plot
    plt.scatter(numbers, accuracies, s=params, alpha=0.5, c=dot_colors, label='Parameters')

    # Labeling each point with the corresponding number
    for i, txt in enumerate(params):
        plt.annotate(txt, (numbers[i], accuracies[i]))

    if(alpha > 10):
        alpha_title = str(alpha/10)
    else:
        alpha_title = str(alpha)

    plt.axhline(accuracies[0], color='orange', linestyle='--')
    plt.text(40, accuracies[0]-0.007, "Baseline", color='orange')
    plt.text(2.5, 4.1, 'Horizontal line at y=4', color='blue')
    plt.title('Accuracy vs Compression - alpha ' + alpha_title + ' - 100 epochs')
    plt.xlabel('Number')
    plt.ylabel('Accuracy')
    plt.colorbar(label='Parameters (KMac)')
    plt.grid(True)
    plt.savefig(path + "image.jpg")
    plt.show()    
