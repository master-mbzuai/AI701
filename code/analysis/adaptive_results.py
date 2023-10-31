import os
import re
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

exp = 0
alpha = 2

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
        line = f.readlines()[0]
        accuracy_match = r'\d+\.\d+'

        matches = re.findall(accuracy_match, line)
        return matches    
    

if __name__ == "__main__":

    results = {}
    path = "../results/adaptive_exp/a" + str(alpha) + "/" + str(exp) + "/"

    for folder in os.listdir(path):        
        results[folder] = {}
        for exp in os.listdir(path + folder):                     
            if("architecture.txt" == exp):
                meta = read_architecture(path + folder + "/" + exp)                
                results[folder]["mac_classifier"] = meta["CLASSIFIER_MACs"]                
                quantity = meta["CLASSIFIER_MACs"]                

                quantity2 = meta["BACKBONE_MACs"]                

                results[folder]["mac_all"] = str(float(quantity) + float(quantity2))
            elif("test_set_result.txt" == exp):
                accuracy, loss = read_results(path + folder + "/" + exp)
                results[folder]["accuracy"] = accuracy
                results[folder]["loss"] = loss

    data = {k: v for k, v in sorted(results.items(), key=lambda item: item[0], reverse=False)}    

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
    plt.scatter(numbers, accuracies, s=50, alpha=0.5, c=dot_colors, label='Parameters')

    # Labeling each point with the corresponding number
    for i, txt in enumerate(numbers):
        plt.annotate(txt, (numbers[i], accuracies[i]))

    if(alpha > 10):
        alpha_title = str(alpha/10)
    else:
        alpha_title = str(alpha)

    plt.title('Accuracy vs Compression - alpha ' + alpha_title + ' - 20 epochs')
    plt.xlabel('Number')
    plt.ylabel('Accuracy')
    plt.colorbar(label='Parameters (KMac)')
    plt.grid(True)
    plt.savefig("./results/adaptive_vanilla_a" + str(alpha))
    plt.show()    