import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


def TP(cm, clas):
    '''
    Function to return True Positive Values from Confusion matrix for specific class
    :param cm: Confusion Matrix
    :param clas: Class ID
    :return: TP Number
    '''
    return cm[clas][clas]

def FP(cm, clas):
    '''
    Function to return False Positive Values from Confusion matrix for specific class
    :param cm: Confusion Matrix
    :param clas: Class ID
    :return: FP Number
    '''
    t = 0
    for i in range(len(cm)):
        if i != clas:
            t += cm[i][clas]
    return t

def FN(cm, clas):
    '''
    Function to return False Negative Values from Confusion matrix for specific class
    :param cm: Confusion Matrix
    :param clas: Class ID
    :return: FN Number
    '''
    t = 0
    for i in range(len(cm)):
        if i!=clas:
            t+= cm[clas][i]
    return t

def TN(cm, clas):
    '''
    Function to return True Negative Values from Confusion matrix for specific class
    :param cm: Confusion Matrix
    :param clas: Class ID
    :return: TN Number
    '''
    t = 0
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i!= clas and j != clas:
                t+=cm[i][j]
    return t

def generate_scores(cm):
    '''
    Function to generate sensitivity, specificity, precision, recall and f1_score for all classes
    :param cm: Confusion Matrix
    :return: Pandas Dataframe with all Values
    '''
    classes = len(cm)
    results = []
    for i in range(classes):
        tp, fp, tn, fn = TP(cm, i), FP(cm, i), TN(cm, i), FN(cm, i)
        sensitivity = tp/(tp+fp)
        specificity = tn/(tn+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*(precision*recall)/(precision+recall)
        results.append([i, sensitivity, specificity, precision, recall, f1_score])
    arr = np.array(results)
    return pd.DataFrame(arr, columns=["Class", "Sensitivity", "Specificity", "Precision", "Recall", "F1-Score"] )


def generate_plot():
    '''
    Function to generate a plot for accuracy vs Number of features used. You need to generate accuracies before doing this.
    :return:
    '''
    accuracies = {
        "Bone": pd.DataFrame(pd.read_csv( r"D://datasets/Osteosarcoma-UT/all_tiles_all_features_256x256/fi_accuracies.csv")).iloc[:, 1:].values.astype(np.int),
        "Prostate": pd.DataFrame(
            pd.read_csv(r"D://datasets/SICAPv2/all_tiles_all_features_256x256/fi_accuracies.csv")).iloc[:, 1:].values.astype(np.int),
        "Breast": pd.DataFrame(
            pd.read_csv(r"D://datasets/ICIAR/all_tiles_all_features_256x256/fi_accuracies.csv")).iloc[:, 1:].values.astype(np.int),
        "Cervical": pd.DataFrame(
            pd.read_csv(r"D://datasets/Cervical/all_tiles_all_features_256x256/fi_accuracies.csv")).iloc[:, 1:].values.astype(np.int),
        "Colon": pd.DataFrame(
            pd.read_csv(r"D://datasets/CRC100K/all_tiles_all_features_224x224/fi_accuracies.csv")).iloc[:, 1:].values.astype(np.int),
    }
    for key, acc in accuracies.items():
        x = acc[:, 0]
        y = acc[:, 1]
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), x.max()-x.min())
        Y_ = X_Y_Spline(X_)
        plt.plot(X_, Y_, label=f"{key}")
    plt.vlines(x=500, ymin=50, ymax=100, linestyles="dotted", label="500 Features")
    plt.xlabel("Number of Features", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend(prop={'size': 18})
    plt.tick_params(labelsize=18)
    plt.tight_layout()

    plt.show()
    print(accuracies)


# Constants for Plot Betti Curve
CURVE_RGB=1
CURVE_HSV=2
def plot_curve(FILE_PATH, class_names, confidence=40, curve=CURVE_RGB):
    '''
    Function to ploy betti curves
    :param FILE_PATH: File path to 2800 features csv file
    :param class_names: Classes names as list.
    :param confidence: Confidance band value
    :param curve: Curve RGB or HSV, select from constant
    :return: None
    '''
    df = pd.DataFrame(pd.read_csv(FILE_PATH, header=None))
    classes = len(class_names)

    if curve==CURVE_RGB:
        channels = ["AVG (Grey)", "Red", "Blue", "Green"]
        starts = [(1+(i*200)) for i in range(4)]
    else:
        channels = ["AVG (HSV)", "Hue", "Saturation", "Value"]
        starts = [(801+(i*200)) for i in range(4)]
    starts2 = [0, 100]
    print(starts, starts2)
    fig, axes = plt.subplots(nrows=len(starts2), ncols=len(starts), figsize=(15, 3))
    print(axes.shape)

    for Q, q in enumerate(starts2):
        for P, p in enumerate(starts):
            classwise_data = []
            for i in range(classes):
                class_data = df[df.iloc[:, -1]==i]
                df1 = class_data.iloc[:, p+q:p+q+100]
                df1.columns = [i for i in range(100)]

                class_data = df1
                n = len(class_data)
                cnf = int((confidence/200)*n)
                median = []
                lower = []
                upper = []

                for i in range(100):
                    values = sorted(class_data.iloc[:, i])
                    median.append(values[n//2])
                    lower.append(values[n//2-cnf])
                    upper.append(values[n//2+cnf])
                classwise_data.append([lower, median, upper])
            X = [i for i in range(100)]
            colors = ["green", "red", "blue", "black"]
            for i in range(classes):
                if P == 0 and Q == 0:
                    axes[Q, P].plot(X, classwise_data[i][1], color=colors[i], linewidth=3, label=class_names[i])
                else:
                    axes[Q, P].plot(X, classwise_data[i][1], color=colors[i], linewidth=3)
                axes[Q, P].fill_between(X, classwise_data[i][0], classwise_data[i][2], color=colors[i], alpha=0.1)
                axes[Q, P].title.set_text(f"RGB - Betti {Q} for channel {channels[P]} ")

    fig.legend()
    fig.tight_layout()
    plt.show()

# Example Call to the betti curve plot function
plot_curve(
    r"/Users/datasets/SICAPv2/all_tiles_all_features_256x256/train/data_2800.csv",
    class_names = [
            "NC",
            "G3",
            "G4",
            "G5",
        ],
    confidence=40,
    curve=CURVE_HSV,
)