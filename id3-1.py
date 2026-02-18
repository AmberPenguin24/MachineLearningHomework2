import pandas as pd
import math
import pdb
from anytree import Node, RenderTree
from sklearn.tree import DecisionTreeClassifier, _tree, export_text
from sklearn.preprocessing import LabelEncoder


CLASSES = 2 # min of binary classification

def get_entropy(data, c_label):
    """
    Calculate the entropy for the given data and class label.
    Steps:
    1. Calculate the total number of samples in the data.
    2. For each unique class label, calculate the probability and its contribution to entropy.
    3. Sum the  entropies for each class label to get the final entropy.
    """
    # TODO: Implement entropy calculation logic here.
    total_samples = len(data)
    entropies = []

    for each_class in c_label:
        class_samples = len(data[data[c_label] == each_class])
        probability = class_samples / total_samples
        if probability > 0:
            entropies.append(probability * math.log2(probability))

    entropy = -(sum(entropies))

    return entropy

def get_information_gain(data, f_label, c_label):
    """
    Calculate the information gain for a given feature.
    Steps:
    1. Calculate the current entropy (without splitting) using get_entropy.
    2. For each unique value in the feature, calculate the entropy for that subset of the data.
    3. Subtract the weighted entropy for each subset from the current entropy to get the information gain.
    """
    # TODO: Implement information gain calculation logic here.
    current_entropy = get_entropy(data, c_label)


    for f_value in data[f_label].unique():
        subset = data[data[f_label] == f_value] #get subset of data where feature has value f_value
        entropy = get_entropy(subset, c_label) #get entropy for subset

        entropy_weight = len(subset) / len(data) #get |Xv| / |X| for weight
        entropy *= entropy_weight 
       
        current_entropy -= entropy
        
    return current_entropy

def build_tree(data, features, c_label, T):
    """
    Recursively build a decision tree.
    Steps:
    1. If no data or features left, return None (base case).
    2. For each feature, calculate its information gain using get_information_gain.
    3. Pick the feature with the highest information gain as the splitting criterion.
    4. Recursively build the tree for each branch using the subset of data corresponding to each feature value.
    """
    # TODO: Implement tree construction logic here.
    if (len(data) == 0):
        return None 
    elif (len(features) == 0):
        return None
    elif (len(data[c_label].unique()) == 1):
        T[data[c_label].unique()[0]] = {}
        return None
    else:
        info_gains = {}
        for feature in features:
            info_gains[feature] = get_information_gain(data, feature, c_label)

        best_feature = max(info_gains, key=info_gains.get) 
        T[best_feature] = {}
        #make a new features list
        new_features = features.copy()
        new_features.remove(best_feature)
        for feature_value in data[best_feature].unique(): #for each value of the best feature, make a new branch
           subset = data[data[best_feature] == feature_value]
           T[best_feature][feature_value] = {}
           build_tree(subset, new_features, c_label, T[best_feature][feature_value])            
        


def sklearn_decision_tree(dataframe):
    """
    Use Sklearn's decision tree to fit and print the tree structure.
    Steps:
    1. Encode categorical columns using encode_categorical.
    2. Separate features and target labels.
    3. Train a DecisionTreeClassifier on the data.
    4. Print the tree structure using export_text.
    """
    # TODO: Implement sklearn decision tree fitting and structure extraction logic here.
    dataframe = encode_categorical(dataframe)
    c_label = dataframe['class']
    features = dataframe.drop(columns=['class'])
    clf = DecisionTreeClassifier.fit(features, c_label)
    print(export_text(clf, feature_names=features.columns))
    pass

def encode_categorical(df):
    """
    Encode categorical features into numerical values using LabelEncoder.
    Steps:
    1. For each column, apply LabelEncoder to convert categorical values to integers.
    2. Return the encoded dataframe and the label encoders used.
    """
    # TODO: Implement categorical encoding logic here.
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        # Store the label encoder for later use if needed (e.g., for inverse transformation)
    return df
    


def convert_to_anytree(tree, parent_name="Root"):
    """ Converts your existing tree structure into an anytree format """
    root = Node(parent_name)
    
    def helper(subtree, parent):
        if not isinstance(subtree, dict):
            return
        for key, branches in subtree.items():
            child = Node(str(key), parent=parent)
            for branch_value, sub_branch in branches.items():
                branch_node = Node(f"{key}={branch_value}", parent=child)
                helper(sub_branch, branch_node)

    helper(tree, root)
    return root

def print_anytree(tree):
    """ Prints the decision tree in a structured way using anytree """


def fetch_and_clean():
    """
    Import and clean the mushroom dataset.
    Steps:
    1. Read the dataset using pandas.
    2. Drop any rows with missing values.
    3. Return the cleaned dataframe.
    """
    # TODO: Implement data fetching and cleaning logic here.
    df = pd.read_csv('mushroom_data.csv')
    df = df.dropna()
    return df
   

if __name__ == "__main__":
    # Example use
    df = fetch_and_clean()

    c_label = 'class'
    CLASSES = len(df[c_label].unique())

    features = df.columns.values.tolist()
    features.remove(c_label)

    T = {}
    build_tree(df, features, c_label, T)

    anytree_root = convert_to_anytree(T)

    # YOUR TREE
    print_anytree(anytree_root)

    # SKLEARN TREE
    sklearn_decision_tree(dataframe=df)
