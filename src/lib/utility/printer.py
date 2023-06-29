import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_scatter(writer, y_true, y_pred, label="Target"):
    """Generate a scatter plot of the predicted vs actual values.

    Args:
        writer (pandas.io.excel._xlsxwriter._XlsxWriter): The ExcelWriter object to which the plot will be added.
        y_test (array-like): The true target values.
        y_pred (array-like): The predicted target values.
        label (str): The label of the target variable.

    Returns:
        pandas.io.excel._xlsxwriter._XlsxWriter: The modified ExcelWriter object.
    """

    # create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title("Scatter plot of true vs predicted values")

    # add plot to Excel file
    sheet_name = "Scatter Plot"
    workbook = writer.book
    worksheet = writer.sheets[sheet_name] if sheet_name in writer.sheets else workbook.add_worksheet(sheet_name)
    writer = pd.ExcelWriter(writer.path, engine="xlsxwriter")
    writer.book = workbook
    writer.sheets = dict((ws.title, ws) for ws in workbook.worksheets())

    # save plot to Excel file
    plt.savefig(writer, format="png")
    plt.close()

    # insert plot into Excel worksheet
    worksheet.insert_image(0, 0, "", {"image_data": writer.sheets[sheet_name].images[0]["data"]})

    return writer


def save_matrix(conf_mat, writer, labels=None):
    """Saves a confusion matrix and its metrics to an Excel file.

    This function takes a confusion matrix and an Excel writer object as
    input and saves the confusion matrix and its metrics (accuracy,
    specificity, and sensitivity) to an Excel file. The confusion matrix
    is saved to a sheet named "Confusion Matrix" and the metrics are saved
    to a sheet named "Metrics".

    Args:
        conf_mat: A 2x2 numpy array representing the confusion matrix.
        writer: An Excel writer object used to save the data to an Excel file.
        labels: An optional list of two strings representing the labels for
            the two classes. If not provided, the labels default to ["0", "1"].

    Returns:
        The writer object passed as input.
    """
    tn, fp, fn, tp = conf_mat.ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    if labels == None:
        labels = ["0", "1"]

    columns = ["Predicted " + labels[0], "Predicted " + labels[1]]
    index = ["Actual " + labels[0], "Actual " + labels[1]]

    metrics_df = pd.DataFrame(
        {"Metric": ["Accuracy", "Specificity", "Sensitivity"], "Value": [accuracy, specificity, sensitivity]}
    )
    conf_mat_df = pd.DataFrame(conf_mat, columns=columns, index=index)

    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    conf_mat_df.to_excel(writer, sheet_name="Confusion Matrix")

    return writer


def plot_feature_importance(importance, names, writer, title=None, outdir=None, first_features=None):
    """Plots and saves the feature importance of a model.

    This function takes the feature importance and feature names of a model
    as input and plots a bar chart showing the importance of each feature.
    The chart is saved to a specified output directory. The function also
    saves the feature importance data to an Excel file using a provided
    Excel writer object.

    Args:
        importance: A list of feature importance values.
        names: A list of feature names.
        writer: An Excel writer object used to save the data to an Excel file.
        title: An optional string representing the title of the chart. If not
            provided, the title defaults to "FeaturesImportance".
        outdir: An optional string representing the output directory where the
            chart will be saved. If not provided, the chart is not saved.
        first_features: An optional integer representing the number of top
            features to plot. If not provided, all features are plotted.

    Returns:
        The writer object passed as input.
    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    df_fi = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    df_fi.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    df_fi.to_excel(writer, sheet_name="Features Importance", index=False)

    if outdir != None:
        # Create the outdir
        os.makedirs(outdir, exist_ok=True)

        # Clear figure
        plt.clf()

        if title != None:
            title = title + "_FeaturesImportance"
        else:
            title = "FeaturesImportance"


        # Select first feature if required
        if first_features != None:
            df_fi = df_fi.iloc[:first_features, :]

        # Define size of bar plot
        plt.figure(figsize=(20, 18))

        # Plot Searborn bar chart
        sns.set(font_scale=2)  # for label size
        barplot = sns.barplot(x=df_fi["feature_importance"], y=df_fi["feature_names"])

        # Add chart labels
        plt.title(title)
        plt.xlabel("FEATURE IMPORTANCE")
        plt.ylabel("FEATURE NAMES")

        # Save figure
        figure = barplot.get_figure()
        figure.savefig(os.path.join(outdir, title+".png"), dpi=400)

    return writer


def plot_confusion_matrix(writer, title=None, outdir=None, **kwargs):
    """Plots and saves a confusion matrix to an Excel file.

    This function takes an Excel writer object and optional title and output
    directory as input and calculates a confusion matrix using the provided
    keyword arguments. The confusion matrix is plotted as a heatmap and saved
    to the specified output directory. The function also saves the confusion
    matrix data to an Excel file using the provided writer object.

    Args:
        writer: An Excel writer object used to save the data to an Excel file.
        title: An optional string representing the title of the chart. If not
            provided, the title defaults to "ConfusionMatrix".
        outdir: An optional string representing the output directory where the
            chart will be saved. If not provided, the chart is not saved.
        **kwargs: Additional keyword arguments passed to the confusion_matrix
            function.

    Returns:
        The writer object passed as input.
    """
    matrix = confusion_matrix(**kwargs)

    writer = save_matrix(matrix, writer, kwargs.get("labels"))

    if outdir != None:
        # Create the outdir
        os.makedirs(outdir, exist_ok=True)
        
        # Clear figure
        plt.clf()

        if title != None:
            title = title + "_ConfusionMatrix"
        else:
            title = "ConfusionMatrix"


        df_cm = pd.DataFrame(matrix, range(matrix.shape[0]), range(matrix.shape[0]))
        sns.set(font_scale=1.4)  # for label size
        heatmap = sns.heatmap(df_cm, annot=True, fmt="g", annot_kws={"size": 16})  # font size

        # Add chart labels
        heatmap.set_title(title)
        heatmap.set_xlabel("Predicted Values")
        heatmap.set_ylabel("Actual Values ")

        ## Ticket labels - List must be in alphabetical order
        heatmap.xaxis.set_ticklabels(["Absence", "Presence"])
        heatmap.yaxis.set_ticklabels(["Absence", "Presence"])

        # Save figure
        figure = heatmap.get_figure()
        figure.savefig(os.path.join(outdir, title + ".png"), dpi=400)

    return writer


def plot_classification_report(writer, title=None, outdir=None, **kwargs):
    """Plots and saves a classification report to an Excel file.

    This function takes an Excel writer object and optional title and output
    directory as input and calculates a classification report using the provided
    keyword arguments. The classification report is plotted as a heatmap and saved
    to the specified output directory. The function also saves the classification
    report data to an Excel file using the provided writer object.

    Args:
        writer: An Excel writer object used to save the data to an Excel file.
        title: An optional string representing the title of the chart. If not
            provided, the title defaults to "ClassificationReport".
        outdir: An optional string representing the output directory where the
            chart will be saved. If not provided, the chart is not saved.
        **kwargs: Additional keyword arguments passed to the classification_report
            function.

    Returns:
        The writer object passed as input.
    """
    n_class = kwargs.pop("n_class", 2)
    report = classification_report(**kwargs)

    # .iloc[:-1, :n_class] to exclude support and plot only the classes
    df_cr = pd.DataFrame(report).iloc[:-1, :n_class].T
    df_cr.to_excel(writer, sheet_name="Classification Report")

    if outdir != None:
        # Create the outdir
        os.makedirs(outdir, exist_ok=True)

        # Clear figure
        plt.clf()

        if title != None:
            title = title + "_ClassificationReport"
        else:
            title = "ClassificationReport"


        sns.set(font_scale=1.4)  # for label size
        heatmap = sns.heatmap(df_cr, annot=True, annot_kws={"size": 16})  # font size

        # Add chart labels
        heatmap.set_title(title)
        heatmap.set_xlabel("Measures")
        heatmap.set_ylabel("Classes")

        # Save figure
        figure = heatmap.get_figure()
        figure.savefig(os.path.join(outdir, title + ".png"), dpi=400)
    
    return writer


def plot_boxplot(data, x, y, hue, outdir, title=None):
    """Plots the data as a boxplot and saves as a png image.

    Args:
        data: data frame of the data to plot
        x: feature to use as the x-axis
        y: feature to use as the y-axis
        hue: feature to use as the hue (optional)
        model_type: name of the model
        outdir: output directory to save the png image

    Returns:
        None
    """
    # Clear figure
    plt.clf()

    # # .iloc[:-1, :n_class] to exclude support and plot only the classes
    # df_cr = pd.DataFrame(report).iloc[:-1, :n_class].T
    # sns.set(font_scale=1.4)  # for label size
    boxplot = sns.boxplot(x=x, y=y, hue=hue, data=data)
    # Add chart labels

    if title:
        boxplot.set_title(title)
    # boxplot.set_xlabel("Measures")
    # boxplot.set_ylabel("Classes")

    # Save figure
    figure = boxplot.get_figure()
    os.makedirs(outdir, exist_ok=True)
    figure.savefig(os.path.join(outdir, (title + "_boxplot.png")), dpi=400)
    # df_cr.to_excel(os.path.join(outdir, "classification_report.xlsx"))