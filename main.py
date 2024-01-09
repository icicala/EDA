import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to save text output to a file
def save_text_to_file(text, filename):
    with open(filename, 'a') as file:
        file.write(text)

# Function to save plots as PNG files
def save_plot_as_png(plot_function, plot_name):
    plt.figure(figsize=(10, 6))
    plot_function()
    plt.savefig(f'{plot_name}.png')
    plt.close()
# Identifying missing values in tabular data
def missing_values():
    dataframe = load_data()
    text_output = str(dataframe.isnull().sum())
    save_text_to_file(text_output, 'result.txt')


# 1.1 Basic information of data set - data distribution and find outliers
def check_data():
    dataframe = load_data()
    pd.set_option('display.max_columns', None)
    text_output = '#################################\n'
    text_output += str(dataframe) + '\n'
    text_output += '#################################\n'
    text_output += str(dataframe.describe()) + '\n'
    text_output += '#################################\n'
    text_output += str(dataframe.head()) + '\n'
    text_output += '#################################\n'
    text_output += str(dataframe.info()) + '\n'
    save_text_to_file(text_output, 'result.txt')
    save_plot_as_png(plot_function=lambda: sns.pairplot(dataframe), plot_name='check_data')


# Sequential feature selection - recursive backward elimination
# TO DO

# Univariate Analysis -label
def label_analysis():
    dataframe = load_data()
    pd.set_option('display.max_columns', None)
    text_output = str(dataframe['class'].describe()) + '\n'
    save_text_to_file(text_output, 'result.txt')

    # Plot function for label analysis
    save_plot_as_png(plot_function=lambda: sns.barplot(
        x=dataframe['class'].value_counts(normalize=True).index,
        y=dataframe['class'].value_counts(normalize=True) * 100,
        color='black', alpha=0.8
    ), plot_name='label_analysis')


# Univariate Analysis - continuous
def continuous_analysis():
    dataframe = load_data()
    text_output = str(dataframe['purchase_value'].describe()) + '\n'
    text_output += str(dataframe['age'].describe()) + '\n'
    save_text_to_file(text_output, 'result.txt')

    # Plot function for continuous analysis
    save_plot_as_png(plot_function=lambda: dataframe[['purchase_value', 'age']].hist(
        figsize=(10, 10), bins=100, xlabelsize=18, ylabelsize=18
    ), plot_name='continuous_analysis')


# Univariate Analysis - category
def category_analysis():
    dataframe = load_data()
    category_columns = dataframe.select_dtypes(include=['object'])
    category_columns = category_columns[category_columns.columns.tolist()[3:]]
    text_output = ''
    for column in category_columns.columns:
        def plot_function():
            dataframe[column].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')

        plot_name = f'{column}_category_analysis'
        save_plot_as_png(plot_function, plot_name)

        text_output += f'{column}:\n{str(dataframe[column].value_counts())}\n'

    save_text_to_file(text_output, 'result.txt')


# category devide_id
def category_device_id_analysis():
    dataframe = load_data()
    device_id = dataframe['device_id'].value_counts()
    count_summary = device_id.value_counts()

    def plot_function():
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=count_summary.index, y=count_summary.values, color='skyblue')
        plt.title('Histogram of Device_ID Counts')
        plt.xlabel('Count of Device_ID')
        plt.ylabel('Frequency')
        for i, count in enumerate(count_summary.values):
            ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

    plot_name = 'category_device_id_analysis'
    save_plot_as_png(plot_function, plot_name)
def country_from_IP(ip, ip_dataframe):
    try:
        ip_int = int(ip)
        country_condition = (ip_int <= ip_dataframe['upper_bound_ip_address']) & (
                ip_int >= ip_dataframe['lower_bound_ip_address'])
        matching_index = np.argmax(country_condition.to_numpy())
        if matching_index:
            country = ip_dataframe.loc[matching_index, 'country']
            return country
        else:
            return "Unknown"
    except ValueError:
        return "Invalid IP"

# need to fix the errors
def ip_address_categorical_analysis():
    ip_path = os.path.join(os.getcwd(), "IpAddress_to_Country.csv")
    ip_dataframe = pd.read_csv(ip_path, delimiter=';')
    ip_dataframe = ip_dataframe.sort_values('lower_bound_ip_address')
    dataframe = load_data()
    dataframe['country'] = dataframe['ip_address'].apply(lambda i: country_from_IP(i, ip_dataframe))
    country_counts = dataframe['country'].value_counts()

    def plot_function():
        plt.figure(figsize=(10, 6))
        country_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Users by Country')
        plt.xlabel('Country')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    plot_name = 'ip_address_categorical_analysis'
    save_plot_as_png(plot_function, plot_name)




# Analyze the relationship/correlation between variables and labels
def corr_analysis():
    dataframe = load_data()
    correlation_matrix = dataframe.corr()
    text_output = str(correlation_matrix) + '\n'
    save_text_to_file(text_output, 'result.txt')


def load_data():
    data_path = os.path.join(os.getcwd(), "Fraud_Data.csv")
    dataframe = pd.read_csv(data_path, delimiter=',')
    return dataframe


if __name__ == '__main__':
    missing_values()
    check_data()
    label_analysis()
    continuous_analysis()
    category_analysis()
    category_device_id_analysis()
    ip_address_categorical_analysis()