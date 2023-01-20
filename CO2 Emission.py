# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:14:01 2023

@author: Shilpa
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

# Read CO2 Emission Dataset from the drive
df = pd.read_csv(
    "C:/Users/rejot/OneDrive - University of Hertfordshire/ADS1 assignment3/CO2 Emissionn.csv")
# Get the data in the year of 2005
data_2005 = df[['Country Name', '2005']]

# Get the Data in the year of 2019
data_2019 = df[['Country Name', '2019']]

# Set the figure size
plt.figure(figsize=(15, 8))

# Plot the Pie Chart for Displaying CO2 Emission in 2005
plt.subplot(1, 2, 1)
plt.pie(data_2005['2005'], labels=data_2005['Country Name'], autopct='%1.1f%%')
plt.title('CO2 Emission in 2005', fontdict={'fontsize': 17, 'weight': 'bold'})
# Plot the Pie Chart for Displaying CO2 Emission in 2019
plt.subplot(1, 2, 2)
plt.pie(data_2019['2019'], labels=data_2019['Country Name'], autopct='%1.1f%%')
# Setting the Title
plt.title('CO2 Emission in 2019', fontdict={'fontsize': 17, 'weight': 'bold'})
# Saving Figure
plt.savefig("pie.png", bbox_inches="tight")
plt.show()
# create a list of countries
countries = df['Country Name'].unique()

# loop over countries
for country in countries:
    # Create a dataframe for the country
    df_country = df[df['Country Name'] == country]

    # Create an array of years
    years = df_country.columns[3:]

    # Create an array of values
    vals = df_country[years].values[0]

    # Plot the line plot
    plt.plot(years, vals, label=country)

# Set the x-limits
plt.xlim('2012', '2019')
# Setting the Labels
plt.xlabel("Year")
plt.ylabel("CO2 Emission(kt)")
# Add a legend
plt.legend()
# Saving the Figure
plt.savefig("line.png", bbox_inches="tight")
# Setting Title
plt.title("Visualisation:CO2 Emission based on Years")
# Show the plot
plt.show()
# Creating the array for 4 countries
country = np.array(['Canada', 'Mexico', 'France', 'Italy'])

# Creating the stacked bar graph
x_pos = np.arange(len(country))

# Adding the 1990 data to obtain bar graph
year_1990 = [1000, 2000, 3000, 4000]
plt.bar(x_pos, year_1990, color='green', label='1990')

# Adding the 2003 data to obtain bar graph
year_2003 = [2000, 4000, 6000, 8000]
plt.bar(x_pos, year_2003, bottom=year_1990, color='red', label='2003')

# Setting the labels
plt.xticks(x_pos, country)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions(kt)')
# Setting Title
plt.title('Stacked Bar Graph')
plt.legend()
# Saving the Figure
plt.savefig("bar.png", bbox_inches="tight")

# Display the graph
plt.show()
# Select columns to use for Clustering
X = df[['2013', '2014', '2015', '2016', '2017', '2018', '2019']]

# Create K-Means Clustering model
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.labels_

# Visualize data using scattering method
sns.pairplot(df, hue='Country Name', palette='husl', diag_kind='hist')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
# Setting the labels
plt.xlabel('Yearwise CO2 Emission(kt)')
plt.ylabel('Yearwise CO2 Emission(kt)')
# Setting The Title
plt.title('CO2 Clustering')
plt.show()
