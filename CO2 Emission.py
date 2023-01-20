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
plt.title('CO2 Emission in 2019', fontdict={'fontsize': 17, 'weight': 'bold'})
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
plt.xlabel("Year")
plt.ylabel("CO2 Emission(kt)")
# Add a legend
plt.legend()
plt.title("Visualisation ")
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
plt.title('Stacked Bar Graph')
plt.legend()

# Display the graph
plt.show()

