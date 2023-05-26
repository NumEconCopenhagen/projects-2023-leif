This zip-file contains:

# 1. Inaugural project. 
The results of the project can be seen from running [inaugrualproject.ipynb](inauguralproject.ipynb).

Dependencies: Apart from a standard Anaconda Python 3 installation, the project requires no further packages.

The goal of this project is to exam how the household maximizes their joint utility when choosing working hours and consumption.

Through exercise 1 and 2 we solve the model in discrete time.

In the first exercise we illustrate the relative difference in hours spent at home seen from the woman's perspective over varying values of alpha and sigma.

In the second exercise we plot the log relative difference of hours spent at home against the log relative wage.

In the third exercise we do exactly the same as in exercise 2 but now in continuous time.

In the fourth exercise we run the regression with values from exercise 3 and afterwards minimize the squared differences of our regressors and the given values in the text. We minimize these to determine the alpha and sigma value that minimize the squared difference. Given our regression model the fit is not that good to the model in the paper. Like the paper we get a positive constant term and a negative slope but the values varies a lot. We are unsure of the economic interpretation as we havent been able to minimize the squared difference correct.

In the last exercise we include the variable kappa to account for the disutility men get from working at home. We then estimate the results from exercise 4 again but this time optimize over kappa and sigma instead of alpha and sigma. This new model fit the data somewhat compared to the paper related to in exercise 4.

# 2. Data project. 

Our project is titled **Consumer spending based on social status** and aims to look at how consumption is different based on the different social statuses in Denmark.

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We have gotten our data from **Danmarks Statistikbank**, and scraped datea using DST API. We have used the following Data:

1. **FU04** - Consumption based on consumptiongroups, socioeconomic status and price units
2. **PRIS112** - Consumer price index

# 3. Model project.

Our project is modelled af the **Ramsey Model** and is about what effects $\alpha$ and $\beta$ have on the long run steady state.

The **results** of the project can be seen from running [modelprojectFIN.ipynb](ModelprojectFIN.ipynb).

In our project we have set up 2 countries, and run the Ramseymodel on both of them. We assigned them  $\alpha$ and $\beta$ values, in order to test for the long-run steady state for different values. 

# 4. Exam project

In our exam we answer the following questions:

1: Optimal taxation with governmentconsumption
2: Labor adjustment costs
3: Global optimizer with refined multi-start

The **results** of the project can be seen from running [Exam-2023-notebook.ipynb](Exam-2023-notebook.ipynb).







