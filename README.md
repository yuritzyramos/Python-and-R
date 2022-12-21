# Python-and-R
This portfolio contains final projects and assignments from two of my economics courses: Probability and Statistics for Economists (ECON 220) and Econometrics (ECON 320). 

In the case of assignments, students were expected to recreate an html provided by the instructor and experiment with packages/functions in a particular programming  language. 

For final projects, students were allowed to work independently or in groups for our ECON 220 course; however, students were expected to collaborate on the final project for ECON 320. 

As a result, these projects reflect my ability to work independently and in a group\team.


## R Programming Language 

### Assignments
These files, imported as .Rmd, contain the code I used to recreate the html file we were assigned for a given week. 

Topics included:
* Modifying data with dplyr
* Using ggplot
* Grouping 
* Factoring 
* Creating tables 
* Creating graphs
* Performing T-tests
* Etc.

### Final Project
For the final project, students were expected to use the data collected from a class survey (taken at the beginning of the semester) to test our preliminary hypotheses and draw conclusions about the population.
This required that I perform the following: 
* Data cleaning 
* Data filtering
* Variable Renaming  
* Factoring to create levels
* Applying T-tests
* Creating tables
* Creating graphs
* Etc. 

## Python Programming Language 

### Assignments
These files, imported as .ipynb, contain the code I used to recreate the html file we were assigned for a given week.

Topics included:
* Simple regression 
* Multiple regression
* Covariance 
* Multicollinearity
* Inference
* Qualitative variables
* Covariance  
* Creating crosstab tables, pivot tables, and other summary statistics 
* Data visualization 
* Etc.

### Final Project 
For the final project, I was in charge of creating data visualizations (scatterplots, heat maps, etc.) to present our summary statistics, performing F-Statistics (joint significance, overall significance, etc.), testing qualitative variables, and testing variable interactions. I presented the hypotheses and results from these tasks using Stargazer as well as markdown on Jupyter notebook. 

I also assisted in finding the data set used for the assignment and I set up the DropBox folder that my teammates and I used to collaborate on the same Jupyter notebook. 

### Machine Learning 

#### Final Project 
For this project, my team and I decided to design and code a program that could predict the success and revenue of video games. The program was trained using a dataset containing the attributes (developer, publisher, genere, etc.) of different video games from 2016. Our program implements KNN, decision tree, random forest, and navie bayes classifers to predict success based on whether a game is classified as Triple-A or Indie. The quantitative value of the user inputted video game is predicted using linear regression, linear regression with stochastic gradient descent, and XGBoost. 
We concluded that the data used to train the classification models was sufficient enough to predict whether a game was indie or triple-A with high accuracy. However, correctly predicting the success of a video game seems to require more extensive research in terms of the “best” data to use. Our regression model also demonstrated mixed results with a decent amount of predictions being within close range of the estimated revenue values. However, the model still faced the problem of calculating negative revenues, a result that may be attributable to the data used. 
