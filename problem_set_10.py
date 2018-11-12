#worked with Marissa Kelley, Taylor Lawrence, and Hannah Weber
#Problem set 9 - Jacob Paul

import csv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import pandas as pd


class AnalysisData:
    def __init__ (self):
        self.dataset = pd.DataFrame()
        self.xs = []


    def openCSV(self, filename):
        self.dataset = pd.read_csv(filename)
        self.xs = [val for val in self.dataset.columns.values if val != "competitorname"]

#Problem 2
class LinearAnalysis:
    def __init__ (self, _targetY):
        self.targetY = _targetY
        self.bestX = ""
    #Problem 3
    def runSimpleAnalysis(self, data):
        best_r2 = -1
        best_var = ""
        for column in data.xs:
            if column != self.targetY:
                #Set up indenpendent variable
                independent_var = data.dataset[column].values
                independent_var = independent_var.reshape(len(independent_var),1)
                # Do regression
                regr = LinearRegression()
                regr.fit(independent_var, data.dataset[self.targetY])
                pred = regr.predict(independent_var)
                r_score = r2_score(data.dataset[self.targetY],pred)
                #If current r_score is better than our previous best, then the current r_score is new best
                if r_score > best_r2:
                    best_r2 = r_score
                    best_var = column
        self.bestX = best_var
        print ("Linear Regression Equation: y = " + str(regr.intercept_) + " + " + str(regr.coef_[0]) + " * x" )
        print("Best variable: " + best_var, best_r2)


#Problem 2
class LogisticAnalysis:
    def __init__ (self, _targetY):
        self.targetY = _targetY
        self.bestX = ""
    def runSimpleAnalysis(self, data):
        best_r2 = -1
        best_var = ""
        for column in data.xs:
            if column != self.targetY:
                #Set up indenpendent variable
                independent_var = data.dataset[column].values
                independent_var = independent_var.reshape(len(independent_var),1)
                # Do regression
                regr = LogisticRegression()
                regr.fit(independent_var, data.dataset[self.targetY])
                pred = regr.predict(independent_var)
                r_score = r2_score(data.dataset[self.targetY],pred)
                #If current r_score is better than our previous best, then the current r_score is new best
                if r_score > best_r2:
                    best_r2 = r_score
                    best_var = column
        print ("Logistic Regression Equation: y = " + str(regr.intercept_[0]) + " + " + str(regr.coef_[0][0]) + " * x" )
        self.bestX = best_var
        print("Best variable: " + best_var, best_r2)

    def runMultipleRegression(self, data):
        regr = LogisticRegression()
        cols = [val for val in data.xs if val != self.targetY]
        regr.fit(data.dataset[cols], data.dataset[self.targetY])
        pred = regr.predict(data.dataset[cols])
        r_score = r2_score(data.dataset[self.targetY],pred)
        equation = []
        for i, coef in enumerate(regr.coef_[0]):
            if i != len(regr.coef_[0])-1:
                equation.append (str("%.3f" % coef) + "x + ")
            else:
                equation.append (str("%.3f" % coef) + "x")
        print("Multiple Regression Equation: " + str("%.3f" % regr.intercept_[0]) + " " + ''.join(equation))
        print(r_score)



#Problem 1
analysis_data = AnalysisData()
analysis_data.openCSV('candy-data.csv')

#Problem3
print("Linear Regression: ")
line_analysis = LinearAnalysis('sugarpercent')
line_analysis.runSimpleAnalysis(analysis_data)

#Problem set 10 -> Problem 1
print("Logistic Regression: ")
log_analysis = LogisticAnalysis('chocolate')
log_analysis.runSimpleAnalysis(analysis_data)

#Problem set 10 -> problem 2
print("Multiple Regression:")
log_analysis.runMultipleRegression(analysis_data)

# The multiple logistic regression worked much better than the simple logistic regression.
# The R2 value for the simple logistic regression was 0.425 and the R2 value for multiple logistic
# regression was 0.760.




#Problem set 10 -> Part 4
#(a)
#independent variable: caramel/chocolate - categorical
#dependent variable: sugar amount - continuous
#null hypothesis: caramel and chocolate candy have the same amount of sugar

#(b)
#independent variable: blue/red state - categorical
#dependent variable: number of split ticket voters - continuous
#null hypothesis: blue/red states have the same amount of split ticket voters

#(c)
#independent variable: battery life - continuous
#dependent variable: selling rate - continuous
#null hypothesis: All phones sell at the same rate no matter the battery life
