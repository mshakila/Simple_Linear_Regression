# Assignment4 -part3 of 4- Simple linear Regression
# Q3) Emp_data -> Build a prediction model for Churn_out_rate
emp_data <- read.csv('file:///E:/EXCELR/Assignments/Simple_Linear_Regression_Assignment/emp_data.csv')
attach(emp_data)
# Details of dataset
dim(emp_data)
# 10 obsvn 2 variables
names(emp_data)
#  varaibles are "Salary_hike"    "Churn_out_rate"
class(emp_data)
str(emp_data)
head(emp_data)

# large dataset
library(moments)
10*(skewness(Salary_hike)^2) #5.239527
10*(kurtosis(Salary_hike)) #25.48327
10*(skewness(Churn_out_rate)^2) # 2.97896
10*(kurtosis(Churn_out_rate)) # 22.68898
summary(emp_data)

# I BM
mean(Salary_hike)
median(Salary_hike)
library(modeest)
mlv(Salary_hike,method='mfv')
mean(Churn_out_rate)
median(Churn_out_rate)
mlv(Churn_out_rate,method='mfv')
# II BM
var(Salary_hike)
sd(Salary_hike)
range(Salary_hike)
max(Salary_hike)-min(Salary_hike)
var(Churn_out_rate)
sd(Churn_out_rate)
range(Churn_out_rate)
max(Churn_out_rate)-min(Churn_out_rate)
# III BM
skewness(Salary_hike)
skewness(Churn_out_rate)

# IV BM
library(moments)
kurtosis(Salary_hike)
kurtosis(Churn_out_rate)

summary(emp_data)

# VISUALIZATIONS
boxplot_salary <- boxplot(Salary_hike,horizontal=T,main='Boxplot of Salary_hike ')
boxplot_churn <- boxplot(Churn_out_rate,horizontal=T,main='Boxplot of Churn_out_rate ')

# finding outliers
boxplot_salary$out
boxplot_churn$out

library(rcompanion)
plotNormalHistogram(Salary_hike)
hist(Salary_hike)
hist(Salary_hike,probability = T)
lines(density(Salary_hike))
lines(density(Salary_hike,adjust = 2), lty='dotted')
hist(Churn_out_rate,probability = TRUE)
lines(density(Churn_out_rate))
lines(density(Churn_out_rate,adjust = 2),lty='dotted')

plot(densityplot(Salary_hike, main='Density plot of Salary_hike'))
plot(densityplot(Churn_out_rate,main='Density plot of Churn_out_rate'))
# densityplot(Salary_hike)
# densityplot(Churn_out_rate)
barplot(Salary_hike,horiz = T,main='Barplot of Salary_hike')
barplot(Churn_out_rate,horiz = T,main = 'Barplot of Churn_out_rate')

dotchart(Salary_hike,xlab = 'Salary_hike')
dotchart(Churn_out_rate,xlab = 'Churn_out_rate')


# Normality
qqnorm(Salary_hike, main='Normal qq plot of Salary_hike')
qqline(Salary_hike)
qqnorm(Churn_out_rate,main='Normal qq plot of Churn_out_rate')
qqline(Churn_out_rate)
shapiro.test(Salary_hike) # p-value = 0.5018, normal
library(nortest)
ad.test(Salary_hike) # p-value = 0.5655 , normal
shapiro.test(Churn_out_rate) # p-value = 0.7342, normal
ad.test(Churn_out_rate) # p-value = 0.8122, not normal

# outliers detection
box <- boxplot(Salary_hike)
box$out

# missing values
# finding missing values
emp_data[!complete.cases(emp_data),]
library(mice)
md.pattern(emp_data)
library(Hmisc)
describe(emp_data)

################# MODEL BUILDING ###########################
# LINEAR RELATIONSHIP
plot(Salary_hike,Churn_out_rate, main='Scatterplot of Churn_out_rate vs Salary_hike')
cor(Churn_out_rate,Salary_hike) # -0.9117216

# standard regression model
reg_simple <- lm(Churn_out_rate ~ Salary_hike)
summary(reg_simple) 
# Multiple R-squared:  0.8312,	Adjusted R-squared:  0.8101

# plotting regression line
plot(Salary_hike,Churn_out_rate, main='Regression Line - Churn_out_rate vs Salary_hike')
abline(reg_simple)

# Residuals
sum(reg_simple$residuals) # is 0
mean(reg_simple$residuals) # is 0
sqrt(mean(reg_simple$residuals^2)) # RMSE= 3.997528

# residual plot, plot residuals against indepen var 
# should show random pattern
plot(Salary_hike,reg_simple$residuals,main='Residual plot: standard regression')
abline(0,0)

# symmetrical disbn of errors across mean
qqnorm(reg_simple$residuals,main = "Normal QQ plot of Residuals")
qqline(reg_simple$residuals) # from plot not normal
shapiro.test(reg_simple$residuals) # p-value = 0.05845 > 0.05, normal
library(nortest)
ad.test(reg_simple$residuals) # p-value = 0.06657, normal

pred_simple <- predict(reg_simple)

################ log(X) = log(Salary_hike) #################

# log(x)  --- 
qqnorm(log(Salary_hike),main='Normal QQ plot using log(X)')
qqline(log(Salary_hike))
shapiro.test(log(Salary_hike)) #0.5992 # DATA normal
library(nortest)
ad.test(log(Salary_hike)) # 0.6753
attach(emp_data)

cor(Churn_out_rate,log(Salary_hike)) # -0.9212077
plot(log(Salary_hike),Churn_out_rate,main='Scatterplot using log(X): log(Salary_hike)')

# regression - log(X)
emp_data <- read.csv(file.choose())
names(emp_data)
attach(emp_data)
reg_log <- lm(Churn_out_rate ~ log(Salary_hike))
summary(reg_log)
pred_log <- predict(reg_log)
# Multiple R-squared:  0.8486,	Adjusted R-squared:  0.8297 
# REGRESSION EQUATION
# Churn_out_rate = 1381.5 - 176.1* log(Salary_hike)
plot(log(Salary_hike),Churn_out_rate,main='Regression line: using log(Salary_hike)')
abline(reg_log)

# RESIDUALS
reg_log$residuals
sum(reg_log$residuals)
mean(reg_log$residuals)
sqrt(mean(reg_log$residuals^2)) # RMSE 3.786004
plot(Salary_hike,reg_log$residuals,main='Residual plot: using log(Salary_hike)')
abline(0,0)

qqnorm(reg_log$residuals,main='Normal QQ plot of Residuals using log(Salary_hike)')
qqline(reg_log$residuals)
shapiro.test(reg_log$residuals) # 0.05982
ad.test(reg_log$residuals) # 0.06504

pred_log <- predict(reg_log)
pred_log_df <- as.data.frame(pred_log)
predicted_Churn_out_rate <- cbind(emp_data,pred_simple,reg_simple$residuals,pred_log,reg_log$residuals)
write.csv(predicted_Churn_out_rate,'E:\\EXCELR\\Datasets\\emp_data_predict.csv')
#################################################################

###################### GIST OF log(X) ################
shapiro.test(log(Salary_hike)) #0.5992
cor(Churn_out_rate,log(Salary_hike)) # -0.9212077
reg_log <- lm(Churn_out_rate ~ log(Salary_hike))
summary(reg_log)
# Multiple R-squared:  0.8486,	Adjusted R-squared:  0.8297 
plot(log(Salary_hike),Churn_out_rate,main='Regression line: using log(X)')
abline(reg_log)
plot(Salary_hike,Churn_out_rate)
abline(reg_simple)
sqrt(mean(reg_log$residuals^2)) # RMSE 3.786004
plot(log(Salary_hike),reg_log$residuals,main='Residual plot: using log(X)')
abline(0,0)
shapiro.test(reg_log$residuals) # p-value = 0.05982

################# exp(x)
exp(Salary_hike) # all values infinity
# >   exp(Salary_hike) # all values infinity
#  [1] Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf

####################### x^2 .. square of X ... Square(Salary_hike)
shapiro.test(Salary_hike^2) #0.4049
cor(Churn_out_rate,Salary_hike^2) # -0.9017223
reg_log <- lm(Churn_out_rate ~ (Salary_hike^2))
summary(reg_log)
# Multiple R-squared:  0.8312,	Adjusted R-squared:  0.8101

plot((Salary_hike^2),Churn_out_rate,main='Regression line: using log(X)')
abline(reg_log)
sqrt(mean(reg_log$residuals^2)) # RMSE 3.997528
plot(Salary_hike^2,reg_log$residuals,main='Residual plot: using log(X)')
abline(0,0)
shapiro.test(reg_log$residuals) # 0.05845

########################### sqrt(x)
shapiro.test(sqrt(Salary_hike)) #0.5509
cor(Churn_out_rate,sqrt(Salary_hike)) # -0.9165311
reg_log <- lm(Churn_out_rate ~ sqrt(Salary_hike))
summary(reg_log)
# Multiple R-squared:   0.84,	Adjusted R-squared:   0.82  
plot(sqrt(Salary_hike),Churn_out_rate,main='Regression line: using log(X)')
abline(reg_log)
sqrt(mean(reg_log$residuals^2)) # RMSE 3.891995
plot(sqrt(Salary_hike),reg_log$residuals,main='Residual plot: using log(X)')
abline(0,0)
shapiro.test(reg_log$residuals) # 0.05949

############################### 1/ sqrt(x)
shapiro.test(1/sqrt(Salary_hike)) #0.6459
cor(Churn_out_rate,(1/sqrt(Salary_hike))) # 0.9257473
reg_log <- lm(Churn_out_rate ~ (1/sqrt(Salary_hike)))
summary(reg_log)
# only intercept... no coefficent of X.. no R2 and no adj R2

################################# 1/x
shapiro.test(1/(Salary_hike)) #0.6901
cor(Salary_hike,(1/(Salary_hike))) # -0.9986812
reg_log <- lm(Churn_out_rate ~ (1/(Salary_hike)))
summary(reg_log)
# only intercept value... no coeff of X.. no R2 and no adj R2

############################## cuberoot(X))
8^(1/3)
salary_cuberoot <- Salary_hike^(1/3)
shapiro.test(salary_cuberoot) #0.5671
cor(Churn_out_rate,salary_cuberoot) # -0.918105
reg_log <- lm(Churn_out_rate ~ salary_cuberoot)
summary(reg_log)
# Multiple R-squared:  0.8429,	Adjusted R-squared:  0.8233 
plot(salary_cuberoot,Churn_out_rate,main='Regression line: using log(X)')
abline(reg_log)
sqrt(mean(reg_log$residuals^2)) # RMSE 3.85671
plot(salary_cuberoot,reg_log$residuals,main='Residual plot: using log(X)')
abline(0,0)
shapiro.test(reg_log$residuals) # 0.05968

######################## XX #########  XX ########### XX ####################