# Assignment4 -part2 of 4- Simple linear Regression

# Q2  Salary_hike -> Build a prediction model for Salary_hike
salary_hike <- read.csv("file:///E:/EXCELR/Assignments/Simple_Linear_Regression_Assignment/Salary_Data.csv")
attach(salary_hike)
# Details of dataset
dim(salary_hike)
# 30 obsvn 2 variables
names(salary_hike)
#  varaibles are "YearsExperience" "Salary" 
class(salary_hike)
str(salary_hike)
head(salary_hike)

# large dataset
10*(skewness(Salary)^2)
10*(kurtosis(Salary))
10*(skewness(YearsExperience)^2)
10*(kurtosis(YearsExperience))
summary(salary_hike)

# I BM
mean(YearsExperience)
median(YearsExperience)
library(modeest)
mlv(YearsExperience,method='mfv')
mean(Salary)
median(Salary)
mlv(Salary,method='mfv')
# II BM
var(YearsExperience)
sd(YearsExperience)
range(YearsExperience)
max(YearsExperience)-min(YearsExperience)
var(Salary)
sd(Salary)
range(Salary)
max(Salary)-min(Salary)
# III BM
skewness(YearsExperience)
skewness(Salary)

# IV BM
library(moments)
kurtosis(YearsExperience)
kurtosis(Salary)

# VISUALIZATIONS
boxplot_exp <- boxplot(YearsExperience,horizontal=T,main='Boxplot of YearsExperience ')
boxplot_sal <- boxplot(Salary,horizontal=T,main='Boxplot of Salary ')

# finding outliers
boxplot_exp$out
boxplot_sal$out

hist(YearsExperience,probability = T)
lines(density(YearsExperience))
lines(density(YearsExperience,adjust = 2), lty='dotted')
hist(Salary,probability = TRUE)
lines(density(Salary))
lines(density(Salary,adjust = 2),lty='dotted')

barplot(YearsExperience,horiz = T,main='Barplot of YearsExperience')
barplot(Salary,horiz = T,main = 'Barplot of salary')

dotchart(YearsExperience,xlab = 'YearsExperience')
dotchart(Salary,xlab = 'salary')
densityplot(YearsExperience)
densityplot(Salary)

# Normality
qqnorm(YearsExperience, main='Normal qq plot of YearsExperience')
qqline(YearsExperience)
qqnorm(Salary,main='Normal qq plot of salary')
qqline(Salary)
shapiro.test(YearsExperience) # p-value = 0.1034, normal
library(nortest)
ad.test(YearsExperience) # p-value = 0.1365 , normal
shapiro.test(Salary) # p-value = 0.01516, not normal
ad.test(Salary) # p-value = 0.01516, not normal

# outliers detection
box <- boxplot(YearsExperience)
box$out

# missing values
# finding missing values
salary_hike[!complete.cases(salary_hike),]
library(mice)
md.pattern(salary_hike)
library(Hmisc)
describe(YearsExperience)
describe(Salary)

################# MODEL BUILDING ###########################
# LINEAR RELATIONSHIP
plot(YearsExperience,Salary, main='Scatterplot of salary vs YearsExperience')
cor(Salary,YearsExperience) # 0.9782416

# standard regression model
reg_simple <- lm(Salary ~ YearsExperience)
summary(reg_simple) 
# Multiple R-squared:  0.957,	Adjusted R-squared:  0.9554

# confidence and prediction intervals
confint(reg_simple,level=0.95)
predict(reg,interval="predict")

# plotting regression line
plot(YearsExperience,Salary, main='Regression Line - Salary vs YearsExperience')
abline(reg_simple)

# Residuals
sum(reg_simple$residuals) # is 0
mean(reg_simple$residuals) # is 0
sqrt(mean(reg_simple$residuals^2)) # RMSE= 5592.044

# residual plot, plot residuals against indepen var 
# should show random pattern
plot(YearsExperience,reg_simple$residuals,main='Residual plot: standard regression')
abline(0,0)

# symmetrical disbn of errors across mean
qqnorm(reg_simple$residuals,main = "Normal QQ plot of Residuals")
qqline(reg_simple$residuals) # from plot not normal
shapiro.test(reg_simple$residuals) # p-value = 0.1952 > 0.05, normal
library(nortest)
ad.test(reg_simple$residuals) # p-value = 0.428, normal

pred_simple <- predict(reg_simple)
############################### sqrt(Y) ##########################

# trying sqrt(Y) -- but data is still not normal
qqnorm(sqrt(Salary), main='Normal QQ plot using sqrt(Y)')
qqline(sqrt(Salary))
shapiro.test(sqrt(Salary)) # 0.03503, not normal
ad.test(sqrt(Salary))

################ log(Y)

# log(Y)  --- data is normal
qqnorm(log(Salary),main='Normal QQ plot using log(Y)')
qqline(log(Salary))
shapiro.test(log(Salary)) #0.05406
ad.test(log(Salary)) # 0.06408

cor(YearsExperience,log(Salary)) # 0.9653844
plot(YearsExperience,log(Salary),main='Scatterplot using log(Y)')

# regression - log(Y)
reg_log <- lm(log(Salary) ~ YearsExperience)
summary(reg_log)
# Multiple R-squared:  0.932,	Adjusted R-squared:  0.9295
exp(0.125453)-1

plot(YearsExperience,log(Salary),main='Regression line: using log(Y)')
abline(reg_log)

# RESIDUALS
reg_log$residuals
sum(reg_log$residuals)
mean(reg_log$residuals)
sqrt(mean(reg_log$residuals^2)) # RMSE 0.09457437
plot(YearsExperience,reg_log$residuals,main='Residual plot: using log(Y)')
abline(0,0)

qqnorm(reg_log$residuals,main='Normal QQ plot of Residuals')
qqline(reg_log$residuals)
shapiro.test(reg_log$residuals)
ad.test(reg_log$residuals)
plot(reg_simple$residuals)
abline(0,0)

pred_log <- predict(reg_log)

predicted_salary <- cbind(salary_hike,pred_simple,reg_simple$residuals,pred_log,reg_log$residuals)
write.csv(predicted_salary,'E:\\EXCELR\\Datasets\\salary_hike_predict.csv')

##########################  XX  #########  XX  ############################

