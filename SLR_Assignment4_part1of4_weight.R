# Assignment 4 - Simple Linear Regression 

# Q1. predict weight gained using calories consumed

cals <- read.csv("file:///E:/EXCELR/Assignments/Simple_Linear_Regression_Assignment/calories_consumed.csv")
# 2 vars weight_gained and calaries_consumed and 14 records
names(cals)
names(cals) <- c("weight_gained","calories_consumed")
head(cals)
attach(cals)
summary(cals) # both mean and median diffrnt hence outliers
str(cals) # both continuous data
library(Hmisc)
describe(cals)
# I BM
mean(weight_gained) # 357.7
median(weight_gained) # 200

install.packages("modeest")
library(modeest)
?modeest
mlv(weight_gained,method='mlv') # 200 frq is 2

mean(calories_consumed) # 2341
median(calories_consumed) # 2250
mlv(calories_consumed,method='mlv') # 1900 frq is 2

# IQR
summary(cals)
IQR(weight_gained)  #  537.5 - 114.5 = 423
# outlier lower limit of weight var
114.5 - 1.5 * (423) # -520 
# outlier upper limit weight var
537.5 + (1.5* IQR(weight_gained)) # 1172
# values below -520 & above  1172 are outliers, hence no outliers in weight var

# outlier limits of calories var
IQR(calories_consumed) # 2775 - 1728 = 1047.5
1728 - (1.5 * IQR(calories_consumed)) #  156.75
2775 + (1.5 * IQR(calories_consumed)) #  4346.25
# values below 156.75 & above  4346.25 are outliers, hence no outliers in calories var

# second BM decisions
var(weight_gained) # 111350.7
sd(weight_gained)  # 333.6925
range(weight_gained) # 62 & 1100
max(weight_gained)-min(weight_gained) # range is 1038
rangevalue <- function(x){max(x)-min(x)}
rangevalue(weight_gained)

var(calories_consumed) # 565668.7
sd(calories_consumed) # 752.1095
range(calories_consumed) #1400 & 3900
rangevalue(calories_consumed) # 2500

# third and fourth BM
skewness(weight_gained) #  1.116977 & (0.9994639)
kurtosis(weight_gained) # 2.891938

skewness(calories_consumed) # 0.5825597
kurtosis(calories_consumed) # 2.403367

# visualization
boxplot_wt <- boxplot(weight_gained, horizontal = TRUE, 
                      main="Boxplot of weight_gained")
boxplot_cals <- boxplot(calories_consumed, horizontal = TRUE, 
                        main = "Boxplot of calories_consumed")

# finding outliers
boxplot_wt$out # no outliers in weight variable
which(weight_gained%in% boxplot_wt)
boxplot_cals$out # no outliers in calories variable
which(weight_gained %in% boxplot_cals)

# finding missing values
is.na(weight_gained)
is.na(calories_consumed) # no missing values in both var

# histogram
hist(weight_gained,probability = T) # prob=TRUE for probabilities not counts
# right skewed
lines(density(weight_gained))     # add a density estimate with defaults
lines(density(weight_gained,adjust = 2),lty='dotted') # add another "smoother" density
hist(calories_consumed,probability = T)
lines(density(calories_consumed))
lines(density(calories_consumed,adjust = 2),lty='dotted')

barplot(weight_gained,main="Barplot of weight_gained",horiz = T)
barplot(calories_consumed,main="Barplot of calories consumed")
dotchart(weight_gained, xlab = "weight_gained")
dotchart(calories_consumed, xlab = "calories_consumed")
plot(densityplot(weight_gained))
plot(densityplot(calories_consumed))

# to identify missing values
library(Hmisc)
describe(cals) # missing values =0

#checking normality of weight_gained
qqnorm(weight_gained,main = 'Normal QQ plot of weight_gained')
qqline(weight_gained)
skewness(weight_gained)
kurtosis(weight_gained)
hist(weight_gained)
shapiro.test(weight_gained) # p value = 0.006646 < 0.05 data not normal
library(nortest)
ad.test(weight_gained) # p-value = 0.004069

# checking normality of calories_consumed
qqnorm(calories_consumed,main='Normal QQ plot of calories_consumed')
qqline(calories_consumed) # data normal
skewness(calories_consumed)
kurtosis(calories_consumed)
shapiro.test(calories_consumed) # p-value = 0.4887
ad.test(calories_consumed) # p-value = 0.6086  calories var is normal

# since Y not normal checking transformation log(Y) and sqrt(Y)
shapiro.test(log(weight_gained))  # pvalue>0.05, normal
shapiro.test(sqrt(weight_gained)) # pvalue > 0.05, normal

# to find missing values in dataset
library(mice)
md.pattern(cals)
# OR
cals[!complete.cases(cals),]

################### MODEL BUILDING #####################################
# PRE-REQUISITES FOR REGRESSION MODEL
# Linear relationship of Y and X
# scatter plot
plot(calories_consumed,weight_gained,main="Scatter plot with Y")
cor(weight_gained,calories_consumed) # 0.946991, strong positive

# Standard Regression Model 
reg_simple <- lm(weight_gained~ calories_consumed)
summary((reg_simple))
# pvalue both intercept and coeff is 0.0
# Multiple R-squared:  0.8968,	Adjusted R-squared:  0.8882

# confidence and prediction intervals
confint(reg_simple,level = 0.95)
predict(reg_simple,interval = 'predict')

# plotting regression line
plot(calories_consumed,weight_gained,main='Scatter plot with regression line using Y')
abline(reg_simple)

# errors or residuals

sum(reg_simple$residuals) # =0, assumption of error
mean(reg_simple$residuals) # =0
sqrt(mean(reg_simple$residuals^2)) # RMSE = 103.3025

# residual plot, plot residuals against indepen var calories_consumed
# should show random pattern
plot(calories_consumed,reg_simple$residuals,main='Residual plot: standard regression')
abline(0,0)

# symmetrical disbn of errors across mean
qqnorm(reg_simple$residuals,main = "Normal QQ plot of Residuals")
qqline(reg_simple$residuals) # from plot not normal
shapiro.test(reg_simple$residuals) # 0.155 > 0.05, normal
library(nortest)
ad.test(reg_simple$residuals)

pred_reg <- predict(reg_simple)

#############################################################################

# using transformed var: log(Y)
plot(calories_consumed,log(weight_gained),main = "Scatter plot using log(Y)")
cor(log(weight_gained),calories_consumed) # 0.9368037, strong positive

qqnorm(log(weight_gained),main='Normal QQ Plot for log(weight_gained)')
qqline(log(weight_gained))
shapiro.test(log(weight_gained))
#linear eqn using log(y)
reg_log <- lm(log(weight_gained)~ calories_consumed)
summary((reg_log))
# pvalue both 0.0
# Multiple R-squared:  0.8776,	Adjusted R-squared:  0.8674
exp(2.8386724) = 17.09306
exp(0.0011336) = 1.001134

# scatter plot with regression line
plot(calories_consumed,log(weight_gained),main = "Scatter plot with Regression line using log(Y)")
abline(reg_log)
# plotting residuals against independent var
plot(calories_consumed,reg_log$residuals,main='Residual plot - Regression using log(y)')
abline(0, 0)
pred_reg_log <- predict(reg_log)
pred_reg_log

# errors or residuals
reg_log$residuals
sum(reg_log$residuals) # =0, assumption of error
mean(reg_log$residuals) # =0
sqrt(mean(reg_log$residuals^2)) # RMSE = 0.3068228
qqnorm(reg_log$residuals, main = 'Normal QQ plot: log(y) residuals')
qqline(reg_log$residuals)
shapiro.test(reg_log$residuals) #  p-value = 0.05947
library(nortest)
ad.test(reg_log$residuals)

pred_log <- predict(reg_log)
pred_exp <- exp(pred_log)
##################################################
# quadratic equation: using suare root of y

plot(calories_consumed,sqrt(weight_gained),main='Scatter plot using sqrt(y)')
cor(sqrt(weight_gained),calories_consumed)
qqnorm(sqrt(weight_gained),main='Normal QQ Plot for sqrt(weight_gained)')
qqline(sqrt(weight_gained))
shapiro.test(sqrt(weight_gained)) # 0.0631, normal data

# regression model
reg_sqrt <- lm(sqrt(weight_gained) ~ calories_consumed)
summary(reg_sqrt)
# Multiple R-squared:  0.9139,	Adjusted R-squared:  0.9067
# squaring the coefficients
(-7.1154342)^2 # 50.6294
(0.0103864)^2 # 0.0001078773
pred_reg_sqrt <- predict(reg_sqrt)
# since used sqrt of y.. predicted values have to be sqaured to bring to original units
pred_sqr <- pred_reg__sqrt^2

# PLOT REGR line
plot(calories_consumed,sqrt(weight_gained),main='Scatter plot with Regression line using sqrt(Y)')
abline(reg_sqrt)
 # RESIDUALS
sum(reg_sqrt$residuals) # 1.498801e-15
mean(reg_sqrt$residuals) # 1.070417e-16
sqrt(mean(reg_sqrt$residuals^2)) # RMSE 2.310718
# residual plot
attach(cals)
plot(calories_consumed,reg_sqrt$residuals,main='Residual plot using sqrt(Y)')
abline(0,0)
# normality of residuals
qqnorm(reg_sqrt$residuals,main = "Normal QQ plot: sqrt(y)")
qqline(reg_sqrt$residuals)
shapiro.test(reg_sqrt$residuals)

pred_sqrt <- predict(reg_sqrt)
pred_sqr <- (pred_sqrt)^2

predict_weight <- cbind(cals,pred_reg,reg_simple$residuals,pred_log,reg_log$residuals,pred_sqrt,pred_sqr,reg_sqrt$residuals)
head(predict_weight)
# saving output file in local drive
write.csv(predict_weight,"E:\\EXCELR\\Datasets\\calories_consumed_predict.csv")

###################### XX ######## XX ######### XX####################


