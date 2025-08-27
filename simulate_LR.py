
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statistics

def standalone_logistic(X, y):
    # Add intercept explicitly
    X_with_intercept = sm.add_constant(X)
    # Fit model using statsmodels for inference
    model = sm.Logit(y, X_with_intercept)
    result = model.fit(disp=False)

    predicted_prob = result.predict(X_with_intercept)  # returns P(y=1|x)
    coef = result.params          # includes intercept and weights
    stderr = result.bse           # standard errors for each beta
    pvals = result.pvalues        # p-values (Wald test)

    return predicted_prob, coef, stderr, pvals

### experiment-1:
### control: generate 1000 obs from a gaussian with mean 3 and variance 1
### case: generate 1000 obs from a gaussian with mean 10 and variance 1
### fit a logostic regression with obs as features and labels as y

exp1_cor_list = []
exp2_cor_list = []

exp1_mean_phat_case = []
exp2_mean_phat_case = []

exp1_mean_phat_control = []
exp2_mean_phat_control = []

NUM_TRIALS = 1000
num_obs = 1000
case_mean = 20
control_mean = 3
sd = 1
case_w = 5
control_w = 5
noise = np.random.normal(0, sd, num_obs*2)

y = np.array([0]*(num_obs) + [1]*num_obs)

for _ in range(NUM_TRIALS):
    # Experiment 1
    control_obs = np.random.normal(loc=control_mean, scale=sd, size=num_obs)
    case_obs = np.random.normal(loc=case_mean, scale=sd, size=num_obs)
    X_exp1 = np.concatenate([control_obs, case_obs])
    X_exp1 += noise  # add noise to both experiments
    try:
        predicted_prob, coef, stderr, pvals = standalone_logistic(X_exp1, y)
        exp1_mean_phat_case.append(np.mean(predicted_prob[y==1]))
        exp1_mean_phat_control.append(np.mean(predicted_prob[y==0]))
        all_corr_exp1 = np.corrcoef(predicted_prob, X_exp1)[0, 1]
        exp1_cor_list.append(abs(all_corr_exp1))
    except np.linalg.LinAlgError:
        # skip singular cases
        continue

    # Experiment 2
    #cases: 90% draws from control mean and 10% from case mean (edge case) 
    
    control_obs = np.random.normal(loc=control_mean, scale=sd, size=int((2*num_obs*control_w)/10))
    case_obs = np.random.normal(loc=case_mean, scale=sd, size=int((2*num_obs*case_w)/10))
    X_exp2 = np.concatenate([control_obs, case_obs])
    X_exp2 += noise  # add noise to both experiments
    try:
        predicted_prob, coef, stderr, pvals = standalone_logistic(X_exp2, y)
        all_corr_exp2 = np.corrcoef(predicted_prob, X_exp2)[0, 1]
        exp2_cor_list.append(abs(all_corr_exp2))
        exp2_mean_phat_case.append(np.mean(predicted_prob[y==1]))
        exp2_mean_phat_control.append(np.mean(predicted_prob[y==0]))
    except np.linalg.LinAlgError:
        continue

print('Exp2: #obs from case dist:', int((2*num_obs*case_w)/10))
print('Exp2: #obs from control dist:', int((2*num_obs*control_w)/10))

mean_exp1 = statistics.mean(exp1_cor_list)
print(f"The mean of exp1 is: {mean_exp1}") #The mean of exp1 is: 0.9495776653940196

mean_exp2 = statistics.mean(exp2_cor_list)
print(f"The mean of exp2 is: {mean_exp2}") #The mean of exp2 is: 0.9918859083517851



### visualize the violin plot of experiment 1 and 3 mean p-hat valuesn for case and control

plt.figure(figsize=(5, 5))
violin_parts = plt.violinplot([exp1_mean_phat_control, exp1_mean_phat_case, 
                               exp2_mean_phat_control, exp2_mean_phat_case], showmeans=True)
plt.xticks([1, 2, 3, 4], ['Exp1 Case', 'Exp1 Control', 'Exp2 Case', 'Exp2 Control'])
plt.ylabel('Mean p-hat')
plt.suptitle('Mean p-hat values over 1000 draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
colors = ['blue', 'blue', 'red', 'red']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
plt.show()


### visualize the violin plot of experiment 1 and 3 correlation values
plt.figure(figsize=(8, 6))
plt.violinplot([exp1_cor_list, exp2_cor_list], showmeans=True)
plt.xticks([1, 2], ['Experiment 1', 'Experiment 2'])
plt.ylabel('Absolute Correlation Coefficient')
plt.suptitle('Correlation values of X vs p-hat over 1000 draws in two experiments')
plt.title('Exp1: ' + str(round(mean_exp1,3)) + ', Exp2: ' + str(round(mean_exp2,3)) + ', case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
### add another subtitle:
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
plt.show()


