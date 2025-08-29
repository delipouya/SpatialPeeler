
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statistics
from sklearn.cluster import KMeans
from SpatialPeeler import plotting as plot
RAND_SEED = 28

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

#### save correlation between X(factor) and p-hat for all data points
exp1_cor_x_phat_case0 = []
exp1_cor_x_phat_case1 = []
exp1_cor_x_phat_control = []

exp2_cor_x_phat_case0 = []
exp2_cor_x_phat_case1 = []
exp2_cor_x_phat_control = []

exp1_cor_res_phat_case0 = []
exp1_cor_res_phat_case1 = []
exp1_cor_res_phat_control = []

exp2_cor_res_phat_case0 = []
exp2_cor_res_phat_case1 = []
exp2_cor_res_phat_control = []

#### save mean phat values (before clustering)
exp1_mean_phat_case = []
exp2_mean_phat_case = []
exp1_mean_phat_control = []
exp2_mean_phat_control = []

#### save mean residual values (before clustering)
exp1_mean_residual_case = []
exp2_mean_residual_case = []
exp1_mean_residual_control = []
exp2_mean_residual_control = []

#### save mean phat values for case cluster 0 and 1, and control cluster 0
exp1_mean_phat_case0 = []
exp1_mean_phat_case1 = []
exp1_mean_phat_control0 = []

exp2_mean_phat_case0 = []
exp2_mean_phat_case1 = []
exp2_mean_phat_control0 = []
#### save mean residuals for case cluster 0 and 1, and control cluster 0
exp1_mean_residual_case0 = []
exp1_mean_residual_case1 = []
exp1_mean_residual_control0 = []

exp2_mean_residual_case0 = []
exp2_mean_residual_case1 = []
exp2_mean_residual_control0 = []


exp1_coef = []
exp1_pvalue = []
exp2_coef = []
exp2_pvalue = []

NUM_TRIALS = 1000
num_obs = 1000

case_mean = 5
control_mean = 3
sd = 1

case_w = 1
control_w = 9

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
        exp1_coef.append(coef)
        exp1_pvalue.append(pvals)

        logit_phat = plot.safe_logit(predicted_prob)
        residual = y - logit_phat

        exp1_mean_phat_case.append(np.mean(predicted_prob[y==1]))
        exp1_mean_phat_control.append(np.mean(predicted_prob[y==0]))

        exp1_mean_residual_case.append(np.mean(residual[y==1]))
        exp1_mean_residual_control.append(np.mean(residual[y==0]))


        # Cluster only the case samples based on p-hat values
        p_hat_factor_case = predicted_prob[y==1]
        kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
        kmeans.fit(p_hat_factor_case.reshape(-1, 1))
    
        #print(num_obs == len(kmeans.labels_))

        # Remap labels so that 1 = higher p-hat, 0 = lower p-hat
        centers = kmeans.cluster_centers_.ravel()
        order = np.argsort(centers)               # [low_center_label, high_center_label]
        remap = {order[0]: 0, order[1]: 1}
        labels_remapped = np.vectorize(remap.get)(kmeans.labels_).astype(int)
        
        # Initialize a vector of length num_obs*2 NaN 
        case_p_hat_clusters = np.nan * np.ones(num_obs*2)
        case_p_hat_clusters[y==1] = labels_remapped.astype(str) #[ 0.,  1., nan]), count ([ 141,  859, 1000]))

        ### p-hat mean for clusters
        exp1_mean_phat_case0.append(np.mean(predicted_prob[(y==1) & (case_p_hat_clusters==0)]))
        exp1_mean_phat_case1.append(np.mean(predicted_prob[(y==1) & (case_p_hat_clusters==1)]))
        exp1_mean_phat_control0.append(np.mean(predicted_prob[y==0]))

        ### residual mean for clusters
        exp1_mean_residual_case0.append(np.mean(residual[(y==1) & (case_p_hat_clusters==0)]))
        exp1_mean_residual_case1.append(np.mean(residual[(y==1) & (case_p_hat_clusters==1)]))
        exp1_mean_residual_control0.append(np.mean(residual[y==0]))

        exp1_cor_x_phat_case0.append(abs(np.corrcoef(predicted_prob[(y==1)&(case_p_hat_clusters==0)], X_exp1[(y==1) & (case_p_hat_clusters==0)])[0, 1]))
        exp1_cor_x_phat_case1.append(abs(np.corrcoef(predicted_prob[(y==1)&(case_p_hat_clusters==1)], X_exp1[(y==1) & (case_p_hat_clusters==1)])[0, 1]))
        exp1_cor_x_phat_control.append(abs(np.corrcoef(predicted_prob[y==0], X_exp1[y==0])[0, 1]))
        
        exp1_cor_res_phat_case0.append(abs(np.corrcoef(residual[(y==1) & (case_p_hat_clusters==0)], X_exp1[(y==1) & (case_p_hat_clusters==0)])[0, 1]))
        exp1_cor_res_phat_case1.append(abs(np.corrcoef(residual[(y==1) & (case_p_hat_clusters==1)], X_exp1[(y==1) & (case_p_hat_clusters==1)])[0, 1]))
        exp1_cor_res_phat_control.append(abs(np.corrcoef(residual[y==0], X_exp1[y==0])[0, 1]))



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
        logit_phat = plot.safe_logit(predicted_prob)
        residual = y - logit_phat
        
        exp2_coef.append(coef)
        exp2_pvalue.append(pvals)


        exp2_mean_phat_case.append(np.mean(predicted_prob[y==1]))
        exp2_mean_phat_control.append(np.mean(predicted_prob[y==0]))

        exp2_mean_residual_case.append(np.mean(residual[y==1]))
        exp2_mean_residual_control.append(np.mean(residual[y==0]))

        # Cluster only the case samples based on p-hat values
        p_hat_factor_case = predicted_prob[y==1]
        kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
        kmeans.fit(p_hat_factor_case.reshape(-1, 1))
    
        #print(num_obs == len(kmeans.labels_))

        # Remap labels so that 1 = higher p-hat, 0 = lower p-hat
        centers = kmeans.cluster_centers_.ravel()
        order = np.argsort(centers)               # [low_center_label, high_center_label]
        remap = {order[0]: 0, order[1]: 1}
        labels_remapped = np.vectorize(remap.get)(kmeans.labels_).astype(int)

        # Initialize a vector of length num_obs*2 NaN 
        case_p_hat_clusters = np.nan * np.ones(num_obs*2)
        case_p_hat_clusters[y==1] = labels_remapped.astype(str) #[ 0.,  1., nan]), count ([ 141,  859, 1000]))

        exp2_mean_phat_case0.append(np.mean(predicted_prob[(y==1) & (case_p_hat_clusters==0)]))
        exp2_mean_phat_case1.append(np.mean(predicted_prob[(y==1) & (case_p_hat_clusters==1)]))
        exp2_mean_phat_control0.append(np.mean(predicted_prob[y==0]))

        exp2_mean_residual_case0.append(np.mean(residual[(y==1) & (case_p_hat_clusters==0)]))
        exp2_mean_residual_case1.append(np.mean(residual[(y==1) & (case_p_hat_clusters==1)]))
        exp2_mean_residual_control0.append(np.mean(residual[y==0]))

        exp2_cor_x_phat_case0.append(abs(np.corrcoef(predicted_prob[(y==1)& (case_p_hat_clusters==0)], X_exp2[(y==1)& (case_p_hat_clusters==0)])[0, 1]))
        exp2_cor_x_phat_case1.append(abs(np.corrcoef(predicted_prob[(y==1)& (case_p_hat_clusters==1)], X_exp2[(y==1)& (case_p_hat_clusters==1)])[0, 1]))
        exp2_cor_x_phat_control.append(abs(np.corrcoef(predicted_prob[y==0], X_exp2[y==0])[0, 1]))

        exp2_cor_res_phat_case0.append(abs(np.corrcoef(residual[(y==1)& (case_p_hat_clusters==0)], X_exp2[(y==1)& (case_p_hat_clusters==0)])[0, 1]))
        exp2_cor_res_phat_case1.append(abs(np.corrcoef(residual[(y==1)& (case_p_hat_clusters==1)], X_exp2[(y==1)& (case_p_hat_clusters==1)])[0, 1]))
        exp2_cor_res_phat_control.append(abs(np.corrcoef(residual[y==0], X_exp2[y==0])[0, 1]))

    except np.linalg.LinAlgError:
        continue

print('Exp2: #obs from case dist:', int((2*num_obs*case_w)/10))
print('Exp2: #obs from control dist:', int((2*num_obs*control_w)/10))

'''
### visualize the violin plot of experiment 1 and 2 mean residuals for case and control - after clusterg
plt.figure(figsize=(16, 10))
violin_parts = plt.violinplot([exp1_mean_residual_control0, exp1_mean_residual_case0, exp1_mean_residual_case1,
                               exp2_mean_residual_control0, exp2_mean_residual_case0, exp2_mean_residual_case1], 
                               showmeans=True)
plt.xticks([1, 2, 3, 4, 5, 6], ['Exp1 Control', 'Exp1 Case0', 'Exp1 Case1', 
                                'Exp2 Control0', 'Exp2 Case0', 'Exp2 Case1'])
plt.ylabel('Mean Residuals')
plt.suptitle('Mean Residuals over ' + str(NUM_TRIALS) + ' draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
colors = ['blue', 'green', 'red', 'blue', 'green', 'red']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
plt.show()

#### visualize the violin plot of experiment 1 and 2 mean p-hat scores for case and control - after clustering
plt.figure(figsize=(16,10))
violin_parts = plt.violinplot([exp1_mean_phat_control0, exp1_mean_phat_case0, exp1_mean_phat_case1,
                               exp2_mean_phat_control0, exp2_mean_phat_case0, exp2_mean_phat_case1])
plt.xticks([1, 2, 3, 4, 5, 6], ['Exp1 Control', 'Exp1 Case0', 'Exp1 Case1', 
                                'Exp2 Control0', 'Exp2 Case0', 'Exp2 Case1'])
plt.ylabel('Mean p-hat')
plt.suptitle('Mean p-hat over ' + str(NUM_TRIALS) + ' draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
colors = ['blue', 'green', 'red', 'blue', 'green', 'red']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
plt.show()


### visualize the violin plot of experiment 1 and 2 mean residuals for case and control - before clusterg
plt.figure(figsize=(10, 10))
violin_parts = plt.violinplot([exp1_mean_residual_control, exp1_mean_residual_case, 
                               exp2_mean_residual_control, exp2_mean_residual_case], 
                               showmeans=True)
plt.xticks([1, 2, 3, 4], ['Exp1 Control', 'Exp1 Case', 'Exp2 Control', 'Exp2 Case'])
plt.ylabel('Mean Residuals')
plt.suptitle('Mean Residuals over ' + str(NUM_TRIALS) + ' draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
colors = ['yellow', 'yellow', 'green', 'green']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
plt.show()


### visualize the violin plot of experiment 1 and 2 mean p-hat values for case and control - before clustering
plt.figure(figsize=(10, 10))
violin_parts = plt.violinplot([exp1_mean_phat_control, exp1_mean_phat_case, 
                               exp2_mean_phat_control, exp2_mean_phat_case], showmeans=True)
plt.xticks([1, 2, 3, 4], ['Exp1 Control', 'Exp1 Case', 'Exp2 Control', 'Exp2 Case'])
plt.ylabel('Mean p-hat')
plt.suptitle('Mean p-hat values over ' + str(NUM_TRIALS) + ' draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
colors = ['blue', 'blue', 'red', 'red']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
plt.show()
'''

### visualize the violin plot of experiment 1 and 3 correlation values
plt.figure(figsize=(16, 10))
plt.violinplot([exp1_cor_res_phat_control, exp1_cor_res_phat_case0, exp1_cor_res_phat_case1,
                exp2_cor_res_phat_control, exp2_cor_res_phat_case0, exp2_cor_res_phat_case1], showmeans=True)
plt.xticks([1, 2, 3, 4, 5, 6], ['Exp1 Control', 'Exp1 Case0', 'Exp1 Case1', 
                                'Exp2 Control', 'Exp2 Case0', 'Exp2 Case1'])
plt.ylabel('Absolute Correlation Coefficient')
plt.suptitle('Correlation values of Residual vs p-hat over ' + str(NUM_TRIALS) + ' draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
### add another subtitle:
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
plt.show()



### visualize the violin plot of experiment 1 and 3 correlation values
plt.figure(figsize=(16, 10))
plt.violinplot([exp1_cor_x_phat_control, exp1_cor_x_phat_case0, exp1_cor_x_phat_case1,
                exp2_cor_x_phat_control, exp2_cor_x_phat_case0, exp2_cor_x_phat_case1], showmeans=True)
plt.xticks([1, 2, 3, 4, 5, 6], ['Exp1 Control', 'Exp1 Case0', 'Exp1 Case1', 
                                'Exp2 Control', 'Exp2 Case0', 'Exp2 Case1'])
plt.ylabel('Absolute Correlation Coefficient')
plt.suptitle('Correlation values of X vs p-hat over' + str(NUM_TRIALS) + 'draws in two experiments')
plt.title('case-mean: ' + str(case_mean) + ', control-mean: ' + str(control_mean))
### add another subtitle:
plt.figtext(0.5, 0.01, 'Case weight: ' + str(case_w) + ', Control weight: ' + str(control_w), 
            wrap=True, horizontalalignment='center', fontsize=10)
plt.show()


