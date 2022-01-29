# Credit_Risk_Analysis


The written analysis has the following:

## Overview of loan prediction risk analysis

We are using Python and scikit-learn to predict the risk of a loan through a Machine learning model. We are going to use different models, evaluate them, compare them, to determine the best option.

## Purpose

We are Doing this prediction risk analysis to create a model to calculate the risk of some loan from the Bank automaticaly using some parameters.

- Results

As a first part of the procedure we had to obtain the dataframe from an Excel. Then we filter the information to create the dataframe with our value to predict, which is the status of the loan. And we separate it from our independent values.

    columns = [
        "loan_amnt", "int_rate", "installment", "home_ownership",
        "annual_inc", "verification_status", "issue_d", "loan_status",
        "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "revol_bal", "total_acc",
        "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
        "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
        "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
        "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
        "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
        "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
        "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
        "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
        "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
        "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
        "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
        "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
        "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
        "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
        "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
    ]

    target = ["loan_status"]

    # Load the data
    file_path = Path('LoanStats_2019Q1.csv')
    df = pd.read_csv(file_path, skiprows=1)[:-2]
    df = df.loc[:, columns].copy()

    # Drop the null columns where all values are null
    df = df.dropna(axis='columns', how='all')

    # Drop the null rows
    df = df.dropna()

    # Remove the `Issued` loan status
    issued_mask = df['loan_status'] != 'Issued'
    df = df.loc[issued_mask]

    # convert interest rate to numerical
    df['int_rate'] = df['int_rate'].str.replace('%', '')
    df['int_rate'] = df['int_rate'].astype('float') / 100


    # Convert the target column values to low_risk and high_risk based on their values
    x = {'Current': 'low_risk'}   
    df = df.replace(x)

    x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
    df = df.replace(x)

    df.reset_index(inplace=True, drop=True)

    df.head()

Here we can see the dataframe created.

![image](https://user-images.githubusercontent.com/88845919/151644817-c81aa428-a733-4989-a0aa-c0fedd6bc80c.png)

The information was then separated into training and testing to put the models into practice and determine the best option.

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    Counter(y_train)

We use the following algorithms.

- Oversampling.

![image](https://user-images.githubusercontent.com/88845919/151645378-977592e0-a445-42b8-b725-f995fda9e975.png)

- SMOTE Oversampling.

![image](https://user-images.githubusercontent.com/88845919/151645393-5b1fbe5c-5e6a-477b-aa28-25501c832629.png)

- Undersampling.

![image](https://user-images.githubusercontent.com/88845919/151645408-15e08d45-5e9b-4ac8-a85d-9e2126914504.png)

- Combination (Over and Under) Sampling.

![image](https://user-images.githubusercontent.com/88845919/151645421-ede3d714-9623-4c90-be71-44bfd68979be.png)

- Balanced Random Forest Classifier

![image](https://user-images.githubusercontent.com/88845919/151645433-b41963ce-6ac4-41cb-bfc3-4beb6546d366.png)

- Easy Ensemble AdaBoost Classifier

![image](https://user-images.githubusercontent.com/88845919/151645445-5be7c03a-4138-44f8-92b8-218649fda82d.png)

- Summary:

We use six different models to find the best option to determine the risk of the loan. We have to categories "High" and "Low" risk, and several  parameters like "home_ownership", "loan_amnt", etc.

The models let us predict the result considering this values for a future case.

We can notice that the model with higher accuracy is the Easy Ensemble AdaBoost Classifier with 93.2%. So this model fits very well to the results. But this is not the only important thing. In this case the most important thing is to have a sensitive model that can catch most of the high risk loans, and in recall section we can see that this model almost catch all the high risk examples.

Also in the condusion matrix we can see that we have very few False Negative, so the risk is lower with this model.

Considering this numbers compare to the other models, the best option to use is the Easy Ensemble AdaBoost Classifier. It has a good feed and we obtain a good sensitivity to catch most of the cases.
