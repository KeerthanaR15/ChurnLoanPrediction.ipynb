#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# In[7]:


df = pd.read_csv('Telco-Customer-Churn.csv')


# In[8]:


c


# In[9]:


print(df.dtypes)


# In[10]:


df.head()


# In[11]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[12]:


median_tc = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_tc, inplace=True)


# In[13]:


df.drop(['customerID'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)


# In[15]:


counts = df['Churn'].value_counts()
plt.figure()
counts.plot.bar()
plt.title('Churn vs. Stay Counts')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()


# In[16]:


plt.figure()
plt.hist(df['MonthlyCharges'], bins=30)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.show()


# In[17]:


corr = df.select_dtypes(include=['float64','int64']).corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Numeric Feature Correlations')
plt.show()


# In[18]:


bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['0-12','13-24','25-36','37-48','49-60','61-72']
df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels)


# In[19]:


df['StreamingBundle'] = np.where(
    (df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes'),
    1, 0
)


# In[20]:


df['LoyaltyScore'] = df['tenure'] * df['MonthlyCharges']


# In[21]:


df.drop(['tenure','StreamingTV','StreamingMovies'], axis=1, inplace=True)


# In[25]:


df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# In[27]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('LR Accuracy:', accuracy_score(y_test, y_pred_lr))
print('LR ROC-AUC:', roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))


# In[28]:


rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))
print('RF ROC-AUC:', roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]))


# In[29]:


test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['ActualChurn']   = y_test.values
test_df['Prob_Churn']    = best_rf.predict_proba(X_test)[:,1]
test_df['PredictedChurn']= (test_df['Prob_Churn'] > 0.5).astype(int)


# In[30]:


test_df.to_csv('churn_predictions_export.csv', index=False)
print("Exported  test data for dashboarding.")


# In[32]:



mask = test_df['Prob_Churn'] > 0.5
at_risk = test_df[mask]


at_risk_revenue = (at_risk['MonthlyCharges'] * at_risk['LoyaltyScore']).sum()
print(f"Projected at-risk revenue: ${at_risk_revenue:,.0f}")


saved = at_risk_revenue * 0.15
print(f"Estimated annual savings (15% retention): ${saved:,.0f}")


# In[ ]:




