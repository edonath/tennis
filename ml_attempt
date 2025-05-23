import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, r2_score
import xgboost as xgb
import seaborn as sns
import hyperopt as hp
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

df = pd.read_csv('/content/drive/MyDrive/df_tennis.csv')
df.info()

df.head()

df.columns

# Let's focus on my own chosen subset of features...
feat_1 = ['GENDER','TOURNAMENT SIZE','SURFACE','GROUND','MONEYLINE US_FAV','RANK_DIFF']

X=df.loc[:,feat_1]
y = df['TOTAL GAMES_FAV'] 

X_train_full, X_test_full, y_train, y_test = train_test_split(X,y,test_size = 400, random_state=0)


# Get the mean and SD of our target variable
#This SD is what we're gonna be aiming for
np.mean(df['TOTAL GAMES_FAV']), np.min(df['TOTAL GAMES_FAV']), np.max(df['TOTAL GAMES_FAV']), np.std(df['TOTAL GAMES_FAV'])

# Make a histogram
plt.hist(df['TOTAL GAMES_FAV'])
plt.show()

# Redo the histogram with "nicer" bins
binpts = np.linspace(10,35)
plt.hist(df['TOTAL GAMES_FAV'], bins=binpts);

# Pairplot 
sns.pairplot(df.loc[:,feat_1+['TOTAL GAMES_FAV']], plot_kws={'alpha':.1});

# MONEYLINE is hard to see due to outliers - let's plot it again here
#DO see fewer games when the favorite is more heavily favored
plt.scatter(df['MONEYLINE US_FAV'], df['TOTAL GAMES_FAV'], alpha=.1)

# Convert 'GENDER', 'GROUND', and 'SURFACE' columns to numerical representations
X_train_full['GENDER'] = X_train_full['GENDER'].map({'m': 0, 'f': 1})  # Map 'm' to 0 and 'f' to 1 in 'GENDER'
X_train_full['GROUND'] = X_train_full['GROUND'].map({'h': 0, 'm': 1})  # Map 'h' to 0 and 'm' to 1 in 'GROUND'
X_train_full['SURFACE'] = X_train_full['SURFACE'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})  # Map surface types to numerical values
X_test_full['GENDER'] = X_test_full['GENDER'].map({'m': 0, 'f': 1})  # Do the same for your test set for 'GENDER'
X_test_full['GROUND'] = X_test_full['GROUND'].map({'h': 0, 'm': 1})  # Do the same for your test set for 'GROUND'
X_test_full['SURFACE'] = X_test_full['SURFACE'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})  # Do the same for your test set for 'SURFACE'



# Create / train Random Forest on just those few variables
rf0 = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rf0.fit(X_train_full, y_train)

#Make predictions on test set
preds_rf0 = rf0.predict(X_test_full)

#We improved our SD slightly
np.sqrt(mean_squared_error(y_test, preds_rf0)), mean_absolute_error(y_test, preds_rf0), r2_score(y_test, preds_rf0)

# Plot predicted (y) vs actual (x)
plt.scatter(x=y_test, y=preds_rf0, alpha=.2, marker='.')
plt.plot([0,40],[0,40], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted');

# THen let's try XGBoost model with default parameters
xgb_def = xgb.XGBRegressor()
xgb_def

xgb_def.fit(X_train_full, y_train)

preds_xgb_def = xgb_def.predict(X_test_full)

np.sqrt(mean_squared_error(y_test, preds_xgb_def)), mean_absolute_error(y_test, preds_xgb_def), r2_score(y_test, preds_xgb_def)

# Add trees until we show no improvement (i.e. no new low) for 10 trees
xgb1 = xgb.XGBRegressor(n_estimators=5000, learning_rate=.01, early_stopping_rounds = 10)


xgb1.fit(X_train_full, y_train, 
         eval_set=[(X_test_full, y_test)])

preds_xgb1 = xgb1.predict(X_test_full)

#XGboost always makes things a bit better
np.sqrt(mean_squared_error(y_test, preds_xgb1)), mean_absolute_error(y_test, preds_xgb1), r2_score(y_test, preds_xgb1)

# Now let's tune the max depth

md_vals_vec=list(range(1,10))
rmse_vec = np.zeros(len(md_vals_vec))
for i,md in enumerate(md_vals_vec):
    print(f'Training with max_depth {md}')
    xgb_temp = xgb.XGBRegressor(max_depth=md, 
                        n_estimators=5000, learning_rate=.01, 
         early_stopping_rounds = 10) #, early_stopping_rounds=10)
    xgb_temp.fit(X_train_full, y_train, 
         eval_set=[(X_test_full, y_test)], 
                 verbose=0)
    preds = xgb_temp.predict(X_test_full)
    rmse_vec[i] = np.sqrt(mean_squared_error(y_test, preds))
  

# Plot performance vs. max_depth
plt.plot(md_vals_vec, rmse_vec, marker='x')
np.min(rmse_vec)

xgb2 = xgb.XGBRegressor(max_depth=7, n_estimators=5000, learning_rate=.01, early_stopping_rounds = 10)


# Add trees until we show no improvement (i.e. no new low) for 10 trees
xgb2.fit(X_train_full, y_train, 
         eval_set=[(X_test_full, y_test)])

preds_xgb2 = xgb2.predict(X_test_full)

np.sqrt(mean_squared_error(y_test, preds_xgb2)), mean_absolute_error(y_test, preds_xgb2), r2_score(y_test, preds_xgb2)

np.sqrt(mean_squared_error(y_test, preds_xgb2))/np.sqrt(mean_squared_error(y_test, preds_xgb1))


# We calculate the residuals - discrepancies between true answer and prediction
# Positive residual => model underpredicted true value
# Negative residual => model overpredicted true value
resids = (y_test - preds_xgb2)
abs_resids = np.abs(resids)

plt.hist(resids);


# Let's focus on a different subset of features...
feat_1 = ['GENDER','SURFACE','MONEYLINE US_FAV','RANK_DIFF','TOTAL POINTS_1MATCH_UND','TOTAL POINTS_1MATCH_FAV']
X=df.loc[:,feat_1]
y = df['TOTAL GAMES_FAV'] 
X_train_full, X_test_full, y_train, y_test = train_test_split(X,y,test_size = 400, random_state=0)

# do a pairplot 
sns.pairplot(df.loc[:,feat_1+['TOTAL GAMES_FAV']], plot_kws={'alpha':.1});

X_train_full['GENDER'] = X_train_full['GENDER'].map({'m': 0, 'f': 1})  # Map 'm' to 0 and 'f' to 1 in 'GENDER'
X_train_full['SURFACE'] = X_train_full['SURFACE'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})  # Map surface types to numerical values
X_test_full['GENDER'] = X_test_full['GENDER'].map({'m': 0, 'f': 1})  # Do the same for your test set for 'GENDER'
X_test_full['SURFACE'] = X_test_full['SURFACE'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})  # Do the same for your test set for 'SURFACE'

rf0 = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rf0.fit(X_train_full, y_train)

preds_rf0 = rf0.predict(X_test_full)

np.sqrt(mean_squared_error(y_test, preds_rf0)), mean_absolute_error(y_test, preds_rf0), r2_score(y_test, preds_rf0)

# Plot predicted (y) vs actual (x)
plt.scatter(x=y_test, y=preds_rf0, alpha=.2, marker='.')
plt.plot([0,40],[0,40], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted');

# Let's focus on a different subset of features...
feat_1 = ['GENDER','SURFACE','MONEYLINE US_FAV','RANK_DIFF','TOTAL POINTS_1MATCH_UND','TOTAL POINTS_1MATCH_FAV','TOTAL POINTS_3MATCH_UND','TOTAL POINTS_3MATCH_FAV','TOTAL POINTS_6MATCH_UND','TOTAL POINTS_6MATCH_FAV','TOTAL POINTS_9MATCH_UND','TOTAL POINTS_9MATCH_FAV']
X=df.loc[:,feat_1]
y = df['TOTAL GAMES_FAV'] 
X_train_full, X_test_full, y_train, y_test = train_test_split(X,y,test_size = 400, random_state=0)

# do a pairplot 
sns.pairplot(df.loc[:,feat_1+['TOTAL GAMES_FAV']], plot_kws={'alpha':.1});

# Convert 'GENDER', 'GROUND', and 'SURFACE' columns to numerical representations
X_train_full['GENDER'] = X_train_full['GENDER'].map({'m': 0, 'f': 1})  # Map 'm' to 0 and 'f' to 1 in 'GENDER'
X_train_full['SURFACE'] = X_train_full['SURFACE'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})  # Map surface types to numerical values
X_test_full['GENDER'] = X_test_full['GENDER'].map({'m': 0, 'f': 1})  # Do the same for your test set for 'GENDER'
X_test_full['SURFACE'] = X_test_full['SURFACE'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})  # Do the same for your test set for 'SURFACE'



# Create / train Random Forest on just 5 variables
rf0 = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rf0.fit(X_train_full, y_train)

preds_rf0 = rf0.predict(X_test_full)
np.sqrt(mean_squared_error(y_test, preds_rf0)), mean_absolute_error(y_test, preds_rf0), r2_score(y_test, preds_rf0)

# Plot predicted (y) vs actual (x)
plt.scatter(x=y_test, y=preds_rf0, alpha=.2, marker='.')
plt.plot([0,40],[0,40], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted');

preds_rf0 = rf0.predict(X_train_full)
np.sqrt(mean_squared_error(y_train, preds_rf0)), mean_absolute_error(y_train, preds_rf0), r2_score(y_train, preds_rf0)

# Plot predicted (y) vs actual (x)
plt.scatter(x=y_train, y=preds_rf0, alpha=.2, marker='.')
plt.plot([0,40],[0,40], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted');

