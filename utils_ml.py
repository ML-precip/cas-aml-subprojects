from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def split_data(df, yy_train, yy_test, attributes, ylabel):
    """"Split the data into train and test
         df is the data\n",
         attributes are the covariates,
         ylabel is the target variable"""
    train_dataset = df[(df.date.dt.year >= yy_train[0]) & (df.date.dt.year <= yy_train[1])]
    test_dataset = df[(df.date.dt.year >= yy_test[0]) & (df.date.dt.year <= yy_test[1])]
    # extract the dates for each datasets
    train_dates = train_dataset['date']
    test_dates = test_dataset['date']
    # extract labels
    train_labels = train_dataset[ylabel].copy()
    test_labels = test_dataset[ylabel].copy()
    # extract predictors\n",
    train_dataset = train_dataset[attributes]
    test_dataset = test_dataset[attributes]

    return(train_dataset, train_labels, test_dataset, test_labels, train_dates, test_dates)


def prepareData(dd):
    
    num_attribs = list(dd)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
    ])

    df_prepared = full_pipeline.fit_transform(dd)
    
    return(df_prepared)




