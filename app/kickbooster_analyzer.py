import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import random

CATEGORICAL_FEATURES = [
'country', 'location.country', 'location.slug', 'location.state', 'location.type', 'category.name', 
'category.parent', 'creator.name', 'currency'
]

STRING_FEATURES = ['blurb_word_lists', 'name_word_lists']
STRING_COL = ['blurb', 'name']
WORDCOUNT_FEATURES = ['blurb_word_count', 'name_word_count']
LOCATION_FEATURES = ['country', 'location.country']
SELECTED_FEATURES = ['category.parent', 'duration', 'goal'] + LOCATION_FEATURES + WORDCOUNT_FEATURES
COLUMN_TO_DROPNA = ['creator.name', 'goal', 'blurb', 'name'] + LOCATION_FEATURES
# REMOVED_SELECTED_FEATURES = ['currency', 'location.slug', 'location.state', 'location.type', 'creator.name']
FEATURES_WITH_CROSSES = SELECTED_FEATURES + ['category.parent_AND_goal', 'category.parent_AND_country', 'goal_AND_duration', 'category.parent_AND_goal_AND_duration']
LABEL_COLUMN = 'state'

def readJsonFile(filename):
  with open(filename, 'r') as f:
    return json.load(f)

def createTrainingTestSets(df, training_set_ratio):
  train_rows = random.sample(df.index, int(len(df.index) * training_set_ratio))
  df_train = df.iloc[train_rows].reset_index(drop=True)
  df_test = df.drop(train_rows).reset_index(drop=True)
  return df_train, df_test

def trainAndEval(model, X_train, y_train, X_test, y_test):
  model.fit(X_train, y_train)
  expected = y_test
  predicted = model.predict(X_test)
  # prob = model.predict_proba(X_test)
  # print prob
  print(metrics.classification_report(expected, predicted))
  print(metrics.confusion_matrix(expected, predicted))
  return model

def blurbWordCount(row):
  return len(row['blurb'].split())

def nameWordCount(row):
  return len(row['name'].split())

def getCategoryParent(row):
  return row['category.slug'].split('/')[0]

def createTestData(df_test, features, dic_vec, blurb_vec, name_vec):
  dic_test = df_test[features].to_dict('record')
  array_test = dic_vec.transform(dic_test).toarray()
  X_test = pd.DataFrame(array_test, columns=dic_vec.get_feature_names())
  print 'X_test_categorical: ', X_test.shape
  # Add string features of 'blurb' and 'name' to testing set
  blurb_test = blurb_vec.transform(df_test['blurb']).toarray() # TODO: Debug
  name_test = name_vec.transform(df_test['name']).toarray()
  X_test_blurb = pd.DataFrame(blurb_test, columns=blurb_vec.get_feature_names())
  X_test_name = pd.DataFrame(name_test, columns=name_vec.get_feature_names())
  X_test = pd.concat([X_test, X_test_blurb, X_test_name], axis=1)
  print 'X_test_categorical+text: ', X_test.shape
  return X_test

def discretizeColTrain(df, col, num_buckets):
  percentile_values = np.percentile(df[col], range(100/num_buckets, 100, 100/num_buckets))
  digitized_col = np.digitize(df[col], percentile_values)
  df[col] = digitized_col
  df[col] = df[col].astype(str)
  return percentile_values

def discretizeColTest(df, col, percentile_values_from_train):
  digitized_col = np.digitize(df[col], percentile_values_from_train)
  df[col] = digitized_col
  df[col] = df[col].astype(str)

def addCross(df, cols):
  if len(cols) < 2:
    return
  crossed_col_name = '_AND_'.join(cols)
  for i, col in enumerate(cols):
    if i == 0:
      df[crossed_col_name] = df[col].astype(str)
    else:
      df[crossed_col_name] = df[crossed_col_name] + '_AND_' + df[col].astype(str)



def main():
  project_sets = readJsonFile('../../data/Kickstarter_Kickstarter_20150402.json')
  print 'Data read'
  projects = []
  for project_set in project_sets:
    projects.extend(project_set['projects'])
  
  df = json_normalize(projects)
  print 'Data normalized'
  df.to_csv('../../data/Kickstarter_Kickstarter_20150402_normalized.csv', encoding='utf-8')

  df = pd.read_csv('../../data/Kickstarter_Kickstarter_20150402_normalized.csv', encoding='utf-8')
  # print df.dtypes  
  print 'df dimensions: ', df.shape
  df.dropna(subset=COLUMN_TO_DROPNA, inplace=True)
  print 'df dimensions na dropped: ', df.shape

  df['duration'] = (df['deadline'] - df['launched_at']) / 86400

  ## Add wordcount for 'blurb' and 'name' columns
  df['blurb_word_count'] = df.apply(blurbWordCount, axis=1)
  df['name_word_count'] = df.apply(nameWordCount, axis=1)
  print 'Duration, word count added'

  # # Add catogory.parent column
  df['category.parent'] = df.apply(getCategoryParent, axis=1)
  print df['category.parent'].head()  

  # print 'Blurb with null string: ', sum(df['blurb'].isnull())
  # print 'Name with null string: ', sum(df['name'].isnull())

  
  # # ## Write dataframe to csv
  df.to_csv('../../data/Kickstarter_20150402_dropna.csv', encoding='utf-8')
  
  # ## Extract random samples
  df.sample(n=1000).to_csv('../../data/project_rand_samples_1000.csv', encoding='utf-8')
  df.sample(n=10000).to_csv('../../data/project_rand_samples_10000.csv', encoding='utf-8')

  ## Read csv file
  df_full = pd.read_csv('../../data/Kickstarter_20150402_dropna.csv')
  print df_full[df_full['name']=='Vending Machine (Canceled)']
  print 'df dimensions: ', df_full.shape
  df_full.dropna(subset=COLUMN_TO_REMOVE, inplace=True)
  print 'df dimensions na dropped: ', df_full.shape

  df_1000 = pd.read_csv('../../data/project_rand_samples_1000.csv')
  df_10000 = pd.read_csv('../../data/project_rand_samples_10000.csv')
  
  ## Drop 'live' projects from the dataframe and reset the index
  df_full = df_full[df_full['state'] != 'live'].reset_index(drop=True)
  df_1000 = df_1000[df_1000['state'] != 'live'].reset_index(drop=True)
  df_10000 = df_10000[df_10000['state'] != 'live'].reset_index(drop=True)

  ## Randomly split data into training and testing sets
  df_train_full, df_test_full = createTrainingTestSets(df_full, 0.7)
  df_train_1000, df_test_1000 = createTrainingTestSets(df_1000, 0.7)
  df_train_10000, df_test_10000 = createTrainingTestSets(df_10000, 0.7)

  ## Write into csv files
  df_train_full.to_csv('../../data/df_train_full.csv', encoding='utf-8')
  df_test_full.to_csv('../../data/df_test_full.csv', encoding='utf-8')
  df_train_1000.to_csv('../../data/df_train_1000.csv', encoding='utf-8')
  df_test_1000.to_csv('../../data/df_test_1000.csv', encoding='utf-8')
  df_train_10000.to_csv('../../data/df_train_10000.csv', encoding='utf-8')
  df_test_10000.to_csv('../../data/df_test_10000.csv', encoding='utf-8')

  ###############################################################
  # # # Read training and testing sets
  df_train = pd.read_csv('../../data/df_train_full.csv', index_col=0)
  df_test = pd.read_csv('../../data/df_test_full.csv', index_col=0)
  # print 'Data Read'

  print 'df_train dimensions: ', df_train.shape
  # print 'df_test dimensions: ', df_test.shape  
  df_train.dropna(subset=SELECTED_FEATURES, inplace=True)
  df_test.dropna(subset=SELECTED_FEATURES, inplace=True)
  # print 'df_train dimensions na dropped: ', df_train.shape
  print 'df_test dimensions na dropped: ', df_test.shape

  # Wrtie first 10 testing data as templates
  df_test_full_top10 = df_test.iloc[0:9].reset_index(drop=True)
  df_test_full_top10.to_csv('../../data/df_test_m.csv', encoding='utf-8')
  print 'Write df_test_top10 to csv'
  
  ### Create training data ####
  # # Discretize training set
  percentile_values_goal = discretizeColTrain(df_train, 'goal', 100)
  # print 'Replace goal with percentile_values'
  
  with open('../../data/vectorizer/percentile_m_train_goal.json', 'w') as f:
        json.dump(percentile_values_goal, f)

  # Add feature crosses to training set
  addCross(df_train, ['category.parent', 'goal'])
  addCross(df_train, ['category.parent', 'country'])
  addCross(df_train, ['goal', 'duration'])
  addCross(df_train, ['category.parent', 'goal', 'duration'])
  print 'df_train dimensions after adding feature crosses: ', df_train.shape

  # Encode categorical features from training set using dic_vec
  dic_train = df_train[FEATURES_WITH_CROSSES].to_dict('record')
  dic_vec = DictVectorizer()
  array_train = dic_vec.fit_transform(dic_train).toarray()
  X_train = pd.DataFrame(array_train, columns=dic_vec.get_feature_names())
  print 'dic_vec transformation for training set'
  print 'X_train dimensions: ', X_train.shape
   
  # Add string features of 'blurb' and 'name' to training set
  blurb_vec = TfidfVectorizer(min_df=0.01, stop_words='english')
  name_vec =  TfidfVectorizer(min_df=0.01, stop_words='english')
  blurb_train = blurb_vec.fit_transform(df_train['blurb']).toarray()
  name_train = name_vec.fit_transform(df_train['name']).toarray()
  X_train_blurb = pd.DataFrame(blurb_train, columns=blurb_vec.get_feature_names())
  X_train_name = pd.DataFrame(name_train, columns=name_vec.get_feature_names())
  X_train = pd.concat([X_train, X_train_blurb, X_train_name], axis=1)
  print 'Added string features to training'
  
  ## Discretize testing set
  print 'df_test dimensions before discretization: ', df_test.shape
  discretizeColTest(df_test, 'goal', percentile_values_goal)
  print 'df_test dimensions after discretization: ', df_test.shape

  # Add feature crosses
  addCross(df_test, ['category.parent', 'goal'])
  addCross(df_test, ['category.parent', 'country'])
  addCross(df_test, ['goal', 'duration'])
  addCross(df_test, ['category.parent', 'goal', 'duration'])
  print 'df_test dimensions after adding feature crooses: ', df_test.shape
  
  X_test = createTestData(df_test, FEATURES_WITH_CROSSES, dic_vec, blurb_vec, name_vec)
  # X_test_top10 = X_test.iloc[0:9].reset_index(drop=True)
  # X_test_top10.to_csv('../../data/X_test_full_top10_discretized_20.csv', encoding='utf-8')
  print 'X_train dimensions: ', X_train.shape
  print 'X_test dimensions: ', X_test.shape
  
  ## Set up label for training and testing sets
  y_train = (df_train['state'] == 'successful')
  y_train = y_train.reset_index(drop=True)
  y_test = (df_test['state'] == 'successful')
  y_test = y_test.reset_index(drop=True)
  # print sum(y_test)*1.0/len(y_test)
  
  # # ## Train a Decision Tree classifier and evaluate on test data.
  # # # decisiontree_model = trainAndEval(model=DecisionTreeClassifier(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  logistic_resgression_model = trainAndEval(model=LogisticRegression(penalty='l2', C=1.0), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  # # # randomforest_model = trainAndEval(model=RandomForestClassifier(max_depth=10, n_estimators=100), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

  
  # ## Export vectorizers and models
  joblib.dump(dic_vec, '../../data/vectorizer/kickbooster_dic_vec_m.pkl')
  joblib.dump(blurb_vec, '../../data/vectorizer/kickbooster_blurb_vec_m.pkl')
  joblib.dump(name_vec, '../../data/vectorizer/kickbooster_name_vec_m.pkl')
  joblib.dump(logistic_resgression_model, '../../data/classifier/kickbooster_lr_m.pkl')
  

  
  pass



if __name__ == '__main__':
  main()