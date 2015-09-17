import json
import pandas as pd
from pandas.io.json import json_normalize
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import random

CATEGORICAL_FEATURES = ['country', 'location.country', 'location.slug', 'location.state', 'location.type', 'category.name', 'creator.name', 'currency']
STRING_FEATURES = ['blurb_word_lists', 'name_word_lists']
WORDCOUNT_FEATURES = ['blurb_word_count', 'name_word_count']
LOCATION_FEATURES = ['country', 'location.country', 'location.slug', 'location.state', 'location.type']
SELECTED_FEATURES = ['category.name', 'creator.name', 'currency', 'duration', 'goal'] + LOCATION_FEATURES + WORDCOUNT_FEATURES
LABEL_COLUMN = 'state'

def readJsonFile(filename):
  with open(filename, 'r') as f:
    return json.load(f)

def createTrainingTestSets(df, training_set_ratio):
  train_rows = random.sample(df.index, int(len(df.index) * training_set_ratio))
  df_train = df.ix[train_rows]
  df_test = df.drop(train_rows)
  return df_train, df_test

def trainAndEval(model, X_train, y_train, X_test, y_test):
  model.fit(X_train, y_train)
  expected = y_test
  predicted = model.predict(X_test)
  predicted_prob = model.predict_proba(X_test)
  print predicted_prob
  # Summarize the fit of the model.
  print(metrics.classification_report(expected, predicted))
  print(metrics.confusion_matrix(expected, predicted))
  return model

def blurbWordCount(row):
  return len(row['blurb'].split())

def nameWordCount(row):
  return len(row['name'].split())


def main():
  # project_sets = readJsonFile("Kickstarter_Kickstarter_20150402.json")
  # projects = []
  # for project_set in project_sets:
  #   projects.extend(project_set['projects'])

  # df = json_normalize(projects)

  # for f in CATEGORICAL_FEATURES:
  #   df.loc[df[f].isnull(), f] = 'UNKNOWN'

  # df['duration'] = (df['deadline'] - df['launched_at']) / 86400
  
  ## Add wordcount for 'blurb' and 'name' columns
  # df['blurb_word_count'] = df.apply(blurbWordCount, axis=1)
  # df['name_word_count'] = df.apply(nameWordCount, axis=1)

  # df.sample(n=10000).to_csv('project_rand_samples_10000.csv', encoding='utf-8')

  ####

  ## Read csv file
  df = pd.read_csv('project_rand_samples_1000_edited.csv')
  
  ## Drop 'live' projects from the dataframe and reset the index
  df = df[df['state'] != 'live'].reset_index()

  ####
  df_train, df_test = createTrainingTestSets(df, 0.7)
  
  ## Encode categorical features from training set using one-hot encoding
  dic_train = df_train[SELECTED_FEATURES].to_dict('record')
  dic_vec = DictVectorizer()
  array_train = dic_vec.fit_transform(dic_train).toarray()
  X_train = pd.DataFrame(array_train, columns=dic_vec.get_feature_names())
  
  ## Add string features of 'blurb' and 'name' to training set
  blurb_vec = TfidfVectorizer(min_df=0.01, stop_words='english')
  name_vec =  TfidfVectorizer(min_df=0.01, stop_words='english')
  blurb_train = blurb_vec.fit_transform(df_train['blurb']).toarray()
  name_train = name_vec.fit_transform(df_train['name']).toarray()
  X_train_blurb = pd.DataFrame(blurb_train, columns=blurb_vec.get_feature_names())
  X_train_name = pd.DataFrame(name_train, columns=name_vec.get_feature_names())
  X_train = pd.concat([X_train, X_train_blurb, X_train_name], axis=1)
  
  
  ## Encode categorical features from testing set using one-hot encoding
  dic_test = df_test[SELECTED_FEATURES].to_dict('record')
  array_test = dic_vec.transform(dic_test).toarray()
  X_test = pd.DataFrame(array_test, columns=dic_vec.get_feature_names())
  
  
  ## Add string features of 'blurb' and 'name' to testing set
  blurb_test = blurb_vec.transform(df_test['blurb']).toarray()
  name_test = name_vec.transform(df_test['name']).toarray()
  X_test_blurb = pd.DataFrame(blurb_test, columns=blurb_vec.get_feature_names())
  X_test_name = pd.DataFrame(name_test, columns=name_vec.get_feature_names())
  X_test = pd.concat([X_test, X_test_blurb, X_test_name], axis=1)

  ## Set up label for training and testing sets
  y_train = (df_train['state'] == 'successful')
  y_train = y_train.reset_index(drop=True)
  y_test = (df_test['state'] == 'successful')
  y_test = y_test.reset_index(drop=True)
  
  X_test_template = pd.DataFrame(index=[0], columns=X_train.columns.values)
  X_test_template.fillna(0, inplace=True)
  X_test_template.to_csv('X_test_template.csv', encoding='utf-8')

  # Train a Decision Tree classifier and evaluate on test data.
  decisiontree_model = trainAndEval(model=DecisionTreeClassifier(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  randomforest_model = trainAndEval(model=RandomForestClassifier(max_depth=10), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

  # Export testing set and classifiers
  X_test.to_csv('X_test_10000.csv', encoding='utf-8')
  y_test.to_csv('y_test_10000.csv', encoding='utf-8')

  joblib.dump(randomforest_model, 'kickbooster_rf_model.pkl')
  

  # TODO:
  # http://scikit-learn.org/stable/modules/model_persistence.html
  # from sklearn.externals import joblib
  # joblib.dump(model, 'saved_model.pkl') 


if __name__ == "__main__":
  main()