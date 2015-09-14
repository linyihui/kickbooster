import json
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import random

CATEGORICAL_FEATURES = ['country', 'location.country', 'location.slug', 'location.state', 'location.type', 'category.name', 'creator.name', 'currency']
LOCATION_FEATURES = ['country', 'location.country', 'location.slug', 'location.state', 'location.type']
SELECTED_FEATURES = ['category.name', 'creator.name', 'currency', 'duration', 'goal'] + LOCATION_FEATURES
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
  # Summarize the fit of the model.
  print(metrics.classification_report(expected, predicted))
  print(metrics.confusion_matrix(expected, predicted))
  return model

def main():
  # project_sets = readJsonFile("Kickstarter_Kickstarter_20150402.json")
  # projects = []
  # for project_set in project_sets:
  #   # projects.extend(project_set['projects'])
  
  # df = json_normalize(projects)
  # df['duration'] = (df['deadline'] - df['launched_at']) / 86400
  # df.sample(n=1000).to_csv('project_rand_samples_1000.csv', encoding='utf-8')
  
  # df.head(100).to_csv('project_samples_100.csv', encoding='utf-8')
  # print df.columns.values.tolist()
  ######

  df = pd.read_csv('project_rand_samples_1000.csv')
  df = df[df['state'] != 'live'].reset_index()

  
  for f in CATEGORICAL_FEATURES:
    df.loc[df[f].isnull(), f] = 'UNKNOWN'
  
  # print df[(df['location.type'].isnull())]['location.type']

  ####
  df_train, df_test = createTrainingTestSets(df, 0.7)

  dic_train = df_train[SELECTED_FEATURES].to_dict('record')
  vec = DictVectorizer()
  array_train = vec.fit_transform(dic_train).toarray()
  X_train = pd.DataFrame(array_train, columns=vec.get_feature_names())

  dic_test = df_test[SELECTED_FEATURES].to_dict('record')
  array_test = vec.transform(dic_test).toarray()
  X_test = pd.DataFrame(array_test, columns=vec.get_feature_names())

  y_train = (df_train['state'] == 'successful')
  y_test = (df_test['state'] == 'successful')

  ## Train a Decision Tree classifier and evaluate on test data.
  model = trainAndEval(model=DecisionTreeClassifier(max_depth=10), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
  main()