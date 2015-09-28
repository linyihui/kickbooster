from flask import render_template, request
from app import app
from random import randint
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import kickbooster_analyzer as ka
import country_dict
import numpy as np

@app.route('/')
# @app.route('/index')
# def index():
#    user = { 'nickname': 'Miguel' } # fake user
#    return render_template("index.html",
#        title = 'Home',
#        user = user)

@app.route('/index')
@app.route('/input')
def kickbooster_input():
  return render_template("input.html")

@app.route('/output')
def kickbooster_output():
  ## Import model and vectorizers
  model = joblib.load('../data/classifier/kickbooster_lr_m.pkl')
  dic_vec = joblib.load('../data/vectorizer/kickbooster_dic_vec_m.pkl')
  blurb_vec = joblib.load('../data/vectorizer/kickbooster_blurb_vec_m.pkl')
  name_vec = joblib.load('../data/vectorizer/kickbooster_name_vec_m.pkl')
  percentile_values_goal = ka.readJsonFile('../data/vectorizer/percentile_m_train_goal.json')
  
  ## Import test_template
  df_test = pd.read_csv('../data/df_test_m.csv', index_col=0)
  test_template = df_test.iloc[[3]].reset_index(drop=True)
  ## TODO: Keep datatype when slice out a row as template

  ## Import user input info
  project_title = request.args.get('project_title')
  print 'Received Project Title: ', project_title

  short_blurb = request.args.get('short_blurb')
  print 'Received Short Blurb: ', short_blurb

  country = request.args.get('country')
  print 'Received Country: ', country

  category = request.args.get('category')
  print 'Received Category: ', category

  funding_goal = int(request.args.get('funding_goal'))
  print 'Received funding_goal: ', funding_goal, type(funding_goal)
  
  funding_duration = int(request.args.get('funding_duration'))
  print 'Received funding_duration: ', funding_duration

  ## Parse user info to template
  test_template.loc[0, 'name'] = project_title
  test_template.loc[0, 'name_word_count'] = len(project_title.split())

  test_template.loc[0,'blurb'] = short_blurb
  test_template.loc[0,'blurb_word_count'] = len(short_blurb.split())
  
  test_template.loc[0,'country'] = country_dict.country_code[country]
  test_template.loc[0,'location.country'] = country_dict.country_code[country]

  test_template.loc[0,'category.parent'] = category.lower()
  
  test_template.loc[0,'duration'] = funding_duration
  test_template.loc[0,'goal'] = funding_goal
  print test_template[['name', 'blurb', 'country', 'category.parent', 'duration', 'goal']]

  ## Create a test set for tips
  ranges_goal = [1] + np.linspace(0.7, 0.9, 3).tolist() + np.linspace(1.1, 2, 10).tolist() #[1, 1.25, 1.5, 2.0, 0.9, 0.8, 0.7]
  test_set_goal = pd.concat([test_template] * len(ranges_goal)).reset_index(drop=True)
  test_set_goal['goal'] = [int(funding_goal*i) for i in ranges_goal]

  ranges_duration = [funding_duration] + range(10, 101, 5)
  test_set_duration = pd.concat([test_template] * len(ranges_duration)).reset_index(drop=True)
  test_set_duration['duration'] = ranges_duration

  ## Discretization & Feature Crosses
  print 'Dimensions Before Discretization & Feature Crosses: ', test_set_goal.shape
  ka.discretizeColTest(test_set_goal, 'goal', percentile_values_goal)
  ka.addCross(test_set_goal, ['category.parent', 'goal'])
  ka.addCross(test_set_goal, ['category.parent', 'country'])
  ka.addCross(test_set_goal, ['goal', 'duration'])
  ka.addCross(test_set_goal, ['category.parent', 'goal', 'duration'])
  print 'Dimensions After Discretization & Feature Crosses: ', test_set_goal.shape
  print 'Percentile of goal: ', test_set_goal['goal']
  print 'Percentile of duration: ', test_set_goal['duration']

  print 'Dimensions Before Discretization & Feature Crosses: ', test_set_duration.shape
  ka.discretizeColTest(test_set_duration, 'goal', percentile_values_goal)
  ka.addCross(test_set_duration, ['category.parent', 'goal'])
  ka.addCross(test_set_duration, ['category.parent', 'country'])
  ka.addCross(test_set_duration, ['goal', 'duration'])
  ka.addCross(test_set_duration, ['category.parent', 'goal', 'duration'])
  print 'Dimensions After Discretization & Feature Crosses: ', test_set_duration.shape
  print 'Percentile of goal: ', test_set_duration['goal']
  print 'Percentile of duration: ', test_set_duration['duration']

  ## Predictions of goal and message
  X_test_goal = ka.createTestData(test_set_goal, ka.FEATURES_WITH_CROSSES, dic_vec, blurb_vec, name_vec)
  print 'X_test_goal dimension: ', X_test_goal.shape

  predictions_goal = model.predict_proba(X_test_goal)
  predicted_prob = int(round(predictions_goal[0][1], 2)*100)
  max_index_goal = np.argmax(predictions_goal[:,1])
  max_success_goal = int(round(predictions_goal[max_index_goal][1], 2)*100)
  diff_goal = int(round(ranges_goal[max_index_goal] - 1, 2) * 100)
  
  if diff_goal == 0:
    message_goal = 'Your funding goal looks great! KickBoostaaah!'
  elif diff_goal > 0:
    message_goal = 'If you increase your funding goal by %d%% to $%s, you can boost your success rate to %d%%.' % (diff_goal, int(funding_goal * (1+diff_goal/100.0)), max_success_goal)  
  else:
    message_goal = 'If you lower your funding goal by %d%% to $%s, you can boost your success rate to %d%%.' % (abs(diff_goal), int(funding_goal * (1+diff_goal/100.0)), max_success_goal)

  ## Predictions of duration and message
  X_test_duration = ka.createTestData(test_set_duration, ka.FEATURES_WITH_CROSSES, dic_vec, blurb_vec, name_vec)
  print 'X_test_duration dimension: ', X_test_duration.shape

  predictions_duration = model.predict_proba(X_test_duration)
  max_index_duration = np.argmax(predictions_duration[:,1])
  max_success_duration = int(round(predictions_duration[max_index_duration][1], 2)*100)
  diff_duration = ranges_duration[max_index_duration] - funding_duration

  if diff_duration == 0:
    message_duration = 'Your funding duration looks great! KickBoostaaah!'
  elif diff_duration > 0:
    message_duration = 'If you extend your funding duration to %d days, you can boost your success rate to %d%%.' % (ranges_duration[max_index_duration], max_success_duration)
  else:
    message_duration = 'If you shorten your funding duration to %d days, you can boost your success rate to %d%%.' % (ranges_duration[max_index_duration], max_success_duration)

 
  return render_template("output.html", 
    funding_goal_val=funding_goal,
    funding_duration=funding_duration, 
    predicted_prob=predicted_prob,
    message_goal=message_goal,
    message_duration=message_duration)
