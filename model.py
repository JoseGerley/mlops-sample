# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init environment
from pycaret.regression import *
r1 = setup(insurance,
                  target = 'charges',
                  session_id=1,
                  fold_shuffle=True,
                  imputation_type='iterative',
                  train_size = 0.8,
                  transform_target = True,
                  normalize = True, #rescale the values of numeric columns
                  handle_unknown_categorical = True, 
                  unknown_categorical_method = 'most_frequent',
                  remove_multicollinearity = True, #rop one of the two features that are highly correlated with each other
                  ignore_low_variance = True,#all categorical features with statistically insignificant variances are removed from the dataset.
                  combine_rare_levels = True,
                  normalize_method='robust',
                 categorical_features=['sex','smoker', 'region'], #categorical features
                  numeric_features=['age',  'bmi', 'children'])

# train a model
lr = create_model('lr')

# save pipeline/model
save_model(lr,'deployment_28042020')


#  setup(insurance, target = 'charges', session_id = 123,
           
#            normalize = True,
#            imputation_type='iterative',
#            polynomial_features = True, trigonometry_features = True,
#            feature_interaction=True,
#            ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],
#            bin_numeric_features= ['age', 'bmi'])