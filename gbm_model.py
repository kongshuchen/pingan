import lightgbm as lgb
import pandas as pd




def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target,
                 early_stopping_rounds=20, num_boost_round=3000, verbose_eval=1, categorical_features=None):

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          # categorical_feature=categorical_features
                          )


    evals_results = {}

    bst1 = lgb.train(params=params,
                     train_set=xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval)

    return bst1

