import numpy as np
import pandas as pd
from colorama import Fore
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prob2label(proba):
    proba = np.array(proba).reshape(proba.shape[0], -1)
    if proba.shape[1] == 1:
        return (proba > 0.5).astype('int')
    else:
        return proba.argmax(axis=1)


_score_name2func = {
    'acc': lambda t, p: accuracy_score(t, prob2label(p)),  # label, label
    'auc': roc_auc_score,  # (bcls)label, proba
    'ovr-auc': lambda t, p: roc_auc_score(t, p, multi_class='ovr'),  # (mcls，对类别不平衡比较敏感)label, proba
    'ovo-auc': lambda t, p: roc_auc_score(t, p, multi_class='ovo'),  # (mcls，对类别不平衡不敏感)label, proba
    'f1': f1_score,  # (bcls)label, label
    'micro-f1': lambda t, p: f1_score(t, prob2label(p), average='micro'),  # (mcls，计算总体 TP，再计算 F1)label, label
    'macro-f1': lambda t, p: f1_score(t, prob2label(p), average='macro'),  # (mcls，各类别 F1 的权重相同)label, label
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': lambda t, p: mean_squared_error(t, p)**0.5,
    'r2': r2_score,
}


def _get_score_names_funcs(score_names='', score_funcs=None):
    # 如果不提供 score_names, score_funcs
    if score_names == '' and score_funcs is None:
        score_names = ['acc']
        score_funcs = [_score_name2func['acc']]
    # 如果只提供 score_names
    elif score_names != '' and score_funcs is None:
        score_names = score_names.replace(' ', '').split(',')
        score_funcs = [_score_name2func[name] for name in score_names]
    # 如果只提供 score_funcs
    elif score_names == '' and score_funcs is not None:
        score_names = [func.__name__ for func in score_funcs]
    # 如果同时提供 score_names, score_funcs
    else:
        n1 = score_names.replace(' ', '').split(',')
        n2 = [func.__name__ for func in score_funcs]
        f1 = [_score_name2func[name] for name in n1]
        f2 = score_funcs
        score_names = n1 + n2
        score_funcs = f1 + f2

    return score_names, score_funcs


def lgb_reg_hold_out(
    train: pd.DataFrame, test: pd.DataFrame, ALL_FEAS: list, LABEL: str, NUM_FEAS: list, CAT_FEAS: list,
    valid: pd.DataFrame = None, IDX_COLS: list = None,
    test_size=0.25, incremental_training=True, incremental_training_rate=0.2,
    params=None, sub_params=None, epochs=15000, eval_epoch=100, early_stopping_rounds=500, eval_metric=None, feval=None,
    hold_out_score_names='', hold_out_score_funcs=None, seed=42,
    use_gpu=False, use_native_api=True,
    impt_type='gain', save_impt=False, impt_file='',
    return_train_pred=False, record_time=True,
):
    '''

    Args:
        train:
        test:
        valid:

        ALL_FEAS:
        LABEL:
        NUM_FEAS:
        CAT_FEAS:
        IDX_COLS:

        incremental_training
        incremental_training_rate

        eval_metric: 验证指标
        feval: (only lgb.train) 验证指标&损失函数

        params:
        sub_param: 更新 params 中的参数
        epochs:
        eval_epoch:
        early_stopping_rounds:

        impt_type: 特征重要性类型
        save_impt: 是否保存特征重要性 df
        impt_file: 保存特征重要性文件名(csv 格式)
    '''
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from time import time
    from sklearn.model_selection import train_test_split

    if valid is not None:
        Xtrain = train[ALL_FEAS]  # (n_train, m)
        ytrain = train[IDX_COLS+[LABEL]]  # (n_train, n_idx+)
        Xvalid = valid[ALL_FEAS]  # (n_valid, m)
        yvalid = valid[IDX_COLS+[LABEL]]  # (n_valid, n_idx+1)
        Xtrain_valid = pd.concat((Xtrain, Xvalid), axis=0, ignore_index=True)  # (n_train+n_valid, m)
        ytrain_valid = pd.concat((ytrain, yvalid), axis=0, ignore_index=True)  # (n_train+n_valid, 1)
        get_train_matrix = lambda: lgb.Dataset(Xtrain, label=ytrain[LABEL])
        get_valid_matrix = lambda: lgb.Dataset(Xvalid, label=yvalid[LABEL])
        train_valid_matrix = lgb.Dataset(Xtrain_valid, label=ytrain_valid[LABEL])
        Xtest = test[ALL_FEAS]  # (n_test, m)
    else:
        Xtrain_valid = train[ALL_FEAS]  # (n_train+n_valid, m)
        ytrain_valid = train[[LABEL]]  # (n_train+n_valid, 1)
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xtrain_valid, ytrain_valid, test_size=test_size, random_state=seed)
        # (n_train, m)， (n_valid, m)， (n_train, 1)， (n_valid, 1)
        get_train_matrix = lambda: lgb.Dataset(Xtrain, label=ytrain[LABEL])
        get_valid_matrix = lambda: lgb.Dataset(Xvalid, label=yvalid[LABEL])
        train_valid_matrix = lgb.Dataset(Xtrain_valid, label=ytrain_valid[LABEL])
        Xtest = test[ALL_FEAS]  # (n_test, m)

    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',

            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_boost_round': 100,  # n_estimators
            'min_split_gain': 0,
            'min_child_samples': 20,  # min_data_in_leaf
            'min_child_weight': 1e-3,  # min_sum_hessian_in_leaf

            'max_depth': -1,
            'num_leaves': 63,
            'bagging_fraction': 0.8,
            'bagging_fraction_seed': seed,
            'feature_fraction': 0.8,
            'feature_fraction_seed': seed,
            'reg_lambda': 5,  # lambda_l2
            'reg_alpha': 2,  # lambda_l1

            'num_threads': -1,
            'verbose': -1,
        }
        _params = {
            'objective': 'regression',
            'metric': 'rmse',

            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_boost_round': 100,  # n_estimators
            'min_split_gain': 0,
            'min_child_samples': 20,  # min_data_in_leaf
            'min_child_weight': 5,  # min_sum_hessian_in_leaf

            'max_depth': -1,
            'num_leaves': 128,
            'bagging_freq': 4,
            'bagging_fraction': 0.8,
            'bagging_fraction_seed': seed,
            'feature_fraction': 0.8,
            'feature_fraction_seed': seed,
            'reg_lambda': 10,  # lambda_l2
            'reg_alpha': 2,  # lambda_l1

            'num_threads': -1,
            'verbose': -1,
        }
    if sub_params is not None:
        for k in sub_params:
            params[k] = sub_params[k]
    if eval_metric is not None:
        params['metric'] = eval_metric
    params['num_iterations'] = epochs  # num_iterations 优先级比 num_boost_round 高
    params['categorical_feature'] = ','.join(map(str, pd.Series(ALL_FEAS)[pd.Series(ALL_FEAS).isin(CAT_FEAS)].index))

    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        # params['gpu_device_id'] = 0

    hold_out_score_names, hold_out_score_funcs = _get_score_names_funcs(hold_out_score_names, hold_out_score_funcs)

    ytrain_pred = ytrain.copy()
    yvalid_pred = yvalid.copy()
    ytest_pred = np.zeros(len(Xtest))

    callbacks = [lgb.log_evaluation(period=eval_epoch),
                 lgb.early_stopping(stopping_rounds=early_stopping_rounds)]

    t0 = time()
    if use_native_api:
        model = lgb.train(params=params,
                          train_set=get_train_matrix(),
                          valid_sets=[get_train_matrix(), get_valid_matrix()],
                          feval=feval,
                          # verbose_eval=eval_epoch,
                          # early_stopping_rounds=early_stopping_rounds,
                          callbacks=callbacks)
        best_iter = model.best_iteration
#         ytrain_pred[list(trn_x.index)] = model.predict(trn_x, num_iteration=best_iter)
#         ytrain_pred[list(val_x.index)] = model.predict(val_x, num_iteration=best_iter)
        ytrain_pred[LABEL] = model.predict(Xtrain, num_iteration=best_iter)
        yvalid_pred[LABEL] = model.predict(Xvalid, num_iteration=best_iter)
        ytest_pred = model.predict(Xtest, num_iteration=best_iter)
        feature_importance = pd.DataFrame(data=model.feature_importance(importance_type=impt_type),
                                          index=model.feature_name(),
                                          columns=["importance"])
        # incremental training
        if incremental_training:
            model.save_model('model-lgb.txt')
            params['num_iterations'] = int(best_iter * incremental_training_rate)
            print(f"{Fore.CYAN}[info]{Fore.RESET} incremental training, best iteration {best_iter}, continue training {params['num_iterations']}")

            model = lgb.train(params=params,
                              train_set=train_valid_matrix,
                              valid_sets=[get_train_matrix(), get_valid_matrix()],
                              feval=feval,
                              # verbose_eval=eval_epoch,
                              # early_stopping_rounds=early_stopping_rounds,
                              callbacks=callbacks,
                              init_model='model-lgb.txt')
            ytest_pred = model.predict(Xtest, num_iteration=model.best_iteration)
            feature_importance = pd.DataFrame(data=model.feature_importance(importance_type=impt_type),
                                              index=model.feature_name(),
                                              columns=["importance"])
    else:
        model = lgb.LGBMRegressor(**params).fit(
            X=Xtrain,
            y=ytrain[LABEL],
            eval_names='valid',
            eval_set=(Xvalid, yvalid[LABEL]),
            callbacks=callbacks)
        best_iter = model.best_iteration_
        ytrain_pred[LABEL] = model.predict(Xtrain, num_iteration=best_iter)
        yvalid_pred[LABEL] = model.predict(Xvalid, num_iteration=best_iter)
        ytest_pred = model.predict(Xtest, num_iteration=best_iter)
        feature_importance = pd.DataFrame(data=model.feature_importances_,
                                          index=model.feature_name_,
                                          columns=["importance"])
        # incremental training
        if incremental_training:
            model.booster_.save_model('model-lgb.txt')
            params['num_iterations'] = int(best_iter * incremental_training_rate)
            print(f"{Fore.CYAN}[info]{Fore.RESET} incremental training, best iteration {best_iter}, continue training {params['num_iterations']}")

            model = lgb.LGBMRegressor(**params).fit(
                X=Xtrain_valid,
                y=ytrain_valid[LABEL],
                eval_names='valid',
                eval_set=(Xvalid, yvalid[LABEL]),
                callbacks=callbacks,
                init_model='model-lgb.txt')
            ytest_pred = model.predict(Xtest, num_iteration=model.best_iteration_)
            feature_importance = pd.DataFrame(data=model.feature_importances_,
                                              index=model.feature_name_,
                                              columns=["importance"])

    train_scores = []
    valid_scores = []
    for hold_out_score_name, hold_out_score_func in zip(hold_out_score_names, hold_out_score_funcs):
        print(f'{Fore.BLUE}train {hold_out_score_name}{Fore.RESET}: ', end='')
        train_scores.append(hold_out_score_func(ytrain, ytrain_pred))
        print(f'{Fore.BLUE}valid {hold_out_score_name}{Fore.RESET}: ', end='')
        valid_scores.append(hold_out_score_func(yvalid, yvalid_pred))
    train_scores = dict(zip(hold_out_score_names, train_scores))
    valid_scores = dict(zip(hold_out_score_names, valid_scores))

    t1 = time()
    if record_time: print(f'{Fore.CYAN}[info]{Fore.RESET} train end, cost time {round(t1 - t0, 3)} s')
    else: print(f'{Fore.CYAN}[info]{Fore.RESET} train end')
    print(f'{Fore.RED}train scores{Fore.RESET}:', train_scores)
    print(f'{Fore.RED}valid scores{Fore.RESET}:', valid_scores)

    feature_importance = feature_importance.sort_values(
        by='importance', ascending=False).reset_index().rename(columns={'index': 'feature'})
    if save_impt:
        impt_file = './feature_importance-lgb-hold_out.csv' if impt_file == '' else impt_file
        feature_importance.to_csv(impt_file)

    ret_res = {
        'model': model,
        'ytest_pred': ytest_pred,
        'ytrain_pred': ytrain_pred if return_train_pred else None,
        'feature_importance': feature_importance,
        'valid_scores': valid_scores,
    }
    return ret_res
