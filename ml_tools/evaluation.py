import pandas as pd
from sklearn.metrics import log_loss


def multi_class_eval(y_p_df, y_t_df, metric_fn=log_loss, metric_name='metric'):

    # Make sure all the prediction columns are in the true df
    assert(len(set(y_p_df.columns) & set(y_t_df.columns)) ==
           len(set(y_p_df.columns)))

    assert(y_p_df.shape[0] == y_t_df.shape[0])

    results = dict()

    for cur_class in y_p_df.columns:

        cur_y_p = y_p_df[cur_class].values
        cur_y_t = y_t_df[cur_class].values

        cur_metric = metric_fn(cur_y_t, cur_y_p)
        results[cur_class] = cur_metric

    return pd.Series(results, name=metric_name)
