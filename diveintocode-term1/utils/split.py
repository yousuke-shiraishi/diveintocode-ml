def train_test_split(X, y, train_size=0.8,):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """
    import random
    random.seed(0)
    random.shuffle(X)
    ss = train_size * len(x)
    X_train = X[0:ss,:]
    X_test = X[ss:,:]
    
    random.shuffle(y)
    ss = train_size * len(y)
    y_train = y[0:ss,:]
    y_test = y[ss:,:]

    return X_train, X_test, y_train, y_test