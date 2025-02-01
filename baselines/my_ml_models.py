# %% [code]
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize, StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.metrics import mean_absolute_error

from warnings import filterwarnings
filterwarnings('ignore')

from tqdm import tqdm
tqdm.pandas()


#    _____________________________________________________
#___/ Tabular Playground Series - Dec 2021                \______________
#

def run_lgbm_cv(n_splits=int(5),params=dict(),train=None,test=None,targets=None):
    train_data = lgb.Dataset(train, label=targets)
    results = lgb.cv(
        params,
        train_data,
        nfold=n_splits
    )

    return results
        

    
def run_lgbm_kfold(n_splits=int(5),params=dict(),train=None,test=None,targets=None):
    valid_accu = list()
    models = list()
    preds  = list()
    kf = KFold(n_splits=n_splits, shuffle=True)
    for n_fold, (train_idx, valid_idx) in enumerate(kf.split(targets)):
        X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval  = lgb.Dataset(X_valid, y_valid,reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_eval,
        )
        y_valid_pred = model.predict(X_valid,num_iteration=model.best_iteration)
        y_valid_pred_max  = np.argmax(y_valid_pred, axis=1)
        accuracy = sum(y_valid == y_valid_pred_max) / len(y_valid)
        print(f'fold {n_fold} accuracy: {accuracy}')
        valid_accu.append(accuracy)
        models.append(model)
        
        pred = model.predict(test,num_iteration=model.best_iteration)
        preds.append(pred)
        

    return valid_accu, models, lgb, preds
    
    
def simple_nn_model_1(X):
    il = layers.Input(shape=(X.shape[-1]), name="input")
    x = layers.Dense(128, activation='selu')(il)
    x1 = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='selu')(x1)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(layers.Concatenate()([x, x1]))
    x = layers.Dense(units=64, activation='relu')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='selu')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(len(le.classes_), activation="softmax", name="output")(x)

    model = tf.keras.Model([il], output)
    return model

def simple_nn_model_2(X):
    model = keras.Sequential([

        # hidden layer 1
        layers.Dense(units=256, activation='relu', input_shape=[X.shape[1]], kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.3),

        # hidden layer 2
        layers.Dense(units=256, activation='relu', kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.3),

        # hidden layer 3
        layers.Dense(units=128, activation='relu', kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.2),
        
        # hidden layer 4
        layers.Dense(units=64, activation='relu', kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.2),

        # output layer
        layers.Dense(units=6, activation='softmax')
    ])

    
    return model

def gres_model(encoding_size,col_len,num_class):
    class GatedLinearUnit(layers.Layer):
        def __init__(self, units):
            super(GatedLinearUnit, self).__init__()
            self.linear = layers.Dense(units)
            self.sigmoid = layers.Dense(units, activation="sigmoid")

        def call(self, inputs):
            return self.linear(inputs) * self.sigmoid(inputs)

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate):
            super(GatedResidualNetwork, self).__init__()
            self.units = units
            self.elu_dense = layers.Dense(units, activation="elu")
            self.linear_dense = layers.Dense(units)
            self.dropout = layers.Dropout(dropout_rate)
            self.gated_linear_unit = GatedLinearUnit(units)
            self.layer_norm = layers.LayerNormalization()
            self.project = layers.Dense(units)

        def call(self, inputs):
            x = self.elu_dense(inputs)
            x = self.linear_dense(x)
            x = self.dropout(x)
            if inputs.shape[-1] != self.units:
                inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x)
            x = self.layer_norm(x)
            return x

    class VariableSelection(layers.Layer):
        def __init__(self, num_features, units, dropout_rate):
            super(VariableSelection, self).__init__()
            self.grns = list()
            for idx in range(num_features):
                grn = GatedResidualNetwork(units, dropout_rate)
                self.grns.append(grn)
            self.grn_concat = GatedResidualNetwork(units, dropout_rate)
            self.softmax = layers.Dense(units=num_features, activation="softmax")

        def call(self, inputs):
            v = layers.concatenate(inputs)
            v = self.grn_concat(v)
            v = tf.expand_dims(self.softmax(v), axis=-1)

            x = []
            for idx, input in enumerate(inputs):
                x.append(self.grns[idx](input))
            x = tf.stack(x, axis=1)

            outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
            return outputs

    def create_model_inputs():
        inputs = {}
        for feature_name in X.columns:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        return inputs

    def encode_inputs(inputs, encoding_size):
        encoded_features = []
        for col in range(inputs.shape[1]):
            encoded_feature = tf.expand_dims(inputs[:, col], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
            encoded_features.append(encoded_feature)
        return encoded_features

    def create_model(encoding_size, col_len, num_class, dropout_rate=0.15):
        inputs = layers.Input(col_len)
        feature_list = encode_inputs(inputs, encoding_size)
        num_features = len(feature_list)

        features = VariableSelection(num_features, encoding_size, dropout_rate)(
            feature_list
        )

        outputs = layers.Dense(units=num_class, activation="softmax")(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    return create_model(encoding_size,col_len,num_class)
        
        
        
#    _____________________________________________________
#___/ Google Brain - Ventilator Pressure Prediction 2021  \______________
#

def run_tf_blstm(epoch=int(50),batch_size=int(1024),train=None,test=None,targets=None):
    #kf = KFold(n_splits=5, shuffle=True, random_state=2000)
    kf = KFold(n_splits=10, shuffle=True)
    test_preds = []
    test_history = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        model = keras.models.Sequential([
            #keras.layers.Embedding(input_dim=train.shape[-2:], output_dim=300, mask_zero=True),
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dense(128, activation='selu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mae")

        scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/batch_size), 1e-5)
        lr = LearningRateScheduler(scheduler, verbose=1)

        #es = EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode="min", restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch_size, callbacks=[lr])
        test_history.append(history.history)
        model.save(f'model_save_fold{fold+1}')
        test_preds.append(model.predict(test).squeeze())

    return test_preds, test_history



def run_tf_blstm_lite(epoch=int(50),batch_size=int(1024),train=None,test=None,targets=None):
    #kf = KFold(n_splits=5, shuffle=True, random_state=2000)
    kf = KFold(n_splits=10, shuffle=True)
    test_preds = []
    test_history = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        model = keras.models.Sequential([
            #keras.layers.Embedding(input_dim=train.shape[-2:], output_dim=300, mask_zero=True),
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            keras.layers.Dense(64, activation='selu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mae")

        scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/batch_size), 1e-5)
        lr = LearningRateScheduler(scheduler, verbose=1)

        #es = EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode="min", restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch_size, callbacks=[lr])
        test_history.append(history.history)
        model.save(f'model_save_fold{fold+1}')
        test_preds.append(model.predict(test).squeeze())

    return test_preds, test_history


def run_tf_lstm(epoch=int(50),batch_size=int(1024),train=None,test=None,targets=None):
    #kf = KFold(n_splits=5, shuffle=True, random_state=2000)
    kf = KFold(n_splits=10, shuffle=True)
    test_preds = []
    test_history = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        model = keras.models.Sequential([
            #keras.layers.Embedding(input_dim=train.shape[-2:], output_dim=300, mask_zero=True),
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.LSTM(1024, return_sequences=True),
            keras.layers.LSTM(512, return_sequences=True),
            keras.layers.LSTM(256, return_sequences=True),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dense(128, activation='selu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mae")

        scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/batch_size), 1e-5)
        lr = LearningRateScheduler(scheduler, verbose=1)

        #es = EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode="min", restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch_size, callbacks=[lr])
        test_history.append(history.history)
        model.save(f'model_save_fold{fold+1}')
        result = model.predict(test)
        test_preds.append(result.squeeze())

    return test_preds, test_history


def dnn_model():

    x_input = Input(shape=(train.shape[-2:]))

    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)

    z2 = Bidirectional(GRU(units=384, return_sequences=True))(x2)

    z31 = Multiply()([x3, z2])
    z31 = BatchNormalization()(z31)
    z3 = Bidirectional(GRU(units=256, return_sequences=True))(z31)

    z41 = Multiply()([x4, z3])
    z41 = BatchNormalization()(z41)
    z4 = Bidirectional(GRU(units=128, return_sequences=True))(z41)

    z51 = Multiply()([x5, z4])
    z51 = BatchNormalization()(z51)
    z5 = Bidirectional(GRU(units=64, return_sequences=True))(z51)

    x = Concatenate(axis=2)([x5, z2, z3, z4, z5])

    x = Dense(units=128, activation='selu')(x)

    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output,
                  name='DNN_Model')
    return model



def run_tf_dnn(epoch=int(50),batch_size=int(1024),train=None,test=None,targets=None):
    VERBOSE = 0
    #kf = KFold(n_splits=5, shuffle=True, random_state=2000)
    kf = KFold(n_splits=5, shuffle=True)
    test_preds = []
    test_history = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        model = dnn_model()
        model.compile(optimizer="adam", loss="mae")

        #scheduler = ExponentialDecay(1e-3, 400*((len(train)*0.8)/batch_size), 1e-5)

        #lr = LearningRateScheduler(scheduler, verbose=1)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.85, patience=7, verbose=VERBOSE)

        #es = EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode="min", restore_best_weights=True)
        es = EarlyStopping(monitor="val_loss", patience=30,verbose=VERBOSE, mode="min", restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch_size, callbacks=[lr])
        test_history.append(history.history)
        model.save(f'model_save_fold{fold+1}')
        result = model.predict(test)
        test_preds.append(result.squeeze())

    return test_preds, test_history



