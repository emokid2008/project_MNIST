import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# подгрузка данных
(data_train, target_train), (data_test, target_test) = mnist.load_data()

# формат данных
print(f'Data train shape: {data_train.shape}\n'
      f'Target train shape: {target_train.shape}\n'
      f'Dtype: {data_train.dtype}\n'
      f'Value range: {data_train.min()} - {data_train.max()}')

print(f'Data test shape: {data_test.shape}\n'
      f'Target test shape: {target_test.shape}\n'
      f'Dtype: {target_train.dtype}\n'
      f'Value range: {target_train.min()} - {target_train.max()}\n')
