import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data(label_name='Species'):

    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],
                                         TRAIN_URL)

    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0)

    train_features, train_label = train, train.pop(label_name)

    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1],
                                                       TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    return (train_features,train_label), (test_features,test_label)

(train_x,train_y),(test_x,test_y) = load_data()
my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]

batch_size = 100
train_steps = 1000

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10,10],
    n_classes=3)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

classifier.train(
    input_fn=lambda:train_input_fn(train_x,train_y,batch_size),
    steps=train_steps)

def eval_input_fn(features, labels=None, batch_size=None):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, 'batch_size must not be None'
    dataset = dataset.batch(batch_size)

    return dataset

eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x,test_y,batch_size))

print(eval_result)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

expected = ['Setosa','Versicolor','Virginica']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength':[5.1,5.9,6.9],
    'SepalWidth':[3.3,3.0,3.1],
    'PetalLength':[1.7,4.2,5.4],
    'PetalWidth':[0.5,1.5,2.1]
}

predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x,
                                  labels=None,
                                  batch_size=batch_size))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(SPECIES[class_id], 100 * probability, expec))
