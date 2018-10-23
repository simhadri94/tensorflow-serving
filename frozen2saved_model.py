import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './models/saved_model/1'
graph_pb = './models/20180402-114759.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    # Get input and output tensors
    images_placeholder = g.get_tensor_by_name("input:0")
    embeddings = g.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = g.get_tensor_by_name("phase_train:0")

    inputs = {'images' : images_placeholder, 'train_phase' : phase_train_placeholder}
            

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(inputs, {"out": embeddings})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()