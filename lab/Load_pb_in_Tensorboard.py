import tensorflow as tf

export_path = './exported_model'

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()) as sess:
    '''
    You can provide 'tags' when saving a model,
    in my case I provided, 'serve' tag 
    '''

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
    graph = tf.get_default_graph()

    LOGDIR='./tensorboard/'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)

    # print your graph's ops, if needed
    #print(graph.get_operations())

    '''
    In my case, I named my input and output tensors as
    input:0 and output:0 respectively
    ''' 
    #y_pred = sess.run('output:0', feed_dict={'input:0': X_test})
