import collections
import tensorflow as tf 
import tf_slim as slim 

def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    net = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)
    return net

def resnet_v2(inputs, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None):
    with tf.compat.v1.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        print (end_points_collection)
        print (inputs)
        with slim.arg_scope([slim.conv2d], outputs_collections = end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, [7, 7], stride=2, scope='conv1')
                    print(net)



def restnet_ve_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_ve_152'):
    resnet_v2(inputs, num_classes, global_pool, reuse=reuse, scope=scope)

if __name__ == "__main__":
    batch_size = 32
    height, width = 224, 224
    
    inputs = tf.compat.v1.random.uniform((batch_size, height, width, 3))

    #with slim.arg_scope()
    restnet_ve_152(inputs, 1000)