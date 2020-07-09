#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def model(images, bn_param, keep_prob, num_classes, device, FLAGS):
  if 'vgg' in FLAGS.model:
    import vgg

    if FLAGS.model=='vgg11': depth = [1,1,2,2,2]
    elif FLAGS.model=='vgg13': depth = [2,2,2,2,2]
    elif FLAGS.model=='vgg16': depth = [2,2,3,3,3]
    elif FLAGS.model=='vgg19': depth = [2,2,4,4,4]
    else:
      import vgg_common
      if FLAGS.model == 'vgg11_c':depth = [1, 1, 2, 2, 2]
      elif FLAGS.model == 'vgg13_c':depth = [2, 2, 2, 2, 2]
      elif FLAGS.model == 'vgg16_c':depth = [2, 2, 3, 3, 3]
      elif FLAGS.model == 'vgg19_c':depth = [2, 2, 4, 4, 4]
      logits = vgg_common.vgg_common(images,bn_param, keep_prob, depth, num_classes, device=device)
      return logits
    logits = vgg.vgg(images, bn_param, keep_prob, depth, num_classes, device=device)
    return logits

  elif 'resnet' in  FLAGS.model:
    assert (FLAGS.depth - 4) % 6 == 0, 'depth should be 6n+4'
    import resnet
    n = (FLAGS.depth-4)//6
    k = FLAGS.widen_factor

    if FLAGS.model == 'resnet':
      import resnet
      logits = resnet.inference(images, bn_param, keep_prob, n=n, k=k, num_classes=num_classes, device=device)
    elif FLAGS.model == 'resnet_sc':
      import resnet_sc
      logits =resnet_sc.inference(images, bn_param, keep_prob, n=n, k=k, num_classes=num_classes, device=device)
    elif FLAGS.model == 'resnet_sc_bn':
      import resnet_sc_bn
      logits = resnet_sc_bn.inference(images, bn_param, keep_prob, n=n, k=k, num_classes=num_classes, device=device)
    elif FLAGS.model == 'resnet_orgin':
      import resnet_orgin
      logits = resnet_orgin.inference(images, bn_param, keep_prob, n=n, k=k, num_classes=num_classes, device=device)
    return logits



  else:
    assert False, 'unknown model'
