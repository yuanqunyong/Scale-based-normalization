for k in range(totol_conv):
    convalue = sess.run(var_con[k])
    conv_watch = convalue.reshape((-1, convalue.shape[-1]))
    conv_max = np.max(conv_watch, axis=0)
    conv_min = np.min(conv_watch, axis=0)
    conv_mean = np.mean(conv_watch, axis=0)
    conv_var = np.std(conv_watch, axis=0)
    norm = np.linalg.norm(conv_watch, axis=0)

    logging.info('this is convalue of layer %f ' % (k))
    logging.info('conv_max =%s conv_min= %s' % (str(conv_max), str(conv_min)))
    logging.info('variance_value= %s meanvalue= %s norm = %s' % (str(conv_var), str(conv_mean), str(norm)))


values = sess.run([losses]+ tf.get_collection('weightcost'), feed_dict=feed_dict)
             print(values[0])
             print(values[1])
             gra = sess.run(tf.gradients(losses, var_con), feed_dict=feed_dict)
             for item in gra:
               tempp1 = np.abs(item)
               logging.info('gramaxvalue= %f graminvalue= %f grameanvalue= %f'%(np.max(tempp1), np.min(tempp1),np.mean(tempp1)))

for k in range(totol_conv):
    print(var_con[k])
    if k == 0:
        conv1, conv2 = sess.run([var_con[0], var_con[1]])

        # print(conv1.shape)
        conv1, conv2 = gutils.conv_scal(conv1, conv2)
        var_con[0].load(conv1, sess)
    elif 0 < k < totol_conv - 1:
        conv2_temp = sess.run(var_con[k + 1])
        conv1, conv2 = gutils.conv_scal(conv2, conv2_temp)
        var_con[k].load(conv1, sess)
    else:
        last_conv = gutils.last_conv_scal(conv2)
        var_con[k].load(last_conv, sess)

for k in range(totol_conv):
    conv_temp = sess.run(var_con[k])
    conv_temp = gutils.conv_adjust(conv_temp, False)
    var_con[k].load(conv_temp, sess)

values0 = self.full_validation([cross_entropy], sess, vali_data, vali_labels)
test_out_value1 = sess.run(logist, feed_dict=feed_dict)
test_out_value3 = sess.run(logist, feed_dict=feed_dict)
computer_error = test_out_value3 - test_out_value1
cross_entropy_value1 = sess.run(cross_entropy, feed_dict=feed_dict)
print(cross_entropy_value1)
conv_list = []
for k in range(totol_conv):
    print(k)
    if k == 0:
        conv1, conv2 = sess.run([var_con[0], var_con[1]])
        conv_list.append((conv1))
        conv_list.append((conv2))

        # print(conv1.shape)
        conv1, conv2 = gutils.conv_scal(conv1, conv2)
        var_con[0].load(conv1, sess)
    elif 0 < k < totol_conv - 1:
        conv2_temp = sess.run(var_con[k + 1])
        conv_list.append((conv2_temp))

        conv1, conv2 = gutils.conv_scal(conv2, conv2_temp)
        var_con[k].load(conv1, sess)
    else:
        last_conv = gutils.last_conv_scal(conv2)
        var_con[k].load(last_conv, sess)
values1 = self.full_validation([cross_entropy], sess, vali_data, vali_labels)
test_out_value2 = sess.run(logist, feed_dict=feed_dict)
test_value_contrast = test_out_value2 - test_out_value1
print(test_value_contrast)

cross_entropy_value2 = sess.run(cross_entropy, feed_dict=feed_dict)

if values0[0] < values1[0]:
    logging.info(' scaling is %s\n' % (values0[0] > values1[0]))
    for k in range(totol_conv):
        var_con[k].load(conv_list[k], sess)