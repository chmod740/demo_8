from gen_check_code import gen_captcha_text_and_image
from gen_test import get_test_captcha_text_and_image
import numpy as np
import tensorflow as tf


text, image = gen_captcha_text_and_image()
print("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = image.shape[0]
IMAGE_WIDTH = image.shape[1]
image_shape = image.shape
MAX_CAPTCHA = len(text)
CHAR_SET_LEN = 95
print("验证码文本最长字符数", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
# 度化是将三分量转化成一样数值的过程
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # int gray = (int) (0.3 * r + 0.59 * g + 0.11 * b);
        return gray
    else:
        return img


""" 
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。 
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行 
"""



# CHAR_SET_LEN = len(char_set)

# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('长度超限')
    while len(text) < MAX_CAPTCHA:
        text = text + " "

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        try:
            k = ord(c)-ord(' ')
            if k > 95:
                k = ord(' ')
        except:
            raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        try:
            vector[idx] = 1
        except:
            pass
    return vector



# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        # if char_idx < 10:
        #     char_code = char_idx + ord('0')
        # elif char_idx < 36:
        #     char_code = char_idx - 10 + ord('A')
        # elif char_idx < 62:
        #     char_code = char_idx - 36 + ord('a')
        # elif char_idx == 62:
        #     char_code = ord('_')
        # else:
        #     raise ValueError('error')
        char_code = char_idx + ord(' ')
        text.append(chr(char_code))
    return "".join(text)



# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == image_shape:
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)


        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout



"""
定义卷积神经网络
"""
def create_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    """
    定义卷积层
    """
    """conv1_1"""
    """
    定义卷积核
    """
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 64]))
    """偏置值"""
    b_c1 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv1 = tf.nn.conv2d(x, w_c1, [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bias=b_c1)
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv1 = tf.nn.dropout(conv1, keep_prob)
    # [40, 64, 64]

    """conv1_2"""
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_c2)
    conv2 = tf.nn.relu(conv2)
    # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv2 = tf.nn.dropout(conv2, keep_prob)
    # [40, 64, 64]

    # pool1
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv2 = tf.nn.dropout(conv2, keep_prob)
    # [20, 32, 64]



    """conv2_1"""
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_c3)
    conv3 = tf.nn.relu(conv3)
    # conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3 = tf.nn.dropout(conv3, keep_prob)
    # [20, 32, 128]

    """conv2_2"""
    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv4 = tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b_c4)
    conv4 = tf.nn.relu(conv4)
    # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv4 = tf.nn.dropout(conv4, keep_prob)
    # [20, 32, 128]

    """pool2"""
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # [10, 16, 128]


    """conv3_1"""
    w_c5 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 256]))
    b_c5 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv5 = tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b_c5)
    conv5 = tf.nn.relu(conv5)
    # conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv5 = tf.nn.dropout(conv5, keep_prob)
    # [10, 16, 256]

    """conv3_2"""
    w_c6 = tf.Variable(w_alpha * tf.random_normal([3, 3, 256, 256]))
    b_c6 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv6 = tf.nn.conv2d(conv5, w_c6, strides=[1, 1, 1, 1], padding='SAME')
    conv6 = tf.nn.bias_add(conv6, b_c6)
    conv6 = tf.nn.relu(conv6)
    # [10, 16, 256]

    """conv3_3"""
    w_c7 = tf.Variable(w_alpha * tf.random_normal([3, 3, 256, 256]))
    b_c7 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv7 = tf.nn.conv2d(conv6, w_c7, strides=[1, 1, 1, 1], padding='SAME')
    conv7 = tf.nn.bias_add(conv7, b_c7)
    conv7 = tf.nn.relu(conv7)
    # [10, 16, 256]

    """pool3"""
    conv7 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # [5, 8, 256]

    """conv4_1"""
    w_c8 = tf.Variable(w_alpha * tf.random_normal([3, 3, 256, 512]))
    b_c8 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv8 = tf.nn.conv2d(conv7, w_c8, strides=[1, 1, 1, 1], padding='SAME')
    conv8 = tf.nn.bias_add(conv8, b_c8)
    conv8 = tf.nn.relu(conv8)
    # [5, 8, 512]


    """conv4_2"""
    w_c9 = tf.Variable(w_alpha * tf.random_normal([3, 3, 512, 512]))
    b_c9 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv9 = tf.nn.conv2d(conv8, w_c9, strides=[1, 1, 1, 1], padding='SAME')
    conv9 = tf.nn.bias_add(conv9, b_c9)
    conv9 = tf.nn.relu(conv9)
    # [5, 8, 512]



    """conv4_3"""
    w_c10 = tf.Variable(w_alpha * tf.random_normal([3, 3, 512, 512]))
    b_c10 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv10 = tf.nn.conv2d(conv9, w_c10, strides=[1, 1, 1, 1], padding='SAME')
    conv10 = tf.nn.bias_add(conv10, b_c10)
    conv10 = tf.nn.relu(conv10)
    # [5, 8, 512]



    """pool4"""
    conv10 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # [3, 4, 512]

    """conv5_1"""
    w_c11 = tf.Variable(w_alpha * tf.random_normal([3, 3, 512, 512]))
    b_c11 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv11 = tf.nn.conv2d(conv10, w_c11, strides=[1, 1, 1, 1], padding='SAME')
    conv11 = tf.nn.bias_add(conv11, b_c11)
    conv11 = tf.nn.relu(conv11)
    # [3, 4, 512]



    """conv5_2"""
    w_c12 = tf.Variable(w_alpha * tf.random_normal([3, 3, 512, 512]))
    b_c12 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv12 = tf.nn.conv2d(conv11, w_c12, strides=[1, 1, 1, 1], padding='SAME')
    conv12 = tf.nn.bias_add(conv12, b_c12)
    conv12 = tf.nn.relu(conv12)
    # [3, 4, 512]


    """conv5_3"""
    w_c13 = tf.Variable(w_alpha * tf.random_normal([3, 3, 512, 512]))
    b_c13 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv13 = tf.nn.conv2d(conv12, w_c13, strides=[1, 1, 1, 1], padding='SAME')
    conv13 = tf.nn.bias_add(conv13, b_c13)
    conv13 = tf.nn.relu(conv13)
    # [3, 4, 512]

    """pool5"""
    conv13 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # [2, 2, 512]


    """
    全连接层
    """
    """fc1"""
    w_fc1 = tf.Variable(w_alpha * tf.random_normal([2048, 1024]))
    b_fc1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense_1 = tf.reshape(conv13, [-1, w_fc1.get_shape().as_list()[0]])
    dense_1 = tf.nn.relu(tf.add(tf.matmul(dense_1, w_fc1), b_fc1))
    # dense = tf.nn.dropout(dense, keep_prob)

    """fc2"""
    w_fc2 = tf.Variable(w_alpha * tf.random_normal([1024, 256]))
    b_fc2 = tf.Variable(b_alpha * tf.random_normal([256]))
    dense_2 = tf.nn.relu(tf.add(tf.matmul(dense_1, w_fc2), b_fc2))


    """fc3"""
    w_out = tf.Variable(w_alpha * tf.random_normal([256, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense_2, w_out), b_out)
    out = tf.reshape(out, shape=[-1, MAX_CAPTCHA, CHAR_SET_LEN])
    out = tf.nn.softmax(out)
    out = tf.reshape(out, shape=[-1, MAX_CAPTCHA * CHAR_SET_LEN])
    print(out)
    return out



# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 定义三层的卷积神经网络

    # 定义第一层的卷积神经网络
    # 定义第一层权重
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    # 定义第一层的偏置
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # 定义第一层的激励函数
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    # conv1 为输入  ksize 表示使用2*2池化，即将2*2的色块转化成1*1的色块
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout防止过拟合。
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # [20,32,32]


    # 定义第二层的卷积神经网络
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # [10,16,64]

    # 定义第三层的卷积神经网络
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # [5,8,64]

    # Fully connected layer
    # 随机生成权重
    w_d = tf.Variable(w_alpha * tf.random_normal([2560, 1024]))
    # 随机生成偏置
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


# 训练
def train_crack_captcha_cnn():
    # X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    # Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    # keep_prob = tf.placeholder(tf.float32)  # dropout
    # output = create_cnn()
    output = crack_captcha_cnn()
    # loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=output))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            step = 0
            while True:
                batch_x, batch_y = get_next_batch(64)
                _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                print(step, loss_)

                # 每100 step计算一次准确率
                if step % 100 == 0:
                    batch_x_test, batch_y_test = get_next_batch(100)
                    acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                    print(step, acc)
                    # 如果准确率大于50%,保存模型,完成训练
                    if acc > 0.99:
                        saver.save(sess, "./crack_capcha.model", global_step=step)
                        break
                step += 1

## 训练(如果要训练则去掉下面一行的注释)
# train_crack_captcha_cnn()

def create_dnn(x, w_alpha=0.01, b_alpha=0.1):
    # input x [1000 1]
    # layer 1
    w1 = tf.Variable(w_alpha * tf.random_normal([96000, 1000]))
    b1 = tf.Variable(b_alpha * tf.random_normal([1000]))
    x = tf.matmul(x, w1 )
    x = tf.add(x, b1)
    # layer 2
    w2 = tf.Variable(w_alpha * tf.random_normal([1000, CHAR_SET_LEN * MAX_CAPTCHA]))
    b2 = tf.Variable(b_alpha * tf.random_normal(CHAR_SET_LEN * MAX_CAPTCHA))
    return tf.add(tf.matmul(x, w2), b2)



def crack_captcha():
    output = crack_captcha_cnn()

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    count = 0
    # 因为测试集共40个...写的很草率
    for i in range(40):
        text, image = get_test_captcha_text_and_image(i)
        image = convert2gray(image)
        captcha_image = image.flatten() / 255
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        predict_text = text_list[0].tolist()
        predict_text = str(predict_text)
        predict_text = predict_text.replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
        if text == predict_text:
            count += 1
            check_result = "，预测结果正确"
        else:
            check_result = "，预测结果不正确"
            print("正确: {}  预测: {}".format(text, predict_text) + check_result)
    print("正确率:" + str(count) + "/40")
# 测试(如果要测试则去掉下面一行的注释)
# crack_captcha()



def train_dnn():


    pass

