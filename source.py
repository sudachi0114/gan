
# GAN source code
import os, time
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.optimizers import Adam


class GAN():

    def __init__(self):
        # define
        # MNIST's datasize
        self.img_size = 28
        self.channel = 1
        self.image_shape = (self.img_size, self.img_size, self.channel)

        # dim of Latent Variable
        self.z_dims = 100

        optimizer = Adam(0.0002, 0.5)

        # Disctiminator model
        self.discriminator = self.buildDescriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Generator model
        self.generator = self.buildGenerator()
        # generator は単体で学習しないので
        #   Comoile する必要はない

        self.combined = self.buildCombined1()
        #self.combined = self.buildCombined2()

        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)



    def buildDescriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.image_shape))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        
        model.summary()

        
        return model

    
    def buildGenerator(self):

        model = Sequential()

        model.add(Dense(256, input_shape=(self.z_dims,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.image_shape), activation='tanh'))
        model.add(Reshape(self.image_shape))

        
        model.summary()

        return model


    def buildCombined1(self):

        self.discriminator.trainable = False

        model = Sequential( [self.generator, self.discriminator] )
        
        return model


    def buildCombined2(self):

        z = Input(shape=(self.z_dims,))
        gen_img =self.generator(z)

        self.discriminator.trainable = False

        pred = self.discriminator(gen_img)
        model = Model(z, pred)

        model.summary()

        return model
        

    def train(self, epochs, batch_size=128, save_interval=50):

        # mnist data の読み込み
        (real_data, _), (_, _) = mnist.load_data()

        # print(real_data.shape)  # (60000, 28, 28)

        # 値を [-1, 1] に正規化
        real_data = ( real_data.astype(np.float32) - 127.5 ) / 127.5
        real_data = np.expand_dims(real_data, axis=3)

        """
        print(real_data.shape)
        print(real_data[0])

        self.display(np.squeeze(real_data[0], axis=2))
        """

        half_batch = int(batch_size / 2)

        start = time.time()
        for epoch in range(epochs):

            # train of discriminator
            print("\ntrain of discriminator.....")

            # batch_size の半数を Generator から生成
            noise = np.random.normal(0, 1, (half_batch, self.z_dims))
            gen_imgs = self.generator.predict(noise)

            # batch_size の半数を Real_data から取得
            idx = np.random.randint(0, real_data.shape[0], half_batch)
            real_imgs = real_data[idx]

            # fit discriminator
            #   !! Real_data と Geneった data は別々に学習させる
            d_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            # それぞれの損失関数の平均を計算
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

            print("  Done.")

            # train of generator
            print("\ntrain of generator.....")

            noise = np.random.normal(0, 1, (batch_size, self.z_dims))

            # gene ったデータでも generator にとっては本物 (label = 1)
            gen_labels = np.array([1]*batch_size)

            # fit generator
            g_loss = self.combined.train_on_batch(noise, gen_labels)

            print("  Done.")

            
            # print progresses
            # print( "{} | [Descriminator loss: {}, acc: {}] [Generator loss: {}, acc: {}] ".format(epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]) )
            print( "{} | [Descriminator loss: {}, acc: {}] [Generator loss: {}] ".format(epoch, d_loss[0], 100*d_loss[1], g_loss) )

            # 指定↓感覚で生成画像を保存
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

        print( "elapsed time: {} [sec]".format(time.time() - start) )



    def save_imgs(self, epoch):

        # 生成画像を敷き詰める行数, 列数
        r, c = 5, 5

        noise = np.random.normal(0, 1, (r*c, self.z_dims))
        gen_imgs = self.generator.predict(noise)

        # 生成画像を [0, 1] に rescale
        gen_imgs = 0.5*gen_imgs+0.5

        fig, ax = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                ax[i, j].axis(False)
                cnt += 1

        cwd = os.getcwd()
        save_loc = os.path.join(cwd, "images")
        os.makedirs(save_loc, exist_ok=True)
        save_file = os.path.join(save_loc, "mnist_{}.jpg".format(epoch))
        fig.savefig(save_file)
        plt.close()



    def display(self, x):

        plt.imshow(x)
        plt.show()
    

if __name__ == '__main__':

    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=100)
