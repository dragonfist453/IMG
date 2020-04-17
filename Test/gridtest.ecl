IMPORT IMG.IMG;
IMPORT GNN.Tensor;
IMPORT Python3 as Python;
TensData := Tensor.R4.TensData;
#option('outputLimit', 2000);

//Train data definitions
imgcount_train := 60000;
imgRows := 28;
imgCols := 28;
imgChannels := 1;
imgSize := imgRows * imgCols;
latentDim := 100;
numClasses := 10;
batchSize := 128;

IMG_FORMAT := RECORD
    STRING filename;
    DATA image;
END;

//Take MNIST dataset using IMG module
mnist := IMG.MNIST_train_image();

images := choosen(mnist, 100);

image_tens := IMG.MNISTtoTens(images);

output_images := IMG.TenstoImg(image_tens);

tensimg := OUTPUT(output_images, ,'~test::image_out_from_tens',OVERWRITE);

DATA makeGrid(SET OF DATA images, Integer r, Integer c) := EMBED(Python)
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            image = images[cnt]
            image_np = np.frombuffer(image, dtype=np.uint8)
            image_mat = image_np.reshape((28,28))
            axs[i,j].imshow(image_mat[:,:], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.canvas.draw()        
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    plt.close()  
    img_encode = cv2.imencode('.png', image_from_plot)[1]
    return bytearray(img_encode)
ENDEMBED;

//Transform IMG_NUMERICAL to IMG_FORMAT with jpg encoding
mnist_png := DATASET(1, TRANSFORM(IMG_FORMAT,
                    SELF.filename := 'Epoch_'+1+'.png',
                    SELF.image := makeGrid(SET(mnist, image), 5, 5)
                    ));


jpgimg := OUTPUT(mnist_png, ,'~test::mnist_as_jpg',OVERWRITE);

SEQUENTIAL(tensimg, jpgimg);