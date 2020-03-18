IMPORT IMG.IMG;
IMPORT GNN.Tensor;
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

//Take MNIST dataset using IMG module
mnist := IMG.MNIST_train_image();

images := choosen(mnist, 10);

image_tens := IMG.MNISTtoTens(images);

output_images := IMG.TenstoImg(image_tens);

tensimg := OUTPUT(output_images, ,'~test::image_out_from_tens',OVERWRITE);

mnist_jpg := IMG.OutputMNIST(output_images);

jpgimg := OUTPUT(mnist_jpg, ,'~test::mnist_as_jpg',OVERWRITE);

SEQUENTIAL(tensimg, jpgimg);