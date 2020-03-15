IMPORT IMG.IMG;
IMPORT GNN.Tensor;
TensData := Tensor.R4.TensData;

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
images := IMG.ReadImage('~test::mnist_train_images');

image_tens := IMG.ImgtoTens(images);

OUTPUT(image_tens, ,'~im_out',OVERWRITE);