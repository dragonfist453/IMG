IMPORT IMG.IMG;

train_img := IMG.MNIST_train_image();

New_format := RECORD
    SET OF UNSIGNED id;
    REAL image;
END;

imageData := NORMALIZE(train_img, 784, TRANSFORM(New_format,
    SELF.id := [LEFT.id,COUNTER],
    SELF.image := (REAL) (>UNSIGNED1<) LEFT.image[counter]));

OUTPUT(imageData, ,'thor::deletethisafterpls',OVERWRITE); 

//Works beautifully to take numerical real numbers from bits for tensors