IMPORT Python3 as Python;
IMPORT GNN.Tensor;
TensData := Tensor.R4.TensData;

EXPORT IMG := MODULE
    //IMG module to work with image datasets
    
    //General format of MNIST images
    SHARED IMG_FORMAT_MNIST := RECORD
        UNSIGNED id;
        DATA image;
    END;

    //Read MNIST training set from ubyte file
    EXPORT DATASET(IMG_FORMAT_MNIST) MNIST_train_image() := FUNCTION
        numImages := 60000;
        numRows := 28;
        numCols := 28;
        imgSize := numRows * numCols;

        // We should have been able to use INTEGER4 for the first four fields, but the ENDIAN seems to be
        // backward from what we are expecting, so we use data4
        MNIST_FORMAT := RECORD
            DATA4 magic;
            DATA4 numImages;
            DATA4 numRows;
            DATA4 numCols;
            DATA47040000 contents;  // 60000 * 28 * 28 gets us to the end of the file.
        END;

        // Read from the landing zone.  You'll probably have to adjust the file path.  Don't use dash(-) in the
        // file name.  Change the file name to use underscores.   Note the escape characters (^) to indicate capital letters.
        // Otherwise will convert to lower case.
        mnist := DATASET('~test::mnist_train_images', MNIST_FORMAT, FLAT);

        // This will create 60,000 records, each with one image.  The id field indicates the image number
        outRecs0 := NORMALIZE(mnist, numImages, TRANSFORM(IMG_FORMAT_MNIST,
                                    SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                    SELF.id := COUNTER));

        //Distribute records on the cluster
        outRecs := DISTRIBUTE(outRecs0,id); 

        RETURN outRecs;                            
    END;

    //Extract MNIST test images and export
    EXPORT DATASET(IMG_FORMAT_MNIST) MNIST_test_image() := FUNCTION
        numImages := 10000;
        numRows := 28;
        numCols := 28;
        imgSize := numRows * numCols;

        // We should have been able to use INTEGER4 for the first four fields, but the ENDIAN seems to be
        // backward from what we are expecting, so we use data4
        MNIST_FORMAT := RECORD
            DATA4 magic;
            DATA4 numImages;
            DATA4 numRows;
            DATA4 numCols;
            DATA7840000 contents;  // 60000 * 28 * 28 gets us to the end of the file.
        END;

        // Read from the landing zone.  You'll probably have to adjust the file path.  Don't use dash(-) in the
        // file name.  Change the file name to use underscores.   Note the escape characters (^) to indicate capital letters.
        // Otherwise will convert to lower case.
        mnist := DATASET('~test::mnist_train_images', MNIST_FORMAT, FLAT);

        // This will create 60,000 records, each with one image.  The id field indicates the image number
        outRecs0 := NORMALIZE(mnist, numImages, TRANSFORM(IMG_FORMAT_MNIST,
                                    SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                    SELF.id := COUNTER));

        //Distribute records on the cluster
        outRecs := DISTRIBUTE(outRecs0,id); 

        RETURN outRecs;                            
    END;

    //Format for labels
    SHARED LABEL_FORMAT_MNIST := RECORD
        UNSIGNED id;
        DATA1 label;
    END;

    //Extract MNIST training labels and export
    EXPORT DATASET(LABEL_FORMAT_MNIST) MNIST_train_label() := FUNCTION
        numImages := 60000;

        MNIST_FORMAT := RECORD
            DATA4 magic;
            DATA4 numImages;
            DATA60000 contents;
        END;

        mnist_labels := DATASET('~test::mnist_train_labelled',MNIST_FORMAT,FLAT);

        outRecs0 := NORMALIZE(mnist_labels, numImages, TRANSFORM(LABEL_FORMAT_MNIST, 
                                            SELF.label := LEFT.contents[COUNTER],
                                            SELF.id := COUNTER;));

        outRecs := DISTRIBUTE(outRecs0,id);
        RETURN outRecs;
    END;

    //Extract MNIST training labels and export
    EXPORT DATASET(LABEL_FORMAT_MNIST) MNIST_test_label() := FUNCTION
        numImages := 10000;

        MNIST_FORMAT := RECORD
            DATA4 magic;
            DATA4 numImages;
            DATA10000 contents;
        END;

        mnist_labels := DATASET('~test::mnist_train_labelled',MNIST_FORMAT,FLAT);

        outRecs0 := NORMALIZE(mnist_labels, numImages, TRANSFORM(LABEL_FORMAT_MNIST, 
                                            SELF.label := LEFT.contents[COUNTER],
                                            SELF.id := COUNTER;));

        outRecs := DISTRIBUTE(outRecs0,id);
        RETURN outRecs;
    END;

    //General format of images
    SHARED IMG_FORMAT := RECORD
        STRING filename;
        DATA image;
    END;

    SHARED IMG_NUMERICAL := RECORD
        UNSIGNED8 id;
        DATA image;
    END;

    //Read image data from a logical file.
    EXPORT DATASET(IMG_NUMERICAL) ReadImage(STRING filename) := FUNCTION
        imageData := DATASET(filename, IMG_FORMAT, FLAT);
        numImages := COUNT(imageData);

        imageNumerical := NORMALIZE(imageData, numImages, TRANSFORM(IMG_NUMERICAL,
                                                        SELF.id := COUNTER,
                                                        SELF.image := LEFT.image
                                                        ));

        return imageNumerical;
    END;

    //Take from MNIST image dataset to convert to tensor
    EXPORT DATASET(Tensdata) MNISTtoTens(DATASET(IMG_FORMAT_MNIST) imgDataset) := FUNCTION
        //MNIST dimensions
        imgRows := 28;
        imgCols := 28;
        imgSize := imgRows * imgCols;
        
        //Build tensor data
        tens := NORMALIZE(imgDataset, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV imgCols+1, (COUNTER-1) % imgCols +1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 ));
        RETURN tens;
    END;    

    //Take from dataset of images to convert to tensor
    EXPORT DATASET(TensData) ImgtoTens(DATASET(IMG_NUMERICAL) imgDataset) := FUNCTION
        SET OF INTEGER GetImageDimensions(DATA image) := EMBED(Python)
            import cv2
            import numpy as np 

            nparr = np.frombuffer(bytes(image), np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            return list(img_np.shape);
        ENDEMBED;

        //Get image shape set
        imgShape := GetImageDimensions(imgDataset[1].image);
        
        //Put them into the subsequent variables
        imgRows := imgShape[1];
        imgCols := imgShape[2];

        //Calculate size to iterate
        imgSize := imgRows * imgCols;

        //Build tensor data
        tens := NORMALIZE(imgDataset, imgSize, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1) DIV imgCols+1, (COUNTER-1) % imgCols +1, 1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter] )/127.5 - 1 ));
        RETURN tens;                    
    END;

    /*
    //Convert tensor data output to Image to send to logical file
    EXPORT DATASET(IMG_NUMERICAL) TenstoJpg(DATASET(TensData) tens) := FUNCTION

    END;
    */
END; 