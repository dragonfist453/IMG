EXPORT IMG := MODULE
    //IMG module to work with image datasets
    
    //General format of images
    IMG_FORMAT := RECORD
        UNSIGNED id;
        DATA image;
    END;
    
    //Read MNIST_set from ubyte file
    EXPORT IMG_FORMAT MNIST_read() := FUNCTION
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
        outRecs0 := NORMALIZE(mnist, numImages, TRANSFORM(IMG_FORMAT,
                                    SELF.image := LEFT.contents[((COUNTER-1)*imgSize+1) .. (COUNTER*imgSize)],
                                    SELF.id := COUNTER));

        //Distribute records on the cluster
        outRecs := DISTRIBUTE(outRecs0,id); 

        RETURN outRecs;                            
    END;

    //Format for labels
    LABEL_FORMAT := RECORD
        UNSIGNED id;
        DATA1 label;
    END;
    //Extract MNIST labels and export
    EXPORT LABEL_FORMAT MNIST_label() := FUNCTION
        numImages := 60000;

        MNIST_FORMAT := RECORD
            DATA4 magic;
            DATA4 numImages;
            DATA60000 contents;
        END;

        mnist_labels := DATASET('~test::mnist_train_labelled',MNIST_FORMAT,FLAT);

        outRecs0 := NORMALIZE(mnist_labels, numImages, TRANSFORM(LABEL_FORMAT, 
                                            SELF.label := LEFT.contents[COUNTER],
                                            SELF.id := COUNTER;));

        outRecs := DISTRIBUTE(outRecs0,id);
        RETURN outRecs;
    END;
    //Read images uploaded as blob. Location of file in server written as blobs taken as parameter. TBD
END;    
