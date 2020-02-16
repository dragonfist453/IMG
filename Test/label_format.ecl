IMPORT IMG.IMG;

label := RECORD
    SET OF UNSIGNED id;
    REAL label;
END;

train_label := IMG.MNIST_train_label();

labelData := NORMALIZE(train_label, 1, TRANSFORM(label,
                            SELF.id := [LEFT.id, COUNTER],
                            SELF.label := (REAL) (>UNSIGNED1<) LEFT.label[counter]));

OUTPUT(labelData, ,'thor::deletethisafterpls', OVERWRITE);

//WORKS BEAUTIFULLY