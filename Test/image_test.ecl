IMPORT IMG.IMG;

images := RECORD
    STRING filename;
    DATA image;
    UNSIGNED4 RecPos{virtual(fileposition)};
END;

something := IMG.ReadImage('test::images');

OUTPUT(something, ,'whatever::images',OVERWRITE);