IMPORT IMG.IMG;

images := RECORD
    STRING filename;
    DATA image;
END;

something := IMG.ReadImage('test::images');

OUTPUT(something, ,'whatever::images',OVERWRITE);