using namespace std;
int off = 0;
byte* data = (byte*)image;
unsigned int dimensions[] = {0,0};
bool flag = true;
while(flag) {
    while(data[off]==0xff) off++;
    byte mrkr = data[off];  off++;

    if(mrkr==0xd8) continue;    // SOI
    if(mrkr==0xd9) break;       // EOI
    if(0xd0<=mrkr && mrkr<=0xd7) continue;
    if(mrkr==0x01) continue;    // TEM

    int len = (data[off]<<8) | data[off+1];  
    off+=2;  

    if(mrkr==0xc0){
        unsigned int w = (unsigned int)( (data[off+1]<<8) | data[off+2] ); //height
        unsigned int h = (unsigned int)( (data[off+3]<<8) | data[off+4] ); //width
        dimensions[0] = h;
        dimensions[1] = w;
        __isAllResult = (bool)true;
        __lenResult = (size32_t)2;
        __result = (void*)dimensions;
        break;
        flag = false;
    }
    off+=len-2;
}