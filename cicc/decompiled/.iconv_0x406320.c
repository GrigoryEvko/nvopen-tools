// Function: .iconv
// Address: 0x406320
//
// attributes: thunk
size_t iconv(iconv_t cd, char **inbuf, size_t *inbytesleft, char **outbuf, size_t *outbytesleft)
{
  return iconv(cd, inbuf, inbytesleft, outbuf, outbytesleft);
}
