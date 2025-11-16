// Function: .fseeko64
// Address: 0x406090
//
// attributes: thunk
int fseeko64(FILE *stream, __off64_t off, int whence)
{
  return fseeko64(stream, off, whence);
}
