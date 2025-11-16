// Function: .__fxstat64
// Address: 0x406ab0
//
// attributes: thunk
int __fxstat64(int ver, int fildes, struct stat64 *stat_buf)
{
  return _fxstat64(ver, fildes, stat_buf);
}
