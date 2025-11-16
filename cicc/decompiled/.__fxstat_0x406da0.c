// Function: .__fxstat
// Address: 0x406da0
//
// attributes: thunk
int __fxstat(int ver, int fildes, struct stat *stat_buf)
{
  return _fxstat(ver, fildes, stat_buf);
}
