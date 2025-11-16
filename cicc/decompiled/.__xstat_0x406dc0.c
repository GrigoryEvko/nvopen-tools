// Function: .__xstat
// Address: 0x406dc0
//
// attributes: thunk
int __xstat(int ver, const char *filename, struct stat *stat_buf)
{
  return _xstat(ver, filename, stat_buf);
}
