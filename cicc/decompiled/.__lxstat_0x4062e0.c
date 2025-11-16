// Function: .__lxstat
// Address: 0x4062e0
//
// attributes: thunk
int __lxstat(int ver, const char *filename, struct stat *stat_buf)
{
  return _lxstat(ver, filename, stat_buf);
}
