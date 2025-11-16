// Function: .statfs
// Address: 0x4065b0
//
// attributes: thunk
int statfs(const char *file, struct statfs *buf)
{
  return statfs(file, buf);
}
