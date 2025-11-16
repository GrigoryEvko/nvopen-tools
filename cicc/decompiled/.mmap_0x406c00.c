// Function: .mmap
// Address: 0x406c00
//
// attributes: thunk
void *mmap(void *addr, size_t len, int prot, int flags, int fd, __off_t offset)
{
  return mmap(addr, len, prot, flags, fd, offset);
}
