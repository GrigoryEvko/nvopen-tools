// Function: .lseek64
// Address: 0x406bb0
//
// attributes: thunk
__off64_t lseek64(int fd, __off64_t offset, int whence)
{
  return lseek64(fd, offset, whence);
}
