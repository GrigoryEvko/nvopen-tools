// Function: .lseek
// Address: 0x4063b0
//
// attributes: thunk
__off_t lseek(int fd, __off_t offset, int whence)
{
  return lseek(fd, offset, whence);
}
