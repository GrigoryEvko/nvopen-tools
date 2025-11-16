// Function: .writev
// Address: 0x406980
//
// attributes: thunk
ssize_t writev(int fd, const struct iovec *iovec, int count)
{
  return writev(fd, iovec, count);
}
