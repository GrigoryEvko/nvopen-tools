// Function: .gettimeofday
// Address: 0x4065f0
//
// attributes: thunk
int gettimeofday(struct timeval *tv, __timezone_ptr_t tz)
{
  return gettimeofday(tv, tz);
}
