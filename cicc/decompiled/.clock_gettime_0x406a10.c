// Function: .clock_gettime
// Address: 0x406a10
//
// attributes: thunk
int clock_gettime(clockid_t clock_id, struct timespec *tp)
{
  return clock_gettime(clock_id, tp);
}
