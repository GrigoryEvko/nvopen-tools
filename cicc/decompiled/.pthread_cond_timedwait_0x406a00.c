// Function: .pthread_cond_timedwait
// Address: 0x406a00
//
// attributes: thunk
int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *abstime)
{
  return pthread_cond_timedwait(cond, mutex, abstime);
}
