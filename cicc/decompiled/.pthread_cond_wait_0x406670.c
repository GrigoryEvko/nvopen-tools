// Function: .pthread_cond_wait
// Address: 0x406670
//
// attributes: thunk
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
  return pthread_cond_wait(cond, mutex);
}
