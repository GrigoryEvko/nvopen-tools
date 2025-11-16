// Function: .pthread_mutex_init
// Address: 0x4064b0
//
// attributes: thunk
int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr)
{
  return pthread_mutex_init(mutex, mutexattr);
}
