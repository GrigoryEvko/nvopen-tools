// Function: .pthread_cond_init
// Address: 0x406520
//
// attributes: thunk
int pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *cond_attr)
{
  return pthread_cond_init(cond, cond_attr);
}
