// Function: .pthread_sigmask
// Address: 0x406a40
//
// attributes: thunk
int pthread_sigmask(int how, const __sigset_t *newmask, __sigset_t *oldmask)
{
  return pthread_sigmask(how, newmask, oldmask);
}
