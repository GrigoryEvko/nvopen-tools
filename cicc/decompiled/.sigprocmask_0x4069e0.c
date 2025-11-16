// Function: .sigprocmask
// Address: 0x4069e0
//
// attributes: thunk
int sigprocmask(int how, const sigset_t *set, sigset_t *oset)
{
  return sigprocmask(how, set, oset);
}
