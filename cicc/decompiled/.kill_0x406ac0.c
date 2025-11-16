// Function: .kill
// Address: 0x406ac0
//
// attributes: thunk
int kill(__pid_t pid, int sig)
{
  return kill(pid, sig);
}
