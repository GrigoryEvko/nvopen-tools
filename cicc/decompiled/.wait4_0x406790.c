// Function: .wait4
// Address: 0x406790
//
// attributes: thunk
__pid_t wait4(__pid_t pid, __WAIT_STATUS stat_loc, int options, struct rusage *usage)
{
  return wait4(pid, stat_loc, options, usage);
}
