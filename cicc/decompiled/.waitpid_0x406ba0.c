// Function: .waitpid
// Address: 0x406ba0
//
// attributes: thunk
__pid_t waitpid(__pid_t pid, int *stat_loc, int options)
{
  return waitpid(pid, stat_loc, options);
}
