// Function: .execve
// Address: 0x406d00
//
// attributes: thunk
int execve(const char *path, char *const argv[], char *const envp[])
{
  return execve(path, argv, envp);
}
