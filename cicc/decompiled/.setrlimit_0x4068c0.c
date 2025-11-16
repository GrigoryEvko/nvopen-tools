// Function: .setrlimit
// Address: 0x4068c0
//
// attributes: thunk
int setrlimit(__rlimit_resource_t resource, const struct rlimit *rlimits)
{
  return setrlimit(resource, rlimits);
}
