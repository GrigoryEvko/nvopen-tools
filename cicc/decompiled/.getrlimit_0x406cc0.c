// Function: .getrlimit
// Address: 0x406cc0
//
// attributes: thunk
int getrlimit(__rlimit_resource_t resource, struct rlimit *rlimits)
{
  return getrlimit(resource, rlimits);
}
