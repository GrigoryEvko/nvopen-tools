// Function: .getpwuid_r
// Address: 0x406890
//
// attributes: thunk
int getpwuid_r(__uid_t uid, struct passwd *resultbuf, char *buffer, size_t buflen, struct passwd **result)
{
  return getpwuid_r(uid, resultbuf, buffer, buflen, result);
}
