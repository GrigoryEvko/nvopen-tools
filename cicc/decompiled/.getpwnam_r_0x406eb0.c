// Function: .getpwnam_r
// Address: 0x406eb0
//
// attributes: thunk
int getpwnam_r(const char *name, struct passwd *resultbuf, char *buffer, size_t buflen, struct passwd **result)
{
  return getpwnam_r(name, resultbuf, buffer, buflen, result);
}
