// Function: .vfprintf
// Address: 0x406d50
//
// attributes: thunk
int vfprintf(FILE *s, const char *format, __gnuc_va_list arg)
{
  return vfprintf(s, format, arg);
}
