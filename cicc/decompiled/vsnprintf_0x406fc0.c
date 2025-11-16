// Function: vsnprintf
// Address: 0x406fc0
//
// attributes: thunk
int vsnprintf(char *s, size_t maxlen, const char *format, __gnuc_va_list arg)
{
  return __imp_vsnprintf(s, maxlen, format, arg);
}
