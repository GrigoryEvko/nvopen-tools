// Function: .strftime
// Address: 0x406b10
//
// attributes: thunk
size_t strftime(char *s, size_t maxsize, const char *format, const struct tm *tp)
{
  return strftime(s, maxsize, format, tp);
}
