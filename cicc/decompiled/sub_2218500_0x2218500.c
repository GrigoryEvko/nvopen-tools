// Function: sub_2218500
// Address: 0x2218500
//
__int64 sub_2218500(__int64 a1, char *a2, int a3, const char *a4, ...)
{
  unsigned int v6; // r12d
  gcc_va_list arg; // [rsp+8h] [rbp-F0h] BYREF

  va_start(arg, a4);
  __uselocale();
  v6 = vsnprintf(a2, a3, a4, arg);
  __uselocale();
  return v6;
}
