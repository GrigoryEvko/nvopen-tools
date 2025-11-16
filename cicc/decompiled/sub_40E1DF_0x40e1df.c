// Function: sub_40E1DF
// Address: 0x40e1df
//
unsigned __int64 sub_40E1DF(__int64 a1, unsigned __int64 a2, char *a3, ...)
{
  gcc_va_list va; // [rsp+8h] [rbp-C8h] BYREF

  va_start(va, a3);
  return sub_40D5CA(a1, a2, a3, (__int64)va);
}
