// Function: sub_16863E0
// Address: 0x16863e0
//
int sub_16863E0(__int64 a1, ...)
{
  int result; // eax
  gcc_va_list va; // [rsp+8h] [rbp-C8h] BYREF

  va_start(va, a1);
  result = (int)va[0].reg_save_area;
  if ( !*(_BYTE *)(a1 + 4) )
    return sub_1685A80((unsigned int *)a1, 0, (__int64)va);
  return result;
}
