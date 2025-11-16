// Function: sub_130D560
// Address: 0x130d560
//
__int64 sub_130D560(char *a1, ...)
{
  gcc_va_list va; // [rsp+8h] [rbp-10D8h] BYREF
  char v3[4096]; // [rsp+20h] [rbp-10C0h] BYREF

  va_start(va, a1);
  sub_40D5CA((__int64)v3, 0x1000u, a1, (__int64)va);
  if ( !qword_4F969D8 )
  {
    sub_130AA40(v3);
    abort();
  }
  return qword_4F969D8(v3);
}
