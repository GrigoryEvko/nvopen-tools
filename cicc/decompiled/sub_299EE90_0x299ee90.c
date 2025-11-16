// Function: sub_299EE90
// Address: 0x299ee90
//
void __fastcall sub_299EE90(char *src, char *a2, unsigned __int8 (__fastcall *a3)(__m128i *, __int8 *))
{
  __int64 v4; // rbx

  if ( a2 - src <= 784 )
  {
    sub_299EA40(src, a2, a3);
  }
  else
  {
    v4 = 56 * ((0x6DB6DB6DB6DB6DB7LL * ((a2 - src) >> 3)) >> 1);
    sub_299EE90(src);
    sub_299EE90(&src[v4]);
    sub_299EC70(
      (__int64)src,
      (__int64)&src[v4],
      (__int64)a2,
      0x6DB6DB6DB6DB6DB7LL * (v4 >> 3),
      0x6DB6DB6DB6DB6DB7LL * ((a2 - &src[v4]) >> 3),
      (unsigned __int8 (__fastcall *)(__int64, __int64))a3);
  }
}
