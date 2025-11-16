// Function: sub_1AA4710
// Address: 0x1aa4710
//
void __fastcall sub_1AA4710(char *src, char *a2, unsigned __int8 (__fastcall *a3)(__m128i *, __int8 *))
{
  __int64 v4; // rbx

  if ( a2 - src <= 784 )
  {
    sub_1AA42C0(src, a2, a3);
  }
  else
  {
    v4 = 56 * ((0x6DB6DB6DB6DB6DB7LL * ((a2 - src) >> 3)) >> 1);
    sub_1AA4710(src);
    sub_1AA4710(&src[v4]);
    sub_1AA44F0(
      (__int64)src,
      (__int64)&src[v4],
      (__int64)a2,
      0x6DB6DB6DB6DB6DB7LL * (v4 >> 3),
      0x6DB6DB6DB6DB6DB7LL * ((a2 - &src[v4]) >> 3),
      (unsigned __int8 (__fastcall *)(__int64, __int64))a3);
  }
}
