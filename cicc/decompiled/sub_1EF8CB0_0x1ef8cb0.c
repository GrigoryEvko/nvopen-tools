// Function: sub_1EF8CB0
// Address: 0x1ef8cb0
//
void __fastcall sub_1EF8CB0(__m128i *a1, __int64 a2)
{
  __int64 v2; // rbx

  if ( a2 - (__int64)a1 <= 560 )
  {
    sub_1EF80B0((__int64)a1, a2);
  }
  else
  {
    v2 = 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)a1) >> 3)) >> 1);
    sub_1EF8CB0(a1, &a1->m128i_i8[v2]);
    sub_1EF8CB0(&a1->m128i_i8[v2], a2);
    sub_1EF8AD0(
      a1,
      (__m128i *)((char *)a1 + v2),
      a2,
      0xCCCCCCCCCCCCCCCDLL * (v2 >> 3),
      0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)&a1->m128i_i64[(unsigned __int64)v2 / 8]) >> 3));
  }
}
