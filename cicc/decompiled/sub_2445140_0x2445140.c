// Function: sub_2445140
// Address: 0x2445140
//
__int64 __fastcall sub_2445140(__m128i *src, __m128i *a2, __m128i *a3, __int64 a4)
{
  __int64 v6; // r9
  __m128i *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (a2 - src + 1) / 2;
  v10 = v6 * 16;
  v7 = &src[v6];
  if ( (a2 - src + 1) / 2 <= a4 )
  {
    sub_2444090(src, &src[v6], a3);
    sub_2444090(v7, a2, a3);
  }
  else
  {
    sub_2445140(src);
    sub_2445140(v7);
  }
  sub_2444CC0(src, v7, (__int64)a2, v10 >> 4, a2 - v7, a3, (const __m128i *)a4);
  return v9;
}
