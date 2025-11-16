// Function: sub_28ECAD0
// Address: 0x28ecad0
//
__int64 __fastcall sub_28ECAD0(__m128i *src, __m128i *a2, char *a3, __int64 a4)
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
    sub_28E9D70(src, &src[v6], a3);
    sub_28E9D70(v7, a2, a3);
  }
  else
  {
    sub_28ECAD0(src);
    sub_28ECAD0(v7);
  }
  sub_28EC640(src, v7, (__int64)a2, v10 >> 4, a2 - v7, a3, a4);
  return v9;
}
