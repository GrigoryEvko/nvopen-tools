// Function: sub_1079160
// Address: 0x1079160
//
__int64 __fastcall sub_1079160(__m128i *src, __m128i *a2, __m128i *a3, const __m128i *a4)
{
  __int64 v6; // rax
  __m128i *v7; // rbx
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)src) >> 3) + 1) / 2;
  v10 = 40 * v6;
  v7 = (__m128i *)((char *)src + 40 * v6);
  if ( v6 <= (__int64)a4 )
  {
    sub_1077800(src->m128i_i8, (__m128i *)((char *)src + 40 * v6), a3);
    sub_1077800(v7->m128i_i8, a2, a3);
  }
  else
  {
    sub_1079160(src);
    sub_1079160(v7);
  }
  sub_1078C30(
    src,
    v7,
    (__int64)a2,
    0xCCCCCCCCCCCCCCCDLL * (v10 >> 3),
    0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)v7) >> 3),
    a3,
    a4);
  return v9;
}
