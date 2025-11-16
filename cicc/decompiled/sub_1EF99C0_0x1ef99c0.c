// Function: sub_1EF99C0
// Address: 0x1ef99c0
//
__int64 __fastcall sub_1EF99C0(__m128i *a1, const __m128i *a2, __m128i *a3, __int64 a4)
{
  __int64 v6; // rax
  __m128i *v7; // rbx
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)a1) >> 3) + 1) / 2;
  v10 = 40 * v6;
  v7 = (__m128i *)((char *)a1 + 40 * v6);
  if ( v6 <= a4 )
  {
    sub_1EF8610(a1, (__m128i *)((char *)a1 + 40 * v6), a3);
    sub_1EF8610(v7, a2, a3);
  }
  else
  {
    sub_1EF99C0(a1, (char *)a1 + 40 * v6, a3);
    sub_1EF99C0(v7, a2, a3);
  }
  sub_1EF8D60(
    (__int64)a1,
    v7,
    (__int64)a2,
    0xCCCCCCCCCCCCCCCDLL * (v10 >> 3),
    0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)v7) >> 3),
    a3,
    a4);
  return v9;
}
