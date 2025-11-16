// Function: sub_101EA00
// Address: 0x101ea00
//
unsigned __int8 *__fastcall sub_101EA00(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        unsigned __int8 *a4,
        const __m128i *a5,
        int a6)
{
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __int64 v12; // rax
  unsigned __int8 *v13; // r14
  unsigned __int8 *v14; // rax
  bool v15; // zf
  unsigned __int8 *result; // rax
  __m128i v18[4]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v19; // [rsp+60h] [rbp-40h]

  v9 = _mm_loadu_si128(a5 + 1);
  v10 = _mm_loadu_si128(a5 + 2);
  v11 = _mm_loadu_si128(a5 + 3);
  v12 = a5[4].m128i_i64[0];
  v18[0] = _mm_loadu_si128(a5);
  v19 = v12;
  v18[1] = v9;
  v18[2] = v10;
  v18[3] = v11;
  BYTE1(v19) = 0;
  v13 = sub_100F630(a4, a1, a2, v18, 0, 0, a6);
  if ( !v13 )
    v13 = a4;
  v14 = sub_100F630(a3, a1, a2, a5, 1u, 0, a6);
  if ( !v14 )
    v14 = a3;
  v15 = v13 == v14;
  result = 0;
  if ( v15 )
    return a4;
  return result;
}
