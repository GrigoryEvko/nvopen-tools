// Function: sub_CF66C0
// Address: 0xcf66c0
//
__int64 __fastcall sub_CF66C0(_QWORD *a1, __int64 a2, __int64 a3, const __m128i *a4, unsigned __int8 a5)
{
  __int64 v5; // r14
  __int64 v6; // r13
  unsigned __int8 *v8; // rsi
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v14[3]; // [rsp+10h] [rbp-70h] BYREF
  char v15; // [rsp+40h] [rbp-40h]

  v5 = a2 + 24;
  v6 = *(_QWORD *)(a3 + 32);
  if ( a2 + 24 == v6 )
    return 0;
  while ( 1 )
  {
    v8 = (unsigned __int8 *)(v5 - 24);
    v9 = _mm_loadu_si128(a4);
    if ( !v5 )
      v8 = 0;
    v15 = 1;
    v10 = _mm_loadu_si128(a4 + 1);
    v11 = _mm_loadu_si128(a4 + 2);
    v14[0] = v9;
    v14[1] = v10;
    v14[2] = v11;
    if ( ((unsigned __int8)sub_CF6520(a1, v8, v14) & a5) != 0 )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v6 == v5 )
      return 0;
  }
  return 1;
}
