// Function: sub_F1FCA0
// Address: 0xf1fca0
//
__m128i *__fastcall sub_F1FCA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *result; // rax
  unsigned int v11; // r13d
  __int64 v12; // rdi
  __int64 v13; // rsi
  const __m128i *v14; // rdx
  const __m128i *i; // rsi

  sub_C8CD80(a1, a1 + 32, a2, a4, a5, a6);
  result = (__m128i *)(a1 + 112);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x800000000LL;
  v11 = *(_DWORD *)(a2 + 104);
  if ( v11 )
  {
    v12 = a1 + 96;
    if ( a1 + 96 != a2 + 96 )
    {
      v13 = v11;
      if ( v11 > 8 )
      {
        sub_CE3550(v12, v11, a2 + 96, 0x800000000LL, v8, v9);
        result = *(__m128i **)(a1 + 96);
        v13 = *(unsigned int *)(a2 + 104);
      }
      v14 = *(const __m128i **)(a2 + 96);
      for ( i = (const __m128i *)((char *)v14 + 40 * v13); i != v14; result = (__m128i *)((char *)result + 40) )
      {
        if ( result )
        {
          *result = _mm_loadu_si128(v14);
          result[1] = _mm_loadu_si128(v14 + 1);
          result[2].m128i_i64[0] = v14[2].m128i_i64[0];
        }
        v14 = (const __m128i *)((char *)v14 + 40);
      }
      *(_DWORD *)(a1 + 104) = v11;
    }
  }
  return result;
}
