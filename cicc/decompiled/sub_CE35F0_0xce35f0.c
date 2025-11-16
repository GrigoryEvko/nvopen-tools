// Function: sub_CE35F0
// Address: 0xce35f0
//
__m128i *__fastcall sub_CE35F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __m128i *result; // rax
  unsigned int v6; // r13d
  __int64 v7; // rcx
  const __m128i *v8; // rdx
  __int64 v9; // rsi
  const __m128i *i; // rsi

  sub_C8CF70(a1, (void *)(a1 + 32), 8, a2 + 32, a2);
  result = (__m128i *)(a1 + 112);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x800000000LL;
  v6 = *(_DWORD *)(a2 + 104);
  if ( v6 && a1 + 96 != a2 + 96 )
  {
    v7 = *(_QWORD *)(a2 + 96);
    v8 = (const __m128i *)(a2 + 112);
    if ( v7 == a2 + 112 )
    {
      v9 = v6;
      if ( v6 > 8 )
      {
        sub_CE3550(a1 + 96, v6, (__int64)v8, v7, v3, v4);
        result = *(__m128i **)(a1 + 96);
        v8 = *(const __m128i **)(a2 + 96);
        v9 = *(unsigned int *)(a2 + 104);
      }
      for ( i = (const __m128i *)((char *)v8 + 40 * v9); i != v8; result = (__m128i *)((char *)result + 40) )
      {
        if ( result )
        {
          *result = _mm_loadu_si128(v8);
          result[1] = _mm_loadu_si128(v8 + 1);
          result[2].m128i_i64[0] = v8[2].m128i_i64[0];
        }
        v8 = (const __m128i *)((char *)v8 + 40);
      }
      *(_DWORD *)(a1 + 104) = v6;
      *(_DWORD *)(a2 + 104) = 0;
    }
    else
    {
      result = (__m128i *)*(unsigned int *)(a2 + 108);
      *(_DWORD *)(a1 + 104) = v6;
      *(_QWORD *)(a1 + 96) = v7;
      *(_DWORD *)(a1 + 108) = (_DWORD)result;
      *(_QWORD *)(a2 + 96) = v8;
      *(_QWORD *)(a2 + 104) = 0;
    }
  }
  return result;
}
