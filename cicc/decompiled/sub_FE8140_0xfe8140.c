// Function: sub_FE8140
// Address: 0xfe8140
//
__m128i *__fastcall sub_FE8140(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 v8; // r12
  unsigned __int64 v9; // r9
  __int64 i; // r11
  __int64 v11; // rdx
  __m128i *result; // rax
  __int64 v13; // r11
  __m128i *v14; // rcx
  __int64 v15; // rcx

  v6 = a3 & 1;
  v8 = (a3 - 1) / 2;
  v9 = HIDWORD(a4);
  if ( a2 >= v8 )
  {
    result = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v11 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    result = (__m128i *)(a1 + 32 * (i + 1));
    if ( result->m128i_i32[1] < (unsigned __int32)result[-1].m128i_i32[1] )
    {
      --v11;
      result = (__m128i *)(a1 + 16 * v11);
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(result);
    if ( v11 >= v8 )
      break;
  }
  if ( !v6 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v11 )
    {
      v15 = v11 + 1;
      v11 = 2 * (v11 + 1) - 1;
      *result = _mm_loadu_si128((const __m128i *)(a1 + 32 * v15 - 16));
      result = (__m128i *)(a1 + 16 * v11);
    }
  }
  v13 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 16 * v11);
      v14 = (__m128i *)(a1 + 16 * v13);
      if ( v14->m128i_i32[1] >= (unsigned int)v9 )
        break;
      *result = _mm_loadu_si128(v14);
      v11 = v13;
      if ( a2 >= v13 )
      {
        v14->m128i_i64[0] = a4;
        v14->m128i_i64[1] = a5;
        return (__m128i *)(a1 + 16 * v13);
      }
      v13 = (v13 - 1) / 2;
    }
  }
LABEL_13:
  result->m128i_i64[0] = a4;
  result->m128i_i64[1] = a5;
  return result;
}
