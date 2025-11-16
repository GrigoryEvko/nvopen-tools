// Function: sub_325DF30
// Address: 0x325df30
//
__m128i *__fastcall sub_325DF30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r11
  __int64 v7; // r12
  __int64 i; // r9
  __int64 v10; // rdx
  __m128i *result; // rax
  __int64 v12; // r9
  __m128i *v13; // rcx
  __int64 v14; // rcx

  v6 = (a3 - 1) / 2;
  v7 = a3 & 1;
  if ( a2 >= v6 )
  {
    result = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v10 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    result = (__m128i *)(a1 + 32 * (i + 1));
    if ( result->m128i_i64[1] < result[-1].m128i_i64[1] )
    {
      --v10;
      result = (__m128i *)(a1 + 16 * v10);
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(result);
    if ( v10 >= v6 )
      break;
  }
  if ( !v7 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v10 )
    {
      v14 = v10 + 1;
      v10 = 2 * (v10 + 1) - 1;
      *result = _mm_loadu_si128((const __m128i *)(a1 + 32 * v14 - 16));
      result = (__m128i *)(a1 + 16 * v10);
    }
  }
  v12 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 16 * v10);
      v13 = (__m128i *)(a1 + 16 * v12);
      if ( v13->m128i_i64[1] >= a5 )
        break;
      *result = _mm_loadu_si128(v13);
      v10 = v12;
      if ( a2 >= v12 )
      {
        v13->m128i_i64[0] = a4;
        v13->m128i_i64[1] = a5;
        return (__m128i *)(a1 + 16 * v12);
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  result->m128i_i64[0] = a4;
  result->m128i_i64[1] = a5;
  return result;
}
