// Function: sub_1411670
// Address: 0x1411670
//
__m128i *__fastcall sub_1411670(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 i; // r10
  __int64 v8; // rcx
  __m128i *result; // rax
  __int64 v10; // r10
  __m128i *v11; // rdx

  v6 = (a3 - 1) / 2;
  if ( a2 >= v6 )
  {
    result = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v8 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    result = (__m128i *)(a1 + 32 * (i + 1));
    if ( result->m128i_i64[0] < *(_QWORD *)(a1 + 16 * (v8 - 1)) )
      result = (__m128i *)(a1 + 16 * --v8);
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(result);
    if ( v8 >= v6 )
      break;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v8 )
    {
      v8 = 2 * v8 + 1;
      *result = _mm_loadu_si128((const __m128i *)(a1 + 16 * v8));
      result = (__m128i *)(a1 + 16 * v8);
    }
  }
  v10 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 16 * v8);
      v11 = (__m128i *)(a1 + 16 * v10);
      if ( v11->m128i_i64[0] >= a4 )
        break;
      *result = _mm_loadu_si128(v11);
      v8 = v10;
      if ( a2 >= v10 )
      {
        v11->m128i_i64[0] = a4;
        v11->m128i_i64[1] = a5;
        return (__m128i *)(a1 + 16 * v10);
      }
      v10 = (v10 - 1) / 2;
    }
  }
LABEL_13:
  result->m128i_i64[0] = a4;
  result->m128i_i64[1] = a5;
  return result;
}
