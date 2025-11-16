// Function: sub_1381540
// Address: 0x1381540
//
__m128i *__fastcall sub_1381540(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r11
  __int64 v8; // r12
  __int64 v9; // rcx
  _QWORD *v10; // rdx
  __m128i *result; // rax
  __int64 v12; // rdx
  __int64 v13; // [rsp+0h] [rbp-40h]

  v5 = (a3 - 1) / 2;
  v8 = a2;
  v13 = a3 & 1;
  if ( a2 >= v5 )
  {
    v9 = a2;
    result = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_15;
    goto LABEL_17;
  }
  while ( 1 )
  {
    v9 = 2 * (a2 + 1) - 1;
    v10 = (_QWORD *)(a1 + 32 * (a2 + 1));
    result = (__m128i *)(a1 + 16 * v9);
    if ( *v10 >= result->m128i_i64[0] && (v10[1] >= result->m128i_i64[1] || *v10 != result->m128i_i64[0]) )
    {
      result = (__m128i *)(a1 + 32 * (a2 + 1));
      v9 = 2 * (a2 + 1);
    }
    *(__m128i *)(a1 + 16 * a2) = _mm_loadu_si128(result);
    if ( v9 >= v5 )
      break;
    a2 = v9;
  }
  if ( !v13 )
  {
LABEL_17:
    if ( (a3 - 2) / 2 == v9 )
    {
      v9 = 2 * v9 + 1;
      *result = _mm_loadu_si128((const __m128i *)(a1 + 16 * v9));
      result = (__m128i *)(a1 + 16 * v9);
    }
  }
  v12 = (v9 - 1) / 2;
  if ( v9 > v8 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 16 * v12);
      if ( result->m128i_i64[0] >= a4 && (result->m128i_i64[1] >= a5 || result->m128i_i64[0] != a4) )
        break;
      *(__m128i *)(a1 + 16 * v9) = _mm_loadu_si128(result);
      v9 = v12;
      if ( v8 >= v12 )
        goto LABEL_15;
      v12 = (v12 - 1) / 2;
    }
    result = (__m128i *)(a1 + 16 * v9);
  }
LABEL_15:
  result->m128i_i64[0] = a4;
  result->m128i_i64[1] = a5;
  return result;
}
