// Function: sub_18A35E0
// Address: 0x18a35e0
//
__m128i *__fastcall sub_18A35E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 i; // rcx
  __int64 v10; // rdx
  __m128i *result; // rax
  __int64 v12; // rax
  _OWORD *v13; // r10
  __int64 v14; // r10
  __int64 v15; // rcx

  v7 = a3 & 1;
  v8 = (a3 - 1) / 2;
  if ( a2 >= v8 )
  {
    result = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_15;
    v10 = a2;
    goto LABEL_18;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    v12 = 32 * (i + 1);
    v13 = (_OWORD *)(a1 + v12 - 16);
    result = (__m128i *)(a1 + v12);
    if ( *(_OWORD *)result > *v13 )
    {
      --v10;
      result = (__m128i *)(a1 + 16 * v10);
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(result);
    if ( v10 >= v8 )
      break;
  }
  if ( !v7 )
  {
LABEL_18:
    if ( (a3 - 2) / 2 == v10 )
    {
      v15 = v10 + 1;
      v10 = 2 * (v10 + 1) - 1;
      *result = _mm_loadu_si128((const __m128i *)(a1 + 32 * v15 - 16));
      result = (__m128i *)(a1 + 16 * v10);
    }
  }
  v14 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 16 * v14);
      if ( *(_OWORD *)result <= __PAIR128__(a5, a4) )
        break;
      *(__m128i *)(a1 + 16 * v10) = _mm_loadu_si128(result);
      v10 = v14;
      if ( a2 >= v14 )
        goto LABEL_15;
      v14 = (v14 - 1) / 2;
    }
    result = (__m128i *)(a1 + 16 * v10);
  }
LABEL_15:
  result->m128i_i64[0] = a4;
  result->m128i_i64[1] = a5;
  return result;
}
