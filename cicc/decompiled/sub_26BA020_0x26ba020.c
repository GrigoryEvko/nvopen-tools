// Function: sub_26BA020
// Address: 0x26ba020
//
__m128i *__fastcall sub_26BA020(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 i; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _OWORD *v12; // r10
  __m128i *result; // rax
  __int64 v14; // r10
  bool v15; // cc
  __int64 v16; // rcx

  v7 = a3 & 1;
  v8 = (a3 - 1) / 2;
  if ( a2 >= v8 )
  {
    result = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_19;
    v10 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    v11 = 32 * (i + 1);
    v12 = (_OWORD *)(a1 + v11 - 16);
    result = (__m128i *)(a1 + v11);
    if ( *(_OWORD *)result > *v12 )
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
LABEL_16:
    if ( (a3 - 2) / 2 == v10 )
    {
      v16 = v10 + 1;
      v10 = 2 * (v10 + 1) - 1;
      *result = _mm_loadu_si128((const __m128i *)(a1 + 32 * v16 - 16));
      result = (__m128i *)(a1 + 16 * v10);
    }
  }
  v14 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 16 * v14);
      v15 = result->m128i_i64[1] <= a5;
      if ( result->m128i_i64[1] == a5 )
        v15 = result->m128i_i64[0] <= a4;
      if ( v15 )
        break;
      *(__m128i *)(a1 + 16 * v10) = _mm_loadu_si128(result);
      v10 = v14;
      if ( a2 >= v14 )
        goto LABEL_19;
      v14 = (v14 - 1) / 2;
    }
    result = (__m128i *)(a1 + 16 * v10);
  }
LABEL_19:
  result->m128i_i64[0] = a4;
  result->m128i_i64[1] = a5;
  return result;
}
