// Function: sub_1977E60
// Address: 0x1977e60
//
__m128i *__fastcall sub_1977E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i *result; // rax
  __int64 v8; // rbx
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  __m128i *v13; // rsi
  __m128i *v14; // rsi
  __int64 v15; // rdx
  __m128i v16; // [rsp+0h] [rbp-30h] BYREF

  result = (__m128i *)(a1 - 24);
  while ( 1 )
  {
    v8 = result->m128i_i64[0];
    if ( a2 == result->m128i_i64[0] )
      break;
    result = (__m128i *)((char *)result - 24);
    if ( (__m128i *)(a1 - 48 - 24LL * ((*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 3)) == result )
      return result;
  }
  if ( v8 )
  {
    v9 = result->m128i_i64[1];
    v10 = result[1].m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  result->m128i_i64[0] = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 8);
    result->m128i_i64[1] = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (unsigned __int64)&result->m128i_u64[1] | *(_QWORD *)(v11 + 16) & 3LL;
    result[1].m128i_i64[0] = (a3 + 8) | result[1].m128i_i64[0] & 3;
    *(_QWORD *)(a3 + 8) = result;
  }
  v12 = *(_QWORD *)(a1 + 40);
  v13 = *(__m128i **)(a4 + 8);
  v16.m128i_i64[1] = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v16.m128i_i64[0] = v12;
  result = *(__m128i **)(a4 + 16);
  if ( v13 == result )
  {
    sub_17F2860((const __m128i **)a4, v13, &v16);
    result = *(__m128i **)(a1 + 40);
    v16.m128i_i64[1] = v8 | 4;
    v14 = *(__m128i **)(a4 + 8);
    v16.m128i_i64[0] = (__int64)result;
    if ( v14 != *(__m128i **)(a4 + 16) )
    {
      if ( !v14 )
        goto LABEL_17;
      goto LABEL_16;
    }
  }
  else
  {
    if ( v13 )
    {
      *v13 = _mm_loadu_si128(&v16);
      v13 = *(__m128i **)(a4 + 8);
      result = *(__m128i **)(a4 + 16);
    }
    v14 = v13 + 1;
    *(_QWORD *)(a4 + 8) = v14;
    v15 = *(_QWORD *)(a1 + 40);
    v16.m128i_i64[1] = v8 | 4;
    v16.m128i_i64[0] = v15;
    if ( v14 != result )
    {
LABEL_16:
      *v14 = _mm_loadu_si128(&v16);
      v14 = *(__m128i **)(a4 + 8);
LABEL_17:
      *(_QWORD *)(a4 + 8) = v14 + 1;
      return result;
    }
  }
  return (__m128i *)sub_17F2860((const __m128i **)a4, v14, &v16);
}
