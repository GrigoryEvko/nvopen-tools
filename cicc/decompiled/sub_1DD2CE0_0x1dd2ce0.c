// Function: sub_1DD2CE0
// Address: 0x1dd2ce0
//
__m128i *__fastcall sub_1DD2CE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rsi
  __int64 v11; // r13
  __int64 i; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r14
  __m128i *result; // rax
  __int64 v18; // r15
  __int32 v19; // r15d
  __m128i *v20; // rcx
  __int64 v21; // rsi
  __m128i *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rcx

  v10 = (a3 - 1) / 2;
  v11 = a3 & 1;
  if ( a2 >= v10 )
  {
    result = (__m128i *)(a1 + 24 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_25;
    v14 = a2;
    goto LABEL_22;
  }
  for ( i = a2; ; i = v14 )
  {
    v14 = 2 * (i + 1);
    v15 = 48 * (i + 1);
    v16 = a1 + v15 - 24;
    result = (__m128i *)(a1 + v15);
    v18 = *(_QWORD *)(v16 + 8);
    if ( result->m128i_i64[1] < v18
      || result->m128i_i64[1] == v18
      && ((v19 = *(_DWORD *)(v16 + 16), result[1].m128i_i32[0] < v19)
       || result[1].m128i_i32[0] == v19 && result[1].m128i_i32[1] < *(_DWORD *)(v16 + 20)) )
    {
      --v14;
      result = (__m128i *)(a1 + 24 * v14);
    }
    v20 = (__m128i *)(a1 + 24 * i);
    *v20 = _mm_loadu_si128(result);
    v20[1].m128i_i64[0] = result[1].m128i_i64[0];
    if ( v14 >= v10 )
      break;
  }
  if ( !v11 )
  {
LABEL_22:
    if ( (a3 - 2) / 2 == v14 )
    {
      v23 = v14 + 1;
      v24 = 2 * (v14 + 1);
      v25 = v24 + 4 * v23;
      v14 = v24 - 1;
      v26 = a1 + 8 * v25;
      *result = _mm_loadu_si128((const __m128i *)(v26 - 24));
      result[1].m128i_i64[0] = *(_QWORD *)(v26 - 8);
      result = (__m128i *)(a1 + 24 * v14);
    }
  }
  v21 = (v14 - 1) / 2;
  if ( v14 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 24 * v21);
      if ( result->m128i_i64[1] >= a8
        && (result->m128i_i64[1] != a8
         || result[1].m128i_i32[0] >= (int)a9
         && (result[1].m128i_i32[0] != (_DWORD)a9 || result[1].m128i_i32[1] >= HIDWORD(a9))) )
      {
        break;
      }
      v22 = (__m128i *)(a1 + 24 * v14);
      *v22 = _mm_loadu_si128(result);
      v22[1].m128i_i64[0] = result[1].m128i_i64[0];
      v14 = v21;
      if ( a2 >= v21 )
        goto LABEL_25;
      v21 = (v21 - 1) / 2;
    }
    result = (__m128i *)(a1 + 24 * v14);
  }
LABEL_25:
  result->m128i_i64[0] = a7;
  result->m128i_i64[1] = a8;
  result[1].m128i_i64[0] = a9;
  return result;
}
