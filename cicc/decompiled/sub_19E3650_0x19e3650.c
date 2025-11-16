// Function: sub_19E3650
// Address: 0x19e3650
//
__m128i *__fastcall sub_19E3650(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int32 a8,
        __int64 a9,
        unsigned __int64 a10)
{
  __int64 v11; // r13
  __int64 i; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r12
  __m128i *result; // rax
  __int32 v17; // r15d
  __int32 v18; // r15d
  __m128i *v19; // rcx
  __int64 v20; // r12
  __m128i *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // r15
  __int64 v25; // [rsp+0h] [rbp-38h]

  v11 = (a3 - 1) / 2;
  v25 = a3 & 1;
  if ( a2 >= v11 )
  {
    result = (__m128i *)(a1 + 32 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_26;
    v13 = a2;
    goto LABEL_22;
  }
  for ( i = a2; ; i = v13 )
  {
    v13 = 2 * (i + 1);
    v14 = (i + 1) << 6;
    v15 = a1 + v14 - 32;
    result = (__m128i *)(a1 + v14);
    if ( result->m128i_i32[0] < *(_DWORD *)v15
      || result->m128i_i32[0] == *(_DWORD *)v15
      && ((v17 = *(_DWORD *)(v15 + 4), result->m128i_i32[1] < v17)
       || result->m128i_i32[1] == v17
       && ((v18 = *(_DWORD *)(v15 + 8), result->m128i_i32[2] < v18)
        || result->m128i_i32[2] == v18
        && ((v24 = *(_QWORD *)(v15 + 16), result[1].m128i_i64[0] < v24)
         || result[1].m128i_i64[0] == v24 && result[1].m128i_i64[1] < *(_QWORD *)(v15 + 24)))) )
    {
      --v13;
      result = (__m128i *)(a1 + 32 * v13);
    }
    v19 = (__m128i *)(a1 + 32 * i);
    *v19 = _mm_loadu_si128(result);
    v19[1] = _mm_loadu_si128(result + 1);
    if ( v13 >= v11 )
      break;
  }
  if ( !v25 )
  {
LABEL_22:
    if ( (a3 - 2) / 2 == v13 )
    {
      v22 = v13 + 1;
      v13 = 2 * (v13 + 1) - 1;
      v23 = a1 + (v22 << 6);
      *result = _mm_loadu_si128((const __m128i *)(v23 - 32));
      result[1] = _mm_loadu_si128((const __m128i *)(v23 - 16));
      result = (__m128i *)(a1 + 32 * v13);
    }
  }
  v20 = (v13 - 1) / 2;
  if ( v13 > a2 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 32 * v20);
      if ( result->m128i_i32[0] >= (int)a7
        && (result->m128i_i32[0] != (_DWORD)a7
         || SHIDWORD(a7) <= result->m128i_i32[1]
         && (HIDWORD(a7) != result->m128i_i32[1]
          || a8 <= result->m128i_i32[2]
          && (a8 != result->m128i_i32[2]
           || a9 <= result[1].m128i_i64[0] && (a9 != result[1].m128i_i64[0] || result[1].m128i_i64[1] >= a10)))) )
      {
        break;
      }
      v21 = (__m128i *)(a1 + 32 * v13);
      *v21 = _mm_loadu_si128(result);
      v21[1] = _mm_loadu_si128(result + 1);
      v13 = v20;
      if ( a2 >= v20 )
        goto LABEL_26;
      v20 = (v20 - 1) / 2;
    }
    result = (__m128i *)(a1 + 32 * v13);
  }
LABEL_26:
  result->m128i_i64[0] = a7;
  result->m128i_i32[2] = a8;
  result[1].m128i_i64[0] = a9;
  result[1].m128i_i64[1] = a10;
  return result;
}
