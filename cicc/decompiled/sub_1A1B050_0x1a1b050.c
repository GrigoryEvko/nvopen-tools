// Function: sub_1A1B050
// Address: 0x1a1b050
//
__m128i *__fastcall sub_1A1B050(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        __int64 a9)
{
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // r12
  __int64 i; // rdx
  __m128i *result; // rax
  __m128i *v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // rdx
  __m128i *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rcx

  v9 = a2;
  v10 = (a3 - 1) / 2;
  v11 = a3 & 1;
  if ( a2 < v10 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * (i + 1);
      v16 = 48 * (i + 1);
      v17 = (_QWORD *)(a1 + v16 - 24);
      result = (__m128i *)(a1 + v16);
      if ( result->m128i_i64[0] < *v17 )
        goto LABEL_4;
      if ( result->m128i_i64[0] <= *v17 )
      {
        v18 = (result[1].m128i_i64[0] >> 2) & 1;
        if ( (_BYTE)v18 != (((__int64)v17[2] >> 2) & 1) )
        {
          if ( (_BYTE)v18 )
            goto LABEL_5;
LABEL_4:
          --a2;
          result = (__m128i *)(a1 + 24 * a2);
          goto LABEL_5;
        }
        if ( result->m128i_i64[1] > v17[1] )
          goto LABEL_4;
      }
LABEL_5:
      v15 = (__m128i *)(a1 + 24 * i);
      *v15 = _mm_loadu_si128(result);
      v15[1].m128i_i64[0] = result[1].m128i_i64[0];
      if ( a2 >= v10 )
      {
        if ( v11 )
          goto LABEL_13;
        goto LABEL_24;
      }
    }
  }
  result = (__m128i *)(a1 + 24 * a2);
  if ( (a3 & 1) != 0 )
    goto LABEL_22;
LABEL_24:
  if ( (a3 - 2) / 2 == a2 )
  {
    v22 = a2 + 1;
    v23 = 6 * v22;
    a2 = 2 * v22 - 1;
    v24 = a1 + 8 * v23;
    *result = _mm_loadu_si128((const __m128i *)(v24 - 24));
    result[1].m128i_i64[0] = *(_QWORD *)(v24 - 8);
    result = (__m128i *)(a1 + 24 * a2);
  }
LABEL_13:
  v19 = (a2 - 1) / 2;
  if ( a2 > v9 )
  {
    while ( 1 )
    {
      result = (__m128i *)(a1 + 24 * v19);
      if ( result->m128i_i64[0] >= a7 )
      {
        if ( result->m128i_i64[0] > a7 )
          goto LABEL_21;
        v20 = (result[1].m128i_i64[0] >> 2) & 1;
        if ( (_BYTE)v20 == ((a9 >> 2) & 1) )
        {
          if ( result->m128i_i64[1] <= a8 )
          {
LABEL_21:
            result = (__m128i *)(a1 + 24 * a2);
            break;
          }
        }
        else if ( (_BYTE)v20 )
        {
          goto LABEL_21;
        }
      }
      v21 = (__m128i *)(a1 + 24 * a2);
      *v21 = _mm_loadu_si128(result);
      v21[1].m128i_i64[0] = result[1].m128i_i64[0];
      a2 = v19;
      if ( v9 >= v19 )
        break;
      v19 = (v19 - 1) / 2;
    }
  }
LABEL_22:
  result->m128i_i64[0] = a7;
  result->m128i_i64[1] = a8;
  result[1].m128i_i64[0] = a9;
  return result;
}
