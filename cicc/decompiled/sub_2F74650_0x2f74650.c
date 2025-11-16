// Function: sub_2F74650
// Address: 0x2f74650
//
__m128i *__fastcall sub_2F74650(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        __int64 a9)
{
  __m128i *result; // rax
  __int64 v10; // rcx
  __m128i *v11; // r8
  signed __int64 v12; // rcx
  __m128i *v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // r8d
  __m128i *v17; // rcx
  unsigned __int64 v18; // rdx
  __m128i v19; // xmm0

  result = *(__m128i **)a1;
  v10 = 24LL * *(unsigned int *)(a1 + 8);
  v11 = (__m128i *)(*(_QWORD *)a1 + v10);
  v12 = 0xAAAAAAAAAAAAAAABLL * (v10 >> 3);
  if ( v12 >> 2 )
  {
    v13 = &result[6 * (v12 >> 2)];
    while ( a7 != result->m128i_i32[0] )
    {
      if ( a7 == result[1].m128i_i32[2] )
      {
        result = (__m128i *)((char *)result + 24);
        goto LABEL_8;
      }
      if ( a7 == result[3].m128i_i32[0] )
      {
        result += 3;
        goto LABEL_8;
      }
      if ( a7 == result[4].m128i_i32[2] )
      {
        result = (__m128i *)((char *)result + 72);
        goto LABEL_8;
      }
      result += 6;
      if ( v13 == result )
      {
        v12 = 0xAAAAAAAAAAAAAAABLL * (((char *)v11 - (char *)result) >> 3);
        goto LABEL_18;
      }
    }
    goto LABEL_8;
  }
LABEL_18:
  if ( v12 == 2 )
  {
LABEL_25:
    if ( a7 != result->m128i_i32[0] )
    {
      result = (__m128i *)((char *)result + 24);
      goto LABEL_21;
    }
    goto LABEL_8;
  }
  if ( v12 != 3 )
  {
    if ( v12 != 1 )
      return result;
LABEL_21:
    if ( a7 != result->m128i_i32[0] )
      return result;
    goto LABEL_8;
  }
  if ( a7 != result->m128i_i32[0] )
  {
    result = (__m128i *)((char *)result + 24);
    goto LABEL_25;
  }
LABEL_8:
  if ( v11 != result )
  {
    v14 = result[1].m128i_i64[0] & ~a9;
    v15 = result->m128i_i64[1] & ~a8;
    result->m128i_i64[1] = v15;
    result[1].m128i_i64[0] = v14;
    if ( __PAIR128__(v15, v14) == 0 )
    {
      v16 = *(_DWORD *)(a1 + 8);
      v17 = (__m128i *)((char *)result + 24);
      v18 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)a1 + 24LL * v16 - (__int64)&result[1].m128i_i64[1]) >> 3);
      if ( *(_QWORD *)a1 + 24LL * v16 - (__int64)&result[1].m128i_i64[1] > 0 )
      {
        while ( 1 )
        {
          v19 = _mm_loadu_si128(result + 2);
          result->m128i_i32[0] = result[1].m128i_i32[2];
          *(__m128i *)((char *)result + 8) = v19;
          result = v17;
          if ( !--v18 )
            break;
          v17 = (__m128i *)((char *)v17 + 24);
        }
        v16 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v16 - 1;
    }
  }
  return result;
}
