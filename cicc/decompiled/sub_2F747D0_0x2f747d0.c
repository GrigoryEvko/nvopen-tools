// Function: sub_2F747D0
// Address: 0x2f747d0
//
__m128i *__fastcall sub_2F747D0(
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
  __int64 v9; // rsi
  __m128i *v10; // rbx
  __m128i *v11; // r8
  signed __int64 v12; // rcx
  __m128i *result; // rax
  __m128i *v14; // rcx
  unsigned __int64 v15; // rdx
  const __m128i *v16; // r12
  const void *v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-18h]

  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(__m128i **)a1;
  v11 = (__m128i *)(*(_QWORD *)a1 + 24 * v9);
  v12 = 0xAAAAAAAAAAAAAAABLL * ((24 * v9) >> 3);
  if ( !(v12 >> 2) )
  {
    result = *(__m128i **)a1;
LABEL_14:
    if ( v12 != 2 )
    {
      if ( v12 != 3 )
      {
        if ( v12 != 1 )
          goto LABEL_17;
        goto LABEL_23;
      }
      if ( a7 == result->m128i_i32[0] )
        goto LABEL_8;
      result = (__m128i *)((char *)result + 24);
    }
    if ( a7 == result->m128i_i32[0] )
      goto LABEL_8;
    result = (__m128i *)((char *)result + 24);
LABEL_23:
    if ( a7 != result->m128i_i32[0] )
      goto LABEL_17;
    goto LABEL_8;
  }
  result = *(__m128i **)a1;
  v14 = &v10[6 * (v12 >> 2)];
  while ( a7 != result->m128i_i32[0] )
  {
    if ( a7 == result[1].m128i_i32[2] )
    {
      result = (__m128i *)((char *)result + 24);
      break;
    }
    if ( a7 == result[3].m128i_i32[0] )
    {
      result += 3;
      break;
    }
    if ( a7 == result[4].m128i_i32[2] )
    {
      result = (__m128i *)((char *)result + 72);
      break;
    }
    result += 6;
    if ( result == v14 )
    {
      a6 = 0xAAAAAAAAAAAAAAABLL;
      v12 = 0xAAAAAAAAAAAAAAABLL * (((char *)v11 - (char *)result) >> 3);
      goto LABEL_14;
    }
  }
LABEL_8:
  if ( v11 != result )
  {
    result->m128i_i64[1] |= a8;
    result[1].m128i_i64[0] |= a9;
    return result;
  }
LABEL_17:
  v15 = v9 + 1;
  v16 = (const __m128i *)&a7;
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v17 = (const void *)(a1 + 16);
    if ( v10 > (__m128i *)&a7 )
    {
      v18 = a1;
    }
    else
    {
      v18 = a1;
      if ( v11 > (__m128i *)&a7 )
      {
        sub_C8D5F0(a1, v17, v15, 0x18u, (__int64)v11, a6);
        v16 = (const __m128i *)(*(_QWORD *)a1 + (char *)&a7 - (char *)v10);
        v11 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
        goto LABEL_18;
      }
    }
    sub_C8D5F0(a1, v17, v15, 0x18u, (__int64)v11, a6);
    a1 = v18;
    v11 = (__m128i *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8));
  }
LABEL_18:
  *v11 = _mm_loadu_si128(v16);
  result = (__m128i *)v16[1].m128i_i64[0];
  v11[1].m128i_i64[0] = (__int64)result;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
