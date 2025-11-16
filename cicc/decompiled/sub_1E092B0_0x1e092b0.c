// Function: sub_1E092B0
// Address: 0x1e092b0
//
__m128i *__fastcall sub_1E092B0(__int64 a1, __m128i *a2, const __m128i *a3)
{
  __m128i *result; // rax
  __m128i *v6; // rdi
  __int8 *v7; // r14
  __m128i v8; // xmm3
  __int64 v9; // rax

  result = a2;
  v6 = *(__m128i **)(a1 + 8);
  v7 = &a2->m128i_i8[-*(_QWORD *)a1];
  if ( v6 == *(__m128i **)(a1 + 16) )
  {
    sub_1E08EE0((const __m128i **)a1, a2, a3);
    return (__m128i *)&v7[*(_QWORD *)a1];
  }
  else if ( v6 == a2 )
  {
    if ( v6 )
    {
      *v6 = _mm_loadu_si128(a3);
      v6[1] = _mm_loadu_si128(a3 + 1);
      v6[2].m128i_i64[0] = a3[2].m128i_i64[0];
      v6 = *(__m128i **)(a1 + 8);
      result = (__m128i *)&v7[*(_QWORD *)a1];
    }
    *(_QWORD *)(a1 + 8) = (char *)v6 + 40;
  }
  else
  {
    if ( v6 )
    {
      v8 = _mm_loadu_si128((__m128i *)((char *)v6 - 24));
      v9 = v6[-1].m128i_i64[1];
      *v6 = _mm_loadu_si128((__m128i *)((char *)v6 - 40));
      v6[2].m128i_i64[0] = v9;
      v6[1] = v8;
      v6 = *(__m128i **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = (char *)v6 + 40;
    if ( a2 != (__m128i *)&v6[-3].m128i_u64[1] )
      memmove(&a2[2].m128i_u64[1], a2, (char *)&v6[-3].m128i_u64[1] - (char *)a2);
    *a2 = _mm_loadu_si128(a3);
    a2[1] = _mm_loadu_si128(a3 + 1);
    a2[2].m128i_i32[0] = a3[2].m128i_i32[0];
    a2[2].m128i_i8[4] = a3[2].m128i_i8[4];
    return (__m128i *)&v7[*(_QWORD *)a1];
  }
  return result;
}
