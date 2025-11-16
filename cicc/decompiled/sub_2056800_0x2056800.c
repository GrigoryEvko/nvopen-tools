// Function: sub_2056800
// Address: 0x2056800
//
__m128i *__fastcall sub_2056800(__int64 a1, __m128i *a2, const __m128i *a3)
{
  __m128i *result; // rax
  __m128i *v5; // rdi
  __int8 *v6; // r13
  __int64 v7; // rax
  __m128i v8; // xmm5
  __int64 v9; // rax
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // [rsp+8h] [rbp-48h] BYREF
  __m128i v13; // [rsp+18h] [rbp-38h] BYREF
  __int64 v14; // [rsp+28h] [rbp-28h]

  result = a2;
  v5 = *(__m128i **)(a1 + 8);
  v6 = &a2->m128i_i8[-*(_QWORD *)a1];
  if ( v5 == *(__m128i **)(a1 + 16) )
  {
    sub_1D27190((const __m128i **)a1, a2, a3);
    return (__m128i *)&v6[*(_QWORD *)a1];
  }
  else if ( v5 == a2 )
  {
    if ( v5 )
    {
      *v5 = _mm_loadu_si128(a3);
      v5[1] = _mm_loadu_si128(a3 + 1);
      v5[2].m128i_i64[0] = a3[2].m128i_i64[0];
      v5 = *(__m128i **)(a1 + 8);
      result = (__m128i *)&v6[*(_QWORD *)a1];
    }
    *(_QWORD *)(a1 + 8) = (char *)v5 + 40;
  }
  else
  {
    v7 = a3[2].m128i_i64[0];
    v12 = _mm_loadu_si128(a3);
    v14 = v7;
    v13 = _mm_loadu_si128(a3 + 1);
    if ( v5 )
    {
      v8 = _mm_loadu_si128((__m128i *)((char *)v5 - 24));
      v9 = v5[-1].m128i_i64[1];
      *v5 = _mm_loadu_si128((__m128i *)((char *)v5 - 40));
      v5[2].m128i_i64[0] = v9;
      v5[1] = v8;
      v5 = *(__m128i **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = (char *)v5 + 40;
    if ( a2 != (__m128i *)&v5[-3].m128i_u64[1] )
      memmove(&a2[2].m128i_u64[1], a2, (char *)&v5[-3].m128i_u64[1] - (char *)a2);
    v10 = _mm_loadu_si128(&v12);
    v11 = _mm_loadu_si128(&v13);
    a2[2].m128i_i32[0] = v14;
    *a2 = v10;
    a2[1] = v11;
    return (__m128i *)&v6[*(_QWORD *)a1];
  }
  return result;
}
