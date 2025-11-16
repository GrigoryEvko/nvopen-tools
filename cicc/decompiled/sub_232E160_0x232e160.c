// Function: sub_232E160
// Address: 0x232e160
//
__m128i *__fastcall sub_232E160(__m128i *a1, __m128i *a2, _WORD *a3, size_t a4)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rbx
  __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  __m128i v11; // xmm0

  v5 = sub_C931B0(a2->m128i_i64, a3, a4, 0);
  if ( v5 == -1 )
  {
    v11 = _mm_loadu_si128(a2);
    a1[1].m128i_i64[0] = 0;
    a1[1].m128i_i64[1] = 0;
    *a1 = v11;
    return a1;
  }
  else
  {
    v6 = a2->m128i_u64[1];
    v7 = v5 + a4;
    v8 = a2->m128i_i64[0];
    if ( v7 > v6 )
    {
      v7 = a2->m128i_u64[1];
      v9 = 0;
    }
    else
    {
      v9 = v6 - v7;
    }
    a1->m128i_i64[0] = v8;
    if ( v5 > v6 )
      v5 = v6;
    a1[1].m128i_i64[1] = v9;
    a1[1].m128i_i64[0] = v8 + v7;
    a1->m128i_i64[1] = v5;
    return a1;
  }
}
