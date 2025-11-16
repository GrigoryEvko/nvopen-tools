// Function: sub_18F2AD0
// Address: 0x18f2ad0
//
__m128i *__fastcall sub_18F2AD0(const __m128i *a1, __int64 a2)
{
  __m128i *v3; // rax
  __int64 v4; // rdi
  __m128i *v5; // r13
  const __m128i *v6; // r12
  __m128i *v7; // rbx
  __m128i *v8; // r14
  __int32 v9; // eax
  __int64 v10; // rdi

  v3 = (__m128i *)sub_22077B0(48);
  v4 = a1[1].m128i_i64[1];
  v5 = v3;
  v3[2] = _mm_loadu_si128(a1 + 2);
  LODWORD(v3) = a1->m128i_i32[0];
  v5[1].m128i_i64[0] = 0;
  v5->m128i_i32[0] = (int)v3;
  v5[1].m128i_i64[1] = 0;
  v5->m128i_i64[1] = a2;
  if ( v4 )
    v5[1].m128i_i64[1] = sub_18F2AD0(v4, v5);
  v6 = (const __m128i *)a1[1].m128i_i64[0];
  if ( v6 )
  {
    v7 = v5;
    do
    {
      v8 = v7;
      v7 = (__m128i *)sub_22077B0(48);
      v7[2] = _mm_loadu_si128(v6 + 2);
      v9 = v6->m128i_i32[0];
      v7[1].m128i_i64[0] = 0;
      v7->m128i_i32[0] = v9;
      v7[1].m128i_i64[1] = 0;
      v8[1].m128i_i64[0] = (__int64)v7;
      v7->m128i_i64[1] = (__int64)v8;
      v10 = v6[1].m128i_i64[1];
      if ( v10 )
        v7[1].m128i_i64[1] = sub_18F2AD0(v10, v7);
      v6 = (const __m128i *)v6[1].m128i_i64[0];
    }
    while ( v6 );
  }
  return v5;
}
