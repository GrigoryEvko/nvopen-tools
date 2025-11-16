// Function: sub_EF8780
// Address: 0xef8780
//
__m128i *__fastcall sub_EF8780(const __m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  __m128i v4; // xmm1
  __m128i *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int32 v10; // eax
  __int64 v11; // rdi
  const __m128i *v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rax
  __int32 v19; // eax
  __int64 v20; // rdi

  v3 = sub_22077B0(104);
  v4 = _mm_loadu_si128(a1 + 5);
  v5 = (__m128i *)v3;
  v6 = a1[2].m128i_i64[0];
  v5[3].m128i_i64[0] = 0;
  v5[2].m128i_i64[0] = v6;
  v7 = a1[2].m128i_i64[1];
  v5[4].m128i_i64[0] = 0;
  v5[2].m128i_i64[1] = v7;
  v8 = a1[3].m128i_i64[1];
  v5[6].m128i_i64[0] = 0;
  v5[3].m128i_i64[1] = v8;
  v9 = a1[4].m128i_i64[1];
  v5[5] = v4;
  v5[4].m128i_i64[1] = v9;
  sub_EF8640((__m128i *)v5[3].m128i_i64, (__int64)a1[3].m128i_i64);
  v10 = a1->m128i_i32[0];
  v11 = a1[1].m128i_i64[1];
  v5->m128i_i64[1] = a2;
  v5[1].m128i_i64[0] = 0;
  v5->m128i_i32[0] = v10;
  v5[1].m128i_i64[1] = 0;
  if ( v11 )
    v5[1].m128i_i64[1] = sub_EF8780(v11, v5);
  v12 = (const __m128i *)a1[1].m128i_i64[0];
  if ( v12 )
  {
    v13 = (__int64)v5;
    do
    {
      v14 = v13;
      v13 = sub_22077B0(104);
      *(_QWORD *)(v13 + 32) = v12[2].m128i_i64[0];
      v15 = v12[2].m128i_i64[1];
      *(_QWORD *)(v13 + 48) = 0;
      *(_QWORD *)(v13 + 40) = v15;
      v16 = v12[3].m128i_i64[1];
      *(_QWORD *)(v13 + 64) = 0;
      v17 = _mm_loadu_si128(v12 + 5);
      *(_QWORD *)(v13 + 56) = v16;
      v18 = v12[4].m128i_i64[1];
      *(_QWORD *)(v13 + 96) = 0;
      *(_QWORD *)(v13 + 72) = v18;
      *(__m128i *)(v13 + 80) = v17;
      sub_EF8640((_QWORD *)(v13 + 48), (__int64)v12[3].m128i_i64);
      v19 = v12->m128i_i32[0];
      *(_QWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 24) = 0;
      *(_DWORD *)v13 = v19;
      *(_QWORD *)(v14 + 16) = v13;
      *(_QWORD *)(v13 + 8) = v14;
      v20 = v12[1].m128i_i64[1];
      if ( v20 )
        *(_QWORD *)(v13 + 24) = sub_EF8780(v20, v13);
      v12 = (const __m128i *)v12[1].m128i_i64[0];
    }
    while ( v12 );
  }
  return v5;
}
