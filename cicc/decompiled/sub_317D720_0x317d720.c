// Function: sub_317D720
// Address: 0x317d720
//
__m128i *__fastcall sub_317D720(const __m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __m128i *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __m128i v15; // xmm1
  __int64 v16; // rax
  __int64 v17; // rax
  const __m128i *v18; // r12
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __m128i v29; // xmm0
  __int64 v30; // rax
  __int64 v31; // rdi

  v3 = sub_22077B0(0x88u);
  v4 = a1[3].m128i_i64[1];
  v5 = (__m128i *)v3;
  v6 = a1[2].m128i_i64[0];
  v5[3].m128i_i32[0] = 0;
  v5[2].m128i_i64[0] = v6;
  v5[3].m128i_i64[1] = 0;
  v5[4].m128i_i64[0] = (__int64)v5[3].m128i_i64;
  v5[4].m128i_i64[1] = (__int64)v5[3].m128i_i64;
  v5[5].m128i_i64[0] = 0;
  if ( v4 )
  {
    v7 = ((__int64 (*)(void))sub_317D720)();
    v8 = v7;
    do
    {
      v9 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v7 );
    v5[4].m128i_i64[0] = v9;
    v10 = v8;
    do
    {
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 24);
    }
    while ( v10 );
    v12 = a1[5].m128i_i64[0];
    v5[4].m128i_i64[1] = v11;
    v5[3].m128i_i64[1] = v8;
    v5[5].m128i_i64[0] = v12;
  }
  v13 = a1[5].m128i_i64[1];
  v14 = a1[1].m128i_i64[1];
  v5[1].m128i_i64[0] = 0;
  v15 = _mm_loadu_si128(a1 + 6);
  v5[1].m128i_i64[1] = 0;
  v5[5].m128i_i64[1] = v13;
  v16 = a1[7].m128i_i64[0];
  v5->m128i_i64[1] = a2;
  v5[7].m128i_i64[0] = v16;
  v17 = a1[7].m128i_i64[1];
  v5[6] = v15;
  v5[7].m128i_i64[1] = v17;
  v5[8].m128i_i64[0] = a1[8].m128i_i64[0];
  v5->m128i_i32[0] = a1->m128i_i32[0];
  if ( v14 )
    v5[1].m128i_i64[1] = sub_317D720(v14, v5);
  v18 = (const __m128i *)a1[1].m128i_i64[0];
  if ( v18 )
  {
    v19 = (__int64)v5;
    do
    {
      v20 = v19;
      v19 = sub_22077B0(0x88u);
      v21 = v18[2].m128i_i64[0];
      *(_DWORD *)(v19 + 48) = 0;
      *(_QWORD *)(v19 + 32) = v21;
      *(_QWORD *)(v19 + 56) = 0;
      *(_QWORD *)(v19 + 64) = v19 + 48;
      *(_QWORD *)(v19 + 72) = v19 + 48;
      *(_QWORD *)(v19 + 80) = 0;
      v22 = v18[3].m128i_i64[1];
      if ( v22 )
      {
        v23 = sub_317D720(v22, v19 + 48);
        v24 = v23;
        do
        {
          v25 = v23;
          v23 = *(_QWORD *)(v23 + 16);
        }
        while ( v23 );
        *(_QWORD *)(v19 + 64) = v25;
        v26 = v24;
        do
        {
          v27 = v26;
          v26 = *(_QWORD *)(v26 + 24);
        }
        while ( v26 );
        *(_QWORD *)(v19 + 72) = v27;
        v28 = v18[5].m128i_i64[0];
        *(_QWORD *)(v19 + 56) = v24;
        *(_QWORD *)(v19 + 80) = v28;
      }
      v29 = _mm_loadu_si128(v18 + 6);
      *(_QWORD *)(v19 + 88) = v18[5].m128i_i64[1];
      v30 = v18[7].m128i_i64[0];
      *(__m128i *)(v19 + 96) = v29;
      *(_QWORD *)(v19 + 112) = v30;
      *(_QWORD *)(v19 + 120) = v18[7].m128i_i64[1];
      *(_QWORD *)(v19 + 128) = v18[8].m128i_i64[0];
      LODWORD(v30) = v18->m128i_i32[0];
      *(_QWORD *)(v19 + 16) = 0;
      *(_DWORD *)v19 = v30;
      *(_QWORD *)(v19 + 24) = 0;
      *(_QWORD *)(v20 + 16) = v19;
      *(_QWORD *)(v19 + 8) = v20;
      v31 = v18[1].m128i_i64[1];
      if ( v31 )
        *(_QWORD *)(v19 + 24) = sub_317D720(v31, v19);
      v18 = (const __m128i *)v18[1].m128i_i64[0];
    }
    while ( v18 );
  }
  return v5;
}
