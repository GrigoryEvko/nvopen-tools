// Function: sub_EF89A0
// Address: 0xef89a0
//
__m128i *__fastcall sub_EF89A0(const __m128i *a1, __int64 a2)
{
  __m128i *v3; // rax
  __m128i v4; // xmm4
  __m128i *v5; // r14
  __m128i v6; // xmm5
  const __m128i *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __m128i *v13; // rax
  __m128i *v14; // rcx
  __m128i *v15; // rdx
  __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  const __m128i *v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r12
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  const __m128i *v36; // rdi
  __m128i *v37; // rax
  __m128i *v38; // rcx
  __m128i *v39; // rdx
  __m128i *v40; // rax
  __m128i *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int32 v49; // eax
  __int64 v50; // rdi

  v3 = (__m128i *)sub_22077B0(224);
  v4 = _mm_loadu_si128(a1 + 4);
  v5 = v3;
  v6 = _mm_loadu_si128(a1 + 5);
  v7 = (const __m128i *)a1[8].m128i_i64[1];
  v3[2] = _mm_loadu_si128(a1 + 2);
  v8 = a1[3].m128i_i64[0];
  v5[8].m128i_i32[0] = 0;
  v5[3].m128i_i64[0] = v8;
  v9 = a1[3].m128i_i64[1];
  v5[8].m128i_i64[1] = 0;
  v5[3].m128i_i64[1] = v9;
  v10 = a1[6].m128i_i64[0];
  v5[9].m128i_i64[0] = (__int64)v5[8].m128i_i64;
  v5[6].m128i_i64[0] = v10;
  v11 = a1[6].m128i_i64[1];
  v5[9].m128i_i64[1] = (__int64)v5[8].m128i_i64;
  v5[6].m128i_i64[1] = v11;
  v12 = a1[7].m128i_i64[0];
  v5[10].m128i_i64[0] = 0;
  v5[7].m128i_i64[0] = v12;
  v5[4] = v4;
  v5[5] = v6;
  if ( v7 )
  {
    v13 = sub_EF8780(v7, (__int64)v5[8].m128i_i64);
    v14 = v13;
    do
    {
      v15 = v13;
      v13 = (__m128i *)v13[1].m128i_i64[0];
    }
    while ( v13 );
    v5[9].m128i_i64[0] = (__int64)v15;
    v16 = v14;
    do
    {
      v17 = (__int64)v16;
      v16 = (__m128i *)v16[1].m128i_i64[1];
    }
    while ( v16 );
    v18 = a1[10].m128i_i64[0];
    v5[9].m128i_i64[1] = v17;
    v5[8].m128i_i64[1] = (__int64)v14;
    v5[10].m128i_i64[0] = v18;
  }
  v19 = a1[11].m128i_i64[1];
  v5[11].m128i_i32[0] = 0;
  v5[11].m128i_i64[1] = 0;
  v5[12].m128i_i64[0] = (__int64)v5[11].m128i_i64;
  v5[12].m128i_i64[1] = (__int64)v5[11].m128i_i64;
  v5[13].m128i_i64[0] = 0;
  if ( v19 )
  {
    v20 = sub_EF8D70();
    v21 = v20;
    do
    {
      v22 = v20;
      v20 = *(_QWORD *)(v20 + 16);
    }
    while ( v20 );
    v5[12].m128i_i64[0] = v22;
    v23 = v21;
    do
    {
      v24 = v23;
      v23 = *(_QWORD *)(v23 + 24);
    }
    while ( v23 );
    v25 = a1[13].m128i_i64[0];
    v5[12].m128i_i64[1] = v24;
    v5[11].m128i_i64[1] = v21;
    v5[13].m128i_i64[0] = v25;
  }
  v26 = a1[13].m128i_i64[1];
  v27 = a1[1].m128i_i64[1];
  v5->m128i_i64[1] = a2;
  v5[1].m128i_i64[0] = 0;
  v5[13].m128i_i64[1] = v26;
  LODWORD(v26) = a1->m128i_i32[0];
  v5[1].m128i_i64[1] = 0;
  v5->m128i_i32[0] = v26;
  if ( v27 )
    v5[1].m128i_i64[1] = sub_EF89A0(v27, v5);
  v28 = (const __m128i *)a1[1].m128i_i64[0];
  if ( v28 )
  {
    v29 = (__int64)v5;
    do
    {
      v30 = v29;
      v29 = sub_22077B0(224);
      *(__m128i *)(v29 + 32) = _mm_loadu_si128(v28 + 2);
      v31 = _mm_loadu_si128(v28 + 4);
      v32 = _mm_loadu_si128(v28 + 5);
      *(_QWORD *)(v29 + 48) = v28[3].m128i_i64[0];
      v33 = v28[3].m128i_i64[1];
      *(__m128i *)(v29 + 64) = v31;
      *(_QWORD *)(v29 + 56) = v33;
      v34 = v28[6].m128i_i64[0];
      *(__m128i *)(v29 + 80) = v32;
      *(_QWORD *)(v29 + 96) = v34;
      *(_QWORD *)(v29 + 104) = v28[6].m128i_i64[1];
      v35 = v28[7].m128i_i64[0];
      *(_DWORD *)(v29 + 128) = 0;
      *(_QWORD *)(v29 + 112) = v35;
      *(_QWORD *)(v29 + 136) = 0;
      *(_QWORD *)(v29 + 144) = v29 + 128;
      *(_QWORD *)(v29 + 152) = v29 + 128;
      *(_QWORD *)(v29 + 160) = 0;
      v36 = (const __m128i *)v28[8].m128i_i64[1];
      if ( v36 )
      {
        v37 = sub_EF8780(v36, v29 + 128);
        v38 = v37;
        do
        {
          v39 = v37;
          v37 = (__m128i *)v37[1].m128i_i64[0];
        }
        while ( v37 );
        *(_QWORD *)(v29 + 144) = v39;
        v40 = v38;
        do
        {
          v41 = v40;
          v40 = (__m128i *)v40[1].m128i_i64[1];
        }
        while ( v40 );
        *(_QWORD *)(v29 + 152) = v41;
        v42 = v28[10].m128i_i64[0];
        *(_QWORD *)(v29 + 136) = v38;
        *(_QWORD *)(v29 + 160) = v42;
      }
      *(_DWORD *)(v29 + 176) = 0;
      *(_QWORD *)(v29 + 184) = 0;
      *(_QWORD *)(v29 + 192) = v29 + 176;
      *(_QWORD *)(v29 + 200) = v29 + 176;
      *(_QWORD *)(v29 + 208) = 0;
      if ( v28[11].m128i_i64[1] )
      {
        v43 = sub_EF8D70();
        v44 = v43;
        do
        {
          v45 = v43;
          v43 = *(_QWORD *)(v43 + 16);
        }
        while ( v43 );
        *(_QWORD *)(v29 + 192) = v45;
        v46 = v44;
        do
        {
          v47 = v46;
          v46 = *(_QWORD *)(v46 + 24);
        }
        while ( v46 );
        *(_QWORD *)(v29 + 200) = v47;
        v48 = v28[13].m128i_i64[0];
        *(_QWORD *)(v29 + 184) = v44;
        *(_QWORD *)(v29 + 208) = v48;
      }
      *(_QWORD *)(v29 + 216) = v28[13].m128i_i64[1];
      v49 = v28->m128i_i32[0];
      *(_QWORD *)(v29 + 16) = 0;
      *(_DWORD *)v29 = v49;
      *(_QWORD *)(v29 + 24) = 0;
      *(_QWORD *)(v30 + 16) = v29;
      *(_QWORD *)(v29 + 8) = v30;
      v50 = v28[1].m128i_i64[1];
      if ( v50 )
        *(_QWORD *)(v29 + 24) = sub_EF89A0(v50, v29);
      v28 = (const __m128i *)v28[1].m128i_i64[0];
    }
    while ( v28 );
  }
  return v5;
}
