// Function: sub_2628F50
// Address: 0x2628f50
//
unsigned __int64 *__fastcall sub_2628F50(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int8 *v8; // rdx
  __int64 m128i_i64; // rbx
  __m128i *v10; // rax
  __m128i v11; // xmm2
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  const __m128i *v30; // r15
  __m128i *v31; // rbx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // r12
  unsigned __int64 v38; // r14
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __m128i *v44; // rdx
  __int64 v45; // r12
  unsigned __int64 v46; // r14
  unsigned __int64 v47; // rdi
  const __m128i *v48; // rax
  __int64 v49; // rdx
  __int32 v50; // ecx
  __m128i v51; // xmm0
  __int32 v52; // ecx
  unsigned __int64 v54; // rbx
  __int64 v55; // rax
  unsigned __int64 v56; // [rsp+8h] [rbp-58h]
  unsigned __int64 v58; // [rsp+18h] [rbp-48h]
  __m128i *v59; // [rsp+20h] [rbp-40h]
  const __m128i *v60; // [rsp+28h] [rbp-38h]

  v60 = (const __m128i *)a1[1];
  v58 = *a1;
  v4 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)((__int64)v60->m128i_i64 - *a1) >> 4);
  if ( v4 == 0xBA2E8BA2E8BA2ELL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)((__int64)v60->m128i_i64 - *a1) >> 4);
  v6 = __CFADD__(v5, v4);
  v7 = v5 + v4;
  v8 = &a2->m128i_i8[-v58];
  if ( v6 )
  {
    v54 = 0x7FFFFFFFFFFFFFA0LL;
  }
  else
  {
    if ( !v7 )
    {
      v56 = 0;
      m128i_i64 = 176;
      v59 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0xBA2E8BA2E8BA2ELL )
      v7 = 0xBA2E8BA2E8BA2ELL;
    v54 = 176 * v7;
  }
  v55 = sub_22077B0(v54);
  v8 = &a2->m128i_i8[-v58];
  v59 = (__m128i *)v55;
  v56 = v55 + v54;
  m128i_i64 = v55 + 176;
LABEL_7:
  v10 = (__m128i *)&v8[(_QWORD)v59];
  if ( &v8[(_QWORD)v59] )
  {
    v11 = _mm_loadu_si128(a3 + 1);
    v10->m128i_i64[0] = a3->m128i_i64[0];
    v12 = a3->m128i_i64[1];
    v10[1] = v11;
    v10->m128i_i64[1] = v12;
    v13 = a3[2].m128i_i64[0];
    a3[2].m128i_i64[0] = 0;
    v10[2].m128i_i64[0] = v13;
    v14 = a3[2].m128i_i64[1];
    a3[2].m128i_i64[1] = 0;
    v10[2].m128i_i64[1] = v14;
    v15 = a3[3].m128i_i64[0];
    a3[3].m128i_i64[0] = 0;
    v10[3].m128i_i64[0] = v15;
    v16 = a3[3].m128i_i64[1];
    a3[3].m128i_i64[1] = 0;
    v10[3].m128i_i64[1] = v16;
    v17 = a3[4].m128i_i64[0];
    a3[4].m128i_i64[0] = 0;
    v10[4].m128i_i64[0] = v17;
    v18 = a3[4].m128i_i64[1];
    a3[4].m128i_i64[1] = 0;
    v10[4].m128i_i64[1] = v18;
    v10[5].m128i_i64[0] = a3[5].m128i_i64[0];
    v19 = a3[5].m128i_i64[1];
    a3[5].m128i_i64[1] = 0;
    v10[5].m128i_i64[1] = v19;
    v20 = a3[6].m128i_i64[0];
    a3[6].m128i_i64[0] = 0;
    v10[6].m128i_i64[0] = v20;
    a3[5].m128i_i64[0] = 0;
    v21 = a3[6].m128i_i64[1];
    a3[6].m128i_i64[1] = 0;
    v10[6].m128i_i64[1] = v21;
    v22 = a3[7].m128i_i64[0];
    a3[7].m128i_i64[0] = 0;
    v10[7].m128i_i64[0] = v22;
    v23 = a3[7].m128i_i64[1];
    a3[7].m128i_i64[1] = 0;
    v10[7].m128i_i64[1] = v23;
    v24 = a3[8].m128i_i64[0];
    a3[8].m128i_i64[0] = 0;
    v10[8].m128i_i64[0] = v24;
    v25 = a3[8].m128i_i64[1];
    a3[8].m128i_i64[1] = 0;
    v10[8].m128i_i64[1] = v25;
    v26 = a3[9].m128i_i64[0];
    a3[9].m128i_i64[0] = 0;
    v10[9].m128i_i64[0] = v26;
    v27 = a3[9].m128i_i64[1];
    a3[9].m128i_i64[1] = 0;
    v10[9].m128i_i64[1] = v27;
    v28 = a3[10].m128i_i64[0];
    a3[10].m128i_i64[0] = 0;
    v10[10].m128i_i64[0] = v28;
    v29 = a3[10].m128i_i64[1];
    a3[10].m128i_i64[1] = 0;
    v10[10].m128i_i64[1] = v29;
  }
  v30 = (const __m128i *)v58;
  if ( a2 != (const __m128i *)v58 )
  {
    v31 = v59;
    if ( !v59 )
      goto LABEL_29;
LABEL_11:
    v31->m128i_i32[0] = v30->m128i_i32[0];
    v31->m128i_i32[1] = v30->m128i_i32[1];
    v31->m128i_i8[8] = v30->m128i_i8[8];
    v31->m128i_i8[9] = v30->m128i_i8[9];
    v31->m128i_i8[10] = v30->m128i_i8[10];
    v31->m128i_i8[11] = v30->m128i_i8[11];
    v31->m128i_i32[3] = v30->m128i_i32[3];
    v31[1] = _mm_loadu_si128(v30 + 1);
    v31[2].m128i_i64[0] = v30[2].m128i_i64[0];
    v31[2].m128i_i64[1] = v30[2].m128i_i64[1];
    v31[3].m128i_i64[0] = v30[3].m128i_i64[0];
    v32 = v30[3].m128i_i64[1];
    v30[3].m128i_i64[0] = 0;
    v30[2].m128i_i64[1] = 0;
    v30[2].m128i_i64[0] = 0;
    v31[3].m128i_i64[1] = v32;
    v31[4].m128i_i64[0] = v30[4].m128i_i64[0];
    v31[4].m128i_i64[1] = v30[4].m128i_i64[1];
    v30[4].m128i_i64[1] = 0;
    v30[4].m128i_i64[0] = 0;
    v33 = v30[5].m128i_i64[0];
    v30[3].m128i_i64[1] = 0;
    v31[5].m128i_i64[0] = v33;
    v31[5].m128i_i64[1] = v30[5].m128i_i64[1];
    v31[6].m128i_i64[0] = v30[6].m128i_i64[0];
    v34 = v30[6].m128i_i64[1];
    v30[6].m128i_i64[0] = 0;
    v30[5].m128i_i64[1] = 0;
    v30[5].m128i_i64[0] = 0;
    v31[6].m128i_i64[1] = v34;
    v31[7].m128i_i64[0] = v30[7].m128i_i64[0];
    v31[7].m128i_i64[1] = v30[7].m128i_i64[1];
    v35 = v30[8].m128i_i64[0];
    v30[7].m128i_i64[1] = 0;
    v30[7].m128i_i64[0] = 0;
    v30[6].m128i_i64[1] = 0;
    v31[8].m128i_i64[0] = v35;
    v31[8].m128i_i64[1] = v30[8].m128i_i64[1];
    v31[9].m128i_i64[0] = v30[9].m128i_i64[0];
    v36 = v30[9].m128i_i64[1];
    v30[9].m128i_i64[0] = 0;
    v30[8].m128i_i64[1] = 0;
    v30[8].m128i_i64[0] = 0;
    v31[9].m128i_i64[1] = v36;
    v31[10].m128i_i64[0] = v30[10].m128i_i64[0];
    v31[10].m128i_i64[1] = v30[10].m128i_i64[1];
    v30[10].m128i_i64[1] = 0;
    v30[10].m128i_i64[0] = 0;
    v30[9].m128i_i64[1] = 0;
    while ( 1 )
    {
      v37 = v30[8].m128i_i64[1];
      v38 = v30[8].m128i_u64[0];
      if ( v37 != v38 )
      {
        do
        {
          v39 = *(_QWORD *)(v38 + 16);
          if ( v39 )
            j_j___libc_free_0(v39);
          v38 += 40LL;
        }
        while ( v37 != v38 );
        v38 = v30[8].m128i_u64[0];
      }
      if ( v38 )
        j_j___libc_free_0(v38);
      v40 = v30[6].m128i_u64[1];
      if ( v40 )
        j_j___libc_free_0(v40);
      v41 = v30[5].m128i_u64[0];
      if ( v41 )
        j_j___libc_free_0(v41);
      v42 = v30[3].m128i_u64[1];
      if ( v42 )
        j_j___libc_free_0(v42);
      v43 = v30[2].m128i_u64[0];
      if ( v43 )
        j_j___libc_free_0(v43);
      v30 += 11;
      v44 = v31 + 11;
      if ( v30 == a2 )
        break;
      v31 += 11;
      if ( v44 )
        goto LABEL_11;
LABEL_29:
      v45 = v30[10].m128i_i64[0];
      v46 = v30[9].m128i_u64[1];
      if ( v45 != v46 )
      {
        do
        {
          v47 = *(_QWORD *)(v46 + 16);
          if ( v47 )
            j_j___libc_free_0(v47);
          v46 += 40LL;
        }
        while ( v45 != v46 );
        v46 = v30[9].m128i_u64[1];
      }
      if ( v46 )
        j_j___libc_free_0(v46);
    }
    m128i_i64 = (__int64)v31[22].m128i_i64;
  }
  if ( a2 != v60 )
  {
    v48 = a2;
    v49 = m128i_i64;
    do
    {
      v50 = v48->m128i_i32[0];
      v51 = _mm_loadu_si128(v48 + 1);
      v49 += 176;
      v48 += 11;
      *(_DWORD *)(v49 - 176) = v50;
      v52 = v48[-11].m128i_i32[1];
      *(__m128i *)(v49 - 160) = v51;
      *(_DWORD *)(v49 - 172) = v52;
      *(_BYTE *)(v49 - 168) = v48[-11].m128i_i8[8];
      *(_BYTE *)(v49 - 167) = v48[-11].m128i_i8[9];
      *(_BYTE *)(v49 - 166) = v48[-11].m128i_i8[10];
      *(_BYTE *)(v49 - 165) = v48[-11].m128i_i8[11];
      *(_DWORD *)(v49 - 164) = v48[-11].m128i_i32[3];
      *(_QWORD *)(v49 - 144) = v48[-9].m128i_i64[0];
      *(_QWORD *)(v49 - 136) = v48[-9].m128i_i64[1];
      *(_QWORD *)(v49 - 128) = v48[-8].m128i_i64[0];
      *(_QWORD *)(v49 - 120) = v48[-8].m128i_i64[1];
      *(_QWORD *)(v49 - 112) = v48[-7].m128i_i64[0];
      *(_QWORD *)(v49 - 104) = v48[-7].m128i_i64[1];
      *(_QWORD *)(v49 - 96) = v48[-6].m128i_i64[0];
      *(_QWORD *)(v49 - 88) = v48[-6].m128i_i64[1];
      *(_QWORD *)(v49 - 80) = v48[-5].m128i_i64[0];
      *(_QWORD *)(v49 - 72) = v48[-5].m128i_i64[1];
      *(_QWORD *)(v49 - 64) = v48[-4].m128i_i64[0];
      *(_QWORD *)(v49 - 56) = v48[-4].m128i_i64[1];
      *(_QWORD *)(v49 - 48) = v48[-3].m128i_i64[0];
      *(_QWORD *)(v49 - 40) = v48[-3].m128i_i64[1];
      *(_QWORD *)(v49 - 32) = v48[-2].m128i_i64[0];
      *(_QWORD *)(v49 - 24) = v48[-2].m128i_i64[1];
      *(_QWORD *)(v49 - 16) = v48[-1].m128i_i64[0];
      *(_QWORD *)(v49 - 8) = v48[-1].m128i_i64[1];
    }
    while ( v48 != v60 );
    m128i_i64 += 176
               * (((0xE8BA2E8BA2E8BA3LL * ((unsigned __int64)((char *)v48 - (char *)a2 - 176) >> 4))
                 & 0xFFFFFFFFFFFFFFFLL)
                + 1);
  }
  if ( v58 )
    j_j___libc_free_0(v58);
  a1[1] = m128i_i64;
  *a1 = (unsigned __int64)v59;
  a1[2] = v56;
  return a1;
}
