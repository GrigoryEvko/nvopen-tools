// Function: sub_298E970
// Address: 0x298e970
//
__int64 __fastcall sub_298E970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int32 v5; // eax
  unsigned int v6; // r8d
  __int32 v7; // r12d
  __int64 v8; // rcx
  __int64 v9; // rdi
  int v10; // r14d
  __int64 *v11; // r9
  unsigned int i; // eax
  _QWORD *v13; // r10
  __int64 v14; // r15
  unsigned int v15; // eax
  __int32 *v16; // r9
  __m128i *v17; // rsi
  __int32 v18; // r13d
  __int64 result; // rax
  __int64 v20; // rsi
  __m128i v21; // xmm2
  __m128i v22; // xmm1
  __m128i v23; // xmm0
  __m128i v24; // xmm3
  __int64 v25; // rcx
  __m128i *v26; // r12
  __m128i v27; // xmm7
  unsigned __int64 v28; // r15
  __int8 *v29; // r8
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdx
  bool v32; // cf
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  __m128i *v36; // r14
  unsigned __int64 v37; // rdx
  __m128i *v38; // r8
  __m128i v39; // xmm4
  __m128i v40; // xmm7
  __m128i v41; // xmm5
  __m128i v42; // xmm7
  __m128i v43; // xmm7
  __m128i *v44; // rcx
  const __m128i *v45; // rax
  int v46; // eax
  int v47; // edx
  __int64 v48; // rax
  __int64 v49; // [rsp+8h] [rbp-138h]
  __int64 v50; // [rsp+10h] [rbp-130h]
  __int64 v51; // [rsp+10h] [rbp-130h]
  unsigned __int64 v52; // [rsp+18h] [rbp-128h]
  unsigned __int64 v53; // [rsp+18h] [rbp-128h]
  __m128i v54[5]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v55; // [rsp+70h] [rbp-D0h]
  __m128i v56; // [rsp+80h] [rbp-C0h] BYREF
  __m128i v57; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v58; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i v59; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v60; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v61; // [rsp+D0h] [rbp-70h] BYREF

  v3 = a1 + 8;
  v5 = *(_DWORD *)a1;
  v6 = *(_DWORD *)(a1 + 32);
  v54[0].m128i_i64[0] = a2;
  v54[0].m128i_i64[1] = a3;
  v7 = v5 + 1;
  *(_DWORD *)a1 = v5 + 1;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 8);
    v56.m128i_i64[0] = 0;
    goto LABEL_53;
  }
  v8 = *(_QWORD *)(a1 + 16);
  v9 = a2;
  v10 = 1;
  v11 = 0;
  for ( i = (v6 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v6 - 1) & v15 )
  {
    v13 = (_QWORD *)(v8 + 24LL * i);
    v14 = *v13;
    if ( *v13 == a2 && v13[1] == a3 )
    {
      v16 = (__int32 *)(v13 + 2);
      goto LABEL_12;
    }
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && v13[1] == -8192 && !v11 )
      v11 = (__int64 *)(v8 + 24LL * i);
LABEL_9:
    v15 = v10 + i;
    ++v10;
  }
  if ( v13[1] != -4096 )
    goto LABEL_9;
  v46 = *(_DWORD *)(a1 + 24);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)(a1 + 8);
  v47 = v46 + 1;
  v56.m128i_i64[0] = (__int64)v11;
  if ( 4 * (v46 + 1) >= 3 * v6 )
  {
LABEL_53:
    sub_298E6A0(v3, 2 * v6);
    goto LABEL_54;
  }
  if ( v6 - *(_DWORD *)(a1 + 28) - v47 > v6 >> 3 )
    goto LABEL_47;
  sub_298E6A0(v3, v6);
LABEL_54:
  sub_298BE50(v3, v54[0].m128i_i64, (__int64 **)&v56);
  v9 = v54[0].m128i_i64[0];
  v11 = (__int64 *)v56.m128i_i64[0];
  v47 = *(_DWORD *)(a1 + 24) + 1;
LABEL_47:
  *(_DWORD *)(a1 + 24) = v47;
  if ( *v11 != -4096 || v11[1] != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v11 = v9;
  v48 = v54[0].m128i_i64[1];
  v16 = (__int32 *)(v11 + 2);
  *v16 = 0;
  *((_QWORD *)v16 - 1) = v48;
LABEL_12:
  *v16 = v7;
  v17 = *(__m128i **)(a1 + 48);
  if ( v17 == *(__m128i **)(a1 + 56) )
  {
    sub_298BF40((unsigned __int64 *)(a1 + 40), v17, v54);
  }
  else
  {
    if ( v17 )
    {
      *v17 = _mm_load_si128(v54);
      v17 = *(__m128i **)(a1 + 48);
    }
    *(_QWORD *)(a1 + 48) = v17 + 1;
  }
  v18 = *(_DWORD *)a1;
  sub_2989F30((__int64)&v56, (__int64)v54);
  result = v60.m128i_i64[0];
  v20 = v54[0].m128i_i64[0];
  v21 = _mm_loadu_si128(&v57);
  v22 = _mm_loadu_si128(&v58);
  v23 = _mm_loadu_si128(&v59);
  v24 = _mm_loadu_si128(&v56);
  v61.m128i_i64[0] = v60.m128i_i64[0];
  v55 = v60.m128i_i64[0];
  v25 = v54[0].m128i_i64[1];
  v26 = *(__m128i **)(a1 + 96);
  v54[1] = v24;
  v54[2] = v21;
  v54[3] = v22;
  v54[4] = v23;
  v57 = v24;
  v58 = v21;
  v59 = v22;
  v60 = v23;
  if ( v26 != *(__m128i **)(a1 + 104) )
  {
    if ( v26 )
    {
      v56 = v54[0];
      v27 = _mm_loadu_si128(&v56);
      v61.m128i_i32[2] = v18;
      *v26 = v27;
      v26[1] = _mm_loadu_si128(&v57);
      v26[2] = _mm_loadu_si128(&v58);
      v26[3] = _mm_loadu_si128(&v59);
      v26[4] = _mm_loadu_si128(&v60);
      v26[5] = _mm_loadu_si128(&v61);
      v26 = *(__m128i **)(a1 + 96);
    }
    *(_QWORD *)(a1 + 96) = v26 + 6;
    return result;
  }
  v28 = *(_QWORD *)(a1 + 88);
  v29 = &v26->m128i_i8[-v28];
  v30 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v26->m128i_i64 - v28) >> 5);
  if ( v30 == 0x155555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v31 = 1;
  if ( v30 )
    v31 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v26->m128i_i64 - v28) >> 5);
  v32 = __CFADD__(v31, v30);
  v33 = v31 - 0x5555555555555555LL * ((__int64)((__int64)v26->m128i_i64 - v28) >> 5);
  if ( v32 )
  {
    v34 = 0x7FFFFFFFFFFFFFE0LL;
    goto LABEL_28;
  }
  if ( v33 )
  {
    if ( v33 > 0x155555555555555LL )
      v33 = 0x155555555555555LL;
    v34 = 96 * v33;
LABEL_28:
    v49 = v54[0].m128i_i64[1];
    v50 = v54[0].m128i_i64[0];
    v52 = v34;
    v35 = sub_22077B0(v34);
    v20 = v50;
    v25 = v49;
    v29 = &v26->m128i_i8[-v28];
    v36 = (__m128i *)v35;
    v37 = v35 + v52;
    result = v35 + 96;
  }
  else
  {
    result = 96;
    v37 = 0;
    v36 = 0;
  }
  v38 = (__m128i *)&v29[(_QWORD)v36];
  if ( v38 )
  {
    v56.m128i_i64[0] = v20;
    v39 = _mm_loadu_si128(&v60);
    v56.m128i_i64[1] = v25;
    v40 = _mm_loadu_si128(&v56);
    v61.m128i_i32[2] = v18;
    v41 = _mm_loadu_si128(&v61);
    *v38 = v40;
    v42 = _mm_loadu_si128(&v57);
    v38[4] = v39;
    v38[1] = v42;
    v43 = _mm_loadu_si128(&v58);
    v38[5] = v41;
    v38[2] = v43;
    v38[3] = _mm_loadu_si128(&v59);
  }
  if ( v26 != (__m128i *)v28 )
  {
    v44 = v36;
    v45 = (const __m128i *)v28;
    do
    {
      if ( v44 )
      {
        *v44 = _mm_loadu_si128(v45);
        v44[1] = _mm_loadu_si128(v45 + 1);
        v44[2] = _mm_loadu_si128(v45 + 2);
        v44[3] = _mm_loadu_si128(v45 + 3);
        v44[4] = _mm_loadu_si128(v45 + 4);
        v44[5] = _mm_loadu_si128(v45 + 5);
      }
      v45 += 6;
      v44 += 6;
    }
    while ( v26 != v45 );
    result = (__int64)v36[6 * ((0x2AAAAAAAAAAAAABLL * (((unsigned __int64)&v26[-6] - v28) >> 5)) & 0x7FFFFFFFFFFFFFFLL)
                        + 12].m128i_i64;
  }
  if ( v28 )
  {
    v51 = result;
    v53 = v37;
    j_j___libc_free_0(v28);
    result = v51;
    v37 = v53;
  }
  *(_QWORD *)(a1 + 88) = v36;
  *(_QWORD *)(a1 + 96) = result;
  *(_QWORD *)(a1 + 104) = v37;
  return result;
}
