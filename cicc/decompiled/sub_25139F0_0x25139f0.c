// Function: sub_25139F0
// Address: 0x25139f0
//
__int64 __fastcall sub_25139F0(__int64 a1, __int64 a2, const void *a3, unsigned __int64 a4, __m128i *a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v11; // rdx
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // r10d
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // rcx
  unsigned __int64 *v19; // r14
  unsigned __int64 v20; // r8
  __int64 result; // rax
  unsigned __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdi
  _QWORD *v25; // rax
  __int64 v26; // r8
  __int64 v27; // rax
  size_t v28; // r12
  void *v29; // rdi
  __m128i *v30; // r9
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdx
  __m128i v34; // xmm1
  __m128i v35; // xmm0
  __m128i v36; // xmm2
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __m128i v40; // xmm0
  __int64 v41; // rax
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rdi
  _QWORD *v44; // rax
  int v45; // ecx
  int v46; // edi
  unsigned __int64 v47; // rsi
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  _QWORD *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // [rsp+8h] [rbp-58h]
  __m128i *v53; // [rsp+8h] [rbp-58h]
  __m128i *v54; // [rsp+8h] [rbp-58h]
  __int64 v55; // [rsp+8h] [rbp-58h]
  __int64 v56; // [rsp+8h] [rbp-58h]
  unsigned __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __m128i *v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  __int64 v62; // [rsp+18h] [rbp-48h]
  int v63; // [rsp+18h] [rbp-48h]
  __int64 v64; // [rsp+18h] [rbp-48h]
  __int64 v65; // [rsp+20h] [rbp-40h] BYREF
  _QWORD *v66; // [rsp+28h] [rbp-38h] BYREF

  v7 = a1 + 168;
  v11 = *(_QWORD *)(a2 + 24);
  v12 = *(_DWORD *)(a1 + 192);
  v65 = v11;
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 168);
    v66 = 0;
    goto LABEL_42;
  }
  v13 = *(_QWORD *)(a1 + 176);
  v14 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v15 = v13 + 88LL * v14;
  v16 = *(_QWORD *)v15;
  if ( v11 != *(_QWORD *)v15 )
  {
    v63 = 1;
    v44 = 0;
    v55 = v13;
    while ( v16 != -4096 )
    {
      if ( !v44 && v16 == -8192 )
        v44 = (_QWORD *)v15;
      v14 = (v12 - 1) & (v63 + v14);
      v13 = (unsigned int)(v63 + 1);
      v15 = v55 + 88LL * v14;
      v16 = *(_QWORD *)v15;
      if ( v11 == *(_QWORD *)v15 )
        goto LABEL_3;
      ++v63;
    }
    if ( !v44 )
      v44 = (_QWORD *)v15;
    v45 = *(_DWORD *)(a1 + 184);
    ++*(_QWORD *)(a1 + 168);
    v46 = v45 + 1;
    v66 = v44;
    if ( 4 * (v45 + 1) < 3 * v12 )
    {
      v15 = v12 - *(_DWORD *)(a1 + 188) - v46;
      v13 = v12 >> 3;
      if ( (unsigned int)v15 > (unsigned int)v13 )
      {
LABEL_28:
        *(_DWORD *)(a1 + 184) = v46;
        if ( *v44 != -4096 )
          --*(_DWORD *)(a1 + 188);
        *v44 = v11;
        v17 = (__int64)(v44 + 1);
        v44[1] = v44 + 3;
        v44[2] = 0x800000000LL;
        goto LABEL_31;
      }
      v64 = a6;
LABEL_43:
      sub_25135D0(v7, v12);
      sub_25109B0(v7, &v65, &v66);
      v11 = v65;
      a6 = v64;
      v46 = *(_DWORD *)(a1 + 184) + 1;
      v44 = v66;
      goto LABEL_28;
    }
LABEL_42:
    v64 = a6;
    v12 *= 2;
    goto LABEL_43;
  }
LABEL_3:
  v17 = v15 + 8;
  if ( *(_DWORD *)(v15 + 16) )
  {
    v18 = *(_QWORD *)(v15 + 8);
    goto LABEL_5;
  }
LABEL_31:
  v47 = *(_QWORD *)(v65 + 104);
  if ( v47 )
  {
    v48 = *(unsigned int *)(v17 + 12);
    v49 = 0;
    if ( v47 > v48 )
    {
      v56 = a6;
      sub_25126F0(v17, v47, v48, v15, v13, a6);
      v49 = *(unsigned int *)(v17 + 8);
      a6 = v56;
    }
    v18 = *(_QWORD *)v17;
    v50 = (_QWORD *)(*(_QWORD *)v17 + 8 * v49);
    v51 = *(_QWORD *)v17 + 8 * v47;
    if ( v50 != (_QWORD *)v51 )
    {
      do
      {
        if ( v50 )
          *v50 = 0;
        ++v50;
      }
      while ( (_QWORD *)v51 != v50 );
      v18 = *(_QWORD *)v17;
    }
    *(_DWORD *)(v17 + 8) = v47;
  }
  else
  {
    v18 = *(_QWORD *)v17;
  }
LABEL_5:
  v19 = (unsigned __int64 *)(v18 + 8LL * *(unsigned int *)(a2 + 32));
  v20 = *v19;
  if ( *v19 )
  {
    result = 0;
    if ( *(unsigned int *)(v20 + 32) <= a4 )
      return result;
    *v19 = 0;
    v52 = a6;
    v58 = v20;
    sub_A17130(v20 + 136);
    sub_A17130(v58 + 104);
    v22 = v58;
    v23 = v52;
    v24 = *(_QWORD *)(v58 + 24);
    if ( v24 != v58 + 40 )
    {
      _libc_free(v24);
      v23 = v52;
      v22 = v58;
    }
    v59 = v23;
    j_j___libc_free_0(v22);
    a6 = v59;
  }
  v60 = (__m128i *)a6;
  v25 = (_QWORD *)sub_22077B0(0xA8u);
  v26 = (__int64)v25;
  if ( v25 )
  {
    *v25 = a1;
    v27 = *(_QWORD *)(a2 + 24);
    v28 = 8 * a4;
    v29 = (void *)(v26 + 40);
    *(_QWORD *)(v26 + 16) = a2;
    v30 = v60;
    *(_QWORD *)(v26 + 8) = v27;
    v31 = (__int64)(8 * a4) >> 3;
    *(_QWORD *)(v26 + 24) = v26 + 40;
    *(_QWORD *)(v26 + 32) = 0x800000000LL;
    if ( v28 > 0x40 )
    {
      v53 = v60;
      v61 = v26;
      sub_C8D5F0(v26 + 24, (const void *)(v26 + 40), v31, 8u, v26, (__int64)v30);
      v26 = v61;
      v30 = v53;
      v29 = (void *)(*(_QWORD *)(v61 + 24) + 8LL * *(unsigned int *)(v61 + 32));
    }
    else if ( !v28 )
    {
LABEL_13:
      v32 = a5[1].m128i_i64[0];
      v33 = *(_QWORD *)(v26 + 128);
      a5[1].m128i_i64[0] = 0;
      v34 = _mm_loadu_si128((const __m128i *)(v26 + 104));
      v35 = _mm_loadu_si128(a5);
      *(_DWORD *)(v26 + 32) = v28 + v31;
      v36 = _mm_loadu_si128((const __m128i *)(v26 + 136));
      *(_QWORD *)(v26 + 120) = v32;
      v37 = a5[1].m128i_i64[1];
      *a5 = v34;
      a5[1].m128i_i64[1] = v33;
      v38 = *(_QWORD *)(v26 + 160);
      *(_QWORD *)(v26 + 128) = v37;
      v39 = v30[1].m128i_i64[0];
      *(__m128i *)(v26 + 104) = v35;
      v40 = _mm_loadu_si128(v30);
      *(_QWORD *)(v26 + 152) = v39;
      v41 = v30[1].m128i_i64[1];
      v30[1].m128i_i64[0] = 0;
      v30[1].m128i_i64[1] = v38;
      *(_QWORD *)(v26 + 160) = v41;
      *v30 = v36;
      *(__m128i *)(v26 + 136) = v40;
      goto LABEL_14;
    }
    v54 = v30;
    v62 = v26;
    memcpy(v29, a3, v28);
    v26 = v62;
    v30 = v54;
    LODWORD(v28) = *(_DWORD *)(v62 + 32);
    goto LABEL_13;
  }
LABEL_14:
  v42 = *v19;
  *v19 = v26;
  if ( v42 )
  {
    sub_A17130(v42 + 136);
    sub_A17130(v42 + 104);
    v43 = *(_QWORD *)(v42 + 24);
    if ( v43 != v42 + 40 )
      _libc_free(v43);
    j_j___libc_free_0(v42);
  }
  return 1;
}
