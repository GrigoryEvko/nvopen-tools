// Function: sub_11FD800
// Address: 0x11fd800
//
void __fastcall sub_11FD800(__int64 a1, unsigned __int64 a2, __int64 a3, int a4)
{
  int v4; // r12d
  __int64 v5; // rbx
  __int64 v6; // r14
  _BYTE *v7; // rdi
  __int64 v8; // rdx
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  _QWORD *v11; // rcx
  _BYTE *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  __int64 v16; // r11
  const __m128i *v17; // r13
  __m128i *v18; // r15
  unsigned __int64 v19; // r9
  __m128i *v20; // rdx
  unsigned __int64 v21; // r8
  const __m128i *v22; // r11
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  __m128i *v25; // rdi
  __m128i v26; // xmm0
  const __m128i *v27; // rcx
  __int64 v28; // rax
  const __m128i *v29; // r14
  const __m128i *v30; // rdi
  __m128i *v31; // r15
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  const __m128i *v34; // r14
  const __m128i *v35; // rdi
  __int64 v36; // rax
  const __m128i *v37; // r14
  const __m128i *v38; // rdi
  __m128i *v39; // r13
  __m128i *v40; // rdi
  size_t v41; // rdx
  size_t v42; // rdx
  size_t v43; // rdx
  __int64 m128i_i64; // rdx
  __m128i *v45; // rbx
  unsigned __int64 v46; // r13
  __m128i *v47; // rdi
  __m128i *v48; // r15
  __int64 v49; // r11
  __int64 v50; // r12
  __m128i *v51; // rbx
  __m128i *v52; // r13
  __int64 v53; // rdx
  __m128i *v54; // rdi
  __m128i *v55; // rax
  size_t v56; // rdx
  __m128i *v57; // r8
  __int64 v58; // r15
  __m128i *v59; // r12
  __m128i *v60; // rbx
  __int64 v61; // rdx
  __m128i *v62; // rdi
  __m128i *v63; // rax
  size_t v64; // rdx
  __int64 v65; // [rsp-1E8h] [rbp-1E8h]
  __int64 v66; // [rsp-1E8h] [rbp-1E8h]
  __int64 v67; // [rsp-1E0h] [rbp-1E0h]
  int v68; // [rsp-1E0h] [rbp-1E0h]
  __int64 v69; // [rsp-1E0h] [rbp-1E0h]
  __int64 v70; // [rsp-1D8h] [rbp-1D8h]
  int v71; // [rsp-1D8h] [rbp-1D8h]
  unsigned int v72; // [rsp-1CCh] [rbp-1CCh]
  _QWORD v73[2]; // [rsp-1A8h] [rbp-1A8h] BYREF
  _QWORD *v74; // [rsp-198h] [rbp-198h]
  size_t v75; // [rsp-190h] [rbp-190h]
  _QWORD v76[2]; // [rsp-188h] [rbp-188h] BYREF
  int v77; // [rsp-178h] [rbp-178h]
  int v78; // [rsp-174h] [rbp-174h]
  int v79; // [rsp-170h] [rbp-170h]
  _QWORD *v80; // [rsp-168h] [rbp-168h]
  size_t v81; // [rsp-160h] [rbp-160h]
  _QWORD v82[2]; // [rsp-158h] [rbp-158h] BYREF
  _QWORD *v83; // [rsp-148h] [rbp-148h]
  size_t v84; // [rsp-140h] [rbp-140h]
  _QWORD v85[2]; // [rsp-138h] [rbp-138h] BYREF
  __int64 v86; // [rsp-128h] [rbp-128h]
  __int64 v87; // [rsp-120h] [rbp-120h]
  __int64 v88; // [rsp-118h] [rbp-118h]
  const __m128i *v89; // [rsp-110h] [rbp-110h] BYREF
  unsigned int v90; // [rsp-108h] [rbp-108h]
  int v91; // [rsp-104h] [rbp-104h]
  _BYTE v92[32]; // [rsp-100h] [rbp-100h] BYREF
  _BYTE v93[224]; // [rsp-E0h] [rbp-E0h] BYREF

  if ( *(_DWORD *)(a1 + 24) > a4 )
    return;
  v4 = a4;
  v5 = a1;
  sub_C917B0((__int64)v73, *(__int64 **)(a1 + 40), a2, 0, a3, *(_QWORD *)(a1 + 40), 0, 0, 0, 0);
  v6 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)v6 = v73[0];
  v7 = *(_BYTE **)(v6 + 16);
  *(_QWORD *)(v6 + 8) = v73[1];
  if ( v74 == v76 )
  {
    v43 = v75;
    if ( v75 )
    {
      if ( v75 == 1 )
        *v7 = v76[0];
      else
        memcpy(v7, v76, v75);
      v43 = v75;
      v7 = *(_BYTE **)(v6 + 16);
    }
    *(_QWORD *)(v6 + 24) = v43;
    v7[v43] = 0;
    v7 = v74;
  }
  else
  {
    if ( v7 == (_BYTE *)(v6 + 32) )
    {
      *(_QWORD *)(v6 + 16) = v74;
      *(_QWORD *)(v6 + 24) = v75;
      *(_QWORD *)(v6 + 32) = v76[0];
    }
    else
    {
      *(_QWORD *)(v6 + 16) = v74;
      v8 = *(_QWORD *)(v6 + 32);
      *(_QWORD *)(v6 + 24) = v75;
      *(_QWORD *)(v6 + 32) = v76[0];
      if ( v7 )
      {
        v74 = v7;
        v76[0] = v8;
        goto LABEL_6;
      }
    }
    v74 = v76;
    v7 = v76;
  }
LABEL_6:
  v75 = 0;
  *v7 = 0;
  v9 = *(_BYTE **)(v6 + 64);
  *(_DWORD *)(v6 + 48) = v77;
  *(_DWORD *)(v6 + 52) = v78;
  *(_DWORD *)(v6 + 56) = v79;
  if ( v80 == v82 )
  {
    v42 = v81;
    if ( v81 )
    {
      if ( v81 == 1 )
        *v9 = v82[0];
      else
        memcpy(v9, v82, v81);
      v42 = v81;
      v9 = *(_BYTE **)(v6 + 64);
    }
    *(_QWORD *)(v6 + 72) = v42;
    v9[v42] = 0;
    v9 = v80;
  }
  else
  {
    if ( v9 == (_BYTE *)(v6 + 80) )
    {
      *(_QWORD *)(v6 + 64) = v80;
      *(_QWORD *)(v6 + 72) = v81;
      *(_QWORD *)(v6 + 80) = v82[0];
    }
    else
    {
      *(_QWORD *)(v6 + 64) = v80;
      v10 = *(_QWORD *)(v6 + 80);
      *(_QWORD *)(v6 + 72) = v81;
      *(_QWORD *)(v6 + 80) = v82[0];
      if ( v9 )
      {
        v80 = v9;
        v82[0] = v10;
        goto LABEL_10;
      }
    }
    v80 = v82;
    v9 = v82;
  }
LABEL_10:
  v81 = 0;
  v11 = v85;
  *v9 = 0;
  v12 = *(_BYTE **)(v6 + 96);
  if ( v83 == v85 )
  {
    v41 = v84;
    if ( v84 )
    {
      if ( v84 == 1 )
        *v12 = v85[0];
      else
        memcpy(v12, v85, v84);
      v41 = v84;
      v12 = *(_BYTE **)(v6 + 96);
    }
    *(_QWORD *)(v6 + 104) = v41;
    v12[v41] = 0;
    v12 = v83;
    goto LABEL_14;
  }
  if ( v12 == (_BYTE *)(v6 + 112) )
  {
    *(_QWORD *)(v6 + 96) = v83;
    *(_QWORD *)(v6 + 104) = v84;
    *(_QWORD *)(v6 + 112) = v85[0];
    goto LABEL_87;
  }
  *(_QWORD *)(v6 + 96) = v83;
  v13 = *(_QWORD *)(v6 + 112);
  *(_QWORD *)(v6 + 104) = v84;
  *(_QWORD *)(v6 + 112) = v85[0];
  if ( !v12 )
  {
LABEL_87:
    v83 = v85;
    v12 = v85;
    goto LABEL_14;
  }
  v83 = v12;
  v85[0] = v13;
LABEL_14:
  v84 = 0;
  *v12 = 0;
  v14 = *(_QWORD *)(v6 + 128);
  v15 = *(_QWORD *)(v6 + 144);
  *(_QWORD *)(v6 + 128) = v86;
  v86 = 0;
  *(_QWORD *)(v6 + 136) = v87;
  v87 = 0;
  *(_QWORD *)(v6 + 144) = v88;
  v88 = 0;
  if ( v14 )
  {
    v15 -= v14;
    j_j___libc_free_0(v14, v15);
  }
  v16 = v6 + 152;
  v17 = v89;
  if ( (const __m128i **)(v6 + 152) == &v89 )
  {
    v28 = 3LL * v90;
    v29 = &v89[v28];
    if ( &v89[v28] != v89 )
    {
      do
      {
        v29 -= 3;
        v30 = (const __m128i *)v29[1].m128i_i64[0];
        if ( v30 != &v29[2] )
        {
          v15 = v29[2].m128i_i64[0] + 1;
          j_j___libc_free_0(v30, v15);
        }
      }
      while ( v29 != v17 );
      v17 = v89;
    }
    goto LABEL_34;
  }
  v18 = *(__m128i **)(v6 + 152);
  v19 = *(unsigned int *)(v6 + 160);
  v20 = v18;
  if ( v89 != (const __m128i *)v92 )
  {
    v39 = &v18[3 * v19];
    if ( v18 != v39 )
    {
      do
      {
        v39 -= 3;
        v40 = (__m128i *)v39[1].m128i_i64[0];
        if ( v40 != &v39[2] )
        {
          v15 = v39[2].m128i_i64[0] + 1;
          j_j___libc_free_0(v40, v15);
        }
      }
      while ( v18 != v39 );
      v39 = *(__m128i **)(v6 + 152);
    }
    if ( v39 != (__m128i *)(v6 + 168) )
      _libc_free(v39, v15);
    *(_QWORD *)(v6 + 152) = v89;
    *(_DWORD *)(v6 + 160) = v90;
    *(_DWORD *)(v6 + 164) = v91;
    goto LABEL_36;
  }
  v21 = v90;
  v72 = v90;
  if ( v90 > v19 )
  {
    if ( v90 > (unsigned __int64)*(unsigned int *)(v6 + 164) )
    {
      m128i_i64 = 48 * v19;
      if ( v18 != &v18[3 * v19] )
      {
        v67 = v5;
        v45 = &v18[3 * v19];
        v46 = v90;
        do
        {
          v45 -= 3;
          v47 = (__m128i *)v45[1].m128i_i64[0];
          m128i_i64 = (__int64)v45[2].m128i_i64;
          if ( v47 != &v45[2] )
            j_j___libc_free_0(v47, v45[2].m128i_i64[0] + 1);
        }
        while ( v18 != v45 );
        v16 = v6 + 152;
        v5 = v67;
        v21 = v46;
      }
      *(_DWORD *)(v6 + 160) = 0;
      sub_C8F9C0(v16, v21, m128i_i64, (__int64)v11, v21, v19);
      v17 = v89;
      v21 = v90;
      v19 = 0;
      v18 = *(__m128i **)(v6 + 152);
      v22 = v89;
LABEL_21:
      v23 = (__m128i *)((char *)v18 + v19);
      v15 = (unsigned __int64)&v17[3 * v21];
      if ( (const __m128i *)v15 != v22 )
      {
        v24 = v22 + 2;
        v15 = (0xAAAAAAAAAAAAAABLL * ((v15 - (unsigned __int64)v22 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL;
        v25 = &v23[3 * v15 + 3];
        do
        {
          if ( v23 )
          {
            v26 = _mm_loadu_si128(v24 - 2);
            v23[1].m128i_i64[0] = (__int64)v23[2].m128i_i64;
            *v23 = v26;
            v27 = (const __m128i *)v24[-1].m128i_i64[0];
            if ( v27 == v24 )
            {
              v23[2] = _mm_loadu_si128(v24);
            }
            else
            {
              v23[1].m128i_i64[0] = (__int64)v27;
              v23[2].m128i_i64[0] = v24->m128i_i64[0];
            }
            v23[1].m128i_i64[1] = v24[-1].m128i_i64[1];
            v24[-1].m128i_i64[0] = (__int64)v24;
            v24[-1].m128i_i64[1] = 0;
            v24->m128i_i8[0] = 0;
          }
          v23 += 3;
          v24 += 3;
        }
        while ( v25 != v23 );
        v17 = v89;
      }
      *(_DWORD *)(v6 + 160) = v72;
      v36 = 3LL * v90;
      v37 = &v17[v36];
      if ( &v17[v36] != v17 )
      {
        do
        {
          v37 -= 3;
          v38 = (const __m128i *)v37[1].m128i_i64[0];
          if ( v38 != &v37[2] )
          {
            v15 = v37[2].m128i_i64[0] + 1;
            j_j___libc_free_0(v38, v15);
          }
        }
        while ( v37 != v17 );
        v17 = v89;
      }
      goto LABEL_34;
    }
    v22 = (const __m128i *)v92;
    if ( !*(_DWORD *)(v6 + 160) )
      goto LABEL_21;
    v48 = v18 + 2;
    v70 = v5;
    v49 = 48 * v19;
    v68 = v4;
    v50 = 48 * v19;
    v51 = (__m128i *)&v93[48 * v19];
    v52 = (__m128i *)v93;
    while ( 1 )
    {
      v54 = (__m128i *)v48[-1].m128i_i64[0];
      v48[-2] = _mm_loadu_si128(v52 - 2);
      v55 = (__m128i *)v52[-1].m128i_i64[0];
      if ( v55 == v52 )
      {
        v56 = v52[-1].m128i_u64[1];
        if ( v56 )
        {
          if ( v56 == 1 )
          {
            v54->m128i_i8[0] = v52->m128i_i8[0];
            v56 = v52[-1].m128i_u64[1];
            v54 = (__m128i *)v48[-1].m128i_i64[0];
          }
          else
          {
            v65 = v49;
            memcpy(v54, v52, v56);
            v56 = v52[-1].m128i_u64[1];
            v54 = (__m128i *)v48[-1].m128i_i64[0];
            v49 = v65;
          }
        }
        v48[-1].m128i_i64[1] = v56;
        v54->m128i_i8[v56] = 0;
        v54 = (__m128i *)v52[-1].m128i_i64[0];
        goto LABEL_104;
      }
      if ( v48 == v54 )
        break;
      v48[-1].m128i_i64[0] = (__int64)v55;
      v53 = v48->m128i_i64[0];
      v48[-1].m128i_i64[1] = v52[-1].m128i_i64[1];
      v48->m128i_i64[0] = v52->m128i_i64[0];
      if ( !v54 )
        goto LABEL_111;
      v52[-1].m128i_i64[0] = (__int64)v54;
      v52->m128i_i64[0] = v53;
LABEL_104:
      v52[-1].m128i_i64[1] = 0;
      v52 += 3;
      v48 += 3;
      v54->m128i_i8[0] = 0;
      if ( v52 == v51 )
      {
        v17 = v89;
        v19 = v50;
        v5 = v70;
        v4 = v68;
        v21 = v90;
        v18 = *(__m128i **)(v6 + 152);
        v22 = (const __m128i *)((char *)v89 + v49);
        goto LABEL_21;
      }
    }
    v48[-1].m128i_i64[0] = (__int64)v55;
    v48[-1].m128i_i64[1] = v52[-1].m128i_i64[1];
    v48->m128i_i64[0] = v52->m128i_i64[0];
LABEL_111:
    v52[-1].m128i_i64[0] = (__int64)v52;
    v54 = v52;
    goto LABEL_104;
  }
  v15 = *(_QWORD *)(v6 + 152);
  if ( !v90 )
    goto LABEL_46;
  v71 = v4;
  v57 = v18 + 2;
  v66 = *(_QWORD *)(v6 + 152);
  v58 = v5;
  v69 = 48LL * v90;
  v59 = v57;
  v60 = (__m128i *)v93;
  do
  {
    v62 = (__m128i *)v59[-1].m128i_i64[0];
    v59[-2] = _mm_loadu_si128(v60 - 2);
    v63 = (__m128i *)v60[-1].m128i_i64[0];
    if ( v63 == v60 )
    {
      v64 = v60[-1].m128i_u64[1];
      if ( v64 )
      {
        if ( v64 == 1 )
          v62->m128i_i8[0] = v60->m128i_i8[0];
        else
          memcpy(v62, v60, v64);
        v64 = v60[-1].m128i_u64[1];
        v62 = (__m128i *)v59[-1].m128i_i64[0];
      }
      v59[-1].m128i_i64[1] = v64;
      v62->m128i_i8[v64] = 0;
      v62 = (__m128i *)v60[-1].m128i_i64[0];
    }
    else
    {
      if ( v59 == v62 )
      {
        v59[-1].m128i_i64[0] = (__int64)v63;
        v59[-1].m128i_i64[1] = v60[-1].m128i_i64[1];
        v59->m128i_i64[0] = v60->m128i_i64[0];
      }
      else
      {
        v59[-1].m128i_i64[0] = (__int64)v63;
        v61 = v59->m128i_i64[0];
        v59[-1].m128i_i64[1] = v60[-1].m128i_i64[1];
        v59->m128i_i64[0] = v60->m128i_i64[0];
        if ( v62 )
        {
          v60[-1].m128i_i64[0] = (__int64)v62;
          v60->m128i_i64[0] = v61;
          goto LABEL_117;
        }
      }
      v60[-1].m128i_i64[0] = (__int64)v60;
      v62 = v60;
    }
LABEL_117:
    v60[-1].m128i_i64[1] = 0;
    v60 += 3;
    v59 += 3;
    v62->m128i_i8[0] = 0;
  }
  while ( &v93[v69] != (_BYTE *)v60 );
  v5 = v58;
  v4 = v71;
  v15 = *(_QWORD *)(v6 + 152);
  v19 = *(unsigned int *)(v6 + 160);
  v20 = (__m128i *)(v66 + v69);
LABEL_46:
  v31 = v20;
  v32 = v15 + 48 * v19;
  if ( (__m128i *)v32 != v20 )
  {
    do
    {
      v32 -= 48LL;
      v33 = *(_QWORD *)(v32 + 16);
      if ( v33 != v32 + 32 )
      {
        v15 = *(_QWORD *)(v32 + 32) + 1LL;
        j_j___libc_free_0(v33, v15);
      }
    }
    while ( v31 != (__m128i *)v32 );
  }
  *(_DWORD *)(v6 + 160) = v72;
  v34 = v89;
  v17 = &v89[3 * v90];
  if ( v89 != v17 )
  {
    do
    {
      v17 -= 3;
      v35 = (const __m128i *)v17[1].m128i_i64[0];
      if ( v35 != &v17[2] )
      {
        v15 = v17[2].m128i_i64[0] + 1;
        j_j___libc_free_0(v35, v15);
      }
    }
    while ( v34 != v17 );
    v17 = v89;
  }
LABEL_34:
  if ( v17 != (const __m128i *)v92 )
    _libc_free(v17, v15);
LABEL_36:
  if ( v86 )
    j_j___libc_free_0(v86, v88 - v86);
  if ( v83 != v85 )
    j_j___libc_free_0(v83, v85[0] + 1LL);
  if ( v80 != v82 )
    j_j___libc_free_0(v80, v82[0] + 1LL);
  if ( v74 != v76 )
    j_j___libc_free_0(v74, v76[0] + 1LL);
  *(_DWORD *)(v5 + 24) = v4;
}
