// Function: sub_CF15A0
// Address: 0xcf15a0
//
__int64 __fastcall sub_CF15A0(__int64 a1, __int64 **a2, __int64 a3, __m128i a4)
{
  __int64 *v4; // r13
  unsigned int v5; // r8d
  __int64 v7; // rbx
  __int64 v8; // r14
  const char *v9; // r12
  size_t v10; // rdx
  size_t v11; // r15
  int v12; // eax
  unsigned int v13; // r13d
  _QWORD *v14; // r9
  __int64 v15; // rax
  _QWORD *v16; // r9
  _QWORD *v17; // rcx
  __int64 *v18; // rbx
  __int64 *v19; // r14
  const char *v20; // r12
  size_t v21; // rdx
  size_t v22; // r15
  int v23; // eax
  unsigned int v24; // r13d
  _QWORD *v25; // r9
  __int64 v26; // rax
  _QWORD *v27; // r9
  _QWORD *v28; // rcx
  __int64 *v29; // r15
  __int64 *v30; // r12
  const char *v31; // r14
  size_t v32; // rdx
  size_t v33; // rbx
  int v34; // eax
  __int64 v35; // r8
  _QWORD *v36; // r9
  __int64 v37; // rax
  unsigned int v38; // r8d
  _QWORD *v39; // r9
  _QWORD *v40; // rcx
  __int64 **v41; // rbx
  __int64 v42; // r14
  size_t **v43; // rsi
  __int64 v44; // r9
  __int8 v45; // al
  __int64 v46; // r14
  _BYTE *v47; // rsi
  __int64 v48; // rdx
  size_t *v49; // rdi
  size_t v50; // rsi
  size_t v51; // rdx
  size_t v52; // r8
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rcx
  unsigned int v57; // r14d
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r14
  __m128i v61; // xmm0
  __m128i v62; // xmm1
  __m128i v63; // xmm2
  __int64 v64; // r8
  __int64 v65; // r12
  __int64 v66; // rbx
  _QWORD *v67; // rdi
  unsigned __int8 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r12
  __int64 v71; // rbx
  unsigned __int8 v72; // r13
  _QWORD *v73; // rdi
  size_t v74; // rdx
  __int64 v75; // r12
  __int64 v76; // r14
  __int64 **v77; // [rsp+0h] [rbp-150h]
  _QWORD *v78; // [rsp+0h] [rbp-150h]
  _QWORD *v79; // [rsp+0h] [rbp-150h]
  _QWORD *v80; // [rsp+0h] [rbp-150h]
  __int64 *src; // [rsp+8h] [rbp-148h]
  _QWORD *srca; // [rsp+8h] [rbp-148h]
  void *srcb; // [rsp+8h] [rbp-148h]
  _QWORD *v85; // [rsp+18h] [rbp-138h]
  _QWORD *v86; // [rsp+18h] [rbp-138h]
  unsigned int v87; // [rsp+18h] [rbp-138h]
  unsigned __int8 v88; // [rsp+28h] [rbp-128h]
  unsigned __int8 v89; // [rsp+28h] [rbp-128h]
  _QWORD *v90; // [rsp+38h] [rbp-118h] BYREF
  __int64 v91; // [rsp+40h] [rbp-110h] BYREF
  __int64 v92; // [rsp+48h] [rbp-108h]
  __int64 v93; // [rsp+50h] [rbp-100h]
  __m128i v94; // [rsp+60h] [rbp-F0h] BYREF
  void (__fastcall *v95)(__m128i *, __m128i *, __int64); // [rsp+70h] [rbp-E0h]
  __int64 v96; // [rsp+78h] [rbp-D8h]
  __m128i v97; // [rsp+80h] [rbp-D0h] BYREF
  void (__fastcall *v98)(__m128i *, __m128i *, __int64); // [rsp+90h] [rbp-C0h]
  __int64 v99; // [rsp+98h] [rbp-B8h]
  size_t *v100; // [rsp+A0h] [rbp-B0h] BYREF
  size_t n[2]; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 (__fastcall *v102)(size_t *, size_t *, int); // [rsp+B8h] [rbp-98h]
  _QWORD v103[18]; // [rsp+C0h] [rbp-90h] BYREF

  v4 = (__int64 *)a1;
  v5 = sub_CEEEB0(a1);
  if ( !(_BYTE)v5 )
    return v5;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = a1 + 8;
  v91 = 0;
  v93 = 0x800000000LL;
  v92 = 0;
  if ( a1 + 8 != v7 )
  {
    while ( 1 )
    {
      if ( !v7 )
LABEL_126:
        BUG();
      if ( (*(_BYTE *)(v7 - 49) & 0x10) == 0 )
        goto LABEL_5;
      v9 = sub_BD5D20(v7 - 56);
      v11 = v10;
      v12 = sub_C92610();
      v13 = sub_C92740((__int64)&v91, v9, v11, v12);
      v14 = (_QWORD *)(v91 + 8LL * v13);
      if ( *v14 )
      {
        if ( *v14 == -8 )
        {
          LODWORD(v93) = v93 - 1;
          goto LABEL_11;
        }
LABEL_5:
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          goto LABEL_14;
      }
      else
      {
LABEL_11:
        v85 = (_QWORD *)(v91 + 8LL * v13);
        v15 = sub_C7D670(v11 + 9, 8);
        v16 = v85;
        v17 = (_QWORD *)v15;
        if ( v11 )
        {
          v78 = (_QWORD *)v15;
          memcpy((void *)(v15 + 8), v9, v11);
          v16 = v85;
          v17 = v78;
        }
        *((_BYTE *)v17 + v11 + 8) = 0;
        *v17 = v11;
        *v16 = v17;
        ++HIDWORD(v92);
        sub_C929D0(&v91, v13);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
        {
LABEL_14:
          v4 = (__int64 *)a1;
          break;
        }
      }
    }
  }
  v18 = (__int64 *)v4[6];
  v19 = v4 + 5;
  if ( v4 + 5 == v18 )
    goto LABEL_27;
  src = v4;
  do
  {
    while ( 1 )
    {
      if ( !v18 )
        goto LABEL_126;
      if ( (*((_BYTE *)v18 - 41) & 0x10) != 0 )
      {
        v20 = sub_BD5D20((__int64)(v18 - 6));
        v22 = v21;
        v23 = sub_C92610();
        v24 = sub_C92740((__int64)&v91, v20, v22, v23);
        v25 = (_QWORD *)(v91 + 8LL * v24);
        if ( !*v25 )
          goto LABEL_23;
        if ( *v25 == -8 )
          break;
      }
      v18 = (__int64 *)v18[1];
      if ( v19 == v18 )
        goto LABEL_26;
    }
    LODWORD(v93) = v93 - 1;
LABEL_23:
    v86 = (_QWORD *)(v91 + 8LL * v24);
    v26 = sub_C7D670(v22 + 9, 8);
    v27 = v86;
    v28 = (_QWORD *)v26;
    if ( v22 )
    {
      v79 = (_QWORD *)v26;
      memcpy((void *)(v26 + 8), v20, v22);
      v27 = v86;
      v28 = v79;
    }
    *((_BYTE *)v28 + v22 + 8) = 0;
    *v28 = v22;
    *v27 = v28;
    ++HIDWORD(v92);
    sub_C929D0(&v91, v24);
    v18 = (__int64 *)v18[1];
  }
  while ( v19 != v18 );
LABEL_26:
  v4 = src;
LABEL_27:
  v29 = (__int64 *)v4[4];
  v30 = v4 + 3;
  if ( v29 != v4 + 3 )
  {
    while ( 1 )
    {
      if ( !v29 )
        goto LABEL_126;
      if ( (*((_BYTE *)v29 - 49) & 0x10) == 0 || sub_B2FC80((__int64)(v29 - 7)) )
        goto LABEL_29;
      v31 = sub_BD5D20((__int64)(v29 - 7));
      v33 = v32;
      v34 = sub_C92610();
      v35 = (unsigned int)sub_C92740((__int64)&v91, v31, v33, v34);
      v36 = (_QWORD *)(v91 + 8 * v35);
      if ( *v36 )
      {
        if ( *v36 == -8 )
        {
          LODWORD(v93) = v93 - 1;
          goto LABEL_36;
        }
LABEL_29:
        v29 = (__int64 *)v29[1];
        if ( v30 == v29 )
          break;
      }
      else
      {
LABEL_36:
        srca = (_QWORD *)(v91 + 8 * v35);
        v87 = v35;
        v37 = sub_C7D670(v33 + 9, 8);
        v38 = v87;
        v39 = srca;
        v40 = (_QWORD *)v37;
        if ( v33 )
        {
          v80 = (_QWORD *)v37;
          memcpy((void *)(v37 + 8), v31, v33);
          v38 = v87;
          v39 = srca;
          v40 = v80;
        }
        *((_BYTE *)v40 + v33 + 8) = 0;
        *v40 = v33;
        *v39 = v40;
        ++HIDWORD(v92);
        sub_C929D0(&v91, v38);
        v29 = (__int64 *)v29[1];
        if ( v30 == v29 )
          break;
      }
    }
  }
  v77 = &a2[a3];
  if ( v77 == a2 )
  {
LABEL_68:
    v54 = (__int64)v4;
    LOBYTE(v100) = 0;
    v94.m128i_i64[0] = (__int64)&v91;
    v61 = _mm_loadu_si128(&v94);
    v62 = _mm_loadu_si128(&v97);
    v95 = 0;
    v96 = v99;
    v63 = _mm_loadu_si128((const __m128i *)n);
    v102 = sub_CEEFE0;
    v98 = 0;
    v99 = v103[0];
    v103[0] = sub_CEEF60;
    v103[1] = 0;
    v103[2] = 0;
    v103[3] = 0x800000000LL;
    v94 = v62;
    v97 = v63;
    *(__m128i *)n = v61;
    sub_F30570(&v100, v4);
    if ( HIDWORD(v103[2]) )
    {
      v64 = v103[1];
      if ( LODWORD(v103[2]) )
      {
        v65 = 8LL * LODWORD(v103[2]);
        v66 = 0;
        do
        {
          v67 = *(_QWORD **)(v64 + v66);
          if ( v67 != (_QWORD *)-8LL && v67 )
          {
            v54 = *v67 + 9LL;
            sub_C7D6A0((__int64)v67, v54, 8);
            v64 = v103[1];
          }
          v66 += 8;
        }
        while ( v65 != v66 );
      }
    }
    else
    {
      v64 = v103[1];
    }
    _libc_free(v64, v54);
    if ( v102 )
    {
      v54 = (__int64)n;
      v102(n, n, 3);
    }
    if ( v98 )
    {
      v54 = (__int64)&v97;
      v98(&v97, &v97, 3);
    }
    if ( v95 )
    {
      v54 = (__int64)&v94;
      v95(&v94, &v94, 3);
    }
    v68 = 0;
    goto LABEL_82;
  }
  v41 = a2;
  while ( 2 )
  {
    sub_C7DA90(&v90, **v41, (*v41)[1], byte_3F871B3, 0, 0);
    v42 = *v4;
    memset(v103, 0, 0x58u);
    sub_C7EC60(&v97, v90);
    v43 = (size_t **)v42;
    sub_A011E0((__int64)&v94, v42, 0, 0, (__int64)&v100, v44, a4, (const __m128i *)v97.m128i_i64[0], v97.m128i_u64[1]);
    if ( LOBYTE(v103[10]) )
    {
      LOBYTE(v103[10]) = 0;
      if ( v103[8] )
      {
        v43 = (size_t **)&v103[6];
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v103[8])(&v103[6], &v103[6], 3);
      }
    }
    if ( LOBYTE(v103[5]) )
    {
      LOBYTE(v103[5]) = 0;
      if ( v103[3] )
      {
        v43 = (size_t **)&v103[1];
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v103[3])(&v103[1], &v103[1], 3);
      }
    }
    if ( LOBYTE(v103[0]) )
    {
      LOBYTE(v103[0]) = 0;
      if ( n[1] )
      {
        v43 = &v100;
        ((void (__fastcall *)(size_t **, size_t **, __int64))n[1])(&v100, &v100, 3);
      }
    }
    v45 = v94.m128i_i8[8];
    v46 = v94.m128i_i64[0];
    v94.m128i_i8[8] &= ~2u;
    if ( (v45 & 1) != 0 )
    {
      v94.m128i_i64[0] = 0;
      v100 = (size_t *)(v46 | 1);
      if ( (v46 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v100, (__int64)v43);
      v46 = 0;
    }
    v47 = (_BYTE *)v4[29];
    v48 = v4[30];
    v100 = &n[1];
    sub_CEF010((__int64 *)&v100, v47, (__int64)&v47[v48]);
    v103[0] = v4[33];
    v103[1] = v4[34];
    v103[2] = v4[35];
    v49 = *(size_t **)(v46 + 232);
    if ( v100 == &n[1] )
    {
      v74 = n[0];
      if ( n[0] )
      {
        if ( n[0] == 1 )
          *(_BYTE *)v49 = n[1];
        else
          memcpy(v49, &n[1], n[0]);
        v74 = n[0];
        v49 = *(size_t **)(v46 + 232);
      }
      *(_QWORD *)(v46 + 240) = v74;
      *((_BYTE *)v49 + v74) = 0;
      v49 = v100;
    }
    else
    {
      v50 = n[0];
      v51 = n[1];
      if ( v49 == (size_t *)(v46 + 248) )
      {
        *(_QWORD *)(v46 + 232) = v100;
        *(_QWORD *)(v46 + 240) = v50;
        *(_QWORD *)(v46 + 248) = v51;
      }
      else
      {
        v52 = *(_QWORD *)(v46 + 248);
        *(_QWORD *)(v46 + 232) = v100;
        *(_QWORD *)(v46 + 240) = v50;
        *(_QWORD *)(v46 + 248) = v51;
        if ( v49 )
        {
          v100 = v49;
          n[1] = v52;
          goto LABEL_52;
        }
      }
      v100 = &n[1];
      v49 = &n[1];
    }
LABEL_52:
    n[0] = 0;
    *(_BYTE *)v49 = 0;
    *(_QWORD *)(v46 + 264) = v103[0];
    *(_QWORD *)(v46 + 272) = v103[1];
    *(_QWORD *)(v46 + 280) = v103[2];
    if ( v100 != &n[1] )
      j_j___libc_free_0(v100, n[1] + 1);
    sub_BA9570(v46, (__int64)(v4 + 39));
    n[1] = 0;
    if ( (v94.m128i_i8[8] & 2) != 0 )
      goto LABEL_106;
    v53 = v94.m128i_i64[0];
    v54 = (__int64)&v97;
    v94.m128i_i64[0] = 0;
    v97.m128i_i64[0] = v53;
    v57 = sub_E4C720(v4, &v97, 3, &v100);
    if ( v97.m128i_i64[0] )
    {
      srcb = (void *)v97.m128i_i64[0];
      sub_BA9C10(v97.m128i_i64[0], (__int64)&v97, v55, v56);
      v54 = 880;
      j_j___libc_free_0(srcb, 880);
    }
    if ( n[1] )
    {
      v54 = (__int64)&v100;
      ((void (__fastcall *)(size_t **, size_t **, __int64))n[1])(&v100, &v100, 3);
    }
    if ( !(_BYTE)v57 )
    {
      if ( (unsigned __int8)sub_CEEEB0((__int64)v4) )
      {
        if ( (v94.m128i_i8[8] & 2) != 0 )
          goto LABEL_106;
        v60 = v94.m128i_i64[0];
        if ( (v94.m128i_i8[8] & 1) != 0 )
        {
          if ( v94.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v94.m128i_i64[0] + 8LL))(v94.m128i_i64[0]);
        }
        else if ( v94.m128i_i64[0] )
        {
          sub_BA9C10(v94.m128i_i64[0], v54, v58, v59);
          j_j___libc_free_0(v60, 880);
        }
        if ( v90 )
          (*(void (__fastcall **)(_QWORD *))(*v90 + 8LL))(v90);
        if ( v77 == ++v41 )
          goto LABEL_68;
        continue;
      }
      if ( (v94.m128i_i8[8] & 2) == 0 )
      {
        v76 = v94.m128i_i64[0];
        if ( (v94.m128i_i8[8] & 1) != 0 )
        {
          if ( v94.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v94.m128i_i64[0] + 8LL))(v94.m128i_i64[0]);
        }
        else if ( v94.m128i_i64[0] )
        {
          sub_BA9C10(v94.m128i_i64[0], v54, v58, v59);
          j_j___libc_free_0(v76, 880);
        }
        if ( v90 )
          (*(void (__fastcall **)(_QWORD *))(*v90 + 8LL))(v90);
        goto LABEL_68;
      }
LABEL_106:
      sub_904700(&v94);
    }
    break;
  }
  v68 = v57;
  if ( (v94.m128i_i8[8] & 2) != 0 )
    goto LABEL_106;
  v75 = v94.m128i_i64[0];
  if ( (v94.m128i_i8[8] & 1) != 0 )
  {
    if ( v94.m128i_i64[0] )
    {
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v94.m128i_i64[0] + 8LL))(
        v94.m128i_i64[0],
        v54,
        v55,
        v56,
        v57);
      v68 = v57;
    }
  }
  else if ( v94.m128i_i64[0] )
  {
    sub_BA9C10(v94.m128i_i64[0], v54, v55, v56);
    v54 = 880;
    j_j___libc_free_0(v75, 880);
    v68 = v57;
  }
  if ( v90 )
  {
    v89 = v68;
    (*(void (__fastcall **)(_QWORD *))(*v90 + 8LL))(v90);
    v68 = v89;
  }
LABEL_82:
  v69 = v91;
  if ( HIDWORD(v92) && (_DWORD)v92 )
  {
    v70 = 8LL * (unsigned int)v92;
    v71 = 0;
    v72 = v68;
    do
    {
      v73 = *(_QWORD **)(v69 + v71);
      if ( v73 && v73 != (_QWORD *)-8LL )
      {
        v54 = *v73 + 9LL;
        sub_C7D6A0((__int64)v73, v54, 8);
        v69 = v91;
      }
      v71 += 8;
    }
    while ( v71 != v70 );
    v68 = v72;
  }
  v88 = v68;
  _libc_free(v69, v54);
  return v88;
}
