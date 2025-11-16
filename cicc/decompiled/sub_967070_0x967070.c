// Function: sub_967070
// Address: 0x967070
//
__int64 __fastcall sub_967070(
        __int64 a1,
        int a2,
        const char **a3,
        int a4,
        int *a5,
        __int64 *a6,
        int *a7,
        __int64 *a8,
        int *a9,
        __int64 *a10,
        int *a11,
        __int64 *a12,
        int *a13,
        __int64 *a14,
        char a15)
{
  __int64 v16; // rbx
  unsigned int v17; // r15d
  int v19; // r12d
  _BYTE *v20; // rdi
  __int64 v21; // rdx
  size_t v22; // rcx
  __int64 v23; // rsi
  _BYTE *v24; // rdi
  size_t v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rsi
  _BYTE *v28; // rdi
  size_t v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rsi
  _BYTE *v32; // rdi
  __int64 v33; // rdx
  size_t v34; // rcx
  __int64 v35; // rsi
  _BYTE *v36; // rdi
  __int64 v37; // rdx
  size_t v38; // rcx
  __int64 v39; // rsi
  _BYTE *v40; // rdi
  __int64 v41; // rdx
  size_t v42; // rcx
  __int64 v43; // rsi
  int v44; // r12d
  void **v45; // r13
  unsigned int v46; // eax
  __int64 v47; // r8
  __int64 i; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rax
  size_t v51; // rbx
  unsigned __int64 v52; // rdx
  void *v53; // r12
  unsigned __int64 v54; // rcx
  __int64 v55; // rax
  unsigned int v56; // r8d
  __int64 *v57; // r9
  __int64 v58; // rcx
  unsigned int v59; // eax
  __int64 *v60; // rdx
  __int64 v61; // rax
  int v62; // edx
  int v63; // ecx
  __int64 v64; // rsi
  __int64 v65; // r8
  __int64 v66; // rax
  unsigned int v67; // edx
  int v68; // ecx
  __int64 v69; // r13
  __int64 v70; // r12
  _QWORD *v71; // rdi
  char *v72; // rsi
  int v73; // r12d
  void **v74; // r13
  void **v75; // r12
  size_t v76; // rdx
  size_t v77; // rdx
  size_t v78; // rdx
  size_t v79; // rdx
  size_t v80; // rdx
  size_t v81; // rdx
  __int64 v82; // [rsp+20h] [rbp-110h]
  void **v83; // [rsp+30h] [rbp-100h]
  __int64 *v84; // [rsp+38h] [rbp-F8h]
  unsigned int v86; // [rsp+48h] [rbp-E8h]
  int v87; // [rsp+4Ch] [rbp-E4h]
  __int64 v88; // [rsp+58h] [rbp-D8h] BYREF
  void *src; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int64 v90; // [rsp+68h] [rbp-C8h]
  __int64 v91; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v92; // [rsp+78h] [rbp-B8h]
  __int64 v93; // [rsp+80h] [rbp-B0h]
  _QWORD v94[2]; // [rsp+90h] [rbp-A0h] BYREF
  char *v95; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int64 v96; // [rsp+A8h] [rbp-88h]
  void **v97; // [rsp+B0h] [rbp-80h] BYREF
  size_t n; // [rsp+B8h] [rbp-78h]
  _QWORD v99[14]; // [rsp+C0h] [rbp-70h] BYREF

  v16 = a1;
  v17 = sub_966FF0(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
  if ( v17 || !a15 || *(_DWORD *)(a1 + 240) == 4 )
    return v17;
  if ( !*a5 )
    goto LABEL_5;
  sub_95D990(&v97, (__m128i *)"-R", 2u, a5, (__m128i *)*a6, 1, 0);
  v45 = v97;
  v93 = 0x1000000000LL;
  v91 = 0;
  v92 = 0;
  v83 = &v97[4 * (unsigned int)n];
  if ( v97 == v83 )
  {
    v64 = 0;
    v63 = 0;
    v62 = 0;
    v61 = 0;
    goto LABEL_77;
  }
  do
  {
    src = *v45;
    v49 = (unsigned __int64)v45[1];
    LOBYTE(v88) = 61;
    v90 = v49;
    v50 = sub_C931B0(&src, &v88, 1, 0);
    if ( v50 == -1 )
    {
      v53 = src;
      v51 = v90;
      v95 = 0;
      v96 = 0;
    }
    else
    {
      v51 = v90;
      v52 = v50 + 1;
      v53 = src;
      if ( v50 + 1 > v90 )
      {
        v52 = v90;
        v54 = 0;
      }
      else
      {
        v54 = v90 - v52;
      }
      v96 = v54;
      if ( v50 <= v90 )
        v51 = v50;
      v95 = (char *)src + v52;
    }
    if ( (unsigned __int8)sub_C93CC0(v95, v96, 10, &v88) )
    {
      v88 = 0;
      v87 = 0;
    }
    else
    {
      v87 = v88;
    }
    v46 = sub_C92610(v53, v51);
    v47 = (unsigned int)sub_C92740(&v91, v53, v51, v46);
    i = *(_QWORD *)(v91 + 8 * v47);
    if ( i )
    {
      if ( i != -8 )
        goto LABEL_58;
      LODWORD(v93) = v93 - 1;
    }
    v84 = (__int64 *)(v91 + 8 * v47);
    v86 = v47;
    v55 = sub_C7D670(v51 + 17, 8);
    v56 = v86;
    v57 = v84;
    v58 = v55;
    if ( v51 )
    {
      v82 = v55;
      memcpy((void *)(v55 + 16), v53, v51);
      v56 = v86;
      v57 = v84;
      v58 = v82;
    }
    *(_BYTE *)(v58 + v51 + 16) = 0;
    *(_QWORD *)v58 = v51;
    *(_DWORD *)(v58 + 8) = 0;
    *v57 = v58;
    ++HIDWORD(v92);
    v59 = sub_C929D0(&v91, v56);
    v60 = (__int64 *)(v91 + 8LL * v59);
    for ( i = *v60; i == -8; ++v60 )
LABEL_71:
      i = v60[1];
    if ( !i )
      goto LABEL_71;
LABEL_58:
    v45 += 4;
    *(_DWORD *)(i + 8) = v87;
  }
  while ( v83 != v45 );
  v17 = 0;
  v16 = a1;
  v61 = v91;
  v62 = v92;
  v63 = HIDWORD(v92);
  v64 = (unsigned int)v93;
LABEL_77:
  v65 = *(_QWORD *)(v16 + 208);
  *(_QWORD *)(v16 + 208) = v61;
  v66 = *(unsigned int *)(v16 + 216);
  *(_DWORD *)(v16 + 216) = v62;
  v67 = *(_DWORD *)(v16 + 220);
  *(_DWORD *)(v16 + 220) = v63;
  v68 = *(_DWORD *)(v16 + 224);
  v91 = v65;
  v92 = __PAIR64__(v67, v66);
  *(_DWORD *)(v16 + 224) = v64;
  LODWORD(v93) = v68;
  if ( v67 && (_DWORD)v66 )
  {
    v69 = 8 * v66;
    v70 = 0;
    do
    {
      v71 = *(_QWORD **)(v65 + v70);
      if ( v71 != (_QWORD *)-8LL && v71 )
      {
        v64 = *v71 + 17LL;
        sub_C7D6A0(v71, v64, 8);
        v65 = v91;
      }
      v70 += 8;
    }
    while ( v69 != v70 );
  }
  _libc_free(v65, v64);
  sub_95E3E0((__int64)v94, (__m128i *)"-lnk-discard-value-names", 0x18u, a5, (__m128i *)*a6);
  v72 = "1";
  v73 = sub_2241AC0(v94, "1");
  if ( (char **)v94[0] != &v95 )
  {
    v72 = v95 + 1;
    j_j___libc_free_0(v94[0], v95 + 1);
  }
  if ( !v73 )
    *(_BYTE *)(v16 + 232) = 1;
  v74 = v97;
  v75 = &v97[4 * (unsigned int)n];
  if ( v97 != v75 )
  {
    do
    {
      v75 -= 4;
      if ( *v75 != v75 + 2 )
      {
        v72 = (char *)v75[2] + 1;
        j_j___libc_free_0(*v75, v72);
      }
    }
    while ( v74 != v75 );
    v75 = v97;
  }
  if ( v75 != v99 )
    _libc_free(v75, v72);
LABEL_5:
  if ( *a7 )
  {
    sub_95E3E0((__int64)&v97, (__m128i *)"-opt-arch", 9u, a7, (__m128i *)*a8);
    v40 = *(_BYTE **)(v16 + 16);
    if ( v97 == v99 )
    {
      v78 = n;
      if ( n )
      {
        if ( n == 1 )
          *v40 = v99[0];
        else
          memcpy(v40, v99, n);
        v78 = n;
        v40 = *(_BYTE **)(v16 + 16);
      }
      *(_QWORD *)(v16 + 24) = v78;
      v40[v78] = 0;
      v40 = v97;
      goto LABEL_47;
    }
    v41 = v99[0];
    v42 = n;
    if ( v40 == (_BYTE *)(v16 + 32) )
    {
      *(_QWORD *)(v16 + 16) = v97;
      *(_QWORD *)(v16 + 24) = v42;
      *(_QWORD *)(v16 + 32) = v41;
    }
    else
    {
      v43 = *(_QWORD *)(v16 + 32);
      *(_QWORD *)(v16 + 16) = v97;
      *(_QWORD *)(v16 + 24) = v42;
      *(_QWORD *)(v16 + 32) = v41;
      if ( v40 )
      {
        v97 = (void **)v40;
        v99[0] = v43;
LABEL_47:
        n = 0;
        *v40 = 0;
        if ( v97 != v99 )
          j_j___libc_free_0(v97, v99[0] + 1LL);
        sub_95E3E0((__int64)&v97, (__m128i *)"-opt-discard-value-names", 0x18u, a7, (__m128i *)*a8);
        v44 = sub_2241AC0(&v97, "1");
        if ( v97 != v99 )
          j_j___libc_free_0(v97, v99[0] + 1LL);
        if ( !v44 )
          *(_BYTE *)(v16 + 232) = 1;
        goto LABEL_6;
      }
    }
    v97 = (void **)v99;
    v40 = v99;
    goto LABEL_47;
  }
LABEL_6:
  if ( !*a11 )
    goto LABEL_7;
  sub_95E3E0((__int64)&v97, (__m128i *)"-mcpu", 5u, a11, (__m128i *)*a12);
  v20 = *(_BYTE **)(v16 + 80);
  if ( v97 == v99 )
  {
    v76 = n;
    if ( n )
    {
      if ( n == 1 )
        *v20 = v99[0];
      else
        memcpy(v20, v99, n);
      v76 = n;
      v20 = *(_BYTE **)(v16 + 80);
    }
    *(_QWORD *)(v16 + 88) = v76;
    v20[v76] = 0;
    v20 = v97;
  }
  else
  {
    v21 = v99[0];
    v22 = n;
    if ( v20 == (_BYTE *)(v16 + 96) )
    {
      *(_QWORD *)(v16 + 80) = v97;
      *(_QWORD *)(v16 + 88) = v22;
      *(_QWORD *)(v16 + 96) = v21;
    }
    else
    {
      v23 = *(_QWORD *)(v16 + 96);
      *(_QWORD *)(v16 + 80) = v97;
      *(_QWORD *)(v16 + 88) = v22;
      *(_QWORD *)(v16 + 96) = v21;
      if ( v20 )
      {
        v97 = (void **)v20;
        v99[0] = v23;
        goto LABEL_17;
      }
    }
    v97 = (void **)v99;
    v20 = v99;
  }
LABEL_17:
  n = 0;
  *v20 = 0;
  if ( v97 != v99 )
    j_j___libc_free_0(v97, v99[0] + 1LL);
  sub_95E3E0((__int64)&v97, (__m128i *)"-march", 6u, a11, (__m128i *)*a12);
  v24 = *(_BYTE **)(v16 + 48);
  if ( v97 == v99 )
  {
    v77 = n;
    if ( n )
    {
      if ( n == 1 )
        *v24 = v99[0];
      else
        memcpy(v24, v99, n);
      v77 = n;
      v24 = *(_BYTE **)(v16 + 48);
    }
    *(_QWORD *)(v16 + 56) = v77;
    v24[v77] = 0;
    v24 = v97;
  }
  else
  {
    v25 = n;
    v26 = v99[0];
    if ( v24 == (_BYTE *)(v16 + 64) )
    {
      *(_QWORD *)(v16 + 48) = v97;
      *(_QWORD *)(v16 + 56) = v25;
      *(_QWORD *)(v16 + 64) = v26;
    }
    else
    {
      v27 = *(_QWORD *)(v16 + 64);
      *(_QWORD *)(v16 + 48) = v97;
      *(_QWORD *)(v16 + 56) = v25;
      *(_QWORD *)(v16 + 64) = v26;
      if ( v24 )
      {
        v97 = (void **)v24;
        v99[0] = v27;
        goto LABEL_23;
      }
    }
    v97 = (void **)v99;
    v24 = v99;
  }
LABEL_23:
  n = 0;
  *v24 = 0;
  if ( v97 != v99 )
    j_j___libc_free_0(v97, v99[0] + 1LL);
  sub_95E3E0((__int64)&v97, (__m128i *)"-nvptx-fma-level", 0x10u, a11, (__m128i *)*a12);
  v28 = *(_BYTE **)(v16 + 112);
  if ( v97 == v99 )
  {
    v81 = n;
    if ( n )
    {
      if ( n == 1 )
        *v28 = v99[0];
      else
        memcpy(v28, v99, n);
      v81 = n;
      v28 = *(_BYTE **)(v16 + 112);
    }
    *(_QWORD *)(v16 + 120) = v81;
    v28[v81] = 0;
    v28 = v97;
  }
  else
  {
    v29 = n;
    v30 = v99[0];
    if ( v28 == (_BYTE *)(v16 + 128) )
    {
      *(_QWORD *)(v16 + 112) = v97;
      *(_QWORD *)(v16 + 120) = v29;
      *(_QWORD *)(v16 + 128) = v30;
    }
    else
    {
      v31 = *(_QWORD *)(v16 + 128);
      *(_QWORD *)(v16 + 112) = v97;
      *(_QWORD *)(v16 + 120) = v29;
      *(_QWORD *)(v16 + 128) = v30;
      if ( v28 )
      {
        v97 = (void **)v28;
        v99[0] = v31;
        goto LABEL_29;
      }
    }
    v97 = (void **)v99;
    v28 = v99;
  }
LABEL_29:
  n = 0;
  *v28 = 0;
  if ( v97 != v99 )
    j_j___libc_free_0(v97, v99[0] + 1LL);
  sub_95E3E0((__int64)&v97, (__m128i *)"-nvptx-prec-divf32", 0x12u, a11, (__m128i *)*a12);
  v32 = *(_BYTE **)(v16 + 144);
  if ( v97 == v99 )
  {
    v80 = n;
    if ( n )
    {
      if ( n == 1 )
        *v32 = v99[0];
      else
        memcpy(v32, v99, n);
      v80 = n;
      v32 = *(_BYTE **)(v16 + 144);
    }
    *(_QWORD *)(v16 + 152) = v80;
    v32[v80] = 0;
    v32 = v97;
  }
  else
  {
    v33 = v99[0];
    v34 = n;
    if ( v32 == (_BYTE *)(v16 + 160) )
    {
      *(_QWORD *)(v16 + 144) = v97;
      *(_QWORD *)(v16 + 152) = v34;
      *(_QWORD *)(v16 + 160) = v33;
    }
    else
    {
      v35 = *(_QWORD *)(v16 + 160);
      *(_QWORD *)(v16 + 144) = v97;
      *(_QWORD *)(v16 + 152) = v34;
      *(_QWORD *)(v16 + 160) = v33;
      if ( v32 )
      {
        v97 = (void **)v32;
        v99[0] = v35;
        goto LABEL_35;
      }
    }
    v97 = (void **)v99;
    v32 = v99;
  }
LABEL_35:
  n = 0;
  *v32 = 0;
  if ( v97 != v99 )
    j_j___libc_free_0(v97, v99[0] + 1LL);
  sub_95E3E0((__int64)&v97, (__m128i *)"-nvptx-prec-sqrtf32", 0x13u, a11, (__m128i *)*a12);
  v36 = *(_BYTE **)(v16 + 176);
  if ( v97 == v99 )
  {
    v79 = n;
    if ( n )
    {
      if ( n == 1 )
        *v36 = v99[0];
      else
        memcpy(v36, v99, n);
      v79 = n;
      v36 = *(_BYTE **)(v16 + 176);
    }
    *(_QWORD *)(v16 + 184) = v79;
    v36[v79] = 0;
    v36 = v97;
    goto LABEL_41;
  }
  v37 = v99[0];
  v38 = n;
  if ( v36 == (_BYTE *)(v16 + 192) )
  {
    *(_QWORD *)(v16 + 176) = v97;
    *(_QWORD *)(v16 + 184) = v38;
    *(_QWORD *)(v16 + 192) = v37;
  }
  else
  {
    v39 = *(_QWORD *)(v16 + 192);
    *(_QWORD *)(v16 + 176) = v97;
    *(_QWORD *)(v16 + 184) = v38;
    *(_QWORD *)(v16 + 192) = v37;
    if ( v36 )
    {
      v97 = (void **)v36;
      v99[0] = v39;
      goto LABEL_41;
    }
  }
  v97 = (void **)v99;
  v36 = v99;
LABEL_41:
  n = 0;
  *v36 = 0;
  if ( v97 != v99 )
    j_j___libc_free_0(v97, v99[0] + 1LL);
LABEL_7:
  if ( *a9 )
  {
    sub_95E3E0((__int64)&v97, (__m128i *)"-lto-discard-value-names", 0x18u, a9, (__m128i *)*a10);
    v19 = sub_2241AC0(&v97, "1");
    if ( v97 != v99 )
      j_j___libc_free_0(v97, v99[0] + 1LL);
    if ( !v19 )
    {
      *(_BYTE *)(v16 + 232) = 1;
      return 0;
    }
  }
  return v17;
}
