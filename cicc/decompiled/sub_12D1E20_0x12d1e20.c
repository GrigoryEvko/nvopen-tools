// Function: sub_12D1E20
// Address: 0x12d1e20
//
__int64 __fastcall sub_12D1E20(
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
  int *v15; // r14
  unsigned int v17; // r15d
  int v19; // r12d
  _BYTE *v20; // rdi
  __int64 v21; // rdx
  size_t v22; // rcx
  __int64 v23; // rsi
  _BYTE *v24; // rdi
  __int64 v25; // rdx
  size_t v26; // rcx
  __int64 v27; // rsi
  _BYTE *v28; // rdi
  __int64 v29; // rdx
  size_t v30; // rcx
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
  void **v45; // r12
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  size_t v50; // rax
  __int64 v51; // rax
  size_t v52; // r14
  unsigned __int64 v53; // rax
  void *v54; // r13
  size_t v55; // rcx
  char *v56; // rax
  __int64 v57; // rax
  __int64 v58; // r8
  unsigned int v59; // r9d
  __int64 *v60; // r10
  __int64 v61; // rcx
  void *v62; // rdi
  unsigned int v63; // eax
  __int64 *v64; // rax
  __int64 *v65; // rax
  __int64 v66; // rax
  void *v67; // rax
  __int64 v68; // rax
  int v69; // edx
  int v70; // ecx
  __int64 v71; // rsi
  __int64 v72; // r8
  __int64 v73; // rax
  unsigned int v74; // edx
  int v75; // ecx
  __int64 v76; // r13
  __int64 v77; // r12
  __int64 v78; // rdi
  char *v79; // rsi
  int v80; // r12d
  void **v81; // r13
  void **v82; // r12
  size_t v83; // rdx
  size_t v84; // rdx
  size_t v85; // rdx
  size_t v86; // rdx
  size_t v87; // rdx
  size_t v88; // rdx
  int *v89; // [rsp+0h] [rbp-130h]
  void **v90; // [rsp+18h] [rbp-118h]
  __int64 *v91; // [rsp+20h] [rbp-110h]
  __int64 *v92; // [rsp+20h] [rbp-110h]
  unsigned int v93; // [rsp+28h] [rbp-108h]
  __int64 *v94; // [rsp+28h] [rbp-108h]
  unsigned int v95; // [rsp+28h] [rbp-108h]
  unsigned int v96; // [rsp+30h] [rbp-100h]
  __int64 v97; // [rsp+30h] [rbp-100h]
  __int64 v98; // [rsp+38h] [rbp-F8h]
  int v100; // [rsp+4Ch] [rbp-E4h]
  __int64 v101; // [rsp+58h] [rbp-D8h] BYREF
  void *src; // [rsp+60h] [rbp-D0h] BYREF
  size_t v103; // [rsp+68h] [rbp-C8h]
  __int64 v104; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v105; // [rsp+78h] [rbp-B8h]
  __int64 v106; // [rsp+80h] [rbp-B0h]
  _QWORD v107[2]; // [rsp+90h] [rbp-A0h] BYREF
  char *v108; // [rsp+A0h] [rbp-90h] BYREF
  size_t v109; // [rsp+A8h] [rbp-88h]
  void **v110; // [rsp+B0h] [rbp-80h] BYREF
  size_t n; // [rsp+B8h] [rbp-78h]
  _QWORD v112[14]; // [rsp+C0h] [rbp-70h] BYREF

  v15 = a5;
  v17 = sub_12D1DA0(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
  if ( v17 || !a15 || *(_DWORD *)(a1 + 248) == 4 )
    return v17;
  if ( !*v15 )
    goto LABEL_5;
  sub_12C7BD0((__int64)&v110, "-R", 2u, v15, (char *)*a6, 1, 0);
  v45 = v110;
  v106 = 0x1000000000LL;
  v104 = 0;
  v105 = 0;
  v90 = &v110[4 * (unsigned int)n];
  if ( v110 == v90 )
  {
    v71 = 0;
    v70 = 0;
    v69 = 0;
    v68 = 0;
    goto LABEL_82;
  }
  v89 = v15;
  do
  {
    src = *v45;
    v50 = (size_t)v45[1];
    LOBYTE(v101) = 61;
    v103 = v50;
    v51 = sub_16D20C0(&src, &v101, 1, 0);
    v52 = v51;
    if ( v51 == -1 )
    {
      v54 = src;
      v52 = v103;
      v108 = 0;
      v109 = 0;
    }
    else
    {
      v53 = v51 + 1;
      v54 = src;
      if ( v53 > v103 )
        v53 = v103;
      v55 = v103 - v53;
      v56 = (char *)src + v53;
      if ( v52 && v52 > v103 )
        v52 = v103;
      v108 = v56;
      v109 = v55;
    }
    if ( (unsigned __int8)sub_16D2BB0(v108, v109, 10, &v101) )
    {
      v101 = 0;
      v100 = 0;
    }
    else
    {
      v100 = v101;
    }
    v48 = (unsigned int)sub_16D19C0(&v104, v54, v52);
    v49 = *(_QWORD *)(v104 + 8 * v48);
    if ( v49 )
    {
      if ( v49 != -8 )
        goto LABEL_58;
      LODWORD(v106) = v106 - 1;
    }
    v91 = (__int64 *)(v104 + 8 * v48);
    v93 = v48;
    v57 = malloc(v52 + 17, v54, v52 + 17, v46, v47, v48);
    v59 = v93;
    v60 = v91;
    v61 = v57;
    if ( !v57 )
    {
      if ( v52 == -17 )
      {
        v66 = malloc(1, v54, 0, 0, v58, v93);
        v61 = 0;
        v59 = v93;
        v60 = v91;
        if ( v66 )
        {
          v62 = (void *)(v66 + 16);
          v61 = v66;
LABEL_80:
          v94 = v60;
          v96 = v59;
          v98 = v61;
          v67 = memcpy(v62, v54, v52);
          v60 = v94;
          v59 = v96;
          v61 = v98;
          v62 = v67;
          goto LABEL_71;
        }
      }
      v92 = v60;
      v95 = v59;
      v97 = v61;
      sub_16BD1C0("Allocation failed");
      v61 = v97;
      v59 = v95;
      v60 = v92;
    }
    v62 = (void *)(v61 + 16);
    if ( v52 + 1 > 1 )
      goto LABEL_80;
LABEL_71:
    *((_BYTE *)v62 + v52) = 0;
    *(_QWORD *)v61 = v52;
    *(_DWORD *)(v61 + 8) = 0;
    *v60 = v61;
    ++HIDWORD(v105);
    v63 = sub_16D1CD0(&v104, v59);
    v64 = (__int64 *)(v104 + 8LL * v63);
    v49 = *v64;
    if ( *v64 == -8 || !v49 )
    {
      v65 = v64 + 1;
      do
      {
        do
          v49 = *v65++;
        while ( !v49 );
      }
      while ( v49 == -8 );
    }
LABEL_58:
    v45 += 4;
    *(_DWORD *)(v49 + 8) = v100;
  }
  while ( v90 != v45 );
  v17 = 0;
  v15 = v89;
  v68 = v104;
  v69 = v105;
  v70 = HIDWORD(v105);
  v71 = (unsigned int)v106;
LABEL_82:
  v72 = *(_QWORD *)(a1 + 208);
  *(_QWORD *)(a1 + 208) = v68;
  v73 = *(unsigned int *)(a1 + 216);
  *(_DWORD *)(a1 + 216) = v69;
  v74 = *(_DWORD *)(a1 + 220);
  *(_DWORD *)(a1 + 220) = v70;
  v75 = *(_DWORD *)(a1 + 224);
  v104 = v72;
  v105 = __PAIR64__(v74, v73);
  *(_DWORD *)(a1 + 224) = v71;
  LODWORD(v106) = v75;
  if ( v74 && (_DWORD)v73 )
  {
    v76 = 8 * v73;
    v77 = 0;
    do
    {
      v78 = *(_QWORD *)(v72 + v77);
      if ( v78 != -8 && v78 )
      {
        _libc_free(v78, v71);
        v72 = v104;
      }
      v77 += 8;
    }
    while ( v76 != v77 );
  }
  _libc_free(v72, v71);
  sub_12C8460((__int64)v107, "-lnk-discard-value-names", 0x18u, v15, (char *)*a6);
  v79 = "1";
  v80 = sub_2241AC0(v107, "1");
  if ( (char **)v107[0] != &v108 )
  {
    v79 = v108 + 1;
    j_j___libc_free_0(v107[0], v108 + 1);
  }
  if ( !v80 )
    *(_BYTE *)(a1 + 240) = 1;
  v81 = v110;
  v82 = &v110[4 * (unsigned int)n];
  if ( v110 != v82 )
  {
    do
    {
      v82 -= 4;
      if ( *v82 != v82 + 2 )
      {
        v79 = (char *)v82[2] + 1;
        j_j___libc_free_0(*v82, v79);
      }
    }
    while ( v81 != v82 );
    v82 = v110;
  }
  if ( v82 != v112 )
    _libc_free(v82, v79);
LABEL_5:
  if ( *a7 )
  {
    sub_12C8460((__int64)&v110, "-opt-arch", 9u, a7, (char *)*a8);
    v40 = *(_BYTE **)(a1 + 16);
    if ( v110 == v112 )
    {
      v85 = n;
      if ( n )
      {
        if ( n == 1 )
          *v40 = v112[0];
        else
          memcpy(v40, v112, n);
        v85 = n;
        v40 = *(_BYTE **)(a1 + 16);
      }
      *(_QWORD *)(a1 + 24) = v85;
      v40[v85] = 0;
      v40 = v110;
      goto LABEL_47;
    }
    v41 = v112[0];
    v42 = n;
    if ( v40 == (_BYTE *)(a1 + 32) )
    {
      *(_QWORD *)(a1 + 16) = v110;
      *(_QWORD *)(a1 + 24) = v42;
      *(_QWORD *)(a1 + 32) = v41;
    }
    else
    {
      v43 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 16) = v110;
      *(_QWORD *)(a1 + 24) = v42;
      *(_QWORD *)(a1 + 32) = v41;
      if ( v40 )
      {
        v110 = (void **)v40;
        v112[0] = v43;
LABEL_47:
        n = 0;
        *v40 = 0;
        if ( v110 != v112 )
          j_j___libc_free_0(v110, v112[0] + 1LL);
        sub_12C8460((__int64)&v110, "-opt-discard-value-names", 0x18u, a7, (char *)*a8);
        v44 = sub_2241AC0(&v110, "1");
        if ( v110 != v112 )
          j_j___libc_free_0(v110, v112[0] + 1LL);
        if ( !v44 )
          *(_BYTE *)(a1 + 240) = 1;
        goto LABEL_6;
      }
    }
    v110 = (void **)v112;
    v40 = v112;
    goto LABEL_47;
  }
LABEL_6:
  if ( !*a11 )
    goto LABEL_7;
  sub_12C8460((__int64)&v110, "-mcpu", 5u, a11, (char *)*a12);
  v20 = *(_BYTE **)(a1 + 80);
  if ( v110 == v112 )
  {
    v83 = n;
    if ( n )
    {
      if ( n == 1 )
        *v20 = v112[0];
      else
        memcpy(v20, v112, n);
      v83 = n;
      v20 = *(_BYTE **)(a1 + 80);
    }
    *(_QWORD *)(a1 + 88) = v83;
    v20[v83] = 0;
    v20 = v110;
  }
  else
  {
    v21 = v112[0];
    v22 = n;
    if ( v20 == (_BYTE *)(a1 + 96) )
    {
      *(_QWORD *)(a1 + 80) = v110;
      *(_QWORD *)(a1 + 88) = v22;
      *(_QWORD *)(a1 + 96) = v21;
    }
    else
    {
      v23 = *(_QWORD *)(a1 + 96);
      *(_QWORD *)(a1 + 80) = v110;
      *(_QWORD *)(a1 + 88) = v22;
      *(_QWORD *)(a1 + 96) = v21;
      if ( v20 )
      {
        v110 = (void **)v20;
        v112[0] = v23;
        goto LABEL_17;
      }
    }
    v110 = (void **)v112;
    v20 = v112;
  }
LABEL_17:
  n = 0;
  *v20 = 0;
  if ( v110 != v112 )
    j_j___libc_free_0(v110, v112[0] + 1LL);
  sub_12C8460((__int64)&v110, "-march", 6u, a11, (char *)*a12);
  v24 = *(_BYTE **)(a1 + 48);
  if ( v110 == v112 )
  {
    v88 = n;
    if ( n )
    {
      if ( n == 1 )
        *v24 = v112[0];
      else
        memcpy(v24, v112, n);
      v88 = n;
      v24 = *(_BYTE **)(a1 + 48);
    }
    *(_QWORD *)(a1 + 56) = v88;
    v24[v88] = 0;
    v24 = v110;
  }
  else
  {
    v25 = v112[0];
    v26 = n;
    if ( v24 == (_BYTE *)(a1 + 64) )
    {
      *(_QWORD *)(a1 + 48) = v110;
      *(_QWORD *)(a1 + 56) = v26;
      *(_QWORD *)(a1 + 64) = v25;
    }
    else
    {
      v27 = *(_QWORD *)(a1 + 64);
      *(_QWORD *)(a1 + 48) = v110;
      *(_QWORD *)(a1 + 56) = v26;
      *(_QWORD *)(a1 + 64) = v25;
      if ( v24 )
      {
        v110 = (void **)v24;
        v112[0] = v27;
        goto LABEL_23;
      }
    }
    v110 = (void **)v112;
    v24 = v112;
  }
LABEL_23:
  n = 0;
  *v24 = 0;
  if ( v110 != v112 )
    j_j___libc_free_0(v110, v112[0] + 1LL);
  sub_12C8460((__int64)&v110, "-nvptx-fma-level", 0x10u, a11, (char *)*a12);
  v28 = *(_BYTE **)(a1 + 112);
  if ( v110 == v112 )
  {
    v86 = n;
    if ( n )
    {
      if ( n == 1 )
        *v28 = v112[0];
      else
        memcpy(v28, v112, n);
      v86 = n;
      v28 = *(_BYTE **)(a1 + 112);
    }
    *(_QWORD *)(a1 + 120) = v86;
    v28[v86] = 0;
    v28 = v110;
  }
  else
  {
    v29 = v112[0];
    v30 = n;
    if ( v28 == (_BYTE *)(a1 + 128) )
    {
      *(_QWORD *)(a1 + 112) = v110;
      *(_QWORD *)(a1 + 120) = v30;
      *(_QWORD *)(a1 + 128) = v29;
    }
    else
    {
      v31 = *(_QWORD *)(a1 + 128);
      *(_QWORD *)(a1 + 112) = v110;
      *(_QWORD *)(a1 + 120) = v30;
      *(_QWORD *)(a1 + 128) = v29;
      if ( v28 )
      {
        v110 = (void **)v28;
        v112[0] = v31;
        goto LABEL_29;
      }
    }
    v110 = (void **)v112;
    v28 = v112;
  }
LABEL_29:
  n = 0;
  *v28 = 0;
  if ( v110 != v112 )
    j_j___libc_free_0(v110, v112[0] + 1LL);
  sub_12C8460((__int64)&v110, "-nvptx-prec-divf32", 0x12u, a11, (char *)*a12);
  v32 = *(_BYTE **)(a1 + 144);
  if ( v110 == v112 )
  {
    v84 = n;
    if ( n )
    {
      if ( n == 1 )
        *v32 = v112[0];
      else
        memcpy(v32, v112, n);
      v84 = n;
      v32 = *(_BYTE **)(a1 + 144);
    }
    *(_QWORD *)(a1 + 152) = v84;
    v32[v84] = 0;
    v32 = v110;
  }
  else
  {
    v33 = v112[0];
    v34 = n;
    if ( v32 == (_BYTE *)(a1 + 160) )
    {
      *(_QWORD *)(a1 + 144) = v110;
      *(_QWORD *)(a1 + 152) = v34;
      *(_QWORD *)(a1 + 160) = v33;
    }
    else
    {
      v35 = *(_QWORD *)(a1 + 160);
      *(_QWORD *)(a1 + 144) = v110;
      *(_QWORD *)(a1 + 152) = v34;
      *(_QWORD *)(a1 + 160) = v33;
      if ( v32 )
      {
        v110 = (void **)v32;
        v112[0] = v35;
        goto LABEL_35;
      }
    }
    v110 = (void **)v112;
    v32 = v112;
  }
LABEL_35:
  n = 0;
  *v32 = 0;
  if ( v110 != v112 )
    j_j___libc_free_0(v110, v112[0] + 1LL);
  sub_12C8460((__int64)&v110, "-nvptx-prec-sqrtf32", 0x13u, a11, (char *)*a12);
  v36 = *(_BYTE **)(a1 + 176);
  if ( v110 == v112 )
  {
    v87 = n;
    if ( n )
    {
      if ( n == 1 )
        *v36 = v112[0];
      else
        memcpy(v36, v112, n);
      v87 = n;
      v36 = *(_BYTE **)(a1 + 176);
    }
    *(_QWORD *)(a1 + 184) = v87;
    v36[v87] = 0;
    v36 = v110;
    goto LABEL_41;
  }
  v37 = v112[0];
  v38 = n;
  if ( v36 == (_BYTE *)(a1 + 192) )
  {
    *(_QWORD *)(a1 + 176) = v110;
    *(_QWORD *)(a1 + 184) = v38;
    *(_QWORD *)(a1 + 192) = v37;
  }
  else
  {
    v39 = *(_QWORD *)(a1 + 192);
    *(_QWORD *)(a1 + 176) = v110;
    *(_QWORD *)(a1 + 184) = v38;
    *(_QWORD *)(a1 + 192) = v37;
    if ( v36 )
    {
      v110 = (void **)v36;
      v112[0] = v39;
      goto LABEL_41;
    }
  }
  v110 = (void **)v112;
  v36 = v112;
LABEL_41:
  n = 0;
  *v36 = 0;
  if ( v110 != v112 )
    j_j___libc_free_0(v110, v112[0] + 1LL);
LABEL_7:
  if ( *a9 )
  {
    sub_12C8460((__int64)&v110, "-lto-discard-value-names", 0x18u, a9, (char *)*a10);
    v19 = sub_2241AC0(&v110, "1");
    if ( v110 != v112 )
      j_j___libc_free_0(v110, v112[0] + 1LL);
    if ( !v19 )
    {
      *(_BYTE *)(a1 + 240) = 1;
      return 0;
    }
  }
  return v17;
}
