// Function: sub_3098640
// Address: 0x3098640
//
__int64 __fastcall sub_3098640(_QWORD *a1, unsigned __int64 a2, unsigned __int64 *a3, __int64 *a4, __int64 *a5)
{
  _DWORD *v5; // rax
  _DWORD *v6; // rax
  __int64 *v7; // r13
  struct __jmp_buf_tag *v8; // r12
  unsigned int v9; // eax
  __int64 v10; // r12
  unsigned int v11; // r12d
  __int64 v13; // rax
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r12
  volatile signed __int32 *v16; // r12
  signed __int32 v17; // eax
  __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  __m128i *v20; // rbx
  __m128i *v21; // r12
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rsi
  __int64 v29; // rdx
  int v30; // r13d
  __int64 *v31; // rax
  __int64 v32; // rcx
  _BYTE *v33; // rdi
  size_t v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r13
  int v37; // ebx
  __int64 v38; // r10
  __int64 v39; // r15
  void *v40; // rax
  __int64 (__fastcall *v41)(__int64, unsigned __int64 *, __int64, __int64, void *, size_t, _QWORD *, __int64, __int64, __int64, _QWORD); // rax
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 *v44; // r13
  int *v45; // rax
  int v46; // eax
  __int64 v47; // rax
  __int64 *v48; // r13
  __int64 *v49; // rax
  __int64 (*v50)(); // rax
  size_t v51; // rbx
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // r13
  unsigned __int64 v54; // rdx
  __int64 v55; // rbx
  __int64 v56; // r13
  unsigned __int64 v57; // rdi
  __m128i *v58; // rax
  __int64 v59; // rdx
  __int64 *v60; // rbx
  unsigned __int64 *v61; // r12
  __m128i *v62; // rbx
  __m128i *v63; // r12
  signed __int32 v64; // eax
  _DWORD *v65; // rax
  void *v66; // rdi
  __int64 v67; // r12
  unsigned __int64 v68; // rdi
  _DWORD *v69; // rax
  void *v71; // [rsp+8h] [rbp-768h]
  size_t v72; // [rsp+10h] [rbp-760h]
  __int64 v73; // [rsp+18h] [rbp-758h]
  _BYTE *v77; // [rsp+40h] [rbp-730h]
  unsigned __int64 v79; // [rsp+58h] [rbp-718h] BYREF
  __int64 v80; // [rsp+60h] [rbp-710h]
  __int64 v81; // [rsp+68h] [rbp-708h]
  _QWORD v82[2]; // [rsp+70h] [rbp-700h] BYREF
  __m128i *v83; // [rsp+80h] [rbp-6F0h] BYREF
  __m128i *v84; // [rsp+88h] [rbp-6E8h]
  __int64 *v85; // [rsp+A0h] [rbp-6D0h] BYREF
  __int64 *v86; // [rsp+A8h] [rbp-6C8h]
  void **v87; // [rsp+C0h] [rbp-6B0h] BYREF
  unsigned __int64 v88; // [rsp+C8h] [rbp-6A8h]
  char v89[16]; // [rsp+D0h] [rbp-6A0h] BYREF
  void *dest; // [rsp+E0h] [rbp-690h]
  size_t v91; // [rsp+E8h] [rbp-688h]
  _QWORD v92[2]; // [rsp+F0h] [rbp-680h] BYREF
  _BYTE *v93[2]; // [rsp+100h] [rbp-670h] BYREF
  __int64 v94; // [rsp+110h] [rbp-660h] BYREF
  _BYTE *v95[2]; // [rsp+120h] [rbp-650h] BYREF
  __int64 v96; // [rsp+130h] [rbp-640h] BYREF
  _BYTE *v97[2]; // [rsp+140h] [rbp-630h] BYREF
  __int64 v98; // [rsp+150h] [rbp-620h] BYREF
  void *src; // [rsp+160h] [rbp-610h] BYREF
  size_t n; // [rsp+168h] [rbp-608h]
  _QWORD v101[2]; // [rsp+170h] [rbp-600h] BYREF
  __int64 v102[2]; // [rsp+180h] [rbp-5F0h] BYREF
  _BYTE v103[16]; // [rsp+190h] [rbp-5E0h] BYREF
  unsigned __int64 v104[2]; // [rsp+1A0h] [rbp-5D0h] BYREF
  _QWORD v105[2]; // [rsp+1B0h] [rbp-5C0h] BYREF
  void *v106[4]; // [rsp+1C0h] [rbp-5B0h] BYREF
  __int16 v107; // [rsp+1E0h] [rbp-590h]
  void *v108[4]; // [rsp+1F0h] [rbp-580h] BYREF
  __int16 v109; // [rsp+210h] [rbp-560h]
  _BYTE v110[4]; // [rsp+220h] [rbp-550h] BYREF
  int v111; // [rsp+224h] [rbp-54Ch]
  _QWORD v112[8]; // [rsp+260h] [rbp-510h] BYREF
  _QWORD *v113; // [rsp+2A0h] [rbp-4D0h] BYREF
  _QWORD v114[6]; // [rsp+2B0h] [rbp-4C0h] BYREF
  __int64 v115[2]; // [rsp+2E0h] [rbp-490h] BYREF
  _QWORD v116[3]; // [rsp+2F0h] [rbp-480h] BYREF
  __int64 v117; // [rsp+308h] [rbp-468h]
  __int64 v118; // [rsp+310h] [rbp-460h]
  unsigned __int64 v119[2]; // [rsp+320h] [rbp-450h] BYREF
  __int64 v120; // [rsp+330h] [rbp-440h] BYREF
  unsigned __int64 v121[2]; // [rsp+360h] [rbp-410h] BYREF
  __int64 v122; // [rsp+370h] [rbp-400h] BYREF
  _QWORD v123[18]; // [rsp+3A0h] [rbp-3D0h] BYREF
  __int64 v124; // [rsp+430h] [rbp-340h]
  unsigned int v125; // [rsp+440h] [rbp-330h]
  unsigned __int64 v126; // [rsp+450h] [rbp-320h]
  unsigned __int64 v127; // [rsp+468h] [rbp-308h]
  _BYTE *v128; // [rsp+480h] [rbp-2F0h] BYREF
  size_t v129; // [rsp+488h] [rbp-2E8h]
  __int64 v130; // [rsp+490h] [rbp-2E0h]
  _BYTE v131[264]; // [rsp+498h] [rbp-2D8h] BYREF
  _QWORD v132[5]; // [rsp+5A0h] [rbp-1D0h] BYREF
  volatile signed __int32 *v133; // [rsp+5C8h] [rbp-1A8h]
  __int64 v134; // [rsp+5D0h] [rbp-1A0h]
  _QWORD *v135; // [rsp+5D8h] [rbp-198h]
  __int64 v136; // [rsp+5E0h] [rbp-190h]
  _QWORD v137[6]; // [rsp+5E8h] [rbp-188h] BYREF
  int v138; // [rsp+618h] [rbp-158h] BYREF
  __int64 *v139; // [rsp+638h] [rbp-138h]
  __int64 v140; // [rsp+648h] [rbp-128h] BYREF
  __int64 *v141; // [rsp+658h] [rbp-118h]
  __int64 v142; // [rsp+668h] [rbp-108h] BYREF
  __int64 *v143; // [rsp+678h] [rbp-F8h]
  __int64 v144; // [rsp+688h] [rbp-E8h] BYREF
  __int64 *v145; // [rsp+698h] [rbp-D8h]
  __int64 v146; // [rsp+6A8h] [rbp-C8h] BYREF
  __int64 *v147; // [rsp+6B8h] [rbp-B8h]
  __int64 v148; // [rsp+6C8h] [rbp-A8h] BYREF
  __int64 *v149; // [rsp+6D8h] [rbp-98h]
  __int64 v150; // [rsp+6E8h] [rbp-88h] BYREF
  unsigned __int64 *v151; // [rsp+6F8h] [rbp-78h]
  unsigned __int64 *v152; // [rsp+700h] [rbp-70h]
  _QWORD *v153; // [rsp+718h] [rbp-58h]
  __int64 v154; // [rsp+720h] [rbp-50h]
  _BYTE v155[72]; // [rsp+728h] [rbp-48h] BYREF

  v77 = (_BYTE *)a1[36];
  if ( !v77 )
  {
    v67 = a1[35];
    v77 = (_BYTE *)sub_22077B0(0x6F0u);
    if ( v77 )
      sub_22623C0((__int64)v77, v67);
    v68 = a1[36];
    a1[36] = v77;
    if ( v68 )
    {
      j_j___libc_free_0(v68);
      v77 = (_BYTE *)a1[36];
    }
  }
  if ( (unsigned __int8)v77[808] + (unsigned __int8)v77[768] == 2 )
  {
    v11 = 0;
    sub_223E0D0(qword_4FD4BE0, "Error: Cannot specify multiple -llcO#\n", 38);
    return v11;
  }
  v128 = v131;
  v112[6] = &v128;
  v129 = 0;
  v112[0] = &unk_49DD288;
  v130 = 256;
  v112[1] = 2;
  memset(&v112[2], 0, 24);
  v112[5] = 0x100000000LL;
  sub_CB5980((__int64)v112, 0, 0, 0);
  if ( !v77[808] || BYTE4(qword_4F862D0[2]) )
  {
    if ( !v77[1184] )
      goto LABEL_6;
LABEL_9:
    v6 = (_DWORD *)sub_CEECD0(4, 4u);
    *v6 = 1;
    sub_C94E10((__int64)qword_4F86370, v6);
    goto LABEL_10;
  }
  v65 = (_DWORD *)sub_CEECD0(4, 4u);
  *v65 = 6;
  sub_C94E10((__int64)qword_4F862D0, v65);
  if ( v77[1184] )
    goto LABEL_9;
LABEL_6:
  if ( v77[1224] )
  {
    v69 = (_DWORD *)sub_CEECD0(4, 4u);
    *v69 = 2;
    sub_C94E10((__int64)qword_4F86370, v69);
  }
  else if ( v77[1264] )
  {
    v5 = (_DWORD *)sub_CEECD0(4, 4u);
    *v5 = 3;
    sub_C94E10((__int64)qword_4F86370, v5);
  }
LABEL_10:
  sub_B848C0(v82);
  v7 = sub_CEACC0();
  v8 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v7);
  if ( !v8 )
  {
    v66 = (void *)sub_CEECD0(200, 8u);
    memset(v66, 0, 0xC8u);
    sub_C94E10((__int64)v7, v66);
    v8 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v7);
  }
  v9 = _setjmp(v8);
  v10 = v9;
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      sub_CEAF80(a4);
LABEL_15:
      v11 = 0;
      goto LABEL_16;
    }
    goto LABEL_71;
  }
  v113 = v114;
  v22 = a2 + 312;
  sub_3098590((__int64 *)&v113, *(_BYTE **)(a2 + 232), *(_QWORD *)(a2 + 232) + *(_QWORD *)(a2 + 240));
  v23 = *(_QWORD *)(a2 + 264);
  strcpy(v89, "nvptx");
  v114[2] = v23;
  v114[3] = *(_QWORD *)(a2 + 272);
  v24 = *(_QWORD *)(a2 + 280);
  v88 = 5;
  v114[4] = v24;
  v87 = (void **)v89;
  if ( sub_AE2980(a2 + 312, 0)[1] == 64 )
    sub_2241130((unsigned __int64 *)&v87, 0, v88, "nvptx64", 7u);
  v91 = 0;
  dest = v92;
  LOBYTE(v92[0]) = 0;
  sub_F064E0((__int64)&v83, (char *)byte_3F871B3, 0, v25, v26, v27);
  v28 = 3;
  if ( sub_AE2980(v22, 3u)[1] == 32 )
  {
    v28 = (__int64)"sharedmem32bitptr";
    sub_F060D0(&v83, "sharedmem32bitptr", 17, 1);
  }
  v30 = 0;
  sub_2D91020(&v85, v28, v29);
  v31 = v85;
  v32 = 0;
  if ( v85 != v86 )
  {
    do
    {
      sub_F060D0(&v83, (_BYTE *)v31[4 * v32], v31[4 * v32 + 1], 1);
      v31 = v85;
      v32 = (unsigned int)++v30;
    }
    while ( v30 != ((char *)v86 - (char *)v85) >> 5 );
  }
  if ( a1[16] )
  {
    sub_8FD6D0((__int64)v93, "fma-level=", a1 + 15);
    sub_F060D0(&v83, v93[0], (__int64)v93[1], 1);
    if ( (__int64 *)v93[0] != &v94 )
      j_j___libc_free_0((unsigned __int64)v93[0]);
  }
  if ( a1[20] )
  {
    sub_8FD6D0((__int64)v95, "prec-divf32=", a1 + 19);
    sub_F060D0(&v83, v95[0], (__int64)v95[1], 1);
    if ( (__int64 *)v95[0] != &v96 )
      j_j___libc_free_0((unsigned __int64)v95[0]);
  }
  if ( a1[24] )
  {
    sub_8FD6D0((__int64)v97, "prec-sqrtf32=", a1 + 23);
    sub_F060D0(&v83, v97[0], (__int64)v97[1], 1);
    if ( (__int64 *)v97[0] != &v98 )
      j_j___libc_free_0((unsigned __int64)v97[0]);
  }
  sub_F05F90((__int64)&src, (__int64 *)&v83);
  v33 = dest;
  v34 = n;
  if ( src == v101 )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = v101[0];
      else
        memcpy(dest, src, n);
      v34 = n;
      v33 = dest;
    }
    v91 = v34;
    v33[v34] = 0;
    v33 = src;
    goto LABEL_87;
  }
  if ( dest == v92 )
  {
    dest = src;
    v91 = n;
    v92[0] = v101[0];
    goto LABEL_157;
  }
  v35 = v92[0];
  dest = src;
  v91 = n;
  v92[0] = v101[0];
  if ( !v33 )
  {
LABEL_157:
    v33 = v101;
    src = v101;
    goto LABEL_87;
  }
  src = v33;
  v101[0] = v35;
LABEL_87:
  n = 0;
  *v33 = 0;
  if ( src != v101 )
    j_j___libc_free_0((unsigned __int64)src);
  v102[1] = 0;
  v102[0] = (__int64)v103;
  v107 = 261;
  v106[0] = v87;
  v103[0] = 0;
  v106[1] = (void *)v88;
  sub_CC9F70((__int64)v119, v106);
  v36 = sub_C0D4F0((__int64)v119, v102);
  if ( (__int64 *)v119[0] != &v120 )
    j_j___libc_free_0(v119[0]);
  if ( !v36 )
  {
    v79 = 30;
    v104[0] = (unsigned __int64)v105;
    v58 = (__m128i *)sub_22409D0((__int64)v104, &v79, 0);
    v104[0] = (unsigned __int64)v58;
    v105[0] = v79;
    *v58 = _mm_load_si128((const __m128i *)&xmmword_4281B20);
    v59 = v104[0];
    qmemcpy(&v58[1], " nvptx target\n", 14);
    v104[1] = v79;
    *(_BYTE *)(v59 + v79) = 0;
    sub_CEB590(v104, 1, v59, (char *)0x7420787470766E20LL);
    if ( (_QWORD *)v104[0] != v105 )
      j_j___libc_free_0(v104[0]);
    if ( (_BYTE *)v102[0] != v103 )
      j_j___libc_free_0(v102[0]);
    v60 = v86;
    v61 = (unsigned __int64 *)v85;
    if ( v86 != v85 )
    {
      do
      {
        if ( (unsigned __int64 *)*v61 != v61 + 2 )
          j_j___libc_free_0(*v61);
        v61 += 4;
      }
      while ( v60 != (__int64 *)v61 );
      v61 = (unsigned __int64 *)v85;
    }
    if ( v61 )
      j_j___libc_free_0((unsigned __int64)v61);
    v62 = v84;
    v63 = v83;
    if ( v84 != v83 )
    {
      do
      {
        if ( (__m128i *)v63->m128i_i64[0] != &v63[1] )
          j_j___libc_free_0(v63->m128i_i64[0]);
        v63 += 2;
      }
      while ( v62 != v63 );
      v63 = v83;
    }
    if ( v63 )
      j_j___libc_free_0((unsigned __int64)v63);
    if ( dest != v92 )
      j_j___libc_free_0((unsigned __int64)dest);
    if ( v87 != (void **)v89 )
      j_j___libc_free_0((unsigned __int64)v87);
    if ( v113 != v114 )
      j_j___libc_free_0((unsigned __int64)v113);
    goto LABEL_15;
  }
  v132[0] = 0;
  v132[1] = 0x100000408LL;
  v132[2] = 0x6000000001LL;
  v135 = v137;
  v137[5] = 4294901760LL;
  v132[3] = 3;
  v132[4] = 0;
  v133 = 0;
  v134 = 16416;
  v136 = 0;
  LOBYTE(v137[0]) = 0;
  v137[2] = 0;
  v137[3] = 1;
  v137[4] = 1;
  sub_EA1890(&v138);
  v154 = 0;
  v153 = v155;
  v155[0] = 0;
  BYTE1(v138) = (8 * (v77[160] & 1)) | BYTE1(v138) & 0xF7;
  if ( v77[768] )
  {
    v10 = 2;
  }
  else if ( v77[808] )
  {
    v10 = 3;
  }
  v37 = sub_2D91100();
  v71 = dest;
  v38 = a1[11];
  v39 = a1[12];
  v72 = v91;
  v109 = 261;
  v73 = v38;
  v40 = *(void **)(a2 + 240);
  v108[0] = *(void **)(a2 + 232);
  v108[1] = v40;
  sub_CC9F70((__int64)v121, v108);
  v41 = *(__int64 (__fastcall **)(__int64, unsigned __int64 *, __int64, __int64, void *, size_t, _QWORD *, __int64, __int64, __int64, _QWORD))(v36 + 96);
  LODWORD(v81) = v37;
  BYTE4(v81) = 1;
  LODWORD(v80) = 1;
  BYTE4(v80) = 1;
  if ( v41 )
    v42 = v41(v36, v121, v73, v39, v71, v72, v132, v81, v80, v10, 0);
  else
    v42 = 0;
  if ( (__int64 *)v121[0] != &v122 )
    j_j___libc_free_0(v121[0]);
  v115[0] = (__int64)v116;
  sub_3098590(v115, *(_BYTE **)(a2 + 232), *(_QWORD *)(a2 + 232) + *(_QWORD *)(a2 + 240));
  v116[2] = *(_QWORD *)(a2 + 264);
  v117 = *(_QWORD *)(a2 + 272);
  v118 = *(_QWORD *)(a2 + 280);
  sub_982C80((__int64)v123, v115);
  sub_97F7E0(v123);
  v43 = sub_22077B0(0x1F0u);
  v44 = (__int64 *)v43;
  if ( v43 )
    sub_980680(v43, (__int64)v123);
  sub_B8B500((__int64)v82, v44, 0);
  sub_31C5AC0(v110, a1[11], a1[12], HIDWORD(v117) == 21);
  v45 = (int *)sub_C94E20((__int64)qword_4F86390);
  if ( v45 )
    v46 = *v45;
  else
    v46 = qword_4F86390[2];
  v111 = v46;
  v47 = sub_22077B0(0xE0u);
  v48 = (__int64 *)v47;
  if ( v47 )
    sub_31C55E0(v47, v110);
  sub_B8B500((__int64)v82, v48, 1u);
  v49 = (__int64 *)sub_31C6290();
  sub_B8B500((__int64)v82, v49, 1u);
  v50 = *(__int64 (**)())(*(_QWORD *)v42 + 136LL);
  if ( v50 != sub_226E280 )
    ((void (__fastcall *)(__int64, _QWORD *, _QWORD *, _QWORD, _QWORD, _QWORD, _QWORD))v50)(
      v42,
      v82,
      v112,
      0,
      0,
      v77[728] ^ 1u,
      0);
  sub_B823B0((__int64)v82, *a5, a5[1]);
  sub_B89FE0((__int64)v82, a2);
  sub_2241130(a3, 0, a3[1], v128, v129);
  v51 = a3[1];
  v52 = *a3;
  v53 = v51 + 1;
  if ( (unsigned __int64 *)*a3 == a3 + 2 )
    v54 = 15;
  else
    v54 = a3[2];
  if ( v53 > v54 )
  {
    sub_2240BB0(a3, v51, 0, 0, 1u);
    v52 = *a3;
  }
  *(_BYTE *)(v52 + v51) = 0;
  a3[1] = v53;
  *(_BYTE *)(*a3 + v51 + 1) = 0;
  if ( v127 )
    j_j___libc_free_0(v127);
  if ( v126 )
    j_j___libc_free_0(v126);
  v13 = v125;
  if ( v125 )
  {
    v55 = v124;
    v56 = v124 + 40LL * v125;
    do
    {
      if ( *(_DWORD *)v55 <= 0xFFFFFFFD )
      {
        v57 = *(_QWORD *)(v55 + 8);
        if ( v57 != v55 + 24 )
          j_j___libc_free_0(v57);
      }
      v55 += 40;
    }
    while ( v56 != v55 );
    v13 = v125;
  }
  sub_C7D6A0(v124, 40 * v13, 8);
  if ( (_QWORD *)v115[0] != v116 )
    j_j___libc_free_0(v115[0]);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42);
  if ( v153 != (_QWORD *)v155 )
    j_j___libc_free_0((unsigned __int64)v153);
  v14 = v152;
  v15 = v151;
  if ( v152 != v151 )
  {
    do
    {
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        j_j___libc_free_0(*v15);
      v15 += 4;
    }
    while ( v14 != v15 );
    v15 = v151;
  }
  if ( v15 )
    j_j___libc_free_0((unsigned __int64)v15);
  if ( v149 != &v150 )
    j_j___libc_free_0((unsigned __int64)v149);
  if ( v147 != &v148 )
    j_j___libc_free_0((unsigned __int64)v147);
  if ( v145 != &v146 )
    j_j___libc_free_0((unsigned __int64)v145);
  if ( v143 != &v144 )
    j_j___libc_free_0((unsigned __int64)v143);
  if ( v141 != &v142 )
    j_j___libc_free_0((unsigned __int64)v141);
  if ( v139 != &v140 )
    j_j___libc_free_0((unsigned __int64)v139);
  if ( v135 != v137 )
    j_j___libc_free_0((unsigned __int64)v135);
  v16 = v133;
  if ( v133 )
  {
    if ( &_pthread_key_create )
    {
      v17 = _InterlockedExchangeAdd(v133 + 2, 0xFFFFFFFF);
    }
    else
    {
      v17 = *((_DWORD *)v133 + 2);
      *((_DWORD *)v133 + 2) = v17 - 1;
    }
    if ( v17 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 16LL))(v16);
      if ( &_pthread_key_create )
      {
        v64 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
      }
      else
      {
        v64 = *((_DWORD *)v16 + 3);
        *((_DWORD *)v16 + 3) = v64 - 1;
      }
      if ( v64 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
    }
  }
  if ( (_BYTE *)v102[0] != v103 )
    j_j___libc_free_0(v102[0]);
  v18 = v86;
  v19 = (unsigned __int64 *)v85;
  if ( v86 != v85 )
  {
    do
    {
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        j_j___libc_free_0(*v19);
      v19 += 4;
    }
    while ( v18 != (__int64 *)v19 );
    v19 = (unsigned __int64 *)v85;
  }
  if ( v19 )
    j_j___libc_free_0((unsigned __int64)v19);
  v20 = v84;
  v21 = v83;
  if ( v84 != v83 )
  {
    do
    {
      if ( (__m128i *)v21->m128i_i64[0] != &v21[1] )
        j_j___libc_free_0(v21->m128i_i64[0]);
      v21 += 2;
    }
    while ( v20 != v21 );
    v21 = v83;
  }
  if ( v21 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( dest != v92 )
    j_j___libc_free_0((unsigned __int64)dest);
  if ( v87 != (void **)v89 )
    j_j___libc_free_0((unsigned __int64)v87);
  if ( v113 != v114 )
    j_j___libc_free_0((unsigned __int64)v113);
LABEL_71:
  v11 = 1;
  sub_CEAF80(a4);
LABEL_16:
  sub_B82680(v82);
  v112[0] = &unk_49DD388;
  sub_CB5840((__int64)v112);
  if ( v128 != v131 )
    _libc_free((unsigned __int64)v128);
  return v11;
}
