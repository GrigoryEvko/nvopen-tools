// Function: sub_3972F10
// Address: 0x3972f10
//
__int64 __fastcall sub_3972F10(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *i; // r14
  _QWORD *v5; // rsi
  _QWORD *v6; // rbx
  unsigned __int8 v7; // dl
  char v8; // dl
  int v9; // r15d
  __int64 v10; // rsi
  _QWORD *v11; // rbx
  void (*v12)(); // rax
  _QWORD *v13; // rax
  _QWORD *v14; // r13
  const char *v15; // rdi
  unsigned int v16; // ebx
  char *v17; // r14
  size_t v18; // rax
  char *v19; // rdi
  double *v20; // r15
  size_t v21; // rax
  const char *v22; // r12
  size_t v23; // r8
  unsigned __int8 *v24; // rsi
  size_t v25; // rdx
  size_t v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  const char *v29; // rdx
  __m128i **v30; // r14
  __m128i **v31; // r12
  const char **v32; // rdi
  __int64 (__fastcall *v33)(__int64); // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __m128i *v36; // rdi
  void (__fastcall *v37)(void *, __int64, __int64); // r14
  __m128i **v38; // r14
  __m128i **v39; // r12
  __int64 *v40; // rdi
  __int64 (__fastcall *v41)(__int64); // rax
  __m128i *v42; // rdi
  void (*v43)(void); // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rbx
  __int64 v47; // r14
  __int64 v48; // r13
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdi
  void (*v52)(); // rax
  __int64 (__fastcall *v53)(__int64, __int64); // rax
  __int64 v54; // rax
  _QWORD *v55; // rax
  void (*v56)(void); // rax
  const char *v57; // rdx
  _QWORD *v58; // rax
  const char *v59; // rsi
  __m128i **v60; // r12
  __m128i **v61; // r14
  const char **v62; // rdi
  __int64 (__fastcall *v63)(__int64); // rax
  __int64 v64; // rax
  __int64 v65; // r15
  __m128i *v66; // rdi
  __m128i **v67; // r15
  __m128i **v68; // r12
  const char **v69; // rdi
  __int64 (__fastcall *v70)(__int64); // rax
  __m128i *v71; // rdi
  void (*v72)(); // rax
  __int64 v73; // rdi
  __int64 v74; // rdi
  __int64 v76; // rdi
  __int64 (*v77)(); // rax
  __int64 v78; // rsi
  const char *v79; // rax
  unsigned __int64 v80; // rdx
  void (*v81)(); // r12
  __int64 v82; // rax
  __int64 v83; // r13
  __int64 v84; // rdi
  void (__fastcall *v85)(__int64, __int64, _QWORD); // rbx
  __int64 v86; // rax
  __int64 v87; // r13
  __int64 v88; // rdi
  void (__fastcall *v89)(__int64, __int64, _QWORD); // rbx
  __int64 v90; // rax
  __int64 v91; // r13
  __int64 (__fastcall *v92)(__int64, __int64, __int64, _QWORD, _QWORD **); // rbx
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rdi
  __int64 v96; // rax
  __int64 *v97; // r14
  __int64 v98; // rax
  __int64 v99; // rax
  _QWORD *v100; // r13
  const char *v101; // rdx
  const char *v102; // rsi
  _QWORD *v103; // rax
  __m128i **v104; // r15
  __m128i **v105; // r12
  const char **v106; // rdi
  __int64 (__fastcall *v107)(__int64); // rax
  __int64 v108; // rax
  __m128i *v109; // rdi
  __int64 v110; // rdx
  void (*v111)(); // rax
  __m128i **v112; // r15
  __m128i **v113; // r12
  const char **v114; // rdi
  __int64 (__fastcall *v115)(__int64); // rax
  __m128i *v116; // rdi
  __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rax
  __int64 *v120; // rbx
  __int64 *v121; // r14
  __int64 v122; // rdx
  void (*v123)(); // rax
  __int64 v124; // r13
  __int64 v125; // rax
  __m128i *v126; // rdi
  __int64 v127; // rbx
  unsigned int v128; // eax
  unsigned int v129; // esi
  __m128i *v130; // r15
  __m128i *v131; // r13
  __int64 v132; // rsi
  __int64 *v133; // r14
  unsigned int v134; // eax
  __int64 v135; // [rsp+0h] [rbp-120h]
  const char *v136; // [rsp+8h] [rbp-118h]
  _QWORD *v138; // [rsp+18h] [rbp-108h]
  _QWORD *v139; // [rsp+18h] [rbp-108h]
  int v140[2]; // [rsp+20h] [rbp-100h]
  int v141[2]; // [rsp+20h] [rbp-100h]
  int v142[2]; // [rsp+28h] [rbp-F8h]
  int v143[2]; // [rsp+28h] [rbp-F8h]
  _QWORD *v144; // [rsp+28h] [rbp-F8h]
  __int64 v145; // [rsp+30h] [rbp-F0h]
  int v146[2]; // [rsp+30h] [rbp-F0h]
  _QWORD *v147; // [rsp+38h] [rbp-E8h]
  void *srcb; // [rsp+40h] [rbp-E0h]
  _QWORD *src; // [rsp+40h] [rbp-E0h]
  _QWORD **srca; // [rsp+40h] [rbp-E0h]
  size_t n; // [rsp+48h] [rbp-D8h]
  _QWORD *v152; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v153; // [rsp+58h] [rbp-C8h]
  _BYTE v154[16]; // [rsp+60h] [rbp-C0h] BYREF
  const char *v155; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v156; // [rsp+78h] [rbp-A8h]
  _QWORD *v157; // [rsp+80h] [rbp-A0h]
  _QWORD *v158; // [rsp+88h] [rbp-98h]
  const char *v159; // [rsp+90h] [rbp-90h]
  const char *v160; // [rsp+98h] [rbp-88h]
  _QWORD *v161; // [rsp+A0h] [rbp-80h]
  _QWORD *v162; // [rsp+A8h] [rbp-78h]
  __m128i *v163; // [rsp+B0h] [rbp-70h] BYREF
  __m128i *v164; // [rsp+B8h] [rbp-68h]
  void *v165; // [rsp+C0h] [rbp-60h]
  __m128i *v166; // [rsp+C8h] [rbp-58h]
  __int64 (__fastcall *v167)(__int64); // [rsp+D0h] [rbp-50h]
  _QWORD **v168; // [rsp+D8h] [rbp-48h]
  __int64 (__fastcall *v169)(__int64 *); // [rsp+E0h] [rbp-40h]
  __int64 v170; // [rsp+E8h] [rbp-38h]

  v2 = (__int64)a1;
  a1[33] = 0;
  sub_3972DD0((__int64)a1, (__int64)a2);
  v3 = (_QWORD *)a2[2];
  v136 = (const char *)(a2 + 1);
  for ( i = a2 + 1; v3 != i; v3 = (_QWORD *)v3[1] )
  {
    v5 = v3 - 7;
    if ( !v3 )
      v5 = 0;
    (*(void (__fastcall **)(_QWORD *, _QWORD *))(*a1 + 208LL))(a1, v5);
  }
  sub_396EF40((__int64)a1);
  v6 = (_QWORD *)a2[4];
  v147 = a2 + 3;
  if ( v6 != a2 + 3 )
  {
    while ( 1 )
    {
      if ( !v6 )
        BUG();
      v7 = *((_BYTE *)v6 - 24);
      if ( (v7 & 0xF) == 1 )
        goto LABEL_10;
      if ( sub_15E4F60((__int64)(v6 - 7)) )
        break;
LABEL_12:
      v6 = (_QWORD *)v6[1];
      if ( v6 == a2 + 3 )
        goto LABEL_13;
    }
    v7 = *((_BYTE *)v6 - 24);
LABEL_10:
    v8 = v7 >> 4;
    v9 = v8 & 3;
    if ( (v8 & 3) != 0 )
    {
      v10 = sub_396EAF0((__int64)a1, (__int64)(v6 - 7));
      sub_39719F0((__int64)a1, v10, v9, 0);
    }
    goto LABEL_12;
  }
LABEL_13:
  v11 = (_QWORD *)sub_396DD80((__int64)a1);
  v12 = *(void (**)())(*v11 + 32LL);
  if ( v12 != nullsub_797 )
    ((void (__fastcall *)(_QWORD *, _QWORD, _QWORD *))v12)(v11, a1[32], a2);
  if ( *(_DWORD *)(a1[29] + 524LL) == 2 )
  {
    v124 = a1[34];
    v125 = *(_QWORD *)(v124 + 1696);
    if ( !v125 )
    {
      v125 = sub_22077B0(0x28u);
      if ( v125 )
      {
        *(_QWORD *)(v125 + 8) = 0;
        *(_QWORD *)(v125 + 16) = 0;
        *(_QWORD *)(v125 + 24) = 0;
        *(_QWORD *)v125 = &unk_4A40348;
        *(_DWORD *)(v125 + 32) = 0;
      }
      *(_QWORD *)(v124 + 1696) = v125;
    }
    sub_39B9440(&v163, v125 + 8);
    v126 = v164;
    if ( v163 != v164 )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v2 + 256) + 160LL))(
        *(_QWORD *)(v2 + 256),
        v11[5],
        0);
      v127 = sub_1632FA0((__int64)a2);
      v128 = sub_15A9520(v127, 0);
      v129 = -1;
      if ( v128 )
      {
        _BitScanReverse(&v128, v128);
        v129 = 31 - (v128 ^ 0x1F);
      }
      sub_396F480(v2, v129, 0);
      v126 = v163;
      v130 = v164;
      if ( v164 != v163 )
      {
        v131 = v163;
        do
        {
          v132 = v131->m128i_i64[0];
          ++v131;
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 256) + 176LL))(
            *(_QWORD *)(v2 + 256),
            v132,
            0);
          v133 = *(__int64 **)(v2 + 256);
          v134 = sub_15A9520(v127, 0);
          sub_38DDC80(v133, v131[-1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL, v134, 0);
        }
        while ( v130 != v131 );
        v126 = v163;
      }
    }
    if ( v126 )
      j_j___libc_free_0((unsigned __int64)v126);
  }
  v13 = *(_QWORD **)(v2 + 424);
  v138 = &v13[5 * *(unsigned int *)(v2 + 432)];
  if ( v13 != v138 )
  {
    v14 = *(_QWORD **)(v2 + 424);
    v135 = v2;
    do
    {
      v15 = (const char *)v14[4];
      v16 = unk_4F9E388;
      v17 = (char *)v15;
      v18 = 0;
      if ( v15 )
        v18 = strlen(v15);
      v19 = (char *)v14[3];
      v20 = (double *)v18;
      v21 = 0;
      if ( v19 )
        v21 = strlen(v19);
      v22 = (const char *)v14[2];
      n = v21;
      v23 = 0;
      if ( v22 )
        v23 = strlen(v22);
      v24 = (unsigned __int8 *)v14[1];
      v25 = 0;
      if ( v24 )
      {
        *(_QWORD *)v142 = v23;
        v26 = strlen((const char *)v14[1]);
        v23 = *(_QWORD *)v142;
        v25 = v26;
      }
      sub_16D8B50(&v163, v24, v25, (__int64)v22, v23, v16, (unsigned __int8 *)v19, n, v17, v20);
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v14 + 24LL))(*v14);
      if ( *v14 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v14 + 8LL))(*v14);
      if ( v163 )
        sub_16D7950((__int64)v163);
      v14 += 5;
    }
    while ( v138 != v14 );
    v2 = v135;
  }
  v27 = *(_QWORD *)(v2 + 240);
  *(_DWORD *)(v2 + 432) = 0;
  *(_QWORD *)(v2 + 504) = 0;
  if ( *(_QWORD *)(v27 + 320) )
  {
    v145 = v2;
    v28 = (_QWORD *)a2[4];
    v29 = (const char *)a2[2];
    v158 = a2 + 3;
    v155 = v29;
    v156 = (__int64)v136;
    v157 = v28;
    if ( v147 == v28 )
      goto LABEL_47;
    while ( 1 )
    {
      v30 = &v163;
      v166 = 0;
      v31 = &v163;
      v32 = &v155;
      v165 = sub_18564C0;
      v33 = sub_18564A0;
      if ( ((unsigned __int8)sub_18564A0 & 1) == 0 )
        goto LABEL_36;
      while ( 1 )
      {
        v33 = *(__int64 (__fastcall **)(__int64))((char *)v33 + (_QWORD)*v32 - 1);
LABEL_36:
        v34 = v33((__int64)v32);
        v35 = v34;
        if ( v34 )
          break;
        while ( 1 )
        {
          v36 = v31[3];
          v33 = (__int64 (__fastcall *)(__int64))v31[2];
          v30 += 2;
          v31 = v30;
          v32 = (const char **)((char *)&v155 + (_QWORD)v36);
          if ( ((unsigned __int8)v33 & 1) != 0 )
            break;
          v34 = ((__int64 (__fastcall *)(const char **, __int64))v33)(v32, v35);
          v35 = v34;
          if ( v34 )
            goto LABEL_39;
        }
      }
LABEL_39:
      if ( (*(_BYTE *)(v34 + 32) & 0xF) == 9 )
      {
        srcb = *(void **)(v145 + 256);
        v37 = *(void (__fastcall **)(void *, __int64, __int64))(*(_QWORD *)srcb + 256LL);
        v35 = sub_396EAF0(v145, v35);
        v37(srcb, v35, 22);
      }
      v38 = &v163;
      v166 = 0;
      v39 = &v163;
      v40 = (__int64 *)&v155;
      v165 = sub_1856470;
      v41 = sub_1856440;
      if ( ((unsigned __int8)sub_1856440 & 1) != 0 )
        goto LABEL_42;
      while ( 2 )
      {
        if ( !(unsigned __int8)v41((__int64)v40) )
        {
          while ( 1 )
          {
            v42 = v39[3];
            v41 = (__int64 (__fastcall *)(__int64))v39[2];
            v38 += 2;
            v39 = v38;
            v40 = (__int64 *)((char *)&v155 + (_QWORD)v42);
            if ( ((unsigned __int8)v41 & 1) != 0 )
              break;
            if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64))v41)(v40, v35) )
              goto LABEL_46;
          }
LABEL_42:
          v35 = *v40;
          v41 = *(__int64 (__fastcall **)(__int64))((char *)v41 + *v40 - 1);
          continue;
        }
        break;
      }
LABEL_46:
      if ( v147 == v157 )
      {
LABEL_47:
        if ( v147 == v158 && v136 == v155 && v136 == (const char *)v156 )
        {
          v2 = v145;
          break;
        }
      }
    }
  }
  v43 = *(void (**)(void))(**(_QWORD **)(v2 + 256) + 144LL);
  if ( v43 != nullsub_581 )
    v43();
  v44 = sub_160F9A0(*(_QWORD *)(v2 + 8), (__int64)&unk_4FC3606, 1u);
  if ( !v44 )
    BUG();
  v45 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v44 + 104LL))(v44, &unk_4FC3606);
  v46 = *(_QWORD *)(v45 + 160);
  v47 = v45;
  v48 = v46 + 8LL * *(unsigned int *)(v45 + 168);
  while ( v48 != v46 )
  {
    while ( 1 )
    {
      v49 = *(_QWORD *)(v48 - 8);
      v48 -= 8;
      v50 = sub_3971E20(v2, v49);
      v51 = v50;
      if ( v50 )
      {
        v52 = *(void (**)())(*(_QWORD *)v50 + 24LL);
        if ( v52 != nullsub_1973 )
          break;
      }
      if ( v48 == v46 )
        goto LABEL_60;
    }
    ((void (__fastcall *)(__int64, _QWORD *, __int64, __int64))v52)(v51, a2, v47, v2);
  }
LABEL_60:
  v53 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 408LL);
  if ( v53 == sub_396BE80 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v2 + 240) + 307LL) )
      sub_396BDC0(v2, (__int64)a2);
  }
  else
  {
    v53(v2, (__int64)a2);
  }
  if ( *(_BYTE *)(*(_QWORD *)(v2 + 272) + 1746LL) )
  {
    LODWORD(v152) = 1;
    v91 = sub_396DD80(v2);
    v92 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD **))(*(_QWORD *)v91 + 40LL);
    v93 = sub_396DDB0(v2);
    v94 = v92(v91, v93, 3, 0, &v152);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 256) + 160LL))(*(_QWORD *)(v2 + 256), v94, 0);
    v163 = (__m128i *)&v155;
    v155 = "__morestack_addr";
    v95 = *(_QWORD *)(v2 + 248);
    LOWORD(v165) = 261;
    v156 = 16;
    v96 = sub_38BF510(v95, (__int64)&v163);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 256) + 176LL))(*(_QWORD *)(v2 + 256), v96, 0);
    v97 = *(__int64 **)(v2 + 256);
    LODWORD(v91) = *(_DWORD *)(*(_QWORD *)(v2 + 240) + 8LL);
    v98 = sub_3970BC0(v2, (__int64)"__morestack", 11);
    sub_38DDC80(v97, v98, v91, 0);
  }
  if ( *(_DWORD *)(*(_QWORD *)(v2 + 232) + 524LL) == 2 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v2 + 272) + 1747LL) )
    {
      v83 = *(_QWORD *)(v2 + 256);
      v84 = *(_QWORD *)(v2 + 248);
      v85 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v83 + 160LL);
      v155 = ".note.GNU-split-stack";
      LOWORD(v165) = 257;
      LOWORD(v157) = 259;
      v86 = sub_38C3B80(v84, (__int64)&v155, 1, 0, 0, (__int64)&v163, -1, 0);
      v85(v83, v86, 0);
      if ( *(_BYTE *)(*(_QWORD *)(v2 + 272) + 1748LL) )
      {
        v87 = *(_QWORD *)(v2 + 256);
        v88 = *(_QWORD *)(v2 + 248);
        v89 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v87 + 160LL);
        v155 = ".note.GNU-no-split-stack";
        LOWORD(v157) = 259;
        LOWORD(v165) = 257;
        v90 = sub_38C3B80(v88, (__int64)&v155, 1, 0, 0, (__int64)&v163, -1, 0);
        v89(v87, v90, 0);
      }
    }
  }
  v54 = sub_16321A0((__int64)a2, (__int64)"llvm.init.trampoline", 20);
  if ( !v54 || !*(_QWORD *)(v54 + 8) )
  {
    v76 = *(_QWORD *)(v2 + 240);
    v77 = *(__int64 (**)())(*(_QWORD *)v76 + 16LL);
    if ( v77 != sub_21BC3B0 )
    {
      v78 = ((__int64 (__fastcall *)(__int64, _QWORD))v77)(v76, *(_QWORD *)(v2 + 248));
      if ( v78 )
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 256) + 160LL))(
          *(_QWORD *)(v2 + 256),
          v78,
          0);
    }
  }
  v55 = *(_QWORD **)(v2 + 232);
  if ( *((_DWORD *)v55 + 131) == 1 )
  {
    v99 = sub_396DD80(v2);
    v153 = 0;
    v100 = (_QWORD *)v99;
    v101 = (const char *)a2[2];
    v102 = (const char *)a2[8];
    v152 = v154;
    v144 = (_QWORD *)v99;
    v103 = (_QWORD *)a2[4];
    v157 = (_QWORD *)a2[6];
    v154[0] = 0;
    v160 = v136;
    *(_QWORD *)v141 = a2 + 7;
    v139 = a2 + 5;
    v155 = v102;
    v156 = (__int64)(a2 + 7);
    v158 = a2 + 5;
    v159 = v101;
    v161 = v103;
    v162 = a2 + 3;
    srca = (_QWORD **)v2;
    if ( v147 == v103 )
      goto LABEL_129;
    while ( 1 )
    {
      v104 = &v163;
      v166 = 0;
      v105 = &v163;
      v106 = &v155;
      v165 = sub_12D3C60;
      v168 = 0;
      v167 = sub_12D3C80;
      v170 = 0;
      v169 = sub_12D3CA0;
      v107 = sub_12D3C40;
      if ( ((unsigned __int8)sub_12D3C40 & 1) == 0 )
        goto LABEL_117;
      while ( 1 )
      {
        v107 = *(__int64 (__fastcall **)(__int64))((char *)v107 + (_QWORD)*v106 - 1);
LABEL_117:
        v108 = v107((__int64)v106);
        if ( v108 )
          break;
        while ( 1 )
        {
          v109 = v105[3];
          v107 = (__int64 (__fastcall *)(__int64))v105[2];
          v104 += 2;
          v105 = v104;
          v106 = (const char **)((char *)&v155 + (_QWORD)v109);
          if ( ((unsigned __int8)v107 & 1) != 0 )
            break;
          v108 = v107((__int64)v106);
          if ( v108 )
            goto LABEL_120;
        }
      }
LABEL_120:
      v110 = v108;
      LODWORD(v167) = 1;
      v166 = 0;
      v165 = 0;
      v163 = (__m128i *)&unk_49EFBE0;
      v164 = 0;
      v168 = &v152;
      v111 = *(void (**)())(*v100 + 136LL);
      if ( v111 != nullsub_798 )
      {
        ((void (__fastcall *)(_QWORD *, __m128i **, __int64))v111)(v100, &v163, v110);
        if ( v166 != v164 )
          sub_16E7BA0((__int64 *)&v163);
      }
      if ( v153 )
      {
        (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*srca[32] + 160LL))(srca[32], v100[80], 0);
        (*(void (__fastcall **)(_QWORD *, _QWORD *, __int64))(*srca[32] + 400LL))(srca[32], v152, v153);
      }
      v112 = &v163;
      v153 = 0;
      v113 = &v163;
      *(_BYTE *)v152 = 0;
      sub_16E7BC0((__int64 *)&v163);
      v166 = 0;
      v168 = 0;
      v114 = &v155;
      v165 = sub_12D3BB0;
      v170 = 0;
      v167 = sub_12D3BE0;
      v169 = sub_12D3C10;
      v115 = sub_12D3B80;
      if ( ((unsigned __int8)sub_12D3B80 & 1) != 0 )
        goto LABEL_124;
      while ( 2 )
      {
        if ( !(unsigned __int8)v115((__int64)v114) )
        {
          while ( 1 )
          {
            v116 = v113[3];
            v115 = (__int64 (__fastcall *)(__int64))v113[2];
            v112 += 2;
            v113 = v112;
            v114 = (const char **)((char *)&v155 + (_QWORD)v116);
            if ( ((unsigned __int8)v115 & 1) != 0 )
              break;
            if ( (unsigned __int8)v115((__int64)v114) )
              goto LABEL_128;
          }
LABEL_124:
          v115 = *(__int64 (__fastcall **)(__int64))((char *)v115 + (_QWORD)*v114 - 1);
          continue;
        }
        break;
      }
LABEL_128:
      if ( v147 == v161 )
      {
LABEL_129:
        if ( v147 == v162
          && v136 == v159
          && v136 == v160
          && v139 == v157
          && v139 == v158
          && *(const char **)v141 == v155
          && *(_QWORD *)v141 == v156 )
        {
          v2 = (__int64)srca;
          v117 = sub_16321C0((__int64)a2, (__int64)"llvm.used", 9, 1);
          if ( v117 )
          {
            v118 = *(_QWORD *)(v117 - 24);
            if ( v118 )
            {
              v119 = 3LL * (*(_DWORD *)(v118 + 20) & 0xFFFFFFF);
              if ( (*(_BYTE *)(v118 + 23) & 0x40) != 0 )
              {
                v120 = *(__int64 **)(v118 - 8);
                v121 = &v120[v119];
              }
              else
              {
                v121 = (__int64 *)v118;
                v120 = (__int64 *)(v118 - v119 * 8);
              }
              for ( ; v121 != v120; v120 += 3 )
              {
                v122 = sub_1649F00(*v120);
                if ( (*(_BYTE *)(v122 + 32) & 0xFu) - 7 > 1 )
                {
                  LODWORD(v167) = 1;
                  v166 = 0;
                  v165 = 0;
                  v163 = (__m128i *)&unk_49EFBE0;
                  v164 = 0;
                  v168 = &v152;
                  v123 = *(void (**)())(*v144 + 144LL);
                  if ( v123 != nullsub_799 )
                  {
                    ((void (__fastcall *)(_QWORD *, __m128i **, __int64))v123)(v144, &v163, v122);
                    if ( v166 != v164 )
                      sub_16E7BA0((__int64 *)&v163);
                  }
                  if ( v153 )
                  {
                    (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*srca[32] + 160LL))(srca[32], v144[80], 0);
                    (*(void (__fastcall **)(_QWORD *, _QWORD *, __int64))(*srca[32] + 400LL))(srca[32], v152, v153);
                  }
                  v153 = 0;
                  *(_BYTE *)v152 = 0;
                  sub_16E7BC0((__int64 *)&v163);
                }
              }
            }
          }
          if ( v152 != (_QWORD *)v154 )
            j_j___libc_free_0((unsigned __int64)v152);
          v55 = srca[29];
          break;
        }
      }
    }
  }
  if ( (*((_BYTE *)v55 + 809) & 0x10) == 0 )
    goto LABEL_93;
  v56 = *(void (**)(void))(**(_QWORD **)(v2 + 256) + 1000LL);
  if ( v56 != nullsub_595 )
    v56();
  src = (_QWORD *)v2;
  v57 = (const char *)a2[2];
  v58 = (_QWORD *)a2[4];
  v59 = (const char *)a2[8];
  *(_QWORD *)v143 = a2 + 7;
  v157 = (_QWORD *)a2[6];
  *(_QWORD *)v140 = a2 + 5;
  v160 = v136;
  v155 = v59;
  v156 = (__int64)(a2 + 7);
  v158 = a2 + 5;
  v159 = v57;
  v161 = v58;
  v162 = a2 + 3;
  if ( v58 == v147 )
    goto LABEL_85;
  do
  {
    do
    {
      v60 = &v163;
      v166 = 0;
      v61 = &v163;
      v62 = &v155;
      v165 = sub_12D3C60;
      v168 = 0;
      v167 = sub_12D3C80;
      v170 = 0;
      v169 = sub_12D3CA0;
      v63 = sub_12D3C40;
      if ( ((unsigned __int8)sub_12D3C40 & 1) == 0 )
        goto LABEL_75;
      while ( 1 )
      {
        v63 = *(__int64 (__fastcall **)(__int64))((char *)v63 + (_QWORD)*v62 - 1);
LABEL_75:
        v64 = v63((__int64)v62);
        v65 = v64;
        if ( v64 )
          break;
        while ( 1 )
        {
          v66 = v61[3];
          v63 = (__int64 (__fastcall *)(__int64))v61[2];
          v60 += 2;
          v61 = v60;
          v62 = (const char **)((char *)&v155 + (_QWORD)v66);
          if ( ((unsigned __int8)v63 & 1) != 0 )
            break;
          v64 = v63((__int64)v62);
          v65 = v64;
          if ( v64 )
            goto LABEL_78;
        }
      }
LABEL_78:
      if ( (*(_BYTE *)(v64 + 33) & 0x1C) != 0 )
        goto LABEL_79;
      v79 = sub_1649960(v64);
      if ( v80 > 4 && *(_DWORD *)v79 == 1836477548 )
      {
        if ( v79[4] == 46 || *(_BYTE *)(v65 + 32) >> 6 )
          goto LABEL_79;
      }
      else if ( *(_BYTE *)(v65 + 32) >> 6 )
      {
        goto LABEL_79;
      }
      *(_QWORD *)v146 = src[32];
      v81 = *(void (**)())(**(_QWORD **)v146 + 1008LL);
      v82 = sub_396EAF0((__int64)src, v65);
      if ( v81 != nullsub_596 )
        ((void (__fastcall *)(_QWORD, __int64))v81)(*(_QWORD *)v146, v82);
LABEL_79:
      v67 = &v163;
      v166 = 0;
      v168 = 0;
      v68 = &v163;
      v69 = &v155;
      v165 = sub_12D3BB0;
      v170 = 0;
      v167 = sub_12D3BE0;
      v169 = sub_12D3C10;
      v70 = sub_12D3B80;
      if ( ((unsigned __int8)sub_12D3B80 & 1) == 0 )
        goto LABEL_81;
      while ( 1 )
      {
        v70 = *(__int64 (__fastcall **)(__int64))((char *)v70 + (_QWORD)*v69 - 1);
LABEL_81:
        if ( (unsigned __int8)v70((__int64)v69) )
          break;
        while ( 1 )
        {
          v71 = v68[3];
          v70 = (__int64 (__fastcall *)(__int64))v68[2];
          v67 += 2;
          v68 = v67;
          v69 = (const char **)((char *)&v155 + (_QWORD)v71);
          if ( ((unsigned __int8)v70 & 1) != 0 )
            break;
          if ( (unsigned __int8)v70((__int64)v69) )
            goto LABEL_84;
        }
      }
LABEL_84:
      ;
    }
    while ( v161 != v147 );
LABEL_85:
    ;
  }
  while ( v147 != v162
       || v136 != v159
       || v136 != v160
       || *(_QWORD **)v140 != v157
       || *(_QWORD **)v140 != v158
       || *(const char **)v143 != v155
       || *(_QWORD *)v143 != v156 );
  v2 = (__int64)src;
LABEL_93:
  v72 = *(void (**)())(*(_QWORD *)v2 + 232LL);
  if ( v72 != nullsub_780 )
    ((void (__fastcall *)(__int64, _QWORD *))v72)(v2, a2);
  *(_QWORD *)(v2 + 272) = 0;
  sub_38DDA30(*(_QWORD **)(v2 + 256));
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 256) + 64LL))(*(_QWORD *)(v2 + 256));
  v73 = *(_QWORD *)(v2 + 488);
  *(_QWORD *)(v2 + 488) = 0;
  if ( v73 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v73 + 8LL))(v73);
  v74 = *(_QWORD *)(v2 + 480);
  *(_QWORD *)(v2 + 480) = 0;
  if ( v74 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v74 + 8LL))(v74);
  return 0;
}
