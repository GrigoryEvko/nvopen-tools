// Function: sub_281F920
// Address: 0x281f920
//
void __fastcall sub_281F920(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        const char **a8,
        unsigned __int8 a9,
        char a10,
        char a11)
{
  unsigned __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // r12
  __int64 v16; // rsi
  const char *v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  const char *v20; // r15
  unsigned int *v21; // rax
  int v22; // ecx
  unsigned int *v23; // rdx
  char v24; // al
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r13
  const char *v31; // rsi
  __int64 v32; // r14
  __int64 v33; // rdi
  __int64 v34; // r15
  _BYTE *v35; // r14
  __int64 v36; // rdx
  __int64 v37; // r13
  int v38; // edx
  __int64 v39; // r14
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rdi
  unsigned int v43; // r15d
  bool v44; // al
  __int64 v45; // r15
  unsigned int v46; // ebx
  unsigned __int8 *v47; // r14
  __int64 *v48; // r15
  unsigned __int64 v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // rbx
  _QWORD *v53; // r11
  __int64 v54; // rax
  int v55; // edx
  int v56; // edx
  unsigned int v57; // ecx
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rcx
  int v61; // edx
  int v62; // edx
  unsigned int v63; // ecx
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rcx
  __int64 *v67; // rcx
  __int16 v68; // si
  bool v69; // zf
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rdx
  unsigned int *v75; // r13
  unsigned int *v76; // rbx
  __int64 v77; // rdx
  unsigned int v78; // esi
  __int64 v79; // rsi
  unsigned __int8 *v80; // rsi
  unsigned int *v81; // r13
  unsigned int *v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  _BYTE *v85; // rax
  __int64 v86; // r13
  __int64 v87; // r15
  unsigned int *v88; // r13
  unsigned int *v89; // rbx
  __int64 v90; // rdx
  unsigned int v91; // esi
  __int64 v92; // r12
  unsigned __int8 *v93; // rax
  unsigned int *v94; // r12
  __int64 v95; // r15
  unsigned int *v96; // rbx
  __int64 v97; // rdx
  unsigned int v98; // esi
  __int64 v99; // rdi
  __int64 v100; // rbx
  unsigned int *v101; // r13
  unsigned int *v102; // rbx
  __int64 v103; // rdx
  unsigned int v104; // esi
  unsigned int *v105; // r13
  unsigned int *v106; // rbx
  __int64 v107; // rdx
  unsigned int v108; // esi
  __int64 v109; // rdi
  __int64 v110; // rbx
  unsigned int *v111; // r13
  unsigned int *v112; // rbx
  __int64 v113; // rdx
  unsigned int v114; // esi
  unsigned int *v115; // r13
  unsigned int *v116; // rbx
  __int64 v117; // rdx
  unsigned int v118; // esi
  unsigned __int64 v119; // rsi
  _QWORD *v120; // [rsp+0h] [rbp-1A0h]
  unsigned __int64 v121; // [rsp+18h] [rbp-188h]
  __int64 v125; // [rsp+40h] [rbp-160h]
  __int64 v126; // [rsp+48h] [rbp-158h]
  __int64 v128; // [rsp+60h] [rbp-140h]
  __int64 v129; // [rsp+68h] [rbp-138h]
  __int64 v130; // [rsp+68h] [rbp-138h]
  __int64 *v131; // [rsp+68h] [rbp-138h]
  unsigned __int8 *v132; // [rsp+68h] [rbp-138h]
  __int64 v133; // [rsp+68h] [rbp-138h]
  __int64 v134; // [rsp+68h] [rbp-138h]
  __int64 v135; // [rsp+70h] [rbp-130h] BYREF
  unsigned int v136; // [rsp+78h] [rbp-128h]
  int v137; // [rsp+7Ch] [rbp-124h]
  _QWORD v138[4]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v139; // [rsp+A0h] [rbp-100h]
  const char *v140[4]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v141; // [rsp+D0h] [rbp-D0h]
  unsigned int *v142; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v143; // [rsp+E8h] [rbp-B8h]
  _BYTE v144[32]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v145; // [rsp+110h] [rbp-90h]
  __int64 v146; // [rsp+118h] [rbp-88h]
  __int64 v147; // [rsp+120h] [rbp-80h]
  _QWORD *v148; // [rsp+128h] [rbp-78h]
  void **v149; // [rsp+130h] [rbp-70h]
  void **v150; // [rsp+138h] [rbp-68h]
  __int64 v151; // [rsp+140h] [rbp-60h]
  int v152; // [rsp+148h] [rbp-58h]
  __int16 v153; // [rsp+14Ch] [rbp-54h]
  char v154; // [rsp+14Eh] [rbp-52h]
  __int64 v155; // [rsp+150h] [rbp-50h]
  __int64 v156; // [rsp+158h] [rbp-48h]
  void *v157; // [rsp+160h] [rbp-40h] BYREF
  void *v158; // [rsp+168h] [rbp-38h] BYREF

  v13 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v13 == a3 + 48 )
  {
    v15 = 0;
  }
  else
  {
    if ( !v13 )
      BUG();
    v14 = *(unsigned __int8 *)(v13 - 24);
    v15 = 0;
    v16 = v13 - 24;
    if ( (unsigned int)(v14 - 30) < 0xB )
      v15 = v16;
  }
  v154 = 7;
  v148 = (_QWORD *)sub_BD5C60(v15);
  v149 = &v157;
  v150 = &v158;
  v143 = 0x200000000LL;
  v157 = &unk_49DA100;
  v142 = (unsigned int *)v144;
  v158 = &unk_49DA0B0;
  v151 = 0;
  v152 = 0;
  v153 = 512;
  v155 = 0;
  v156 = 0;
  v145 = 0;
  v146 = 0;
  LOWORD(v147) = 0;
  sub_D5F1F0((__int64)&v142, v15);
  v17 = *a8;
  v140[0] = v17;
  if ( v17 && (sub_B96E90((__int64)v140, (__int64)v17, 1), (v20 = v140[0]) != 0) )
  {
    v21 = v142;
    v22 = v143;
    v23 = &v142[4 * (unsigned int)v143];
    if ( v142 != v23 )
    {
      while ( *v21 )
      {
        v21 += 4;
        if ( v23 == v21 )
          goto LABEL_100;
      }
      *((const char **)v21 + 1) = v140[0];
      goto LABEL_12;
    }
LABEL_100:
    if ( (unsigned int)v143 >= (unsigned __int64)HIDWORD(v143) )
    {
      v119 = (unsigned int)v143 + 1LL;
      if ( HIDWORD(v143) < v119 )
      {
        sub_C8D5F0((__int64)&v142, v144, v119, 0x10u, v18, v19);
        v23 = &v142[4 * (unsigned int)v143];
      }
      *(_QWORD *)v23 = 0;
      *((_QWORD *)v23 + 1) = v20;
      v20 = v140[0];
      LODWORD(v143) = v143 + 1;
    }
    else
    {
      if ( v23 )
      {
        *v23 = 0;
        *((_QWORD *)v23 + 1) = v20;
        v22 = v143;
        v20 = v140[0];
      }
      LODWORD(v143) = v22 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v142, 0);
    v20 = v140[0];
  }
  if ( v20 )
LABEL_12:
    sub_B91220((__int64)v140, (__int64)v20);
  if ( a10 )
  {
    v24 = *a7;
    if ( *a7 == 56 )
    {
      v99 = *(_QWORD *)(a6 + 8);
      v139 = 257;
      v100 = sub_AD64C0(v99, 1, 0);
      v27 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD))*v149 + 3))(v149, 27, a6, v100, 0);
      if ( !v27 )
      {
        v141 = 257;
        v27 = sub_B504D0(27, a6, v100, (__int64)v140, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v150 + 2))(v150, v27, v138, v146, v147);
        v101 = v142;
        v102 = &v142[4 * (unsigned int)v143];
        if ( v142 != v102 )
        {
          do
          {
            v103 = *((_QWORD *)v101 + 1);
            v104 = *v101;
            v101 += 4;
            sub_B99FD0(v27, v104, v103);
          }
          while ( v102 != v101 );
        }
      }
    }
    else if ( v24 == 55 )
    {
      v109 = *(_QWORD *)(a6 + 8);
      v139 = 257;
      v110 = sub_AD64C0(v109, 1, 0);
      v27 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD))*v149 + 3))(v149, 26, a6, v110, 0);
      if ( !v27 )
      {
        v141 = 257;
        v27 = sub_B504D0(26, a6, v110, (__int64)v140, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v150 + 2))(v150, v27, v138, v146, v147);
        v111 = v142;
        v112 = &v142[4 * (unsigned int)v143];
        if ( v142 != v112 )
        {
          do
          {
            v113 = *((_QWORD *)v111 + 1);
            v114 = *v111;
            v111 += 4;
            sub_B99FD0(v27, v114, v113);
          }
          while ( v112 != v111 );
        }
      }
    }
    else
    {
      if ( v24 != 54 )
        BUG();
      v25 = *(_QWORD *)(a6 + 8);
      v139 = 257;
      v26 = sub_AD64C0(v25, 1, 0);
      v27 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v149 + 4))(
              v149,
              25,
              a6,
              v26,
              0,
              0);
      if ( !v27 )
      {
        v141 = 257;
        v27 = sub_B504D0(25, a6, v26, (__int64)v140, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v150 + 2))(v150, v27, v138, v146, v147);
        v105 = v142;
        v106 = &v142[4 * (unsigned int)v143];
        if ( v142 != v106 )
        {
          do
          {
            v107 = *((_QWORD *)v105 + 1);
            v108 = *v105;
            v105 += 4;
            sub_B99FD0(v27, v108, v107);
          }
          while ( v106 != v105 );
        }
      }
    }
    a6 = v27;
  }
  v138[0] = a6;
  v28 = sub_BCB2A0(v148);
  v138[1] = sub_ACD640(v28, a9, 0);
  v29 = *(_QWORD *)(a6 + 8);
  v137 = 0;
  v135 = v29;
  v141 = 257;
  v30 = sub_B33D10((__int64)&v142, a2, (__int64)&v135, 1, (int)v138, 2, v136, (__int64)v140);
  v31 = *a8;
  v140[0] = v31;
  if ( !v31 )
  {
    v32 = v30 + 48;
    if ( (const char **)(v30 + 48) == v140 )
      goto LABEL_23;
    v79 = *(_QWORD *)(v30 + 48);
    if ( !v79 )
      goto LABEL_23;
LABEL_97:
    sub_B91220(v32, v79);
    goto LABEL_98;
  }
  v32 = v30 + 48;
  sub_B96E90((__int64)v140, (__int64)v31, 1);
  if ( (const char **)(v30 + 48) == v140 )
  {
    if ( v140[0] )
      sub_B91220((__int64)v140, (__int64)v140[0]);
    goto LABEL_23;
  }
  v79 = *(_QWORD *)(v30 + 48);
  if ( v79 )
    goto LABEL_97;
LABEL_98:
  v80 = (unsigned __int8 *)v140[0];
  *(const char **)(v30 + 48) = v140[0];
  if ( v80 )
    sub_B976B0((__int64)v140, v80, v32);
LABEL_23:
  v33 = *(_QWORD *)(v30 + 8);
  v139 = 257;
  v128 = v33;
  v34 = sub_AD64C0(v33, *(_DWORD *)(v33 + 8) >> 8, 0);
  v35 = (_BYTE *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v149 + 4))(
                   v149,
                   15,
                   v34,
                   v30,
                   0,
                   0);
  if ( !v35 )
  {
    v141 = 257;
    v35 = (_BYTE *)sub_B504D0(15, v34, v30, (__int64)v140, 0, 0);
    (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v150 + 2))(v150, v35, v138, v146, v147);
    v88 = v142;
    v89 = &v142[4 * (unsigned int)v143];
    if ( v142 != v89 )
    {
      do
      {
        v90 = *((_QWORD *)v88 + 1);
        v91 = *v88;
        v88 += 4;
        sub_B99FD0((__int64)v35, v91, v90);
      }
      while ( v89 != v88 );
    }
  }
  if ( a11 )
  {
    v139 = 257;
    v86 = sub_AD64C0(v33, 1, 0);
    v87 = (*((__int64 (__fastcall **)(void **, __int64, _BYTE *, __int64, _QWORD, _QWORD))*v149 + 4))(
            v149,
            15,
            v35,
            v86,
            0,
            0);
    if ( !v87 )
    {
      v141 = 257;
      v87 = sub_B504D0(15, (__int64)v35, v86, (__int64)v140, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v150 + 2))(v150, v87, v138, v146, v147);
      v115 = v142;
      v116 = &v142[4 * (unsigned int)v143];
      if ( v142 != v116 )
      {
        do
        {
          v117 = *((_QWORD *)v115 + 1);
          v118 = *v115;
          v115 += 4;
          sub_B99FD0(v87, v118, v117);
        }
        while ( v116 != v115 );
      }
    }
    v35 = (_BYTE *)v87;
  }
  v125 = (__int64)v35;
  if ( a10 )
  {
    v141 = 257;
    v85 = (_BYTE *)sub_AD64C0(v33, 1, 0);
    v125 = sub_929C50(&v142, v35, v85, (__int64)v140, 0, 0);
  }
  v36 = *(_QWORD *)(a4 + 8);
  v141 = 257;
  v37 = sub_A830B0(&v142, (__int64)v35, v36, (__int64)v140);
  v38 = *(_DWORD *)(a5 + 4) & 0x7FFFFFF;
  if ( v38 )
  {
    v39 = *(_QWORD *)(a5 - 8);
    v40 = 0;
    do
    {
      if ( a3 == *(_QWORD *)(v39 + 32LL * *(unsigned int *)(a5 + 72) + 8 * v40) )
      {
        v41 = 32 * v40;
        goto LABEL_32;
      }
      ++v40;
    }
    while ( v38 != (_DWORD)v40 );
    v41 = 0x1FFFFFFFE0LL;
  }
  else
  {
    v41 = 0x1FFFFFFFE0LL;
    v39 = *(_QWORD *)(a5 - 8);
  }
LABEL_32:
  if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
  {
    v42 = *(_QWORD *)(*(_QWORD *)(a4 - 8) + 32LL);
    v43 = *(_DWORD *)(v42 + 32);
    if ( v43 > 0x40 )
    {
LABEL_34:
      v44 = v43 - 1 == (unsigned int)sub_C444A0(v42 + 24);
      goto LABEL_35;
    }
  }
  else
  {
    v42 = *(_QWORD *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) + 32);
    v43 = *(_DWORD *)(v42 + 32);
    if ( v43 > 0x40 )
      goto LABEL_34;
  }
  v44 = *(_QWORD *)(v42 + 24) == 1;
LABEL_35:
  v45 = *(_QWORD *)(v39 + v41);
  if ( v44 )
  {
    if ( *(_BYTE *)v45 == 17 )
    {
      v46 = *(_DWORD *)(v45 + 32);
      if ( v46 <= 0x40 )
      {
        v47 = (unsigned __int8 *)v37;
        if ( !*(_QWORD *)(v45 + 24) )
          goto LABEL_39;
      }
      else
      {
        v47 = (unsigned __int8 *)v37;
        if ( v46 == (unsigned int)sub_C444A0(v45 + 24) )
          goto LABEL_39;
      }
    }
    v139 = 257;
    v47 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v149 + 4))(
                               v149,
                               13,
                               v37,
                               v45,
                               0,
                               0);
    if ( !v47 )
    {
      v141 = 257;
      v47 = (unsigned __int8 *)sub_B504D0(13, v37, v45, (__int64)v140, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v150 + 2))(
        v150,
        v47,
        v138,
        v146,
        v147);
      v75 = v142;
      v76 = &v142[4 * (unsigned int)v143];
      if ( v142 != v76 )
      {
        do
        {
          v77 = *((_QWORD *)v75 + 1);
          v78 = *v75;
          v75 += 4;
          sub_B99FD0((__int64)v47, v78, v77);
        }
        while ( v76 != v75 );
      }
    }
  }
  else
  {
    v139 = 257;
    v47 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v149 + 4))(
                               v149,
                               15,
                               v45,
                               v37,
                               0,
                               0);
    if ( !v47 )
    {
      v141 = 257;
      v47 = (unsigned __int8 *)sub_B504D0(15, v45, v37, (__int64)v140, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v150 + 2))(
        v150,
        v47,
        v138,
        v146,
        v147);
      v81 = v142;
      v82 = &v142[4 * (unsigned int)v143];
      if ( v142 != v82 )
      {
        do
        {
          v83 = *((_QWORD *)v81 + 1);
          v84 = *v81;
          v81 += 4;
          sub_B99FD0((__int64)v47, v84, v83);
        }
        while ( v82 != v81 );
      }
    }
  }
LABEL_39:
  v48 = **(__int64 ***)(*a1 + 32);
  v49 = v48[6] & 0xFFFFFFFFFFFFFFF8LL;
  v121 = v49;
  if ( (__int64 *)v49 == v48 + 6 )
    goto LABEL_143;
  if ( !v49 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v49 - 24) - 30 > 0xA )
LABEL_143:
    BUG();
  v50 = *(_QWORD *)(v49 - 120);
  v140[0] = "tcphi";
  v141 = 259;
  v51 = sub_BD2DA0(80);
  v52 = v51;
  if ( v51 )
  {
    v120 = (_QWORD *)v51;
    sub_B44260(v51, v128, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v52 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v52, v140);
    sub_BD2A10(v52, *(_DWORD *)(v52 + 72), 1);
    v53 = v120;
  }
  else
  {
    v53 = 0;
  }
  sub_B44220(v53, v48[7], 1);
  sub_D5F1F0((__int64)&v142, v50);
  v138[0] = "tcdec";
  v139 = 259;
  v129 = sub_AD64C0(v128, 1, 0);
  v54 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, __int64))*v149 + 4))(
          v149,
          15,
          v52,
          v129,
          0,
          1);
  if ( v54 )
  {
    v55 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
    if ( v55 != *(_DWORD *)(v52 + 72) )
      goto LABEL_46;
LABEL_120:
    v133 = v54;
    sub_B48D90(v52);
    v54 = v133;
    v55 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
    goto LABEL_46;
  }
  v141 = 257;
  v130 = sub_B504D0(15, v52, v129, (__int64)v140, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v150 + 2))(v150, v130, v138, v146, v147);
  v92 = 4LL * (unsigned int)v143;
  v93 = (unsigned __int8 *)v130;
  if ( v142 != &v142[v92] )
  {
    v131 = v48;
    v94 = &v142[v92];
    v95 = (__int64)v93;
    v126 = v52;
    v96 = v142;
    do
    {
      v97 = *((_QWORD *)v96 + 1);
      v98 = *v96;
      v96 += 4;
      sub_B99FD0(v95, v98, v97);
    }
    while ( v94 != v96 );
    v93 = (unsigned __int8 *)v95;
    v52 = v126;
    v48 = v131;
  }
  v132 = v93;
  sub_B44850(v93, 1);
  v54 = (__int64)v132;
  v55 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
  if ( v55 == *(_DWORD *)(v52 + 72) )
    goto LABEL_120;
LABEL_46:
  v56 = (v55 + 1) & 0x7FFFFFF;
  v57 = v56 | *(_DWORD *)(v52 + 4) & 0xF8000000;
  v58 = *(_QWORD *)(v52 - 8) + 32LL * (unsigned int)(v56 - 1);
  *(_DWORD *)(v52 + 4) = v57;
  if ( *(_QWORD *)v58 )
  {
    v59 = *(_QWORD *)(v58 + 8);
    **(_QWORD **)(v58 + 16) = v59;
    if ( v59 )
      *(_QWORD *)(v59 + 16) = *(_QWORD *)(v58 + 16);
  }
  *(_QWORD *)v58 = v125;
  if ( v125 )
  {
    v60 = *(_QWORD *)(v125 + 16);
    *(_QWORD *)(v58 + 8) = v60;
    if ( v60 )
      *(_QWORD *)(v60 + 16) = v58 + 8;
    *(_QWORD *)(v58 + 16) = v125 + 16;
    *(_QWORD *)(v125 + 16) = v58;
  }
  *(_QWORD *)(*(_QWORD *)(v52 - 8) + 32LL * *(unsigned int *)(v52 + 72)
                                   + 8LL * ((*(_DWORD *)(v52 + 4) & 0x7FFFFFFu) - 1)) = a3;
  v61 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
  if ( v61 == *(_DWORD *)(v52 + 72) )
  {
    v134 = v54;
    sub_B48D90(v52);
    v54 = v134;
    v61 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
  }
  v62 = (v61 + 1) & 0x7FFFFFF;
  v63 = v62 | *(_DWORD *)(v52 + 4) & 0xF8000000;
  v64 = *(_QWORD *)(v52 - 8) + 32LL * (unsigned int)(v62 - 1);
  *(_DWORD *)(v52 + 4) = v63;
  if ( *(_QWORD *)v64 )
  {
    v65 = *(_QWORD *)(v64 + 8);
    **(_QWORD **)(v64 + 16) = v65;
    if ( v65 )
      *(_QWORD *)(v65 + 16) = *(_QWORD *)(v64 + 16);
  }
  *(_QWORD *)v64 = v54;
  if ( v54 )
  {
    v66 = *(_QWORD *)(v54 + 16);
    *(_QWORD *)(v64 + 8) = v66;
    if ( v66 )
      *(_QWORD *)(v66 + 16) = v64 + 8;
    *(_QWORD *)(v64 + 16) = v54 + 16;
    *(_QWORD *)(v54 + 16) = v64;
  }
  *(_QWORD *)(*(_QWORD *)(v52 - 8) + 32LL * *(unsigned int *)(v52 + 72)
                                   + 8LL * ((*(_DWORD *)(v52 + 4) & 0x7FFFFFFu) - 1)) = v48;
  v67 = *(__int64 **)(v121 - 56);
  if ( !v67 || (v68 = 33, v48 != v67) )
    v68 = 32;
  v69 = *(_QWORD *)(v50 - 64) == 0;
  *(_WORD *)(v50 + 2) = v68 | *(_WORD *)(v50 + 2) & 0xFFC0;
  if ( !v69 )
  {
    v70 = *(_QWORD *)(v50 - 56);
    **(_QWORD **)(v50 - 48) = v70;
    if ( v70 )
      *(_QWORD *)(v70 + 16) = *(_QWORD *)(v50 - 48);
  }
  *(_QWORD *)(v50 - 64) = v54;
  if ( v54 )
  {
    v71 = *(_QWORD *)(v54 + 16);
    *(_QWORD *)(v50 - 56) = v71;
    if ( v71 )
      *(_QWORD *)(v71 + 16) = v50 - 56;
    *(_QWORD *)(v50 - 48) = v54 + 16;
    *(_QWORD *)(v54 + 16) = v50 - 64;
  }
  v72 = sub_AD64C0(v128, 0, 0);
  if ( *(_QWORD *)(v50 - 32) )
  {
    v73 = *(_QWORD *)(v50 - 24);
    **(_QWORD **)(v50 - 16) = v73;
    if ( v73 )
      *(_QWORD *)(v73 + 16) = *(_QWORD *)(v50 - 16);
  }
  *(_QWORD *)(v50 - 32) = v72;
  if ( v72 )
  {
    v74 = *(_QWORD *)(v72 + 16);
    *(_QWORD *)(v50 - 24) = v74;
    if ( v74 )
      *(_QWORD *)(v74 + 16) = v50 - 24;
    *(_QWORD *)(v50 - 16) = v72 + 16;
    *(_QWORD *)(v72 + 16) = v50 - 32;
  }
  if ( a10 )
    sub_BD7E80((unsigned __int8 *)a5, v47, v48);
  else
    sub_BD7E80((unsigned __int8 *)a4, v47, v48);
  sub_DAC210(a1[4], *a1);
  nullsub_61();
  v157 = &unk_49DA100;
  nullsub_63();
  if ( v142 != (unsigned int *)v144 )
    _libc_free((unsigned __int64)v142);
}
