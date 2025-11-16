// Function: sub_2C5B160
// Address: 0x2c5b160
//
__int64 __fastcall sub_2C5B160(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rbx
  unsigned __int8 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned int v9; // r14d
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r14
  void *v13; // r11
  __int64 v14; // r10
  _DWORD *v15; // rax
  _BYTE *v16; // rsi
  _DWORD *v17; // rdx
  char *v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // r10
  int v26; // edx
  int v27; // r8d
  int v28; // edx
  bool v29; // of
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rdx
  int v35; // r14d
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r10
  signed __int64 v41; // rax
  signed __int64 v42; // r11
  int v43; // eax
  _QWORD *v44; // rax
  __int64 v45; // rax
  int v46; // edx
  unsigned int v47; // eax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rdi
  unsigned __int8 *v52; // r9
  unsigned int v53; // r15d
  __int64 (__fastcall *v54)(__int64, unsigned int, _BYTE *); // rax
  __int64 v55; // rax
  __int64 v56; // r14
  __int64 v57; // r15
  __int64 v58; // rax
  __int64 v59; // rdi
  _BYTE *v60; // r11
  __int64 (__fastcall *v61)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v62; // rax
  _BYTE *v63; // r10
  __int64 v64; // rdi
  _DWORD *v65; // r11
  __int64 v66; // r14
  __int64 (__fastcall *v67)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 v70; // rbx
  __int64 i; // r14
  __int64 v72; // rax
  int v73; // edx
  bool v74; // zf
  int v75; // edx
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 v80; // rbx
  __int64 v81; // r15
  __int64 v82; // rdx
  unsigned int v83; // esi
  __int64 v84; // rdi
  _DWORD *v85; // r11
  __int64 v86; // r10
  __int64 (__fastcall *v87)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v88; // rax
  _QWORD *v89; // rax
  __int64 v90; // rax
  __int64 v91; // rcx
  unsigned __int8 *v92; // rax
  __int64 v93; // r12
  __int64 v94; // rbx
  unsigned __int8 *v95; // r14
  __int64 v96; // rdx
  unsigned int v97; // esi
  __int64 v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // rax
  __int64 v101; // rcx
  __int64 v102; // rbx
  __int64 v103; // r14
  __int64 v104; // rdx
  unsigned int v105; // esi
  _QWORD *v106; // rax
  _QWORD *v107; // r10
  void *v108; // rcx
  __int64 v109; // r15
  __int64 v110; // r14
  __int64 v111; // rbx
  __int64 v112; // r15
  __int64 v113; // r14
  __int64 v114; // rdx
  unsigned int v115; // esi
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rax
  bool v119; // cc
  unsigned __int64 v120; // rax
  __int64 v121; // [rsp+0h] [rbp-190h]
  unsigned __int64 n; // [rsp+8h] [rbp-188h]
  signed __int64 v123; // [rsp+18h] [rbp-178h]
  __int64 v124; // [rsp+20h] [rbp-170h]
  __int64 v125; // [rsp+20h] [rbp-170h]
  signed __int64 v126; // [rsp+20h] [rbp-170h]
  __int64 v127; // [rsp+28h] [rbp-168h]
  __int64 v128; // [rsp+28h] [rbp-168h]
  int v129; // [rsp+28h] [rbp-168h]
  int v130; // [rsp+38h] [rbp-158h]
  int v131; // [rsp+38h] [rbp-158h]
  __int64 v132; // [rsp+38h] [rbp-158h]
  void *s; // [rsp+40h] [rbp-150h]
  int sb; // [rsp+40h] [rbp-150h]
  void *sa; // [rsp+40h] [rbp-150h]
  signed __int64 v136; // [rsp+48h] [rbp-148h]
  signed __int64 v137; // [rsp+48h] [rbp-148h]
  unsigned int v138; // [rsp+50h] [rbp-140h]
  _BYTE *v139; // [rsp+50h] [rbp-140h]
  _DWORD *v140; // [rsp+50h] [rbp-140h]
  _DWORD *v141; // [rsp+50h] [rbp-140h]
  __int64 v142; // [rsp+50h] [rbp-140h]
  __int64 v143; // [rsp+50h] [rbp-140h]
  __int64 v144; // [rsp+50h] [rbp-140h]
  _BYTE *v145; // [rsp+50h] [rbp-140h]
  _DWORD *v146; // [rsp+50h] [rbp-140h]
  _DWORD *v147; // [rsp+50h] [rbp-140h]
  _BYTE *v148; // [rsp+58h] [rbp-138h]
  unsigned __int64 v149; // [rsp+60h] [rbp-130h]
  __int64 v150; // [rsp+60h] [rbp-130h]
  __int64 v151; // [rsp+68h] [rbp-128h]
  unsigned __int8 *v152; // [rsp+68h] [rbp-128h]
  _DWORD *v153; // [rsp+68h] [rbp-128h]
  _BYTE *v154; // [rsp+68h] [rbp-128h]
  __int64 v155; // [rsp+68h] [rbp-128h]
  __int64 v156; // [rsp+68h] [rbp-128h]
  void *v157; // [rsp+68h] [rbp-128h]
  __int64 v158; // [rsp+68h] [rbp-128h]
  unsigned __int8 *v159; // [rsp+68h] [rbp-128h]
  void *v160; // [rsp+68h] [rbp-128h]
  __int64 v161; // [rsp+68h] [rbp-128h]
  _QWORD *v162; // [rsp+68h] [rbp-128h]
  _BYTE *v163; // [rsp+68h] [rbp-128h]
  __int64 v164; // [rsp+68h] [rbp-128h]
  _BYTE *v165; // [rsp+68h] [rbp-128h]
  __int64 v166; // [rsp+68h] [rbp-128h]
  unsigned __int8 *v167; // [rsp+70h] [rbp-120h] BYREF
  __int64 v168; // [rsp+78h] [rbp-118h] BYREF
  _BYTE v169[32]; // [rsp+80h] [rbp-110h] BYREF
  __int16 v170; // [rsp+A0h] [rbp-F0h]
  _BYTE v171[32]; // [rsp+B0h] [rbp-E0h] BYREF
  __int16 v172; // [rsp+D0h] [rbp-C0h]
  _DWORD *v173; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v174; // [rsp+E8h] [rbp-A8h]
  _BYTE v175[48]; // [rsp+F0h] [rbp-A0h] BYREF
  void *v176; // [rsp+120h] [rbp-70h] BYREF
  __int64 v177; // [rsp+128h] [rbp-68h]
  _QWORD v178[12]; // [rsp+130h] [rbp-60h] BYREF

  if ( *(_BYTE *)a2 != 91 )
    return 0;
  v4 = a1;
  v5 = (unsigned __int8 *)a2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a2 - 8);
    v148 = *(_BYTE **)v6;
    if ( !*(_QWORD *)v6 )
      return 0;
  }
  else
  {
    v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v148 = *(_BYTE **)v6;
    if ( !*(_QWORD *)v6 )
      return 0;
  }
  v151 = *(_QWORD *)(v6 + 32);
  v7 = *(_QWORD *)(v151 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v151 <= 0x1Cu )
    return 0;
  v8 = *(_QWORD *)(v6 + 64);
  if ( *(_BYTE *)v8 != 17 )
    return 0;
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
  {
    v149 = *(_QWORD *)(v8 + 24);
    goto LABEL_12;
  }
  if ( v9 - (unsigned int)sub_C444A0(v8 + 24) > 0x40 )
    return 0;
  v149 = **(_QWORD **)(v8 + 24);
LABEL_12:
  v176 = &v168;
  v177 = (__int64)&v167;
  v178[0] = v149;
  v2 = sub_2C530C0((__int64)&v176, (unsigned __int8 *)v151);
  if ( !(_BYTE)v2 )
    return v2;
  v12 = *(_QWORD *)(a2 + 8);
  v13 = (void *)v12;
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
    v13 = **(void ***)(v12 + 16);
  v14 = *((_QWORD *)v167 + 1);
  if ( *(_BYTE *)(v14 + 8) != 17 )
    return 0;
  if ( v13 != **(void ***)(v14 + 16) )
    return 0;
  v138 = *(_DWORD *)(v12 + 32);
  if ( v138 <= v149 )
    return 0;
  v15 = v175;
  v16 = v175;
  v173 = v175;
  v174 = 0xC00000000LL;
  if ( v138 > 0xCuLL )
  {
    v132 = v14;
    sa = v13;
    sub_C8D5F0((__int64)&v173, v175, v138, 4u, v10, v11);
    v16 = v173;
    v14 = v132;
    v13 = sa;
    v15 = &v173[(unsigned int)v174];
  }
  v17 = &v16[4 * v138];
  n = 4LL * v138;
  if ( v17 != v15 )
  {
    do
    {
      if ( v15 )
        *v15 = 0;
      ++v15;
    }
    while ( v17 != v15 );
    v16 = v173;
    v15 = &v173[n / 4];
  }
  v18 = (char *)(v15 - 1);
  v19 = 0;
  LODWORD(v174) = v138;
  v20 = (unsigned __int64)(v18 - v16) >> 2;
  do
  {
    v21 = v19;
    *(_DWORD *)&v16[4 * v19] = v19;
    ++v19;
  }
  while ( v20 != v21 );
  v127 = v14;
  v124 = (__int64)v13;
  v173[v149] = v149 + v138;
  v22 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
  v130 = v23;
  s = (void *)v22;
  v24 = sub_DFD800(*(_QWORD *)(a1 + 152), 0xCu, v124, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
  v25 = v127;
  v27 = v26;
  v28 = 1;
  if ( v130 != 1 )
    v28 = v27;
  v29 = __OFADD__(s, v24);
  v30 = (unsigned __int64)s + v24;
  v131 = v28;
  if ( v29 )
  {
    v30 = 0x8000000000000000LL;
    if ( (__int64)s > 0 )
      v30 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v123 = v30;
  v31 = *(_QWORD *)(v168 + 16);
  if ( v31 && !*(_QWORD *)(v31 + 8) )
  {
    v72 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
    v25 = v127;
    v74 = v73 == 1;
    v75 = 1;
    if ( !v74 )
      v75 = v131;
    v131 = v75;
    if ( __OFADD__(v72, v123) )
    {
      v119 = v72 <= 0;
      v120 = 0x8000000000000000LL;
      if ( !v119 )
        v120 = 0x7FFFFFFFFFFFFFFFLL;
      v123 = v120;
    }
    else
    {
      v123 += v72;
    }
  }
  v125 = v25;
  v32 = sub_DFBC30(
          *(__int64 **)(a1 + 152),
          6,
          v12,
          (__int64)v173,
          (unsigned int)v174,
          *(unsigned int *)(a1 + 192),
          0,
          0,
          0,
          0,
          0);
  sb = v33;
  v34 = v12;
  v35 = 1;
  v128 = v32;
  v36 = sub_DFD800(*(_QWORD *)(a1 + 152), 0xCu, v34, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
  if ( sb != 1 )
    v35 = v37;
  v40 = v125;
  v29 = __OFADD__(v128, v36);
  v41 = v128 + v36;
  if ( v29 )
  {
    v42 = 0x8000000000000000LL;
    if ( v128 > 0 )
      v42 = 0x7FFFFFFFFFFFFFFFLL;
  }
  else
  {
    v42 = v41;
  }
  v43 = *(_DWORD *)(v125 + 32);
  v176 = v178;
  v129 = v43;
  v177 = 0xC00000000LL;
  if ( v138 != v43 )
  {
    if ( v138 > 0xCuLL )
    {
      v121 = v125;
      v126 = v42;
      sub_C8D5F0((__int64)&v176, v178, v138, 4u, v38, v39);
      memset(v176, 255, n);
      v42 = v126;
      v40 = v121;
      LODWORD(v177) = v138;
      v44 = v176;
    }
    else
    {
      if ( v138 )
      {
        v136 = v42;
        memset(v178, 255, n);
        v40 = v125;
        v42 = v136;
      }
      LODWORD(v177) = v138;
      v44 = v178;
    }
    v137 = v42;
    *((_DWORD *)v44 + v149) = v149;
    v45 = sub_DFBC30(
            *(__int64 **)(a1 + 152),
            7,
            v40,
            (__int64)v176,
            (unsigned int)v177,
            *(unsigned int *)(a1 + 192),
            0,
            0,
            0,
            0,
            0);
    if ( v46 == 1 )
      v35 = 1;
    v42 = v45 + v137;
    if ( __OFADD__(v45, v137) )
    {
      v42 = 0x8000000000000000LL;
      if ( v45 > 0 )
        v42 = 0x7FFFFFFFFFFFFFFFLL;
    }
  }
  if ( v131 == v35 )
  {
    if ( v42 > v123 )
      goto LABEL_45;
LABEL_53:
    v150 = a1 + 8;
    v170 = 257;
    v47 = sub_B45210(v151);
    v51 = *(_QWORD *)(a1 + 88);
    v52 = v167;
    v53 = v47;
    v54 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*(_QWORD *)v51 + 48LL);
    if ( v54 == sub_9288C0 )
    {
      if ( *v167 > 0x15u )
      {
LABEL_82:
        v172 = 257;
        v76 = sub_B50340(12, (__int64)v52, (__int64)v171, 0, 0);
        v77 = *(_QWORD *)(v4 + 104);
        v56 = v76;
        if ( v77 )
          sub_B99FD0(v76, 3u, v77);
        sub_B45150(v56, v53);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 96) + 16LL))(
          *(_QWORD *)(v4 + 96),
          v56,
          v169,
          *(_QWORD *)(v4 + 64),
          *(_QWORD *)(v4 + 72));
        v78 = *(_QWORD *)(v4 + 8);
        v79 = v78 + 16LL * *(unsigned int *)(v4 + 16);
        if ( v78 != v79 )
        {
          v155 = v4;
          v80 = *(_QWORD *)(v4 + 8);
          v81 = v79;
          do
          {
            v82 = *(_QWORD *)(v80 + 8);
            v83 = *(_DWORD *)v80;
            v80 += 16;
            sub_B99FD0(v56, v83, v82);
          }
          while ( v81 != v80 );
          v4 = v155;
        }
LABEL_57:
        v170 = 257;
        if ( v138 != v129 )
        {
          v57 = (unsigned int)v177;
          v153 = v176;
          v58 = sub_ACADE0(*(__int64 ***)(v56 + 8));
          v59 = *(_QWORD *)(v4 + 88);
          v60 = (_BYTE *)v58;
          v61 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v59 + 112LL);
          if ( v61 == sub_9B6630 )
          {
            if ( *(_BYTE *)v56 > 0x15u || *v60 > 0x15u )
            {
LABEL_107:
              v144 = (__int64)v60;
              v172 = 257;
              v106 = sub_BD2C40(112, unk_3F1FE60);
              v107 = v106;
              if ( v106 )
              {
                v108 = v153;
                v162 = v106;
                sub_B4E9E0((__int64)v106, v56, v144, v108, v57, (__int64)v171, 0, 0);
                v107 = v162;
              }
              v163 = v107;
              (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 96) + 16LL))(
                *(_QWORD *)(v4 + 96),
                v107,
                v169,
                *(_QWORD *)(v150 + 56),
                *(_QWORD *)(v150 + 64));
              v109 = *(_QWORD *)(v4 + 8);
              v63 = v163;
              v110 = v109 + 16LL * *(unsigned int *)(v4 + 16);
              if ( v109 != v110 )
              {
                v164 = v4;
                v111 = *(_QWORD *)(v4 + 8);
                v112 = v110;
                v113 = (__int64)v63;
                do
                {
                  v114 = *(_QWORD *)(v111 + 8);
                  v115 = *(_DWORD *)v111;
                  v111 += 16;
                  sub_B99FD0(v113, v115, v114);
                }
                while ( v112 != v111 );
                v4 = v164;
                v63 = (_BYTE *)v113;
              }
LABEL_63:
              v64 = *(_QWORD *)(v4 + 88);
              v65 = v173;
              v170 = 257;
              v66 = (unsigned int)v174;
              v67 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v64 + 112LL);
              if ( v67 == sub_9B6630 )
              {
                if ( *v148 > 0x15u || *v63 > 0x15u )
                  goto LABEL_101;
                v140 = v173;
                v154 = v63;
                v68 = sub_AD5CE0((__int64)v148, (__int64)v63, v173, (unsigned int)v174, 0);
                v63 = v154;
                v65 = v140;
                v69 = v68;
              }
              else
              {
                v146 = v173;
                v165 = v63;
                v117 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, _QWORD))v67)(
                         v64,
                         v148,
                         v63,
                         v173,
                         (unsigned int)v174);
                v65 = v146;
                v63 = v165;
                v69 = v117;
              }
              if ( v69 )
              {
LABEL_68:
                v70 = v4 + 200;
                sub_BD84D0((__int64)v5, v69);
                if ( *(_BYTE *)v69 > 0x1Cu )
                {
                  sub_BD6B90((unsigned __int8 *)v69, v5);
                  for ( i = *(_QWORD *)(v69 + 16); i; i = *(_QWORD *)(i + 8) )
                    sub_F15FC0(v70, *(_QWORD *)(i + 24));
                  if ( *(_BYTE *)v69 > 0x1Cu )
                    sub_F15FC0(v70, v69);
                }
                if ( *v5 > 0x1Cu )
                  sub_F15FC0(v70, (__int64)v5);
                goto LABEL_46;
              }
LABEL_101:
              v143 = (__int64)v63;
              v160 = v65;
              v172 = 257;
              v99 = sub_BD2C40(112, unk_3F1FE60);
              v69 = (__int64)v99;
              if ( v99 )
                sub_B4E9E0((__int64)v99, (__int64)v148, v143, v160, v66, (__int64)v171, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 96) + 16LL))(
                *(_QWORD *)(v4 + 96),
                v69,
                v169,
                *(_QWORD *)(v150 + 56),
                *(_QWORD *)(v150 + 64));
              v100 = *(_QWORD *)(v4 + 8);
              v101 = v100 + 16LL * *(unsigned int *)(v4 + 16);
              if ( v100 != v101 )
              {
                v161 = v4;
                v102 = *(_QWORD *)(v4 + 8);
                v103 = v101;
                do
                {
                  v104 = *(_QWORD *)(v102 + 8);
                  v105 = *(_DWORD *)v102;
                  v102 += 16;
                  sub_B99FD0(v69, v105, v104);
                }
                while ( v103 != v102 );
                v4 = v161;
              }
              goto LABEL_68;
            }
            v139 = v60;
            v62 = sub_AD5CE0(v56, (__int64)v60, v153, v57, 0);
            v60 = v139;
            v63 = (_BYTE *)v62;
          }
          else
          {
            v145 = v60;
            v116 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v61)(v59, v56, v60, v153, v57);
            v60 = v145;
            v63 = (_BYTE *)v116;
          }
          if ( v63 )
            goto LABEL_63;
          goto LABEL_107;
        }
        v84 = *(_QWORD *)(v4 + 88);
        v85 = v173;
        v86 = (unsigned int)v174;
        v87 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v84 + 112LL);
        if ( v87 == sub_9B6630 )
        {
          if ( *v148 > 0x15u || *(_BYTE *)v56 > 0x15u )
            goto LABEL_93;
          v141 = v173;
          v156 = (unsigned int)v174;
          v88 = sub_AD5CE0((__int64)v148, v56, v173, (unsigned int)v174, 0);
          v86 = v156;
          v85 = v141;
          v69 = v88;
        }
        else
        {
          v147 = v173;
          v166 = (unsigned int)v174;
          v118 = v87(v84, v148, (_BYTE *)v56, (__int64)v173, (unsigned int)v174, (__int64)v52);
          v85 = v147;
          v86 = v166;
          v69 = v118;
        }
        if ( v69 )
          goto LABEL_68;
LABEL_93:
        v142 = v86;
        v157 = v85;
        v172 = 257;
        v89 = sub_BD2C40(112, unk_3F1FE60);
        v69 = (__int64)v89;
        if ( v89 )
          sub_B4E9E0((__int64)v89, (__int64)v148, v56, v157, v142, (__int64)v171, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 96) + 16LL))(
          *(_QWORD *)(v4 + 96),
          v69,
          v169,
          *(_QWORD *)(v150 + 56),
          *(_QWORD *)(v150 + 64));
        v90 = *(_QWORD *)(v4 + 8);
        v91 = v90 + 16LL * *(unsigned int *)(v4 + 16);
        if ( v90 != v91 )
        {
          v92 = v5;
          v158 = v4;
          v93 = *(_QWORD *)(v4 + 8);
          v94 = v91;
          v95 = v92;
          do
          {
            v96 = *(_QWORD *)(v93 + 8);
            v97 = *(_DWORD *)v93;
            v93 += 16;
            sub_B99FD0(v69, v97, v96);
          }
          while ( v94 != v93 );
          v4 = v158;
          v5 = v95;
        }
        goto LABEL_68;
      }
      v152 = v167;
      v55 = sub_AAAFF0(12, v167, v48, v49, v50);
      v52 = v152;
      v56 = v55;
    }
    else
    {
      v159 = v167;
      v98 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int8 *, _QWORD))v54)(v51, 12, v167, v53);
      v52 = v159;
      v56 = v98;
    }
    if ( v56 )
      goto LABEL_57;
    goto LABEL_82;
  }
  if ( v131 >= v35 )
    goto LABEL_53;
LABEL_45:
  v2 = 0;
LABEL_46:
  if ( v176 != v178 )
    _libc_free((unsigned __int64)v176);
  if ( v173 != (_DWORD *)v175 )
    _libc_free((unsigned __int64)v173);
  return v2;
}
