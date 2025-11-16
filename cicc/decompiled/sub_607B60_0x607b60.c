// Function: sub_607B60
// Address: 0x607b60
//
__int64 __fastcall sub_607B60(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int16 a5,
        unsigned int a6,
        unsigned int a7,
        unsigned int a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r12
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r13
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // ebx
  bool v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  char v28; // dl
  char v29; // al
  char v30; // al
  __int64 v31; // rax
  char v32; // dl
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // r13
  __int64 v38; // rcx
  __int64 v39; // r9
  __int64 v40; // rbx
  __int64 v41; // r13
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  _QWORD **v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rdx
  _BYTE *v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rcx
  unsigned __int16 v54; // ax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rsi
  unsigned __int64 v58; // rcx
  __int64 v59; // rdi
  unsigned int v60; // r15d
  __int64 v61; // rdx
  int v62; // r14d
  __int64 v63; // rdx
  __int64 v64; // rcx
  unsigned __int16 v65; // ax
  void *v66; // rdx
  int v67; // eax
  unsigned int i; // ecx
  __int64 v69; // rax
  __int64 v70; // rsi
  unsigned __int16 v71; // ax
  __int64 v72; // rax
  __int64 v73; // r14
  unsigned int v74; // eax
  int v75; // eax
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // rcx
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // rcx
  unsigned __int64 v96; // rax
  __int64 v97; // rdi
  __int64 v98; // rdi
  __int16 v99; // ax
  __int64 v100; // rdi
  __int64 v101; // rax
  int v102; // edx
  __int64 v103; // r14
  __int64 v104; // r14
  __int64 v105; // rax
  __int64 j; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __m128i v115; // xmm1
  __m128i v116; // xmm2
  __m128i v117; // xmm3
  __int64 v118; // r14
  __int64 v119; // rax
  __int64 v120; // rdi
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rdi
  __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // rsi
  __int64 v128; // r10
  char v129; // r8
  __int64 v130; // r13
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rdi
  __int64 v134; // rax
  __int64 *v135; // r14
  __int64 *v136; // rsi
  __int64 **v137; // rdi
  __int64 *v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rax
  __int64 v141; // r8
  __int64 v142; // rdi
  __int64 v143; // rax
  void *v144; // [rsp+0h] [rbp-320h]
  __int64 v145; // [rsp+8h] [rbp-318h]
  __int64 v146; // [rsp+10h] [rbp-310h]
  __int64 v147; // [rsp+18h] [rbp-308h]
  _DWORD *v148; // [rsp+20h] [rbp-300h]
  __int64 v149; // [rsp+30h] [rbp-2F0h]
  __int64 v150; // [rsp+38h] [rbp-2E8h]
  __int64 v151; // [rsp+38h] [rbp-2E8h]
  unsigned int v152; // [rsp+40h] [rbp-2E0h]
  unsigned int v153; // [rsp+44h] [rbp-2DCh]
  __int64 v154; // [rsp+48h] [rbp-2D8h]
  __int64 v155; // [rsp+48h] [rbp-2D8h]
  __int64 v156; // [rsp+50h] [rbp-2D0h]
  __int64 v157; // [rsp+50h] [rbp-2D0h]
  __int64 v158; // [rsp+50h] [rbp-2D0h]
  unsigned int v159; // [rsp+58h] [rbp-2C8h]
  __int64 v160; // [rsp+60h] [rbp-2C0h]
  int v161; // [rsp+68h] [rbp-2B8h]
  _BOOL4 v162; // [rsp+6Ch] [rbp-2B4h]
  __int64 v164; // [rsp+78h] [rbp-2A8h]
  unsigned int v165; // [rsp+80h] [rbp-2A0h]
  int v167; // [rsp+88h] [rbp-298h]
  __int64 v168; // [rsp+88h] [rbp-298h]
  __int64 v169; // [rsp+88h] [rbp-298h]
  unsigned int v170; // [rsp+90h] [rbp-290h]
  __int64 v172; // [rsp+98h] [rbp-288h]
  __int16 v173; // [rsp+A2h] [rbp-27Eh] BYREF
  unsigned int v174; // [rsp+A4h] [rbp-27Ch] BYREF
  __int64 v175; // [rsp+A8h] [rbp-278h] BYREF
  __int64 v176; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v177; // [rsp+B8h] [rbp-268h] BYREF
  _DWORD v178[4]; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v179; // [rsp+D0h] [rbp-250h] BYREF
  unsigned int v180; // [rsp+D8h] [rbp-248h]
  unsigned __int8 v181; // [rsp+DCh] [rbp-244h]
  __int64 v182; // [rsp+E0h] [rbp-240h]
  __int64 v183; // [rsp+E8h] [rbp-238h]
  __int64 v184; // [rsp+F0h] [rbp-230h]
  __int64 v185; // [rsp+F8h] [rbp-228h]
  __int64 v186; // [rsp+100h] [rbp-220h]
  _OWORD v187[33]; // [rsp+110h] [rbp-210h] BYREF

  v10 = (__int64)a1;
  v12 = a1[21];
  v13 = *(_QWORD *)(*a1 + 96);
  v172 = *a1;
  if ( a2 )
  {
    a1 = (__int64 *)a2;
    sub_643D30(a2);
  }
  v14 = qword_4F04C68[0];
  if ( dword_4F04C58 == -1 )
    v15 = 776LL * unk_4F04C24;
  else
    v15 = 776LL * dword_4F04C58;
  *(_BYTE *)(v13 + 182) |= 2u;
  v16 = v15 + v14;
  v17 = (_QWORD *)sub_8784C0();
  v17[1] = v10;
  *v17 = *(_QWORD *)(v16 + 744);
  *(_QWORD *)(v16 + 744) = v17;
  *(_BYTE *)(v13 + 182) |= 1u;
  v179 = v10;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v180 = v180 & 0xF8000000 | (a5 << 10) & 0x400 | 1;
  v181 = 0;
  if ( dword_4F04C58 == -1 )
    v18 = 776LL * unk_4F04C24;
  else
    v18 = 776LL * dword_4F04C58;
  ++*(_QWORD *)(qword_4F04C68[0] + v18 + 720);
  v19 = *(_QWORD *)(v13 + 80);
  *(_BYTE *)(v13 + 177) |= 0x60u;
  v164 = v19;
  v20 = sub_7A7D00();
  v21 = v20;
  if ( v20 )
  {
    if ( *(_DWORD *)(v10 + 184) )
    {
      v42 = sub_727670();
      v168 = sub_7276D0();
      *(_QWORD *)&v187[0] = sub_724DC0(a1, a2, v43, v44, v45, v46);
      sub_72BBE0(*(_QWORD *)&v187[0], v21, 8);
      *(_BYTE *)(v42 + 11) |= 1u;
      *(_BYTE *)(v42 + 8) = 85;
      *(_QWORD *)(v42 + 32) = v168;
      *(_BYTE *)(v168 + 10) = 3;
      sub_7296C0(v178);
      *(_QWORD *)(v168 + 40) = sub_73A460(*(_QWORD *)&v187[0]);
      sub_729730(v178[0]);
      v47 = (_QWORD **)(v10 + 104);
      if ( *(_QWORD *)(v10 + 104) )
        v47 = sub_5CB9F0(v47);
      *v47 = (_QWORD *)v42;
      sub_724E30(v187);
    }
    else
    {
      *(_DWORD *)(v10 + 184) = v20;
    }
  }
  if ( (*(_BYTE *)(v172 + 81) & 0x10) != 0 )
  {
    v22 = a6 == 0 && a8 != 0;
    v162 = v22;
  }
  else
  {
    v162 = 0;
    v22 = 0;
  }
  *(_BYTE *)(v10 + 178) = (4 * v22) | *(_BYTE *)(v10 + 178) & 0xFB;
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] != 73 )
    {
      v170 = 1;
      v167 = 0;
      v23 = qword_4F04C68[0];
      goto LABEL_15;
    }
    v170 = 1;
    v153 = -1;
    v165 = 0;
    v161 = 0;
    v167 = 0;
    goto LABEL_83;
  }
  v26 = qword_4F04C68[0];
  v27 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v27 + 4) == 6
    && (*(_BYTE *)(v10 + 89) & 4) != 0
    && *(_QWORD *)(v27 + 208) == *(_QWORD *)(*(_QWORD *)(v10 + 40) + 32LL) )
  {
    *(_BYTE *)(v12 + 110) |= 0x80u;
  }
  v28 = *(_BYTE *)(v10 + 177);
  v29 = *(_BYTE *)(v27 + 6);
  if ( v28 < 0 || (v29 & 0xA) != 0 )
  {
    if ( (v29 & 2) != 0 )
    {
      LOBYTE(v180) = v180 | 0x80;
      *(_BYTE *)(v10 + 177) |= 0x20u;
    }
    else if ( (v29 & 8) != 0 )
    {
      BYTE1(v180) |= 1u;
    }
    v30 = *(_BYTE *)(v172 + 81) & 0x10;
    if ( !v30 || !v164 )
      goto LABEL_72;
    v31 = *(_QWORD *)(v172 + 64);
    if ( (word_4F06418[0] == 73 || word_4F06418[0] == 55)
      && *(_QWORD *)(v172 + 88) != *(_QWORD *)(v26 + 776LL * unk_4F04C48 + 208) )
    {
      while ( *(_BYTE *)(v31 + 140) == 12 )
        v31 = *(_QWORD *)(v31 + 160);
      v122 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v31 + 96LL) + 80LL);
      *(_QWORD *)(v164 + 176) = v172;
      v169 = v122;
      sub_879080(v164, 0, *(_QWORD *)(v122 + 32));
      *(_BYTE *)(v164 + 265) = *(_BYTE *)(v169 + 265) & 0x1C | *(_BYTE *)(v164 + 265) & 0xE3;
      v123 = sub_878940(v172);
      *(_QWORD *)(v164 + 80) = v123;
      *(_DWORD *)(v123 + 24) = dword_4F06650[0];
      if ( (unsigned int)sub_879510(v172) )
        *(_BYTE *)(v164 + 265) |= 0x20u;
      v30 = *(_BYTE *)(v172 + 81) & 0x10;
      goto LABEL_72;
    }
  }
  else
  {
    v30 = *(_BYTE *)(v172 + 81) & 0x10;
    if ( (v28 & 0x20) != 0 )
    {
      LOBYTE(v180) = v180 | 0x80;
      if ( !v30 )
        goto LABEL_40;
      v31 = *(_QWORD *)(v172 + 64);
      if ( (*(_BYTE *)(v31 + 177) & 0x40) == 0 )
        goto LABEL_37;
      v32 = v28 | 0x40;
      goto LABEL_327;
    }
    if ( !a7 )
    {
LABEL_72:
      if ( !v30 )
        goto LABEL_40;
      v31 = *(_QWORD *)(v172 + 64);
      goto LABEL_37;
    }
    if ( !v30 )
      goto LABEL_40;
    v31 = *(_QWORD *)(v172 + 64);
    if ( (*(_BYTE *)(v31 + 177) & 0x20) != 0 )
    {
      LOBYTE(v180) = v180 | 0x80;
      v32 = v28 | 0x20;
LABEL_327:
      *(_BYTE *)(v10 + 177) = v32;
      v30 = *(_BYTE *)(v172 + 81) & 0x10;
      goto LABEL_72;
    }
  }
LABEL_37:
  if ( (*(_BYTE *)(v31 + 178) & 4) != 0 && (*(char *)(v10 + 177) >= 0 || !*(_QWORD *)(v12 + 168)) )
    *(_BYTE *)(v10 + 178) |= 4u;
LABEL_40:
  LOBYTE(v180) = ((a7 & 1) << 6) | v180 & 0xBF;
  v33 = sub_8788F0(v172);
  v184 = v33;
  if ( v33 )
    *(_DWORD *)(v10 + 184) = *(_DWORD *)(*(_QWORD *)(v33 + 88) + 184LL);
  v161 = 0;
  v36 = a7;
  if ( a7 && *(char *)(v10 + 177) >= 0 )
  {
    v36 = (__int64)v178;
    sub_7A74B0(*(unsigned int *)(v10 + 184), v178);
    v161 = 1;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 > 201102 || (v34 = (__int64)&dword_4F07774, (v35 = dword_4F07774) != 0) )
    {
      v34 = qword_4F04C68[0];
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 8) != 0 )
      {
        v165 = 0;
        v153 = dword_4F04C40;
        goto LABEL_50;
      }
LABEL_369:
      v153 = dword_4F04C40;
      *(_BYTE *)(v34 + 776LL * (int)dword_4F04C40 + 7) |= 8u;
      v165 = 1;
      goto LABEL_50;
    }
  }
  v35 = (__int64)&dword_4F04C40;
  v36 = dword_4F04C40;
  v165 = dword_4F077BC;
  v153 = dword_4F04C40;
  if ( dword_4F077BC )
  {
    v34 = (__int64)&qword_4F077A8;
    if ( qword_4F077A8 > 0x76BFu )
    {
      v34 = a7;
      if ( a7 )
      {
        v34 = qword_4F04C68[0];
        v35 = 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(qword_4F04C68[0] + v35 + 7) & 8) != 0 )
        {
          v165 = 0;
        }
        else
        {
          if ( dword_4F077C4 == 2 )
            goto LABEL_369;
          v165 = 1;
        }
        if ( !unk_4D04238 || v22 )
        {
LABEL_74:
          v167 = 0;
          goto LABEL_55;
        }
        v35 = v172;
        v167 = 0;
        if ( (unsigned __int8)(*(_BYTE *)(v172 + 80) - 4) > 1u )
          goto LABEL_55;
        v107 = *(_QWORD *)(v172 + 88);
        if ( !v107 )
          goto LABEL_55;
        goto LABEL_316;
      }
    }
    v165 = 0;
  }
LABEL_50:
  if ( !unk_4D04238
    || v22
    || (v36 = v172, (unsigned __int8)(*(_BYTE *)(v172 + 80) - 4) > 1u)
    || (v107 = *(_QWORD *)(v172 + 88)) == 0 )
  {
LABEL_52:
    if ( a6 && !a7 )
    {
      v36 = 0;
      sub_865D70(*(_QWORD *)(v172 + 64), 0, a8, 1, 0, 0);
      v167 = 0;
      goto LABEL_55;
    }
    goto LABEL_74;
  }
LABEL_316:
  if ( (*(_DWORD *)(v107 + 176) & 0x12000) != 0x10000 )
    goto LABEL_52;
  v36 = 1;
  sub_865B60(v10, 1);
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) |= 0x80u;
  v167 = 1;
LABEL_55:
  if ( word_4F06418[0] != 55
    || (++*(_BYTE *)(qword_4F061C8 + 81LL),
        sub_5EBF70((__int64)&v179, v36, v34, v35),
        --*(_BYTE *)(qword_4F061C8 + 81LL),
        word_4F06418[0] == 73) )
  {
    v170 = 1;
  }
  else
  {
    if ( *(_BYTE *)(v10 + 140) != 11 )
      sub_6851D0(243);
    *(_QWORD *)v12 = 0;
    *(_QWORD *)(v12 + 8) = 0;
    *(_QWORD *)(v12 + 24) = 0;
    *(_QWORD *)(v12 + 80) = 0;
    *(_WORD *)(v10 + 176) &= 0xFEEFu;
    *(_QWORD *)(v12 + 152) = sub_8600D0(6, 0xFFFFFFFFLL, v10, 0);
    v108 = *(_QWORD *)v10;
    *(_BYTE *)(v10 + 141) &= ~0x20u;
    *(_BYTE *)(v108 + 81) |= 2u;
    sub_601910((_BYTE *)v10, a3, (__int64)&v179, v109, v110, v111);
    sub_863FC0();
    if ( a7 == 0 && a6 != 0 && !v167 )
    {
      LOBYTE(v112) = a6 != 0;
      sub_866010(v10, a3, v112, v113, v114);
      v170 = 0;
    }
    else
    {
      v170 = 0;
    }
  }
  if ( !*(_QWORD *)v12 )
  {
    *(_BYTE *)(v13 + 184) |= 1u;
    goto LABEL_77;
  }
  v37 = *(_QWORD *)(v179 + 168);
  v156 = v179;
  sub_5E6440(v179, 0, (__int64 *)(v37 + 16));
  v38 = v156;
  if ( !v37 )
    goto LABEL_60;
  v39 = *(_QWORD *)(v37 + 80);
  if ( v39 )
    goto LABEL_60;
  v127 = *(_QWORD *)(*(_QWORD *)(v156 + 168) + 16LL);
  if ( !v127 )
    goto LABEL_60;
  v128 = v156;
  while ( (*(_BYTE *)(v127 + 96) & 2) == 0 || !(unsigned int)sub_5E4940(*(_QWORD *)(v127 + 40)) )
  {
LABEL_351:
    v127 = *(_QWORD *)(v127 + 16);
    if ( !v127 )
    {
      v38 = v128;
      goto LABEL_364;
    }
  }
  if ( (v129 & 8) != 0 )
  {
    if ( !v39 )
      v39 = v127;
    goto LABEL_351;
  }
  v38 = v128;
  v139 = *(_QWORD *)(*(_QWORD *)(v127 + 56) + 168LL);
  *(_QWORD *)(v139 + 24) = v127;
  v140 = *(_QWORD *)(v127 + 40);
  v141 = *(_QWORD *)(v140 + 168);
  v142 = *(_QWORD *)(v141 + 80);
  if ( v142 )
  {
    v149 = *(_QWORD *)(v140 + 168);
    v151 = v139;
    v155 = v39;
    v158 = v128;
    v143 = sub_8E5650(v142);
    v139 = v151;
    v141 = v149;
    v39 = v155;
    v38 = v158;
    *(_QWORD *)(v151 + 80) = v143;
  }
  else
  {
    *(_QWORD *)(v139 + 80) = v127;
  }
  *(_WORD *)(v139 + 44) = *(_WORD *)(v141 + 44);
  v127 = *(_QWORD *)(v37 + 80);
LABEL_364:
  if ( v39 && !v127 )
  {
    v130 = *(_QWORD *)(*(_QWORD *)(v39 + 56) + 168LL);
    *(_QWORD *)(v130 + 24) = v39;
    v131 = *(_QWORD *)(v39 + 40);
    v132 = *(_QWORD *)(v131 + 168);
    v133 = *(_QWORD *)(v132 + 80);
    if ( v133 )
    {
      v154 = *(_QWORD *)(v131 + 168);
      v157 = v38;
      v134 = sub_8E5650(v133);
      v132 = v154;
      v38 = v157;
      *(_QWORD *)(v130 + 80) = v134;
    }
    else
    {
      *(_QWORD *)(v130 + 80) = v39;
    }
    *(_WORD *)(v130 + 44) = *(_WORD *)(v132 + 44);
  }
LABEL_60:
  if ( (*(_BYTE *)(v38 + 176) & 0x10) != 0 && **(_QWORD **)(v38 + 168) )
  {
    v40 = **(_QWORD **)(v38 + 168);
    v41 = v38;
    do
    {
      if ( (*(_BYTE *)(v40 + 96) & 2) != 0 )
        sub_5E9440(v40);
      v40 = *(_QWORD *)v40;
    }
    while ( v40 );
    v38 = v41;
  }
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v38 + 96LL) + 184LL) |= 1u;
LABEL_77:
  if ( word_4F06418[0] != 73 )
  {
    v23 = qword_4F04C68[0];
    if ( v165 )
    {
      if ( dword_4F077C4 == 2 )
      {
        v48 = 776LL * (int)dword_4F04C40;
        *(_BYTE *)(qword_4F04C68[0] + v48 + 7) &= ~8u;
        v23 = qword_4F04C68[0];
        if ( *(_QWORD *)(qword_4F04C68[0] + v48 + 456) )
        {
          ((void (*)(void))sub_8845B0)();
          v23 = qword_4F04C68[0];
        }
      }
    }
    goto LABEL_15;
  }
LABEL_83:
  v176 = *(_QWORD *)&dword_4F063F8;
  v150 = sub_8600D0(6, 0xFFFFFFFFLL, v10, 0);
  v49 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_QWORD *)(v49 + 600) = &v179;
  LODWORD(v146) = sub_5E4740(6, 0xFFFFFFFFLL, v50, v49);
  v148 = &dword_4F04C44;
  if ( (_DWORD)v146 && (dword_4F04C44 != -1 || (v51[6] & 6) != 0 || v51[4] == 12) )
    *(_BYTE *)(v10 + 177) |= 0x20u;
  if ( dword_4F077C4 == 2 )
    v51[704] = *(_BYTE *)(v12 + 109) & 7;
  sub_7B80F0();
  *(_QWORD *)(v12 + 152) = v150;
  sub_7B8B50(6, 0xFFFFFFFFLL, v52, v53);
  v147 = 0;
  ++*(_BYTE *)(qword_4F061C8 + 82LL);
  if ( dword_4F077C4 == 2 )
  {
    v119 = qword_4CF8008;
    qword_4CF8008 = 0;
    v147 = v119;
    if ( unk_4D047DC )
      sub_886390(v172);
  }
  v54 = word_4F06418[0];
  if ( word_4F06418[0] == 74 )
  {
    if ( dword_4F077C0 )
    {
      *(_BYTE *)(v10 + 179) |= 1u;
    }
    else if ( dword_4F077C4 != 2 )
    {
      sub_6851C0(169, dword_4F07508);
      v160 = sub_725D60();
      v115 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v116 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v117 = _mm_loadu_si128(&xmmword_4F06660[3]);
      *(_QWORD *)&v187[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v187[1] = v115;
      v187[2] = v116;
      v187[3] = v117;
      BYTE1(v187[1]) = v115.m128i_i8[1] | 0x20;
      *((_QWORD *)&v187[0] + 1) = *(_QWORD *)dword_4F07508;
      *(_QWORD *)(v160 + 120) = sub_72C930();
      v149 = v160;
      v118 = sub_647630(8, v187, (unsigned int)dword_4F04C64, 1);
      *(_QWORD *)(v118 + 88) = v160;
      sub_877D80(v160, v118);
      sub_877E20(v118, v160, v10);
      if ( v183 )
        *(_QWORD *)(v183 + 112) = v160;
      else
        *(_QWORD *)(v10 + 160) = v160;
      v183 = v160;
    }
    goto LABEL_157;
  }
  HIDWORD(v146) = 0;
  v152 = unk_4F066AC;
  v181 = 2 * (*(_BYTE *)(v10 + 140) == 9);
  *(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C5C + 5) = v181
                                                         | *(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C5C + 5) & 0xFC;
  v55 = dword_4D047AC;
  if ( dword_4D047AC )
  {
    if ( a9 || !a7 )
    {
      HIDWORD(v146) = 0;
    }
    else
    {
      v56 = *(_QWORD *)(v13 + 104);
      switch ( *(_BYTE *)(v56 + 80) )
      {
        case 4:
        case 5:
          v124 = *(_QWORD *)(*(_QWORD *)(v56 + 96) + 80LL);
          break;
        case 6:
          v124 = *(_QWORD *)(*(_QWORD *)(v56 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v124 = *(_QWORD *)(*(_QWORD *)(v56 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v124 = *(_QWORD *)(v56 + 88);
          break;
        default:
          v124 = 0;
          break;
      }
      v55 = (unsigned int)(unk_4F066AC - *(_DWORD *)(*(_QWORD *)(sub_892400(v124) + 32) + 48LL));
      v54 = word_4F06418[0];
      HIDWORD(v146) = v55;
    }
  }
  v57 = (__int64)dword_4F06650;
  v58 = (unsigned __int64)&dword_4F077BC;
  v59 = (__int64)v187;
  v145 = v13;
  v149 = v10;
  while ( 2 )
  {
    v60 = 0;
    if ( v54 == 187 )
    {
      sub_7B8B50(v59, v57, v55, v58);
      v60 = 1;
      v54 = word_4F06418[0];
    }
    v61 = qword_4F061C8;
    v62 = 0;
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    if ( v54 == 170 )
    {
      v62 = 1;
      v177 = *(_QWORD *)&dword_4F063F8;
      sub_7B8B50(v59, v57, v61, v58);
    }
    v59 = 0;
    sub_854590(0);
    if ( dword_4F077C4 != 2 )
    {
      v65 = word_4F06418[0];
LABEL_103:
      if ( v65 == 170 )
        goto LABEL_220;
      if ( !v62 )
        goto LABEL_106;
LABEL_105:
      v57 = (__int64)&v177;
      v59 = 3104;
      sub_6851C0(3104, &v177);
      v65 = word_4F06418[0];
      goto LABEL_106;
    }
    v64 = 0;
    while ( 2 )
    {
      v65 = word_4F06418[0];
      v63 = word_4F06418[0] & 0xFFFD;
      if ( (word_4F06418[0] & 0xFFFD) == 0x9D )
      {
        if ( word_4F06418[0] == 159 )
        {
          v181 = 0;
          v91 = 0;
          goto LABEL_199;
        }
        if ( word_4F06418[0] != 158 )
        {
          v181 = 2;
          v91 = 2;
          goto LABEL_199;
        }
LABEL_198:
        v181 = 1;
        v91 = 1;
LABEL_199:
        v57 = (__int64)qword_4F04C68;
        v92 = qword_4F04C68[0] + 776LL * unk_4F04C5C;
        v93 = (unsigned int)v91 | *(_BYTE *)(v92 + 5) & 0xFC;
        *(_BYTE *)(v92 + 5) = v91 | *(_BYTE *)(v92 + 5) & 0xFC;
        sub_7B8B50(v59, qword_4F04C68, v93, v91);
        if ( word_4F06418[0] == 55 )
        {
          sub_7B8B50(v59, qword_4F04C68, v94, v95);
        }
        else
        {
          if ( word_4F06418[0] == 1 || (unsigned int)sub_651AF0(0) )
            goto LABEL_212;
          if ( dword_4F077C4 != 2 )
            goto LABEL_203;
          v96 = (0xE00000000000009uLL >> (LOBYTE(word_4F06418[0]) - 100)) & 1;
          if ( (unsigned __int16)(word_4F06418[0] - 100) >= 0x3Cu )
            LOBYTE(v96) = 0;
          if ( word_4F06418[0] == 37 || (_BYTE)v96 )
          {
LABEL_212:
            v57 = (__int64)dword_4F07508;
            sub_6851C0(53, dword_4F07508);
          }
          else
          {
LABEL_203:
            sub_6851D0(53);
          }
        }
        v59 = 1;
        sub_854590(1);
        v64 = 1;
        continue;
      }
      break;
    }
    if ( word_4F06418[0] == 158 )
      goto LABEL_198;
    if ( !(_DWORD)v64 )
      goto LABEL_103;
    if ( !v62 )
    {
      if ( word_4F06418[0] == 74 )
        goto LABEL_389;
      if ( word_4F06418[0] == 170 )
        goto LABEL_220;
LABEL_106:
      if ( v65 == 75 )
      {
        if ( dword_4F077C4 != 2 )
        {
          if ( (v180 & 1) == 0 )
            goto LABEL_192;
          v57 = 0;
          v59 = 0;
          if ( (unsigned __int16)sub_7BE840(0, 0) == 74 )
          {
            v65 = word_4F06418[0];
            if ( word_4F06418[0] != 184 )
              goto LABEL_108;
LABEL_133:
            v59 = 0;
            sub_64FFE0(0);
            goto LABEL_150;
          }
          if ( dword_4F077C4 != 2 )
            goto LABEL_192;
        }
        if ( unk_4F07778 <= 201401 )
        {
LABEL_192:
          v59 = 5;
          if ( dword_4D04964 )
            v59 = unk_4F07471;
          v57 = 381;
          sub_684AA0(v59, 381, &dword_4F063F8);
        }
        sub_854AB0();
        sub_7B8B50(v59, v57, v89, v90);
        goto LABEL_150;
      }
      if ( v65 == 184 )
        goto LABEL_133;
LABEL_108:
      if ( v65 == 149 || v65 == 137 )
      {
        v57 = 0;
        v59 = 0;
        sub_64F620(0, 0, v187);
        goto LABEL_150;
      }
      if ( dword_4F077C4 != 2 )
        goto LABEL_138;
      if ( v65 == 191 && (unsigned int)sub_645420() )
      {
        sub_854AB0();
        v57 = 65;
        v59 = 75;
        sub_7BE280(75, 65, 0, 0);
        goto LABEL_150;
      }
      if ( unk_4D0418C )
      {
        v98 = sub_5CC190(1);
        if ( v98 )
          sub_5CC9F0(v98);
      }
      if ( word_4F06418[0] == 179 )
      {
        if ( (unsigned __int16)sub_7BE840(0, 0) == 87 )
        {
          sub_7B8B50(0, 0, v125, v126);
          v57 = v181;
          v59 = v10;
          sub_65B760(v10, v181);
          sub_854AB0();
          goto LABEL_150;
        }
        goto LABEL_228;
      }
      v66 = (void *)qword_4D04168;
      qword_4D04168 = 0;
      if ( dword_4F077C4 != 2 )
      {
        v67 = word_4F06418[0] == 1;
        goto LABEL_116;
      }
      if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
      {
        qword_4D04168 = v66;
LABEL_223:
        v97 = 0;
        if ( (unk_4D04A12 & 2) != 0 )
          v97 = xmmword_4D04A20.m128i_i64[0];
        v144 = &unk_4D04A00;
        if ( (unsigned int)sub_8D0520(v97, v10) )
          goto LABEL_117;
        v64 = (__int64)&unk_4D04A00;
        if ( (unk_4D04A10 & 1) == 0 || (unsigned __int16)sub_7BE840(0, 0) != 75 )
          goto LABEL_117;
LABEL_228:
        v57 = v60;
        v59 = (__int64)&v179;
        v73 = sub_5EEEC0((__int64)&v179, v60);
        goto LABEL_144;
      }
      v144 = v66;
      v67 = sub_7C0F00(0, 0);
      v66 = v144;
LABEL_116:
      qword_4D04168 = v66;
      if ( v67 )
        goto LABEL_223;
LABEL_117:
      if ( !a9 && a7 && dword_4D04490 )
      {
        for ( i = dword_4F06650[0]; ; i = v64 + 1 )
        {
          v64 = (_DWORD)qword_4CF7FF0[1] & i;
          v69 = *qword_4CF7FF0 + 16LL * (unsigned int)v64;
          if ( dword_4F06650[0] == *(_DWORD *)v69 )
            break;
          if ( !*(_DWORD *)v69 )
            goto LABEL_134;
        }
        v70 = *(_QWORD *)(v69 + 8);
        if ( v70 )
        {
          memset(v187, 0, 0x1D8u);
          *((_QWORD *)&v187[9] + 1) = v187;
          *((_QWORD *)&v187[1] + 1) = *(_QWORD *)&dword_4F063F8;
          if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
            BYTE2(v187[11]) |= 1u;
          v59 = (__int64)v187;
          *(_QWORD *)&v187[23] = v70;
          v57 = (__int64)&v173;
          BYTE5(v187[8]) |= 0x10u;
          LODWORD(v187[4]) = dword_4F06650[0];
          sub_8BE2A0(v187, &v173, dword_4F06650[0], 0, dword_4F077BC, dword_4D04490);
          goto LABEL_150;
        }
      }
LABEL_134:
      v71 = word_4F06418[0];
      if ( word_4F06418[0] == 160 || word_4F06418[0] == 169 )
      {
LABEL_233:
        v173 = 75;
        v57 = 0;
        *(_QWORD *)&v187[0] = *(_QWORD *)&dword_4F063F8;
        if ( v71 == 88 )
        {
          sub_7B8B50(&dword_4F063F8, 0, *(_QWORD *)&dword_4F063F8, v64);
          v57 = 1;
        }
        v59 = (__int64)&v173;
        v73 = sub_8BF8E0(&v173, v57, v187);
        if ( ((word_4F06418[0] - 9) & 0xFFF7) == 0 )
          sub_7B8B50(&v173, v57, v55, v58);
        v99 = v173;
        if ( v173 == 75 )
        {
          v57 = 65;
          v59 = 75;
          sub_7BE5B0(75, 65, 0, 0);
          v99 = v173;
        }
        if ( word_4F06418[0] == v99 )
          sub_7B8B50(v59, v57, v55, v58);
      }
      else
      {
        if ( unk_4D04218 && word_4F06418[0] == 88 && (unsigned __int16)sub_7BE840(0, 0) == 160 )
        {
          v71 = word_4F06418[0];
          goto LABEL_233;
        }
LABEL_138:
        v57 = 0;
        v59 = (__int64)&v179;
        v72 = sub_6040F0((__int64)&v179, 0, 0, v60, 0, &v174, &v175, 0, 0, 0);
        v58 = v174;
        v73 = v72;
        if ( !v174 )
        {
          if ( word_4F06418[0] == 74 )
          {
            if ( dword_4F077C4 != 1 )
            {
              if ( dword_4D04964 )
                v59 = unk_4F07471;
              else
                v59 = 2 * (unsigned int)(dword_4F077C4 == 2) + 5;
              v57 = 65;
              sub_684AC0(v59, 65);
            }
          }
          else
          {
            v57 = 65;
            v59 = 75;
            sub_7BE280(75, 65, 0, 0);
          }
        }
      }
LABEL_144:
      if ( dword_4D047AC )
      {
        if ( v73 )
        {
          if ( (*(_BYTE *)(v73 + 81) & 0x10) == 0 )
          {
            v74 = *(_DWORD *)(v73 + 44);
            if ( v74 > v152 )
            {
              v55 = *(unsigned __int8 *)(v73 + 80);
              if ( (_BYTE)v55 == 20 || (_BYTE)v55 == 11 )
                *(_DWORD *)(v73 + 44) = v74 - HIDWORD(v146);
            }
          }
        }
      }
LABEL_150:
      if ( qword_4CF8008 )
        sub_5E9610(v59, v57, v55);
      --*(_BYTE *)(qword_4F061C8 + 83LL);
      v54 = word_4F06418[0];
      if ( word_4F06418[0] == 74 || word_4F06418[0] == 9 )
      {
        v13 = v145;
        goto LABEL_155;
      }
      continue;
    }
    break;
  }
  v59 = 3103;
  v57 = (__int64)&v177;
  sub_6851C0(3103, &v177);
  v65 = word_4F06418[0];
  if ( word_4F06418[0] != 74 )
  {
    if ( word_4F06418[0] != 170 )
      goto LABEL_106;
LABEL_220:
    v177 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(v59, v57, v63, v64);
    goto LABEL_105;
  }
LABEL_389:
  v13 = v145;
  sub_854AB0();
  --*(_BYTE *)(qword_4F061C8 + 83LL);
LABEL_155:
  if ( dword_4F077C4 != 2 && (v180 & 8) == 0 )
  {
    v120 = 5;
    if ( dword_4D04964 )
      v120 = unk_4F07471;
    sub_684AC0(v120, 618);
  }
LABEL_157:
  if ( a6 )
  {
    v75 = -1;
    if ( !a7 )
      v75 = a3;
    a3 = v75;
  }
  sub_854430();
  v159 = dword_4F06650[0];
  *(_QWORD *)&v187[0] = *(_QWORD *)&dword_4F063F8;
  if ( a10 )
    *(_QWORD *)(a10 + 40) = *(_QWORD *)&dword_4F063F8;
  sub_7BE280(74, 67, 3196, &v176);
  if ( dword_4F077B8 )
  {
    if ( word_4F06418[0] == 142 )
    {
      v135 = (__int64 *)sub_5CC970(3);
      if ( v135 )
      {
        v159 = unk_4D04180;
        *(_QWORD *)&v187[0] = unk_4D04178;
        if ( a10 )
          *(_QWORD *)(a10 + 40) = unk_4F061D8;
        sub_5CF700(v135);
        if ( a2 )
        {
          v136 = 0;
          v137 = 0;
          do
          {
            v138 = v135;
            v135 = (__int64 *)*v135;
            if ( (unsigned __int8)(*((_BYTE *)v138 + 8) - 86) <= 0x1Cu
              && ((1LL << (*((_BYTE *)v138 + 8) - 86)) & 0x107BFFFF) != 0 )
            {
              *v138 = *(_QWORD *)(a2 + 184);
              *(_QWORD *)(a2 + 184) = v138;
              if ( v137 )
                *v137 = v135;
              *((_WORD *)v138 + 5) = *((_WORD *)v138 + 5) & 0xFE00 | 1;
            }
            else
            {
              v137 = (__int64 **)v138;
              if ( !v136 )
                v136 = v138;
            }
          }
          while ( v135 );
          if ( v136 )
          {
            v135 = v136;
            goto LABEL_387;
          }
        }
        else
        {
LABEL_387:
          sub_5CEC90(v135, v10, 6);
        }
      }
    }
  }
  sub_6030F0((unsigned int *)v187);
  if ( a3 == -1 || *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)a3 + 4) != 8 )
    sub_601910((_BYTE *)v10, a3, (__int64)&v179, v76, v77, v78);
  if ( v161 )
    sub_7A7500(v178);
  if ( a9 )
  {
    v79 = a9;
    v80 = 59;
    sub_869D70(a9, 59);
  }
  else
  {
    v80 = 6;
    v79 = v10;
    sub_869D70(v10, 6);
  }
  if ( dword_4F077BC && qword_4F077A8 > 0x76BFu && a7 )
  {
    if ( v153 != -1 )
    {
      v79 = v153;
      sub_87DD20(v153, v80, v81, v82, v83, v84, v144, v145, v146, v147, &dword_4F04C44, &dword_4F077BC, v149);
    }
  }
  else if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) && v153 != -1 )
  {
    v79 = v153;
    sub_8845B0(v153);
  }
  sub_863FC0();
  if ( a6 )
  {
    if ( (!v164 || (v87 = *(_QWORD *)(*(_QWORD *)(v164 + 32) + 16LL)) == 0 || *(_BYTE *)(v87 + 28) != 6)
      && ((v180 & 0x80u) != 0
       || (v88 = *(_QWORD *)(v13 + 104)) == 0
       || (*(_BYTE *)(*(_QWORD *)(v88 + 88) + 177LL) & 2) != 0) )
    {
      *(_BYTE *)(v10 + 177) |= 2u;
    }
    if ( !(a7 | v167) )
      sub_866010(v79, v80, v85, v86, a6);
  }
  --*(_BYTE *)(qword_4F061C8 + 82LL);
  sub_7B8160();
  if ( !v165 )
  {
LABEL_243:
    if ( *v148 == -1 )
      goto LABEL_348;
    goto LABEL_244;
  }
  if ( dword_4F077C4 == 2 )
  {
    v100 = (int)dword_4F04C40;
    v101 = 776LL * (int)dword_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + v101 + 7) &= ~8u;
    if ( !*(_QWORD *)(qword_4F04C68[0] + v101 + 456) )
    {
      if ( *v148 == -1 )
        goto LABEL_348;
      goto LABEL_245;
    }
    sub_8845B0(v100);
    goto LABEL_243;
  }
  if ( *v148 != -1 )
    goto LABEL_189;
LABEL_348:
  sub_880400(v172);
LABEL_244:
  if ( dword_4F077C4 == 2 )
  {
LABEL_245:
    sub_880640(v10);
    v102 = dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      if ( !(_DWORD)v146
        && (!a2 || *(char *)(a2 + 122) >= 0)
        && ((*(_BYTE *)(v172 + 81) & 0x10) == 0 || v162 | a6)
        && (*(_BYTE *)(v10 + 177) & 0x40) == 0 )
      {
        if ( a7 || dword_4F04C58 == -1 )
          v103 = 776LL * unk_4F04C24;
        else
          v103 = 776LL * dword_4F04C58;
        v104 = qword_4F04C68[0] + v103;
        v105 = qword_4CF7FF8;
        if ( qword_4CF7FF8 )
          qword_4CF7FF8 = *(_QWORD *)qword_4CF7FF8;
        else
          v105 = sub_823970(24);
        *(_QWORD *)v105 = 0;
        *(_QWORD *)(v105 + 8) = v10;
        *(_DWORD *)(v105 + 16) = a7;
        if ( *(_QWORD *)(v104 + 728) )
          **(_QWORD **)(v104 + 736) = v105;
        else
          *(_QWORD *)(v104 + 728) = v105;
        *(_QWORD *)(v104 + 736) = v105;
        v102 = dword_4F077C4;
      }
      qword_4CF8008 = v147;
      if ( (*(_DWORD *)(v10 + 176) & 0x18000) == 0x8000 )
      {
        *(_BYTE *)(v164 + 265) |= 2u;
        *(_QWORD *)(v164 + 176) = v172;
        if ( (*(_BYTE *)(v172 + 81) & 0x10) != 0 )
        {
          v121 = *(_QWORD *)(v164 + 80);
          if ( v121 )
            *(_DWORD *)(v121 + 28) = v159;
        }
        if ( word_4F06418[0] != 75 )
          *(_BYTE *)(v164 + 265) |= 0x20u;
      }
      if ( v102 == 2 )
      {
        for ( j = *(_QWORD *)(v150 + 144); j; j = *(_QWORD *)(j + 112) )
        {
          if ( *(char *)(j + 192) < 0 )
            *(_BYTE *)(j + 205) |= 1u;
        }
      }
    }
  }
LABEL_189:
  v23 = qword_4F04C68[0];
LABEL_15:
  if ( dword_4F04C58 == -1 )
    v24 = 776LL * unk_4F04C24;
  else
    v24 = 776LL * dword_4F04C58;
  --*(_QWORD *)(v23 + v24 + 720);
  if ( v167 )
    sub_863FE0();
  *(_BYTE *)(v13 + 182) &= ~1u;
  return v170;
}
