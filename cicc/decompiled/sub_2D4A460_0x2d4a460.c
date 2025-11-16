// Function: sub_2D4A460
// Address: 0x2d4a460
//
_BOOL8 __fastcall sub_2D4A460(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        int *a10)
{
  unsigned __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned int v15; // eax
  char v16; // cl
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // r14
  __int64 v33; // rbx
  unsigned __int8 v34; // al
  unsigned int v35; // ebx
  unsigned __int8 v36; // r15
  _QWORD *v37; // rax
  __int64 v38; // r12
  unsigned __int64 v39; // r14
  _BYTE *v40; // rbx
  __int64 v41; // rdx
  unsigned int v42; // esi
  _QWORD *v43; // rax
  __int64 v44; // r9
  __int64 v45; // r14
  unsigned int *v46; // r15
  unsigned int *v47; // rbx
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  __int64 v52; // rbx
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // rax
  _BYTE *v57; // rbx
  __int64 v58; // r15
  _BYTE *v59; // rcx
  _BYTE *v60; // r12
  __int64 v61; // r14
  __int64 v62; // rdx
  _BYTE *v63; // rsi
  unsigned __int64 v64; // rax
  size_t v65; // rdx
  __int64 v66; // r14
  __int64 v67; // rbx
  unsigned __int64 v68; // rax
  int v69; // edx
  __int64 v70; // rax
  __int64 v71; // r15
  _QWORD *v72; // rax
  _BYTE *v73; // r14
  unsigned int *v74; // r15
  unsigned int *v75; // rbx
  __int64 v76; // rdx
  unsigned int v77; // esi
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 *v81; // r14
  __int64 v82; // rbx
  unsigned __int8 v83; // al
  unsigned int v84; // ebx
  _QWORD *v85; // rax
  unsigned __int64 v86; // r14
  _BYTE *v87; // rbx
  __int64 v88; // rdx
  unsigned int v89; // esi
  _QWORD *v90; // rax
  __int64 v91; // r9
  __int64 v92; // rbx
  unsigned int *v93; // r14
  unsigned int *v94; // r13
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int64 v97; // rax
  unsigned __int64 v98; // rdx
  __int64 *v99; // r13
  __int64 v100; // rbx
  unsigned __int8 v101; // al
  unsigned int v102; // ebx
  unsigned __int64 v103; // r13
  _BYTE *v104; // rbx
  __int64 v105; // rdx
  unsigned int v106; // esi
  __int64 v107; // rax
  unsigned __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rbx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rax
  unsigned __int64 v114; // rdx
  __int64 v115; // r12
  _QWORD *v116; // rax
  __int64 v117; // r14
  unsigned int *v118; // r12
  unsigned int *v119; // rbx
  __int64 v120; // rdx
  unsigned int v121; // esi
  int v122; // r12d
  unsigned int *v123; // r12
  unsigned int *v124; // rbx
  __int64 v125; // rdx
  unsigned int v126; // esi
  __int64 v127; // rdx
  __int64 v128; // [rsp-10h] [rbp-320h]
  __int64 v129; // [rsp+18h] [rbp-2F8h]
  __int64 v130; // [rsp+20h] [rbp-2F0h]
  _QWORD *v132; // [rsp+60h] [rbp-2B0h]
  unsigned __int8 v133; // [rsp+60h] [rbp-2B0h]
  __int64 v134; // [rsp+68h] [rbp-2A8h]
  __int64 v135; // [rsp+70h] [rbp-2A0h]
  __int16 v136; // [rsp+7Ah] [rbp-296h]
  unsigned __int8 v137; // [rsp+7Ch] [rbp-294h]
  unsigned __int8 v138; // [rsp+7Fh] [rbp-291h]
  __int64 *v139; // [rsp+80h] [rbp-290h]
  __int64 v140; // [rsp+88h] [rbp-288h]
  __int64 v141; // [rsp+98h] [rbp-278h]
  int v143; // [rsp+B0h] [rbp-260h]
  bool v145; // [rsp+B8h] [rbp-258h]
  __int64 v146; // [rsp+B8h] [rbp-258h]
  __int64 *v148; // [rsp+C8h] [rbp-248h]
  __int64 v149; // [rsp+C8h] [rbp-248h]
  __int64 v150; // [rsp+C8h] [rbp-248h]
  unsigned __int64 v151; // [rsp+D8h] [rbp-238h] BYREF
  _DWORD v152[8]; // [rsp+E0h] [rbp-230h] BYREF
  __int16 v153; // [rsp+100h] [rbp-210h]
  _BYTE v154[32]; // [rsp+110h] [rbp-200h] BYREF
  __int16 v155; // [rsp+130h] [rbp-1E0h]
  _BYTE *v156; // [rsp+140h] [rbp-1D0h] BYREF
  __int64 v157; // [rsp+148h] [rbp-1C8h]
  _BYTE v158[48]; // [rsp+150h] [rbp-1C0h] BYREF
  _BYTE *v159; // [rsp+180h] [rbp-190h] BYREF
  __int64 v160; // [rsp+188h] [rbp-188h]
  _BYTE v161[16]; // [rsp+190h] [rbp-180h] BYREF
  __int16 v162; // [rsp+1A0h] [rbp-170h]
  unsigned int *v163; // [rsp+1C0h] [rbp-150h] BYREF
  __int64 v164; // [rsp+1C8h] [rbp-148h]
  _BYTE v165[32]; // [rsp+1D0h] [rbp-140h] BYREF
  __int64 v166; // [rsp+1F0h] [rbp-120h]
  __int64 v167; // [rsp+1F8h] [rbp-118h]
  __int64 v168; // [rsp+200h] [rbp-110h]
  __int64 v169; // [rsp+208h] [rbp-108h]
  void **v170; // [rsp+210h] [rbp-100h]
  void **v171; // [rsp+218h] [rbp-F8h]
  __int64 v172; // [rsp+220h] [rbp-F0h]
  int v173; // [rsp+228h] [rbp-E8h]
  __int16 v174; // [rsp+22Ch] [rbp-E4h]
  char v175; // [rsp+22Eh] [rbp-E2h]
  __int64 v176; // [rsp+230h] [rbp-E0h]
  __int64 v177; // [rsp+238h] [rbp-D8h]
  void *v178; // [rsp+240h] [rbp-D0h] BYREF
  void *v179; // [rsp+248h] [rbp-C8h] BYREF
  _BYTE *v180; // [rsp+250h] [rbp-C0h] BYREF
  __int64 v181; // [rsp+258h] [rbp-B8h]
  _BYTE v182[32]; // [rsp+260h] [rbp-B0h] BYREF
  __int64 v183; // [rsp+280h] [rbp-90h]
  __int64 v184; // [rsp+288h] [rbp-88h]
  __int64 v185; // [rsp+290h] [rbp-80h]
  __int64 v186; // [rsp+298h] [rbp-78h]
  void **v187; // [rsp+2A0h] [rbp-70h]
  void **v188; // [rsp+2A8h] [rbp-68h]
  __int64 v189; // [rsp+2B0h] [rbp-60h]
  int v190; // [rsp+2B8h] [rbp-58h]
  __int16 v191; // [rsp+2BCh] [rbp-54h]
  char v192; // [rsp+2BEh] [rbp-52h]
  __int64 v193; // [rsp+2C0h] [rbp-50h]
  __int64 v194; // [rsp+2C8h] [rbp-48h]
  void *v195; // [rsp+2D0h] [rbp-40h] BYREF
  void *v196; // [rsp+2D8h] [rbp-38h] BYREF

  v10 = a3;
  v148 = (__int64 *)sub_BD5C60(a2);
  v130 = sub_B43CA0(a2);
  v11 = v130 + 312;
  v164 = 0x200000000LL;
  v169 = sub_BD5C60(a2);
  v170 = &v178;
  v171 = &v179;
  v174 = 512;
  LOWORD(v168) = 0;
  v163 = (unsigned int *)v165;
  v175 = 7;
  v172 = 0;
  v173 = 0;
  v176 = 0;
  v177 = 0;
  v166 = 0;
  v167 = 0;
  v178 = &unk_49DA100;
  v179 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v163, a2);
  v12 = *(_QWORD *)(sub_B43CB0(a2) + 80);
  if ( !v12 )
    BUG();
  v13 = *(_QWORD *)(v12 + 32);
  if ( v13 )
    v13 -= 24;
  v14 = sub_BD5C60(v13);
  v181 = 0x200000000LL;
  v186 = v14;
  v180 = v182;
  v187 = &v195;
  v188 = &v196;
  LOWORD(v185) = 0;
  v196 = &unk_49DA0B0;
  v189 = 0;
  v190 = 0;
  v191 = 512;
  v192 = 7;
  v193 = 0;
  v194 = 0;
  v183 = 0;
  v184 = 0;
  v195 = &unk_49DA100;
  sub_D5F1F0((__int64)&v180, v13);
  v15 = sub_AE44F0(v11);
  v16 = a4;
  v145 = 0;
  v17 = v15 < 0x40 ? 8 : 16;
  if ( (unsigned int)v10 <= (unsigned __int64)(1LL << v16) && (unsigned int)v10 <= 0x10 )
  {
    v23 = 65814;
    if ( _bittest64(&v23, v10) )
      v145 = (unsigned int)v10 <= v17;
  }
  v139 = (__int64 *)sub_BCD140(v148, 8 * (int)v10);
  v138 = sub_AE5260(v11, (__int64)v139);
  v18 = sub_BCB2E0(v148);
  v140 = sub_ACD640(v18, (unsigned int)v10, 0);
  v19 = dword_444C480[a8];
  v20 = sub_BCB2D0(v148);
  v141 = 0;
  v129 = sub_ACD640(v20, v19, 0);
  if ( a7 )
  {
    v21 = dword_444C480[a9];
    v22 = sub_BCB2D0(v148);
    v141 = sub_ACD640(v22, v21, 0);
  }
  v135 = *(_QWORD *)(a2 + 8);
  v134 = sub_BCB120(v148);
  if ( v145 )
  {
    switch ( (int)v10 )
    {
      case 1:
        v143 = a10[1];
        break;
      case 2:
        v143 = a10[2];
        break;
      case 4:
        v143 = a10[3];
        break;
      case 8:
        v143 = a10[4];
        break;
      case 16:
        v143 = a10[5];
        break;
      default:
        break;
    }
  }
  else
  {
    v143 = *a10;
    if ( *a10 == 729 )
      goto LABEL_66;
  }
  if ( !*(_QWORD *)(*a1 + 8LL * v143 + 525288) )
  {
    v145 = 0;
    goto LABEL_66;
  }
  v151 = 0;
  v156 = v158;
  v157 = 0x600000000LL;
  if ( !v145 )
  {
    v109 = sub_AE4420(v11, (__int64)v148, 0);
    v110 = sub_ACD640(v109, (unsigned int)v10, 0);
    v113 = (unsigned int)v157;
    v114 = (unsigned int)v157 + 1LL;
    if ( v114 > HIDWORD(v157) )
    {
      sub_C8D5F0((__int64)&v156, v158, v114, 8u, v111, v112);
      v113 = (unsigned int)v157;
    }
    *(_QWORD *)&v156[8 * v113] = v110;
    LODWORD(v157) = v157 + 1;
  }
  v155 = 257;
  v24 = sub_BCE3C0(v148, 0);
  v27 = v24;
  if ( v24 == *(_QWORD *)(a5 + 8) )
  {
    v28 = a5;
  }
  else
  {
    v28 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v170 + 15))(v170, 50, a5, v24);
    if ( !v28 )
    {
      v162 = 257;
      v28 = sub_B51D30(50, a5, v27, (__int64)&v159, 0, 0);
      if ( (unsigned __int8)sub_920620(v28) )
      {
        v122 = v173;
        if ( v172 )
          sub_B99FD0(v28, 3u, v172);
        sub_B45150(v28, v122);
      }
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v171 + 2))(v171, v28, v154, v167, v168);
      v123 = v163;
      v124 = &v163[4 * (unsigned int)v164];
      if ( v163 != v124 )
      {
        do
        {
          v125 = *((_QWORD *)v123 + 1);
          v126 = *v123;
          v123 += 4;
          sub_B99FD0(v28, v126, v125);
        }
        while ( v124 != v123 );
      }
    }
  }
  v29 = (unsigned int)v157;
  v30 = (unsigned int)v157 + 1LL;
  if ( v30 > HIDWORD(v157) )
  {
    sub_C8D5F0((__int64)&v156, v158, v30, 8u, v25, v26);
    v29 = (unsigned int)v157;
  }
  *(_QWORD *)&v156[8 * v29] = v28;
  v31 = (unsigned int)(v157 + 1);
  LODWORD(v157) = v157 + 1;
  if ( !a7 )
  {
    if ( a6 )
    {
      v38 = 0;
      if ( v145 )
        goto LABEL_32;
      goto LABEL_73;
    }
    if ( v135 != v134 && !v145 )
    {
      v38 = 0;
      v55 = 0;
      v136 = v138;
LABEL_85:
      v155 = 257;
      v99 = *(__int64 **)(a2 + 8);
      v100 = sub_AA4E30(v183);
      v101 = sub_AE5260(v100, (__int64)v99);
      v102 = *(_DWORD *)(v100 + 4);
      v137 = v101;
      v162 = 257;
      v132 = sub_BD2C40(80, 1u);
      if ( v132 )
        sub_B4CCA0((__int64)v132, v99, v102, 0, v137, (__int64)&v159, 0, 0);
      (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64))*v188 + 2))(v188, v132, v154, v184, v185);
      v103 = (unsigned __int64)v180;
      v104 = &v180[16 * (unsigned int)v181];
      if ( v180 != v104 )
      {
        do
        {
          v105 = *(_QWORD *)(v103 + 8);
          v106 = *(_DWORD *)v103;
          v103 += 16LL;
          sub_B99FD0((__int64)v132, v106, v105);
        }
        while ( v104 != (_BYTE *)v103 );
      }
      *((_WORD *)v132 + 1) = v136 | *((_WORD *)v132 + 1) & 0xFFC0;
      sub_B34940((__int64)&v163, (__int64)v132, v140);
      v107 = (unsigned int)v157;
      v108 = (unsigned int)v157 + 1LL;
      if ( v108 > HIDWORD(v157) )
      {
        sub_C8D5F0((__int64)&v156, v158, v108, 8u, v25, v26);
        v107 = (unsigned int)v157;
      }
      *(_QWORD *)&v156[8 * v107] = v132;
      v31 = (unsigned int)(v157 + 1);
      LODWORD(v157) = v157 + 1;
      goto LABEL_35;
    }
    v38 = 0;
LABEL_108:
    v55 = 0;
    goto LABEL_109;
  }
  v155 = 257;
  v32 = *(__int64 **)(a7 + 8);
  v33 = sub_AA4E30(v183);
  v34 = sub_AE5260(v33, (__int64)v32);
  v35 = *(_DWORD *)(v33 + 4);
  v36 = v34;
  v162 = 257;
  v37 = sub_BD2C40(80, 1u);
  v38 = (__int64)v37;
  if ( v37 )
    sub_B4CCA0((__int64)v37, v32, v35, 0, v36, (__int64)&v159, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v188 + 2))(v188, v38, v154, v184, v185);
  v39 = (unsigned __int64)v180;
  v40 = &v180[16 * (unsigned int)v181];
  if ( v180 != v40 )
  {
    do
    {
      v41 = *(_QWORD *)(v39 + 8);
      v42 = *(_DWORD *)v39;
      v39 += 16LL;
      sub_B99FD0(v38, v42, v41);
    }
    while ( v40 != (_BYTE *)v39 );
  }
  *(_WORD *)(v38 + 2) = v138 | *(_WORD *)(v38 + 2) & 0xFFC0;
  sub_B34940((__int64)&v163, v38, v140);
  v162 = 257;
  v43 = sub_BD2C40(80, unk_3F10A10);
  v45 = (__int64)v43;
  if ( v43 )
    sub_B4D3C0((__int64)v43, a7, v38, 0, v138, v44, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v171 + 2))(v171, v45, &v159, v167, v168);
  v46 = v163;
  v47 = &v163[4 * (unsigned int)v164];
  if ( v163 != v47 )
  {
    do
    {
      v48 = *((_QWORD *)v46 + 1);
      v49 = *v46;
      v46 += 4;
      sub_B99FD0(v45, v49, v48);
    }
    while ( v47 != v46 );
  }
  v50 = (unsigned int)v157;
  v51 = (unsigned int)v157 + 1LL;
  if ( v51 > HIDWORD(v157) )
  {
    sub_C8D5F0((__int64)&v156, v158, v51, 8u, v25, v26);
    v50 = (unsigned int)v157;
  }
  *(_QWORD *)&v156[8 * v50] = v38;
  v31 = (unsigned int)(v157 + 1);
  LODWORD(v157) = v157 + 1;
  if ( !a6 )
    goto LABEL_108;
  if ( v145 )
  {
LABEL_32:
    v162 = 257;
    v52 = sub_10E0940((__int64 *)&v163, a6, (__int64)v139, (__int64)&v159);
    v53 = (unsigned int)v157;
    v54 = (unsigned int)v157 + 1LL;
    if ( v54 > HIDWORD(v157) )
    {
      sub_C8D5F0((__int64)&v156, v158, v54, 8u, v25, v26);
      v53 = (unsigned int)v157;
    }
    v55 = 0;
    v132 = 0;
    *(_QWORD *)&v156[8 * v53] = v52;
    v31 = (unsigned int)(v157 + 1);
    LODWORD(v157) = v157 + 1;
    goto LABEL_35;
  }
LABEL_73:
  v155 = 257;
  v81 = *(__int64 **)(a6 + 8);
  v82 = sub_AA4E30(v183);
  v83 = sub_AE5260(v82, (__int64)v81);
  v84 = *(_DWORD *)(v82 + 4);
  v133 = v83;
  v162 = 257;
  v85 = sub_BD2C40(80, 1u);
  v55 = (__int64)v85;
  if ( v85 )
    sub_B4CCA0((__int64)v85, v81, v84, 0, v133, (__int64)&v159, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v188 + 2))(v188, v55, v154, v184, v185);
  v86 = (unsigned __int64)v180;
  v87 = &v180[16 * (unsigned int)v181];
  if ( v180 != v87 )
  {
    do
    {
      v88 = *(_QWORD *)(v86 + 8);
      v89 = *(_DWORD *)v86;
      v86 += 16LL;
      sub_B99FD0(v55, v89, v88);
    }
    while ( v87 != (_BYTE *)v86 );
  }
  v136 = v138;
  *(_WORD *)(v55 + 2) = v138 | *(_WORD *)(v55 + 2) & 0xFFC0;
  sub_B34940((__int64)&v163, v55, v140);
  v162 = 257;
  v90 = sub_BD2C40(80, unk_3F10A10);
  v92 = (__int64)v90;
  if ( v90 )
  {
    sub_B4D3C0((__int64)v90, a6, v55, 0, v138, v91, 0, 0);
    v91 = v128;
  }
  (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64, __int64))*v171 + 2))(
    v171,
    v92,
    &v159,
    v167,
    v168,
    v91);
  v93 = v163;
  v94 = &v163[4 * (unsigned int)v164];
  if ( v163 != v94 )
  {
    do
    {
      v95 = *((_QWORD *)v93 + 1);
      v96 = *v93;
      v93 += 4;
      sub_B99FD0(v92, v96, v95);
    }
    while ( v94 != v93 );
  }
  v97 = (unsigned int)v157;
  v98 = (unsigned int)v157 + 1LL;
  if ( v98 > HIDWORD(v157) )
  {
    sub_C8D5F0((__int64)&v156, v158, v98, 8u, v25, v26);
    v97 = (unsigned int)v157;
  }
  *(_QWORD *)&v156[8 * v97] = v55;
  v31 = (unsigned int)(v157 + 1);
  LODWORD(v157) = v157 + 1;
  if ( !a7 && v135 != v134 )
    goto LABEL_85;
LABEL_109:
  v132 = 0;
LABEL_35:
  if ( v31 + 1 > (unsigned __int64)HIDWORD(v157) )
  {
    sub_C8D5F0((__int64)&v156, v158, v31 + 1, 8u, v25, v26);
    v31 = (unsigned int)v157;
  }
  *(_QWORD *)&v156[8 * v31] = v129;
  v56 = (unsigned int)(v157 + 1);
  LODWORD(v157) = v157 + 1;
  if ( v141 )
  {
    if ( v56 + 1 > (unsigned __int64)HIDWORD(v157) )
    {
      sub_C8D5F0((__int64)&v156, v158, v56 + 1, 8u, v25, v26);
      v56 = (unsigned int)v157;
    }
    *(_QWORD *)&v156[8 * v56] = v141;
    LODWORD(v157) = v157 + 1;
  }
  if ( a7 )
  {
    v139 = (__int64 *)sub_BCB2A0(v148);
    v151 = sub_A7A090((__int64 *)&v151, v148, 0, 79);
  }
  else if ( v135 == v134 || !v145 )
  {
    v139 = (__int64 *)sub_BCB120(v148);
  }
  v160 = 0x600000000LL;
  v159 = v161;
  if ( v156 == &v156[8 * (unsigned int)v157] )
  {
    v62 = 0;
    v63 = v161;
  }
  else
  {
    v149 = v55;
    v57 = v156 + 8;
    v58 = v38;
    v59 = v161;
    v60 = &v156[8 * (unsigned int)v157];
    v61 = *(_QWORD *)(*(_QWORD *)v156 + 8LL);
    v62 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v59[8 * v62] = v61;
      v62 = (unsigned int)(v160 + 1);
      LODWORD(v160) = v160 + 1;
      if ( v60 == v57 )
        break;
      v61 = *(_QWORD *)(*(_QWORD *)v57 + 8LL);
      if ( v62 + 1 > (unsigned __int64)HIDWORD(v160) )
      {
        sub_C8D5F0((__int64)&v159, v161, v62 + 1, 8u, v62 + 1, v26);
        v62 = (unsigned int)v160;
      }
      v59 = v159;
      v57 += 8;
    }
    v38 = v58;
    v63 = v159;
    v55 = v149;
  }
  v64 = sub_BCF480(v139, v63, v62, 0);
  v65 = 0;
  v66 = v64;
  v67 = *(_QWORD *)(*a1 + 8LL * v143 + 525288);
  if ( v67 )
    v65 = strlen(*(const char **)(*a1 + 8LL * v143 + 525288));
  v68 = sub_BA8C10(v130, v67, v65, v66, v151);
  v155 = 257;
  v150 = sub_921880(&v163, v68, v69, (int)v156, v157, (__int64)v154, 0);
  *(_QWORD *)(v150 + 72) = v151;
  if ( a6 && !v145 )
    sub_B349D0((__int64)&v163, v55, v140);
  if ( a7 )
  {
    v70 = sub_ACADE0(*(__int64 ***)(a2 + 8));
    v153 = 257;
    v146 = v70;
    v155 = 257;
    v71 = *(_QWORD *)(a7 + 8);
    v72 = sub_BD2C40(80, 1u);
    v73 = v72;
    if ( v72 )
      sub_B4D190((__int64)v72, v71, v38, (__int64)v154, 0, v138, 0, 0);
    (*((void (__fastcall **)(void **, _BYTE *, _DWORD *, __int64, __int64))*v171 + 2))(v171, v73, v152, v167, v168);
    v74 = v163;
    v75 = &v163[4 * (unsigned int)v164];
    if ( v163 != v75 )
    {
      do
      {
        v76 = *((_QWORD *)v74 + 1);
        v77 = *v74;
        v74 += 4;
        sub_B99FD0((__int64)v73, v77, v76);
      }
      while ( v75 != v74 );
    }
    sub_B349D0((__int64)&v163, v38, v140);
    v152[0] = 0;
    v155 = 257;
    v78 = sub_2466140((__int64 *)&v163, v146, v73, v152, 1, (__int64)v154);
    v155 = 257;
    v152[0] = 1;
    v79 = sub_2466140((__int64 *)&v163, v78, (_BYTE *)v150, v152, 1, (__int64)v154);
    sub_BD84D0(a2, v79);
  }
  else if ( v135 != v134 )
  {
    v115 = *(_QWORD *)(a2 + 8);
    if ( v145 )
    {
      v127 = *(_QWORD *)(a2 + 8);
      v155 = 257;
      v117 = sub_10E0940((__int64 *)&v163, v150, v127, (__int64)v154);
    }
    else
    {
      v155 = 257;
      v153 = 257;
      v116 = sub_BD2C40(80, 1u);
      v117 = (__int64)v116;
      if ( v116 )
        sub_B4D190((__int64)v116, v115, (__int64)v132, (__int64)v154, 0, v138, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v171 + 2))(v171, v117, v152, v167, v168);
      v118 = v163;
      v119 = &v163[4 * (unsigned int)v164];
      if ( v163 != v119 )
      {
        do
        {
          v120 = *((_QWORD *)v118 + 1);
          v121 = *v118;
          v118 += 4;
          sub_B99FD0(v117, v121, v120);
        }
        while ( v119 != v118 );
      }
      sub_B349D0((__int64)&v163, (__int64)v132, v140);
    }
    sub_BD84D0(a2, v117);
  }
  sub_B43D60((_QWORD *)a2);
  if ( v159 != v161 )
    _libc_free((unsigned __int64)v159);
  if ( v156 != v158 )
    _libc_free((unsigned __int64)v156);
  v145 = 1;
LABEL_66:
  nullsub_61();
  v195 = &unk_49DA100;
  nullsub_63();
  if ( v180 != v182 )
    _libc_free((unsigned __int64)v180);
  nullsub_61();
  v178 = &unk_49DA100;
  nullsub_63();
  if ( v163 != (unsigned int *)v165 )
    _libc_free((unsigned __int64)v163);
  return v145;
}
