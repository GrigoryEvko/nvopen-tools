// Function: sub_24963E0
// Address: 0x24963e0
//
__int64 __fastcall sub_24963E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 **a4)
{
  __int64 *v4; // r13
  unsigned __int64 v5; // r12
  __int64 v7; // rax
  unsigned int *v8; // rsi
  __int64 v9; // r9
  unsigned int *v10; // r8
  unsigned int *v11; // rax
  int v12; // ecx
  unsigned int *v13; // rdx
  unsigned int *v14; // rsi
  __int64 v15; // r9
  unsigned int *v16; // r8
  unsigned int *v17; // rax
  int v18; // ecx
  unsigned int *v19; // rdx
  char *v20; // rax
  char v21; // dl
  __int64 v22; // r13
  int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // r14
  __int64 v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // r14
  unsigned __int64 v43; // rsi
  unsigned int *v44; // rax
  int v45; // ecx
  unsigned int *v46; // rdx
  _BYTE *v47; // rax
  __int64 v48; // r15
  _QWORD *v49; // rax
  __int64 v50; // r14
  unsigned int *v51; // r15
  unsigned int *v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // rax
  unsigned int *v56; // rsi
  __int64 v57; // r8
  __int64 v58; // r9
  unsigned int *v59; // r14
  unsigned __int64 v60; // rax
  int v61; // ecx
  _QWORD *v62; // rdx
  __int16 v63; // bx
  char v64; // bl
  _QWORD *v65; // rax
  __int64 v66; // r14
  unsigned __int64 v67; // r15
  _BYTE *v68; // rbx
  __int64 v69; // rdx
  unsigned int v70; // esi
  _QWORD *v71; // rax
  __int64 v72; // r15
  __int64 v73; // rbx
  _BYTE *v74; // rbx
  unsigned __int64 v75; // r12
  __int64 v76; // rdx
  unsigned int v77; // esi
  __int64 v78; // rax
  unsigned int *v79; // rsi
  __int64 v80; // r8
  __int64 v81; // r9
  unsigned int *v82; // r15
  unsigned __int64 v83; // rax
  int v84; // ecx
  _QWORD *v85; // rdx
  _QWORD *v86; // rax
  __int64 v87; // r15
  __int64 v88; // rbx
  unsigned __int64 v89; // rbx
  unsigned __int64 v90; // r12
  __int64 v91; // rdx
  unsigned int v92; // esi
  __int64 v93; // rsi
  const char *v94; // rsi
  __int64 v95; // r8
  __int64 v96; // r9
  const char *v97; // r12
  unsigned int *v98; // rax
  int v99; // ecx
  unsigned int *v100; // rdx
  __int64 v101; // rax
  __int64 v102; // r12
  unsigned int *v103; // r12
  unsigned int *v104; // rbx
  __int64 v105; // rdx
  unsigned int v106; // esi
  int v107; // eax
  int v108; // eax
  unsigned int v109; // edx
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 v112; // rdx
  int v113; // eax
  int v114; // eax
  unsigned int v115; // edx
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rdx
  int v119; // r12d
  unsigned __int64 v120; // rsi
  unsigned __int64 v121; // rsi
  unsigned __int64 v122; // rsi
  unsigned __int64 v123; // rsi
  unsigned __int64 v124; // rsi
  __int64 v125; // [rsp+48h] [rbp-328h]
  __int64 v126; // [rsp+70h] [rbp-300h]
  __int64 v127; // [rsp+78h] [rbp-2F8h]
  unsigned __int64 v128; // [rsp+80h] [rbp-2F0h]
  __int64 v130; // [rsp+90h] [rbp-2E0h]
  unsigned int *v131; // [rsp+90h] [rbp-2E0h]
  unsigned int *v132; // [rsp+90h] [rbp-2E0h]
  unsigned __int64 v134; // [rsp+98h] [rbp-2D8h]
  unsigned __int64 v135; // [rsp+98h] [rbp-2D8h]
  unsigned int v136[8]; // [rsp+A0h] [rbp-2D0h] BYREF
  __int16 v137; // [rsp+C0h] [rbp-2B0h]
  const char *v138[4]; // [rsp+D0h] [rbp-2A0h] BYREF
  __int16 v139; // [rsp+F0h] [rbp-280h]
  unsigned int *v140; // [rsp+100h] [rbp-270h] BYREF
  __int64 v141; // [rsp+108h] [rbp-268h]
  _BYTE v142[32]; // [rsp+110h] [rbp-260h] BYREF
  __int64 v143; // [rsp+130h] [rbp-240h]
  __int64 *v144; // [rsp+138h] [rbp-238h]
  unsigned __int16 v145; // [rsp+140h] [rbp-230h]
  __int64 v146; // [rsp+148h] [rbp-228h]
  void **v147; // [rsp+150h] [rbp-220h]
  void **v148; // [rsp+158h] [rbp-218h]
  __int64 v149; // [rsp+160h] [rbp-210h]
  int v150; // [rsp+168h] [rbp-208h]
  __int16 v151; // [rsp+16Ch] [rbp-204h]
  char v152; // [rsp+16Eh] [rbp-202h]
  __int64 v153; // [rsp+170h] [rbp-200h]
  __int64 v154; // [rsp+178h] [rbp-1F8h]
  void *v155; // [rsp+180h] [rbp-1F0h] BYREF
  void *v156; // [rsp+188h] [rbp-1E8h] BYREF
  _BYTE *v157; // [rsp+190h] [rbp-1E0h] BYREF
  __int64 v158; // [rsp+198h] [rbp-1D8h]
  _BYTE v159[16]; // [rsp+1A0h] [rbp-1D0h] BYREF
  __int16 v160; // [rsp+1B0h] [rbp-1C0h]
  __int64 v161; // [rsp+1C0h] [rbp-1B0h]
  __int64 v162; // [rsp+1C8h] [rbp-1A8h]
  __int64 v163; // [rsp+1D0h] [rbp-1A0h]
  __int64 v164; // [rsp+1D8h] [rbp-198h]
  void **v165; // [rsp+1E0h] [rbp-190h]
  void **v166; // [rsp+1E8h] [rbp-188h]
  __int64 v167; // [rsp+1F0h] [rbp-180h]
  int v168; // [rsp+1F8h] [rbp-178h]
  __int16 v169; // [rsp+1FCh] [rbp-174h]
  char v170; // [rsp+1FEh] [rbp-172h]
  __int64 v171; // [rsp+200h] [rbp-170h]
  __int64 v172; // [rsp+208h] [rbp-168h]
  void *v173; // [rsp+210h] [rbp-160h] BYREF
  void *v174; // [rsp+218h] [rbp-158h] BYREF
  unsigned __int64 v175; // [rsp+220h] [rbp-150h] BYREF
  __int64 v176; // [rsp+228h] [rbp-148h]
  _BYTE v177[16]; // [rsp+230h] [rbp-140h] BYREF
  __int16 v178; // [rsp+240h] [rbp-130h]
  __int64 v179; // [rsp+250h] [rbp-120h]
  __int64 v180; // [rsp+258h] [rbp-118h]
  __int64 v181; // [rsp+260h] [rbp-110h]
  __int64 v182; // [rsp+268h] [rbp-108h]
  void **v183; // [rsp+270h] [rbp-100h]
  void **v184; // [rsp+278h] [rbp-F8h]
  __int64 v185; // [rsp+280h] [rbp-F0h]
  int v186; // [rsp+288h] [rbp-E8h]
  __int16 v187; // [rsp+28Ch] [rbp-E4h]
  char v188; // [rsp+28Eh] [rbp-E2h]
  __int64 v189; // [rsp+290h] [rbp-E0h]
  __int64 v190; // [rsp+298h] [rbp-D8h]
  void *v191; // [rsp+2A0h] [rbp-D0h] BYREF
  void *v192; // [rsp+2A8h] [rbp-C8h] BYREF
  unsigned int *v193; // [rsp+2B0h] [rbp-C0h] BYREF
  __int64 v194; // [rsp+2B8h] [rbp-B8h]
  _BYTE v195[16]; // [rsp+2C0h] [rbp-B0h] BYREF
  __int16 v196; // [rsp+2D0h] [rbp-A0h]
  _QWORD *v197; // [rsp+2E0h] [rbp-90h]
  _QWORD *v198; // [rsp+2E8h] [rbp-88h]
  __int64 v199; // [rsp+2F0h] [rbp-80h]
  __int64 v200; // [rsp+2F8h] [rbp-78h]
  void **v201; // [rsp+300h] [rbp-70h]
  void **v202; // [rsp+308h] [rbp-68h]
  __int64 v203; // [rsp+310h] [rbp-60h]
  int v204; // [rsp+318h] [rbp-58h]
  __int16 v205; // [rsp+31Ch] [rbp-54h]
  char v206; // [rsp+31Eh] [rbp-52h]
  __int64 v207; // [rsp+320h] [rbp-50h]
  __int64 v208; // [rsp+328h] [rbp-48h]
  void *v209; // [rsp+330h] [rbp-40h] BYREF
  void *v210; // [rsp+338h] [rbp-38h] BYREF

  v4 = *(__int64 **)(a2 + 32);
  if ( v4 == (__int64 *)(*(_QWORD *)(a2 + 40) + 48LL) || !v4 )
  {
    v149 = 0;
    v146 = sub_BD5C60(0);
    v147 = &v155;
    v148 = &v156;
    v140 = (unsigned int *)v142;
    v141 = 0x200000000LL;
    v155 = &unk_49DA100;
    v150 = 0;
    v151 = 512;
    v152 = 7;
    v153 = 0;
    v154 = 0;
    v143 = 0;
    v144 = 0;
    v145 = 0;
    v156 = &unk_49DA0B0;
    BUG();
  }
  v5 = a2;
  v143 = 0;
  v146 = sub_BD5C60((__int64)(v4 - 3));
  v147 = &v155;
  v148 = &v156;
  v144 = 0;
  v151 = 512;
  v155 = &unk_49DA100;
  v149 = 0;
  v150 = 0;
  v152 = 7;
  v153 = 0;
  v154 = 0;
  v145 = 0;
  v156 = &unk_49DA0B0;
  v7 = v4[2];
  v140 = (unsigned int *)v142;
  v144 = v4;
  v141 = 0x200000000LL;
  v143 = v7;
  v8 = *(unsigned int **)sub_B46C60((__int64)(v4 - 3));
  v193 = v8;
  if ( !v8 || (sub_B96E90((__int64)&v193, (__int64)v8, 1), (v10 = v193) == 0) )
  {
    sub_93FB40((__int64)&v140, 0);
    v10 = v193;
    goto LABEL_27;
  }
  v11 = v140;
  v12 = v141;
  v13 = &v140[4 * (unsigned int)v141];
  if ( v140 == v13 )
  {
LABEL_128:
    if ( (unsigned int)v141 >= (unsigned __int64)HIDWORD(v141) )
    {
      v120 = (unsigned int)v141 + 1LL;
      if ( HIDWORD(v141) < v120 )
      {
        v131 = v193;
        sub_C8D5F0((__int64)&v140, v142, v120, 0x10u, (__int64)v193, v9);
        v10 = v131;
        v13 = &v140[4 * (unsigned int)v141];
      }
      *(_QWORD *)v13 = 0;
      *((_QWORD *)v13 + 1) = v10;
      v10 = v193;
      LODWORD(v141) = v141 + 1;
    }
    else
    {
      if ( v13 )
      {
        *v13 = 0;
        *((_QWORD *)v13 + 1) = v10;
        v12 = v141;
        v10 = v193;
      }
      LODWORD(v141) = v12 + 1;
    }
LABEL_27:
    if ( !v10 )
      goto LABEL_11;
    goto LABEL_10;
  }
  while ( 1 )
  {
    v9 = *v11;
    if ( !(_DWORD)v9 )
      break;
    v11 += 4;
    if ( v13 == v11 )
      goto LABEL_128;
  }
  *((_QWORD *)v11 + 1) = v193;
LABEL_10:
  sub_B91220((__int64)&v193, (__int64)v10);
LABEL_11:
  v14 = *(unsigned int **)(v5 + 48);
  v193 = v14;
  if ( v14 && (sub_B96E90((__int64)&v193, (__int64)v14, 1), (v16 = v193) != 0) )
  {
    v17 = v140;
    v18 = v141;
    v19 = &v140[4 * (unsigned int)v141];
    if ( v140 != v19 )
    {
      while ( *v17 )
      {
        v17 += 4;
        if ( v19 == v17 )
          goto LABEL_132;
      }
      *((_QWORD *)v17 + 1) = v193;
      goto LABEL_18;
    }
LABEL_132:
    if ( (unsigned int)v141 >= (unsigned __int64)HIDWORD(v141) )
    {
      v121 = (unsigned int)v141 + 1LL;
      if ( HIDWORD(v141) < v121 )
      {
        v132 = v193;
        sub_C8D5F0((__int64)&v140, v142, v121, 0x10u, (__int64)v193, v15);
        v16 = v132;
        v19 = &v140[4 * (unsigned int)v141];
      }
      *(_QWORD *)v19 = 0;
      *((_QWORD *)v19 + 1) = v16;
      v16 = v193;
      LODWORD(v141) = v141 + 1;
    }
    else
    {
      if ( v19 )
      {
        *v19 = 0;
        *((_QWORD *)v19 + 1) = v16;
        v18 = v141;
        v16 = v193;
      }
      LODWORD(v141) = v18 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v140, 0);
    v16 = v193;
  }
  if ( !v16 )
  {
    v20 = *(char **)(v5 - 32);
    v21 = *v20;
    if ( *v20 == 63 )
      goto LABEL_127;
    goto LABEL_19;
  }
LABEL_18:
  sub_B91220((__int64)&v193, (__int64)v16);
  v20 = *(char **)(v5 - 32);
  v21 = *v20;
  if ( *v20 == 63 )
  {
LABEL_127:
    v20 = *(char **)&v20[-32 * (*((_DWORD *)v20 + 1) & 0x7FFFFFF)];
    v21 = *v20;
  }
LABEL_19:
  if ( v21 == 3 && (v20[80] & 1) != 0 )
  {
    HIDWORD(v157) = 0;
    v196 = 257;
    v175 = (unsigned int)v157;
    if ( (_BYTE)v151 )
      v22 = sub_B358C0((__int64)&v140, 0x6Eu, v5, (__int64)a4, (unsigned int)v157, (__int64)&v193, 0, 0, 0);
    else
      v22 = sub_24932B0((__int64 *)&v140, 0x2Eu, v5, a4, (__int64)&v193, 0, (int)v157, 0);
    goto LABEL_23;
  }
  v24 = sub_24915C0(a3);
  v196 = 257;
  v25 = *(_QWORD *)(a1 + 48);
  v26 = 16LL * v24;
  v175 = *(_QWORD *)(v5 - 32);
  v176 = sub_ACD640(v25, v27, 0);
  v28 = sub_921880(&v140, *(_QWORD *)(a1 + v26 + 104), *(_QWORD *)(a1 + v26 + 112), (int)&v175, 2, (__int64)&v193, 0);
  v29 = *(_QWORD **)(v5 + 40);
  v125 = v28;
  v196 = 257;
  v30 = sub_AA8550(v29, v144, v145, (__int64)&v193, 0);
  v31 = v29[9];
  v130 = v30;
  v196 = 257;
  v32 = *(_QWORD *)(a1 + 8);
  v33 = sub_22077B0(0x50u);
  v126 = v33;
  if ( v33 )
    sub_AA4D50(v33, v32, (__int64)&v193, v31, v130);
  v34 = v29[9];
  v196 = 257;
  v35 = *(_QWORD *)(a1 + 8);
  v36 = sub_22077B0(0x50u);
  v127 = v36;
  if ( v36 )
    sub_AA4D50(v36, v35, (__int64)&v193, v34, v130);
  v37 = (_QWORD *)((v29[6] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v29[6] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v37 = 0;
  sub_B43D60(v37);
  v38 = sub_AA48A0((__int64)v29);
  v39 = *(_QWORD *)(v5 + 48);
  v206 = 7;
  v200 = v38;
  v201 = &v209;
  v202 = &v210;
  v193 = (unsigned int *)v195;
  v209 = &unk_49DA100;
  v194 = 0x200000000LL;
  v197 = v29;
  LOWORD(v199) = 0;
  v203 = 0;
  v204 = 0;
  v205 = 512;
  v207 = 0;
  v208 = 0;
  v210 = &unk_49DA0B0;
  v198 = v29 + 6;
  v175 = v39;
  if ( v39 && (sub_B96E90((__int64)&v175, v39, 1), (v42 = v175) != 0) )
  {
    v43 = (unsigned int)v194;
    v44 = v193;
    v45 = v194;
    v46 = &v193[4 * (unsigned int)v194];
    if ( v193 != v46 )
    {
      while ( *v44 )
      {
        v44 += 4;
        if ( v46 == v44 )
          goto LABEL_154;
      }
      *((_QWORD *)v44 + 1) = v175;
      goto LABEL_42;
    }
LABEL_154:
    if ( (unsigned int)v194 >= (unsigned __int64)HIDWORD(v194) )
    {
      v43 = (unsigned int)v194 + 1LL;
      if ( HIDWORD(v194) < v43 )
      {
        v43 = (unsigned __int64)v195;
        sub_C8D5F0((__int64)&v193, v195, (unsigned int)v194 + 1LL, 0x10u, v40, v41);
        v46 = &v193[4 * (unsigned int)v194];
      }
      *(_QWORD *)v46 = 0;
      *((_QWORD *)v46 + 1) = v42;
      v42 = v175;
      LODWORD(v194) = v194 + 1;
    }
    else
    {
      if ( v46 )
      {
        *v46 = 0;
        *((_QWORD *)v46 + 1) = v42;
        v45 = v194;
        v42 = v175;
      }
      LODWORD(v194) = v45 + 1;
    }
  }
  else
  {
    v43 = 0;
    sub_93FB40((__int64)&v193, 0);
    v42 = v175;
  }
  if ( v42 )
  {
LABEL_42:
    v43 = v42;
    sub_B91220((__int64)&v175, v42);
  }
  v160 = 257;
  v47 = (_BYTE *)sub_AD6530(*(_QWORD *)(v125 + 8), v43);
  v48 = sub_92B530(&v193, 0x20u, v125, v47, (__int64)&v157);
  v178 = 257;
  v49 = sub_BD2C40(72, 3u);
  v50 = (__int64)v49;
  if ( v49 )
    sub_B4C9A0((__int64)v49, v127, v126, v48, 3u, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, _QWORD *, __int64))*v202 + 2))(
    v202,
    v50,
    &v175,
    v198,
    v199);
  v51 = v193;
  v52 = &v193[4 * (unsigned int)v194];
  if ( v193 != v52 )
  {
    do
    {
      v53 = *((_QWORD *)v51 + 1);
      v54 = *v51;
      v51 += 4;
      sub_B99FD0(v50, v54, v53);
    }
    while ( v52 != v51 );
  }
  nullsub_61();
  v209 = &unk_49DA100;
  nullsub_63();
  if ( v193 != (unsigned int *)v195 )
    _libc_free((unsigned __int64)v193);
  v55 = sub_AA48A0(v126);
  v56 = *(unsigned int **)(v5 + 48);
  v164 = v55;
  v165 = &v173;
  v166 = &v174;
  v157 = v159;
  v173 = &unk_49DA100;
  v158 = 0x200000000LL;
  LOWORD(v163) = 0;
  v174 = &unk_49DA0B0;
  v167 = 0;
  v168 = 0;
  v169 = 512;
  v170 = 7;
  v171 = 0;
  v172 = 0;
  v161 = v126;
  v162 = v126 + 48;
  v193 = v56;
  if ( v56 && (sub_B96E90((__int64)&v193, (__int64)v56, 1), (v59 = v193) != 0) )
  {
    v60 = (unsigned __int64)v157;
    v61 = v158;
    v62 = &v157[16 * (unsigned int)v158];
    if ( v157 != (_BYTE *)v62 )
    {
      while ( *(_DWORD *)v60 )
      {
        v60 += 16LL;
        if ( v62 == (_QWORD *)v60 )
          goto LABEL_150;
      }
      *(_QWORD *)(v60 + 8) = v193;
      goto LABEL_56;
    }
LABEL_150:
    if ( (unsigned int)v158 >= (unsigned __int64)HIDWORD(v158) )
    {
      v122 = (unsigned int)v158 + 1LL;
      if ( HIDWORD(v158) < v122 )
      {
        sub_C8D5F0((__int64)&v157, v159, v122, 0x10u, v57, v58);
        v62 = &v157[16 * (unsigned int)v158];
      }
      *v62 = 0;
      v62[1] = v59;
      v59 = v193;
      LODWORD(v158) = v158 + 1;
    }
    else
    {
      if ( v62 )
      {
        *(_DWORD *)v62 = 0;
        v62[1] = v59;
        v61 = v158;
        v59 = v193;
      }
      LODWORD(v158) = v61 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v157, 0);
    v59 = v193;
  }
  if ( v59 )
LABEL_56:
    sub_B91220((__int64)&v193, (__int64)v59);
  v63 = *(_WORD *)(v5 + 2);
  v196 = 257;
  v178 = 257;
  v64 = v63 & 1;
  v65 = sub_BD2C40(80, unk_3F10A14);
  v66 = (__int64)v65;
  if ( v65 )
    sub_B4D190((__int64)v65, (__int64)a4, v125, (__int64)&v193, v64, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64))*v166 + 2))(
    v166,
    v66,
    &v175,
    v162,
    v163);
  v67 = (unsigned __int64)v157;
  v68 = &v157[16 * (unsigned int)v158];
  if ( v157 != v68 )
  {
    do
    {
      v69 = *(_QWORD *)(v67 + 8);
      v70 = *(_DWORD *)v67;
      v67 += 16LL;
      sub_B99FD0(v66, v70, v69);
    }
    while ( v68 != (_BYTE *)v67 );
  }
  if ( (_BYTE)qword_4FEA3C8 )
    v66 = sub_2495170(a1, v5, (_BYTE *)v66, (__int64)&v157, *(_QWORD *)(v5 - 32), 0xFFFFFFFF00000003LL);
  v196 = 257;
  v71 = sub_BD2C40(72, 1u);
  v72 = (__int64)v71;
  if ( v71 )
    sub_B4C8F0((__int64)v71, v130, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, unsigned int **, __int64, __int64))*v166 + 2))(
    v166,
    v72,
    &v193,
    v162,
    v163);
  v73 = 16LL * (unsigned int)v158;
  if ( v157 != &v157[v73] )
  {
    v134 = v5;
    v74 = &v157[v73];
    v75 = (unsigned __int64)v157;
    do
    {
      v76 = *(_QWORD *)(v75 + 8);
      v77 = *(_DWORD *)v75;
      v75 += 16LL;
      sub_B99FD0(v72, v77, v76);
    }
    while ( v74 != (_BYTE *)v75 );
    v5 = v134;
  }
  v78 = sub_AA48A0(v127);
  v79 = *(unsigned int **)(v5 + 48);
  LOWORD(v181) = 0;
  v182 = v78;
  v183 = &v191;
  v184 = &v192;
  v187 = 512;
  v175 = (unsigned __int64)v177;
  v191 = &unk_49DA100;
  v176 = 0x200000000LL;
  v188 = 7;
  v192 = &unk_49DA0B0;
  v185 = 0;
  v186 = 0;
  v189 = 0;
  v190 = 0;
  v179 = v127;
  v180 = v127 + 48;
  v193 = v79;
  if ( v79 && (sub_B96E90((__int64)&v193, (__int64)v79, 1), (v82 = v193) != 0) )
  {
    v83 = v175;
    v84 = v176;
    v85 = (_QWORD *)(v175 + 16LL * (unsigned int)v176);
    if ( (_QWORD *)v175 != v85 )
    {
      while ( *(_DWORD *)v83 )
      {
        v83 += 16LL;
        if ( v85 == (_QWORD *)v83 )
          goto LABEL_162;
      }
      *(_QWORD *)(v83 + 8) = v193;
      goto LABEL_76;
    }
LABEL_162:
    if ( (unsigned int)v176 >= (unsigned __int64)HIDWORD(v176) )
    {
      v124 = (unsigned int)v176 + 1LL;
      if ( HIDWORD(v176) < v124 )
      {
        sub_C8D5F0((__int64)&v175, v177, v124, 0x10u, v80, v81);
        v85 = (_QWORD *)(v175 + 16LL * (unsigned int)v176);
      }
      *v85 = 0;
      v85[1] = v82;
      v82 = v193;
      LODWORD(v176) = v176 + 1;
    }
    else
    {
      if ( v85 )
      {
        *(_DWORD *)v85 = 0;
        v85[1] = v82;
        v84 = v176;
        v82 = v193;
      }
      LODWORD(v176) = v84 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v175, 0);
    v82 = v193;
  }
  if ( v82 )
LABEL_76:
    sub_B91220((__int64)&v193, (__int64)v82);
  v136[1] = 0;
  v196 = 257;
  v138[0] = (const char *)v136[0];
  if ( (_BYTE)v187 )
    v135 = sub_B358C0((__int64)&v175, 0x6Eu, v5, (__int64)a4, v136[0], (__int64)&v193, 0, 0, 0);
  else
    v135 = sub_24932B0((__int64 *)&v175, 0x2Eu, v5, a4, (__int64)&v193, 0, v136[0], 0);
  v196 = 257;
  v86 = sub_BD2C40(72, 1u);
  v87 = (__int64)v86;
  if ( v86 )
    sub_B4C8F0((__int64)v86, v130, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, unsigned int **, __int64, __int64))*v184 + 2))(
    v184,
    v87,
    &v193,
    v180,
    v181);
  v88 = 16LL * (unsigned int)v176;
  if ( v175 != v175 + v88 )
  {
    v128 = v5;
    v89 = v175 + v88;
    v90 = v175;
    do
    {
      v91 = *(_QWORD *)(v90 + 8);
      v92 = *(_DWORD *)v90;
      v90 += 16LL;
      sub_B99FD0(v87, v92, v91);
    }
    while ( v89 != v90 );
    v5 = v128;
  }
  v93 = *(_QWORD *)(v130 + 56);
  if ( v93 )
    v93 -= 24;
  sub_23D0AB0((__int64)&v193, v93, 0, 0, 0);
  v94 = *(const char **)(v5 + 48);
  v138[0] = v94;
  if ( !v94 || (sub_B96E90((__int64)v138, (__int64)v94, 1), (v97 = v138[0]) == 0) )
  {
    sub_93FB40((__int64)&v193, 0);
    v97 = v138[0];
    goto LABEL_142;
  }
  v98 = v193;
  v99 = v194;
  v100 = &v193[4 * (unsigned int)v194];
  if ( v193 == v100 )
  {
LABEL_158:
    if ( (unsigned int)v194 >= (unsigned __int64)HIDWORD(v194) )
    {
      v123 = (unsigned int)v194 + 1LL;
      if ( HIDWORD(v194) < v123 )
      {
        sub_C8D5F0((__int64)&v193, v195, v123, 0x10u, v95, v96);
        v100 = &v193[4 * (unsigned int)v194];
      }
      *(_QWORD *)v100 = 0;
      *((_QWORD *)v100 + 1) = v97;
      v97 = v138[0];
      LODWORD(v194) = v194 + 1;
    }
    else
    {
      if ( v100 )
      {
        *v100 = 0;
        *((_QWORD *)v100 + 1) = v97;
        v99 = v194;
        v97 = v138[0];
      }
      LODWORD(v194) = v99 + 1;
    }
LABEL_142:
    if ( !v97 )
      goto LABEL_95;
    goto LABEL_94;
  }
  while ( *v98 )
  {
    v98 += 4;
    if ( v100 == v98 )
      goto LABEL_158;
  }
  *((const char **)v98 + 1) = v138[0];
LABEL_94:
  sub_B91220((__int64)v138, (__int64)v97);
LABEL_95:
  v137 = 257;
  v139 = 257;
  v101 = sub_BD2DA0(80);
  v22 = v101;
  if ( v101 )
  {
    v102 = v101;
    sub_B44260(v101, (__int64)a4, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v22 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v22, v138);
    sub_BD2A10(v22, *(_DWORD *)(v22 + 72), 1);
  }
  else
  {
    v102 = 0;
  }
  if ( (unsigned __int8)sub_920620(v102) )
  {
    v119 = v204;
    if ( v203 )
      sub_B99FD0(v22, 3u, v203);
    sub_B45150(v22, v119);
  }
  (*((void (__fastcall **)(void **, __int64, unsigned int *, _QWORD *, __int64))*v202 + 2))(v202, v22, v136, v198, v199);
  v103 = v193;
  v104 = &v193[4 * (unsigned int)v194];
  if ( v193 != v104 )
  {
    do
    {
      v105 = *((_QWORD *)v103 + 1);
      v106 = *v103;
      v103 += 4;
      sub_B99FD0(v22, v106, v105);
    }
    while ( v104 != v103 );
  }
  v107 = *(_DWORD *)(v22 + 4) & 0x7FFFFFF;
  if ( v107 == *(_DWORD *)(v22 + 72) )
  {
    sub_B48D90(v22);
    v107 = *(_DWORD *)(v22 + 4) & 0x7FFFFFF;
  }
  v108 = (v107 + 1) & 0x7FFFFFF;
  v109 = v108 | *(_DWORD *)(v22 + 4) & 0xF8000000;
  v110 = *(_QWORD *)(v22 - 8) + 32LL * (unsigned int)(v108 - 1);
  *(_DWORD *)(v22 + 4) = v109;
  if ( *(_QWORD *)v110 )
  {
    v111 = *(_QWORD *)(v110 + 8);
    **(_QWORD **)(v110 + 16) = v111;
    if ( v111 )
      *(_QWORD *)(v111 + 16) = *(_QWORD *)(v110 + 16);
  }
  *(_QWORD *)v110 = v66;
  if ( v66 )
  {
    v112 = *(_QWORD *)(v66 + 16);
    *(_QWORD *)(v110 + 8) = v112;
    if ( v112 )
      *(_QWORD *)(v112 + 16) = v110 + 8;
    *(_QWORD *)(v110 + 16) = v66 + 16;
    *(_QWORD *)(v66 + 16) = v110;
  }
  *(_QWORD *)(*(_QWORD *)(v22 - 8) + 32LL * *(unsigned int *)(v22 + 72)
                                   + 8LL * ((*(_DWORD *)(v22 + 4) & 0x7FFFFFFu) - 1)) = v126;
  v113 = *(_DWORD *)(v22 + 4) & 0x7FFFFFF;
  if ( v113 == *(_DWORD *)(v22 + 72) )
  {
    sub_B48D90(v22);
    v113 = *(_DWORD *)(v22 + 4) & 0x7FFFFFF;
  }
  v114 = (v113 + 1) & 0x7FFFFFF;
  v115 = v114 | *(_DWORD *)(v22 + 4) & 0xF8000000;
  v116 = *(_QWORD *)(v22 - 8) + 32LL * (unsigned int)(v114 - 1);
  *(_DWORD *)(v22 + 4) = v115;
  if ( *(_QWORD *)v116 )
  {
    v117 = *(_QWORD *)(v116 + 8);
    **(_QWORD **)(v116 + 16) = v117;
    if ( v117 )
      *(_QWORD *)(v117 + 16) = *(_QWORD *)(v116 + 16);
  }
  *(_QWORD *)v116 = v135;
  if ( v135 )
  {
    v118 = *(_QWORD *)(v135 + 16);
    *(_QWORD *)(v116 + 8) = v118;
    if ( v118 )
      *(_QWORD *)(v118 + 16) = v116 + 8;
    *(_QWORD *)(v116 + 16) = v135 + 16;
    *(_QWORD *)(v135 + 16) = v116;
  }
  *(_QWORD *)(*(_QWORD *)(v22 - 8) + 32LL * *(unsigned int *)(v22 + 72)
                                   + 8LL * ((*(_DWORD *)(v22 + 4) & 0x7FFFFFFu) - 1)) = v127;
  nullsub_61();
  v209 = &unk_49DA100;
  nullsub_63();
  if ( v193 != (unsigned int *)v195 )
    _libc_free((unsigned __int64)v193);
  nullsub_61();
  v191 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v175 != v177 )
    _libc_free(v175);
  nullsub_61();
  v173 = &unk_49DA100;
  nullsub_63();
  if ( v157 != v159 )
    _libc_free((unsigned __int64)v157);
LABEL_23:
  nullsub_61();
  v155 = &unk_49DA100;
  nullsub_63();
  if ( v140 != (unsigned int *)v142 )
    _libc_free((unsigned __int64)v140);
  return v22;
}
