// Function: sub_2C5E6D0
// Address: 0x2c5e6d0
//
bool __fastcall sub_2C5E6D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int8 *v3; // r13
  __int64 v4; // rax
  bool result; // al
  unsigned __int8 v6; // al
  unsigned __int8 *v7; // r15
  __int64 v8; // rdx
  _DWORD *v9; // r8
  unsigned __int64 v10; // r14
  __int64 v12; // rsi
  _DWORD *v13; // rax
  const void *v14; // r8
  int v15; // ecx
  unsigned int v16; // ecx
  bool v17; // dl
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // rcx
  _BYTE *v21; // rax
  __int64 v22; // rdi
  int v23; // r10d
  unsigned __int64 v24; // r9
  int v25; // eax
  int v26; // edx
  int v27; // r9d
  __int64 *v28; // rdi
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 **v31; // r10
  __int64 v32; // r8
  int v33; // edx
  unsigned __int8 *v34; // rax
  unsigned __int8 *v35; // r11
  __int64 v36; // r11
  unsigned __int64 v37; // rdx
  __int64 v38; // r14
  unsigned __int8 **v39; // rdi
  int v40; // ecx
  unsigned __int8 **v41; // rsi
  unsigned __int64 v42; // rcx
  int v43; // edx
  __int64 **v44; // r10
  __int64 v45; // r8
  unsigned __int8 *v46; // rax
  unsigned __int8 *v47; // r11
  __int64 v48; // r11
  unsigned __int64 v49; // rdx
  __int64 v50; // r14
  unsigned __int8 **v51; // rdi
  int v52; // esi
  unsigned __int8 **v53; // rcx
  unsigned __int64 v54; // rcx
  int v55; // edx
  __int64 v56; // r14
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rax
  __int64 v59; // r9
  char v60; // al
  __int64 *v61; // rdi
  __int64 v62; // rax
  __int64 *v63; // rdi
  int v64; // edx
  int v65; // r14d
  __int64 v66; // rax
  int v67; // edx
  bool v68; // zf
  int v69; // r14d
  unsigned __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  int v73; // edx
  signed __int64 v74; // rdx
  int v75; // ecx
  bool v76; // sf
  bool v77; // of
  __int64 v78; // rdi
  int *v79; // r10
  __int64 v80; // r11
  __int64 v81; // r14
  __int64 (__fastcall *v82)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v83; // rax
  __int64 v84; // rdi
  __int64 v85; // r10
  _BYTE *v86; // r11
  _BYTE *v87; // r14
  __int64 (__fastcall *v88)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v89; // rax
  __int64 v90; // r14
  __int64 v91; // r13
  __int64 i; // rbx
  unsigned int v93; // edx
  char v94; // al
  int *v95; // rdi
  int *v96; // rdi
  int *v97; // rax
  int *v98; // rcx
  _BYTE *v99; // rax
  int *v100; // rax
  int *v101; // rcx
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rcx
  __int64 v105; // rax
  __int64 v106; // r12
  __int64 v107; // r14
  __int64 v108; // rbx
  __int64 v109; // rdx
  unsigned int v110; // esi
  __int64 v111; // rax
  __int64 v112; // rcx
  __int64 v113; // rax
  __int64 v114; // r12
  __int64 v115; // r14
  __int64 v116; // rbx
  __int64 v117; // rdx
  unsigned int v118; // esi
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // [rsp-210h] [rbp-210h]
  unsigned __int8 *v122; // [rsp-208h] [rbp-208h]
  int v123; // [rsp-200h] [rbp-200h]
  __int64 v124; // [rsp-200h] [rbp-200h]
  __int64 **v125; // [rsp-1F8h] [rbp-1F8h]
  unsigned __int8 *v126; // [rsp-1F8h] [rbp-1F8h]
  int v127; // [rsp-1F0h] [rbp-1F0h]
  int v128; // [rsp-1F0h] [rbp-1F0h]
  int v129; // [rsp-1E8h] [rbp-1E8h]
  __int64 **v130; // [rsp-1E8h] [rbp-1E8h]
  int v131; // [rsp-1E0h] [rbp-1E0h]
  char v132; // [rsp-1E0h] [rbp-1E0h]
  int v133; // [rsp-1D8h] [rbp-1D8h]
  unsigned __int8 v134; // [rsp-1D8h] [rbp-1D8h]
  int v135; // [rsp-1D8h] [rbp-1D8h]
  __int64 v136; // [rsp-1D0h] [rbp-1D0h]
  char v137; // [rsp-1D0h] [rbp-1D0h]
  __int64 v138; // [rsp-1D0h] [rbp-1D0h]
  unsigned __int64 v139; // [rsp-1D0h] [rbp-1D0h]
  __int64 v140; // [rsp-1D0h] [rbp-1D0h]
  __int64 v141; // [rsp-1D0h] [rbp-1D0h]
  __int64 v142; // [rsp-1C8h] [rbp-1C8h]
  char v143; // [rsp-1C8h] [rbp-1C8h]
  int *v144; // [rsp-1C8h] [rbp-1C8h]
  _BYTE *v145; // [rsp-1C8h] [rbp-1C8h]
  const void *v146; // [rsp-1C8h] [rbp-1C8h]
  __int64 v147; // [rsp-1C8h] [rbp-1C8h]
  unsigned __int64 v148; // [rsp-1C8h] [rbp-1C8h]
  __int64 v149; // [rsp-1C8h] [rbp-1C8h]
  __int64 v150; // [rsp-1C8h] [rbp-1C8h]
  _BYTE *v151; // [rsp-1C8h] [rbp-1C8h]
  int *v152; // [rsp-1C8h] [rbp-1C8h]
  unsigned int v153; // [rsp-1C0h] [rbp-1C0h]
  __int64 v154; // [rsp-1C0h] [rbp-1C0h]
  __int64 v155; // [rsp-1C0h] [rbp-1C0h]
  int *v156; // [rsp-1C0h] [rbp-1C0h]
  __int64 v157; // [rsp-1C0h] [rbp-1C0h]
  int v158; // [rsp-1C0h] [rbp-1C0h]
  __int64 v159; // [rsp-1C0h] [rbp-1C0h]
  const void *v160; // [rsp-1C0h] [rbp-1C0h]
  __int64 v161; // [rsp-1C0h] [rbp-1C0h]
  int *v162; // [rsp-1C0h] [rbp-1C0h]
  __int64 v163; // [rsp-1C0h] [rbp-1C0h]
  __int64 v164; // [rsp-1C0h] [rbp-1C0h]
  unsigned int v165; // [rsp-1B8h] [rbp-1B8h]
  unsigned __int64 v166; // [rsp-1B8h] [rbp-1B8h]
  __int64 *v167; // [rsp-1B8h] [rbp-1B8h]
  const void *v168; // [rsp-1B8h] [rbp-1B8h]
  int v169; // [rsp-1B8h] [rbp-1B8h]
  int v170; // [rsp-1B0h] [rbp-1B0h]
  const void *v171; // [rsp-1B0h] [rbp-1B0h]
  __int64 v172; // [rsp-1B0h] [rbp-1B0h]
  unsigned __int64 v173; // [rsp-1B0h] [rbp-1B0h]
  const void *v174; // [rsp-1B0h] [rbp-1B0h]
  __int64 v175; // [rsp-1B0h] [rbp-1B0h]
  int v176; // [rsp-1A8h] [rbp-1A8h]
  const void *v177; // [rsp-1A8h] [rbp-1A8h]
  int v178; // [rsp-1A8h] [rbp-1A8h]
  __int64 v179; // [rsp-198h] [rbp-198h]
  unsigned int v180; // [rsp-18Ch] [rbp-18Ch]
  __int64 **v181; // [rsp-188h] [rbp-188h]
  unsigned __int8 *v182; // [rsp-188h] [rbp-188h]
  _DWORD *v183; // [rsp-188h] [rbp-188h]
  __int64 v184; // [rsp-180h] [rbp-180h]
  __int64 v185; // [rsp-180h] [rbp-180h]
  _BYTE *v186; // [rsp-180h] [rbp-180h]
  unsigned __int8 *v187; // [rsp-180h] [rbp-180h]
  bool v188; // [rsp-180h] [rbp-180h]
  bool v189; // [rsp-180h] [rbp-180h]
  int v190; // [rsp-16Ch] [rbp-16Ch] BYREF
  _BYTE *v191; // [rsp-168h] [rbp-168h] BYREF
  _BYTE *v192; // [rsp-160h] [rbp-160h] BYREF
  __int64 v193; // [rsp-158h] [rbp-158h] BYREF
  _BYTE *v194; // [rsp-150h] [rbp-150h] BYREF
  signed __int64 v195; // [rsp-148h] [rbp-148h] BYREF
  int v196; // [rsp-140h] [rbp-140h]
  int *v197[4]; // [rsp-138h] [rbp-138h] BYREF
  _BYTE *v198; // [rsp-118h] [rbp-118h] BYREF
  __int64 v199; // [rsp-110h] [rbp-110h]
  __int16 v200; // [rsp-F8h] [rbp-F8h]
  unsigned __int8 **v201; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v202; // [rsp-E0h] [rbp-E0h]
  _BYTE v203[16]; // [rsp-D8h] [rbp-D8h] BYREF
  __int16 v204; // [rsp-C8h] [rbp-C8h]
  int *v205; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v206; // [rsp-B0h] [rbp-B0h]
  _BYTE v207[48]; // [rsp-A8h] [rbp-A8h] BYREF
  int *v208; // [rsp-78h] [rbp-78h] BYREF
  __int64 v209; // [rsp-70h] [rbp-70h]
  _BYTE v210[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( *(_BYTE *)a2 != 92 )
    return 0;
  v2 = a2;
  v3 = *(unsigned __int8 **)(a2 - 64);
  v4 = *((_QWORD *)v3 + 2);
  if ( !v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 8) )
    return 0;
  v6 = *v3;
  if ( *v3 <= 0x1Cu )
    return 0;
  v7 = *(unsigned __int8 **)(a2 - 32);
  v8 = *((_QWORD *)v7 + 2);
  if ( !v8 )
    return 0;
  v184 = *(_QWORD *)(v8 + 8);
  if ( v184 )
    return 0;
  if ( *v7 <= 0x1Cu )
    return 0;
  v9 = *(_DWORD **)(a2 + 72);
  v10 = *(unsigned int *)(a2 + 80);
  if ( v6 != *v7 )
    return 0;
  if ( (unsigned int)v6 - 42 > 0x11 || !*((_QWORD *)v3 - 8) )
  {
LABEL_110:
    if ( (unsigned __int8)(v6 - 82) <= 1u )
    {
      v183 = v9;
      if ( *((_QWORD *)v3 - 8) )
      {
        v191 = (_BYTE *)*((_QWORD *)v3 - 8);
        if ( *((_QWORD *)v3 - 4) )
        {
          v192 = (_BYTE *)*((_QWORD *)v3 - 4);
          v180 = sub_B53900((__int64)v3);
          v93 = v180;
          if ( (unsigned __int8)(*v7 - 82) <= 1u )
          {
            if ( *((_QWORD *)v7 - 8) )
            {
              v193 = *((_QWORD *)v7 - 8);
              if ( *((_QWORD *)v7 - 4) )
              {
                v194 = (_BYTE *)*((_QWORD *)v7 - 4);
                if ( (unsigned int)sub_B53900((__int64)v7) == v93 )
                {
                  v94 = sub_B527F0((__int64)v3);
                  v14 = v183;
                  v17 = v94;
                  goto LABEL_23;
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  v191 = (_BYTE *)*((_QWORD *)v3 - 8);
  if ( !*((_QWORD *)v3 - 4)
    || (v192 = (_BYTE *)*((_QWORD *)v3 - 4), (unsigned int)*v7 - 42 > 0x11)
    || !*((_QWORD *)v7 - 8)
    || (v193 = *((_QWORD *)v7 - 8), !*((_QWORD *)v7 - 4)) )
  {
    v6 = *v3;
    goto LABEL_110;
  }
  v12 = (__int64)&v9[v10];
  v194 = (_BYTE *)*((_QWORD *)v7 - 4);
  v13 = sub_2C4D170(v9, v12, &dword_43A18EC);
  v15 = *v3;
  if ( (_DWORD *)v12 != v13 && ((unsigned __int8)(v15 - 51) <= 1u || (unsigned int)(v15 - 48) <= 1) )
    return 0;
  v16 = v15 - 29;
  v180 = 42;
  v17 = v16 <= 0x1E && ((1LL << v16) & 0x70066000) != 0;
LABEL_23:
  v18 = *(_QWORD *)(v2 + 8);
  v19 = *((_QWORD *)v3 + 1);
  if ( *(_BYTE *)(v18 + 8) != 17 )
    v18 = v184;
  v185 = v18;
  v20 = v18;
  if ( *(_BYTE *)(v19 + 8) != 17 )
    return 0;
  v21 = v191;
  v22 = *((_QWORD *)v191 + 1);
  v181 = (__int64 **)v22;
  if ( *(_BYTE *)(v22 + 8) != 17 || !v20 || v22 != *(_QWORD *)(v193 + 8) )
    return 0;
  v23 = *(_DWORD *)(v22 + 32);
  v190 = v23;
  if ( v191 != (_BYTE *)v193 && v17 && v192 != v194 && ((_BYTE *)v193 == v192 || v191 == v194) )
  {
    v191 = v192;
    v192 = v21;
  }
  v24 = 4 * v10;
  v205 = (int *)v207;
  v206 = 0xC00000000LL;
  if ( v10 > 0xC )
  {
    v168 = v14;
    v172 = v19;
    v176 = v23;
    sub_C8D5F0((__int64)&v205, v207, v10, 4u, (__int64)v14, v24);
    v23 = v176;
    v19 = v172;
    v14 = v168;
    v24 = 4 * v10;
    v96 = &v205[(unsigned int)v206];
  }
  else
  {
    v25 = 0;
    if ( !v24 )
      goto LABEL_36;
    v96 = (int *)v207;
  }
  v159 = v19;
  v169 = v23;
  v173 = v24;
  v177 = v14;
  memcpy(v96, v14, v24);
  v25 = v206;
  v19 = v159;
  v23 = v169;
  v24 = v173;
  v14 = v177;
LABEL_36:
  v165 = 6;
  v26 = v10 + v25;
  LODWORD(v206) = v10 + v25;
  if ( v191 == (_BYTE *)v193 )
  {
    v100 = v205;
    v101 = &v205[v26];
    if ( v205 != v101 )
    {
      do
      {
        if ( *v100 >= v23 )
          *v100 -= v23;
        ++v100;
      }
      while ( v101 != v100 );
    }
    v148 = v24;
    v160 = v14;
    v175 = v19;
    v178 = v23;
    v102 = sub_ACADE0(v181);
    v23 = v178;
    v165 = 7;
    v193 = v102;
    v19 = v175;
    v14 = v160;
    v24 = v148;
  }
  v208 = (int *)v210;
  v209 = 0xC00000000LL;
  if ( v24 > 0x30 )
  {
    v139 = v24;
    v146 = v14;
    v157 = v19;
    v170 = v23;
    sub_C8D5F0((__int64)&v208, v210, v10, 4u, (__int64)v14, v24);
    v23 = v170;
    v19 = v157;
    v14 = v146;
    v24 = v139;
    v95 = &v208[(unsigned int)v209];
  }
  else
  {
    if ( !v24 )
      goto LABEL_39;
    v95 = (int *)v210;
  }
  v147 = v19;
  v158 = v23;
  v171 = v14;
  memcpy(v95, v14, v24);
  LODWORD(v24) = v209;
  v19 = v147;
  v23 = v158;
  v14 = v171;
LABEL_39:
  v27 = v10 + v24;
  v153 = 6;
  LODWORD(v209) = v27;
  if ( v192 == v194 )
  {
    v97 = v208;
    v98 = &v208[v27];
    if ( v208 != v98 )
    {
      do
      {
        if ( *v97 >= v23 )
          *v97 -= v23;
        ++v97;
      }
      while ( v98 != v97 );
    }
    v174 = v14;
    v179 = v19;
    v99 = (_BYTE *)sub_ACADE0(v181);
    v19 = v179;
    v153 = 7;
    v194 = v99;
    v14 = v174;
  }
  v28 = *(__int64 **)(a1 + 152);
  v29 = *(unsigned int *)(a1 + 192);
  v198 = v3;
  v199 = (__int64)v7;
  v30 = sub_DFBC30(v28, 6, v19, (__int64)v14, v10, v29, 0, 0, (__int64)&v198, 2, v2);
  v31 = *(__int64 ***)(a1 + 152);
  v32 = *(unsigned int *)(a1 + 192);
  v136 = v30;
  v131 = v33;
  if ( (v7[7] & 0x40) != 0 )
  {
    v34 = (unsigned __int8 *)*((_QWORD *)v7 - 1);
    v35 = &v34[32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
  }
  else
  {
    v35 = v7;
    v34 = &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
  }
  v36 = v35 - v34;
  v37 = v36 >> 5;
  v201 = (unsigned __int8 **)v203;
  v202 = 0x400000000LL;
  v38 = v36 >> 5;
  if ( (unsigned __int64)v36 > 0x80 )
  {
    v124 = v36;
    v126 = v34;
    v128 = v32;
    v130 = v31;
    v135 = v36 >> 5;
    sub_C8D5F0((__int64)&v201, v203, v37, 8u, v32, (__int64)v203);
    v41 = v201;
    v40 = v202;
    LODWORD(v37) = v135;
    v31 = v130;
    LODWORD(v32) = v128;
    v39 = &v201[(unsigned int)v202];
    v34 = v126;
    v36 = v124;
  }
  else
  {
    v39 = (unsigned __int8 **)v203;
    v40 = 0;
    v41 = (unsigned __int8 **)v203;
  }
  if ( v36 > 0 )
  {
    v42 = 0;
    do
    {
      v39[v42 / 8] = *(unsigned __int8 **)&v34[4 * v42];
      v42 += 8LL;
      --v38;
    }
    while ( v38 );
    v41 = v201;
    v40 = v202;
  }
  LODWORD(v202) = v40 + v37;
  v142 = sub_DFCEF0(v31, v7, v41, (unsigned int)(v40 + v37), v32);
  v133 = v43;
  if ( v201 != (unsigned __int8 **)v203 )
    _libc_free((unsigned __int64)v201);
  v44 = *(__int64 ***)(a1 + 152);
  v45 = *(unsigned int *)(a1 + 192);
  if ( (v3[7] & 0x40) != 0 )
  {
    v46 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
    v47 = &v46[32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  }
  else
  {
    v47 = v3;
    v46 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  }
  v48 = v47 - v46;
  v201 = (unsigned __int8 **)v203;
  v202 = 0x400000000LL;
  v49 = v48 >> 5;
  v50 = v48 >> 5;
  if ( (unsigned __int64)v48 > 0x80 )
  {
    v121 = v48;
    v122 = v46;
    v123 = v45;
    v125 = v44;
    v127 = v48 >> 5;
    sub_C8D5F0((__int64)&v201, v203, v49, 8u, v45, (__int64)v203);
    v53 = v201;
    v52 = v202;
    LODWORD(v49) = v127;
    v44 = v125;
    LODWORD(v45) = v123;
    v51 = &v201[(unsigned int)v202];
    v46 = v122;
    v48 = v121;
  }
  else
  {
    v51 = (unsigned __int8 **)v203;
    v52 = 0;
    v53 = (unsigned __int8 **)v203;
  }
  if ( v48 > 0 )
  {
    v54 = 0;
    do
    {
      v51[v54 / 8] = *(unsigned __int8 **)&v46[4 * v54];
      v54 += 8LL;
      --v50;
    }
    while ( v50 );
    v53 = v201;
    v52 = v202;
  }
  LODWORD(v202) = v52 + v49;
  v56 = sub_DFCEF0(v44, v3, v53, (unsigned int)(v52 + v49), v45);
  if ( v201 != (unsigned __int8 **)v203 )
  {
    v129 = v55;
    _libc_free((unsigned __int64)v201);
    v55 = v129;
  }
  if ( v133 == 1 )
    v55 = 1;
  v57 = v142 + v56;
  if ( __OFADD__(v142, v56) )
  {
    v57 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v142 <= 0 )
      v57 = 0x8000000000000000LL;
  }
  if ( v131 == 1 )
    v55 = 1;
  v77 = __OFADD__(v136, v57);
  v58 = v136 + v57;
  if ( v77 )
  {
    v58 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v136 <= 0 )
      v58 = 0x8000000000000000LL;
  }
  v195 = v58;
  v59 = *(unsigned int *)(a1 + 192);
  v196 = v55;
  v197[0] = &v190;
  v197[1] = (int *)&v195;
  v197[2] = (int *)a1;
  v143 = sub_2C525B0(v197, (__int64 *)&v191, 0, v205, (unsigned int)v206, v59);
  v137 = sub_2C525B0(v197, (__int64 *)&v192, 0, v208, (unsigned int)v209, *(unsigned int *)(a1 + 192));
  v134 = sub_2C525B0(v197, &v193, v190, v205, (unsigned int)v206, *(unsigned int *)(a1 + 192));
  v60 = sub_2C525B0(v197, (__int64 *)&v194, v190, v208, (unsigned int)v209, *(unsigned int *)(a1 + 192));
  v61 = *(__int64 **)(a1 + 152);
  v132 = v60;
  v201 = (unsigned __int8 **)v192;
  v202 = (__int64)v194;
  v62 = sub_DFBC30(
          v61,
          v153,
          (__int64)v181,
          (__int64)v208,
          (unsigned int)v209,
          *(unsigned int *)(a1 + 192),
          0,
          0,
          (__int64)&v201,
          2,
          0);
  v63 = *(__int64 **)(a1 + 152);
  v154 = v62;
  v65 = v64;
  v198 = v191;
  v199 = v193;
  v66 = sub_DFBC30(
          v63,
          v165,
          (__int64)v181,
          (__int64)v205,
          (unsigned int)v206,
          *(unsigned int *)(a1 + 192),
          0,
          0,
          (__int64)&v198,
          2,
          0);
  v68 = v65 == 1;
  v69 = 1;
  if ( !v68 )
    v69 = v67;
  v77 = __OFADD__(v154, v66);
  v70 = v154 + v66;
  if ( v77 )
  {
    v70 = 0x8000000000000000LL;
    if ( v154 > 0 )
      v70 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v166 = v70;
  if ( v180 == 42 )
  {
    v72 = sub_DFD800(*(_QWORD *)(a1 + 152), (unsigned int)*v3 - 29, v185, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
  }
  else
  {
    v71 = sub_BCDA70(v181[3], *(_DWORD *)(v185 + 32));
    v72 = sub_DFD2D0(*(__int64 **)(a1 + 152), (unsigned int)*v3 - 29, v71);
  }
  if ( v73 == 1 )
    v69 = 1;
  v74 = v72 + v166;
  if ( __OFADD__(v72, v166) )
  {
    v74 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v72 <= 0 )
      v74 = 0x8000000000000000LL;
  }
  v186 = v191;
  if ( *v191 <= 0x15u && *(_BYTE *)v193 <= 0x15u )
  {
    v75 = v196;
    goto LABEL_77;
  }
  v75 = v196;
  if ( *v192 <= 0x15u && *v194 <= 0x15u || (unsigned __int8)(v137 | v143) | v134 || v132 )
  {
LABEL_77:
    v77 = __OFSUB__(v75, v69);
    v76 = v75 - v69 < 0;
    if ( v75 == v69 )
    {
      v77 = __OFSUB__(v195, v74);
      v76 = v195 - v74 < 0;
    }
    result = 0;
    if ( v76 != v77 )
      goto LABEL_105;
LABEL_80:
    v78 = *(_QWORD *)(a1 + 88);
    v79 = v205;
    v167 = (__int64 *)(a1 + 8);
    v80 = (unsigned int)v206;
    v200 = 257;
    v81 = v193;
    v82 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v78 + 112LL);
    if ( v82 == sub_9B6630 )
    {
      if ( *v191 > 0x15u || *(_BYTE *)v193 > 0x15u )
      {
LABEL_156:
        v150 = v80;
        v204 = 257;
        v162 = v79;
        v182 = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
        if ( v182 )
          sub_B4E9E0((__int64)v182, (__int64)v186, v81, v162, v150, (__int64)&v201, 0, 0);
        (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
          *(_QWORD *)(a1 + 96),
          v182,
          &v198,
          *(_QWORD *)(a1 + 64),
          *(_QWORD *)(a1 + 72));
        v111 = *(_QWORD *)(a1 + 8);
        v112 = v111 + 16LL * *(unsigned int *)(a1 + 16);
        if ( v111 != v112 )
        {
          v113 = v2;
          v114 = *(_QWORD *)(a1 + 8);
          v163 = a1;
          v115 = v113;
          v116 = v112;
          do
          {
            v117 = *(_QWORD *)(v114 + 8);
            v118 = *(_DWORD *)v114;
            v114 += 16;
            sub_B99FD0((__int64)v182, v118, v117);
          }
          while ( v116 != v114 );
          a1 = v163;
          v2 = v115;
        }
LABEL_85:
        v84 = *(_QWORD *)(a1 + 88);
        v200 = 257;
        v85 = (unsigned int)v209;
        v156 = v208;
        v86 = v194;
        v87 = v192;
        v88 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v84 + 112LL);
        if ( v88 == sub_9B6630 )
        {
          if ( *v192 > 0x15u || *v194 > 0x15u )
            goto LABEL_150;
          v138 = (unsigned int)v209;
          v145 = v194;
          v89 = sub_AD5CE0((__int64)v192, (__int64)v194, v208, (unsigned int)v209, 0);
          v86 = v145;
          v85 = v138;
          v187 = (unsigned __int8 *)v89;
        }
        else
        {
          v141 = (unsigned int)v209;
          v151 = v194;
          v119 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, int *, _QWORD))v88)(
                   v84,
                   v192,
                   v194,
                   v208,
                   (unsigned int)v209);
          v85 = v141;
          v86 = v151;
          v187 = (unsigned __int8 *)v119;
        }
        if ( v187 )
        {
LABEL_90:
          v204 = 257;
          if ( v180 == 42 )
            v90 = sub_2C51350(v167, *v3 - 29, v182, v187, (int)v198, 0, (__int64)&v201, 0);
          else
            v90 = sub_2B22A00((__int64)v167, v180, (__int64)v182, (__int64)v187, (__int64)&v201, 0);
          if ( *(_BYTE *)v90 > 0x1Cu )
          {
            sub_B45260((unsigned __int8 *)v90, (__int64)v3, 1);
            sub_B45560((unsigned __int8 *)v90, (unsigned __int64)v7);
          }
          v91 = a1 + 200;
          if ( *v182 > 0x1Cu )
            sub_F15FC0(a1 + 200, (__int64)v182);
          if ( *v187 > 0x1Cu )
            sub_F15FC0(a1 + 200, (__int64)v187);
          sub_BD84D0(v2, v90);
          if ( *(_BYTE *)v90 > 0x1Cu )
          {
            sub_BD6B90((unsigned __int8 *)v90, (unsigned __int8 *)v2);
            for ( i = *(_QWORD *)(v90 + 16); i; i = *(_QWORD *)(i + 8) )
              sub_F15FC0(v91, *(_QWORD *)(i + 24));
            if ( *(_BYTE *)v90 > 0x1Cu )
              sub_F15FC0(v91, v90);
          }
          result = 1;
          if ( *(_BYTE *)v2 > 0x1Cu )
          {
            sub_F15FC0(v91, v2);
            result = 1;
          }
          goto LABEL_105;
        }
LABEL_150:
        v140 = v85;
        v149 = (__int64)v86;
        v204 = 257;
        v187 = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
        if ( v187 )
          sub_B4E9E0((__int64)v187, (__int64)v87, v149, v156, v140, (__int64)&v201, 0, 0);
        (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE **, __int64, __int64))(**(_QWORD **)(a1 + 96) + 16LL))(
          *(_QWORD *)(a1 + 96),
          v187,
          &v198,
          v167[7],
          v167[8]);
        v103 = *(_QWORD *)(a1 + 8);
        v104 = v103 + 16LL * *(unsigned int *)(a1 + 16);
        if ( v103 != v104 )
        {
          v105 = v2;
          v106 = *(_QWORD *)(a1 + 8);
          v161 = a1;
          v107 = v105;
          v108 = v104;
          do
          {
            v109 = *(_QWORD *)(v106 + 8);
            v110 = *(_DWORD *)v106;
            v106 += 16;
            sub_B99FD0((__int64)v187, v110, v109);
          }
          while ( v108 != v106 );
          a1 = v161;
          v2 = v107;
        }
        goto LABEL_90;
      }
      v144 = v205;
      v155 = (unsigned int)v206;
      v83 = sub_AD5CE0((__int64)v191, v193, v205, (unsigned int)v206, 0);
      v80 = v155;
      v79 = v144;
      v182 = (unsigned __int8 *)v83;
    }
    else
    {
      v152 = v205;
      v164 = (unsigned int)v206;
      v120 = ((__int64 (__fastcall *)(__int64, _BYTE *, __int64, int *, _QWORD))v82)(
               v78,
               v191,
               v193,
               v205,
               (unsigned int)v206);
      v79 = v152;
      v80 = v164;
      v182 = (unsigned __int8 *)v120;
    }
    if ( v182 )
      goto LABEL_85;
    goto LABEL_156;
  }
  if ( v69 == v196 )
    result = v195 > v74;
  else
    result = v69 < v196;
  if ( result )
    goto LABEL_80;
LABEL_105:
  if ( v208 != (int *)v210 )
  {
    v188 = result;
    _libc_free((unsigned __int64)v208);
    result = v188;
  }
  if ( v205 != (int *)v207 )
  {
    v189 = result;
    _libc_free((unsigned __int64)v205);
    return v189;
  }
  return result;
}
