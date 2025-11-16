// Function: sub_29A8CE0
// Address: 0x29a8ce0
//
__int64 *__fastcall sub_29A8CE0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64),
        __int64 a5)
{
  const char *v5; // rax
  size_t v6; // rdx
  const char *v7; // rax
  size_t v8; // rdx
  __int64 v10; // r14
  _BYTE *v11; // r13
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rbx
  const char **v15; // r12
  _QWORD *v16; // rdi
  size_t v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rbx
  _BYTE *v21; // r14
  size_t v22; // r13
  _QWORD *v23; // rax
  _QWORD *v24; // rdi
  size_t v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 *v28; // rbx
  unsigned __int64 *v29; // r14
  unsigned __int64 *v30; // rdi
  size_t v31; // rcx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r9
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // r12
  const char *v41; // rax
  size_t v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // rbx
  _QWORD *v45; // r14
  __int64 v46; // rax
  __int64 v47; // rbx
  const char *v48; // rax
  size_t v49; // rdx
  __int64 v50; // rax
  __int64 v51; // r14
  _QWORD *v52; // r13
  __int64 v53; // rax
  __int64 v54; // r14
  const char *v55; // rax
  size_t v56; // rdx
  unsigned __int8 v57; // r13
  __int16 v58; // r13
  _QWORD *v59; // rax
  __int64 v60; // rbx
  _QWORD *v61; // r12
  __int64 v62; // rax
  __int64 v63; // r12
  __int64 v64; // rbx
  const char *v65; // rax
  size_t v66; // rdx
  __int64 v67; // rax
  const char *v68; // rax
  size_t v69; // rdx
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // rbx
  __int64 v73; // r13
  const char *v74; // rax
  size_t v75; // rdx
  __int64 v76; // r13
  _QWORD *v77; // rdi
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // r13
  __int64 v82; // rbx
  __int64 v83; // r12
  __int64 v84; // r15
  unsigned int v85; // r12d
  __int64 v86; // rsi
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // r15
  char v92; // al
  char v93; // al
  __int64 v94; // r13
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // rcx
  __int64 v98; // r8
  unsigned __int8 *v99; // r14
  __int64 v100; // r13
  const char *v101; // rax
  size_t v102; // rdx
  _QWORD *v103; // rdi
  unsigned __int8 *v104; // rax
  unsigned __int8 *v105; // rax
  __int64 v106; // rbx
  _QWORD *v107; // rdi
  __int64 v108; // r13
  void *v109; // rax
  size_t v110; // rdx
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // rbx
  __int64 v115; // r12
  void *v116; // rax
  size_t v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rcx
  __int64 v121; // rbx
  __int64 v122; // rsi
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // rbx
  __int64 v128; // rdx
  __int64 v129; // r14
  __int64 v130; // rdx
  __int64 v131; // r15
  _QWORD *v132; // rax
  __int64 v133; // rsi
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 m; // rbx
  _QWORD *v139; // rax
  __int64 v140; // r14
  __int64 v141; // r13
  __int64 v142; // rsi
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // r8
  __int64 v146; // r9
  __int64 v147; // r14
  __int64 v148; // rdx
  __int64 v149; // rdx
  __int64 v150; // rbx
  __int64 v151; // r13
  _QWORD *v152; // rax
  size_t v153; // rdx
  unsigned int v154; // r15d
  __int64 v155; // r14
  unsigned int v156; // esi
  __int64 v157; // r13
  __int64 v158; // rsi
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // r8
  __int64 v162; // r9
  __int64 v163; // r13
  __int64 v165; // r12
  __int64 v166; // rsi
  __int64 v167; // rdx
  __int64 v168; // rcx
  __int64 v169; // r8
  __int64 v170; // r9
  __int64 v171; // r12
  _QWORD *v172; // r13
  __int64 v173; // rbx
  __int64 v174; // r12
  __int64 v175; // r14
  unsigned int v176; // r13d
  __int64 v177; // rdx
  __int64 v178; // rcx
  __int64 v179; // r8
  __int64 v180; // r9
  _QWORD *v181; // rdi
  size_t v182; // rdx
  size_t v183; // rdx
  size_t v184; // rdx
  const char **v185; // [rsp+0h] [rbp-100h]
  __int64 j; // [rsp+10h] [rbp-F0h]
  __int64 v188; // [rsp+18h] [rbp-E8h]
  __int64 i; // [rsp+20h] [rbp-E0h]
  __int64 v190; // [rsp+28h] [rbp-D8h]
  _QWORD *v192; // [rsp+38h] [rbp-C8h]
  __int64 v193; // [rsp+38h] [rbp-C8h]
  __int64 v194; // [rsp+38h] [rbp-C8h]
  __int64 v195; // [rsp+38h] [rbp-C8h]
  __int64 v196; // [rsp+38h] [rbp-C8h]
  _QWORD *v197; // [rsp+38h] [rbp-C8h]
  char v198; // [rsp+40h] [rbp-C0h]
  unsigned int v199; // [rsp+40h] [rbp-C0h]
  _QWORD *v200; // [rsp+40h] [rbp-C0h]
  __int64 v201; // [rsp+40h] [rbp-C0h]
  __int64 v202; // [rsp+40h] [rbp-C0h]
  __int64 v203; // [rsp+40h] [rbp-C0h]
  __int64 v204; // [rsp+40h] [rbp-C0h]
  void *src; // [rsp+50h] [rbp-B0h]
  char v208; // [rsp+60h] [rbp-A0h]
  char v209; // [rsp+60h] [rbp-A0h]
  __int64 v210; // [rsp+60h] [rbp-A0h]
  unsigned int v211; // [rsp+60h] [rbp-A0h]
  _QWORD *v212; // [rsp+60h] [rbp-A0h]
  __int64 k; // [rsp+60h] [rbp-A0h]
  __int64 v214; // [rsp+60h] [rbp-A0h]
  __int64 v215; // [rsp+60h] [rbp-A0h]
  __int64 v216; // [rsp+60h] [rbp-A0h]
  __int16 v217; // [rsp+68h] [rbp-98h]
  __int64 v218; // [rsp+68h] [rbp-98h]
  __int64 v219; // [rsp+68h] [rbp-98h]
  __int64 v220; // [rsp+68h] [rbp-98h]
  __int64 v221; // [rsp+68h] [rbp-98h]
  int v222; // [rsp+68h] [rbp-98h]
  size_t v223; // [rsp+78h] [rbp-88h] BYREF
  _QWORD *v224; // [rsp+80h] [rbp-80h] BYREF
  size_t n; // [rsp+88h] [rbp-78h]
  _QWORD v226[2]; // [rsp+90h] [rbp-70h] BYREF
  __int64 v227; // [rsp+A0h] [rbp-60h]
  __int64 v228; // [rsp+A8h] [rbp-58h]
  __int64 v229; // [rsp+B0h] [rbp-50h]

  v10 = *(_QWORD *)a2;
  v11 = *(_BYTE **)(a2 + 168);
  v12 = *(_QWORD *)(a2 + 176);
  v13 = sub_22077B0(0x370u);
  v14 = (_QWORD *)v13;
  if ( v13 )
    sub_BA8740(v13, v11, v12, v10);
  v15 = (const char **)&v224;
  *a1 = (__int64)v14;
  v224 = v226;
  sub_29A88A0((__int64 *)&v224, *(_BYTE **)(a2 + 200), *(_QWORD *)(a2 + 200) + *(_QWORD *)(a2 + 208));
  v16 = (_QWORD *)v14[25];
  if ( v224 == v226 )
  {
    v182 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v16 = v226[0];
      else
        memcpy(v16, v226, n);
      v182 = n;
      v16 = (_QWORD *)v14[25];
    }
    v14[26] = v182;
    *((_BYTE *)v16 + v182) = 0;
    v16 = v224;
  }
  else
  {
    v17 = n;
    v18 = v226[0];
    if ( v16 == v14 + 27 )
    {
      v14[25] = v224;
      v14[26] = v17;
      v14[27] = v18;
    }
    else
    {
      v19 = v14[27];
      v14[25] = v224;
      v14[26] = v17;
      v14[27] = v18;
      if ( v16 )
      {
        v224 = v16;
        v226[0] = v19;
        goto LABEL_7;
      }
    }
    v224 = v226;
    v16 = v226;
  }
LABEL_7:
  n = 0;
  *(_BYTE *)v16 = 0;
  if ( v224 != v226 )
    j_j___libc_free_0((unsigned __int64)v224);
  sub_BA9570(*a1, a2 + 312);
  v20 = (_QWORD *)*a1;
  v224 = v226;
  v21 = *(_BYTE **)(a2 + 232);
  v22 = *(_QWORD *)(a2 + 240);
  if ( &v21[v22] && !v21 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v223 = *(_QWORD *)(a2 + 240);
  if ( v22 > 0xF )
  {
    v224 = (_QWORD *)sub_22409D0((__int64)&v224, &v223, 0);
    v181 = v224;
    v226[0] = v223;
  }
  else
  {
    if ( v22 == 1 )
    {
      LOBYTE(v226[0]) = *v21;
      v23 = v226;
      goto LABEL_14;
    }
    if ( !v22 )
    {
      v23 = v226;
      goto LABEL_14;
    }
    v181 = v226;
  }
  memcpy(v181, v21, v22);
  v22 = v223;
  v23 = v224;
LABEL_14:
  n = v22;
  *((_BYTE *)v23 + v22) = 0;
  v227 = *(_QWORD *)(a2 + 264);
  v228 = *(_QWORD *)(a2 + 272);
  v229 = *(_QWORD *)(a2 + 280);
  v24 = (_QWORD *)v20[29];
  if ( v224 == v226 )
  {
    v184 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v24 = v226[0];
      else
        memcpy(v24, v226, n);
      v184 = n;
      v24 = (_QWORD *)v20[29];
    }
    v20[30] = v184;
    *((_BYTE *)v24 + v184) = 0;
    v24 = v224;
  }
  else
  {
    v25 = n;
    v26 = v226[0];
    if ( v24 == v20 + 31 )
    {
      v20[29] = v224;
      v20[30] = v25;
      v20[31] = v26;
    }
    else
    {
      v27 = v20[31];
      v20[29] = v224;
      v20[30] = v25;
      v20[31] = v26;
      if ( v24 )
      {
        v224 = v24;
        v226[0] = v27;
        goto LABEL_18;
      }
    }
    v224 = v226;
    v24 = v226;
  }
LABEL_18:
  n = 0;
  *(_BYTE *)v24 = 0;
  v20[33] = v227;
  v20[34] = v228;
  v20[35] = v229;
  if ( v224 != v226 )
    j_j___libc_free_0((unsigned __int64)v224);
  v28 = (unsigned __int64 *)*a1;
  v224 = v226;
  v29 = v28 + 13;
  sub_29A88A0((__int64 *)&v224, *(_BYTE **)(a2 + 88), *(_QWORD *)(a2 + 88) + *(_QWORD *)(a2 + 96));
  v30 = (unsigned __int64 *)v28[11];
  if ( v224 == v226 )
  {
    v183 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v30 = v226[0];
      else
        memcpy(v30, v226, n);
      v183 = n;
      v30 = (unsigned __int64 *)v28[11];
    }
    v28[12] = v183;
    *((_BYTE *)v30 + v183) = 0;
    v30 = v224;
  }
  else
  {
    v31 = n;
    v32 = v226[0];
    if ( v30 == v29 )
    {
      v28[11] = (unsigned __int64)v224;
      v28[12] = v31;
      v28[13] = v32;
    }
    else
    {
      v33 = v28[13];
      v28[11] = (unsigned __int64)v224;
      v28[12] = v31;
      v28[13] = v32;
      if ( v30 )
      {
        v224 = v30;
        v226[0] = v33;
        goto LABEL_24;
      }
    }
    v224 = v226;
    v30 = v226;
  }
LABEL_24:
  n = 0;
  *(_BYTE *)v30 = 0;
  if ( v224 != v226 )
    j_j___libc_free_0((unsigned __int64)v224);
  v34 = v28[12];
  if ( v34 )
  {
    v35 = v28[11];
    if ( *(_BYTE *)(v35 + v34 - 1) != 10 )
    {
      v36 = v34 + 1;
      if ( (unsigned __int64 *)v35 == v29 )
        v37 = 15;
      else
        v37 = v28[13];
      if ( v36 > v37 )
      {
        sub_2240BB0(v28 + 11, v28[12], 0, 0, 1u);
        v35 = v28[11];
        v36 = v34 + 1;
      }
      *(_BYTE *)(v35 + v34) = 10;
      v38 = v28[11];
      v28[12] = v36;
      *(_BYTE *)(v38 + v34 + 1) = 0;
    }
  }
  *(_BYTE *)(*a1 + 872) = *(_BYTE *)(a2 + 872);
  v39 = *(_QWORD *)(a2 + 16);
  v190 = a2 + 8;
  if ( v39 != a2 + 8 )
  {
    do
    {
      if ( !v39 )
        BUG();
      v192 = *(_QWORD **)(v39 - 32);
      v40 = *a1;
      v198 = *(_BYTE *)(v39 + 24) & 1;
      v208 = *(_BYTE *)(v39 - 24) & 0xF;
      v41 = sub_BD5D20(v39 - 56);
      LOWORD(v227) = 261;
      v224 = v41;
      n = v42;
      LOBYTE(v42) = *(_BYTE *)(v39 - 23);
      LODWORD(v41) = *(_DWORD *)(*(_QWORD *)(v39 - 48) + 8LL);
      BYTE4(v223) = 1;
      LODWORD(v223) = (unsigned int)v41 >> 8;
      v217 = ((unsigned __int8)v42 >> 2) & 7;
      v43 = sub_BD2C40(88, unk_3F0FAE8);
      v44 = (__int64)v43;
      if ( v43 )
        sub_B30000((__int64)v43, v40, v192, v198, v208, 0, (__int64)&v224, 0, v217, v223, 0);
      sub_B32030(v44, v39 - 56);
      v45 = sub_29A8950(a3, v39 - 56);
      v46 = v45[2];
      if ( v44 != v46 )
      {
        if ( v46 != 0 && v46 != -4096 && v46 != -8192 )
          sub_BD60C0(v45);
        v45[2] = v44;
        if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
          sub_BD73F0((__int64)v45);
      }
      v39 = *(_QWORD *)(v39 + 8);
    }
    while ( v190 != v39 );
    v15 = (const char **)&v224;
  }
  v47 = *(_QWORD *)(a2 + 32);
  for ( i = a2 + 24; v47 != i; v47 = *(_QWORD *)(v47 + 8) )
  {
    if ( !v47 )
    {
      v5 = sub_BD5D20(0);
      LOWORD(v227) = 261;
      v224 = v5;
      n = v6;
      BUG();
    }
    v193 = *a1;
    v48 = sub_BD5D20(v47 - 56);
    LOWORD(v227) = 261;
    v224 = v48;
    n = v49;
    v218 = *(_QWORD *)(v47 - 32);
    v209 = *(_BYTE *)(v47 - 24) & 0xF;
    v199 = *(_DWORD *)(*(_QWORD *)(v47 - 48) + 8LL) >> 8;
    v50 = sub_BD2DA0(136);
    v51 = v50;
    if ( v50 )
      sub_B2C3B0(v50, v218, v209, v199, (__int64)&v224, v193);
    sub_B2EC90(v51, v47 - 56);
    v52 = sub_29A8950(a3, v47 - 56);
    v53 = v52[2];
    if ( v53 != v51 )
    {
      if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
        sub_BD60C0(v52);
      v52[2] = v51;
      if ( v51 != 0 && v51 != -4096 && v51 != -8192 )
        sub_BD73F0((__int64)v52);
    }
  }
  v54 = *(_QWORD *)(a2 + 48);
  v188 = a2 + 40;
  if ( v54 != a2 + 40 )
  {
    do
    {
      v63 = v54 - 48;
      if ( !v54 )
        v63 = 0;
      if ( a4(a5, v63) )
      {
        v64 = *a1;
        v65 = sub_BD5D20(v63);
        LOWORD(v227) = 261;
        v224 = v65;
        n = v66;
        v67 = *(_QWORD *)(v63 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v67 + 8) - 17 <= 1 )
          v67 = **(_QWORD **)(v67 + 16);
        v60 = sub_B30580(
                *(_QWORD **)(v63 + 24),
                *(_DWORD *)(v67 + 8) >> 8,
                *(_BYTE *)(v63 + 32) & 0xF,
                (__int64)&v224,
                v64);
        sub_B31710(v60, (_BYTE *)v63);
      }
      else if ( *(_BYTE *)(*(_QWORD *)(v63 + 24) + 8LL) == 13 )
      {
        v201 = *a1;
        v68 = sub_BD5D20(v63);
        LOWORD(v227) = 261;
        v224 = v68;
        n = v69;
        v70 = *(_QWORD *)(v63 + 24);
        v211 = *(_DWORD *)(*(_QWORD *)(v63 + 8) + 8LL) >> 8;
        v71 = sub_BD2DA0(136);
        v60 = v71;
        if ( v71 )
          sub_B2C3B0(v71, v70, 0, v211, (__int64)&v224, v201);
      }
      else
      {
        v200 = *(_QWORD **)(v63 + 24);
        v210 = *a1;
        v55 = sub_BD5D20(v63);
        LOWORD(v227) = 261;
        v224 = v55;
        n = v56;
        v57 = *(_BYTE *)(v63 + 33);
        LODWORD(v55) = *(_DWORD *)(*(_QWORD *)(v63 + 8) + 8LL);
        BYTE4(v223) = 1;
        v58 = (v57 >> 2) & 7;
        LODWORD(v223) = (unsigned int)v55 >> 8;
        v59 = sub_BD2C40(88, unk_3F0FAE8);
        v60 = (__int64)v59;
        if ( v59 )
          sub_B30000((__int64)v59, v210, v200, 0, 0, 0, (__int64)&v224, 0, v58, v223, 0);
      }
      v61 = sub_29A8950(a3, v63);
      v62 = v61[2];
      if ( v60 != v62 )
      {
        if ( v62 != -4096 && v62 != 0 && v62 != -8192 )
          sub_BD60C0(v61);
        v61[2] = v60;
        if ( v60 != -4096 && v60 != 0 && v60 != -8192 )
          sub_BD73F0((__int64)v61);
      }
      v54 = *(_QWORD *)(v54 + 8);
    }
    while ( v188 != v54 );
    v15 = (const char **)&v224;
  }
  v72 = *(_QWORD *)(a2 + 64);
  for ( j = a2 + 56; j != v72; v72 = *(_QWORD *)(v72 + 8) )
  {
    if ( !v72 )
    {
      v7 = sub_BD5D20(0);
      LOWORD(v227) = 261;
      v224 = v7;
      n = v8;
      BUG();
    }
    v73 = *a1;
    v74 = sub_BD5D20(v72 - 56);
    LOWORD(v227) = 261;
    v224 = v74;
    n = v75;
    v76 = sub_B30730(
            *(_QWORD **)(v72 - 32),
            *(_DWORD *)(*(_QWORD *)(v72 - 48) + 8LL) >> 8,
            *(_BYTE *)(v72 - 24) & 0xF,
            (__int64)&v224,
            0,
            v73);
    sub_B31FB0(v76, v72 - 56);
    v77 = sub_29A8950(a3, v72 - 56);
    v78 = v77[2];
    if ( v76 != v78 )
    {
      if ( v78 != -4096 && v78 != 0 && v78 != -8192 )
        sub_BD60C0(v77);
      v77[2] = v76;
      if ( v76 != -4096 && v76 != 0 && v76 != -8192 )
        sub_BD73F0((__int64)v77);
    }
  }
  v194 = *(_QWORD *)(a2 + 16);
  if ( v190 != v194 )
  {
    v219 = a3;
    do
    {
      v79 = 0;
      if ( v194 )
        v79 = v194 - 56;
      v80 = v79;
      v202 = v79;
      v81 = sub_29A8950(v219, v79)[2];
      v224 = v226;
      n = 0x100000000LL;
      sub_B9A9D0(v80, (__int64)&v224);
      v82 = (__int64)v224;
      v83 = 2LL * (unsigned int)n;
      v212 = &v224[v83];
      if ( &v224[v83] != v224 )
      {
        do
        {
          v84 = *(_QWORD *)(v82 + 8);
          v85 = *(_DWORD *)v82;
          sub_FC75A0((__int64 *)&v223, v219, 0, 0, 0, 0);
          v86 = v84;
          v82 += 16;
          v91 = sub_FCD270((__int64 *)&v223, v84, v87, v88, v89, v90);
          sub_FC7680((__int64 *)&v223, v86);
          sub_B994D0(v81, v85, v91);
        }
        while ( v212 != (_QWORD *)v82 );
      }
      if ( !sub_B2FC80(v202) )
      {
        if ( a4(a5, v202) )
        {
          if ( !sub_B2FC80(v202) )
          {
            v165 = *(_QWORD *)(v202 - 32);
            sub_FC75A0((__int64 *)&v223, v219, 0, 0, 0, 0);
            v166 = v165;
            v171 = sub_FCD390((__int64 *)&v223, v165, v167, v168, v169, v170);
            sub_FC7680((__int64 *)&v223, v166);
            sub_B30160(v81, v171);
          }
          v114 = *(_QWORD *)(v202 + 48);
          if ( v114 )
          {
            v115 = *(_QWORD *)(v81 + 40);
            v116 = (void *)sub_AA8810(*(_QWORD **)(v202 + 48));
            v118 = sub_BAA410(v115, v116, v117);
            *(_DWORD *)(v118 + 8) = *(_DWORD *)(v114 + 8);
            sub_B2F990(v81, v118, v119, v120);
          }
        }
        else
        {
          v92 = *(_BYTE *)(v81 + 32);
          *(_BYTE *)(v81 + 32) = v92 & 0xF0;
          if ( (v92 & 0x30) != 0 )
            *(_BYTE *)(v81 + 33) |= 0x40u;
        }
      }
      if ( v224 != v226 )
        _libc_free((unsigned __int64)v224);
      v194 = *(_QWORD *)(v194 + 8);
    }
    while ( v190 != v194 );
    a3 = v219;
    v15 = (const char **)&v224;
  }
  for ( k = *(_QWORD *)(a2 + 32); i != k; k = *(_QWORD *)(k + 8) )
  {
    v94 = k - 56;
    if ( !k )
      v94 = 0;
    v220 = sub_29A8950(a3, v94)[2];
    if ( sub_B2FC80(v94) )
    {
      v224 = v226;
      n = 0x100000000LL;
      sub_B9A9D0(v94, (__int64)v15);
      v172 = v224;
      v173 = 2LL * (unsigned int)n;
      v197 = &v224[v173];
      if ( &v224[v173] != v224 )
      {
        v185 = v15;
        v174 = (__int64)v224;
        do
        {
          v175 = *(_QWORD *)(v174 + 8);
          v176 = *(_DWORD *)v174;
          sub_FC75A0((__int64 *)&v223, a3, 0, 0, 0, 0);
          v174 += 16;
          v204 = sub_FCD270((__int64 *)&v223, v175, v177, v178, v179, v180);
          sub_FC7680((__int64 *)&v223, v175);
          sub_B994D0(v220, v176, v204);
        }
        while ( v197 != (_QWORD *)v174 );
        v15 = v185;
        v172 = v224;
      }
      if ( v172 != v226 )
        _libc_free((unsigned __int64)v172);
    }
    else if ( a4(a5, v94) )
    {
      if ( (*(_BYTE *)(v220 + 2) & 1) != 0 )
        sub_B2C6D0(v220, v94, v95, v96);
      v97 = *(_QWORD *)(v220 + 96);
      if ( (*(_BYTE *)(v94 + 2) & 1) != 0 )
      {
        v196 = *(_QWORD *)(v220 + 96);
        sub_B2C6D0(v94, v94, v95, v97);
        v98 = *(_QWORD *)(v94 + 96);
        v97 = v196;
        v203 = v98 + 40LL * *(_QWORD *)(v94 + 104);
        if ( (*(_BYTE *)(v94 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v94, v94, v128, v196);
          v98 = *(_QWORD *)(v94 + 96);
          v97 = v196;
        }
      }
      else
      {
        v98 = *(_QWORD *)(v94 + 96);
        v203 = v98 + 40LL * *(_QWORD *)(v94 + 104);
      }
      v99 = (unsigned __int8 *)v97;
      if ( v98 != v203 )
      {
        v195 = v94;
        v100 = v98;
        do
        {
          v101 = sub_BD5D20(v100);
          LOWORD(v227) = 261;
          v224 = v101;
          n = v102;
          sub_BD6B50(v99, v15);
          v103 = sub_29A8950(a3, v100);
          v104 = (unsigned __int8 *)v103[2];
          if ( v104 != v99 )
          {
            if ( v104 + 4096 != 0 && v104 != 0 && v104 != (unsigned __int8 *)-8192LL )
              sub_BD60C0(v103);
            v105 = v99;
            v103[2] = v99;
            BYTE1(v105) = BYTE1(v99) & 0xEF;
            if ( v105 != (unsigned __int8 *)-8192LL && v99 )
              sub_BD73F0((__int64)v103);
          }
          v100 += 40;
          v99 += 40;
        }
        while ( v203 != v100 );
        v94 = v195;
      }
      v224 = v226;
      n = 0x800000000LL;
      sub_F4BB00(v220, (_QWORD *)v94, a3, 3, (__int64)v15, byte_3F871B3, 0, 0, 0);
      if ( (*(_BYTE *)(v94 + 2) & 8) != 0 )
      {
        v121 = sub_B2E500(v94);
        sub_FC75A0((__int64 *)&v223, a3, 0, 0, 0, 0);
        v122 = v121;
        v127 = sub_FCD390((__int64 *)&v223, v121, v123, v124, v125, v126);
        sub_FC7680((__int64 *)&v223, v122);
        sub_B2E8C0(v220, v127);
      }
      v106 = *(_QWORD *)(v94 + 48);
      if ( v106 )
      {
        v107 = *(_QWORD **)(v94 + 48);
        v108 = *(_QWORD *)(v220 + 40);
        v109 = (void *)sub_AA8810(v107);
        v111 = sub_BAA410(v108, v109, v110);
        *(_DWORD *)(v111 + 8) = *(_DWORD *)(v106 + 8);
        sub_B2F990(v220, v111, v112, v113);
      }
      if ( v224 != v226 )
        _libc_free((unsigned __int64)v224);
    }
    else
    {
      v93 = *(_BYTE *)(v220 + 32);
      *(_BYTE *)(v220 + 32) = v93 & 0xF0;
      if ( (v93 & 0x30) != 0 )
        *(_BYTE *)(v220 + 33) |= 0x40u;
      sub_B2E8C0(v220, 0);
    }
  }
  if ( v188 != *(_QWORD *)(a2 + 48) )
  {
    v221 = a3;
    v129 = *(_QWORD *)(a2 + 48);
    do
    {
      v130 = v129 - 48;
      if ( !v129 )
        v130 = 0;
      v131 = v130;
      if ( a4(a5, v130) )
      {
        v132 = sub_29A8950(v221, v131);
        v214 = *(_QWORD *)(v131 - 32);
        if ( v214 )
        {
          src = (void *)v132[2];
          sub_FC75A0((__int64 *)v15, v221, 0, 0, 0, 0);
          v133 = v214;
          v215 = sub_FCD390((__int64 *)v15, v214, v134, v135, v136, v137);
          sub_FC7680((__int64 *)v15, v133);
          sub_B303B0((__int64)src, v215);
        }
      }
      v129 = *(_QWORD *)(v129 + 8);
    }
    while ( v188 != v129 );
    a3 = v221;
  }
  for ( m = *(_QWORD *)(a2 + 64); j != m; m = *(_QWORD *)(m + 8) )
  {
    if ( !m )
    {
      sub_29A8950(a3, 0);
      BUG();
    }
    v139 = sub_29A8950(a3, m - 56);
    v140 = *(_QWORD *)(m - 88);
    if ( v140 )
    {
      v141 = v139[2];
      sub_FC75A0((__int64 *)v15, a3, 0, 0, 0, 0);
      v142 = v140;
      v147 = sub_FCD390((__int64 *)v15, v140, v143, v144, v145, v146);
      sub_FC7680((__int64 *)v15, v142);
      if ( *(_QWORD *)(v141 - 32) )
      {
        v148 = *(_QWORD *)(v141 - 24);
        **(_QWORD **)(v141 - 16) = v148;
        if ( v148 )
          *(_QWORD *)(v148 + 16) = *(_QWORD *)(v141 - 16);
      }
      *(_QWORD *)(v141 - 32) = v147;
      if ( v147 )
      {
        v149 = *(_QWORD *)(v147 + 16);
        *(_QWORD *)(v141 - 24) = v149;
        if ( v149 )
          *(_QWORD *)(v149 + 16) = v141 - 24;
        *(_QWORD *)(v141 - 16) = v147 + 16;
        *(_QWORD *)(v147 + 16) = v141 - 32;
      }
    }
  }
  v150 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v150 )
  {
    v216 = a3;
    do
    {
      v151 = *a1;
      v152 = (_QWORD *)sub_B91B20(v150);
      v154 = 0;
      v155 = sub_BA8E40(v151, v152, v153);
      v222 = sub_B91A00(v150);
      if ( v222 )
      {
        do
        {
          v156 = v154++;
          v157 = sub_B91A10(v150, v156);
          sub_FC75A0((__int64 *)v15, v216, 0, 0, 0, 0);
          v158 = v157;
          v163 = sub_FCD270((__int64 *)v15, v157, v159, v160, v161, v162);
          sub_FC7680((__int64 *)v15, v158);
          sub_B979A0(v155, v163);
        }
        while ( v222 != v154 );
      }
      v150 = *(_QWORD *)(v150 + 8);
    }
    while ( a2 + 72 != v150 );
  }
  return a1;
}
