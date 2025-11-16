// Function: sub_19BB5C0
// Address: 0x19bb5c0
//
__int64 __fastcall sub_19BB5C0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        unsigned int *a8,
        unsigned int a9,
        unsigned int *a10,
        unsigned __int32 a11,
        int *a12,
        bool *a13)
{
  __int64 v13; // r13
  int *v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // r14
  _QWORD *v18; // r15
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rax
  _DWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned int v25; // eax
  unsigned int v26; // r14d
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  _QWORD *v33; // r14
  _QWORD *v34; // r15
  _QWORD *v35; // rdi
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // r14
  _QWORD *v40; // r15
  _QWORD *v41; // rdi
  char v42; // r14
  unsigned int v43; // eax
  __int64 v45; // rax
  _DWORD *v46; // r8
  _DWORD *v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rdx
  bool v51; // zf
  __int64 v52; // rax
  _BYTE *v53; // r15
  _QWORD *v54; // rbx
  _QWORD *v55; // r15
  _QWORD *v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rax
  __int64 v64; // r8
  __int64 v65; // r9
  _BYTE *v66; // r15
  _QWORD *v67; // rbx
  _QWORD *v68; // r15
  _QWORD *v69; // rdi
  __int64 v70; // rdx
  __int64 v71; // rcx
  unsigned __int32 v72; // eax
  __int64 v73; // rax
  int v74; // eax
  int v75; // eax
  __int64 v76; // rax
  _QWORD *v77; // r14
  _QWORD *v78; // r15
  _QWORD *v79; // rdi
  unsigned int v80; // r14d
  __int64 v81; // rax
  _QWORD *v82; // r14
  _QWORD *v83; // r15
  _QWORD *v84; // rdi
  __int64 v85; // rax
  _QWORD *v86; // r14
  _QWORD *v87; // r15
  _QWORD *v88; // rdi
  __int64 v89; // rax
  __int64 v90; // r14
  _QWORD *v91; // r14
  _QWORD *v92; // rdi
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  _QWORD *v96; // r14
  _QWORD *v97; // rdi
  unsigned int v98; // ecx
  unsigned int v99; // eax
  __int64 v100; // rax
  __int64 v101; // r14
  _QWORD *v102; // r14
  _QWORD *v103; // rdi
  __int64 v104; // rax
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // rcx
  __int64 v108; // rdx
  _BYTE *v109; // r15
  _QWORD *v110; // rbx
  _QWORD *v111; // r15
  _QWORD *v112; // rdi
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  _QWORD *v116; // r14
  _QWORD *v117; // r15
  _QWORD *v118; // rdi
  __int64 v119; // rax
  _QWORD *v120; // r14
  _QWORD *v121; // r15
  _QWORD *v122; // rdi
  char v123; // r14
  __int64 v124; // rax
  __int64 v125; // r14
  _QWORD *v126; // rbx
  _QWORD *v127; // r12
  _QWORD *v128; // rdi
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // rax
  _BYTE *v136; // r15
  _QWORD *v137; // rbx
  _QWORD *v138; // r15
  _QWORD *v139; // rdi
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  _QWORD *v143; // r14
  _QWORD *v144; // r15
  _QWORD *v145; // rdi
  unsigned int v146; // eax
  __int64 v147; // rax
  _QWORD *v148; // r14
  _QWORD *v149; // r15
  _QWORD *v150; // rdi
  __int64 v151; // rax
  _QWORD *v152; // r14
  _QWORD *v153; // r15
  _QWORD *v154; // rdi
  __int64 v155; // rdi
  __int64 v156; // rax
  __int64 v157; // r14
  char *v158; // rsi
  size_t v159; // rdx
  int v160; // edx
  __int64 v161; // rax
  __int64 v162; // r14
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // rax
  __int64 v166; // rax
  unsigned int v167; // edx
  unsigned int v168; // esi
  __int64 v169; // r9
  unsigned __int64 v170; // rdi
  __int64 v171; // r8
  __int64 v172; // rcx
  unsigned int v173; // eax
  unsigned int v174; // eax
  unsigned int v175; // edx
  unsigned int v176; // eax
  unsigned int v177; // eax
  __int64 v178; // rax
  __int64 v179; // r14
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // rax
  __int64 v183; // rax
  _QWORD *v184; // rax
  unsigned int v185; // edx
  unsigned int v186; // ecx
  __int64 v187; // rdx
  unsigned __int64 v188; // rdi
  __int64 v189; // rsi
  unsigned int v190; // ecx
  unsigned int i; // eax
  unsigned int v192; // eax
  __int64 v193; // rax
  __int64 v194; // rax
  __int64 v195; // rax
  __int64 v196; // rax
  __int64 v197; // rax
  __int64 v198; // rax
  unsigned int v199; // ecx
  __int64 v200; // rax
  __int64 v201; // r14
  __int64 v202; // rax
  __int64 v203; // rax
  unsigned int v204; // edx
  __int64 v205; // rax
  __int64 v206; // rax
  __int64 v207; // rax
  __int64 v208; // rax
  __int64 v209; // rax
  __int64 v210; // rax
  __int64 v211; // rax
  __int64 v212; // rax
  __int64 v213; // rax
  __int64 v214; // rax
  __int64 v215; // rax
  unsigned __int64 v216; // r13
  __int64 v217; // rdx
  _QWORD *v218; // rax
  __int64 v219; // rax
  __int64 v220; // rax
  __int64 v221; // rax
  __int64 v222; // rax
  __int64 v223; // rax
  __int64 v224; // rax
  __int64 v225; // rax
  __int64 v226; // rax
  __int64 v227; // rax
  __int64 v228; // rax
  __int64 v229; // rax
  __int64 v230; // r15
  __int64 v231; // rax
  __int64 v232; // r14
  __int64 v233; // rax
  __int64 v234; // rax
  __int64 v235; // rax
  __int64 v236; // rax
  __int64 v237; // rax
  __int64 v238; // rax
  char v239; // [rsp+Fh] [rbp-391h]
  __int64 v240; // [rsp+10h] [rbp-390h]
  __int64 v241; // [rsp+18h] [rbp-388h]
  __int64 v243; // [rsp+20h] [rbp-380h]
  __int64 v245; // [rsp+28h] [rbp-378h]
  __int64 v246; // [rsp+28h] [rbp-378h]
  __int64 v247; // [rsp+28h] [rbp-378h]
  __int64 v248; // [rsp+28h] [rbp-378h]
  __int64 v249; // [rsp+28h] [rbp-378h]
  __int64 v250; // [rsp+30h] [rbp-370h]
  __int64 v251; // [rsp+38h] [rbp-368h]
  __int64 v252; // [rsp+38h] [rbp-368h]
  bool v253; // [rsp+38h] [rbp-368h]
  __int64 v254; // [rsp+38h] [rbp-368h]
  __int64 v255; // [rsp+38h] [rbp-368h]
  __int64 v256; // [rsp+38h] [rbp-368h]
  __int64 v257; // [rsp+38h] [rbp-368h]
  __int64 v258; // [rsp+38h] [rbp-368h]
  int v260; // [rsp+48h] [rbp-358h]
  __int64 v261; // [rsp+48h] [rbp-358h]
  __int64 v262; // [rsp+50h] [rbp-350h]
  __int64 v263; // [rsp+50h] [rbp-350h]
  __int64 v264; // [rsp+50h] [rbp-350h]
  unsigned __int8 v265; // [rsp+50h] [rbp-350h]
  __int64 v266; // [rsp+50h] [rbp-350h]
  __int64 v267; // [rsp+50h] [rbp-350h]
  unsigned int v268; // [rsp+68h] [rbp-338h] BYREF
  unsigned int v269; // [rsp+6Ch] [rbp-334h] BYREF
  __int64 v270; // [rsp+70h] [rbp-330h] BYREF
  __int64 v271[5]; // [rsp+78h] [rbp-328h] BYREF
  __int64 *v272; // [rsp+A0h] [rbp-300h]
  __int64 *v273; // [rsp+A8h] [rbp-2F8h]
  unsigned __int32 *v274; // [rsp+B0h] [rbp-2F0h]
  int *v275; // [rsp+B8h] [rbp-2E8h]
  unsigned int *v276; // [rsp+C0h] [rbp-2E0h]
  _QWORD v277[2]; // [rsp+D0h] [rbp-2D0h] BYREF
  _QWORD v278[2]; // [rsp+E0h] [rbp-2C0h] BYREF
  _QWORD *v279; // [rsp+F0h] [rbp-2B0h]
  _QWORD v280[6]; // [rsp+100h] [rbp-2A0h] BYREF
  __m128i v281; // [rsp+130h] [rbp-270h] BYREF
  _QWORD v282[2]; // [rsp+140h] [rbp-260h] BYREF
  _QWORD *v283; // [rsp+150h] [rbp-250h]
  _QWORD v284[6]; // [rsp+160h] [rbp-240h] BYREF
  void *v285; // [rsp+190h] [rbp-210h] BYREF
  char v286; // [rsp+198h] [rbp-208h]
  _QWORD *v287; // [rsp+1E8h] [rbp-1B8h] BYREF
  unsigned int v288; // [rsp+1F0h] [rbp-1B0h]
  _BYTE v289[424]; // [rsp+1F8h] [rbp-1A8h] BYREF

  v13 = a1;
  v14 = a12;
  sub_13FD840(&v270, a1);
  v15 = *a7;
  v271[0] = **(_QWORD **)(v13 + 32);
  v16 = sub_15E0530(v15);
  if ( sub_1602790(v16)
    || (v57 = sub_15E0530(*a7),
        v58 = sub_16033E0(v57),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v58 + 48LL))(v58)) )
  {
    v262 = v271[0];
    sub_15C9090((__int64)&v281, &v270);
    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v262);
    sub_15CAB20((__int64)&v285, "  Analyzing unrolling strategy...", 0x21u);
    sub_143AA50(a7, (__int64)&v285);
    v17 = v287;
    v285 = &unk_49ECF68;
    v18 = &v287[11 * v288];
    if ( v287 != v18 )
    {
      do
      {
        v18 -= 11;
        v19 = (_QWORD *)v18[4];
        if ( v19 != v18 + 6 )
          j_j___libc_free_0(v19, v18[6] + 1LL);
        if ( (_QWORD *)*v18 != v18 + 2 )
          j_j___libc_free_0(*v18, v18[2] + 1LL);
      }
      while ( v17 != v18 );
      v18 = v287;
    }
    if ( v18 != (_QWORD *)v289 )
      _libc_free((unsigned __int64)v18);
  }
  v20 = sub_16D5D50();
  v21 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_18;
  v22 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v23 = v21[2];
      v24 = v21[3];
      if ( v20 <= v21[4] )
        break;
      v21 = (_QWORD *)v21[3];
      if ( !v24 )
        goto LABEL_16;
    }
    v22 = v21;
    v21 = (_QWORD *)v21[2];
  }
  while ( v23 );
LABEL_16:
  if ( v22 == dword_4FA0208 )
    goto LABEL_18;
  if ( v20 < *((_QWORD *)v22 + 4) )
    goto LABEL_18;
  v45 = *((_QWORD *)v22 + 7);
  v46 = v22 + 12;
  if ( !v45 )
    goto LABEL_18;
  v20 = (unsigned int)dword_4FB2EA8;
  v47 = v22 + 12;
  do
  {
    while ( 1 )
    {
      v48 = *(_QWORD *)(v45 + 16);
      v49 = *(_QWORD *)(v45 + 24);
      if ( *(_DWORD *)(v45 + 32) >= dword_4FB2EA8 )
        break;
      v45 = *(_QWORD *)(v45 + 24);
      if ( !v49 )
        goto LABEL_64;
    }
    v47 = (_DWORD *)v45;
    v45 = *(_QWORD *)(v45 + 16);
  }
  while ( v48 );
LABEL_64:
  if ( v47 == v46 || dword_4FB2EA8 < v47[8] )
  {
LABEL_18:
    v260 = 0;
    goto LABEL_19;
  }
  v260 = v47[9];
  if ( v260 <= 0
    || (v50 = (unsigned int)dword_4FB2F40,
        v51 = *((_BYTE *)v14 + 46) == 0,
        *(_WORD *)((char *)v14 + 47) = 257,
        v14[5] = v50,
        v51) )
  {
LABEL_19:
    v25 = sub_19B5DD0(v13);
    v26 = v25;
    if ( v25 <= 1 )
      goto LABEL_20;
    goto LABEL_70;
  }
  v20 = (unsigned int)*v14;
  if ( v20 > (unsigned int)v14[10] + v50 * (unsigned __int64)(a11 - v14[10]) )
    goto LABEL_55;
  v25 = sub_19B5DD0(v13);
  v26 = v25;
  if ( v25 > 1 )
  {
LABEL_70:
    v52 = sub_15E0530(*a7);
    if ( sub_1602790(v52)
      || (v140 = sub_15E0530(*a7),
          v141 = sub_16033E0(v140),
          (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v141 + 48LL))(v141, v20)) )
    {
      v251 = v271[0];
      sub_15C9090((__int64)&v281, &v270);
      sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v251);
      sub_15CAB20((__int64)&v285, "    Reminder: loop accesses local arrays with approximate size of ", 0x42u);
      sub_15C9C50((__int64)&v281, "LocalArraySize", 14, v26);
      sub_17C2270((__int64)&v285, (__int64)&v281);
      if ( v283 != v284 )
        j_j___libc_free_0(v283, v284[0] + 1LL);
      if ( (_QWORD *)v281.m128i_i64[0] != v282 )
        j_j___libc_free_0(v281.m128i_i64[0], v282[0] + 1LL);
      v20 = (unsigned __int64)&v285;
      sub_143AA50(a7, (__int64)&v285);
      v285 = &unk_49ECF68;
      v53 = &v287[11 * v288];
      if ( v287 != (_QWORD *)v53 )
      {
        a12 = v14;
        v54 = &v287[11 * v288];
        v55 = v287;
        do
        {
          v54 -= 11;
          v56 = (_QWORD *)v54[4];
          if ( v56 != v54 + 6 )
          {
            v20 = v54[6] + 1LL;
            j_j___libc_free_0(v56, v20);
          }
          if ( (_QWORD *)*v54 != v54 + 2 )
          {
            v20 = v54[2] + 1LL;
            j_j___libc_free_0(*v54, v20);
          }
        }
        while ( v55 != v54 );
        v14 = a12;
        v53 = v287;
      }
      if ( v53 != v289 )
        _libc_free((unsigned __int64)v53);
    }
    v25 = 6;
    if ( v26 <= 6 )
      v25 = v26;
  }
LABEL_20:
  v268 = v25;
  v27 = sub_13FD000(v13);
  if ( v27
    && (v20 = (unsigned __int64)"llvm.loop.unroll.count", (v28 = sub_1AFD990(v27, "llvm.loop.unroll.count", 22)) != 0) )
  {
    v29 = *(_QWORD *)(*(_QWORD *)(v28 + 8 * (1LL - *(unsigned int *)(v28 + 8))) + 136LL);
    v30 = *(_QWORD **)(v29 + 24);
    if ( *(_DWORD *)(v29 + 32) > 0x40u )
      v30 = (_QWORD *)*v30;
  }
  else
  {
    LODWORD(v30) = 0;
  }
  v31 = *a7;
  v269 = (unsigned int)v30;
  v32 = sub_15E0530(v31);
  if ( sub_1602790(v32)
    || (v93 = sub_15E0530(*a7),
        v94 = sub_16033E0(v93),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v94 + 48LL))(v94)) )
  {
    v263 = v271[0];
    sub_15C9090((__int64)&v281, &v270);
    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v263);
    sub_15CAB20((__int64)&v285, "    Trying to unroll by #pragma (2nd priority)...", 0x31u);
    v20 = (unsigned __int64)&v285;
    sub_143AA50(a7, (__int64)&v285);
    v33 = v287;
    v285 = &unk_49ECF68;
    v34 = &v287[11 * v288];
    if ( v287 != v34 )
    {
      do
      {
        v34 -= 11;
        v35 = (_QWORD *)v34[4];
        if ( v35 != v34 + 6 )
        {
          v20 = v34[6] + 1LL;
          j_j___libc_free_0(v35, v20);
        }
        if ( (_QWORD *)*v34 != v34 + 2 )
        {
          v20 = v34[2] + 1LL;
          j_j___libc_free_0(*v34, v20);
        }
      }
      while ( v33 != v34 );
      v34 = v287;
    }
    if ( v34 != (_QWORD *)v289 )
      _libc_free((unsigned __int64)v34);
  }
  if ( !v269 )
  {
    v42 = 0;
    goto LABEL_92;
  }
  v36 = *a7;
  if ( v269 != 1 )
  {
    v37 = sub_15E0530(v36);
    if ( !sub_1602790(v37) )
    {
      v133 = sub_15E0530(*a7);
      v134 = sub_16033E0(v133);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v134 + 48LL))(v134) )
        goto LABEL_50;
    }
    v264 = v271[0];
    sub_15C9090((__int64)&v281, &v270);
    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v264);
    sub_15CAB20((__int64)&v285, "      Loop has \"#pragma unroll ", 0x1Fu);
    sub_15C9C50((__int64)&v281, "PragmaCount", 11, v269);
    v38 = sub_17C2270((__int64)&v285, (__int64)&v281);
    sub_15CAB20(v38, "\"", 1u);
    if ( v283 != v284 )
      j_j___libc_free_0(v283, v284[0] + 1LL);
    if ( (_QWORD *)v281.m128i_i64[0] != v282 )
      j_j___libc_free_0(v281.m128i_i64[0], v282[0] + 1LL);
    v20 = (unsigned __int64)&v285;
    sub_143AA50(a7, (__int64)&v285);
    v39 = v287;
    v285 = &unk_49ECF68;
    v40 = &v287[11 * v288];
    if ( v287 == v40 )
      goto LABEL_48;
    do
    {
      v40 -= 11;
      v41 = (_QWORD *)v40[4];
      if ( v41 != v40 + 6 )
      {
        v20 = v40[6] + 1LL;
        j_j___libc_free_0(v41, v20);
      }
      if ( (_QWORD *)*v40 != v40 + 2 )
      {
        v20 = v40[2] + 1LL;
        j_j___libc_free_0(*v40, v20);
      }
    }
    while ( v39 != v40 );
    goto LABEL_47;
  }
  v95 = sub_15E0530(v36);
  if ( !sub_1602790(v95) )
  {
    v180 = sub_15E0530(*a7);
    v181 = sub_16033E0(v180);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v181 + 48LL))(v181) )
      goto LABEL_50;
  }
  v266 = v271[0];
  sub_15C9090((__int64)&v281, &v270);
  sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v266);
  sub_15CAB20((__int64)&v285, "      Unrolling is disabled by source code \"#pragma unroll 1\"", 0x3Du);
  v20 = (unsigned __int64)&v285;
  sub_143AA50(a7, (__int64)&v285);
  v96 = v287;
  v285 = &unk_49ECF68;
  v40 = &v287[11 * v288];
  if ( v287 != v40 )
  {
    do
    {
      v40 -= 11;
      v97 = (_QWORD *)v40[4];
      if ( v97 != v40 + 6 )
      {
        v20 = v40[6] + 1LL;
        j_j___libc_free_0(v97, v20);
      }
      if ( (_QWORD *)*v40 != v40 + 2 )
      {
        v20 = v40[2] + 1LL;
        j_j___libc_free_0(*v40, v20);
      }
    }
    while ( v96 != v40 );
LABEL_47:
    v40 = v287;
  }
LABEL_48:
  if ( v40 != (_QWORD *)v289 )
    _libc_free((unsigned __int64)v40);
LABEL_50:
  *((_BYTE *)v14 + 45) = 1;
  v42 = 0;
  v43 = v269;
  *(_WORD *)((char *)v14 + 47) = 257;
  v14[5] = v43;
  if ( v43 != *a8 )
  {
    if ( v43 != 1 )
      goto LABEL_52;
    goto LABEL_265;
  }
  v20 = dword_4FB2760 * v268;
  if ( v20 > (unsigned int)v14[10] + v43 * (unsigned __int64)(a11 - v14[10]) )
    goto LABEL_55;
  v42 = 1;
  if ( v43 == 1 )
LABEL_265:
    sub_13FD1C0(v13);
LABEL_52:
  if ( *((_BYTE *)v14 + 46) )
  {
    v42 = *((_BYTE *)v14 + 46);
    v20 = (unsigned int)v14[5];
    if ( (unsigned int)v14[10] + v20 * (a11 - v14[10]) < (unsigned int)dword_4FB2760 )
      goto LABEL_55;
  }
  else if ( *a10 % v269 )
  {
    v104 = sub_15E0530(*a7);
    if ( sub_1602790(v104)
      || (v193 = sub_15E0530(*a7),
          v194 = sub_16033E0(v193),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v194 + 48LL))(v194)) )
    {
      v254 = v271[0];
      sub_15C9090((__int64)&v281, &v270);
      sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v254);
      sub_15CAB20((__int64)&v285, "      Failed : not allow remainder loops", 0x28u);
      v20 = (unsigned __int64)&v285;
      sub_143AA50(a7, (__int64)&v285);
      v107 = (__int64)v287;
      v285 = &unk_49ECF68;
      v108 = 5LL * v288;
      v109 = &v287[11 * v288];
      if ( v287 != (_QWORD *)v109 )
      {
        a12 = v14;
        v110 = &v287[11 * v288];
        v111 = v287;
        do
        {
          v110 -= 11;
          v112 = (_QWORD *)v110[4];
          if ( v112 != v110 + 6 )
          {
            v20 = v110[6] + 1LL;
            j_j___libc_free_0(v112, v20);
          }
          v108 = (__int64)(v110 + 2);
          if ( (_QWORD *)*v110 != v110 + 2 )
          {
            v20 = v110[2] + 1LL;
            j_j___libc_free_0(*v110, v20);
          }
        }
        while ( v111 != v110 );
        v14 = a12;
        v109 = v287;
      }
      if ( v109 != v289 )
        _libc_free((unsigned __int64)v109);
    }
    v271[1] = (__int64)&v270;
    v271[2] = (__int64)v271;
    v271[3] = (__int64)&v269;
    v271[4] = (__int64)a10;
    sub_19B78B0(a7, v20, v108, v107, v105, v106, &v270, v271, &v269, a10);
    v20 = (unsigned int)v14[5];
    if ( (unsigned int)dword_4FB2760 <= (unsigned int)v14[10] + v20 * (a11 - v14[10]) )
      v42 = 1;
  }
  else
  {
    v42 = 1;
    if ( (unsigned int)v14[10] + (unsigned int)v14[5] * (unsigned __int64)(a11 - v14[10]) < (unsigned int)dword_4FB2760 )
    {
LABEL_55:
      v265 = 1;
      goto LABEL_56;
    }
  }
LABEL_92:
  v60 = sub_13FD000(v13);
  if ( v60 )
  {
    v20 = (unsigned __int64)"llvm.loop.unroll.full";
    v240 = sub_1AFD990(v60, "llvm.loop.unroll.full", 21);
    if ( v240 )
    {
      v63 = sub_15E0530(*a7);
      if ( sub_1602790(v63)
        || (v113 = sub_15E0530(*a7),
            v114 = sub_16033E0(v113),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v114 + 48LL))(v114)) )
      {
        v252 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v252);
        sub_15CAB20((__int64)&v285, "      Loop has \"#pragma unroll\" (full unroll directive)", 0x37u);
        v20 = (unsigned __int64)&v285;
        sub_143AA50(a7, (__int64)&v285);
        v285 = &unk_49ECF68;
        v66 = &v287[11 * v288];
        if ( v287 != (_QWORD *)v66 )
        {
          a12 = v14;
          v67 = &v287[11 * v288];
          v68 = v287;
          do
          {
            v67 -= 11;
            v69 = (_QWORD *)v67[4];
            if ( v69 != v67 + 6 )
            {
              v20 = v67[6] + 1LL;
              j_j___libc_free_0(v69, v20);
            }
            if ( (_QWORD *)*v67 != v67 + 2 )
            {
              v20 = v67[2] + 1LL;
              j_j___libc_free_0(*v67, v20);
            }
          }
          while ( v68 != v67 );
          v14 = a12;
          v66 = v287;
        }
        if ( v66 != v289 )
          _libc_free((unsigned __int64)v66);
      }
      v70 = *a8;
      if ( (_DWORD)v70 )
      {
        v71 = (unsigned int)v14[10];
        v72 = a11;
        v14[5] = v70;
        v20 = dword_4FB2760 * v268;
        if ( v20 > v71 + v70 * (unsigned __int64)(v72 - (unsigned int)v71) )
          goto LABEL_55;
      }
      else
      {
        v265 = *((_BYTE *)v14 + 49) & (a9 != 0);
        if ( v265 )
        {
          v70 = (unsigned int)v14[10];
          v20 = a9;
          v199 = v268;
          v14[5] = a9;
          if ( dword_4FB2760 * v199 > v70 + a9 * (unsigned __int64)(a11 - (unsigned int)v70) )
          {
            *a13 = 1;
            *a8 = a9;
            *a10 = 1;
            goto LABEL_56;
          }
        }
        else
        {
          v14[5] = 0;
          v135 = sub_15E0530(*a7);
          if ( sub_1602790(v135)
            || (v210 = sub_15E0530(*a7),
                v211 = sub_16033E0(v210),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v211 + 48LL))(v211)) )
          {
            v255 = v271[0];
            sub_15C9090((__int64)&v281, &v270);
            sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v255);
            sub_15CAB20(
              (__int64)&v285,
              "      Failed : trip count or its upper bound could not be statically calculated",
              0x4Fu);
            v20 = (unsigned __int64)&v285;
            sub_143AA50(a7, (__int64)&v285);
            v285 = &unk_49ECF68;
            v70 = 5LL * v288;
            v136 = &v287[11 * v288];
            if ( v287 != (_QWORD *)v136 )
            {
              a12 = v14;
              v137 = &v287[11 * v288];
              v138 = v287;
              do
              {
                v137 -= 11;
                v139 = (_QWORD *)v137[4];
                if ( v139 != v137 + 6 )
                {
                  v20 = v137[6] + 1LL;
                  j_j___libc_free_0(v139, v20);
                }
                v70 = (__int64)(v137 + 2);
                if ( (_QWORD *)*v137 != v137 + 2 )
                {
                  v20 = v137[2] + 1LL;
                  j_j___libc_free_0(*v137, v20);
                }
              }
              while ( v138 != v137 );
              v14 = a12;
              v136 = v287;
            }
            if ( v136 != v289 )
              _libc_free((unsigned __int64)v136);
          }
          if ( !v42 )
            goto LABEL_108;
        }
      }
      v273 = v271;
      v274 = &a11;
      v275 = v14;
      v276 = &v268;
      v272 = &v270;
      sub_19B7B10(a7, v20, v70, (__int64)&v270, v64, v65, &v270, v271, &a11, (__int64)v14, &v268);
LABEL_108:
      v239 = 1;
      goto LABEL_112;
    }
  }
  if ( v42 )
  {
    v273 = v271;
    v274 = &a11;
    v275 = v14;
    v276 = &v268;
    v272 = &v270;
    sub_19B7B10(a7, v20, v59, (__int64)&v270, v61, v62, &v270, v271, &a11, (__int64)v14, &v268);
  }
  v239 = 0;
  v240 = 0;
  if ( !v269 )
  {
    v142 = sub_15E0530(*a7);
    if ( sub_1602790(v142)
      || (v202 = sub_15E0530(*a7),
          v203 = sub_16033E0(v202),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v203 + 48LL))(v203)) )
    {
      v267 = v271[0];
      sub_15C9090((__int64)&v281, &v270);
      sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v267);
      sub_15CAB20((__int64)&v285, "      Failed : \"#pragma unroll\" is not set", 0x2Au);
      sub_143AA50(a7, (__int64)&v285);
      v143 = v287;
      v285 = &unk_49ECF68;
      v144 = &v287[11 * v288];
      if ( v287 != v144 )
      {
        do
        {
          v144 -= 11;
          v145 = (_QWORD *)v144[4];
          if ( v145 != v144 + 6 )
            j_j___libc_free_0(v145, v144[6] + 1LL);
          if ( (_QWORD *)*v144 != v144 + 2 )
            j_j___libc_free_0(*v144, v144[2] + 1LL);
        }
        while ( v143 != v144 );
        v144 = v287;
      }
      if ( v144 != (_QWORD *)v289 )
        _libc_free((unsigned __int64)v144);
    }
    v240 = 0;
    v239 = 0;
  }
LABEL_112:
  v73 = sub_13FD000(v13);
  v253 = 0;
  v250 = v73;
  if ( v73 )
  {
    v250 = sub_1AFD990(v73, "llvm.loop.unroll.enable", 23);
    v253 = v250 != 0;
  }
  if ( v269 || (v265 = (v260 > 0) | v239 | v253) != 0 )
  {
    v265 = 1;
    if ( *a8 )
    {
      v74 = dword_4FB2760;
      if ( *v14 >= (unsigned int)dword_4FB2760 )
        v74 = *v14;
      *v14 = v74;
      v75 = dword_4FB2760;
      if ( v14[3] >= (unsigned int)dword_4FB2760 )
        v75 = v14[3];
      v14[3] = v75;
    }
  }
  v76 = sub_15E0530(*a7);
  if ( sub_1602790(v76)
    || (v129 = sub_15E0530(*a7),
        v130 = sub_16033E0(v129),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v130 + 48LL))(v130)) )
  {
    v241 = v271[0];
    sub_15C9090((__int64)&v281, &v270);
    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v241);
    sub_15CAB20((__int64)&v285, "    Trying to fully unroll (3rd priority)...", 0x2Cu);
    sub_143AA50(a7, (__int64)&v285);
    v77 = v287;
    v285 = &unk_49ECF68;
    v78 = &v287[11 * v288];
    if ( v287 != v78 )
    {
      do
      {
        v78 -= 11;
        v79 = (_QWORD *)v78[4];
        if ( v79 != v78 + 6 )
          j_j___libc_free_0(v79, v78[6] + 1LL);
        if ( (_QWORD *)*v78 != v78 + 2 )
          j_j___libc_free_0(*v78, v78[2] + 1LL);
      }
      while ( v77 != v78 );
      v78 = v287;
    }
    if ( v78 != (_QWORD *)v289 )
      _libc_free((unsigned __int64)v78);
  }
  v80 = *a8;
  if ( *a8 )
  {
    v14[5] = v80;
  }
  else
  {
    v14[5] = a9;
    if ( !a9 )
    {
      v81 = sub_15E0530(*a7);
      if ( !sub_1602790(v81) )
      {
        v182 = sub_15E0530(*a7);
        v183 = sub_16033E0(v182);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v183 + 48LL))(v183) )
          goto LABEL_144;
      }
      v245 = v271[0];
      sub_15C9090((__int64)&v281, &v270);
      sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v245);
      sub_15CAB20(
        (__int64)&v285,
        "      Failed : trip count or its upper bound could not be statically calculated",
        0x4Fu);
      sub_143AA50(a7, (__int64)&v285);
      v82 = v287;
      v285 = &unk_49ECF68;
      v83 = &v287[11 * v288];
      if ( v287 == v83 )
        goto LABEL_142;
      do
      {
        v83 -= 11;
        v84 = (_QWORD *)v83[4];
        if ( v84 != v83 + 6 )
          j_j___libc_free_0(v84, v83[6] + 1LL);
        if ( (_QWORD *)*v83 != v83 + 2 )
          j_j___libc_free_0(*v83, v83[2] + 1LL);
      }
      while ( v82 != v83 );
      goto LABEL_141;
    }
    v80 = a9;
  }
  if ( v14[9] < v80 )
  {
    v89 = sub_15E0530(*a7);
    if ( !sub_1602790(v89) )
    {
      v197 = sub_15E0530(*a7);
      v198 = sub_16033E0(v197);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v198 + 48LL))(v198) )
        goto LABEL_144;
    }
    v243 = v271[0];
    sub_15C9090((__int64)&v281, &v270);
    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v243);
    sub_15CAB20((__int64)&v285, "      Failed : trip count ", 0x1Au);
    sub_15C9C50((__int64)&v281, "FullUnrollTripCount", 19, v80);
    v90 = sub_17C2270((__int64)&v285, (__int64)&v281);
    sub_15CAB20(v90, " exceeds threshold ", 0x13u);
    sub_15C9C50((__int64)v277, "FullUnrollTripCount", 19, v14[9]);
    sub_17C2270(v90, (__int64)v277);
    if ( v279 != v280 )
      j_j___libc_free_0(v279, v280[0] + 1LL);
    if ( (_QWORD *)v277[0] != v278 )
      j_j___libc_free_0(v277[0], v278[0] + 1LL);
    if ( v283 != v284 )
      j_j___libc_free_0(v283, v284[0] + 1LL);
    if ( (_QWORD *)v281.m128i_i64[0] != v282 )
      j_j___libc_free_0(v281.m128i_i64[0], v282[0] + 1LL);
    sub_143AA50(a7, (__int64)&v285);
    v91 = v287;
    v285 = &unk_49ECF68;
    v83 = &v287[11 * v288];
    if ( v287 == v83 )
      goto LABEL_142;
    do
    {
      v83 -= 11;
      v92 = (_QWORD *)v83[4];
      if ( v92 != v83 + 6 )
        j_j___libc_free_0(v92, v83[6] + 1LL);
      if ( (_QWORD *)*v83 != v83 + 2 )
        j_j___libc_free_0(*v83, v83[2] + 1LL);
    }
    while ( v91 != v83 );
    goto LABEL_141;
  }
  if ( v268 * *v14 > (unsigned int)v14[10] + v80 * (unsigned __int64)(a11 - v14[10]) )
    goto LABEL_300;
  sub_19B9A90((__int64)&v285, v13, v80, a5, a6, a2, v14[1] * *v14 / 0x64u);
  if ( !v286 )
  {
LABEL_197:
    v100 = sub_15E0530(*a7);
    if ( !sub_1602790(v100) )
    {
      v195 = sub_15E0530(*a7);
      v196 = sub_16033E0(v195);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v196 + 48LL))(v196) )
        goto LABEL_144;
    }
    v247 = v271[0];
    sub_15C9090((__int64)&v281, &v270);
    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v247);
    sub_15CAB20((__int64)&v285, "      Failed : estimated unrolled loop size ", 0x2Cu);
    sub_15C9D40(
      (__int64)&v281,
      "UnrolledLoopSize",
      16,
      (unsigned int)v14[10] + (unsigned int)v14[5] * (unsigned __int64)(a11 - v14[10]));
    v101 = sub_17C2270((__int64)&v285, (__int64)&v281);
    sub_15CAB20(v101, " exceeds threshold ", 0x13u);
    sub_15C9C50((__int64)v277, "Threshold", 9, v268 * *v14);
    sub_17C2270(v101, (__int64)v277);
    if ( v279 != v280 )
      j_j___libc_free_0(v279, v280[0] + 1LL);
    if ( (_QWORD *)v277[0] != v278 )
      j_j___libc_free_0(v277[0], v278[0] + 1LL);
    if ( v283 != v284 )
      j_j___libc_free_0(v283, v284[0] + 1LL);
    if ( (_QWORD *)v281.m128i_i64[0] != v282 )
      j_j___libc_free_0(v281.m128i_i64[0], v282[0] + 1LL);
    sub_143AA50(a7, (__int64)&v285);
    v102 = v287;
    v285 = &unk_49ECF68;
    v83 = &v287[11 * v288];
    if ( v287 == v83 )
    {
LABEL_142:
      if ( v83 != (_QWORD *)v289 )
        _libc_free((unsigned __int64)v83);
LABEL_144:
      v85 = sub_15E0530(*a7);
      if ( sub_1602790(v85)
        || (v131 = sub_15E0530(*a7),
            v132 = sub_16033E0(v131),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v132 + 48LL))(v132)) )
      {
        v246 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v246);
        sub_15CAB20((__int64)&v285, "    Trying loop peeling (4th priority)...", 0x29u);
        sub_143AA50(a7, (__int64)&v285);
        v86 = v287;
        v285 = &unk_49ECF68;
        v87 = &v287[11 * v288];
        if ( v287 != v87 )
        {
          do
          {
            v87 -= 11;
            v88 = (_QWORD *)v87[4];
            if ( v88 != v87 + 6 )
              j_j___libc_free_0(v88, v87[6] + 1LL);
            if ( (_QWORD *)*v87 != v87 + 2 )
              j_j___libc_free_0(*v87, v87[2] + 1LL);
          }
          while ( v86 != v87 );
          v87 = v287;
        }
        if ( v87 != (_QWORD *)v289 )
          _libc_free((unsigned __int64)v87);
      }
      sub_1B0B080(v13, a11, v14, a8, a5);
      if ( v14[6] )
      {
        *((_BYTE *)v14 + 45) = 0;
        v14[5] = 1;
        goto LABEL_56;
      }
      v115 = sub_15E0530(*a7);
      if ( sub_1602790(v115)
        || (v165 = sub_15E0530(*a7),
            v166 = sub_16033E0(v165),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v166 + 48LL))(v166)) )
      {
        v248 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v248);
        sub_15CAB20((__int64)&v285, "      Failed : cannot do loop peeling", 0x25u);
        sub_143AA50(a7, (__int64)&v285);
        v116 = v287;
        v285 = &unk_49ECF68;
        v117 = &v287[11 * v288];
        if ( v287 != v117 )
        {
          do
          {
            v117 -= 11;
            v118 = (_QWORD *)v117[4];
            if ( v118 != v117 + 6 )
              j_j___libc_free_0(v118, v117[6] + 1LL);
            if ( (_QWORD *)*v117 != v117 + 2 )
              j_j___libc_free_0(*v117, v117[2] + 1LL);
          }
          while ( v116 != v117 );
          v117 = v287;
        }
        if ( v117 != (_QWORD *)v289 )
          _libc_free((unsigned __int64)v117);
      }
      v119 = sub_15E0530(*a7);
      if ( sub_1602790(v119)
        || (v163 = sub_15E0530(*a7),
            v164 = sub_16033E0(v163),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v164 + 48LL))(v164)) )
      {
        v249 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v249);
        sub_15CAB20((__int64)&v285, "    Trying static partial unrolling (5th priority)...", 0x35u);
        sub_143AA50(a7, (__int64)&v285);
        v120 = v287;
        v285 = &unk_49ECF68;
        v121 = &v287[11 * v288];
        if ( v287 != v121 )
        {
          do
          {
            v121 -= 11;
            v122 = (_QWORD *)v121[4];
            if ( v122 != v121 + 6 )
              j_j___libc_free_0(v122, v121[6] + 1LL);
            if ( (_QWORD *)*v121 != v121 + 2 )
              j_j___libc_free_0(*v121, v121[2] + 1LL);
          }
          while ( v120 != v121 );
          v121 = v287;
        }
        if ( v121 != (_QWORD *)v289 )
          _libc_free((unsigned __int64)v121);
      }
      if ( *a8 )
      {
        v123 = *((_BYTE *)v14 + 44) | v265;
        *((_BYTE *)v14 + 44) = v123;
        if ( v123 )
        {
          if ( *(_QWORD *)(v13 + 16) != *(_QWORD *)(v13 + 8) )
          {
            v14[5] = 0;
            v124 = sub_15E0530(*a7);
            if ( sub_1602790(v124)
              || (v212 = sub_15E0530(*a7),
                  v213 = sub_16033E0(v212),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v213 + 48LL))(v213)) )
            {
              v125 = v271[0];
              sub_15C9090((__int64)&v281, &v270);
              sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v125);
              sub_15CAB20((__int64)&v285, "      Failed : not an innermost loop", 0x24u);
              sub_143AA50(a7, (__int64)&v285);
              v126 = v287;
              v285 = &unk_49ECF68;
              v127 = &v287[11 * v288];
              if ( v287 != v127 )
              {
                do
                {
                  v127 -= 11;
                  v128 = (_QWORD *)v127[4];
                  if ( v128 != v127 + 6 )
                    j_j___libc_free_0(v128, v127[6] + 1LL);
                  if ( (_QWORD *)*v127 != v127 + 2 )
                    j_j___libc_free_0(*v127, v127[2] + 1LL);
                }
                while ( v126 != v127 );
                v127 = v287;
              }
              if ( v127 != (_QWORD *)v289 )
                _libc_free((unsigned __int64)v127);
            }
            goto LABEL_262;
          }
          if ( !v14[5] )
            v14[5] = *a8;
          v167 = v14[3];
          v168 = v14[8];
          if ( v167 == -1 )
          {
            v177 = *a8;
            v14[5] = *a8;
LABEL_356:
            v123 = 0;
            if ( v168 < v177 )
              v14[5] = v168;
LABEL_358:
            if ( v253 || v239 )
            {
              if ( *a8 )
              {
                if ( v14[5] != *a8 && !v123 )
                {
                  v178 = sub_15E0530(*a7);
                  if ( sub_1602790(v178)
                    || (v233 = sub_15E0530(*a7),
                        v234 = sub_16033E0(v233),
                        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v234 + 48LL))(v234)) )
                  {
                    v179 = v271[0];
                    sub_15C9090((__int64)&v281, &v270);
                    sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v179);
                    sub_15CAB20(
                      (__int64)&v285,
                      "      Warning : unable to fully unroll loop as directed by unroll pragma ",
                      0x49u);
                    sub_15CAB20((__int64)&v285, "or unroll(enable) pragma because unrolled size is too large", 0x3Bu);
                    sub_143AA50(a7, (__int64)&v285);
                    v285 = &unk_49ECF68;
                    sub_1897B80((__int64)&v287);
                  }
                }
              }
            }
            goto LABEL_56;
          }
          v169 = (unsigned int)v14[10];
          v170 = v167;
          v171 = a11 - (unsigned int)v169;
          v172 = (unsigned int)v14[5];
          if ( v167 < (unsigned __int64)(v169 + v171 * v172) )
          {
            v173 = v169 + 1;
            if ( (int)v169 + 1 < v167 )
              v173 = v14[3];
            v174 = (v173 - (unsigned int)v169) / (a11 - (unsigned int)v169);
            v14[5] = v174;
            LODWORD(v172) = v174;
          }
          if ( v168 < (unsigned int)v172 )
          {
            v14[5] = v168;
            LODWORD(v172) = v168;
          }
          if ( (_DWORD)v172 )
          {
            while ( 1 )
            {
              v175 = *a8 % (unsigned int)v172;
              v176 = v172;
              LODWORD(v172) = v172 - 1;
              if ( !(v175 | v176 & (unsigned int)v172) )
                break;
              v14[5] = v172;
              if ( !(_DWORD)v172 )
                goto LABEL_410;
            }
            if ( !*((_BYTE *)v14 + 46) || v176 > 1 )
            {
LABEL_355:
              v177 = v14[5];
              if ( v177 > 1 )
                goto LABEL_356;
LABEL_414:
              if ( v250 )
              {
                v225 = sub_15E0530(*a7);
                if ( sub_1602790(v225)
                  || (v227 = sub_15E0530(*a7),
                      v228 = sub_16033E0(v227),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v228 + 48LL))(v228)) )
                {
                  v261 = v271[0];
                  sub_15C9090((__int64)&v281, &v270);
                  sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v261);
                  sub_15CAB20((__int64)&v285, "      Failed : estimated unrolled loop size ", 0x2Cu);
                  sub_15C9D40(
                    (__int64)&v281,
                    "UnrolledLoopSize",
                    16,
                    (unsigned int)v14[10] + (unsigned int)v14[5] * (unsigned __int64)(a11 - v14[10]));
                  v226 = sub_17C2270((__int64)&v285, (__int64)&v281);
                  sub_15CAB20(v226, " is too large", 0xDu);
                  if ( v283 != v284 )
                    j_j___libc_free_0(v283, v284[0] + 1LL);
                  sub_2240A30(&v281);
                  sub_143AA50(a7, (__int64)&v285);
                  v285 = &unk_49ECF68;
                  sub_1897B80((__int64)&v287);
                }
              }
              else
              {
                v123 = 0;
              }
              v14[5] = 0;
              goto LABEL_358;
            }
          }
          else
          {
LABEL_410:
            if ( !*((_BYTE *)v14 + 46) )
              goto LABEL_414;
          }
          v204 = v14[7];
          v14[5] = v204;
          if ( !v204 )
            goto LABEL_414;
          while ( v170 < v169 + v171 * (unsigned __int64)v204 )
          {
            v204 >>= 1;
            v14[5] = v204;
            if ( !v204 )
              goto LABEL_414;
          }
          goto LABEL_355;
        }
        v14[5] = 0;
        v200 = sub_15E0530(*a7);
        if ( !sub_1602790(v200) )
        {
          v214 = sub_15E0530(*a7);
          v215 = sub_16033E0(v214);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v215 + 48LL))(v215) )
          {
LABEL_262:
            v265 = 0;
            goto LABEL_56;
          }
        }
        v201 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v201);
        v158 = "      Failed : static partial unrolling is disabled";
        v159 = 51;
LABEL_328:
        sub_15CAB20((__int64)&v285, v158, v159);
        sub_143AA50(a7, (__int64)&v285);
        v285 = &unk_49ECF68;
        sub_1897B80((__int64)&v287);
        goto LABEL_262;
      }
      v147 = sub_15E0530(*a7);
      if ( sub_1602790(v147)
        || (v208 = sub_15E0530(*a7),
            v209 = sub_16033E0(v208),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v209 + 48LL))(v209)) )
      {
        v256 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v256);
        sub_15CAB20((__int64)&v285, "      Failed : trip count could not be statically calculated", 0x3Cu);
        sub_143AA50(a7, (__int64)&v285);
        v148 = v287;
        v285 = &unk_49ECF68;
        v149 = &v287[11 * v288];
        if ( v287 != v149 )
        {
          do
          {
            v149 -= 11;
            v150 = (_QWORD *)v149[4];
            if ( v150 != v149 + 6 )
              j_j___libc_free_0(v150, v149[6] + 1LL);
            if ( (_QWORD *)*v149 != v149 + 2 )
              j_j___libc_free_0(*v149, v149[2] + 1LL);
          }
          while ( v148 != v149 );
          v149 = v287;
        }
        if ( v149 != (_QWORD *)v289 )
          _libc_free((unsigned __int64)v149);
      }
      if ( v240 )
      {
        v205 = sub_15E0530(*a7);
        if ( sub_1602790(v205)
          || (v219 = sub_15E0530(*a7),
              v220 = sub_16033E0(v219),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v220 + 48LL))(v220)) )
        {
          v258 = v271[0];
          sub_15C9090((__int64)&v281, &v270);
          sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v258);
          sub_15CAB20((__int64)&v285, "    Warning : unable to fully unroll loop as directed by unroll(full) ", 0x46u);
          sub_15CAB20((__int64)&v285, "pragma because loop has a runtime trip count", 0x2Cu);
          sub_143AA50(a7, (__int64)&v285);
          v285 = &unk_49ECF68;
          sub_1897B80((__int64)&v287);
        }
      }
      v151 = sub_15E0530(*a7);
      if ( sub_1602790(v151)
        || (v206 = sub_15E0530(*a7),
            v207 = sub_16033E0(v206),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v207 + 48LL))(v207)) )
      {
        v257 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v257);
        sub_15CAB20((__int64)&v285, "    Trying runtime unrolling (6th priority)...", 0x2Eu);
        sub_143AA50(a7, (__int64)&v285);
        v152 = v287;
        v285 = &unk_49ECF68;
        v153 = &v287[11 * v288];
        if ( v287 != v153 )
        {
          do
          {
            v153 -= 11;
            v154 = (_QWORD *)v153[4];
            if ( v154 != v153 + 6 )
              j_j___libc_free_0(v154, v153[6] + 1LL);
            if ( (_QWORD *)*v153 != v153 + 2 )
              j_j___libc_free_0(*v153, v153[2] + 1LL);
          }
          while ( v152 != v153 );
          v153 = v287;
        }
        if ( v153 != (_QWORD *)v289 )
          _libc_free((unsigned __int64)v153);
      }
      v155 = sub_13FD000(v13);
      if ( v155 && sub_1AFD990(v155, "llvm.loop.unroll.runtime.disable", 32) )
      {
        v14[5] = 0;
        v156 = sub_15E0530(*a7);
        if ( !sub_1602790(v156) )
        {
          v221 = sub_15E0530(*a7);
          v222 = sub_16033E0(v221);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v222 + 48LL))(v222) )
            goto LABEL_262;
        }
        v157 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v157);
        v158 = "      Failed : runtime unrolling is disabled by pragma";
        v159 = 54;
        goto LABEL_328;
      }
      sub_15E44B0(*(_QWORD *)(**(_QWORD **)(v13 + 32) + 56LL));
      if ( v160 )
      {
        sub_1B18810(v277, v13);
        if ( BYTE4(v277[0]) )
        {
          if ( LODWORD(v277[0]) < dword_4FB2680 )
          {
            v231 = sub_15E0530(*a7);
            if ( !sub_1602790(v231) )
            {
              v235 = sub_15E0530(*a7);
              v236 = sub_16033E0(v235);
              if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v236 + 48LL))(v236) )
                goto LABEL_262;
            }
            v232 = v271[0];
            sub_15C9090((__int64)&v281, &v270);
            sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v232);
            v158 = "      Failed : runtime trip count is too small";
            v159 = 46;
            goto LABEL_328;
          }
          *((_BYTE *)v14 + 47) = 1;
        }
      }
      if ( v250 || v269 || v260 > 0 )
      {
        *((_BYTE *)v14 + 45) = 1;
      }
      else if ( !*((_BYTE *)v14 + 45) )
      {
        v14[5] = 0;
        v161 = sub_15E0530(*a7);
        if ( !sub_1602790(v161) )
        {
          v223 = sub_15E0530(*a7);
          v224 = sub_16033E0(v223);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v224 + 48LL))(v224) )
            goto LABEL_262;
        }
        v162 = v271[0];
        sub_15C9090((__int64)&v281, &v270);
        sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v162);
        v158 = "      Failed : runtime unrolling is disabled";
        v159 = 44;
        goto LABEL_328;
      }
      v285 = (void *)v13;
      v281.m128i_i32[0] = a11;
      v184 = *(_QWORD **)v13;
      if ( !*(_QWORD *)v13 )
        goto LABEL_431;
      v185 = 1;
      do
      {
        v184 = (_QWORD *)*v184;
        ++v185;
      }
      while ( v184 );
      if ( v185 <= 1 )
      {
LABEL_431:
        v216 = sub_1474260(a5, v13);
        if ( !sub_14562D0(v216) && !*(_WORD *)(v216 + 24) )
        {
          v217 = *(_QWORD *)(v216 + 32);
          v218 = *(_QWORD **)(v217 + 24);
          if ( *(_DWORD *)(v217 + 32) > 0x40u )
            v218 = (_QWORD *)*v218;
          if ( (unsigned int)dword_4FB3480 > (unsigned __int64)v218 )
          {
            sub_19B6500(a7, (__int64 *)&v285);
            goto LABEL_428;
          }
        }
        v13 = (__int64)v285;
      }
      if ( *(_QWORD *)(v13 + 16) == *(_QWORD *)(v13 + 8) )
      {
        if ( (unsigned int)dword_4FB3560 >= v281.m128i_i32[0] )
        {
          v186 = v14[5];
          if ( !v186 )
          {
            v186 = v14[7];
            v14[5] = v186;
            if ( !v186 )
            {
LABEL_392:
              v14[5] = 0;
              goto LABEL_56;
            }
          }
          v187 = (unsigned int)v14[10];
          v188 = (unsigned int)v14[3];
          v189 = a11 - (unsigned int)v187;
          while ( v188 < v187 + v189 * (unsigned __int64)v186 )
          {
            v186 >>= 1;
            v14[5] = v186;
            if ( !v186 )
              goto LABEL_392;
          }
          if ( *((_BYTE *)v14 + 46) || !(*a10 % v186) )
            goto LABEL_389;
          v190 = v14[5];
          if ( v190 )
          {
            for ( i = *a10; i % v190; i = *a10 )
            {
              v190 >>= 1;
              v14[5] = v190;
              if ( !v190 )
                goto LABEL_452;
            }
            if ( !v269 )
              goto LABEL_389;
          }
          else
          {
LABEL_452:
            if ( !v269 )
            {
              v192 = v14[5];
              goto LABEL_391;
            }
          }
          v229 = sub_15E0530(*a7);
          if ( sub_1602790(v229)
            || (v237 = sub_15E0530(*a7),
                v238 = sub_16033E0(v237),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v238 + 48LL))(v238)) )
          {
            v230 = v271[0];
            sub_15C9090((__int64)&v281, &v270);
            sub_15CA330((__int64)&v285, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v281, v230);
            sub_15CAB20((__int64)&v285, "      Warning : remainder loops not allowed", 0x2Bu);
            sub_143AA50(a7, (__int64)&v285);
            v285 = &unk_49ECF68;
            sub_1897B80((__int64)&v287);
          }
LABEL_389:
          v192 = v14[8];
          if ( v14[5] <= v192 )
            v192 = v14[5];
          else
            v14[5] = v192;
LABEL_391:
          if ( v192 > 1 )
            goto LABEL_56;
          goto LABEL_392;
        }
        sub_19B7D80(a7, (__int64 *)&v285, (unsigned int *)&v281);
      }
      else
      {
        sub_19B6370(a7, (__int64 *)&v285);
      }
LABEL_428:
      v14[5] = 0;
      goto LABEL_262;
    }
    do
    {
      v83 -= 11;
      v103 = (_QWORD *)v83[4];
      if ( v103 != v83 + 6 )
        j_j___libc_free_0(v103, v83[6] + 1LL);
      if ( (_QWORD *)*v83 != v83 + 2 )
        j_j___libc_free_0(*v83, v83[2] + 1LL);
    }
    while ( v102 != v83 );
LABEL_141:
    v83 = v287;
    goto LABEL_142;
  }
  v98 = v14[1];
  if ( HIDWORD(v285) > 0x28F5C27 )
  {
    v98 = 100;
  }
  else if ( (_DWORD)v285 && v98 > 100 * HIDWORD(v285) / (unsigned int)v285 )
  {
    v98 = 100 * HIDWORD(v285) / (unsigned int)v285;
  }
  v99 = *v14;
  if ( !v240 && v269 <= 1 )
  {
    if ( (unsigned int)v285 < v98 * v99 / 0x64 )
      goto LABEL_300;
    goto LABEL_197;
  }
  if ( (unsigned int)v285 >= v98 * v99 / 0x64 && (v99 >= HIDWORD(v285) || v98 <= 0x64) )
    goto LABEL_197;
LABEL_300:
  *a13 = a9 == v80;
  *a8 = v80;
  v146 = 1;
  if ( !*((_BYTE *)v14 + 49) )
    v146 = *a10;
  *a10 = v146;
LABEL_56:
  if ( v270 )
    sub_161E7C0((__int64)&v270, v270);
  return v265;
}
