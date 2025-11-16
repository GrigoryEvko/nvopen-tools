// Function: sub_19C97B0
// Address: 0x19c97b0
//
__int64 __fastcall sub_19C97B0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r14
  unsigned __int64 *v14; // r13
  unsigned __int64 v15; // rdx
  _QWORD *v16; // r15
  _QWORD *v17; // r12
  __int64 v18; // rax
  _QWORD *v20; // r12
  _QWORD *m; // r14
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rcx
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // r14
  unsigned __int64 *v45; // r12
  unsigned __int64 v46; // rdx
  _QWORD *v47; // r15
  _QWORD *v48; // r13
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 *v54; // r14
  __int64 *v55; // r13
  __int64 v56; // rbx
  int v57; // eax
  __int64 v58; // rsi
  int v59; // edi
  __int64 v60; // rcx
  unsigned int v61; // edx
  __int64 *v62; // rax
  __int64 v63; // r8
  __int64 v64; // r12
  __int64 v65; // rax
  __int64 v66; // rsi
  __int64 v67; // r12
  __m128i v68; // xmm4
  __m128i v69; // xmm5
  char *v70; // r14
  int v71; // eax
  int v72; // r9d
  _QWORD *v73; // r14
  _QWORD *i; // r15
  __int64 v75; // rax
  char *v76; // rbx
  char *v77; // rdi
  _QWORD *v78; // rbx
  _QWORD *v79; // r12
  _QWORD *v80; // rdi
  __int64 v81; // r13
  _QWORD *v82; // rax
  __int64 v83; // r12
  unsigned int v84; // r15d
  __int64 v85; // r12
  _QWORD *v86; // rsi
  _QWORD *v87; // rax
  __int64 v88; // r12
  __int64 v89; // rsi
  __int64 v90; // rbx
  __int64 v91; // r13
  __int64 v92; // r15
  __int64 v93; // r12
  __int64 v94; // rsi
  __int64 v95; // r12
  __int64 *v96; // r13
  __int64 v97; // rax
  __m128i v98; // xmm2
  __m128i v99; // xmm3
  char *v100; // r14
  _QWORD *v101; // rbx
  _QWORD *v102; // r12
  _QWORD *v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rax
  unsigned __int64 v106; // rsi
  unsigned __int8 v107; // al
  __int64 v108; // r12
  __int64 v109; // rsi
  __int64 v110; // rax
  unsigned int v111; // eax
  int v112; // eax
  __int64 *v113; // rsi
  __int64 v114; // r9
  __m128i v115; // xmm1
  double v116; // xmm0_8
  __int64 v117; // rsi
  __int64 v118; // rax
  __int64 v119; // rsi
  char v120; // dl
  char v121; // r8
  char v122; // di
  unsigned __int8 v123; // cl
  __int64 **v124; // rdx
  __int64 v125; // r11
  __int64 *v126; // rdx
  char *v127; // rbx
  char *v128; // rdi
  __int64 v129; // rax
  __int64 v130; // rsi
  __int64 v131; // r12
  __m128i v132; // xmm5
  __m128i v133; // xmm6
  char *v134; // r14
  _QWORD *v135; // rbx
  _QWORD *v136; // rdi
  __int64 v137; // rax
  __int64 v138; // rsi
  __int64 v139; // r13
  __m128i v140; // xmm4
  char *v141; // r13
  char *v142; // r12
  char *v143; // rbx
  char *v144; // rdi
  _QWORD *v145; // r12
  _QWORD *v146; // r13
  _QWORD *v147; // rdi
  __int64 *v148; // rdx
  __int64 v149; // rax
  __int64 v150; // rdx
  int v151; // eax
  double v152; // xmm4_8
  double v153; // xmm5_8
  double v154; // xmm4_8
  double v155; // xmm5_8
  double v156; // xmm4_8
  double v157; // xmm5_8
  __int64 v158; // rax
  unsigned __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // r13
  __int64 v162; // rax
  __int64 v163; // rbx
  __int64 v164; // rdi
  __int64 v165; // rax
  __int64 *v166; // r14
  __int64 *v167; // r15
  __int64 *v168; // rax
  __int64 v169; // rax
  __int64 v170; // rdi
  __int64 v171; // r14
  __int64 v172; // rax
  __int64 v173; // r12
  int v174; // r8d
  int v175; // r9d
  __int64 v176; // rax
  __int64 v177; // rax
  __int64 *v178; // r15
  unsigned __int32 v179; // r14d
  __int64 *v180; // rax
  __int64 v181; // rax
  __int64 v182; // r14
  _QWORD *v183; // rbx
  _QWORD *v184; // r12
  unsigned __int64 v185; // rdi
  _QWORD *v186; // r12
  __int64 v187; // rax
  unsigned __int64 v188; // r13
  __int64 v189; // rdx
  __int64 v190; // rax
  _QWORD *v191; // rbx
  __int64 v192; // r12
  __int64 v193; // rsi
  __int64 v194; // rax
  __int64 v195; // rax
  char *v196; // rbx
  char *v197; // rdi
  __int64 v198; // rax
  __int64 v199; // rax
  __int64 v200; // rax
  __int64 v201; // rax
  __int64 *v202; // rbx
  __int64 v203; // rax
  __int64 v204; // rsi
  __int64 v205; // r12
  __int64 v206; // r14
  __int64 v207; // r14
  char *v208; // r8
  char *v209; // rbx
  char *v210; // r12
  char *v211; // rdi
  _QWORD *v212; // rbx
  _QWORD *v213; // rdi
  __int64 v214; // rax
  __int64 v215; // rax
  __int64 v216; // rax
  __int64 v217; // rsi
  __int64 v218; // r12
  __int64 v219; // r12
  __int64 v220; // rbx
  char *v221; // r8
  char *v222; // r14
  char *v223; // rbx
  char *v224; // rdi
  _QWORD *v225; // rbx
  _QWORD *v226; // rdi
  __int64 v227; // rax
  __int64 v228; // rax
  int v229; // [rsp+10h] [rbp-540h]
  __int64 v230; // [rsp+18h] [rbp-538h]
  int v231; // [rsp+18h] [rbp-538h]
  __int64 v232; // [rsp+18h] [rbp-538h]
  unsigned __int8 v233; // [rsp+27h] [rbp-529h]
  char v234; // [rsp+28h] [rbp-528h]
  __int64 j; // [rsp+28h] [rbp-528h]
  __int64 v237; // [rsp+38h] [rbp-518h]
  __int64 v238; // [rsp+38h] [rbp-518h]
  __int64 *v239; // [rsp+38h] [rbp-518h]
  __int64 v240; // [rsp+38h] [rbp-518h]
  __int64 v241; // [rsp+38h] [rbp-518h]
  __int64 *v242; // [rsp+38h] [rbp-518h]
  __int64 k; // [rsp+38h] [rbp-518h]
  __int64 *v244; // [rsp+38h] [rbp-518h]
  __int64 v245; // [rsp+48h] [rbp-508h] BYREF
  __m128i v246[2]; // [rsp+50h] [rbp-500h] BYREF
  __m128i v247; // [rsp+70h] [rbp-4E0h] BYREF
  _QWORD v248[2]; // [rsp+80h] [rbp-4D0h] BYREF
  _QWORD *v249; // [rsp+90h] [rbp-4C0h]
  _QWORD v250[6]; // [rsp+A0h] [rbp-4B0h] BYREF
  __m128i v251; // [rsp+D0h] [rbp-480h] BYREF
  __int64 v252; // [rsp+E0h] [rbp-470h] BYREF
  __int64 v253; // [rsp+E8h] [rbp-468h]
  _BYTE *v254; // [rsp+F0h] [rbp-460h]
  __int64 v255; // [rsp+F8h] [rbp-458h]
  _QWORD v256[2]; // [rsp+100h] [rbp-450h] BYREF
  __m128i v257; // [rsp+110h] [rbp-440h] BYREF
  __int64 v258; // [rsp+120h] [rbp-430h]
  __m128i v259; // [rsp+130h] [rbp-420h] BYREF
  __int64 v260; // [rsp+140h] [rbp-410h] BYREF
  __m128i v261; // [rsp+148h] [rbp-408h] BYREF
  __int64 v262; // [rsp+158h] [rbp-3F8h]
  __int64 v263; // [rsp+160h] [rbp-3F0h] BYREF
  _BYTE v264[24]; // [rsp+168h] [rbp-3E8h]
  __int64 v265; // [rsp+180h] [rbp-3D0h]
  _BYTE *v266; // [rsp+188h] [rbp-3C8h] BYREF
  __int64 v267; // [rsp+190h] [rbp-3C0h]
  _BYTE v268[352]; // [rsp+198h] [rbp-3B8h] BYREF
  char v269; // [rsp+2F8h] [rbp-258h]
  int v270; // [rsp+2FCh] [rbp-254h]
  __int64 v271; // [rsp+300h] [rbp-250h]
  void *v272; // [rsp+310h] [rbp-240h] BYREF
  __int64 v273; // [rsp+318h] [rbp-238h] BYREF
  __int64 v274; // [rsp+320h] [rbp-230h]
  __m128i v275; // [rsp+328h] [rbp-228h] BYREF
  __int64 v276; // [rsp+338h] [rbp-218h]
  __int64 v277; // [rsp+340h] [rbp-210h]
  __m128i v278; // [rsp+348h] [rbp-208h] BYREF
  __int64 v279; // [rsp+358h] [rbp-1F8h]
  char v280; // [rsp+360h] [rbp-1F0h]
  char *v281; // [rsp+368h] [rbp-1E8h] BYREF
  unsigned __int64 v282; // [rsp+370h] [rbp-1E0h]
  char v283; // [rsp+378h] [rbp-1D8h] BYREF
  char v284; // [rsp+380h] [rbp-1D0h] BYREF
  void *v285; // [rsp+3C0h] [rbp-190h]
  char *v286; // [rsp+3E8h] [rbp-168h]
  char v287; // [rsp+3F8h] [rbp-158h] BYREF
  _QWORD *v288; // [rsp+480h] [rbp-D0h]
  unsigned int v289; // [rsp+490h] [rbp-C0h]
  __int64 v290; // [rsp+4A0h] [rbp-B0h]
  __int64 v291; // [rsp+4C0h] [rbp-90h]
  char v292; // [rsp+4D8h] [rbp-78h]
  int v293; // [rsp+4DCh] [rbp-74h]
  __int64 v294; // [rsp+4E0h] [rbp-70h]

  v10 = a2;
  if ( (unsigned __int8)sub_1404700(a1, a2) )
    goto LABEL_2;
  v23 = *(__int64 **)(a1 + 8);
  v24 = *v23;
  v25 = v23[1];
  if ( v24 == v25 )
LABEL_422:
    BUG();
  while ( *(_UNKNOWN **)v24 != &unk_4F96DB4 )
  {
    v24 += 16;
    if ( v25 == v24 )
      goto LABEL_422;
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(*(_QWORD *)(v24 + 8), &unk_4F96DB4);
  v27 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 160) = *(_QWORD *)(v26 + 160);
  v28 = *v27;
  v29 = v27[1];
  if ( v28 == v29 )
LABEL_424:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F9A488 )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_424;
  }
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_4F9A488);
  v31 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 168) = *(_QWORD *)(v30 + 160);
  v32 = *v31;
  v33 = v31[1];
  if ( v32 == v33 )
LABEL_423:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_5051F8C )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_423;
  }
  v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(*(_QWORD *)(v32 + 8), &unk_5051F8C);
  v35 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 176) = v34;
  v36 = *v35;
  v37 = v35[1];
  if ( v36 == v37 )
LABEL_419:
    BUG();
  while ( *(_UNKNOWN **)v36 != &unk_4F99CB0 )
  {
    v36 += 16;
    if ( v37 == v36 )
      goto LABEL_419;
  }
  v38 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v36 + 8) + 104LL))(
                      *(_QWORD *)(v36 + 8),
                      &unk_4F99CB0)
                  + 160);
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 232) = v38;
  *(_QWORD *)(a1 + 192) = a2;
  v39 = sub_22077B0(72);
  if ( v39 )
  {
    *(_QWORD *)(v39 + 24) = 0;
    *(_QWORD *)(v39 + 32) = 0;
    v40 = *(_QWORD *)(a1 + 160);
    *(_QWORD *)(v39 + 40) = 0;
    *(_DWORD *)(v39 + 48) = 0;
    *(_QWORD *)v39 = v40;
    *(_QWORD *)(v39 + 16) = v39 + 8;
    *(_DWORD *)(v39 + 56) = 0;
    *(_QWORD *)(v39 + 8) = (v39 + 8) | 4;
    *(_QWORD *)(v39 + 64) = 0;
  }
  v41 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 200) = v39;
  v238 = v41;
  if ( v41 )
  {
    sub_1359CD0(v41);
    if ( *(_DWORD *)(v41 + 48) )
    {
      sub_1359800(&v259, -8, 0);
      sub_1359800(&v272, -16, 0);
      v73 = *(_QWORD **)(v41 + 32);
      for ( i = &v73[6 * *(unsigned int *)(v41 + 48)]; i != v73; v73 += 6 )
      {
        v75 = v73[3];
        *v73 = &unk_49EE2B0;
        if ( v75 != 0 && v75 != -8 && v75 != -16 )
          sub_1649B30(v73 + 1);
      }
      v272 = &unk_49EE2B0;
      if ( v275.m128i_i64[0] != 0 && v275.m128i_i64[0] != -8 && v275.m128i_i64[0] != -16 )
        sub_1649B30(&v273);
      v259.m128i_i64[0] = (__int64)&unk_49EE2B0;
      if ( v261.m128i_i64[0] != -8 && v261.m128i_i64[0] != 0 && v261.m128i_i64[0] != -16 )
        sub_1649B30(&v259.m128i_i64[1]);
    }
    j___libc_free_0(*(_QWORD *)(v41 + 32));
    v42 = v41 + 8;
    v43 = *(unsigned __int64 **)(v41 + 16);
    if ( (unsigned __int64 *)(v238 + 8) != v43 )
    {
      v44 = (unsigned __int64 *)v42;
      do
      {
        v45 = v43;
        v43 = (unsigned __int64 *)v43[1];
        v46 = *v45 & 0xFFFFFFFFFFFFFFF8LL;
        *v43 = v46 | *v43 & 7;
        *(_QWORD *)(v46 + 8) = v43;
        v47 = (_QWORD *)v45[6];
        v48 = (_QWORD *)v45[5];
        *v45 &= 7u;
        v45[1] = 0;
        if ( v47 != v48 )
        {
          do
          {
            v49 = v48[2];
            if ( v49 != -8 && v49 != 0 && v49 != -16 )
              sub_1649B30(v48);
            v48 += 3;
          }
          while ( v47 != v48 );
          v48 = (_QWORD *)v45[5];
        }
        if ( v48 )
          j_j___libc_free_0(v48, v45[7] - (_QWORD)v48);
        j_j___libc_free_0(v45, 72);
      }
      while ( v44 != v43 );
      v10 = a2;
    }
    j_j___libc_free_0(v238, 72);
  }
  v50 = *(__int64 **)(a1 + 8);
  v51 = *v50;
  v52 = v50[1];
  if ( v51 == v52 )
LABEL_421:
    BUG();
  while ( *(_UNKNOWN **)v51 != &unk_4F9920C )
  {
    v51 += 16;
    if ( v52 == v51 )
      goto LABEL_421;
  }
  v53 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v51 + 8) + 104LL))(*(_QWORD *)(v51 + 8), &unk_4F9920C);
  v54 = *(__int64 **)(v10 + 40);
  v55 = *(__int64 **)(v10 + 32);
  v56 = v53;
  if ( v55 != v54 )
  {
    while ( 1 )
    {
      v57 = *(_DWORD *)(v56 + 184);
      if ( !v57 )
        goto LABEL_69;
      v58 = *v55;
      v59 = v57 - 1;
      v60 = *(_QWORD *)(v56 + 168);
      v61 = (v57 - 1) & (((unsigned int)*v55 >> 9) ^ ((unsigned int)*v55 >> 4));
      v62 = (__int64 *)(v60 + 16LL * v61);
      v63 = *v62;
      if ( *v55 != *v62 )
      {
        v71 = 1;
        while ( v63 != -8 )
        {
          v72 = v71 + 1;
          v61 = v59 & (v71 + v61);
          v62 = (__int64 *)(v60 + 16LL * v61);
          v63 = *v62;
          if ( v58 == *v62 )
            goto LABEL_72;
          v71 = v72;
        }
        goto LABEL_69;
      }
LABEL_72:
      if ( v10 == v62[1] )
      {
        ++v55;
        sub_135CE70(*(_QWORD *)(a1 + 200), v58);
        if ( v54 == v55 )
          break;
      }
      else
      {
LABEL_69:
        if ( v54 == ++v55 )
          break;
      }
    }
  }
  sub_1B17B90(&v272, *(_QWORD *)(a1 + 192), "llvm.loop.licm_versioning.disable", 33);
  if ( (_BYTE)v273 )
    goto LABEL_2;
  v234 = sub_13FCBF0(*(_QWORD *)(a1 + 192));
  if ( !v234 )
    goto LABEL_77;
  v64 = *(_QWORD *)(a1 + 192);
  if ( *(_QWORD *)(v64 + 16) != *(_QWORD *)(v64 + 8) )
    goto LABEL_77;
  v81 = *(_QWORD *)(**(_QWORD **)(v64 + 32) + 8LL);
  if ( !v81 )
    goto LABEL_77;
  while ( 1 )
  {
    v82 = sub_1648700(v81);
    if ( (unsigned __int8)(*((_BYTE *)v82 + 16) - 25) <= 9u )
      break;
    v81 = *(_QWORD *)(v81 + 8);
    if ( !v81 )
      goto LABEL_77;
  }
  v83 = v64 + 56;
  v84 = 0;
LABEL_121:
  v84 -= !sub_1377F70(v83, v82[5]) - 1;
  while ( 1 )
  {
    v81 = *(_QWORD *)(v81 + 8);
    if ( !v81 )
      break;
    v82 = sub_1648700(v81);
    if ( (unsigned __int8)(*((_BYTE *)v82 + 16) - 25) <= 9u )
      goto LABEL_121;
  }
  if ( v84 != 1 )
    goto LABEL_77;
  if ( !sub_13F9E70(*(_QWORD *)(a1 + 192)) )
    goto LABEL_77;
  v85 = sub_13F9E70(*(_QWORD *)(a1 + 192));
  if ( v85 != sub_13FCB50(*(_QWORD *)(a1 + 192)) )
    goto LABEL_77;
  v233 = sub_13FD440(*(_QWORD *)(a1 + 192));
  if ( v233 )
    goto LABEL_77;
  v86 = *(_QWORD **)(a1 + 192);
  v87 = (_QWORD *)*v86;
  if ( *v86 )
  {
    do
    {
      v87 = (_QWORD *)*v87;
      ++v84;
    }
    while ( v87 );
  }
  if ( *(_DWORD *)(a1 + 208) < v84
    || (v88 = sub_1481F60(*(_QWORD **)(a1 + 168), (__int64)v86, a3, a4), v88 == sub_1456E90(*(_QWORD *)(a1 + 168))) )
  {
LABEL_77:
    v239 = *(__int64 **)(a1 + 232);
    v65 = sub_15E0530(*v239);
    if ( !sub_1602790(v65) )
    {
      v104 = sub_15E0530(*v239);
      v105 = sub_16033E0(v104);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v105 + 48LL))(v105) )
        goto LABEL_2;
    }
    v66 = *(_QWORD *)(a1 + 192);
    v67 = **(_QWORD **)(v66 + 32);
    sub_13FD840(&v247, v66);
    sub_15C9090((__int64)&v251, &v247);
    sub_15CA540((__int64)&v272, (__int64)"loop-versioning-licm", (__int64)"IllegalLoopStruct", 17, &v251, v67);
    sub_15CAB20((__int64)&v272, " Unsafe Loop structure", 0x16u);
    v68 = _mm_loadu_si128(&v275);
    v69 = _mm_loadu_si128(&v278);
    v259.m128i_i32[2] = v273;
    v261 = v68;
    v259.m128i_i8[12] = BYTE4(v273);
    *(__m128i *)v264 = v69;
    v260 = v274;
    v262 = v276;
    v259.m128i_i64[0] = (__int64)&unk_49ECF68;
    v263 = v277;
    LOBYTE(v265) = v280;
    if ( v280 )
      *(_QWORD *)&v264[16] = v279;
    v266 = v268;
    v267 = 0x400000000LL;
    if ( (_DWORD)v282 )
    {
      sub_19C9520((__int64)&v266, (__int64)&v281);
      v76 = v281;
      v269 = v292;
      v270 = v293;
      v271 = v294;
      v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v272 = &unk_49ECF68;
      v70 = &v281[88 * (unsigned int)v282];
      if ( v281 != v70 )
      {
        do
        {
          v70 -= 88;
          v77 = (char *)*((_QWORD *)v70 + 4);
          if ( v77 != v70 + 48 )
            j_j___libc_free_0(v77, *((_QWORD *)v70 + 6) + 1LL);
          if ( *(char **)v70 != v70 + 16 )
            j_j___libc_free_0(*(_QWORD *)v70, *((_QWORD *)v70 + 2) + 1LL);
        }
        while ( v76 != v70 );
        v70 = v281;
      }
    }
    else
    {
      v70 = v281;
      v269 = v292;
      v270 = v293;
      v271 = v294;
      v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    if ( v70 != &v283 )
      _libc_free((unsigned __int64)v70);
    if ( v247.m128i_i64[0] )
      sub_161E7C0((__int64)&v247, v247.m128i_i64[0]);
    sub_143AA50(v239, (__int64)&v259);
    v78 = v266;
    v259.m128i_i64[0] = (__int64)&unk_49ECF68;
    v79 = &v266[88 * (unsigned int)v267];
    if ( v266 != (_BYTE *)v79 )
    {
      do
      {
        v79 -= 11;
        v80 = (_QWORD *)v79[4];
        if ( v80 != v79 + 6 )
          j_j___libc_free_0(v80, v79[6] + 1LL);
        if ( (_QWORD *)*v79 != v79 + 2 )
          j_j___libc_free_0(*v79, v79[2] + 1LL);
      }
      while ( v78 != v79 );
      goto LABEL_114;
    }
    goto LABEL_115;
  }
  v89 = *(_QWORD *)(a1 + 192);
  *(_BYTE *)(a1 + 224) = 1;
  *(_QWORD *)(a1 + 216) = 0;
  v230 = *(_QWORD *)(v89 + 40);
  if ( *(_QWORD *)(v89 + 32) == v230 )
  {
LABEL_182:
    v110 = sub_38694E0(*(_QWORD *)(a1 + 176));
    *(_QWORD *)(a1 + 184) = v110;
    v111 = *(_DWORD *)(*(_QWORD *)(v110 + 8) + 280LL);
    if ( !v111 )
      goto LABEL_2;
    if ( v111 > dword_5052308[0] )
    {
      v202 = *(__int64 **)(a1 + 232);
      v203 = sub_15E0530(*v202);
      if ( !sub_1602790(v203) )
      {
        v214 = sub_15E0530(*v202);
        v215 = sub_16033E0(v214);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v215 + 48LL))(v215, v89) )
          goto LABEL_3;
      }
      v204 = *(_QWORD *)(a1 + 192);
      v205 = **(_QWORD **)(v204 + 32);
      sub_13FD840(&v245, v204);
      sub_15C9090((__int64)v246, &v245);
      sub_15CA540((__int64)&v272, (__int64)"loop-versioning-licm", (__int64)"RuntimeCheck", 12, v246, v205);
      sub_15CAB20((__int64)&v272, "Number of runtime checks ", 0x19u);
      sub_15C9C50((__int64)&v251, "RuntimeChecks", 13, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL) + 280LL));
      v206 = sub_17C21B0((__int64)&v272, (__int64)&v251);
      sub_15CAB20(v206, " exceeds threshold ", 0x13u);
      sub_15C9C50((__int64)&v247, "Threshold", 9, dword_5052308[0]);
      v207 = sub_17C21B0(v206, (__int64)&v247);
      v259.m128i_i32[2] = *(_DWORD *)(v207 + 8);
      v259.m128i_i8[12] = *(_BYTE *)(v207 + 12);
      v260 = *(_QWORD *)(v207 + 16);
      v261 = _mm_loadu_si128((const __m128i *)(v207 + 24));
      v262 = *(_QWORD *)(v207 + 40);
      v259.m128i_i64[0] = (__int64)&unk_49ECF68;
      v263 = *(_QWORD *)(v207 + 48);
      *(__m128i *)v264 = _mm_loadu_si128((const __m128i *)(v207 + 56));
      LOBYTE(v265) = *(_BYTE *)(v207 + 80);
      if ( (_BYTE)v265 )
        *(_QWORD *)&v264[16] = *(_QWORD *)(v207 + 72);
      v266 = v268;
      v267 = 0x400000000LL;
      if ( *(_DWORD *)(v207 + 96) )
        sub_19C9520((__int64)&v266, v207 + 88);
      v269 = *(_BYTE *)(v207 + 456);
      v270 = *(_DWORD *)(v207 + 460);
      v271 = *(_QWORD *)(v207 + 464);
      v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
      if ( v249 != v250 )
        j_j___libc_free_0(v249, v250[0] + 1LL);
      if ( (_QWORD *)v247.m128i_i64[0] != v248 )
        j_j___libc_free_0(v247.m128i_i64[0], v248[0] + 1LL);
      if ( v254 != (_BYTE *)v256 )
        j_j___libc_free_0(v254, v256[0] + 1LL);
      if ( (__int64 *)v251.m128i_i64[0] != &v252 )
        j_j___libc_free_0(v251.m128i_i64[0], v252 + 1);
      v208 = v281;
      v272 = &unk_49ECF68;
      if ( v281 != &v281[88 * (unsigned int)v282] )
      {
        v244 = v202;
        v209 = v281;
        v210 = &v281[88 * (unsigned int)v282];
        do
        {
          v210 -= 88;
          v211 = (char *)*((_QWORD *)v210 + 4);
          if ( v211 != v210 + 48 )
            j_j___libc_free_0(v211, *((_QWORD *)v210 + 6) + 1LL);
          if ( *(char **)v210 != v210 + 16 )
            j_j___libc_free_0(*(_QWORD *)v210, *((_QWORD *)v210 + 2) + 1LL);
        }
        while ( v209 != v210 );
        v202 = v244;
        v208 = v281;
      }
      if ( v208 != &v283 )
        _libc_free((unsigned __int64)v208);
      if ( v245 )
        sub_161E7C0((__int64)&v245, v245);
      sub_143AA50(v202, (__int64)&v259);
      v102 = v266;
      v259.m128i_i64[0] = (__int64)&unk_49ECF68;
      v212 = &v266[88 * (unsigned int)v267];
      if ( v266 == (_BYTE *)v212 )
        goto LABEL_159;
      do
      {
        v212 -= 11;
        v213 = (_QWORD *)v212[4];
        if ( v213 != v212 + 6 )
          j_j___libc_free_0(v213, v212[6] + 1LL);
        if ( (_QWORD *)*v212 != v212 + 2 )
          j_j___libc_free_0(*v212, v212[2] + 1LL);
      }
      while ( v102 != v212 );
      goto LABEL_158;
    }
    v112 = *(_DWORD *)(a1 + 220);
    if ( !v112 || *(_BYTE *)(a1 + 224) )
    {
LABEL_2:
      v233 = 0;
      goto LABEL_3;
    }
    v115.m128i_i32[1] = 0;
    HIDWORD(v116) = 0;
    v113 = *(__int64 **)(a1 + 232);
    v242 = v113;
    v114 = *v113;
    *(float *)v115.m128i_i32 = (float)(100 * v112);
    *(float *)&v116 = (float)*(int *)(a1 + 216) * *(float *)(a1 + 212);
    if ( *(float *)&v116 > *(float *)v115.m128i_i32 )
    {
      v216 = sub_15E0530(*v113);
      if ( !sub_1602790(v216) )
      {
        v227 = sub_15E0530(*v113);
        v228 = sub_16033E0(v227);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v228 + 48LL))(v228) )
          goto LABEL_2;
      }
      v217 = *(_QWORD *)(a1 + 192);
      v218 = **(_QWORD **)(v217 + 32);
      sub_13FD840(&v245, v217);
      sub_15C9090((__int64)v246, &v245);
      sub_15CA540((__int64)&v272, (__int64)"loop-versioning-licm", (__int64)"InvariantThreshold", 18, v246, v218);
      sub_15CAB20((__int64)&v272, "Invariant load & store ", 0x17u);
      sub_15C9C50(
        (__int64)&v251,
        "LoadAndStoreCounter",
        19,
        (unsigned int)(100 * *(_DWORD *)(a1 + 220)) / *(_DWORD *)(a1 + 216));
      v219 = sub_17C21B0((__int64)&v272, (__int64)&v251);
      sub_15CAB20(v219, " are less then defined threshold ", 0x21u);
      sub_15C99E0((__int64)&v247, "Threshold", 9, *(float *)(a1 + 212));
      v220 = sub_17C21B0(v219, (__int64)&v247);
      v259.m128i_i32[2] = *(_DWORD *)(v220 + 8);
      v259.m128i_i8[12] = *(_BYTE *)(v220 + 12);
      v260 = *(_QWORD *)(v220 + 16);
      v261 = _mm_loadu_si128((const __m128i *)(v220 + 24));
      v262 = *(_QWORD *)(v220 + 40);
      v259.m128i_i64[0] = (__int64)&unk_49ECF68;
      v263 = *(_QWORD *)(v220 + 48);
      *(__m128i *)v264 = _mm_loadu_si128((const __m128i *)(v220 + 56));
      LOBYTE(v265) = *(_BYTE *)(v220 + 80);
      if ( (_BYTE)v265 )
        *(_QWORD *)&v264[16] = *(_QWORD *)(v220 + 72);
      v266 = v268;
      v267 = 0x400000000LL;
      if ( *(_DWORD *)(v220 + 96) )
        sub_19C9520((__int64)&v266, v220 + 88);
      v269 = *(_BYTE *)(v220 + 456);
      v270 = *(_DWORD *)(v220 + 460);
      v271 = *(_QWORD *)(v220 + 464);
      v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
      if ( v249 != v250 )
        j_j___libc_free_0(v249, v250[0] + 1LL);
      if ( (_QWORD *)v247.m128i_i64[0] != v248 )
        j_j___libc_free_0(v247.m128i_i64[0], v248[0] + 1LL);
      if ( v254 != (_BYTE *)v256 )
        j_j___libc_free_0(v254, v256[0] + 1LL);
      if ( (__int64 *)v251.m128i_i64[0] != &v252 )
        j_j___libc_free_0(v251.m128i_i64[0], v252 + 1);
      v221 = v281;
      v272 = &unk_49ECF68;
      v222 = v281;
      v223 = &v281[88 * (unsigned int)v282];
      if ( v281 != v223 )
      {
        do
        {
          v223 -= 88;
          v224 = (char *)*((_QWORD *)v223 + 4);
          if ( v224 != v223 + 48 )
            j_j___libc_free_0(v224, *((_QWORD *)v223 + 6) + 1LL);
          if ( *(char **)v223 != v223 + 16 )
            j_j___libc_free_0(*(_QWORD *)v223, *((_QWORD *)v223 + 2) + 1LL);
        }
        while ( v222 != v223 );
        v221 = v281;
      }
      if ( v221 != &v283 )
        _libc_free((unsigned __int64)v221);
      if ( v245 )
        sub_161E7C0((__int64)&v245, v245);
      sub_143AA50(v242, (__int64)&v259);
      v79 = v266;
      v259.m128i_i64[0] = (__int64)&unk_49ECF68;
      v225 = &v266[88 * (unsigned int)v267];
      if ( v266 != (_BYTE *)v225 )
      {
        do
        {
          v225 -= 11;
          v226 = (_QWORD *)v225[4];
          if ( v226 != v225 + 6 )
            j_j___libc_free_0(v226, v225[6] + 1LL);
          if ( (_QWORD *)*v225 != v225 + 2 )
            j_j___libc_free_0(*v225, v225[2] + 1LL);
        }
        while ( v79 != v225 );
        goto LABEL_114;
      }
    }
    else
    {
      v117 = *(_QWORD *)(a1 + 200);
      v118 = *(_QWORD *)(v117 + 16);
      v119 = v117 + 8;
      if ( v118 != v119 )
      {
        v120 = 0;
        v121 = 0;
        v122 = 0;
        do
        {
          if ( !*(_QWORD *)(v118 + 32) )
          {
            v123 = *(_BYTE *)(v118 + 67);
            if ( (v123 & 0x40) == 0 )
              goto LABEL_206;
            v124 = *(__int64 ***)(v118 + 16);
            v122 |= (unsigned __int8)((v123 >> 4) & 3) >> 1;
            v125 = **v124;
            do
            {
              v124 = (__int64 **)v124[2];
              if ( !v124 )
              {
                v121 = v234;
                goto LABEL_197;
              }
            }
            while ( **v124 == v125 );
            do
            {
              v126 = v124[2];
              if ( !v126 )
                break;
              v124 = (__int64 **)v126[2];
            }
            while ( v124 );
LABEL_197:
            v120 = v234;
          }
          v118 = *(_QWORD *)(v118 + 8);
        }
        while ( v119 != v118 );
        v233 = v122 & v121 & v120;
        if ( v233 )
        {
          v137 = sub_15E0530(v114);
          if ( sub_1602790(v137)
            || (v198 = sub_15E0530(*v242),
                v199 = sub_16033E0(v198),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v199 + 48LL))(v199)) )
          {
            v138 = *(_QWORD *)(a1 + 192);
            v139 = **(_QWORD **)(v138 + 32);
            sub_13FD840(v246, v138);
            sub_15C9090((__int64)&v247, v246);
            sub_15CA330(
              (__int64)&v272,
              (__int64)"loop-versioning-licm",
              (__int64)"IsLegalForVersioning",
              20,
              &v247,
              v139);
            sub_15CAB20((__int64)&v272, " Versioned loop for LICM.", 0x19u);
            sub_15CAB20((__int64)&v272, " Number of runtime checks we had to insert ", 0x2Bu);
            sub_15C9C50(
              (__int64)&v251,
              "RuntimeChecks",
              13,
              *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL) + 280LL));
            v259.m128i_i64[0] = (__int64)&v260;
            sub_19C88F0(v259.m128i_i64, v251.m128i_i64[0], v251.m128i_i64[0] + v251.m128i_i64[1]);
            v261.m128i_i64[1] = (__int64)&v263;
            sub_19C88F0(&v261.m128i_i64[1], v254, (__int64)&v254[v255]);
            a10 = (__m128)_mm_loadu_si128(&v257);
            *(__m128 *)&v264[8] = a10;
            v265 = v258;
            sub_15CAC60((__int64)&v272, &v259);
            if ( (__int64 *)v261.m128i_i64[1] != &v263 )
              j_j___libc_free_0(v261.m128i_i64[1], v263 + 1);
            if ( (__int64 *)v259.m128i_i64[0] != &v260 )
              j_j___libc_free_0(v259.m128i_i64[0], v260 + 1);
            v115 = _mm_loadu_si128(&v275);
            v140 = _mm_loadu_si128(&v278);
            v259.m128i_i32[2] = v273;
            v261 = v115;
            v259.m128i_i8[12] = BYTE4(v273);
            *(__m128i *)v264 = v140;
            v260 = v274;
            v262 = v276;
            v259.m128i_i64[0] = (__int64)&unk_49ECF68;
            v263 = v277;
            LOBYTE(v265) = v280;
            if ( v280 )
              *(_QWORD *)&v264[16] = v279;
            v266 = v268;
            v267 = 0x400000000LL;
            if ( (_DWORD)v282 )
              sub_19C9520((__int64)&v266, (__int64)&v281);
            v269 = v292;
            v270 = v293;
            v271 = v294;
            v259.m128i_i64[0] = (__int64)&unk_49ECF98;
            if ( v254 != (_BYTE *)v256 )
              j_j___libc_free_0(v254, v256[0] + 1LL);
            if ( (__int64 *)v251.m128i_i64[0] != &v252 )
              j_j___libc_free_0(v251.m128i_i64[0], v252 + 1);
            v272 = &unk_49ECF68;
            v141 = &v281[88 * (unsigned int)v282];
            if ( v281 != v141 )
            {
              v231 = v56;
              v142 = &v281[88 * (unsigned int)v282];
              v143 = v281;
              do
              {
                v142 -= 88;
                v144 = (char *)*((_QWORD *)v142 + 4);
                if ( v144 != v142 + 48 )
                  j_j___libc_free_0(v144, *((_QWORD *)v142 + 6) + 1LL);
                if ( *(char **)v142 != v142 + 16 )
                  j_j___libc_free_0(*(_QWORD *)v142, *((_QWORD *)v142 + 2) + 1LL);
              }
              while ( v143 != v142 );
              LODWORD(v56) = v231;
              v141 = v281;
            }
            if ( v141 != &v283 )
              _libc_free((unsigned __int64)v141);
            if ( v246[0].m128i_i64[0] )
              sub_161E7C0((__int64)v246, v246[0].m128i_i64[0]);
            sub_143AA50(v242, (__int64)&v259);
            v145 = v266;
            v259.m128i_i64[0] = (__int64)&unk_49ECF68;
            v146 = &v266[88 * (unsigned int)v267];
            if ( v266 != (_BYTE *)v146 )
            {
              do
              {
                v146 -= 11;
                v147 = (_QWORD *)v146[4];
                if ( v147 != v146 + 6 )
                  j_j___libc_free_0(v147, v146[6] + 1LL);
                if ( (_QWORD *)*v146 != v146 + 2 )
                  j_j___libc_free_0(*v146, v146[2] + 1LL);
              }
              while ( v145 != v146 );
              v146 = v266;
            }
            if ( v146 != (_QWORD *)v268 )
              _libc_free((unsigned __int64)v146);
          }
          v148 = *(__int64 **)(a1 + 8);
          v149 = *v148;
          v150 = v148[1];
          if ( v149 == v150 )
LABEL_420:
            BUG();
          while ( *(_UNKNOWN **)v149 != &unk_4F9E06C )
          {
            v149 += 16;
            if ( v150 == v149 )
              goto LABEL_420;
          }
          v151 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v149 + 8) + 104LL))(
                   *(_QWORD *)(v149 + 8),
                   &unk_4F9E06C);
          sub_1B1E040(
            (unsigned int)&v272,
            *(_QWORD *)(a1 + 184),
            *(_QWORD *)(a1 + 192),
            v56 + 160,
            v151 + 160,
            *(_QWORD *)(a1 + 168),
            1);
          sub_1B17630(&v259, v272);
          sub_1B1F0F0(&v272, &v259);
          if ( (__int64 *)v259.m128i_i64[0] != &v260 )
            _libc_free(v259.m128i_u64[0]);
          sub_19C8F50(
            v273,
            "llvm.loop.licm_versioning.disable",
            0,
            v116,
            *(double *)v115.m128i_i64,
            a5,
            a6,
            v152,
            v153,
            a9,
            a10);
          sub_19C8F50(
            (__int64)v272,
            "llvm.loop.licm_versioning.disable",
            0,
            v116,
            *(double *)v115.m128i_i64,
            a5,
            a6,
            v154,
            v155,
            a9,
            a10);
          sub_19C8F50(
            (__int64)v272,
            "llvm.mem.parallel_loop_access",
            0,
            v116,
            *(double *)v115.m128i_i64,
            a5,
            a6,
            v156,
            v157,
            a9,
            a10);
          v158 = sub_13FCB50((__int64)v272);
          v159 = sub_157EBA0(v158);
          v247.m128i_i64[0] = sub_16498A0(v159);
          v160 = sub_161C490(&v247, (__int64)"LVDomain", 8, 0);
          v251.m128i_i64[0] = (__int64)&v252;
          v259.m128i_i64[0] = (__int64)&v260;
          v251.m128i_i64[1] = 0x400000000LL;
          v259.m128i_i64[1] = 0x400000000LL;
          v161 = sub_161C490(&v247, (__int64)"LVAliasScope", 12, v160);
          v162 = *(_QWORD *)(a1 + 192);
          v232 = *(_QWORD *)(v162 + 40);
          for ( j = *(_QWORD *)(v162 + 32); v232 != j; j += 8 )
          {
            v163 = *(_QWORD *)(*(_QWORD *)j + 48LL);
            for ( k = *(_QWORD *)j + 40LL; k != v163; v163 = *(_QWORD *)(v163 + 8) )
            {
              v173 = v163 - 24;
              if ( !v163 )
                v173 = 0;
              if ( (unsigned __int8)sub_15F2ED0(v173) || (unsigned __int8)sub_15F3040(v173) )
              {
                v176 = v251.m128i_u32[2];
                if ( v251.m128i_i32[2] >= (unsigned __int32)v251.m128i_i32[3] )
                {
                  sub_16CD150((__int64)&v251, &v252, 0, 8, v174, v175);
                  v176 = v251.m128i_u32[2];
                }
                *(_QWORD *)(v251.m128i_i64[0] + 8 * v176) = v161;
                v177 = v259.m128i_u32[2];
                ++v251.m128i_i32[2];
                if ( v259.m128i_i32[2] >= (unsigned __int32)v259.m128i_i32[3] )
                {
                  sub_16CD150((__int64)&v259, &v260, 0, 8, v174, v175);
                  v177 = v259.m128i_u32[2];
                }
                *(_QWORD *)(v259.m128i_i64[0] + 8 * v177) = v161;
                v178 = (__int64 *)v259.m128i_i64[0];
                v179 = ++v259.m128i_i32[2];
                v180 = (__int64 *)sub_16498A0(v173);
                v181 = sub_1627350(v180, v178, (__int64 *)v179, 0, 1);
                v164 = *(_QWORD *)(v173 + 48);
                v182 = v181;
                if ( v164 || *(__int16 *)(v173 + 18) < 0 )
                  v164 = sub_1625790(v173, 8);
                v165 = sub_1631960(v164, v182);
                sub_1625C10(v173, 8, v165);
                v166 = (__int64 *)v251.m128i_u32[2];
                v167 = (__int64 *)v251.m128i_i64[0];
                v168 = (__int64 *)sub_16498A0(v173);
                v169 = sub_1627350(v168, v167, v166, 0, 1);
                v170 = *(_QWORD *)(v173 + 48);
                v171 = v169;
                if ( v170 || *(__int16 *)(v173 + 18) < 0 )
                  v170 = sub_1625790(v173, 7);
                v172 = sub_1631960(v170, v171);
                sub_1625C10(v173, 7, v172);
              }
            }
          }
          if ( (__int64 *)v259.m128i_i64[0] != &v260 )
            _libc_free(v259.m128i_u64[0]);
          if ( (__int64 *)v251.m128i_i64[0] != &v252 )
            _libc_free(v251.m128i_u64[0]);
          j___libc_free_0(v294);
          j___libc_free_0(v291);
          j___libc_free_0(v290);
          v285 = &unk_49EC708;
          if ( v289 )
          {
            v183 = v288;
            v184 = &v288[7 * v289];
            do
            {
              if ( *v183 != -16 && *v183 != -8 )
              {
                v185 = v183[1];
                if ( (_QWORD *)v185 != v183 + 3 )
                  _libc_free(v185);
              }
              v183 += 7;
            }
            while ( v184 != v183 );
          }
          j___libc_free_0(v288);
          if ( v286 != &v287 )
            _libc_free((unsigned __int64)v286);
          if ( (char *)v282 != &v284 )
            _libc_free(v282);
          if ( v280 )
          {
            if ( (_DWORD)v279 )
            {
              v191 = (_QWORD *)v278.m128i_i64[0];
              v192 = v278.m128i_i64[0] + 16LL * (unsigned int)v279;
              do
              {
                if ( *v191 != -8 && *v191 != -4 )
                {
                  v193 = v191[1];
                  if ( v193 )
                    sub_161E7C0((__int64)(v191 + 1), v193);
                }
                v191 += 2;
              }
              while ( (_QWORD *)v192 != v191 );
            }
            j___libc_free_0(v278.m128i_i64[0]);
          }
          if ( (_DWORD)v276 )
          {
            v186 = (_QWORD *)v275.m128i_i64[0];
            v251.m128i_i64[1] = 2;
            v252 = 0;
            v187 = -8;
            v188 = v275.m128i_i64[0] + ((unsigned __int64)(unsigned int)v276 << 6);
            v253 = -8;
            v251.m128i_i64[0] = (__int64)&unk_49E6B50;
            v254 = 0;
            v259.m128i_i64[1] = 2;
            v260 = 0;
            v261 = (__m128i)0xFFFFFFFFFFFFFFF0LL;
            v259.m128i_i64[0] = (__int64)&unk_49E6B50;
            while ( 1 )
            {
              v189 = v186[3];
              if ( v187 != v189 )
              {
                v187 = v261.m128i_i64[0];
                if ( v189 != v261.m128i_i64[0] )
                {
                  v190 = v186[7];
                  if ( v190 != 0 && v190 != -8 && v190 != -16 )
                  {
                    sub_1649B30(v186 + 5);
                    v189 = v186[3];
                  }
                  v187 = v189;
                }
              }
              *v186 = &unk_49EE2B0;
              if ( v187 != 0 && v187 != -8 && v187 != -16 )
                sub_1649B30(v186 + 1);
              v186 += 8;
              if ( (_QWORD *)v188 == v186 )
                break;
              v187 = v253;
            }
            v259.m128i_i64[0] = (__int64)&unk_49EE2B0;
            if ( v261.m128i_i64[0] != -8 && v261.m128i_i64[0] != 0 && v261.m128i_i64[0] != -16 )
              sub_1649B30(&v259.m128i_i64[1]);
            v251.m128i_i64[0] = (__int64)&unk_49EE2B0;
            if ( v253 != 0 && v253 != -8 && v253 != -16 )
              sub_1649B30(&v251.m128i_i64[1]);
          }
          j___libc_free_0(v275.m128i_i64[0]);
          goto LABEL_3;
        }
      }
LABEL_206:
      v129 = sub_15E0530(v114);
      if ( !sub_1602790(v129) )
      {
        v200 = sub_15E0530(*v242);
        v201 = sub_16033E0(v200);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v201 + 48LL))(v201) )
          goto LABEL_2;
      }
      v130 = *(_QWORD *)(a1 + 192);
      v131 = **(_QWORD **)(v130 + 32);
      sub_13FD840(&v247, v130);
      sub_15C9090((__int64)&v251, &v247);
      sub_15CA540((__int64)&v272, (__int64)"loop-versioning-licm", (__int64)"IllegalLoopMemoryAccess", 23, &v251, v131);
      sub_15CAB20((__int64)&v272, " Unsafe Loop memory access", 0x1Au);
      v132 = _mm_loadu_si128(&v275);
      v133 = _mm_loadu_si128(&v278);
      v259.m128i_i32[2] = v273;
      v261 = v132;
      v259.m128i_i8[12] = BYTE4(v273);
      *(__m128i *)v264 = v133;
      v260 = v274;
      v262 = v276;
      v259.m128i_i64[0] = (__int64)&unk_49ECF68;
      v263 = v277;
      LOBYTE(v265) = v280;
      if ( v280 )
        *(_QWORD *)&v264[16] = v279;
      v266 = v268;
      v267 = 0x400000000LL;
      if ( (_DWORD)v282 )
      {
        sub_19C9520((__int64)&v266, (__int64)&v281);
        v196 = v281;
        v269 = v292;
        v270 = v293;
        v271 = v294;
        v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
        v272 = &unk_49ECF68;
        v134 = &v281[88 * (unsigned int)v282];
        if ( v281 != v134 )
        {
          do
          {
            v134 -= 88;
            v197 = (char *)*((_QWORD *)v134 + 4);
            if ( v197 != v134 + 48 )
              j_j___libc_free_0(v197, *((_QWORD *)v134 + 6) + 1LL);
            if ( *(char **)v134 != v134 + 16 )
              j_j___libc_free_0(*(_QWORD *)v134, *((_QWORD *)v134 + 2) + 1LL);
          }
          while ( v196 != v134 );
          v134 = v281;
        }
      }
      else
      {
        v134 = v281;
        v269 = v292;
        v270 = v293;
        v271 = v294;
        v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
      }
      if ( v134 != &v283 )
        _libc_free((unsigned __int64)v134);
      if ( v247.m128i_i64[0] )
        sub_161E7C0((__int64)&v247, v247.m128i_i64[0]);
      sub_143AA50(v242, (__int64)&v259);
      v135 = v266;
      v259.m128i_i64[0] = (__int64)&unk_49ECF68;
      v79 = &v266[88 * (unsigned int)v267];
      if ( v266 != (_BYTE *)v79 )
      {
        do
        {
          v79 -= 11;
          v136 = (_QWORD *)v79[4];
          if ( v136 != v79 + 6 )
            j_j___libc_free_0(v136, v79[6] + 1LL);
          if ( (_QWORD *)*v79 != v79 + 2 )
            j_j___libc_free_0(*v79, v79[2] + 1LL);
        }
        while ( v135 != v79 );
LABEL_114:
        v79 = v266;
      }
    }
LABEL_115:
    if ( v79 != (_QWORD *)v268 )
      _libc_free((unsigned __int64)v79);
    goto LABEL_2;
  }
  v229 = v56;
  v90 = *(_QWORD *)(v89 + 32);
  while ( 1 )
  {
    v91 = *(_QWORD *)(*(_QWORD *)v90 + 48LL);
    v92 = *(_QWORD *)v90 + 40LL;
    if ( v91 != v92 )
      break;
LABEL_180:
    v90 += 8;
    if ( v230 == v90 )
    {
      LODWORD(v56) = v229;
      v89 = *(_QWORD *)(a1 + 192);
      goto LABEL_182;
    }
  }
  while ( 1 )
  {
    if ( !v91 )
      BUG();
    v95 = v91 - 24;
    if ( *(_BYTE *)(v91 - 8) == 78 )
    {
      v106 = v95 & 0xFFFFFFFFFFFFFFF8LL;
      v107 = *(_BYTE *)((v95 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      if ( v107 <= 0x17u )
      {
        v106 = 0;
      }
      else if ( v107 == 78 )
      {
        v106 |= 4u;
      }
      else if ( v107 != 29 )
      {
        v106 = 0;
      }
      if ( (unsigned int)sub_134CC90(*(_QWORD *)(a1 + 160), v106) != 4 )
        break;
    }
    if ( sub_15F3330(v91 - 24) )
      break;
    if ( (unsigned __int8)sub_15F2ED0(v91 - 24) )
    {
      if ( *(_BYTE *)(v91 - 8) != 54 || sub_15F32D0(v91 - 24) || (*(_BYTE *)(v91 - 6) & 1) != 0 )
        break;
      ++*(_DWORD *)(a1 + 216);
      v93 = *(_QWORD *)(a1 + 168);
      v240 = *(_QWORD *)(a1 + 192);
      v94 = sub_146F1B0(v93, *(_QWORD *)(v91 - 48));
      if ( sub_146CEE0(v93, v94, v240) )
        ++*(_DWORD *)(a1 + 220);
    }
    else if ( (unsigned __int8)sub_15F3040(v91 - 24) )
    {
      if ( *(_BYTE *)(v91 - 8) != 55 || sub_15F32D0(v91 - 24) || (*(_BYTE *)(v91 - 6) & 1) != 0 )
        break;
      ++*(_DWORD *)(a1 + 216);
      v108 = *(_QWORD *)(a1 + 168);
      v241 = *(_QWORD *)(a1 + 192);
      v109 = sub_146F1B0(v108, *(_QWORD *)(v91 - 48));
      if ( sub_146CEE0(v108, v109, v241) )
        ++*(_DWORD *)(a1 + 220);
      *(_BYTE *)(a1 + 224) = 0;
    }
    v91 = *(_QWORD *)(v91 + 8);
    if ( v92 == v91 )
      goto LABEL_180;
  }
  v96 = *(__int64 **)(a1 + 232);
  v97 = sub_15E0530(*v96);
  if ( sub_1602790(v97)
    || (v194 = sub_15E0530(*v96),
        v195 = sub_16033E0(v194),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v195 + 48LL))(v195)) )
  {
    sub_15CA5C0((__int64)&v272, (__int64)"loop-versioning-licm", (__int64)"IllegalLoopInst", 15, v95);
    sub_15CAB20((__int64)&v272, " Unsafe Loop Instruction", 0x18u);
    v98 = _mm_loadu_si128(&v275);
    v99 = _mm_loadu_si128(&v278);
    v259.m128i_i32[2] = v273;
    v261 = v98;
    v259.m128i_i8[12] = BYTE4(v273);
    *(__m128i *)v264 = v99;
    v260 = v274;
    v262 = v276;
    v259.m128i_i64[0] = (__int64)&unk_49ECF68;
    v263 = v277;
    LOBYTE(v265) = v280;
    if ( v280 )
      *(_QWORD *)&v264[16] = v279;
    v266 = v268;
    v267 = 0x400000000LL;
    if ( (_DWORD)v282 )
    {
      sub_19C9520((__int64)&v266, (__int64)&v281);
      v127 = v281;
      v269 = v292;
      v270 = v293;
      v271 = v294;
      v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v272 = &unk_49ECF68;
      v100 = &v281[88 * (unsigned int)v282];
      if ( v281 != v100 )
      {
        do
        {
          v100 -= 88;
          v128 = (char *)*((_QWORD *)v100 + 4);
          if ( v128 != v100 + 48 )
            j_j___libc_free_0(v128, *((_QWORD *)v100 + 6) + 1LL);
          if ( *(char **)v100 != v100 + 16 )
            j_j___libc_free_0(*(_QWORD *)v100, *((_QWORD *)v100 + 2) + 1LL);
        }
        while ( v127 != v100 );
        v100 = v281;
      }
    }
    else
    {
      v100 = v281;
      v269 = v292;
      v270 = v293;
      v271 = v294;
      v259.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    if ( v100 != &v283 )
      _libc_free((unsigned __int64)v100);
    sub_143AA50(v96, (__int64)&v259);
    v101 = v266;
    v259.m128i_i64[0] = (__int64)&unk_49ECF68;
    v102 = &v266[88 * (unsigned int)v267];
    if ( v266 == (_BYTE *)v102 )
      goto LABEL_159;
    do
    {
      v102 -= 11;
      v103 = (_QWORD *)v102[4];
      if ( v103 != v102 + 6 )
        j_j___libc_free_0(v103, v102[6] + 1LL);
      if ( (_QWORD *)*v102 != v102 + 2 )
        j_j___libc_free_0(*v102, v102[2] + 1LL);
    }
    while ( v101 != v102 );
LABEL_158:
    v102 = v266;
LABEL_159:
    if ( v102 != (_QWORD *)v268 )
      _libc_free((unsigned __int64)v102);
  }
LABEL_3:
  v11 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_BYTE *)(a1 + 224) = 1;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  if ( v11 )
  {
    sub_1359CD0(v11);
    if ( *(_DWORD *)(v11 + 48) )
    {
      sub_1359800(&v259, -8, 0);
      sub_1359800(&v272, -16, 0);
      v20 = *(_QWORD **)(v11 + 32);
      for ( m = &v20[6 * *(unsigned int *)(v11 + 48)]; m != v20; v20 += 6 )
      {
        v22 = v20[3];
        *v20 = &unk_49EE2B0;
        if ( v22 != -8 && v22 != 0 && v22 != -16 )
          sub_1649B30(v20 + 1);
      }
      v272 = &unk_49EE2B0;
      if ( v275.m128i_i64[0] != -8 && v275.m128i_i64[0] != 0 && v275.m128i_i64[0] != -16 )
        sub_1649B30(&v273);
      v259.m128i_i64[0] = (__int64)&unk_49EE2B0;
      if ( v261.m128i_i64[0] != 0 && v261.m128i_i64[0] != -8 && v261.m128i_i64[0] != -16 )
        sub_1649B30(&v259.m128i_i64[1]);
    }
    j___libc_free_0(*(_QWORD *)(v11 + 32));
    v12 = *(unsigned __int64 **)(v11 + 16);
    if ( (unsigned __int64 *)(v11 + 8) != v12 )
    {
      v237 = v11;
      v13 = (unsigned __int64 *)(v11 + 8);
      do
      {
        v14 = v12;
        v12 = (unsigned __int64 *)v12[1];
        v15 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
        *v12 = v15 | *v12 & 7;
        *(_QWORD *)(v15 + 8) = v12;
        v16 = (_QWORD *)v14[6];
        v17 = (_QWORD *)v14[5];
        *v14 &= 7u;
        v14[1] = 0;
        if ( v16 != v17 )
        {
          do
          {
            v18 = v17[2];
            if ( v18 != -8 && v18 != 0 && v18 != -16 )
              sub_1649B30(v17);
            v17 += 3;
          }
          while ( v16 != v17 );
          v17 = (_QWORD *)v14[5];
        }
        if ( v17 )
          j_j___libc_free_0(v17, v14[7] - (_QWORD)v17);
        j_j___libc_free_0(v14, 72);
      }
      while ( v13 != v12 );
      v11 = v237;
    }
    j_j___libc_free_0(v11, 72);
  }
  return v233;
}
