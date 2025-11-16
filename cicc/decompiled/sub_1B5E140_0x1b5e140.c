// Function: sub_1B5E140
// Address: 0x1b5e140
//
__int64 __fastcall sub_1B5E140(
        __int64 a1,
        __int64 **a2,
        _QWORD *a3,
        __int64 a4,
        unsigned int *a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        __m128i a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v17; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rbx
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rcx
  __int64 v27; // r15
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // rdx
  unsigned int v34; // eax
  unsigned int v35; // r13d
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned __int64 v43; // r14
  __int64 v44; // rdx
  __int64 v45; // r15
  _QWORD *v46; // rax
  _QWORD *v47; // rbx
  unsigned __int64 *v48; // r15
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  unsigned int v53; // eax
  double v54; // xmm4_8
  double v55; // xmm5_8
  bool v56; // bl
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // rax
  unsigned __int64 v60; // rax
  __int64 v61; // rsi
  unsigned __int64 v62; // r15
  double v63; // xmm4_8
  double v64; // xmm5_8
  char v65; // al
  __int64 v66; // rdi
  __int64 v67; // rdx
  __int64 **v68; // rdi
  double v69; // xmm4_8
  double v70; // xmm5_8
  __int64 v71; // rax
  __int64 v72; // rdx
  unsigned int v73; // eax
  _QWORD *v74; // rax
  __int64 v75; // r15
  __int64 *v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // rbx
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rcx
  unsigned __int64 v85; // r12
  __int64 v86; // rbx
  _QWORD *v87; // rax
  char v88; // r12
  __int64 v89; // r14
  int v90; // ebx
  unsigned __int64 v91; // rax
  __int64 v92; // rsi
  __int64 v93; // rax
  __int64 v94; // rcx
  __int64 v95; // r12
  __int64 v96; // rbx
  __int64 v97; // r14
  __int64 v98; // r13
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // r12
  __int64 v102; // rdx
  unsigned int v103; // ebx
  __int64 v104; // r12
  __int64 v105; // rdi
  __int64 *v106; // rax
  __int64 v107; // rsi
  int v108; // ebx
  __int64 v109; // r12
  __int64 v110; // r14
  double v111; // xmm4_8
  double v112; // xmm5_8
  __int64 j; // rbx
  int v114; // eax
  __int64 v115; // rdx
  int v116; // eax
  __int64 v117; // rdi
  double v118; // xmm4_8
  double v119; // xmm5_8
  unsigned __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rdi
  double v123; // xmm4_8
  double v124; // xmm5_8
  unsigned __int64 v125; // r12
  __int64 v126; // rax
  __int64 v127; // rbx
  _QWORD *v128; // rax
  unsigned __int64 v129; // rax
  double v130; // xmm4_8
  double v131; // xmm5_8
  __int64 v132; // rdi
  __int64 v133; // r12
  _QWORD *v134; // rax
  __int64 v135; // rbx
  __int64 v136; // rax
  unsigned int v137; // eax
  __int64 k; // rax
  int v139; // edx
  __int64 v140; // rax
  __int64 v141; // rdx
  __int64 v142; // r12
  _QWORD *v143; // rax
  __int64 v144; // r13
  __int64 v145; // r12
  __int64 v146; // r14
  __int64 v147; // rbx
  __int64 v148; // rdx
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 m; // rax
  char v152; // cl
  char v153; // al
  __int64 v154; // rbx
  __int64 v155; // r13
  _QWORD *v156; // rax
  __int64 *v157; // r12
  __int64 *v158; // rbx
  __int64 v159; // r8
  __int64 *v160; // rdi
  __int64 *v161; // rax
  __int64 *v162; // rsi
  __int64 v163; // rcx
  unsigned __int64 v164; // r12
  __int64 v165; // rsi
  __int64 *v166; // rax
  __int64 v167; // rdx
  unsigned __int64 v168; // rax
  double v169; // xmm4_8
  double v170; // xmm5_8
  __int64 v171; // rdx
  unsigned __int64 *v172; // rax
  __int64 v173; // r14
  __int64 v174; // r15
  __int64 v175; // r13
  __int64 v176; // r14
  __int64 v177; // rdi
  __int64 v178; // rax
  __int64 v179; // rbx
  __int64 v180; // rsi
  __int64 v181; // rax
  __int64 v182; // rbx
  unsigned __int64 v183; // rax
  __int64 v184; // rdx
  __int64 v185; // rcx
  __int64 v186; // r8
  __int64 *v187; // r9
  __int64 v188; // rsi
  __int64 v189; // r14
  __int64 i; // r12
  unsigned int v191; // esi
  int v192; // edx
  __int64 v193; // rax
  __int64 v194; // rcx
  __int64 v195; // r15
  _QWORD *v196; // rbx
  double v197; // xmm4_8
  double v198; // xmm5_8
  _QWORD *v199; // rax
  _QWORD *v200; // rax
  __int64 v201; // rbx
  _QWORD *v202; // rdi
  __int64 v203; // rbx
  __int64 v204; // rax
  char v205; // r9
  int v206; // r14d
  int v207; // ebx
  unsigned __int64 v208; // rdx
  __int64 v209; // r12
  __int64 *v210; // rax
  unsigned int v211; // r8d
  char v212; // dl
  __int64 *v213; // rdi
  __int64 *v214; // rsi
  __int64 *v215; // rax
  __int64 *v216; // rdx
  __int64 v217; // rdi
  __int64 *v218; // r13
  __int64 v219; // rax
  __int64 v220; // rbx
  __int64 v221; // r12
  unsigned __int64 v222; // rdi
  int v223; // ebx
  unsigned __int64 v224; // r12
  unsigned int v225; // r13d
  __int64 *v226; // rcx
  __int64 v227; // r9
  __int64 v228; // rax
  __int64 v229; // r8
  __int64 *v230; // rdi
  __int64 *v231; // rax
  __int64 *v232; // rsi
  __int64 v233; // rdx
  __int64 v234; // rax
  _QWORD *v235; // r13
  __int64 v236; // r13
  _QWORD *v237; // rdi
  char v238; // bl
  __int64 v239; // rax
  unsigned __int64 v240; // rdx
  __int64 v241; // rax
  __int64 v242; // r12
  __int64 v243; // r14
  __int64 *v244; // rbx
  unsigned __int64 v245; // rax
  unsigned __int64 v246; // rdi
  __int64 v247; // rsi
  unsigned __int64 v248; // rdx
  __int64 v249; // rdx
  __int64 *v250; // rax
  __int64 *v251; // r12
  __int64 *v252; // rax
  __int64 v253; // rdi
  __int64 *v254; // rbx
  __int64 v255; // rax
  _QWORD *v256; // rax
  __int64 v257; // r12
  __int64 *v258; // rbx
  __int64 v259; // rax
  __int64 v260; // rcx
  __int64 *v261; // rax
  __int64 v262; // r13
  __int64 v263; // rdi
  _QWORD *v264; // rax
  double v265; // xmm4_8
  double v266; // xmm5_8
  __int64 v267; // r12
  _QWORD *v268; // rax
  double v269; // xmm4_8
  double v270; // xmm5_8
  __int64 v271; // r12
  _QWORD *v272; // rdx
  _QWORD *v273; // rax
  _QWORD *v274; // rcx
  __int64 v275; // rdx
  _QWORD *v276; // rdx
  __int64 v277; // rdx
  __int64 v278; // rcx
  __int64 v279; // r12
  _QWORD *v280; // rdi
  __int64 v281; // r12
  _QWORD *v282; // rdi
  __int64 v283; // rcx
  _QWORD *v284; // rsi
  char v285; // [rsp+7h] [rbp-2A9h]
  unsigned __int8 v286; // [rsp+8h] [rbp-2A8h]
  __int64 v287; // [rsp+10h] [rbp-2A0h]
  unsigned __int8 v288; // [rsp+10h] [rbp-2A0h]
  unsigned __int8 v289; // [rsp+10h] [rbp-2A0h]
  __int64 v290; // [rsp+10h] [rbp-2A0h]
  char v291; // [rsp+10h] [rbp-2A0h]
  bool v292; // [rsp+18h] [rbp-298h]
  __int64 *v293; // [rsp+18h] [rbp-298h]
  __int64 v294; // [rsp+20h] [rbp-290h]
  unsigned __int8 v295; // [rsp+20h] [rbp-290h]
  unsigned __int8 v296; // [rsp+20h] [rbp-290h]
  unsigned __int64 v297; // [rsp+20h] [rbp-290h]
  __int64 *v298; // [rsp+20h] [rbp-290h]
  __int64 v299; // [rsp+20h] [rbp-290h]
  __int64 v300; // [rsp+28h] [rbp-288h]
  __int64 v301; // [rsp+28h] [rbp-288h]
  __int64 v302; // [rsp+28h] [rbp-288h]
  __int64 v303; // [rsp+30h] [rbp-280h] BYREF
  __int16 v304; // [rsp+40h] [rbp-270h]
  __int64 **v305; // [rsp+50h] [rbp-260h] BYREF
  unsigned __int64 v306; // [rsp+58h] [rbp-258h]
  _QWORD *v307; // [rsp+60h] [rbp-250h]
  __int64 v308; // [rsp+68h] [rbp-248h]
  __int64 v309; // [rsp+70h] [rbp-240h]
  unsigned int *v310; // [rsp+78h] [rbp-238h]
  __int64 v311; // [rsp+80h] [rbp-230h] BYREF
  __int64 v312; // [rsp+88h] [rbp-228h]
  __int64 v313; // [rsp+90h] [rbp-220h]
  __int64 v314; // [rsp+98h] [rbp-218h]
  __int64 v315; // [rsp+A0h] [rbp-210h]
  int v316; // [rsp+A8h] [rbp-208h]
  __int64 v317; // [rsp+B0h] [rbp-200h]
  __int64 v318; // [rsp+B8h] [rbp-1F8h]
  unsigned __int8 *v319; // [rsp+D0h] [rbp-1E0h] BYREF
  __int64 v320; // [rsp+D8h] [rbp-1D8h]
  __int64 *v321; // [rsp+E0h] [rbp-1D0h]
  __int64 v322; // [rsp+E8h] [rbp-1C8h]
  __int64 v323; // [rsp+F0h] [rbp-1C0h]
  int v324; // [rsp+F8h] [rbp-1B8h]
  __int64 v325; // [rsp+100h] [rbp-1B0h]
  __int64 v326; // [rsp+108h] [rbp-1A8h]
  __m128i v327; // [rsp+120h] [rbp-190h] BYREF
  __int64 *v328; // [rsp+130h] [rbp-180h] BYREF
  __int64 v329; // [rsp+138h] [rbp-178h]
  __int64 v330; // [rsp+140h] [rbp-170h]
  _QWORD v331[17]; // [rsp+148h] [rbp-168h] BYREF
  __m128i v332; // [rsp+1D0h] [rbp-E0h] BYREF
  __int64 *v333; // [rsp+1E0h] [rbp-D0h] BYREF
  __int64 v334; // [rsp+1E8h] [rbp-C8h]
  __int64 v335; // [rsp+1F0h] [rbp-C0h]
  int v336; // [rsp+1F8h] [rbp-B8h] BYREF
  __int64 *v337; // [rsp+200h] [rbp-B0h] BYREF
  __int64 v338; // [rsp+208h] [rbp-A8h]
  _BYTE v339[160]; // [rsp+210h] [rbp-A0h] BYREF

  v17 = a1;
  v19 = sub_157EB90(a1);
  v20 = sub_1632FA0(v19);
  v310 = a5;
  v23 = *(_QWORD *)(a1 + 8);
  v306 = v20;
  v305 = a2;
  v307 = a3;
  v308 = a4;
  v309 = a6;
  if ( v23 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v23) + 16) - 25) > 9u )
    {
      v23 = *(_QWORD *)(v23 + 8);
      if ( !v23 )
        goto LABEL_17;
    }
  }
  else
  {
LABEL_17:
    v37 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 80LL);
    if ( !v37 || a1 != v37 - 24 )
      goto LABEL_19;
  }
  if ( a1 == sub_157F0B0(a1) )
  {
LABEL_19:
    v35 = 1;
    sub_1AA7270(
      a1,
      0,
      a7,
      *(double *)a8.m128i_i64,
      *(double *)a9.m128i_i64,
      *(double *)a10.m128i_i64,
      v21,
      v22,
      a13,
      a14);
    return v35;
  }
  v286 = sub_1AEE9C0(a1, 1u, 0, 0);
  v285 = sub_1AEE3A0(
           a1,
           a7,
           *(double *)a8.m128i_i64,
           *(double *)a9.m128i_i64,
           *(double *)a10.m128i_i64,
           v24,
           v25,
           a13,
           a14);
  v27 = sub_157F280(a1);
  v287 = v30;
  if ( v27 == v30 )
  {
LABEL_15:
    v35 = v286;
    LOBYTE(v35) = v285 | v286;
    goto LABEL_47;
  }
  while ( 1 )
  {
    v31 = 0;
    v32 = 8LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
    if ( (*(_DWORD *)(v27 + 20) & 0xFFFFFFF) != 0 )
      break;
LABEL_10:
    v36 = *(_QWORD *)(v27 + 32);
    if ( !v36 )
      goto LABEL_508;
    v27 = 0;
    if ( *(_BYTE *)(v36 - 8) == 77 )
      v27 = v36 - 24;
    if ( v287 == v27 )
    {
      v17 = a1;
      goto LABEL_15;
    }
  }
  while ( 1 )
  {
    v33 = (*(_BYTE *)(v27 + 23) & 0x40) != 0 ? *(_QWORD *)(v27 - 8) : v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
    LOBYTE(v34) = sub_1B43710(*(_QWORD *)(v33 + 3 * v31), v27, v33, v26);
    v35 = v34;
    if ( (_BYTE)v34 )
      break;
LABEL_9:
    v31 += 8;
    if ( v32 == v31 )
      goto LABEL_10;
  }
  if ( (*(_BYTE *)(v27 + 23) & 0x40) != 0 )
    v39 = *(_QWORD *)(v27 - 8);
  else
    v39 = v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
  v40 = sub_157EBA0(*(_QWORD *)(v31 + v39 + 24LL * *(unsigned int *)(v27 + 56) + 8));
  v41 = sub_16498A0(v40);
  v332.m128i_i64[0] = 0;
  v334 = v41;
  v335 = 0;
  v336 = 0;
  v337 = 0;
  v338 = 0;
  v332.m128i_i64[1] = *(_QWORD *)(v40 + 40);
  v333 = (__int64 *)(v40 + 24);
  v42 = *(_QWORD *)(v40 + 48);
  v327.m128i_i64[0] = v42;
  if ( v42 )
  {
    sub_1623A60((__int64)&v327, v42, 2);
    if ( v332.m128i_i64[0] )
      sub_161E7C0((__int64)&v332, v332.m128i_i64[0]);
    v332.m128i_i64[0] = v327.m128i_i64[0];
    if ( v327.m128i_i64[0] )
      sub_1623210((__int64)&v327, (unsigned __int8 *)v327.m128i_i64[0], (__int64)&v332);
  }
  if ( *(_BYTE *)(v40 + 16) != 26 )
  {
    if ( v332.m128i_i64[0] )
      sub_161E7C0((__int64)&v332, v332.m128i_i64[0]);
    goto LABEL_9;
  }
  v43 = v40;
  v17 = a1;
  if ( (*(_BYTE *)(v27 + 23) & 0x40) != 0 )
    v44 = *(_QWORD *)(v27 - 8);
  else
    v44 = v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
  sub_157F2D0(a1, *(_QWORD *)(v31 + v44 + 24LL * *(unsigned int *)(v27 + 56) + 8), 0);
  if ( (*(_DWORD *)(v43 + 20) & 0xFFFFFFF) == 1 )
  {
    LOWORD(v328) = 257;
    v74 = sub_1648A60(56, 0);
    v75 = (__int64)v74;
    if ( v74 )
      sub_15F82A0((__int64)v74, v334, 0);
    if ( v332.m128i_i64[1] )
    {
      v76 = v333;
      sub_157E9D0(v332.m128i_i64[1] + 40, v75);
      v77 = *(_QWORD *)(v75 + 24);
      v78 = *v76;
      *(_QWORD *)(v75 + 32) = v76;
      v78 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v75 + 24) = v78 | v77 & 7;
      *(_QWORD *)(v78 + 8) = v75 + 24;
      *v76 = *v76 & 7 | (v75 + 24);
    }
    sub_164B780(v75, v327.m128i_i64);
    sub_12A86E0(v332.m128i_i64, v75);
  }
  else
  {
    v45 = *(_QWORD *)(v43 - 24);
    if ( v45 && a1 == v45 )
      v45 = *(_QWORD *)(v43 - 48);
    LOWORD(v328) = 257;
    v46 = sub_1648A60(56, 1u);
    v47 = v46;
    if ( v46 )
      sub_15F8320((__int64)v46, v45, 0);
    if ( v332.m128i_i64[1] )
    {
      v48 = (unsigned __int64 *)v333;
      sub_157E9D0(v332.m128i_i64[1] + 40, (__int64)v47);
      v49 = v47[3];
      v50 = *v48;
      v47[4] = v48;
      v50 &= 0xFFFFFFFFFFFFFFF8LL;
      v47[3] = v50 | v49 & 7;
      *(_QWORD *)(v50 + 8) = v47 + 3;
      *v48 = *v48 & 7 | (unsigned __int64)(v47 + 3);
    }
    sub_164B780((__int64)v47, v327.m128i_i64);
    if ( v332.m128i_i64[0] )
    {
      v319 = (unsigned __int8 *)v332.m128i_i64[0];
      sub_1623A60((__int64)&v319, v332.m128i_i64[0], 2);
      v51 = v47[6];
      if ( v51 )
        sub_161E7C0((__int64)(v47 + 6), v51);
      v52 = v319;
      v47[6] = v319;
      if ( v52 )
        sub_1623210((__int64)&v319, v52, (__int64)(v47 + 6));
    }
  }
  sub_15F20C0((_QWORD *)v43);
  if ( v332.m128i_i64[0] )
    sub_161E7C0((__int64)&v332, v332.m128i_i64[0]);
LABEL_47:
  v53 = sub_1AA7EA0(v17, 0, 0, 0, 0, a7, a8, a9, *(double *)a10.m128i_i64, v28, v29, a13, a14);
  v56 = v53;
  if ( (_BYTE)v53 )
    return v53;
  if ( byte_4FB74C0 && *((_BYTE *)v310 + 7) )
    v35 |= sub_1B51110(
             v17,
             a7,
             *(double *)a8.m128i_i64,
             *(double *)a9.m128i_i64,
             *(double *)a10.m128i_i64,
             v54,
             v55,
             a13,
             a14);
  v312 = v17;
  v314 = sub_157E9C0(v17);
  v313 = v17 + 40;
  v59 = *(_QWORD *)(v17 + 48);
  v311 = 0;
  v315 = 0;
  v316 = 0;
  v317 = 0;
  v318 = 0;
  if ( !v59 )
    goto LABEL_508;
  if ( *(_BYTE *)(v59 - 8) == 77 && (*(_DWORD *)(v59 - 4) & 0xFFFFFFF) == 2 )
    v35 |= sub_1B4CB10(
             v59 - 24,
             v305,
             v306,
             a7,
             *(double *)a8.m128i_i64,
             *(double *)a9.m128i_i64,
             *(double *)a10.m128i_i64,
             v57,
             v58,
             a13,
             a14);
  v60 = sub_157EBA0(v17);
  v61 = *(_QWORD *)(v60 + 48);
  v312 = *(_QWORD *)(v60 + 40);
  v313 = v60 + 24;
  v332.m128i_i64[0] = v61;
  if ( v61 )
  {
    sub_1623A60((__int64)&v332, v61, 2);
    v311 = v332.m128i_i64[0];
    if ( v332.m128i_i64[0] )
      sub_1623210((__int64)&v332, (unsigned __int8 *)v332.m128i_i64[0], (__int64)&v311);
  }
  v62 = sub_157EBA0(v17);
  v65 = *(_BYTE *)(v62 + 16);
  if ( v65 != 26 )
  {
    switch ( v65 )
    {
      case 25:
        if ( (unsigned __int8)sub_1B481C0((__int64)&v305, v62, &v311) )
          goto LABEL_66;
        goto LABEL_67;
      case 30:
        v79 = *(_QWORD *)(v62 - 24);
        v301 = *(_QWORD *)(v62 + 40);
        if ( *(_BYTE *)(v79 + 16) != 77 )
        {
          if ( *(_BYTE *)(sub_157ED20(v301) + 16) != 88 )
            goto LABEL_67;
          v80 = *(_QWORD *)(v62 - 24);
          if ( v80 != sub_157ED20(*(_QWORD *)(v62 + 40)) )
            goto LABEL_67;
          v81 = *(_QWORD *)(v62 + 40);
          v82 = sub_157ED20(v81);
          if ( *(_BYTE *)(v82 + 16) != 88 )
            v82 = 0;
          v83 = v82 + 24;
          while ( 1 )
          {
            v83 = *(_QWORD *)(v83 + 8);
            if ( v62 + 24 == v83 )
              break;
            if ( !v83 )
              goto LABEL_508;
            if ( *(_BYTE *)(v83 - 8) == 78 )
            {
              v84 = *(_QWORD *)(v83 - 48);
              if ( !*(_BYTE *)(v84 + 16)
                && (*(_BYTE *)(v84 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v84 + 36) - 35) <= 3 )
              {
                continue;
              }
            }
            goto LABEL_67;
          }
          v262 = *(_QWORD *)(v81 + 8);
          if ( v262 )
          {
            while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v262) + 16) - 25) > 9u )
            {
              v262 = *(_QWORD *)(v262 + 8);
              if ( !v262 )
                goto LABEL_472;
            }
            while ( 1 )
            {
              v267 = *(_QWORD *)(v262 + 8);
              if ( !v267 )
                break;
              while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v267) + 16) - 25) > 9u )
              {
                v267 = *(_QWORD *)(v267 + 8);
                if ( !v267 )
                  goto LABEL_471;
              }
              v263 = v262;
              v262 = v267;
              v264 = sub_1648700(v263);
              sub_1AF0970(
                v264[5],
                0,
                a7,
                *(double *)a8.m128i_i64,
                *(double *)a9.m128i_i64,
                *(double *)a10.m128i_i64,
                v265,
                v266,
                a13,
                a14);
            }
LABEL_471:
            v268 = sub_1648700(v262);
            sub_1AF0970(
              v268[5],
              0,
              a7,
              *(double *)a8.m128i_i64,
              *(double *)a9.m128i_i64,
              *(double *)a10.m128i_i64,
              v269,
              v270,
              a13,
              a14);
          }
LABEL_472:
          sub_157F980(v81);
          v271 = v309;
          if ( !v309 )
            goto LABEL_66;
          v272 = *(_QWORD **)(v309 + 16);
          v273 = *(_QWORD **)(v309 + 8);
          if ( v272 == v273 )
          {
            v283 = *(unsigned int *)(v309 + 28);
            v284 = &v272[v283];
            if ( v272 == v284 )
            {
LABEL_506:
              v273 = &v272[v283];
            }
            else
            {
              while ( v81 != *v273 )
              {
                if ( v284 == ++v273 )
                  goto LABEL_506;
              }
            }
          }
          else
          {
            v273 = sub_16CC9F0(v309, v81);
            if ( v81 == *v273 )
            {
              v274 = *(_QWORD **)(v309 + 16);
              v272 = *(_QWORD **)(v309 + 8);
              if ( v274 != v272 )
              {
                v275 = *(unsigned int *)(v309 + 24);
                goto LABEL_477;
              }
              v283 = *(unsigned int *)(v309 + 28);
            }
            else
            {
              v272 = *(_QWORD **)(v309 + 16);
              v274 = v272;
              if ( v272 != *(_QWORD **)(v309 + 8) )
              {
                v275 = *(unsigned int *)(v309 + 24);
                v273 = &v274[v275];
LABEL_477:
                v276 = &v274[v275];
LABEL_478:
                if ( v276 == v273 )
                  goto LABEL_66;
                *v273 = -2;
                v35 = 1;
                ++*(_DWORD *)(v271 + 32);
                goto LABEL_67;
              }
              v283 = *(unsigned int *)(v309 + 28);
              v273 = &v272[v283];
            }
          }
          v276 = &v272[v283];
          goto LABEL_478;
        }
        v149 = v79 + 24;
        while ( 1 )
        {
          v149 = *(_QWORD *)(v149 + 8);
          if ( v62 + 24 == v149 )
            break;
          if ( !v149 )
            goto LABEL_508;
          if ( *(_BYTE *)(v149 - 8) == 78 )
          {
            v150 = *(_QWORD *)(v149 - 48);
            if ( !*(_BYTE *)(v150 + 16)
              && (*(_BYTE *)(v150 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v150 + 36) - 35) <= 3 )
            {
              continue;
            }
          }
          goto LABEL_67;
        }
        v172 = (unsigned __int64 *)&v333;
        v332.m128i_i64[0] = 0;
        v332.m128i_i64[1] = 1;
        do
          *v172++ = -8;
        while ( v172 != (unsigned __int64 *)&v337 );
        v337 = (__int64 *)v339;
        v338 = 0x400000000LL;
        v173 = *(_QWORD *)(v62 - 24);
        if ( (*(_DWORD *)(v173 + 20) & 0xFFFFFFF) != 0 )
        {
          v297 = v62;
          v174 = *(_QWORD *)(v62 - 24);
          v289 = v35;
          v175 = 0;
          v292 = v56;
          v176 = 8LL * (*(_DWORD *)(v173 + 20) & 0xFFFFFFF);
          do
          {
            v180 = v175 + 24LL * *(unsigned int *)(v174 + 56) + 8;
            if ( (*(_BYTE *)(v174 + 23) & 0x40) != 0 )
            {
              v177 = *(_QWORD *)(*(_QWORD *)(v174 - 8) + v180);
              v327.m128i_i64[0] = v177;
              v178 = *(_QWORD *)(v174 - 8);
            }
            else
            {
              v178 = v174 - 24LL * (*(_DWORD *)(v174 + 20) & 0xFFFFFFF);
              v177 = *(_QWORD *)(v178 + v180);
              v327.m128i_i64[0] = v177;
            }
            v179 = *(_QWORD *)(v178 + 3 * v175);
            if ( v301 == sub_157F210(v177) )
            {
              v181 = sub_157ED20(v327.m128i_i64[0]);
              if ( *(_BYTE *)(v181 + 16) != 88 )
                v181 = 0;
              if ( v181 == v179 )
              {
                v182 = sub_157ED20(v327.m128i_i64[0]) + 24;
                v183 = sub_157EBA0(v327.m128i_i64[0]) + 24;
                while ( 1 )
                {
                  v182 = *(_QWORD *)(v182 + 8);
                  if ( v183 == v182 )
                    break;
                  if ( !v182 )
                    goto LABEL_508;
                  if ( *(_BYTE *)(v182 - 8) == 78 )
                  {
                    v188 = *(_QWORD *)(v182 - 48);
                    if ( !*(_BYTE *)(v188 + 16)
                      && (*(_BYTE *)(v188 + 33) & 0x20) != 0
                      && (unsigned int)(*(_DWORD *)(v188 + 36) - 35) <= 3 )
                    {
                      continue;
                    }
                  }
                  goto LABEL_305;
                }
                sub_1B5A4D0((__int64)&v332, v327.m128i_i64, v184, v185, v186, v187);
              }
            }
LABEL_305:
            v175 += 8;
          }
          while ( v175 != v176 );
          v189 = v174;
          v56 = v292;
          v35 = v289;
          if ( (_DWORD)v338 )
          {
            v290 = v297;
            v298 = v337;
            v293 = &v337[(unsigned int)v338];
            do
            {
              for ( i = *v298; ; sub_157F2D0(v301, i, 1) )
              {
                v191 = *(_DWORD *)(v189 + 20) & 0xFFFFFFF;
                if ( !v191 )
                  break;
                v192 = 0;
                v193 = 24LL * *(unsigned int *)(v189 + 56) + 8;
                while ( 1 )
                {
                  v194 = v189 - 24LL * v191;
                  if ( (*(_BYTE *)(v189 + 23) & 0x40) != 0 )
                    v194 = *(_QWORD *)(v189 - 8);
                  if ( i == *(_QWORD *)(v194 + v193) )
                    break;
                  ++v192;
                  v193 += 8;
                  if ( v191 == v192 )
                    goto LABEL_329;
                }
              }
LABEL_329:
              v195 = *(_QWORD *)(i + 8);
              if ( v195 )
              {
                while ( 1 )
                {
                  v196 = sub_1648700(v195);
                  if ( (unsigned __int8)(*((_BYTE *)v196 + 16) - 25) <= 9u )
                    break;
                  v195 = *(_QWORD *)(v195 + 8);
                  if ( !v195 )
                    goto LABEL_337;
                }
                while ( 1 )
                {
                  v195 = *(_QWORD *)(v195 + 8);
                  if ( !v195 )
                    break;
                  while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v195) + 16) - 25) <= 9u )
                  {
                    sub_1AF0970(
                      v196[5],
                      0,
                      a7,
                      *(double *)a8.m128i_i64,
                      *(double *)a9.m128i_i64,
                      *(double *)a10.m128i_i64,
                      v197,
                      v198,
                      a13,
                      a14);
                    v199 = sub_1648700(v195);
                    v195 = *(_QWORD *)(v195 + 8);
                    v196 = v199;
                    if ( !v195 )
                      goto LABEL_336;
                  }
                }
LABEL_336:
                sub_1AF0970(
                  v196[5],
                  0,
                  a7,
                  *(double *)a8.m128i_i64,
                  *(double *)a9.m128i_i64,
                  *(double *)a10.m128i_i64,
                  v197,
                  v198,
                  a13,
                  a14);
              }
LABEL_337:
              v200 = (_QWORD *)sub_157EBA0(i);
              sub_15F20C0(v200);
              v201 = sub_16498A0(v290);
              v202 = sub_1648A60(56, 0);
              if ( v202 )
                sub_15F82E0((__int64)v202, v201, i);
              ++v298;
            }
            while ( v293 != v298 );
            v35 = (unsigned __int8)v35;
            v203 = *(_QWORD *)(v301 + 8);
            if ( v203 )
            {
              while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v203) + 16) - 25) > 9u )
              {
                v203 = *(_QWORD *)(v203 + 8);
                if ( !v203 )
                  goto LABEL_349;
              }
            }
            else
            {
LABEL_349:
              sub_157F980(v301);
            }
            v56 = (_DWORD)v338 != 0;
          }
          if ( v337 != (__int64 *)v339 )
            _libc_free((unsigned __int64)v337);
          if ( (v332.m128i_i8[8] & 1) != 0 )
          {
LABEL_346:
            if ( v56 )
              goto LABEL_66;
            goto LABEL_67;
          }
        }
        else if ( (v332.m128i_i8[8] & 1) != 0 )
        {
          goto LABEL_67;
        }
        j___libc_free_0(v333);
        goto LABEL_346;
      case 32:
        if ( *(_BYTE *)(*(_QWORD *)(v62 - 24LL * (*(_DWORD *)(v62 + 20) & 0xFFFFFFF)) + 16LL) == 9 )
          goto LABEL_67;
        v108 = *(_BYTE *)(v62 + 18) & 1;
        if ( (*(_BYTE *)(v62 + 18) & 1) != 0 )
        {
          v109 = *(_QWORD *)(v62 + 24 * (1LL - (*(_DWORD *)(v62 + 20) & 0xFFFFFFF)));
          if ( v109 )
          {
            if ( *(_QWORD *)(v62 + 40) == sub_157F0B0(*(_QWORD *)(v62 + 24 * (1LL - (*(_DWORD *)(v62 + 20) & 0xFFFFFFF)))) )
            {
              v234 = *(_QWORD *)(v109 + 48);
              if ( !v234 )
                goto LABEL_508;
              if ( *(_BYTE *)(v234 - 8) == 73 )
              {
                v235 = (_QWORD *)(v234 - 24);
                sub_164D160(
                  v234 - 24,
                  *(_QWORD *)(v62 - 24LL * (*(_DWORD *)(v62 + 20) & 0xFFFFFFF)),
                  a7,
                  *(double *)a8.m128i_i64,
                  *(double *)a9.m128i_i64,
                  *(double *)a10.m128i_i64,
                  v63,
                  v64,
                  a13,
                  a14);
                sub_15F20C0(v235);
                v236 = *(_QWORD *)(v62 + 40);
                v237 = sub_1648A60(56, 1u);
                if ( v237 )
                  sub_15F8590((__int64)v237, v109, v236);
                v35 = v108;
                sub_15F20C0((_QWORD *)v62);
                goto LABEL_67;
              }
            }
          }
        }
        if ( (unsigned __int8)sub_1B491E0(
                                v62,
                                a7,
                                *(double *)a8.m128i_i64,
                                *(double *)a9.m128i_i64,
                                *(double *)a10.m128i_i64,
                                v63,
                                v64,
                                a13,
                                a14) )
          goto LABEL_66;
        goto LABEL_67;
      case 27:
        if ( (unsigned __int8)sub_1B60700(&v305, v62, &v311) )
          goto LABEL_66;
        goto LABEL_67;
      case 31:
        if ( sub_1B49850(
               (__int64)&v305,
               v62,
               a7,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               *(double *)a10.m128i_i64,
               v63,
               v64,
               a13,
               a14) )
        {
          goto LABEL_66;
        }
        goto LABEL_67;
    }
    if ( v65 != 28 )
      goto LABEL_67;
    v204 = *(_QWORD *)(v62 + 40);
    v205 = v56;
    v332.m128i_i64[0] = 0;
    v334 = 8;
    v206 = 0;
    v302 = v204;
    v332.m128i_i64[1] = (__int64)&v336;
    v333 = (__int64 *)&v336;
    LODWORD(v335) = 0;
    v207 = (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) - 1;
    if ( (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) == 1 )
      goto LABEL_493;
    while ( 1 )
    {
      v211 = v206 + 1;
      if ( (*(_BYTE *)(v62 + 23) & 0x40) != 0 )
        v208 = *(_QWORD *)(v62 - 8);
      else
        v208 = v62 - 24LL * (*(_DWORD *)(v62 + 20) & 0xFFFFFFF);
      v209 = *(_QWORD *)(v208 + 24LL * v211);
      if ( !*(_WORD *)(v209 + 18) )
        goto LABEL_369;
      v210 = (__int64 *)v332.m128i_i64[1];
      if ( v333 != (__int64 *)v332.m128i_i64[1] )
        goto LABEL_358;
      v213 = (__int64 *)(v332.m128i_i64[1] + 8LL * HIDWORD(v334));
      if ( (__int64 *)v332.m128i_i64[1] != v213 )
      {
        v214 = 0;
        while ( v209 != *v210 )
        {
          if ( *v210 == -2 )
            v214 = v210;
          if ( v213 == ++v210 )
          {
            if ( !v214 )
              goto LABEL_490;
            *v214 = v209;
            LODWORD(v335) = v335 - 1;
            ++v332.m128i_i64[0];
            goto LABEL_359;
          }
        }
LABEL_369:
        --v207;
        sub_157F2D0(v209, v302, 0);
        sub_1600500(v62, v206);
        v205 = 1;
        goto LABEL_360;
      }
LABEL_490:
      if ( HIDWORD(v334) < (unsigned int)v334 )
      {
        ++HIDWORD(v334);
        *v213 = v209;
        ++v332.m128i_i64[0];
      }
      else
      {
LABEL_358:
        v291 = v205;
        sub_16CCBA0((__int64)&v332, v209);
        v211 = v206 + 1;
        v205 = v291;
        if ( !v212 )
          goto LABEL_369;
      }
LABEL_359:
      v206 = v211;
LABEL_360:
      if ( v206 == v207 )
      {
        v238 = v205;
        v239 = *(_DWORD *)(v62 + 20) & 0xFFFFFFF;
        if ( (_DWORD)v239 != 1 )
        {
          if ( (_DWORD)v239 != 2 )
          {
            if ( (*(_BYTE *)(v62 + 23) & 0x40) != 0 )
              v240 = *(_QWORD *)(v62 - 8);
            else
              v240 = v62 - 24 * v239;
            v241 = *(_QWORD *)v240;
            if ( *(_BYTE *)(*(_QWORD *)v240 + 16LL) == 79 )
            {
              v277 = *(_QWORD *)(v241 - 48);
              if ( *(_BYTE *)(v277 + 16) == 4 )
              {
                v278 = *(_QWORD *)(v241 - 24);
                if ( *(_BYTE *)(v278 + 16) == 4 )
                {
                  if ( (unsigned __int8)sub_1B45090(
                                          v62,
                                          *(_QWORD *)(v241 - 72),
                                          *(_QWORD *)(v277 - 24),
                                          *(_QWORD *)(v278 - 24),
                                          0,
                                          0) )
                  {
                    sub_1B5E140(v302, v305, v307, v308, v310);
                    goto LABEL_488;
                  }
                }
              }
            }
            if ( v333 != (__int64 *)v332.m128i_i64[1] )
              _libc_free((unsigned __int64)v333);
            if ( v238 )
LABEL_66:
              v35 = 1;
            goto LABEL_67;
          }
          v281 = *(_QWORD *)(sub_13CF970(v62) + 24);
          v282 = sub_1648A60(56, 1u);
          if ( v282 )
            sub_15F8320((__int64)v282, v281, v62);
LABEL_495:
          sub_1B44FE0(v62);
LABEL_488:
          v35 = 1;
          if ( v333 != (__int64 *)v332.m128i_i64[1] )
          {
            _libc_free((unsigned __int64)v333);
            goto LABEL_66;
          }
          goto LABEL_67;
        }
LABEL_493:
        v279 = sub_16498A0(v62);
        v280 = sub_1648A60(56, 0);
        if ( v280 )
          sub_15F82A0((__int64)v280, v279, v62);
        goto LABEL_495;
      }
    }
  }
  v300 = *(_QWORD *)(v62 + 40);
  if ( (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) == 1 )
  {
    v294 = *(_QWORD *)(v62 - 24);
    v85 = sub_157EBA0(v294);
    if ( *(_BYTE *)(v85 + 16) == 26 )
    {
      v332.m128i_i64[0] = *(_QWORD *)(v300 + 8);
      sub_15CDD40(v332.m128i_i64);
      if ( v332.m128i_i64[0] )
      {
        if ( (*(_DWORD *)(v85 + 20) & 0xFFFFFFF) == 3 && (unsigned __int8)sub_1B432F0(v85) )
          goto LABEL_67;
      }
    }
    v86 = *(_QWORD *)(*(_QWORD *)(sub_157EBA0(v300) - 24) + 8LL);
    if ( v86 )
    {
      while ( 1 )
      {
        v87 = sub_1648700(v86);
        if ( (unsigned __int8)(*((_BYTE *)v87 + 16) - 25) <= 9u )
          break;
        v86 = *(_QWORD *)(v86 + 8);
        if ( !v86 )
          goto LABEL_110;
      }
LABEL_122:
      v91 = sub_157EBA0(v87[5]);
      if ( *(_BYTE *)(v91 + 16) == 26 && (*(_DWORD *)(v91 + 20) & 0xFFFFFFF) == 3 && (unsigned __int8)sub_1B432F0(v91) )
        goto LABEL_67;
      while ( 1 )
      {
        v86 = *(_QWORD *)(v86 + 8);
        if ( !v86 )
          break;
        v87 = sub_1648700(v86);
        if ( (unsigned __int8)(*((_BYTE *)v87 + 16) - 25) <= 9u )
          goto LABEL_122;
      }
    }
LABEL_110:
    v88 = *((_BYTE *)v310 + 6);
    if ( v88 )
    {
      if ( !v309 )
        goto LABEL_169;
      v89 = *(_QWORD *)(v300 + 8);
      if ( !v89 )
        goto LABEL_169;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v89) + 16) - 25) > 9u )
      {
        v89 = *(_QWORD *)(v89 + 8);
        if ( !v89 )
          goto LABEL_169;
      }
      v90 = 0;
      while ( 1 )
      {
        v89 = *(_QWORD *)(v89 + 8);
        if ( !v89 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v89) + 16) - 25) <= 9u )
        {
          v89 = *(_QWORD *)(v89 + 8);
          ++v90;
          if ( !v89 )
            goto LABEL_118;
        }
      }
LABEL_118:
      if ( (unsigned int)(v90 + 1) > 1 )
      {
        if ( !sub_183E920(v309, v300) )
          v88 = sub_183E920(v309, v294);
      }
      else
      {
LABEL_169:
        v88 = 0;
      }
    }
    v110 = sub_157ED60(v300);
    j = v110 + 24;
    v114 = *(unsigned __int8 *)(v110 + 16);
    if ( (unsigned int)(v114 - 25) <= 9 )
    {
      v115 = *(_QWORD *)(*(_QWORD *)(v300 + 56) + 80LL);
      if ( (!v115 || v300 != v115 - 24) && !v88 )
      {
        if ( (unsigned __int8)sub_1AF2B30(v300, 0, a7, a8, a9, *(double *)a10.m128i_i64, v111, v112, a13, a14) )
          goto LABEL_66;
        LOBYTE(v114) = *(_BYTE *)(v110 + 16);
      }
    }
    if ( (_BYTE)v114 == 75 )
    {
      v116 = *(unsigned __int16 *)(v110 + 18);
      BYTE1(v116) &= ~0x80u;
      if ( (unsigned int)(v116 - 32) > 1 || *(_BYTE *)(*(_QWORD *)(v110 - 24) + 16LL) != 13 )
        goto LABEL_176;
      for ( j = *(_QWORD *)(v110 + 32); ; j = *(_QWORD *)(j + 8) )
      {
        if ( !j )
          goto LABEL_508;
        LOBYTE(v114) = *(_BYTE *)(j - 8);
        if ( (_BYTE)v114 != 78 )
          break;
        v171 = *(_QWORD *)(j - 48);
        if ( *(_BYTE *)(v171 + 16)
          || (*(_BYTE *)(v171 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v171 + 36) - 35) > 3 )
        {
          goto LABEL_212;
        }
      }
      if ( (unsigned int)(unsigned __int8)v114 - 25 <= 9 )
      {
        if ( (unsigned __int8)sub_1B63B00(
                                v110,
                                (unsigned int)&v311,
                                v306,
                                (_DWORD)v305,
                                (_DWORD)v307,
                                v308,
                                (__int64)v310) )
          goto LABEL_66;
        LOBYTE(v114) = *(_BYTE *)(j - 8);
      }
    }
LABEL_212:
    if ( (_BYTE)v114 != 88 )
      goto LABEL_176;
    for ( k = *(_QWORD *)(j + 8); ; k = *(_QWORD *)(k + 8) )
    {
      if ( !k )
        goto LABEL_508;
      v139 = *(unsigned __int8 *)(k - 8);
      if ( (_BYTE)v139 != 78 )
        break;
      v148 = *(_QWORD *)(k - 48);
      if ( *(_BYTE *)(v148 + 16)
        || (*(_BYTE *)(v148 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v148 + 36) - 35) > 3 )
      {
        goto LABEL_176;
      }
    }
    if ( (unsigned int)(v139 - 25) > 9 )
    {
LABEL_176:
      v103 = sub_1B5C580(
               v62,
               *v310,
               a7,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               *(double *)a10.m128i_i64,
               v111,
               v112,
               a13,
               a14);
      if ( !(_BYTE)v103 )
        goto LABEL_67;
      goto LABEL_159;
    }
    v140 = sub_157F210(v300);
    v141 = *(_QWORD *)(v140 + 48);
    if ( v141 )
    {
      if ( *(_BYTE *)(v141 - 8) == 77 )
        goto LABEL_176;
      v332.m128i_i64[0] = *(_QWORD *)(v140 + 8);
      sub_15CDD40(v332.m128i_i64);
      v142 = v332.m128i_i64[0];
      if ( !v332.m128i_i64[0] )
        goto LABEL_176;
      v143 = sub_1648700(v332.m128i_i64[0]);
      v296 = v35;
      v144 = v142;
      v145 = j - 24;
LABEL_222:
      v146 = v143[5];
      if ( v146 == v300 )
        goto LABEL_226;
      v147 = *(_QWORD *)(v146 + 48);
      if ( v147 )
      {
        if ( *(_BYTE *)(v147 - 8) != 88 || !sub_15F41F0(v147 - 24, v145) )
        {
LABEL_226:
          while ( 1 )
          {
            v144 = *(_QWORD *)(v144 + 8);
            if ( !v144 )
              break;
            v143 = sub_1648700(v144);
            if ( (unsigned __int8)(*((_BYTE *)v143 + 16) - 25) <= 9u )
              goto LABEL_222;
          }
          v35 = v296;
          goto LABEL_176;
        }
        for ( m = *(_QWORD *)(v147 + 8); ; m = *(_QWORD *)(m + 8) )
        {
          if ( !m )
            goto LABEL_508;
          v152 = *(_BYTE *)(m - 8);
          if ( v152 != 78 )
            break;
          v163 = *(_QWORD *)(m - 48);
          if ( *(_BYTE *)(v163 + 16)
            || (*(_BYTE *)(v163 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v163 + 36) - 35) > 3 )
          {
            goto LABEL_226;
          }
        }
        if ( v152 != 26 )
          goto LABEL_226;
        v153 = sub_15F41F0(m - 24, v62);
        if ( !v153 )
          goto LABEL_226;
        v288 = v153;
        v154 = v146;
        v327.m128i_i64[1] = (__int64)v331;
        v328 = v331;
        v327.m128i_i64[0] = 0;
        v329 = 16;
        LODWORD(v330) = 0;
        v332.m128i_i64[0] = *(_QWORD *)(v300 + 8);
        sub_15CDD40(v332.m128i_i64);
        v155 = v332.m128i_i64[0];
        if ( v332.m128i_i64[0] )
        {
          v156 = sub_1648700(v332.m128i_i64[0]);
          v157 = v328;
          v158 = (__int64 *)v327.m128i_i64[1];
LABEL_253:
          v159 = v156[5];
          if ( v157 != v158 )
            goto LABEL_250;
          v160 = &v157[HIDWORD(v329)];
          if ( v160 != v157 )
          {
            v161 = v157;
            v162 = 0;
            do
            {
              if ( v159 == *v161 )
                goto LABEL_251;
              if ( *v161 == -2 )
                v162 = v161;
              ++v161;
            }
            while ( v160 != v161 );
            if ( v162 )
            {
              *v162 = v159;
              v157 = v328;
              LODWORD(v330) = v330 - 1;
              v158 = (__int64 *)v327.m128i_i64[1];
              ++v327.m128i_i64[0];
              goto LABEL_251;
            }
          }
          if ( HIDWORD(v329) >= (unsigned int)v329 )
          {
LABEL_250:
            sub_16CCBA0((__int64)&v327, v159);
            v157 = v328;
            v158 = (__int64 *)v327.m128i_i64[1];
            goto LABEL_251;
          }
          ++HIDWORD(v329);
          *v160 = v159;
          v158 = (__int64 *)v327.m128i_i64[1];
          ++v327.m128i_i64[0];
          v157 = v328;
LABEL_251:
          while ( 1 )
          {
            v155 = *(_QWORD *)(v155 + 8);
            if ( !v155 )
              break;
            v156 = sub_1648700(v155);
            if ( (unsigned __int8)(*((_BYTE *)v156 + 16) - 25) <= 9u )
              goto LABEL_253;
          }
          v215 = v158;
          v154 = v146;
        }
        else
        {
          v157 = v328;
          v215 = (__int64 *)v327.m128i_i64[1];
        }
        if ( v157 == v215 )
          v216 = &v157[HIDWORD(v329)];
        else
          v216 = &v157[(unsigned int)v329];
        while ( v216 != v157 )
        {
          v217 = *v157;
          v218 = v157;
          if ( (unsigned __int64)*v157 < 0xFFFFFFFFFFFFFFFELL )
          {
            if ( v216 != v157 )
            {
              v242 = v154 + 8;
              v243 = v154;
              v244 = v216;
              do
              {
                v245 = sub_157EBA0(v217);
                v246 = v245 - 24;
                if ( *(_QWORD *)(v245 - 24) )
                {
                  v247 = *(_QWORD *)(v245 - 16);
                  v248 = *(_QWORD *)(v245 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v248 = v247;
                  if ( v247 )
                    *(_QWORD *)(v247 + 16) = *(_QWORD *)(v247 + 16) & 3LL | v248;
                }
                *(_QWORD *)(v245 - 24) = v243;
                v249 = *(_QWORD *)(v243 + 8);
                *(_QWORD *)(v245 - 16) = v249;
                if ( v249 )
                  *(_QWORD *)(v249 + 16) = (v245 - 16) | *(_QWORD *)(v249 + 16) & 3LL;
                *(_QWORD *)(v245 - 8) = v242 | *(_QWORD *)(v245 - 8) & 3LL;
                v250 = v218 + 1;
                *(_QWORD *)(v243 + 8) = v246;
                if ( v218 + 1 == v244 )
                  break;
                while ( 1 )
                {
                  v217 = *v250;
                  v218 = v250;
                  if ( (unsigned __int64)*v250 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v244 == ++v250 )
                    goto LABEL_428;
                }
              }
              while ( v244 != v250 );
LABEL_428:
              v154 = v243;
            }
            break;
          }
          ++v157;
        }
        v219 = *(_QWORD *)(v154 + 48);
        v220 = v154 + 40;
        if ( v219 == v220 )
        {
LABEL_382:
          v332.m128i_i64[0] = 0;
          v332.m128i_i64[1] = (__int64)&v336;
          v333 = (__int64 *)&v336;
          v334 = 16;
          LODWORD(v335) = 0;
          v222 = sub_157EBA0(v300);
          if ( v222 )
          {
            v223 = sub_15F4D60(v222);
            v224 = sub_157EBA0(v300);
            if ( v223 )
            {
              v225 = 0;
              while ( 1 )
              {
                v228 = sub_15F4DF0(v224, v225);
                v226 = v333;
                v227 = v332.m128i_i64[1];
                v229 = v228;
                if ( v333 != (__int64 *)v332.m128i_i64[1] )
                  goto LABEL_385;
                v230 = &v333[HIDWORD(v334)];
                if ( v333 != v230 )
                {
                  v231 = v333;
                  v232 = 0;
                  while ( v229 != *v231 )
                  {
                    if ( *v231 == -2 )
                      v232 = v231;
                    if ( v230 == ++v231 )
                    {
                      if ( !v232 )
                        goto LABEL_431;
                      *v232 = v229;
                      v226 = v333;
                      LODWORD(v335) = v335 - 1;
                      v227 = v332.m128i_i64[1];
                      ++v332.m128i_i64[0];
                      goto LABEL_386;
                    }
                  }
                  goto LABEL_386;
                }
LABEL_431:
                if ( HIDWORD(v334) < (unsigned int)v334 )
                {
                  ++HIDWORD(v334);
                  *v230 = v229;
                  v227 = v332.m128i_i64[1];
                  ++v332.m128i_i64[0];
                  v226 = v333;
                }
                else
                {
LABEL_385:
                  sub_16CCBA0((__int64)&v332, v229);
                  v226 = v333;
                  v227 = v332.m128i_i64[1];
                }
LABEL_386:
                if ( v223 == ++v225 )
                  goto LABEL_434;
              }
            }
            v226 = v333;
            v227 = v332.m128i_i64[1];
LABEL_434:
            if ( (__int64 *)v227 != v226 )
            {
              v251 = &v226[(unsigned int)v334];
LABEL_436:
              v252 = v226;
              if ( v251 != v226 )
              {
                while ( 1 )
                {
                  v253 = *v252;
                  v254 = v252;
                  if ( (unsigned __int64)*v252 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v251 == ++v252 )
                    goto LABEL_439;
                }
                if ( v251 != v252 )
                {
                  do
                  {
                    sub_157F2D0(v253, v300, 0);
                    v261 = v254 + 1;
                    if ( v254 + 1 == v251 )
                      break;
                    while ( 1 )
                    {
                      v253 = *v261;
                      v254 = v261;
                      if ( (unsigned __int64)*v261 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v251 == ++v261 )
                        goto LABEL_439;
                    }
                  }
                  while ( v251 != v261 );
                }
              }
LABEL_439:
              v255 = sub_16498A0(v62);
              v319 = 0;
              v322 = v255;
              v321 = 0;
              v323 = 0;
              v324 = 0;
              v325 = 0;
              v326 = 0;
              v320 = 0;
              sub_17050D0((__int64 *)&v319, v62);
              v304 = 257;
              v256 = sub_1648A60(56, 0);
              v257 = (__int64)v256;
              if ( v256 )
                sub_15F82A0((__int64)v256, v322, 0);
              if ( v320 )
              {
                v258 = v321;
                sub_157E9D0(v320 + 40, v257);
                v259 = *(_QWORD *)(v257 + 24);
                v260 = *v258;
                *(_QWORD *)(v257 + 32) = v258;
                v260 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v257 + 24) = v260 | v259 & 7;
                *(_QWORD *)(v260 + 8) = v257 + 24;
                *v258 = *v258 & 7 | (v257 + 24);
              }
              sub_164B780(v257, &v303);
              sub_12A86E0((__int64 *)&v319, v257);
              sub_15F20C0((_QWORD *)v62);
              if ( v319 )
                sub_161E7C0((__int64)&v319, (__int64)v319);
              if ( v333 != (__int64 *)v332.m128i_i64[1] )
                _libc_free((unsigned __int64)v333);
              if ( v328 != (__int64 *)v327.m128i_i64[1] )
                _libc_free((unsigned __int64)v328);
              v35 = v288;
              goto LABEL_67;
            }
          }
          else
          {
            v226 = (__int64 *)&v336;
          }
          v251 = &v226[HIDWORD(v334)];
          goto LABEL_436;
        }
        while ( v219 )
        {
          v221 = *(_QWORD *)(v219 + 8);
          if ( *(_BYTE *)(v219 - 8) == 78 )
          {
            v233 = *(_QWORD *)(v219 - 48);
            if ( !*(_BYTE *)(v233 + 16)
              && (*(_BYTE *)(v233 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v233 + 36) - 35) <= 3 )
            {
              sub_15F20C0((_QWORD *)(v219 - 24));
            }
          }
          if ( v220 == v221 )
            goto LABEL_382;
          v219 = v221;
        }
      }
    }
LABEL_508:
    BUG();
  }
  v66 = *(_QWORD *)(v300 + 56);
  if ( v66 && (unsigned __int8)sub_1560180(v66 + 112, 33) )
    goto LABEL_67;
  if ( sub_1B44C50((__int64)&v305, v62) )
  {
    v67 = sub_157F0B0(v300);
    if ( v67 && (unsigned __int8)sub_1B45D90((__int64)&v305, v62, v67, &v311) )
    {
      sub_1B5E140(v300, v305, v307, v308, v310);
      goto LABEL_66;
    }
    sub_1580910(&v332);
    v327 = v332;
    sub_1974F30((__int64)&v328, (__int64)&v333);
    sub_A17130((__int64)v339);
    v68 = &v333;
    sub_A17130((__int64)&v333);
    v71 = v327.m128i_i64[0];
    if ( !v327.m128i_i64[0] )
    {
      v72 = 0;
      goto LABEL_76;
    }
    v72 = v327.m128i_i64[0] - 24;
    if ( v62 == v327.m128i_i64[0] - 24 )
    {
LABEL_135:
      if ( sub_1B5AB40((__int64)&v305, v62, &v311, a7, a8, a9, a10, v69, v70, a13, a14) )
      {
        sub_1B5E140(v300, v305, v307, v308, v310);
        sub_A17130((__int64)&v328);
        goto LABEL_66;
      }
    }
    else
    {
LABEL_76:
      if ( *(_QWORD *)(v62 - 72) == v72 )
      {
        while ( 1 )
        {
          v92 = *(_QWORD *)(v71 + 8);
          v327.m128i_i64[0] = v92;
          if ( v327.m128i_i64[1] == v92 )
            break;
          if ( v92 )
            v92 -= 24;
          if ( !v330 )
            sub_4263D6(v68, v92, v72);
          v68 = &v328;
          if ( ((unsigned __int8 (__fastcall *)(__int64 **, __int64))v331[0])(&v328, v92) )
          {
            v92 = v327.m128i_i64[0];
            break;
          }
          v71 = v327.m128i_i64[0];
        }
        if ( v92 && v62 == v92 - 24 )
          goto LABEL_135;
      }
    }
    sub_A17130((__int64)&v328);
  }
  v73 = sub_1B4A120(v62, &v311, v306);
  if ( (_BYTE)v73 )
  {
    v35 = v73;
    goto LABEL_67;
  }
  if ( (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) != 1 )
  {
    v93 = *(_QWORD *)(v62 - 72);
    if ( *(_BYTE *)(v93 + 16) == 76 )
    {
      v94 = *(_QWORD *)(v93 - 48);
      if ( v94 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v93 - 24) + 16LL) == 14 )
        {
          v95 = *((_QWORD *)v310 + 2);
          if ( !*(_BYTE *)(v95 + 184) )
          {
            v299 = *(_QWORD *)(v93 - 48);
            sub_14CDF70(*((_QWORD *)v310 + 2));
            v94 = v299;
          }
          v96 = *(_QWORD *)(v95 + 8);
          v97 = v96 + 32LL * *(unsigned int *)(v95 + 16);
          if ( v97 != v96 )
          {
            v295 = v35;
            v98 = v94;
            while ( 1 )
            {
              v99 = *(_QWORD *)(v96 + 16);
              if ( v99 )
              {
                if ( *(_BYTE *)(v99 + 16) == 78 )
                {
                  v100 = *(_QWORD *)(v99 - 24);
                  if ( !*(_BYTE *)(v100 + 16) && (*(_BYTE *)(v100 + 33) & 0x20) != 0 )
                  {
                    v101 = *(_QWORD *)(v99 - 24LL * (*(_DWORD *)(v99 + 20) & 0xFFFFFFF));
                    if ( *(_BYTE *)(v101 + 16) == 76 )
                    {
                      v102 = *(_QWORD *)(v101 - 48);
                      if ( v102 )
                      {
                        if ( *(_BYTE *)(*(_QWORD *)(v101 - 24) + 16LL) == 14
                          && v98 == v102
                          && (unsigned __int8)sub_1B56760(*(__int64 **)(v99 + 40), *(_QWORD *)(v62 + 40)) )
                        {
                          break;
                        }
                      }
                    }
                  }
                }
              }
              v96 += 32;
              if ( v97 == v96 )
              {
                v35 = v295;
                goto LABEL_179;
              }
            }
            v35 = v295;
            sub_14BCF40((bool *)v332.m128i_i8, v101, *(_QWORD *)(v62 - 72), v306, 1u, 0);
            v103 = v332.m128i_u8[1];
            if ( !v332.m128i_i8[1] )
              goto LABEL_179;
            v104 = *(_QWORD *)(v62 - 72);
            v105 = v300;
            if ( v332.m128i_i8[0] )
            {
LABEL_157:
              v106 = (__int64 *)sub_157E9C0(v105);
              v107 = sub_159C4F0(v106);
              goto LABEL_158;
            }
LABEL_279:
            v166 = (__int64 *)sub_157E9C0(v105);
            v107 = sub_159C540(v166);
LABEL_158:
            sub_1593B40((_QWORD *)(v62 - 72), v107);
            sub_1AEB370(v104, 0);
            goto LABEL_159;
          }
        }
      }
    }
  }
LABEL_179:
  v117 = sub_157F0B0(v300);
  if ( v117 )
  {
    v120 = sub_157EBA0(v117);
    if ( v120 )
    {
      if ( *(_BYTE *)(v120 + 16) == 26 && (*(_DWORD *)(v120 + 20) & 0xFFFFFFF) == 3 )
      {
        v167 = *(_QWORD *)(v120 - 24);
        if ( v167 != *(_QWORD *)(v120 - 48) )
        {
          sub_14BCF40((bool *)v332.m128i_i8, *(_QWORD *)(v120 - 72), *(_QWORD *)(v62 - 72), v306, v300 == v167, 0);
          v103 = v332.m128i_u8[1];
          if ( v332.m128i_i8[1] )
          {
            v104 = *(_QWORD *)(v62 - 72);
            if ( v332.m128i_i8[0] )
            {
              v105 = v300;
              goto LABEL_157;
            }
            v105 = v300;
            goto LABEL_279;
          }
        }
      }
    }
  }
  v103 = sub_1B5C580(
           v62,
           *v310,
           a7,
           *(double *)a8.m128i_i64,
           *(double *)a9.m128i_i64,
           *(double *)a10.m128i_i64,
           v118,
           v119,
           a13,
           a14);
  if ( (_BYTE)v103 )
  {
LABEL_159:
    v35 = v103;
    sub_1B5E140(v300, v305, v307, v308, v310);
    goto LABEL_67;
  }
  v121 = sub_157F0B0(*(_QWORD *)(v62 - 24));
  v122 = *(_QWORD *)(v62 - 48);
  if ( v121 )
  {
    if ( sub_157F0B0(v122) )
    {
      if ( (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) != 3 || !(unsigned __int8)sub_1B432F0(v62) )
      {
        v103 = sub_1B4F570(
                 v62,
                 (__int64)v305,
                 a7,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 *(double *)a10.m128i_i64,
                 v123,
                 v124,
                 a13,
                 a14);
        if ( (_BYTE)v103 )
          goto LABEL_159;
      }
    }
    else
    {
      v125 = sub_157EBA0(*(_QWORD *)(v62 - 24));
      if ( (unsigned int)sub_15F4D60(v125) == 1 && *(_QWORD *)(v62 - 48) == sub_15F4DF0(v125, 0) )
      {
        v165 = *(_QWORD *)(v62 - 24);
        goto LABEL_276;
      }
    }
  }
  else if ( sub_157F0B0(v122) )
  {
    v164 = sub_157EBA0(*(_QWORD *)(v62 - 48));
    if ( (unsigned int)sub_15F4D60(v164) == 1 && *(_QWORD *)(v62 - 24) == sub_15F4DF0(v164, 0) )
    {
      v165 = *(_QWORD *)(v62 - 48);
LABEL_276:
      v103 = sub_1B534C0((_QWORD *)v62, v165, v305);
      if ( (_BYTE)v103 )
        goto LABEL_159;
    }
  }
  v126 = *(_QWORD *)(v62 - 72);
  if ( *(_BYTE *)(v126 + 16) == 77 && *(_QWORD *)(v126 + 40) == *(_QWORD *)(v62 + 40) )
  {
    v103 = sub_1B54A50(
             v62,
             v306,
             v308,
             *((_QWORD *)v310 + 2),
             a7,
             *(double *)a8.m128i_i64,
             *(double *)a9.m128i_i64,
             *(double *)a10.m128i_i64,
             v123,
             v124,
             a13,
             a14);
    if ( (_BYTE)v103 )
      goto LABEL_159;
  }
  v332.m128i_i64[0] = *(_QWORD *)(v300 + 8);
  sub_15CDD40(v332.m128i_i64);
  v127 = v332.m128i_i64[0];
  if ( v332.m128i_i64[0] )
  {
    v128 = sub_1648700(v332.m128i_i64[0]);
LABEL_191:
    v129 = sub_157EBA0(v128[5]);
    if ( *(_BYTE *)(v129 + 16) == 26 && v62 != v129 && (*(_DWORD *)(v129 + 20) & 0xFFFFFFF) == 3 )
    {
      v137 = sub_1B55A30(
               v129,
               v62,
               v306,
               v307,
               a7,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               *(double *)a10.m128i_i64,
               v130,
               v131,
               a13,
               a14);
      if ( (_BYTE)v137 )
      {
        v35 = v137;
        sub_1B5E140(v300, v305, v307, v308, v310);
        goto LABEL_67;
      }
    }
    while ( 1 )
    {
      v127 = *(_QWORD *)(v127 + 8);
      if ( !v127 )
        break;
      v128 = sub_1648700(v127);
      if ( (unsigned __int8)(*((_BYTE *)v128 + 16) - 25) <= 9u )
        goto LABEL_191;
    }
  }
  if ( byte_4FB7300 )
  {
    v332.m128i_i64[0] = *(_QWORD *)(v300 + 8);
    sub_15CDD40(v332.m128i_i64);
    v132 = v332.m128i_i64[0];
    if ( v332.m128i_i64[0] )
    {
      v133 = 0;
      while ( 1 )
      {
        v134 = sub_1648700(v132);
        v135 = v133;
        v136 = sub_157F0B0(v134[5]);
        v133 = v136;
        if ( !v136 || v135 && v136 != v135 )
          break;
        v332.m128i_i64[0] = *(_QWORD *)(v332.m128i_i64[0] + 8);
        sub_15CDD40(v332.m128i_i64);
        v132 = v332.m128i_i64[0];
        if ( !v332.m128i_i64[0] )
        {
          v168 = sub_157EBA0(v133);
          if ( *(_BYTE *)(v168 + 16) != 26 )
            break;
          if ( v62 == v168 )
            break;
          if ( (*(_DWORD *)(v168 + 20) & 0xFFFFFFF) != 3 )
            break;
          v103 = sub_1B4BCE0(
                   v168,
                   v62,
                   v306,
                   a7,
                   *(double *)a8.m128i_i64,
                   *(double *)a9.m128i_i64,
                   *(double *)a10.m128i_i64,
                   v169,
                   v170,
                   a13,
                   a14);
          if ( !(_BYTE)v103 )
            break;
          goto LABEL_159;
        }
      }
    }
  }
LABEL_67:
  if ( v311 )
    sub_161E7C0((__int64)&v311, v311);
  return v35;
}
