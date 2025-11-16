// Function: sub_1C82A50
// Address: 0x1c82a50
//
void __fastcall sub_1C82A50(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        __int64 **a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        unsigned int a15,
        unsigned __int8 a16,
        unsigned __int8 a17,
        __int64 a18,
        __int64 a19)
{
  _QWORD *v20; // rbx
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  unsigned __int64 v24; // r14
  __int64 v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rbx
  _QWORD *v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  _QWORD *v33; // r15
  __int64 **v34; // r15
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rcx
  __int64 v38; // rax
  __int64 v39; // r14
  _QWORD *v40; // rax
  _QWORD *v41; // rbx
  unsigned __int64 *v42; // r14
  __int64 v43; // rax
  unsigned __int64 v44; // rcx
  __int64 v45; // rsi
  unsigned __int8 *v46; // rsi
  __int64 v47; // rax
  bool v48; // zf
  _QWORD *v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // r14
  _QWORD *v53; // r13
  unsigned __int64 *v54; // r14
  __int64 v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rsi
  unsigned __int8 *v58; // rsi
  __int64 v59; // rax
  _QWORD *v60; // rax
  _QWORD *v61; // r14
  unsigned __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rdx
  unsigned __int8 *v66; // rsi
  _QWORD *v68; // rax
  _QWORD *v69; // rbx
  unsigned __int64 *v70; // r14
  __int64 v71; // rax
  unsigned __int64 v72; // rcx
  unsigned __int8 *v73; // rsi
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 *v76; // r14
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // rcx
  char **v81; // r8
  __int64 v82; // r9
  unsigned __int8 *v83; // rsi
  __int64 v84; // rsi
  int v85; // eax
  __int64 v86; // rax
  int v87; // edx
  __int64 v88; // rdx
  __int64 *v89; // rax
  __int64 v90; // rcx
  unsigned __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // rax
  unsigned __int8 *v96; // rsi
  __int64 v97; // rdx
  __int64 v98; // rcx
  char **v99; // r8
  __int64 v100; // r9
  __int64 v101; // r14
  int v102; // eax
  __int64 v103; // rax
  int v104; // edx
  __int64 v105; // rdx
  __int64 *v106; // rax
  __int64 v107; // rcx
  unsigned __int64 v108; // rdx
  __int64 v109; // rdx
  __int64 v110; // rdx
  __int64 v111; // rcx
  _QWORD *v112; // rax
  _QWORD *v113; // rbx
  unsigned __int64 v114; // rsi
  __int64 v115; // rax
  __int64 v116; // rsi
  __int64 v117; // rdx
  unsigned __int8 *v118; // rsi
  _QWORD *v119; // rax
  __int64 v120; // r9
  __int64 v121; // rdx
  __int64 *v122; // rbx
  __int64 v123; // rcx
  __int64 v124; // rax
  __int64 v125; // r9
  __int64 v126; // rsi
  __int64 v127; // rbx
  unsigned __int8 *v128; // rsi
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // r14
  _QWORD *v132; // rax
  unsigned __int64 *v133; // r14
  __int64 v134; // rax
  unsigned __int64 v135; // rcx
  __int64 v136; // rsi
  unsigned __int8 *v137; // rsi
  __int64 v138; // rax
  _QWORD *v139; // rax
  __int64 v140; // r14
  __int64 v141; // rax
  _QWORD *v142; // rax
  _QWORD *v143; // r15
  unsigned __int64 v144; // rsi
  __int64 v145; // rax
  __int64 v146; // rsi
  __int64 v147; // rdx
  unsigned __int8 *v148; // rsi
  __int64 v149; // rax
  _QWORD *v150; // rax
  _QWORD *v151; // r14
  unsigned __int64 v152; // rsi
  __int64 v153; // rax
  __int64 v154; // rsi
  __int64 v155; // rdx
  unsigned __int8 *v156; // rsi
  _QWORD *v157; // rax
  _QWORD *v158; // rbx
  unsigned __int64 *v159; // r14
  __int64 v160; // rax
  unsigned __int64 v161; // rcx
  __int64 v162; // rsi
  unsigned __int8 *v163; // rsi
  __int64 v164; // rax
  __int64 v165; // rbx
  __int64 *v166; // r14
  __int64 v167; // rax
  __int64 v168; // rcx
  __int64 v169; // rsi
  unsigned __int8 *v170; // rsi
  __int64 v171; // rdx
  __int64 v172; // rcx
  __int64 v173; // r8
  __int64 v174; // r9
  __int64 v175; // r14
  int v176; // eax
  __int64 v177; // rax
  int v178; // edx
  __int64 v179; // rdx
  __int64 *v180; // rax
  __int64 v181; // rcx
  unsigned __int64 v182; // rdx
  __int64 v183; // rdx
  __int64 v184; // rdx
  __int64 v185; // rcx
  _QWORD *v186; // rax
  __int64 v187; // r10
  __int64 v188; // rdx
  __int64 *v189; // r14
  __int64 v190; // rcx
  __int64 v191; // rax
  __int64 v192; // r10
  __int64 v193; // rsi
  __int64 v194; // r14
  unsigned __int8 *v195; // rsi
  _QWORD *v196; // rax
  _QWORD *v197; // r14
  unsigned __int64 v198; // rsi
  __int64 v199; // rax
  __int64 v200; // rsi
  __int64 v201; // rdx
  unsigned __int8 *v202; // rsi
  __int64 v203; // rax
  unsigned __int8 *v204; // rsi
  __int64 v205; // rdx
  __int64 v206; // rcx
  char **v207; // r8
  __int64 v208; // r9
  __int64 v209; // r14
  int v210; // eax
  __int64 v211; // rax
  int v212; // edx
  __int64 v213; // rdx
  __int64 *v214; // rax
  __int64 v215; // rcx
  unsigned __int64 v216; // rdx
  __int64 v217; // rdx
  __int64 v218; // rdx
  __int64 v219; // rcx
  __int64 v220; // rax
  __int64 v221; // r14
  _QWORD *v222; // rax
  _QWORD *v223; // rbx
  unsigned __int64 *v224; // r14
  __int64 v225; // rax
  unsigned __int64 v226; // rcx
  __int64 v227; // rsi
  unsigned __int8 *v228; // rsi
  __int64 v229; // rax
  __int64 *v230; // r15
  __int64 v231; // rax
  __int64 v232; // rsi
  __int64 v233; // rsi
  __int64 v234; // rax
  __int64 v235; // rsi
  __int64 v236; // rax
  __int64 v237; // rsi
  __int64 v238; // rax
  __int64 *v239; // rbx
  __int64 v240; // rax
  __int64 v241; // rcx
  __int64 v242; // rsi
  unsigned __int8 *v243; // rsi
  __int64 v244; // rax
  __int64 v245; // r15
  __int64 *v246; // rbx
  __int64 v247; // rax
  __int64 v248; // rcx
  __int64 v249; // rsi
  unsigned __int8 *v250; // rsi
  __int64 v251; // rax
  __int64 v252; // r15
  __int64 *v253; // rbx
  __int64 v254; // rax
  __int64 v255; // rcx
  __int64 v256; // rsi
  unsigned __int8 *v257; // rsi
  __int64 v258; // rax
  __int64 v259; // rcx
  __int64 *v260; // rbx
  __int64 v261; // rsi
  __int64 v262; // rax
  __int64 v263; // rsi
  __int64 v264; // rbx
  unsigned __int8 *v265; // rsi
  _QWORD *v266; // [rsp+18h] [rbp-178h]
  __int64 v267; // [rsp+20h] [rbp-170h]
  __int64 v268; // [rsp+28h] [rbp-168h]
  unsigned __int64 *v269; // [rsp+28h] [rbp-168h]
  __int64 v270; // [rsp+28h] [rbp-168h]
  __int64 *v271; // [rsp+28h] [rbp-168h]
  __int64 v272; // [rsp+30h] [rbp-160h]
  __int64 v273; // [rsp+30h] [rbp-160h]
  unsigned __int64 *v274; // [rsp+30h] [rbp-160h]
  __int64 v275; // [rsp+30h] [rbp-160h]
  unsigned __int64 *v276; // [rsp+30h] [rbp-160h]
  __int64 v277; // [rsp+38h] [rbp-158h]
  __int64 v279; // [rsp+40h] [rbp-150h]
  __int64 v280; // [rsp+40h] [rbp-150h]
  unsigned __int64 *v281; // [rsp+40h] [rbp-150h]
  __int64 v282; // [rsp+40h] [rbp-150h]
  _QWORD *v283; // [rsp+40h] [rbp-150h]
  __int64 v284; // [rsp+40h] [rbp-150h]
  __int64 v285; // [rsp+40h] [rbp-150h]
  __int64 v286; // [rsp+40h] [rbp-150h]
  __int64 v287; // [rsp+48h] [rbp-148h]
  __int64 v288; // [rsp+48h] [rbp-148h]
  __int64 v290; // [rsp+50h] [rbp-140h]
  unsigned int v291; // [rsp+58h] [rbp-138h]
  __int64 v292; // [rsp+58h] [rbp-138h]
  __int64 v293; // [rsp+58h] [rbp-138h]
  __int64 v294; // [rsp+58h] [rbp-138h]
  __int64 v297; // [rsp+68h] [rbp-128h]
  _QWORD *v298; // [rsp+68h] [rbp-128h]
  __int64 v299; // [rsp+68h] [rbp-128h]
  __int64 v300; // [rsp+68h] [rbp-128h]
  __int64 v301; // [rsp+68h] [rbp-128h]
  __int64 v302; // [rsp+68h] [rbp-128h]
  unsigned __int64 *v303; // [rsp+68h] [rbp-128h]
  char *v304; // [rsp+78h] [rbp-118h] BYREF
  __int64 v305[2]; // [rsp+80h] [rbp-110h] BYREF
  __int16 v306; // [rsp+90h] [rbp-100h]
  __int64 v307[2]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v308; // [rsp+B0h] [rbp-E0h]
  unsigned __int8 *v309; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v310; // [rsp+C8h] [rbp-C8h]
  __int64 *v311; // [rsp+D0h] [rbp-C0h]
  __int64 v312; // [rsp+D8h] [rbp-B8h]
  __int64 v313; // [rsp+E0h] [rbp-B0h]
  int v314; // [rsp+E8h] [rbp-A8h]
  __int64 v315; // [rsp+F0h] [rbp-A0h]
  __int64 v316; // [rsp+F8h] [rbp-98h]
  char *v317; // [rsp+110h] [rbp-80h] BYREF
  __int64 v318; // [rsp+118h] [rbp-78h]
  __int64 *v319; // [rsp+120h] [rbp-70h]
  __int64 v320; // [rsp+128h] [rbp-68h]
  __int64 v321; // [rsp+130h] [rbp-60h]
  int v322; // [rsp+138h] [rbp-58h]
  __int64 v323; // [rsp+140h] [rbp-50h]
  __int64 v324; // [rsp+148h] [rbp-48h]

  v20 = *(_QWORD **)(a1 + 40);
  v317 = "split";
  LOWORD(v319) = 259;
  v277 = sub_157FBF0(v20, (__int64 *)(a1 + 24), (__int64)&v317);
  v317 = "forward.for";
  LOWORD(v319) = 259;
  v21 = (_QWORD *)sub_22077B0(64);
  v272 = (__int64)v21;
  if ( v21 )
    sub_157FB60(v21, a18, (__int64)&v317, a19, v277);
  v317 = "reverse.for";
  LOWORD(v319) = 259;
  v22 = (_QWORD *)sub_22077B0(64);
  v287 = (__int64)v22;
  if ( v22 )
    sub_157FB60(v22, a18, (__int64)&v317, a19, v272);
  v317 = "nonzerotrip";
  LOWORD(v319) = 259;
  v23 = (_QWORD *)sub_22077B0(64);
  v267 = (__int64)v23;
  if ( v23 )
    sub_157FB60(v23, a18, (__int64)&v317, a19, v287);
  v24 = sub_157EBA0((__int64)v20);
  v25 = sub_16498A0(v24);
  v315 = 0;
  v316 = 0;
  v26 = *(unsigned __int8 **)(v24 + 48);
  v312 = v25;
  v314 = 0;
  v27 = *(_QWORD *)(v24 + 40);
  v309 = 0;
  v310 = v27;
  v311 = (__int64 *)(v24 + 24);
  v313 = 0;
  v317 = (char *)v26;
  if ( v26 )
  {
    sub_1623A60((__int64)&v317, (__int64)v26, 2);
    v309 = (unsigned __int8 *)v317;
    if ( v317 )
      sub_1623210((__int64)&v317, (unsigned __int8 *)v317, (__int64)&v309);
  }
  v308 = 257;
  if ( a3 != *(__int64 ***)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      LOWORD(v319) = 257;
      v251 = sub_15FDBD0(47, a2, (__int64)a3, (__int64)&v317, 0);
      a2 = v251;
      v252 = v251;
      if ( v310 )
      {
        v253 = v311;
        sub_157E9D0(v310 + 40, v251);
        v254 = *(_QWORD *)(v252 + 24);
        v255 = *v253;
        *(_QWORD *)(v252 + 32) = v253;
        v255 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v252 + 24) = v255 | v254 & 7;
        *(_QWORD *)(v255 + 8) = v252 + 24;
        *v253 = *v253 & 7 | (v252 + 24);
      }
      sub_164B780(a2, v307);
      if ( v309 )
      {
        v305[0] = (__int64)v309;
        sub_1623A60((__int64)v305, (__int64)v309, 2);
        v256 = *(_QWORD *)(a2 + 48);
        if ( v256 )
          sub_161E7C0(a2 + 48, v256);
        v257 = (unsigned __int8 *)v305[0];
        *(_QWORD *)(a2 + 48) = v305[0];
        if ( v257 )
          sub_1623210((__int64)v305, v257, a2 + 48);
      }
    }
    else
    {
      a2 = sub_15A46C0(47, (__int64 ***)a2, a3, 0);
    }
  }
  v308 = 257;
  if ( a5 != *(__int64 ***)a4 )
  {
    if ( *(_BYTE *)(a4 + 16) > 0x10u )
    {
      LOWORD(v319) = 257;
      v244 = sub_15FDBD0(47, a4, (__int64)a5, (__int64)&v317, 0);
      a4 = v244;
      v245 = v244;
      if ( v310 )
      {
        v246 = v311;
        sub_157E9D0(v310 + 40, v244);
        v247 = *(_QWORD *)(v245 + 24);
        v248 = *v246;
        *(_QWORD *)(v245 + 32) = v246;
        v248 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v245 + 24) = v248 | v247 & 7;
        *(_QWORD *)(v248 + 8) = v245 + 24;
        *v246 = *v246 & 7 | (v245 + 24);
      }
      sub_164B780(a4, v307);
      if ( v309 )
      {
        v305[0] = (__int64)v309;
        sub_1623A60((__int64)v305, (__int64)v309, 2);
        v249 = *(_QWORD *)(a4 + 48);
        if ( v249 )
          sub_161E7C0(a4 + 48, v249);
        v250 = (unsigned __int8 *)v305[0];
        *(_QWORD *)(a4 + 48) = v305[0];
        if ( v250 )
          sub_1623210((__int64)v305, v250, a4 + 48);
      }
    }
    else
    {
      a4 = sub_15A46C0(47, (__int64 ***)a4, a5, 0);
    }
  }
  LOWORD(v319) = 257;
  v28 = sub_15A0680(*(_QWORD *)a6, 0, 0);
  v29 = sub_12AA0C0((__int64 *)&v309, 0x22u, (_BYTE *)a6, v28, (__int64)&v317);
  v30 = sub_1648A60(56, 3u);
  v33 = v30;
  if ( v30 )
    sub_15F83E0((__int64)v30, v267, v277, v29, 0);
  sub_1AA6530(v24, v33, a7, a8, a9, a10, v31, v32, a13, a14);
  v34 = *(__int64 ***)a6;
  v35 = sub_157E9C0(v267);
  v317 = 0;
  v320 = v35;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v318 = v267;
  v319 = (__int64 *)(v267 + 40);
  v306 = 257;
  if ( v34 == *(__int64 ***)a2 )
  {
    v36 = a2;
  }
  else if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v308 = 257;
    v238 = sub_15FDBD0(45, a2, (__int64)v34, (__int64)v307, 0);
    v36 = v238;
    if ( v318 )
    {
      v239 = v319;
      sub_157E9D0(v318 + 40, v238);
      v240 = *(_QWORD *)(v36 + 24);
      v241 = *v239;
      *(_QWORD *)(v36 + 32) = v239;
      v241 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v36 + 24) = v241 | v240 & 7;
      *(_QWORD *)(v241 + 8) = v36 + 24;
      *v239 = *v239 & 7 | (v36 + 24);
    }
    sub_164B780(v36, v305);
    if ( v317 )
    {
      v304 = v317;
      sub_1623A60((__int64)&v304, (__int64)v317, 2);
      v242 = *(_QWORD *)(v36 + 48);
      if ( v242 )
        sub_161E7C0(v36 + 48, v242);
      v243 = (unsigned __int8 *)v304;
      *(_QWORD *)(v36 + 48) = v304;
      if ( v243 )
        sub_1623210((__int64)&v304, v243, v36 + 48);
    }
  }
  else
  {
    v36 = sub_15A46C0(45, (__int64 ***)a2, v34, 0);
  }
  v306 = 257;
  if ( v34 == *(__int64 ***)a4 )
  {
    v37 = a4;
  }
  else if ( *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v308 = 257;
    v258 = sub_15FDBD0(45, a4, (__int64)v34, (__int64)v307, 0);
    v259 = v258;
    if ( v318 )
    {
      v260 = v319;
      v292 = v258;
      sub_157E9D0(v318 + 40, v258);
      v259 = v292;
      v261 = *v260;
      v262 = *(_QWORD *)(v292 + 24);
      *(_QWORD *)(v292 + 32) = v260;
      v261 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v292 + 24) = v261 | v262 & 7;
      *(_QWORD *)(v261 + 8) = v292 + 24;
      *v260 = *v260 & 7 | (v292 + 24);
    }
    v293 = v259;
    sub_164B780(v259, v305);
    v37 = v293;
    if ( v317 )
    {
      v304 = v317;
      sub_1623A60((__int64)&v304, (__int64)v317, 2);
      v37 = v293;
      v263 = *(_QWORD *)(v293 + 48);
      v264 = v293 + 48;
      if ( v263 )
      {
        sub_161E7C0(v293 + 48, v263);
        v37 = v293;
      }
      v265 = (unsigned __int8 *)v304;
      *(_QWORD *)(v37 + 48) = v304;
      if ( v265 )
      {
        v294 = v37;
        sub_1623210((__int64)&v304, v265, v264);
        v37 = v294;
      }
    }
  }
  else
  {
    v37 = sub_15A46C0(45, (__int64 ***)a4, v34, 0);
  }
  v308 = 257;
  v38 = sub_12AA0C0((__int64 *)&v317, 0x24u, (_BYTE *)v36, v37, (__int64)v307);
  v308 = 257;
  v39 = v38;
  v40 = sub_1648A60(56, 3u);
  v41 = v40;
  if ( v40 )
    sub_15F83E0((__int64)v40, v287, v272, v39, 0);
  if ( v318 )
  {
    v42 = (unsigned __int64 *)v319;
    sub_157E9D0(v318 + 40, (__int64)v41);
    v43 = v41[3];
    v44 = *v42;
    v41[4] = v42;
    v44 &= 0xFFFFFFFFFFFFFFF8LL;
    v41[3] = v44 | v43 & 7;
    *(_QWORD *)(v44 + 8) = v41 + 3;
    *v42 = *v42 & 7 | (unsigned __int64)(v41 + 3);
  }
  sub_164B780((__int64)v41, v307);
  if ( v317 )
  {
    v305[0] = (__int64)v317;
    sub_1623A60((__int64)v305, (__int64)v317, 2);
    v45 = v41[6];
    if ( v45 )
      sub_161E7C0((__int64)(v41 + 6), v45);
    v46 = (unsigned __int8 *)v305[0];
    v41[6] = v305[0];
    if ( v46 )
      sub_1623210((__int64)v305, v46, (__int64)(v41 + 6));
    if ( v317 )
      sub_161E7C0((__int64)&v317, (__int64)v317);
  }
  v47 = sub_157E9C0(v287);
  v318 = v287;
  v320 = v47;
  v317 = 0;
  v48 = *(_BYTE *)(a6 + 16) == 13;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v319 = (__int64 *)(v287 + 40);
  if ( v48 )
  {
    v49 = *(_QWORD **)(a6 + 24);
    if ( *(_DWORD *)(a6 + 32) > 0x40u )
      v49 = (_QWORD *)*v49;
    if ( (unsigned int)dword_4FBD560 >= (unsigned __int64)v49 )
    {
      if ( v49 )
      {
        v50 = (unsigned int)((_DWORD)v49 - 1);
        do
        {
          v307[0] = (__int64)"src.memmove.gep.unroll";
          v308 = 259;
          v51 = sub_15A0680((__int64)v34, v50, 0);
          v52 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a2, v51, (__int64)v307);
          v308 = 257;
          v53 = sub_1648A60(64, 1u);
          if ( v53 )
            sub_15F9210((__int64)v53, *(_QWORD *)(*(_QWORD *)v52 + 24LL), v52, 0, a16, 0);
          if ( v318 )
          {
            v54 = (unsigned __int64 *)v319;
            sub_157E9D0(v318 + 40, (__int64)v53);
            v55 = v53[3];
            v56 = *v54;
            v53[4] = v54;
            v56 &= 0xFFFFFFFFFFFFFFF8LL;
            v53[3] = v56 | v55 & 7;
            *(_QWORD *)(v56 + 8) = v53 + 3;
            *v54 = *v54 & 7 | (unsigned __int64)(v53 + 3);
          }
          sub_164B780((__int64)v53, v307);
          if ( v317 )
          {
            v305[0] = (__int64)v317;
            sub_1623A60((__int64)v305, (__int64)v317, 2);
            v57 = v53[6];
            if ( v57 )
              sub_161E7C0((__int64)(v53 + 6), v57);
            v58 = (unsigned __int8 *)v305[0];
            v53[6] = v305[0];
            if ( v58 )
              sub_1623210((__int64)v305, v58, (__int64)(v53 + 6));
          }
          sub_15F8F50((__int64)v53, a15);
          v308 = 259;
          v307[0] = (__int64)"dst.memmove.gep,unroll";
          v59 = sub_15A0680((__int64)v34, v50, 0);
          v268 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a4, v59, (__int64)v307);
          v308 = 257;
          v60 = sub_1648A60(64, 2u);
          v61 = v60;
          if ( v60 )
            sub_15F9650((__int64)v60, (__int64)v53, v268, a17, 0);
          if ( v318 )
          {
            v269 = (unsigned __int64 *)v319;
            sub_157E9D0(v318 + 40, (__int64)v61);
            v62 = *v269;
            v63 = v61[3] & 7LL;
            v61[4] = v269;
            v62 &= 0xFFFFFFFFFFFFFFF8LL;
            v61[3] = v62 | v63;
            *(_QWORD *)(v62 + 8) = v61 + 3;
            *v269 = *v269 & 7 | (unsigned __int64)(v61 + 3);
          }
          sub_164B780((__int64)v61, v307);
          if ( v317 )
          {
            v305[0] = (__int64)v317;
            sub_1623A60((__int64)v305, (__int64)v317, 2);
            v64 = v61[6];
            v65 = (__int64)(v61 + 6);
            if ( v64 )
            {
              sub_161E7C0((__int64)(v61 + 6), v64);
              v65 = (__int64)(v61 + 6);
            }
            v66 = (unsigned __int8 *)v305[0];
            v61[6] = v305[0];
            if ( v66 )
              sub_1623210((__int64)v305, v66, v65);
          }
          sub_15F9450((__int64)v61, a15);
        }
        while ( v50-- != 0 );
      }
      v308 = 257;
      v68 = sub_1648A60(56, 1u);
      v69 = v68;
      if ( v68 )
        sub_15F8320((__int64)v68, v277, 0);
      if ( v318 )
      {
        v70 = (unsigned __int64 *)v319;
        sub_157E9D0(v318 + 40, (__int64)v69);
        v71 = v69[3];
        v72 = *v70;
        v69[4] = v70;
        v72 &= 0xFFFFFFFFFFFFFFF8LL;
        v69[3] = v72 | v71 & 7;
        *(_QWORD *)(v72 + 8) = v69 + 3;
        *v70 = *v70 & 7 | (unsigned __int64)(v69 + 3);
      }
      sub_164B780((__int64)v69, v307);
      v73 = (unsigned __int8 *)v317;
      if ( v317 )
      {
        v305[0] = (__int64)v317;
LABEL_125:
        sub_1623A60((__int64)v305, (__int64)v73, 2);
        v136 = v69[6];
        if ( v136 )
          sub_161E7C0((__int64)(v69 + 6), v136);
        v137 = (unsigned __int8 *)v305[0];
        v69[6] = v305[0];
        if ( v137 )
          sub_1623210((__int64)v305, v137, (__int64)(v69 + 6));
        if ( v317 )
          sub_161E7C0((__int64)&v317, (__int64)v317);
        goto LABEL_131;
      }
      goto LABEL_131;
    }
  }
  v306 = 257;
  v308 = 257;
  v74 = sub_1648B60(64);
  v75 = v74;
  if ( v74 )
  {
    v279 = v74;
    sub_15F1EA0(v74, (__int64)v34, 53, 0, 0, 0);
    *(_DWORD *)(v75 + 56) = 0;
    sub_164B780(v75, v307);
    sub_1648880(v75, *(_DWORD *)(v75 + 56), 1);
  }
  else
  {
    v279 = 0;
  }
  if ( v318 )
  {
    v76 = v319;
    sub_157E9D0(v318 + 40, v75);
    v77 = *(_QWORD *)(v75 + 24);
    v78 = *v76;
    *(_QWORD *)(v75 + 32) = v76;
    v78 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v75 + 24) = v78 | v77 & 7;
    *(_QWORD *)(v78 + 8) = v75 + 24;
    *v76 = *v76 & 7 | (v75 + 24);
  }
  sub_164B780(v279, v305);
  v83 = (unsigned __int8 *)v317;
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v84 = *(_QWORD *)(v75 + 48);
    v81 = &v304;
    if ( v84 )
    {
      sub_161E7C0(v75 + 48, v84);
      v81 = &v304;
    }
    v83 = (unsigned __int8 *)v304;
    *(_QWORD *)(v75 + 48) = v304;
    if ( v83 )
      sub_1623210((__int64)&v304, v83, v75 + 48);
  }
  v85 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
  if ( v85 == *(_DWORD *)(v75 + 56) )
  {
    sub_15F55D0(v75, (__int64)v83, v79, v80, (__int64)v81, v82);
    v85 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
  }
  v86 = (v85 + 1) & 0xFFFFFFF;
  v87 = v86 | *(_DWORD *)(v75 + 20) & 0xF0000000;
  *(_DWORD *)(v75 + 20) = v87;
  if ( (v87 & 0x40000000) != 0 )
    v88 = *(_QWORD *)(v75 - 8);
  else
    v88 = v279 - 24 * v86;
  v89 = (__int64 *)(v88 + 24LL * (unsigned int)(v86 - 1));
  if ( *v89 )
  {
    v90 = v89[1];
    v91 = v89[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v91 = v90;
    if ( v90 )
      *(_QWORD *)(v90 + 16) = *(_QWORD *)(v90 + 16) & 3LL | v91;
  }
  *v89 = a6;
  v92 = *(_QWORD *)(a6 + 8);
  v89[1] = v92;
  if ( v92 )
    *(_QWORD *)(v92 + 16) = (unsigned __int64)(v89 + 1) | *(_QWORD *)(v92 + 16) & 3LL;
  v89[2] = (a6 + 8) | v89[2] & 3;
  *(_QWORD *)(a6 + 8) = v89;
  v93 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
    v94 = *(_QWORD *)(v75 - 8);
  else
    v94 = v279 - 24 * v93;
  *(_QWORD *)(v94 + 8LL * (unsigned int)(v93 - 1) + 24LL * *(unsigned int *)(v75 + 56) + 8) = v267;
  v306 = 257;
  v95 = sub_15A0680((__int64)v34, 1, 0);
  v96 = (unsigned __int8 *)v95;
  if ( *(_BYTE *)(v75 + 16) > 0x10u || *(_BYTE *)(v95 + 16) > 0x10u )
  {
    v308 = 257;
    v234 = sub_15FB440(13, (__int64 *)v75, v95, (__int64)v307, 0);
    v101 = v234;
    if ( v318 )
    {
      v271 = v319;
      sub_157E9D0(v318 + 40, v234);
      v235 = *v271;
      v236 = *(_QWORD *)(v101 + 24) & 7LL;
      *(_QWORD *)(v101 + 32) = v271;
      v235 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v101 + 24) = v235 | v236;
      *(_QWORD *)(v235 + 8) = v101 + 24;
      *v271 = *v271 & 7 | (v101 + 24);
    }
    sub_164B780(v101, v305);
    v96 = (unsigned __int8 *)v317;
    if ( v317 )
    {
      v304 = v317;
      sub_1623A60((__int64)&v304, (__int64)v317, 2);
      v237 = *(_QWORD *)(v101 + 48);
      v99 = &v304;
      v97 = v101 + 48;
      if ( v237 )
      {
        sub_161E7C0(v101 + 48, v237);
        v99 = &v304;
        v97 = v101 + 48;
      }
      v96 = (unsigned __int8 *)v304;
      *(_QWORD *)(v101 + 48) = v304;
      if ( v96 )
      {
        sub_1623210((__int64)&v304, v96, v97);
        v102 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        if ( v102 != *(_DWORD *)(v75 + 56) )
          goto LABEL_90;
        goto LABEL_257;
      }
    }
  }
  else
  {
    v101 = sub_15A2B60((__int64 *)v75, v95, 0, 0, *(double *)a7.m128_u64, a8, a9);
  }
  v102 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
  if ( v102 != *(_DWORD *)(v75 + 56) )
    goto LABEL_90;
LABEL_257:
  sub_15F55D0(v75, (__int64)v96, v97, v98, (__int64)v99, v100);
  v102 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
LABEL_90:
  v103 = (v102 + 1) & 0xFFFFFFF;
  v104 = v103 | *(_DWORD *)(v75 + 20) & 0xF0000000;
  *(_DWORD *)(v75 + 20) = v104;
  if ( (v104 & 0x40000000) != 0 )
    v105 = *(_QWORD *)(v75 - 8);
  else
    v105 = v279 - 24 * v103;
  v106 = (__int64 *)(v105 + 24LL * (unsigned int)(v103 - 1));
  if ( *v106 )
  {
    v107 = v106[1];
    v108 = v106[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v108 = v107;
    if ( v107 )
      *(_QWORD *)(v107 + 16) = *(_QWORD *)(v107 + 16) & 3LL | v108;
  }
  *v106 = v101;
  if ( v101 )
  {
    v109 = *(_QWORD *)(v101 + 8);
    v106[1] = v109;
    if ( v109 )
      *(_QWORD *)(v109 + 16) = (unsigned __int64)(v106 + 1) | *(_QWORD *)(v109 + 16) & 3LL;
    v106[2] = (v101 + 8) | v106[2] & 3;
    *(_QWORD *)(v101 + 8) = v106;
  }
  v110 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
    v111 = *(_QWORD *)(v75 - 8);
  else
    v111 = v279 - 24 * v110;
  *(_QWORD *)(v111 + 8LL * (unsigned int)(v110 - 1) + 24LL * *(unsigned int *)(v75 + 56) + 8) = v287;
  v308 = 257;
  v306 = 257;
  v280 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a2, v101, (__int64)v305);
  v112 = sub_1648A60(64, 1u);
  v113 = v112;
  if ( v112 )
    sub_15F9210((__int64)v112, *(_QWORD *)(*(_QWORD *)v280 + 24LL), v280, 0, a16, 0);
  if ( v318 )
  {
    v281 = (unsigned __int64 *)v319;
    sub_157E9D0(v318 + 40, (__int64)v113);
    v114 = *v281;
    v115 = v113[3] & 7LL;
    v113[4] = v281;
    v114 &= 0xFFFFFFFFFFFFFFF8LL;
    v113[3] = v114 | v115;
    *(_QWORD *)(v114 + 8) = v113 + 3;
    *v281 = *v281 & 7 | (unsigned __int64)(v113 + 3);
  }
  sub_164B780((__int64)v113, v307);
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v116 = v113[6];
    v117 = (__int64)(v113 + 6);
    if ( v116 )
    {
      sub_161E7C0((__int64)(v113 + 6), v116);
      v117 = (__int64)(v113 + 6);
    }
    v118 = (unsigned __int8 *)v304;
    v113[6] = v304;
    if ( v118 )
      sub_1623210((__int64)&v304, v118, v117);
  }
  sub_15F8F50((__int64)v113, a15);
  v306 = 257;
  v282 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a4, v101, (__int64)v305);
  v308 = 257;
  v119 = sub_1648A60(64, 2u);
  v120 = (__int64)v119;
  if ( v119 )
  {
    v121 = v282;
    v283 = v119;
    sub_15F9650((__int64)v119, (__int64)v113, v121, a17, 0);
    v120 = (__int64)v283;
  }
  if ( v318 )
  {
    v122 = v319;
    v284 = v120;
    sub_157E9D0(v318 + 40, v120);
    v120 = v284;
    v123 = *v122;
    v124 = *(_QWORD *)(v284 + 24);
    *(_QWORD *)(v284 + 32) = v122;
    v123 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v284 + 24) = v123 | v124 & 7;
    *(_QWORD *)(v123 + 8) = v284 + 24;
    *v122 = *v122 & 7 | (v284 + 24);
  }
  v285 = v120;
  sub_164B780(v120, v307);
  v125 = v285;
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v125 = v285;
    v126 = *(_QWORD *)(v285 + 48);
    v127 = v285 + 48;
    if ( v126 )
    {
      sub_161E7C0(v285 + 48, v126);
      v125 = v285;
    }
    v128 = (unsigned __int8 *)v304;
    *(_QWORD *)(v125 + 48) = v304;
    if ( v128 )
    {
      v286 = v125;
      sub_1623210((__int64)&v304, v128, v127);
      v125 = v286;
    }
  }
  sub_15F9450(v125, a15);
  v308 = 257;
  v129 = sub_15A0680((__int64)v34, 0, 0);
  v130 = sub_12AA0C0((__int64 *)&v317, 0x22u, (_BYTE *)v101, v129, (__int64)v307);
  v308 = 257;
  v131 = v130;
  v132 = sub_1648A60(56, 3u);
  v69 = v132;
  if ( v132 )
    sub_15F83E0((__int64)v132, v287, v277, v131, 0);
  if ( v318 )
  {
    v133 = (unsigned __int64 *)v319;
    sub_157E9D0(v318 + 40, (__int64)v69);
    v134 = v69[3];
    v135 = *v133;
    v69[4] = v133;
    v135 &= 0xFFFFFFFFFFFFFFF8LL;
    v69[3] = v135 | v134 & 7;
    *(_QWORD *)(v135 + 8) = v69 + 3;
    *v133 = *v133 & 7 | (unsigned __int64)(v69 + 3);
  }
  sub_164B780((__int64)v69, v307);
  v73 = (unsigned __int8 *)v317;
  if ( v317 )
  {
    v305[0] = (__int64)v317;
    goto LABEL_125;
  }
LABEL_131:
  v138 = sub_157E9C0(v272);
  v318 = v272;
  v320 = v138;
  v317 = 0;
  v48 = *(_BYTE *)(a6 + 16) == 13;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v319 = (__int64 *)(v272 + 40);
  if ( v48 )
  {
    v139 = *(_QWORD **)(a6 + 24);
    if ( *(_DWORD *)(a6 + 32) > 0x40u )
      v139 = (_QWORD *)*v139;
    v266 = v139;
    if ( (unsigned int)dword_4FBD560 >= (unsigned __int64)v139 )
    {
      if ( v139 )
      {
        v290 = (__int64)v34;
        v140 = 0;
        v291 = 0;
        do
        {
          v307[0] = (__int64)"src.memmove.gep.unroll";
          v308 = 259;
          v141 = sub_15A0680(v290, v140, 0);
          v273 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a2, v141, (__int64)v307);
          v308 = 257;
          v142 = sub_1648A60(64, 1u);
          v143 = v142;
          if ( v142 )
            sub_15F9210((__int64)v142, *(_QWORD *)(*(_QWORD *)v273 + 24LL), v273, 0, a16, 0);
          if ( v318 )
          {
            v274 = (unsigned __int64 *)v319;
            sub_157E9D0(v318 + 40, (__int64)v143);
            v144 = *v274;
            v145 = v143[3] & 7LL;
            v143[4] = v274;
            v144 &= 0xFFFFFFFFFFFFFFF8LL;
            v143[3] = v144 | v145;
            *(_QWORD *)(v144 + 8) = v143 + 3;
            *v274 = *v274 & 7 | (unsigned __int64)(v143 + 3);
          }
          sub_164B780((__int64)v143, v307);
          if ( v317 )
          {
            v305[0] = (__int64)v317;
            sub_1623A60((__int64)v305, (__int64)v317, 2);
            v146 = v143[6];
            v147 = (__int64)(v143 + 6);
            if ( v146 )
            {
              sub_161E7C0((__int64)(v143 + 6), v146);
              v147 = (__int64)(v143 + 6);
            }
            v148 = (unsigned __int8 *)v305[0];
            v143[6] = v305[0];
            if ( v148 )
              sub_1623210((__int64)v305, v148, v147);
          }
          sub_15F8F50((__int64)v143, a15);
          v307[0] = (__int64)"dst.memmove.gep,unroll";
          v308 = 259;
          v149 = sub_15A0680(v290, v140, 0);
          v275 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a4, v149, (__int64)v307);
          v308 = 257;
          v150 = sub_1648A60(64, 2u);
          v151 = v150;
          if ( v150 )
            sub_15F9650((__int64)v150, (__int64)v143, v275, a17, 0);
          if ( v318 )
          {
            v276 = (unsigned __int64 *)v319;
            sub_157E9D0(v318 + 40, (__int64)v151);
            v152 = *v276;
            v153 = v151[3] & 7LL;
            v151[4] = v276;
            v152 &= 0xFFFFFFFFFFFFFFF8LL;
            v151[3] = v152 | v153;
            *(_QWORD *)(v152 + 8) = v151 + 3;
            *v276 = *v276 & 7 | (unsigned __int64)(v151 + 3);
          }
          sub_164B780((__int64)v151, v307);
          if ( v317 )
          {
            v305[0] = (__int64)v317;
            sub_1623A60((__int64)v305, (__int64)v317, 2);
            v154 = v151[6];
            v155 = (__int64)(v151 + 6);
            if ( v154 )
            {
              sub_161E7C0((__int64)(v151 + 6), v154);
              v155 = (__int64)(v151 + 6);
            }
            v156 = (unsigned __int8 *)v305[0];
            v151[6] = v305[0];
            if ( v156 )
              sub_1623210((__int64)v305, v156, v155);
          }
          sub_15F9450((__int64)v151, a15);
          v140 = ++v291;
        }
        while ( v266 != (_QWORD *)v291 );
      }
      v308 = 257;
      v157 = sub_1648A60(56, 1u);
      v158 = v157;
      if ( v157 )
        sub_15F8320((__int64)v157, v277, 0);
      if ( v318 )
      {
        v159 = (unsigned __int64 *)v319;
        sub_157E9D0(v318 + 40, (__int64)v158);
        v160 = v158[3];
        v161 = *v159;
        v158[4] = v159;
        v161 &= 0xFFFFFFFFFFFFFFF8LL;
        v158[3] = v161 | v160 & 7;
        *(_QWORD *)(v161 + 8) = v158 + 3;
        *v159 = *v159 & 7 | (unsigned __int64)(v158 + 3);
      }
      sub_164B780((__int64)v158, v307);
      if ( v317 )
      {
        v305[0] = (__int64)v317;
        sub_1623A60((__int64)v305, (__int64)v317, 2);
        v162 = v158[6];
        if ( v162 )
          sub_161E7C0((__int64)(v158 + 6), v162);
        v163 = (unsigned __int8 *)v305[0];
        v158[6] = v305[0];
        if ( v163 )
          sub_1623210((__int64)v305, v163, (__int64)(v158 + 6));
LABEL_230:
        if ( v317 )
          sub_161E7C0((__int64)&v317, (__int64)v317);
        goto LABEL_232;
      }
      goto LABEL_232;
    }
  }
  v306 = 257;
  v308 = 257;
  v164 = sub_1648B60(64);
  v165 = v164;
  if ( v164 )
  {
    v288 = v164;
    sub_15F1EA0(v164, (__int64)v34, 53, 0, 0, 0);
    *(_DWORD *)(v165 + 56) = 0;
    sub_164B780(v165, v307);
    sub_1648880(v165, *(_DWORD *)(v165 + 56), 1);
  }
  else
  {
    v288 = 0;
  }
  if ( v318 )
  {
    v166 = v319;
    sub_157E9D0(v318 + 40, v165);
    v167 = *(_QWORD *)(v165 + 24);
    v168 = *v166;
    *(_QWORD *)(v165 + 32) = v166;
    v168 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v165 + 24) = v168 | v167 & 7;
    *(_QWORD *)(v168 + 8) = v165 + 24;
    *v166 = *v166 & 7 | (v165 + 24);
  }
  sub_164B780(v288, v305);
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v169 = *(_QWORD *)(v165 + 48);
    if ( v169 )
      sub_161E7C0(v165 + 48, v169);
    v170 = (unsigned __int8 *)v304;
    *(_QWORD *)(v165 + 48) = v304;
    if ( v170 )
      sub_1623210((__int64)&v304, v170, v165 + 48);
  }
  v175 = sub_15A0680((__int64)v34, 0, 0);
  v176 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
  if ( v176 == *(_DWORD *)(v165 + 56) )
  {
    sub_15F55D0(v165, 0, v171, v172, v173, v174);
    v176 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
  }
  v177 = (v176 + 1) & 0xFFFFFFF;
  v178 = v177 | *(_DWORD *)(v165 + 20) & 0xF0000000;
  *(_DWORD *)(v165 + 20) = v178;
  if ( (v178 & 0x40000000) != 0 )
    v179 = *(_QWORD *)(v165 - 8);
  else
    v179 = v288 - 24 * v177;
  v180 = (__int64 *)(v179 + 24LL * (unsigned int)(v177 - 1));
  if ( *v180 )
  {
    v181 = v180[1];
    v182 = v180[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v182 = v181;
    if ( v181 )
      *(_QWORD *)(v181 + 16) = *(_QWORD *)(v181 + 16) & 3LL | v182;
  }
  *v180 = v175;
  if ( v175 )
  {
    v183 = *(_QWORD *)(v175 + 8);
    v180[1] = v183;
    if ( v183 )
      *(_QWORD *)(v183 + 16) = (unsigned __int64)(v180 + 1) | *(_QWORD *)(v183 + 16) & 3LL;
    v180[2] = (v175 + 8) | v180[2] & 3;
    *(_QWORD *)(v175 + 8) = v180;
  }
  v184 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v165 + 23) & 0x40) != 0 )
    v185 = *(_QWORD *)(v165 - 8);
  else
    v185 = v288 - 24 * v184;
  *(_QWORD *)(v185 + 8LL * (unsigned int)(v184 - 1) + 24LL * *(unsigned int *)(v165 + 56) + 8) = v267;
  v308 = 257;
  v306 = 257;
  v297 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a2, v165, (__int64)v305);
  v186 = sub_1648A60(64, 1u);
  v187 = (__int64)v186;
  if ( v186 )
  {
    v188 = v297;
    v298 = v186;
    sub_15F9210((__int64)v186, *(_QWORD *)(*(_QWORD *)v188 + 24LL), v188, 0, a16, 0);
    v187 = (__int64)v298;
  }
  if ( v318 )
  {
    v189 = v319;
    v299 = v187;
    sub_157E9D0(v318 + 40, v187);
    v187 = v299;
    v190 = *v189;
    v191 = *(_QWORD *)(v299 + 24);
    *(_QWORD *)(v299 + 32) = v189;
    v190 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v299 + 24) = v190 | v191 & 7;
    *(_QWORD *)(v190 + 8) = v299 + 24;
    *v189 = *v189 & 7 | (v299 + 24);
  }
  v300 = v187;
  sub_164B780(v187, v307);
  v192 = v300;
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v192 = v300;
    v193 = *(_QWORD *)(v300 + 48);
    v194 = v300 + 48;
    if ( v193 )
    {
      sub_161E7C0(v300 + 48, v193);
      v192 = v300;
    }
    v195 = (unsigned __int8 *)v304;
    *(_QWORD *)(v192 + 48) = v304;
    if ( v195 )
    {
      v301 = v192;
      sub_1623210((__int64)&v304, v195, v194);
      v192 = v301;
    }
  }
  v270 = v192;
  sub_15F8F50(v192, a15);
  v306 = 257;
  v302 = sub_12815B0((__int64 *)&v317, 0, (_BYTE *)a4, v165, (__int64)v305);
  v308 = 257;
  v196 = sub_1648A60(64, 2u);
  v197 = v196;
  if ( v196 )
    sub_15F9650((__int64)v196, v270, v302, a17, 0);
  if ( v318 )
  {
    v303 = (unsigned __int64 *)v319;
    sub_157E9D0(v318 + 40, (__int64)v197);
    v198 = *v303;
    v199 = v197[3] & 7LL;
    v197[4] = v303;
    v198 &= 0xFFFFFFFFFFFFFFF8LL;
    v197[3] = v198 | v199;
    *(_QWORD *)(v198 + 8) = v197 + 3;
    *v303 = *v303 & 7 | (unsigned __int64)(v197 + 3);
  }
  sub_164B780((__int64)v197, v307);
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v200 = v197[6];
    v201 = (__int64)(v197 + 6);
    if ( v200 )
    {
      sub_161E7C0((__int64)(v197 + 6), v200);
      v201 = (__int64)(v197 + 6);
    }
    v202 = (unsigned __int8 *)v304;
    v197[6] = v304;
    if ( v202 )
      sub_1623210((__int64)&v304, v202, v201);
  }
  sub_15F9450((__int64)v197, a15);
  v306 = 257;
  v203 = sub_15A0680((__int64)v34, 1, 0);
  v204 = (unsigned __int8 *)v203;
  if ( *(_BYTE *)(v165 + 16) > 0x10u || *(_BYTE *)(v203 + 16) > 0x10u )
  {
    v308 = 257;
    v229 = sub_15FB440(11, (__int64 *)v165, v203, (__int64)v307, 0);
    v209 = v229;
    if ( v318 )
    {
      v230 = v319;
      sub_157E9D0(v318 + 40, v229);
      v231 = *(_QWORD *)(v209 + 24);
      v232 = *v230;
      *(_QWORD *)(v209 + 32) = v230;
      v232 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v209 + 24) = v232 | v231 & 7;
      *(_QWORD *)(v232 + 8) = v209 + 24;
      *v230 = *v230 & 7 | (v209 + 24);
    }
    sub_164B780(v209, v305);
    v204 = (unsigned __int8 *)v317;
    if ( v317 )
    {
      v304 = v317;
      sub_1623A60((__int64)&v304, (__int64)v317, 2);
      v233 = *(_QWORD *)(v209 + 48);
      v207 = &v304;
      if ( v233 )
      {
        sub_161E7C0(v209 + 48, v233);
        v207 = &v304;
      }
      v204 = (unsigned __int8 *)v304;
      *(_QWORD *)(v209 + 48) = v304;
      if ( v204 )
      {
        sub_1623210((__int64)&v304, v204, v209 + 48);
        v210 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
        if ( v210 != *(_DWORD *)(v165 + 56) )
          goto LABEL_210;
        goto LABEL_249;
      }
    }
  }
  else
  {
    v209 = sub_15A2B30((__int64 *)v165, v203, 0, 0, *(double *)a7.m128_u64, a8, a9);
  }
  v210 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
  if ( v210 != *(_DWORD *)(v165 + 56) )
    goto LABEL_210;
LABEL_249:
  sub_15F55D0(v165, (__int64)v204, v205, v206, (__int64)v207, v208);
  v210 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
LABEL_210:
  v211 = (v210 + 1) & 0xFFFFFFF;
  v212 = v211 | *(_DWORD *)(v165 + 20) & 0xF0000000;
  *(_DWORD *)(v165 + 20) = v212;
  if ( (v212 & 0x40000000) != 0 )
    v213 = *(_QWORD *)(v165 - 8);
  else
    v213 = v288 - 24 * v211;
  v214 = (__int64 *)(v213 + 24LL * (unsigned int)(v211 - 1));
  if ( *v214 )
  {
    v215 = v214[1];
    v216 = v214[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v216 = v215;
    if ( v215 )
      *(_QWORD *)(v215 + 16) = *(_QWORD *)(v215 + 16) & 3LL | v216;
  }
  *v214 = v209;
  if ( v209 )
  {
    v217 = *(_QWORD *)(v209 + 8);
    v214[1] = v217;
    if ( v217 )
      *(_QWORD *)(v217 + 16) = (unsigned __int64)(v214 + 1) | *(_QWORD *)(v217 + 16) & 3LL;
    v214[2] = (v209 + 8) | v214[2] & 3;
    *(_QWORD *)(v209 + 8) = v214;
  }
  v218 = *(_DWORD *)(v165 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v165 + 23) & 0x40) != 0 )
    v219 = *(_QWORD *)(v165 - 8);
  else
    v219 = v288 - 24 * v218;
  *(_QWORD *)(v219 + 8LL * (unsigned int)(v218 - 1) + 24LL * *(unsigned int *)(v165 + 56) + 8) = v272;
  v306 = 257;
  v220 = sub_12AA0C0((__int64 *)&v317, 0x24u, (_BYTE *)v209, a6, (__int64)v305);
  v308 = 257;
  v221 = v220;
  v222 = sub_1648A60(56, 3u);
  v223 = v222;
  if ( v222 )
    sub_15F83E0((__int64)v222, v272, v277, v221, 0);
  if ( v318 )
  {
    v224 = (unsigned __int64 *)v319;
    sub_157E9D0(v318 + 40, (__int64)v223);
    v225 = v223[3];
    v226 = *v224;
    v223[4] = v224;
    v226 &= 0xFFFFFFFFFFFFFFF8LL;
    v223[3] = v226 | v225 & 7;
    *(_QWORD *)(v226 + 8) = v223 + 3;
    *v224 = *v224 & 7 | (unsigned __int64)(v223 + 3);
  }
  sub_164B780((__int64)v223, v307);
  if ( v317 )
  {
    v304 = v317;
    sub_1623A60((__int64)&v304, (__int64)v317, 2);
    v227 = v223[6];
    if ( v227 )
      sub_161E7C0((__int64)(v223 + 6), v227);
    v228 = (unsigned __int8 *)v304;
    v223[6] = v304;
    if ( v228 )
      sub_1623210((__int64)&v304, v228, (__int64)(v223 + 6));
    goto LABEL_230;
  }
LABEL_232:
  if ( v309 )
    sub_161E7C0((__int64)&v309, (__int64)v309);
}
