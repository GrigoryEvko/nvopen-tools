// Function: sub_20CEB70
// Address: 0x20ceb70
//
__int64 __fastcall sub_20CEB70(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // r15
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  char v15; // r14
  __int64 v16; // rbx
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  unsigned __int8 *v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rdi
  _QWORD *(__fastcall *v28)(__int64, __int64 *, __int64, int); // rax
  _QWORD *v29; // rax
  _QWORD *v30; // r12
  unsigned __int64 *v31; // r14
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // r12
  unsigned __int64 *v38; // r14
  __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // r14
  _QWORD *v46; // rax
  _QWORD *v47; // r15
  unsigned __int64 *v48; // r14
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned __int8 *v53; // rsi
  __int64 v54; // rdi
  _QWORD *(__fastcall *v55)(__int64, __int64 *, __int64, int); // rax
  _QWORD *v56; // rax
  _QWORD *v57; // r15
  unsigned __int64 *v58; // r14
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rsi
  __int64 v62; // rdx
  unsigned __int8 *v63; // rsi
  _QWORD *v64; // rax
  _QWORD *v65; // r15
  unsigned __int64 *v66; // r14
  __int64 v67; // rax
  unsigned __int64 v68; // rcx
  __int64 v69; // rsi
  __int64 v70; // rdx
  unsigned __int8 *v71; // rsi
  __int64 v72; // rdx
  _BYTE *v73; // r14
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r14
  _QWORD *v78; // rax
  _QWORD *v79; // r15
  unsigned __int64 *v80; // r14
  __int64 v81; // rax
  unsigned __int64 v82; // rcx
  __int64 v83; // rsi
  __int64 v84; // rdx
  unsigned __int8 *v85; // rsi
  _BYTE *v86; // rax
  __int64 v87; // rcx
  __int64 v88; // rax
  __int64 v89; // r14
  _QWORD *v90; // rax
  _QWORD *v91; // r12
  unsigned __int64 *v92; // r14
  __int64 v93; // rax
  unsigned __int64 v94; // rcx
  __int64 v95; // rsi
  unsigned __int8 *v96; // rsi
  __int64 v97; // rdi
  _QWORD *(__fastcall *v98)(__int64, __int64 *, __int64, int); // rax
  _QWORD *v99; // rax
  _QWORD *v100; // r12
  unsigned __int64 *v101; // r14
  __int64 v102; // rax
  unsigned __int64 v103; // rcx
  __int64 v104; // rsi
  unsigned __int8 *v105; // rsi
  _QWORD *v106; // rax
  _QWORD *v107; // r12
  unsigned __int64 *v108; // r14
  __int64 v109; // rax
  unsigned __int64 v110; // rcx
  __int64 v111; // rsi
  unsigned __int8 *v112; // rsi
  __int64 v113; // rdi
  void (*v114)(); // rax
  _QWORD *v115; // rax
  _QWORD *v116; // r12
  unsigned __int64 *v117; // r14
  __int64 v118; // rax
  unsigned __int64 v119; // rcx
  __int64 v120; // rsi
  unsigned __int8 *v121; // rsi
  __int64 v122; // rdi
  _QWORD *(__fastcall *v123)(__int64, __int64 *, __int64, int); // rax
  _QWORD *v124; // rax
  _QWORD *v125; // r12
  unsigned __int64 *v126; // r14
  __int64 v127; // rax
  unsigned __int64 v128; // rcx
  __int64 v129; // rsi
  unsigned __int8 *v130; // rsi
  _QWORD *v131; // rax
  _QWORD *v132; // r12
  unsigned __int64 *v133; // r14
  __int64 v134; // rax
  unsigned __int64 v135; // rcx
  __int64 v136; // rsi
  unsigned __int8 *v137; // rsi
  __int64 v138; // rax
  unsigned __int8 *v139; // rsi
  char *v140; // rsi
  __int64 v141; // rax
  __int64 v142; // r14
  __int64 v143; // rax
  __int64 v144; // r12
  __int64 v145; // r15
  __int64 *v146; // r14
  __int64 v147; // rax
  __int64 v148; // rcx
  unsigned __int8 *v149; // rsi
  __int64 v150; // rsi
  __int64 v151; // rdx
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  int v156; // eax
  __int64 v157; // rax
  __int64 v158; // rsi
  int v159; // edx
  __int64 v160; // rdx
  __int64 *v161; // rax
  unsigned __int64 v162; // rdx
  __int64 v163; // rdx
  __int64 v164; // rdx
  __int64 v165; // rcx
  __int64 v166; // rdx
  __int64 v167; // rcx
  __int64 v168; // r8
  __int64 v169; // r9
  int v170; // eax
  __int64 v171; // rax
  int v172; // edx
  __int64 v173; // rdx
  __int64 *v174; // rax
  __int64 v175; // rsi
  unsigned __int64 v176; // rdx
  __int64 v177; // rdx
  __int64 v178; // rdx
  __int64 v179; // r15
  __int64 v180; // r13
  _QWORD *v181; // rax
  double v182; // xmm4_8
  double v183; // xmm5_8
  _QWORD *v184; // r14
  int v185; // r8d
  int v186; // r9d
  __int64 v187; // rdx
  unsigned __int8 *v188; // rbx
  unsigned __int8 *v189; // r15
  _QWORD *v190; // rdi
  __int64 v191; // rax
  __int64 v192; // rax
  __int64 v193; // rax
  double v194; // xmm4_8
  double v195; // xmm5_8
  __int64 v197; // rax
  unsigned __int8 *v198; // rsi
  char *v199; // rsi
  __int64 v200; // rsi
  __int64 v201; // rax
  __int64 v202; // r15
  __int64 v203; // rsi
  __int64 v204; // rax
  __int64 v205; // rdx
  __int64 v206; // rcx
  char **v207; // r8
  __int64 v208; // r9
  unsigned __int8 *v209; // rsi
  __int64 v210; // rsi
  int v211; // eax
  __int64 v212; // rax
  int v213; // edx
  __int64 v214; // rdx
  __int64 *v215; // rax
  __int64 v216; // rcx
  unsigned __int64 v217; // rdx
  __int64 v218; // rdx
  __int64 v219; // rdx
  __int64 v220; // rax
  __int64 v221; // rcx
  __int64 v222; // rdx
  int v223; // eax
  __int64 v224; // rax
  int v225; // edx
  __int64 v226; // rdx
  _QWORD *v227; // rax
  __int64 v228; // rcx
  unsigned __int64 v229; // rdx
  __int64 v230; // rdx
  __int64 v231; // rdx
  __int64 v232; // rcx
  __int64 v233; // rax
  unsigned __int8 *v234; // rsi
  char *v235; // rsi
  __int64 v236; // rsi
  __int64 v237; // rax
  __int64 v238; // r11
  __int64 v239; // rax
  __int64 v240; // rsi
  __int64 v241; // rdx
  __int64 v242; // rcx
  char **v243; // r8
  __int64 v244; // r9
  unsigned __int8 *v245; // rsi
  __int64 v246; // r11
  __int64 v247; // rsi
  int v248; // eax
  __int64 v249; // rax
  int v250; // edx
  __int64 v251; // rdx
  __int64 *v252; // rax
  __int64 v253; // rcx
  unsigned __int64 v254; // rdx
  __int64 v255; // rdx
  __int64 v256; // rdx
  __int64 v257; // rax
  __int64 v258; // rcx
  __int64 v259; // rdx
  int v260; // eax
  __int64 v261; // rax
  int v262; // edx
  __int64 v263; // rdx
  _QWORD *v264; // rax
  __int64 v265; // rcx
  unsigned __int64 v266; // rdx
  __int64 v267; // rdx
  __int64 v268; // rdx
  __int64 v269; // rcx
  __int64 v270; // rax
  unsigned __int8 *v271; // rsi
  char *v272; // rsi
  __int64 v273; // rsi
  __int64 v274; // rax
  __int64 v275; // r11
  __int64 v276; // rsi
  __int64 v277; // rax
  __int64 v278; // rdx
  char **v279; // r8
  __int64 v280; // r9
  unsigned __int8 *v281; // rsi
  __int64 v282; // r11
  __int64 v283; // rsi
  int v284; // eax
  __int64 v285; // rax
  __int64 v286; // rdx
  __int64 *v287; // rax
  __int64 v288; // rcx
  unsigned __int64 v289; // rdx
  __int64 v290; // rdx
  __int64 v291; // rdx
  __int64 v292; // rax
  __int64 v293; // rcx
  __int64 v294; // rdx
  int v295; // eax
  __int64 v296; // rax
  __int64 v297; // rdx
  _QWORD *v298; // rax
  __int64 v299; // rcx
  unsigned __int64 v300; // rdx
  __int64 v301; // rdx
  __int64 v302; // rdx
  __int64 v303; // r15
  _QWORD *v304; // rax
  _BYTE *v305; // [rsp+8h] [rbp-168h]
  char v306; // [rsp+10h] [rbp-160h]
  __int64 v307; // [rsp+10h] [rbp-160h]
  unsigned int v308; // [rsp+18h] [rbp-158h]
  unsigned int v309; // [rsp+1Ch] [rbp-154h]
  __int64 v310; // [rsp+20h] [rbp-150h]
  unsigned int v311; // [rsp+28h] [rbp-148h]
  char v312; // [rsp+30h] [rbp-140h]
  __int64 v313; // [rsp+30h] [rbp-140h]
  __int64 v314; // [rsp+38h] [rbp-138h]
  __int64 v315; // [rsp+40h] [rbp-130h]
  char v316; // [rsp+48h] [rbp-128h]
  __int64 *v317; // [rsp+48h] [rbp-128h]
  __int64 v318; // [rsp+50h] [rbp-120h]
  __int64 v319; // [rsp+58h] [rbp-118h]
  __int64 v320; // [rsp+58h] [rbp-118h]
  __int64 v321; // [rsp+58h] [rbp-118h]
  __int64 v322; // [rsp+58h] [rbp-118h]
  __int64 v323; // [rsp+58h] [rbp-118h]
  __int64 v324; // [rsp+58h] [rbp-118h]
  __int64 v325; // [rsp+60h] [rbp-110h]
  __int64 *v326; // [rsp+60h] [rbp-110h]
  __int64 v327; // [rsp+60h] [rbp-110h]
  __int64 v328; // [rsp+60h] [rbp-110h]
  __int64 v329; // [rsp+68h] [rbp-108h]
  __int64 v330; // [rsp+70h] [rbp-100h]
  __int64 v331; // [rsp+70h] [rbp-100h]
  __int64 v332; // [rsp+70h] [rbp-100h]
  __int64 v333; // [rsp+70h] [rbp-100h]
  _QWORD *v334; // [rsp+78h] [rbp-F8h]
  __int64 v336; // [rsp+80h] [rbp-F0h]
  __int64 *v337; // [rsp+80h] [rbp-F0h]
  __int64 v338; // [rsp+80h] [rbp-F0h]
  __int64 v339; // [rsp+80h] [rbp-F0h]
  int v340; // [rsp+80h] [rbp-F0h]
  int v341; // [rsp+80h] [rbp-F0h]
  __int16 v342; // [rsp+88h] [rbp-E8h]
  __int64 v343; // [rsp+88h] [rbp-E8h]
  __int64 *v345; // [rsp+98h] [rbp-D8h]
  __int64 v346; // [rsp+98h] [rbp-D8h]
  __int64 v347; // [rsp+98h] [rbp-D8h]
  __int64 v348; // [rsp+98h] [rbp-D8h]
  __int64 v349; // [rsp+98h] [rbp-D8h]
  __int64 v350; // [rsp+98h] [rbp-D8h]
  __int64 v351; // [rsp+98h] [rbp-D8h]
  __int64 v352; // [rsp+98h] [rbp-D8h]
  char *v353; // [rsp+A8h] [rbp-C8h] BYREF
  __int64 v354[2]; // [rsp+B0h] [rbp-C0h] BYREF
  __int16 v355; // [rsp+C0h] [rbp-B0h]
  char *v356; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 v357; // [rsp+D8h] [rbp-98h]
  _WORD v358[8]; // [rsp+E0h] [rbp-90h] BYREF
  char *v359; // [rsp+F0h] [rbp-80h] BYREF
  _QWORD *v360; // [rsp+F8h] [rbp-78h]
  unsigned __int64 *v361; // [rsp+100h] [rbp-70h]
  __int64 v362; // [rsp+108h] [rbp-68h]
  __int64 v363; // [rsp+110h] [rbp-60h]
  int v364; // [rsp+118h] [rbp-58h]
  __int64 v365; // [rsp+120h] [rbp-50h]
  __int64 v366; // [rsp+128h] [rbp-48h]

  v10 = *(_QWORD **)(a2 + 40);
  v308 = *(unsigned __int16 *)(a2 + 18);
  v11 = v10[7];
  v309 = (v308 >> 2) & 7;
  v310 = *(_QWORD *)(a2 - 72);
  v12 = sub_15E0530(v11);
  v13 = *(_QWORD *)(a1 + 160);
  v345 = (__int64 *)v12;
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 600LL);
  if ( v14 == sub_1F3CAD0 || (v312 = ((__int64 (__fastcall *)(__int64, __int64))v14)(v13, a2)) == 0 )
  {
    v312 = 0;
    v316 = 0;
    v311 = (v308 >> 2) & 7;
  }
  else if ( (*(_BYTE *)(a2 + 19) & 1) == 0 && ((v309 - 2) & 0xFFFFFFFD) != 0 )
  {
    v311 = 2;
    v316 = sub_1560180(v11 + 112, 17) ^ 1;
  }
  else
  {
    v311 = 2;
    v316 = 0;
  }
  v15 = sub_1560180(v11 + 112, 17);
  v306 = v312;
  if ( v15 )
  {
    v342 = *(_WORD *)(a2 + 18);
    v15 = v312 & ((v342 & 0x100) == 0);
    v306 = v312 & ((v342 & 0x100) != 0);
  }
  v359 = "cmpxchg.end";
  LOWORD(v361) = 259;
  v16 = a2 + 24;
  v330 = sub_157FBF0(v10, (__int64 *)(a2 + 24), (__int64)&v359);
  v359 = "cmpxchg.failure";
  LOWORD(v361) = 259;
  v334 = (_QWORD *)sub_22077B0(64);
  if ( v334 )
    sub_157FB60(v334, (__int64)v345, (__int64)&v359, v11, v330);
  v359 = "cmpxchg.nostore";
  LOWORD(v361) = 259;
  v17 = (_QWORD *)sub_22077B0(64);
  v319 = (__int64)v17;
  if ( v17 )
    sub_157FB60(v17, (__int64)v345, (__int64)&v359, v11, (__int64)v334);
  v359 = "cmpxchg.success";
  LOWORD(v361) = 259;
  v18 = (_QWORD *)sub_22077B0(64);
  v329 = (__int64)v18;
  if ( v18 )
    sub_157FB60(v18, (__int64)v345, (__int64)&v359, v11, v319);
  v359 = "cmpxchg.releasedload";
  LOWORD(v361) = 259;
  v19 = (_QWORD *)sub_22077B0(64);
  v318 = (__int64)v19;
  if ( v19 )
    sub_157FB60(v19, (__int64)v345, (__int64)&v359, v11, v329);
  v359 = "cmpxchg.trystore";
  LOWORD(v361) = 259;
  v20 = (_QWORD *)sub_22077B0(64);
  v325 = (__int64)v20;
  if ( v20 )
    sub_157FB60(v20, (__int64)v345, (__int64)&v359, v11, v318);
  v359 = "cmpxchg.fencedstore";
  LOWORD(v361) = 259;
  v21 = (_QWORD *)sub_22077B0(64);
  v314 = (__int64)v21;
  if ( v21 )
    sub_157FB60(v21, (__int64)v345, (__int64)&v359, v11, v325);
  v359 = "cmpxchg.start";
  LOWORD(v361) = 259;
  v22 = (_QWORD *)sub_22077B0(64);
  v315 = (__int64)v22;
  if ( v22 )
    sub_157FB60(v22, (__int64)v345, (__int64)&v359, v11, v314);
  v23 = sub_16498A0(a2);
  v24 = *(unsigned __int8 **)(a2 + 48);
  v361 = (unsigned __int64 *)v16;
  v362 = v23;
  v25 = *(_QWORD **)(a2 + 40);
  v359 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v366 = 0;
  v360 = v25;
  v356 = (char *)v24;
  if ( v24 )
  {
    sub_1623A60((__int64)&v356, (__int64)v24, 2);
    if ( v359 )
      sub_161E7C0((__int64)&v359, (__int64)v359);
    v359 = v356;
    if ( v356 )
      sub_1623210((__int64)&v356, (unsigned __int8 *)v356, (__int64)&v359);
  }
  v26 = (_QWORD *)((v10[5] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v10[5] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v26 = 0;
  sub_15F20C0(v26);
  v360 = v10;
  v361 = v10 + 5;
  if ( v15 )
  {
    v27 = *(_QWORD *)(a1 + 160);
    v28 = *(_QWORD *(__fastcall **)(__int64, __int64 *, __int64, int))(*(_QWORD *)v27 + 624LL);
    if ( v28 == sub_1F3D1E0 )
    {
      if ( byte_428C1E0[8 * v309 + 5] && (unsigned __int8)sub_15F3310(a2) )
      {
        v358[0] = 257;
        v29 = sub_1648A60(64, 0);
        v30 = v29;
        if ( v29 )
          sub_15F9C80((__int64)v29, v362, v309, 1, 0);
        if ( v360 )
        {
          v31 = v361;
          sub_157E9D0((__int64)(v360 + 5), (__int64)v30);
          v32 = v30[3];
          v33 = *v31;
          v30[4] = v31;
          v33 &= 0xFFFFFFFFFFFFFFF8LL;
          v30[3] = v33 | v32 & 7;
          *(_QWORD *)(v33 + 8) = v30 + 3;
          *v31 = *v31 & 7 | (unsigned __int64)(v30 + 3);
        }
        sub_164B780((__int64)v30, (__int64 *)&v356);
        if ( v359 )
        {
          v354[0] = (__int64)v359;
          sub_1623A60((__int64)v354, (__int64)v359, 2);
          v34 = v30[6];
          if ( v34 )
            sub_161E7C0((__int64)(v30 + 6), v34);
          v35 = (unsigned __int8 *)v354[0];
          v30[6] = v354[0];
          if ( v35 )
            sub_1623210((__int64)v354, v35, (__int64)(v30 + 6));
        }
      }
    }
    else
    {
      v28(v27, (__int64 *)&v359, a2, v309);
    }
  }
  v358[0] = 257;
  v36 = sub_1648A60(56, 1u);
  v37 = v36;
  if ( v36 )
    sub_15F8320((__int64)v36, v315, 0);
  if ( v360 )
  {
    v38 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v37);
    v39 = v37[3];
    v40 = *v38;
    v37[4] = v38;
    v40 &= 0xFFFFFFFFFFFFFFF8LL;
    v37[3] = v40 | v39 & 7;
    *(_QWORD *)(v40 + 8) = v37 + 3;
    *v38 = *v38 & 7 | (unsigned __int64)(v37 + 3);
  }
  sub_164B780((__int64)v37, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v41 = v37[6];
    if ( v41 )
      sub_161E7C0((__int64)(v37 + 6), v41);
    v42 = (unsigned __int8 *)v354[0];
    v37[6] = v354[0];
    if ( v42 )
      sub_1623210((__int64)v354, v42, (__int64)(v37 + 6));
  }
  v360 = (_QWORD *)v315;
  v361 = (unsigned __int64 *)(v315 + 40);
  v343 = (*(__int64 (__fastcall **)(_QWORD, char **, __int64, _QWORD))(**(_QWORD **)(a1 + 160) + 608LL))(
           *(_QWORD *)(a1 + 160),
           &v359,
           v310,
           v311);
  v43 = *(_QWORD *)(a2 - 48);
  v356 = "should_store";
  v358[0] = 259;
  v44 = sub_12AA0C0((__int64 *)&v359, 0x20u, (_BYTE *)v343, v43, (__int64)&v356);
  v358[0] = 257;
  v45 = v44;
  v46 = sub_1648A60(56, 3u);
  v47 = v46;
  if ( v46 )
    sub_15F83E0((__int64)v46, v314, v319, v45, 0);
  if ( v360 )
  {
    v48 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v47);
    v49 = v47[3];
    v50 = *v48;
    v47[4] = v48;
    v50 &= 0xFFFFFFFFFFFFFFF8LL;
    v47[3] = v50 | v49 & 7;
    *(_QWORD *)(v50 + 8) = v47 + 3;
    *v48 = *v48 & 7 | (unsigned __int64)(v47 + 3);
  }
  sub_164B780((__int64)v47, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v51 = v47[6];
    v52 = (__int64)(v47 + 6);
    if ( v51 )
    {
      sub_161E7C0((__int64)(v47 + 6), v51);
      v52 = (__int64)(v47 + 6);
    }
    v53 = (unsigned __int8 *)v354[0];
    v47[6] = v354[0];
    if ( v53 )
      sub_1623210((__int64)v354, v53, v52);
  }
  v360 = (_QWORD *)v314;
  v361 = (unsigned __int64 *)(v314 + 40);
  if ( v306 )
  {
    v54 = *(_QWORD *)(a1 + 160);
    v55 = *(_QWORD *(__fastcall **)(__int64, __int64 *, __int64, int))(*(_QWORD *)v54 + 624LL);
    if ( v55 == sub_1F3D1E0 )
    {
      if ( byte_428C1E0[8 * v309 + 5] && (unsigned __int8)sub_15F3310(a2) )
      {
        v358[0] = 257;
        v56 = sub_1648A60(64, 0);
        v57 = v56;
        if ( v56 )
          sub_15F9C80((__int64)v56, v362, v309, 1, 0);
        if ( v360 )
        {
          v58 = v361;
          sub_157E9D0((__int64)(v360 + 5), (__int64)v57);
          v59 = v57[3];
          v60 = *v58;
          v57[4] = v58;
          v60 &= 0xFFFFFFFFFFFFFFF8LL;
          v57[3] = v60 | v59 & 7;
          *(_QWORD *)(v60 + 8) = v57 + 3;
          *v58 = *v58 & 7 | (unsigned __int64)(v57 + 3);
        }
        sub_164B780((__int64)v57, (__int64 *)&v356);
        if ( v359 )
        {
          v354[0] = (__int64)v359;
          sub_1623A60((__int64)v354, (__int64)v359, 2);
          v61 = v57[6];
          v62 = (__int64)(v57 + 6);
          if ( v61 )
          {
            sub_161E7C0((__int64)(v57 + 6), v61);
            v62 = (__int64)(v57 + 6);
          }
          v63 = (unsigned __int8 *)v354[0];
          v57[6] = v354[0];
          if ( v63 )
            sub_1623210((__int64)v354, v63, v62);
        }
      }
    }
    else
    {
      v55(v54, (__int64 *)&v359, a2, v309);
    }
  }
  v358[0] = 257;
  v64 = sub_1648A60(56, 1u);
  v65 = v64;
  if ( v64 )
    sub_15F8320((__int64)v64, v325, 0);
  if ( v360 )
  {
    v66 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v65);
    v67 = v65[3];
    v68 = *v66;
    v65[4] = v66;
    v68 &= 0xFFFFFFFFFFFFFFF8LL;
    v65[3] = v68 | v67 & 7;
    *(_QWORD *)(v68 + 8) = v65 + 3;
    *v66 = *v66 & 7 | (unsigned __int64)(v65 + 3);
  }
  sub_164B780((__int64)v65, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v69 = v65[6];
    v70 = (__int64)(v65 + 6);
    if ( v69 )
    {
      sub_161E7C0((__int64)(v65 + 6), v69);
      v70 = (__int64)(v65 + 6);
    }
    v71 = (unsigned __int8 *)v354[0];
    v65[6] = v354[0];
    if ( v71 )
      sub_1623210((__int64)v354, v71, v70);
  }
  v360 = (_QWORD *)v325;
  v72 = *(_QWORD *)(a2 - 24);
  v361 = (unsigned __int64 *)(v325 + 40);
  v73 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, char **, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 160) + 616LL))(
                   *(_QWORD *)(a1 + 160),
                   &v359,
                   v72,
                   v310,
                   v311);
  v358[0] = 259;
  v356 = "success";
  v74 = sub_1643350(v345);
  v75 = sub_159C470(v74, 0, 0);
  v76 = sub_12AA0C0((__int64 *)&v359, 0x20u, v73, v75, (__int64)&v356);
  v358[0] = 257;
  v77 = v315;
  v307 = v76;
  if ( v316 )
    v77 = v318;
  if ( (*(_BYTE *)(a2 + 19) & 1) != 0 )
    v77 = (__int64)v334;
  v78 = sub_1648A60(56, 3u);
  v79 = v78;
  if ( v78 )
    sub_15F83E0((__int64)v78, v329, v77, v307, 0);
  if ( v360 )
  {
    v80 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v79);
    v81 = v79[3];
    v82 = *v80;
    v79[4] = v80;
    v82 &= 0xFFFFFFFFFFFFFFF8LL;
    v79[3] = v82 | v81 & 7;
    *(_QWORD *)(v82 + 8) = v79 + 3;
    *v80 = *v80 & 7 | (unsigned __int64)(v79 + 3);
  }
  sub_164B780((__int64)v79, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v83 = v79[6];
    v84 = (__int64)(v79 + 6);
    if ( v83 )
    {
      sub_161E7C0((__int64)(v79 + 6), v83);
      v84 = (__int64)(v79 + 6);
    }
    v85 = (unsigned __int8 *)v354[0];
    v79[6] = v354[0];
    if ( v85 )
      sub_1623210((__int64)v354, v85, v84);
  }
  v360 = (_QWORD *)v318;
  v361 = (unsigned __int64 *)(v318 + 40);
  if ( v316 )
  {
    v86 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, char **, __int64, _QWORD))(**(_QWORD **)(a1 + 160) + 608LL))(
                     *(_QWORD *)(a1 + 160),
                     &v359,
                     v310,
                     v311);
    v356 = "should_store";
    v87 = *(_QWORD *)(a2 - 48);
    v305 = v86;
    v358[0] = 259;
    v88 = sub_12AA0C0((__int64 *)&v359, 0x20u, v86, v87, (__int64)&v356);
    v358[0] = 257;
    v89 = v88;
    v90 = sub_1648A60(56, 3u);
    v91 = v90;
    if ( v90 )
      sub_15F83E0((__int64)v90, v325, v319, v89, 0);
  }
  else
  {
    v358[0] = 257;
    v304 = sub_1648A60(56, 0);
    v91 = v304;
    if ( v304 )
      sub_15F82A0((__int64)v304, v362, 0);
  }
  if ( v360 )
  {
    v92 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v91);
    v93 = v91[3];
    v94 = *v92;
    v91[4] = v92;
    v94 &= 0xFFFFFFFFFFFFFFF8LL;
    v91[3] = v94 | v93 & 7;
    *(_QWORD *)(v94 + 8) = v91 + 3;
    *v92 = *v92 & 7 | (unsigned __int64)(v91 + 3);
  }
  sub_164B780((__int64)v91, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v95 = v91[6];
    if ( v95 )
      sub_161E7C0((__int64)(v91 + 6), v95);
    v96 = (unsigned __int8 *)v354[0];
    v91[6] = v354[0];
    if ( v96 )
      sub_1623210((__int64)v354, v96, (__int64)(v91 + 6));
  }
  v360 = (_QWORD *)v329;
  v361 = (unsigned __int64 *)(v329 + 40);
  if ( v312 )
  {
    v97 = *(_QWORD *)(a1 + 160);
    v98 = *(_QWORD *(__fastcall **)(__int64, __int64 *, __int64, int))(*(_QWORD *)v97 + 632LL);
    if ( v98 == sub_1F3D380 )
    {
      if ( byte_428C1E0[8 * v309 + 4] )
      {
        v358[0] = 257;
        v99 = sub_1648A60(64, 0);
        v100 = v99;
        if ( v99 )
          sub_15F9C80((__int64)v99, v362, v309, 1, 0);
        if ( v360 )
        {
          v101 = v361;
          sub_157E9D0((__int64)(v360 + 5), (__int64)v100);
          v102 = v100[3];
          v103 = *v101;
          v100[4] = v101;
          v103 &= 0xFFFFFFFFFFFFFFF8LL;
          v100[3] = v103 | v102 & 7;
          *(_QWORD *)(v103 + 8) = v100 + 3;
          *v101 = *v101 & 7 | (unsigned __int64)(v100 + 3);
        }
        sub_164B780((__int64)v100, (__int64 *)&v356);
        if ( v359 )
        {
          v354[0] = (__int64)v359;
          sub_1623A60((__int64)v354, (__int64)v359, 2);
          v104 = v100[6];
          if ( v104 )
            sub_161E7C0((__int64)(v100 + 6), v104);
          v105 = (unsigned __int8 *)v354[0];
          v100[6] = v354[0];
          if ( v105 )
            sub_1623210((__int64)v354, v105, (__int64)(v100 + 6));
        }
      }
    }
    else
    {
      v98(v97, (__int64 *)&v359, a2, v309);
    }
  }
  v358[0] = 257;
  v106 = sub_1648A60(56, 1u);
  v107 = v106;
  if ( v106 )
    sub_15F8320((__int64)v106, v330, 0);
  if ( v360 )
  {
    v108 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v107);
    v109 = v107[3];
    v110 = *v108;
    v107[4] = v108;
    v110 &= 0xFFFFFFFFFFFFFFF8LL;
    v107[3] = v110 | v109 & 7;
    *(_QWORD *)(v110 + 8) = v107 + 3;
    *v108 = *v108 & 7 | (unsigned __int64)(v107 + 3);
  }
  sub_164B780((__int64)v107, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v111 = v107[6];
    if ( v111 )
      sub_161E7C0((__int64)(v107 + 6), v111);
    v112 = (unsigned __int8 *)v354[0];
    v107[6] = v354[0];
    if ( v112 )
      sub_1623210((__int64)v354, v112, (__int64)(v107 + 6));
  }
  v360 = (_QWORD *)v319;
  v361 = (unsigned __int64 *)(v319 + 40);
  v113 = *(_QWORD *)(a1 + 160);
  v114 = *(void (**)())(*(_QWORD *)v113 + 640LL);
  if ( v114 != nullsub_760 )
    ((void (__fastcall *)(__int64, char **))v114)(v113, &v359);
  v358[0] = 257;
  v115 = sub_1648A60(56, 1u);
  v116 = v115;
  if ( v115 )
    sub_15F8320((__int64)v115, (__int64)v334, 0);
  if ( v360 )
  {
    v117 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v116);
    v118 = v116[3];
    v119 = *v117;
    v116[4] = v117;
    v119 &= 0xFFFFFFFFFFFFFFF8LL;
    v116[3] = v119 | v118 & 7;
    *(_QWORD *)(v119 + 8) = v116 + 3;
    *v117 = *v117 & 7 | (unsigned __int64)(v116 + 3);
  }
  sub_164B780((__int64)v116, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v120 = v116[6];
    if ( v120 )
      sub_161E7C0((__int64)(v116 + 6), v120);
    v121 = (unsigned __int8 *)v354[0];
    v116[6] = v354[0];
    if ( v121 )
      sub_1623210((__int64)v354, v121, (__int64)(v116 + 6));
  }
  v360 = v334;
  v361 = v334 + 5;
  if ( v312 )
  {
    v122 = *(_QWORD *)(a1 + 160);
    v123 = *(_QWORD *(__fastcall **)(__int64, __int64 *, __int64, int))(*(_QWORD *)v122 + 632LL);
    if ( v123 == sub_1F3D380 )
    {
      if ( byte_428C1E0[8 * ((unsigned __int8)v308 >> 5) + 4] )
      {
        v358[0] = 257;
        v124 = sub_1648A60(64, 0);
        v125 = v124;
        if ( v124 )
          sub_15F9C80((__int64)v124, v362, (unsigned __int8)v308 >> 5, 1, 0);
        if ( v360 )
        {
          v126 = v361;
          sub_157E9D0((__int64)(v360 + 5), (__int64)v125);
          v127 = v125[3];
          v128 = *v126;
          v125[4] = v126;
          v128 &= 0xFFFFFFFFFFFFFFF8LL;
          v125[3] = v128 | v127 & 7;
          *(_QWORD *)(v128 + 8) = v125 + 3;
          *v126 = *v126 & 7 | (unsigned __int64)(v125 + 3);
        }
        sub_164B780((__int64)v125, (__int64 *)&v356);
        if ( v359 )
        {
          v354[0] = (__int64)v359;
          sub_1623A60((__int64)v354, (__int64)v359, 2);
          v129 = v125[6];
          if ( v129 )
            sub_161E7C0((__int64)(v125 + 6), v129);
          v130 = (unsigned __int8 *)v354[0];
          v125[6] = v354[0];
          if ( v130 )
            sub_1623210((__int64)v354, v130, (__int64)(v125 + 6));
        }
      }
    }
    else
    {
      v123(v122, (__int64 *)&v359, a2, (unsigned __int8)v308 >> 5);
    }
  }
  v358[0] = 257;
  v131 = sub_1648A60(56, 1u);
  v132 = v131;
  if ( v131 )
    sub_15F8320((__int64)v131, v330, 0);
  if ( v360 )
  {
    v133 = v361;
    sub_157E9D0((__int64)(v360 + 5), (__int64)v132);
    v134 = v132[3];
    v135 = *v133;
    v132[4] = v133;
    v135 &= 0xFFFFFFFFFFFFFFF8LL;
    v132[3] = v135 | v134 & 7;
    *(_QWORD *)(v135 + 8) = v132 + 3;
    *v133 = *v133 & 7 | (unsigned __int64)(v132 + 3);
  }
  sub_164B780((__int64)v132, (__int64 *)&v356);
  if ( v359 )
  {
    v354[0] = (__int64)v359;
    sub_1623A60((__int64)v354, (__int64)v359, 2);
    v136 = v132[6];
    if ( v136 )
      sub_161E7C0((__int64)(v132 + 6), v136);
    v137 = (unsigned __int8 *)v354[0];
    v132[6] = v354[0];
    if ( v137 )
      sub_1623210((__int64)v354, v137, (__int64)(v132 + 6));
  }
  v138 = *(_QWORD *)(v330 + 48);
  v360 = (_QWORD *)v330;
  v361 = (unsigned __int64 *)v138;
  if ( v138 != v330 + 40 )
  {
    if ( !v138 )
      BUG();
    v139 = *(unsigned __int8 **)(v138 + 24);
    v356 = (char *)v139;
    if ( v139 )
    {
      sub_1623A60((__int64)&v356, (__int64)v139, 2);
      v140 = v359;
      if ( !v359 )
        goto LABEL_160;
    }
    else
    {
      v140 = v359;
      if ( !v359 )
        goto LABEL_162;
    }
    sub_161E7C0((__int64)&v359, (__int64)v140);
LABEL_160:
    v359 = v356;
    if ( v356 )
      sub_1623210((__int64)&v356, (unsigned __int8 *)v356, (__int64)&v359);
  }
LABEL_162:
  v355 = 257;
  v141 = sub_1643320(v345);
  v358[0] = 257;
  v142 = v141;
  v143 = sub_1648B60(64);
  v144 = v143;
  if ( v143 )
  {
    v145 = v143;
    sub_15F1EA0(v143, v142, 53, 0, 0, 0);
    *(_DWORD *)(v144 + 56) = 2;
    sub_164B780(v144, (__int64 *)&v356);
    sub_1648880(v144, *(_DWORD *)(v144 + 56), 1);
  }
  else
  {
    v145 = 0;
  }
  if ( v360 )
  {
    v146 = (__int64 *)v361;
    sub_157E9D0((__int64)(v360 + 5), v144);
    v147 = *(_QWORD *)(v144 + 24);
    v148 = *v146;
    *(_QWORD *)(v144 + 32) = v146;
    v148 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v144 + 24) = v148 | v147 & 7;
    *(_QWORD *)(v148 + 8) = v144 + 24;
    *v146 = *v146 & 7 | (v144 + 24);
  }
  sub_164B780(v145, v354);
  v149 = (unsigned __int8 *)v359;
  if ( v359 )
  {
    v353 = v359;
    sub_1623A60((__int64)&v353, (__int64)v359, 2);
    v150 = *(_QWORD *)(v144 + 48);
    v151 = v144 + 48;
    if ( v150 )
    {
      sub_161E7C0(v144 + 48, v150);
      v151 = v144 + 48;
    }
    v149 = (unsigned __int8 *)v353;
    *(_QWORD *)(v144 + 48) = v353;
    if ( v149 )
      sub_1623210((__int64)&v353, v149, v151);
  }
  v153 = sub_159C4F0(v345);
  v156 = *(_DWORD *)(v144 + 20) & 0xFFFFFFF;
  if ( v156 == *(_DWORD *)(v144 + 56) )
  {
    v313 = v153;
    sub_15F55D0(v144, (__int64)v149, v152, v153, v154, v155);
    v153 = v313;
    v156 = *(_DWORD *)(v144 + 20) & 0xFFFFFFF;
  }
  v157 = (v156 + 1) & 0xFFFFFFF;
  v158 = (unsigned int)(v157 - 1);
  v159 = v157 | *(_DWORD *)(v144 + 20) & 0xF0000000;
  *(_DWORD *)(v144 + 20) = v159;
  if ( (v159 & 0x40000000) != 0 )
    v160 = *(_QWORD *)(v144 - 8);
  else
    v160 = v145 - 24 * v157;
  v161 = (__int64 *)(v160 + 24LL * (unsigned int)v158);
  if ( *v161 )
  {
    v158 = v161[1];
    v162 = v161[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v162 = v158;
    if ( v158 )
      *(_QWORD *)(v158 + 16) = *(_QWORD *)(v158 + 16) & 3LL | v162;
  }
  *v161 = v153;
  if ( v153 )
  {
    v163 = *(_QWORD *)(v153 + 8);
    v161[1] = v163;
    if ( v163 )
    {
      v158 = (unsigned __int64)(v161 + 1) | *(_QWORD *)(v163 + 16) & 3LL;
      *(_QWORD *)(v163 + 16) = v158;
    }
    v161[2] = (v153 + 8) | v161[2] & 3;
    *(_QWORD *)(v153 + 8) = v161;
  }
  v164 = *(_DWORD *)(v144 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v144 + 23) & 0x40) != 0 )
    v165 = *(_QWORD *)(v144 - 8);
  else
    v165 = v145 - 24 * v164;
  *(_QWORD *)(v165 + 8LL * (unsigned int)(v164 - 1) + 24LL * *(unsigned int *)(v144 + 56) + 8) = v329;
  v167 = sub_159C540(v345);
  v170 = *(_DWORD *)(v144 + 20) & 0xFFFFFFF;
  if ( v170 == *(_DWORD *)(v144 + 56) )
  {
    v352 = v167;
    sub_15F55D0(v144, v158, v166, v167, v168, v169);
    v167 = v352;
    v170 = *(_DWORD *)(v144 + 20) & 0xFFFFFFF;
  }
  v171 = (v170 + 1) & 0xFFFFFFF;
  v172 = v171 | *(_DWORD *)(v144 + 20) & 0xF0000000;
  *(_DWORD *)(v144 + 20) = v172;
  if ( (v172 & 0x40000000) != 0 )
    v173 = *(_QWORD *)(v144 - 8);
  else
    v173 = v145 - 24 * v171;
  v174 = (__int64 *)(v173 + 24LL * (unsigned int)(v171 - 1));
  if ( *v174 )
  {
    v175 = v174[1];
    v176 = v174[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v176 = v175;
    if ( v175 )
      *(_QWORD *)(v175 + 16) = *(_QWORD *)(v175 + 16) & 3LL | v176;
  }
  *v174 = v167;
  if ( v167 )
  {
    v177 = *(_QWORD *)(v167 + 8);
    v174[1] = v177;
    if ( v177 )
      *(_QWORD *)(v177 + 16) = (unsigned __int64)(v174 + 1) | *(_QWORD *)(v177 + 16) & 3LL;
    v174[2] = (v167 + 8) | v174[2] & 3;
    *(_QWORD *)(v167 + 8) = v174;
  }
  v178 = *(_DWORD *)(v144 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v144 + 23) & 0x40) != 0 )
    v179 = *(_QWORD *)(v144 - 8);
  else
    v179 = v145 - 24 * v178;
  *(_QWORD *)(v179 + 8LL * (unsigned int)(v178 - 1) + 24LL * *(unsigned int *)(v144 + 56) + 8) = v334;
  if ( !v316 )
    goto LABEL_198;
  v197 = *(_QWORD *)(v325 + 48);
  v360 = (_QWORD *)v325;
  v361 = (unsigned __int64 *)v197;
  if ( v325 + 40 != v197 )
  {
    if ( !v197 )
      BUG();
    v198 = *(unsigned __int8 **)(v197 + 24);
    v356 = (char *)v198;
    if ( v198 )
    {
      sub_1623A60((__int64)&v356, (__int64)v198, 2);
      v199 = v359;
      if ( !v359 )
        goto LABEL_228;
    }
    else
    {
      v199 = v359;
      if ( !v359 )
        goto LABEL_230;
    }
    sub_161E7C0((__int64)&v359, (__int64)v199);
LABEL_228:
    v359 = v356;
    if ( v356 )
      sub_1623210((__int64)&v356, (unsigned __int8 *)v356, (__int64)&v359);
  }
LABEL_230:
  v355 = 257;
  v200 = *(_QWORD *)v343;
  v358[0] = 257;
  v201 = sub_1648B60(64);
  v202 = v201;
  if ( v201 )
  {
    v346 = v201;
    sub_15F1EA0(v201, v200, 53, 0, 0, 0);
    *(_DWORD *)(v202 + 56) = 2;
    sub_164B780(v202, (__int64 *)&v356);
    sub_1648880(v202, *(_DWORD *)(v202 + 56), 1);
  }
  else
  {
    v346 = 0;
  }
  if ( v360 )
  {
    v326 = (__int64 *)v361;
    sub_157E9D0((__int64)(v360 + 5), v202);
    v203 = *v326;
    v204 = *(_QWORD *)(v202 + 24) & 7LL;
    *(_QWORD *)(v202 + 32) = v326;
    v203 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v202 + 24) = v203 | v204;
    *(_QWORD *)(v203 + 8) = v202 + 24;
    *v326 = *v326 & 7 | (v202 + 24);
  }
  sub_164B780(v346, v354);
  v209 = (unsigned __int8 *)v359;
  if ( v359 )
  {
    v353 = v359;
    sub_1623A60((__int64)&v353, (__int64)v359, 2);
    v210 = *(_QWORD *)(v202 + 48);
    v207 = &v353;
    v205 = v202 + 48;
    if ( v210 )
    {
      sub_161E7C0(v202 + 48, v210);
      v207 = &v353;
      v205 = v202 + 48;
    }
    v209 = (unsigned __int8 *)v353;
    *(_QWORD *)(v202 + 48) = v353;
    if ( v209 )
      sub_1623210((__int64)&v353, v209, v205);
  }
  v211 = *(_DWORD *)(v202 + 20) & 0xFFFFFFF;
  if ( v211 == *(_DWORD *)(v202 + 56) )
  {
    sub_15F55D0(v202, (__int64)v209, v205, v206, (__int64)v207, v208);
    v211 = *(_DWORD *)(v202 + 20) & 0xFFFFFFF;
  }
  v212 = (v211 + 1) & 0xFFFFFFF;
  v213 = v212 | *(_DWORD *)(v202 + 20) & 0xF0000000;
  *(_DWORD *)(v202 + 20) = v213;
  if ( (v213 & 0x40000000) != 0 )
    v214 = *(_QWORD *)(v202 - 8);
  else
    v214 = v346 - 24 * v212;
  v215 = (__int64 *)(v214 + 24LL * (unsigned int)(v212 - 1));
  if ( *v215 )
  {
    v216 = v215[1];
    v217 = v215[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v217 = v216;
    if ( v216 )
      *(_QWORD *)(v216 + 16) = *(_QWORD *)(v216 + 16) & 3LL | v217;
  }
  *v215 = v343;
  v218 = *(_QWORD *)(v343 + 8);
  v327 = v343 + 8;
  v215[1] = v218;
  if ( v218 )
    *(_QWORD *)(v218 + 16) = (unsigned __int64)(v215 + 1) | *(_QWORD *)(v218 + 16) & 3LL;
  v215[2] = v327 | v215[2] & 3;
  *(_QWORD *)(v343 + 8) = v215;
  v219 = *(_DWORD *)(v202 + 20) & 0xFFFFFFF;
  v220 = (unsigned int)(v219 - 1);
  if ( (*(_BYTE *)(v202 + 23) & 0x40) != 0 )
    v221 = *(_QWORD *)(v202 - 8);
  else
    v221 = v346 - 24 * v219;
  v222 = 3LL * *(unsigned int *)(v202 + 56);
  *(_QWORD *)(v221 + 8 * v220 + 24LL * *(unsigned int *)(v202 + 56) + 8) = v314;
  v223 = *(_DWORD *)(v202 + 20) & 0xFFFFFFF;
  if ( v223 == *(_DWORD *)(v202 + 56) )
  {
    sub_15F55D0(v202, v314, v222, v221, (__int64)v207, v208);
    v223 = *(_DWORD *)(v202 + 20) & 0xFFFFFFF;
  }
  v224 = (v223 + 1) & 0xFFFFFFF;
  v225 = v224 | *(_DWORD *)(v202 + 20) & 0xF0000000;
  *(_DWORD *)(v202 + 20) = v225;
  if ( (v225 & 0x40000000) != 0 )
    v226 = *(_QWORD *)(v202 - 8);
  else
    v226 = v346 - 24 * v224;
  v227 = (_QWORD *)(v226 + 24LL * (unsigned int)(v224 - 1));
  if ( *v227 )
  {
    v228 = v227[1];
    v229 = v227[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v229 = v228;
    if ( v228 )
      *(_QWORD *)(v228 + 16) = *(_QWORD *)(v228 + 16) & 3LL | v229;
  }
  *v227 = v305;
  if ( v305 )
  {
    v230 = *((_QWORD *)v305 + 1);
    v227[1] = v230;
    if ( v230 )
      *(_QWORD *)(v230 + 16) = (unsigned __int64)(v227 + 1) | *(_QWORD *)(v230 + 16) & 3LL;
    v227[2] = (unsigned __int64)(v305 + 8) | v227[2] & 3LL;
    *((_QWORD *)v305 + 1) = v227;
  }
  v231 = *(_DWORD *)(v202 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v202 + 23) & 0x40) != 0 )
    v232 = *(_QWORD *)(v202 - 8);
  else
    v232 = v346 - 24 * v231;
  *(_QWORD *)(v232 + 8LL * (unsigned int)(v231 - 1) + 24LL * *(unsigned int *)(v202 + 56) + 8) = v318;
  v233 = *(_QWORD *)(v319 + 48);
  v360 = (_QWORD *)v319;
  v361 = (unsigned __int64 *)v233;
  if ( v319 + 40 != v233 )
  {
    if ( !v233 )
      BUG();
    v234 = *(unsigned __int8 **)(v233 + 24);
    v356 = (char *)v234;
    if ( v234 )
    {
      sub_1623A60((__int64)&v356, (__int64)v234, 2);
      v235 = v359;
      if ( !v359 )
        goto LABEL_268;
    }
    else
    {
      v235 = v359;
      if ( !v359 )
        goto LABEL_270;
    }
    sub_161E7C0((__int64)&v359, (__int64)v235);
LABEL_268:
    v359 = v356;
    if ( v356 )
      sub_1623210((__int64)&v356, (unsigned __int8 *)v356, (__int64)&v359);
  }
LABEL_270:
  v355 = 257;
  v236 = *(_QWORD *)v343;
  v358[0] = 257;
  v237 = sub_1648B60(64);
  v238 = v237;
  if ( v237 )
  {
    v347 = v237;
    v320 = v237;
    sub_15F1EA0(v237, v236, 53, 0, 0, 0);
    *(_DWORD *)(v320 + 56) = 2;
    sub_164B780(v320, (__int64 *)&v356);
    sub_1648880(v320, *(_DWORD *)(v320 + 56), 1);
    v238 = v320;
  }
  else
  {
    v347 = 0;
  }
  if ( v360 )
  {
    v321 = v238;
    v317 = (__int64 *)v361;
    sub_157E9D0((__int64)(v360 + 5), v238);
    v238 = v321;
    v239 = *(_QWORD *)(v321 + 24);
    v240 = *v317;
    *(_QWORD *)(v321 + 32) = v317;
    v240 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v321 + 24) = v240 | v239 & 7;
    *(_QWORD *)(v240 + 8) = v321 + 24;
    *v317 = *v317 & 7 | (v321 + 24);
  }
  v322 = v238;
  sub_164B780(v347, v354);
  v245 = (unsigned __int8 *)v359;
  v246 = v322;
  if ( v359 )
  {
    v353 = v359;
    sub_1623A60((__int64)&v353, (__int64)v359, 2);
    v246 = v322;
    v243 = &v353;
    v247 = *(_QWORD *)(v322 + 48);
    v241 = v322 + 48;
    if ( v247 )
    {
      sub_161E7C0(v322 + 48, v247);
      v243 = &v353;
      v246 = v322;
      v241 = v322 + 48;
    }
    v245 = (unsigned __int8 *)v353;
    *(_QWORD *)(v246 + 48) = v353;
    if ( v245 )
    {
      v323 = v246;
      sub_1623210((__int64)&v353, v245, v241);
      v246 = v323;
    }
  }
  v248 = *(_DWORD *)(v246 + 20) & 0xFFFFFFF;
  if ( v248 == *(_DWORD *)(v246 + 56) )
  {
    v324 = v246;
    sub_15F55D0(v246, (__int64)v245, v241, v242, (__int64)v243, v244);
    v246 = v324;
    v248 = *(_DWORD *)(v324 + 20) & 0xFFFFFFF;
  }
  v249 = (v248 + 1) & 0xFFFFFFF;
  v250 = v249 | *(_DWORD *)(v246 + 20) & 0xF0000000;
  *(_DWORD *)(v246 + 20) = v250;
  if ( (v250 & 0x40000000) != 0 )
    v251 = *(_QWORD *)(v246 - 8);
  else
    v251 = v347 - 24 * v249;
  v252 = (__int64 *)(v251 + 24LL * (unsigned int)(v249 - 1));
  if ( *v252 )
  {
    v253 = v252[1];
    v254 = v252[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v254 = v253;
    if ( v253 )
      *(_QWORD *)(v253 + 16) = *(_QWORD *)(v253 + 16) & 3LL | v254;
  }
  *v252 = v343;
  v255 = *(_QWORD *)(v343 + 8);
  v252[1] = v255;
  if ( v255 )
    *(_QWORD *)(v255 + 16) = (unsigned __int64)(v252 + 1) | *(_QWORD *)(v255 + 16) & 3LL;
  v252[2] = v327 | v252[2] & 3;
  *(_QWORD *)(v343 + 8) = v252;
  v256 = *(_DWORD *)(v246 + 20) & 0xFFFFFFF;
  v257 = (unsigned int)(v256 - 1);
  if ( (*(_BYTE *)(v246 + 23) & 0x40) != 0 )
    v258 = *(_QWORD *)(v246 - 8);
  else
    v258 = v347 - 24 * v256;
  v259 = 3LL * *(unsigned int *)(v246 + 56);
  *(_QWORD *)(v258 + 8 * v257 + 24LL * *(unsigned int *)(v246 + 56) + 8) = v315;
  v260 = *(_DWORD *)(v246 + 20) & 0xFFFFFFF;
  if ( v260 == *(_DWORD *)(v246 + 56) )
  {
    v328 = v246;
    sub_15F55D0(v246, v315, v259, v258, (__int64)v243, v244);
    v246 = v328;
    v260 = *(_DWORD *)(v328 + 20) & 0xFFFFFFF;
  }
  v261 = (v260 + 1) & 0xFFFFFFF;
  v262 = v261 | *(_DWORD *)(v246 + 20) & 0xF0000000;
  *(_DWORD *)(v246 + 20) = v262;
  if ( (v262 & 0x40000000) != 0 )
    v263 = *(_QWORD *)(v246 - 8);
  else
    v263 = v347 - 24 * v261;
  v264 = (_QWORD *)(v263 + 24LL * (unsigned int)(v261 - 1));
  if ( *v264 )
  {
    v265 = v264[1];
    v266 = v264[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v266 = v265;
    if ( v265 )
      *(_QWORD *)(v265 + 16) = *(_QWORD *)(v265 + 16) & 3LL | v266;
  }
  *v264 = v305;
  if ( v305 )
  {
    v267 = *((_QWORD *)v305 + 1);
    v264[1] = v267;
    if ( v267 )
      *(_QWORD *)(v267 + 16) = (unsigned __int64)(v264 + 1) | *(_QWORD *)(v267 + 16) & 3LL;
    v264[2] = (unsigned __int64)(v305 + 8) | v264[2] & 3LL;
    *((_QWORD *)v305 + 1) = v264;
  }
  v268 = *(_DWORD *)(v246 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v246 + 23) & 0x40) != 0 )
    v269 = *(_QWORD *)(v246 - 8);
  else
    v269 = v347 - 24 * v268;
  *(_QWORD *)(v269 + 8LL * (unsigned int)(v268 - 1) + 24LL * *(unsigned int *)(v246 + 56) + 8) = v318;
  v270 = *(_QWORD *)(*(_QWORD *)(v330 + 48) + 8LL);
  v360 = (_QWORD *)v330;
  v361 = (unsigned __int64 *)v270;
  if ( v270 == v330 + 40 )
    goto LABEL_310;
  if ( !v270 )
    BUG();
  v271 = *(unsigned __int8 **)(v270 + 24);
  v356 = (char *)v271;
  if ( v271 )
  {
    v348 = v246;
    sub_1623A60((__int64)&v356, (__int64)v271, 2);
    v272 = v359;
    v246 = v348;
    if ( !v359 )
      goto LABEL_308;
    goto LABEL_307;
  }
  v272 = v359;
  if ( v359 )
  {
LABEL_307:
    v349 = v246;
    sub_161E7C0((__int64)&v359, (__int64)v272);
    v246 = v349;
LABEL_308:
    v359 = v356;
    if ( v356 )
    {
      v350 = v246;
      sub_1623210((__int64)&v356, (unsigned __int8 *)v356, (__int64)&v359);
      v246 = v350;
    }
  }
LABEL_310:
  v336 = v246;
  v355 = 257;
  v273 = *(_QWORD *)v343;
  v358[0] = 257;
  v274 = sub_1648B60(64);
  v275 = v336;
  v343 = v274;
  if ( v274 )
  {
    v351 = v274;
    sub_15F1EA0(v274, v273, 53, 0, 0, 0);
    *(_DWORD *)(v343 + 56) = 2;
    sub_164B780(v343, (__int64 *)&v356);
    sub_1648880(v343, *(_DWORD *)(v343 + 56), 1);
    v275 = v336;
  }
  else
  {
    v351 = 0;
  }
  if ( v360 )
  {
    v331 = v275;
    v337 = (__int64 *)v361;
    sub_157E9D0((__int64)(v360 + 5), v343);
    v275 = v331;
    v276 = *v337;
    v277 = *(_QWORD *)(v343 + 24);
    *(_QWORD *)(v343 + 32) = v337;
    v276 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v343 + 24) = v276 | v277 & 7;
    *(_QWORD *)(v276 + 8) = v343 + 24;
    *v337 = *v337 & 7 | (v343 + 24);
  }
  v338 = v275;
  sub_164B780(v351, v354);
  v281 = (unsigned __int8 *)v359;
  v282 = v338;
  if ( v359 )
  {
    v353 = v359;
    sub_1623A60((__int64)&v353, (__int64)v359, 2);
    v279 = &v353;
    v282 = v338;
    v283 = *(_QWORD *)(v343 + 48);
    v278 = v343 + 48;
    if ( v283 )
    {
      sub_161E7C0(v343 + 48, v283);
      v279 = &v353;
      v282 = v338;
      v278 = v343 + 48;
    }
    v281 = (unsigned __int8 *)v353;
    *(_QWORD *)(v343 + 48) = v353;
    if ( v281 )
    {
      v339 = v282;
      sub_1623210((__int64)&v353, v281, v278);
      v282 = v339;
    }
  }
  v284 = *(_DWORD *)(v343 + 20) & 0xFFFFFFF;
  if ( v284 == *(_DWORD *)(v343 + 56) )
  {
    v332 = v282;
    sub_15F55D0(v343, (__int64)v281, v278, v343, (__int64)v279, v280);
    v282 = v332;
    v284 = *(_DWORD *)(v343 + 20) & 0xFFFFFFF;
  }
  v285 = (v284 + 1) & 0xFFFFFFF;
  v340 = *(_DWORD *)(v343 + 20);
  *(_DWORD *)(v343 + 20) = v285 | v340 & 0xF0000000;
  if ( v285 & 0x40000000 | v340 & 0x40000000 )
    v286 = *(_QWORD *)(v343 - 8);
  else
    v286 = v351 - 24 * v285;
  v287 = (__int64 *)(v286 + 24LL * (unsigned int)(v285 - 1));
  if ( *v287 )
  {
    v288 = v287[1];
    v289 = v287[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v289 = v288;
    if ( v288 )
      *(_QWORD *)(v288 + 16) = *(_QWORD *)(v288 + 16) & 3LL | v289;
  }
  *v287 = v202;
  v290 = *(_QWORD *)(v202 + 8);
  v287[1] = v290;
  if ( v290 )
    *(_QWORD *)(v290 + 16) = (unsigned __int64)(v287 + 1) | *(_QWORD *)(v290 + 16) & 3LL;
  v287[2] = (v202 + 8) | v287[2] & 3;
  *(_QWORD *)(v202 + 8) = v287;
  v291 = *(_DWORD *)(v343 + 20) & 0xFFFFFFF;
  v292 = (unsigned int)(v291 - 1);
  if ( (*(_BYTE *)(v343 + 23) & 0x40) != 0 )
    v293 = *(_QWORD *)(v343 - 8);
  else
    v293 = v351 - 24 * v291;
  v294 = 3LL * *(unsigned int *)(v343 + 56);
  *(_QWORD *)(v293 + 8 * v292 + 24LL * *(unsigned int *)(v343 + 56) + 8) = v329;
  v295 = *(_DWORD *)(v343 + 20) & 0xFFFFFFF;
  if ( v295 == *(_DWORD *)(v343 + 56) )
  {
    v333 = v282;
    sub_15F55D0(v343, v329, v294, v293, (__int64)v279, v280);
    v282 = v333;
    v295 = *(_DWORD *)(v343 + 20) & 0xFFFFFFF;
  }
  v296 = (v295 + 1) & 0xFFFFFFF;
  v341 = *(_DWORD *)(v343 + 20);
  *(_DWORD *)(v343 + 20) = v296 | v341 & 0xF0000000;
  if ( v296 & 0x40000000 | v341 & 0x40000000 )
    v297 = *(_QWORD *)(v343 - 8);
  else
    v297 = v351 - 24 * v296;
  v298 = (_QWORD *)(v297 + 24LL * (unsigned int)(v296 - 1));
  if ( *v298 )
  {
    v299 = v298[1];
    v300 = v298[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v300 = v299;
    if ( v299 )
      *(_QWORD *)(v299 + 16) = *(_QWORD *)(v299 + 16) & 3LL | v300;
  }
  *v298 = v282;
  v301 = *(_QWORD *)(v282 + 8);
  v298[1] = v301;
  if ( v301 )
    *(_QWORD *)(v301 + 16) = (unsigned __int64)(v298 + 1) | *(_QWORD *)(v301 + 16) & 3LL;
  v298[2] = (v282 + 8) | v298[2] & 3LL;
  *(_QWORD *)(v282 + 8) = v298;
  v302 = *(_DWORD *)(v343 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v343 + 23) & 0x40) != 0 )
    v303 = *(_QWORD *)(v343 - 8);
  else
    v303 = v351 - 24 * v302;
  *(_QWORD *)(v303 + 8LL * (unsigned int)(v302 - 1) + 24LL * *(unsigned int *)(v343 + 56) + 8) = v334;
LABEL_198:
  v356 = (char *)v358;
  v357 = 0x200000000LL;
  if ( !*(_QWORD *)(a2 + 8) )
    goto LABEL_210;
  v180 = *(_QWORD *)(a2 + 8);
  do
  {
    while ( 1 )
    {
      v181 = sub_1648700(v180);
      v184 = v181;
      if ( *((_BYTE *)v181 + 16) == 86 )
        break;
      v180 = *(_QWORD *)(v180 + 8);
      if ( !v180 )
        goto LABEL_205;
    }
    if ( *(_DWORD *)v181[7] )
    {
      sub_164D160((__int64)v181, v144, a3, a4, a5, a6, v182, v183, a9, a10);
      v187 = (unsigned int)v357;
      if ( (unsigned int)v357 < HIDWORD(v357) )
        goto LABEL_204;
    }
    else
    {
      sub_164D160((__int64)v181, v343, a3, a4, a5, a6, v182, v183, a9, a10);
      v187 = (unsigned int)v357;
      if ( (unsigned int)v357 < HIDWORD(v357) )
        goto LABEL_204;
    }
    sub_16CD150((__int64)&v356, v358, 0, 8, v185, v186);
    v187 = (unsigned int)v357;
LABEL_204:
    *(_QWORD *)&v356[8 * v187] = v184;
    LODWORD(v357) = v357 + 1;
    v180 = *(_QWORD *)(v180 + 8);
  }
  while ( v180 );
LABEL_205:
  v188 = (unsigned __int8 *)&v356[8 * (unsigned int)v357];
  if ( v356 != (char *)v188 )
  {
    v189 = (unsigned __int8 *)v356;
    do
    {
      v190 = *(_QWORD **)v189;
      v189 += 8;
      sub_15F20C0(v190);
    }
    while ( v188 != v189 );
  }
  if ( *(_QWORD *)(a2 + 8) )
  {
    LODWORD(v353) = 0;
    v355 = 257;
    v191 = sub_1599EF0(*(__int64 ***)a2);
    v192 = sub_17FE490((__int64 *)&v359, v191, v343, &v353, 1, v354);
    v355 = 257;
    LODWORD(v353) = 1;
    v193 = sub_17FE490((__int64 *)&v359, v192, v144, &v353, 1, v354);
    sub_164D160(a2, v193, a3, a4, a5, a6, v194, v195, a9, a10);
  }
LABEL_210:
  sub_15F20C0((_QWORD *)a2);
  if ( v356 != (char *)v358 )
    _libc_free((unsigned __int64)v356);
  if ( v359 )
    sub_161E7C0((__int64)&v359, (__int64)v359);
  return 1;
}
