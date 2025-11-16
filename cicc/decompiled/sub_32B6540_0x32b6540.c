// Function: sub_32B6540
// Address: 0x32b6540
//
__int64 __fastcall sub_32B6540(__int64 **a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 **v3; // r12
  const __m128i *v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // r14
  __int64 v7; // r15
  __int32 v8; // ebx
  unsigned __int8 v9; // al
  __int64 v10; // rsi
  __int16 *v11; // rax
  __int16 v12; // cx
  __int64 v13; // rax
  int v14; // ebx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rsi
  __int64 *v22; // rdi
  __m128i v23; // xmm2
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // ecx
  int v30; // esi
  int v31; // r8d
  unsigned int v32; // edx
  __int64 v33; // rax
  unsigned __int8 v34; // r10
  __int64 *v35; // rdi
  char v36; // al
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // rax
  unsigned int v40; // edx
  int v41; // edx
  __int64 v42; // rax
  unsigned __int8 v43; // r8
  __int64 *v44; // rdi
  __int64 v45; // rax
  unsigned int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rcx
  unsigned __int64 *v52; // rax
  __int64 v53; // rsi
  __m128i v54; // xmm3
  unsigned __int64 v55; // r15
  __m128i v56; // xmm4
  __int64 v57; // r14
  unsigned __int64 v58; // rbx
  __int16 *v59; // rax
  __int16 v60; // dx
  __int64 v61; // rax
  int v62; // eax
  __int64 *v63; // rdi
  __int64 *v64; // rsi
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 (__fastcall *v67)(__int64, __int64, __int64, __int64, unsigned int); // rax
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 *v70; // rcx
  __int64 v71; // rax
  char v72; // al
  __int64 *v73; // rdi
  __int64 (*v74)(); // rax
  __int64 (*v75)(); // rax
  int v76; // ecx
  __int64 v77; // rax
  int v78; // esi
  __int32 v79; // eax
  unsigned __int64 v80; // rt0
  __int64 v81; // rax
  __int64 v82; // rdi
  int v83; // r8d
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rdx
  int v87; // edx
  int v88; // ecx
  __int64 v89; // rax
  __int64 v90; // rsi
  __int64 v91; // r11
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 (*v94)(); // rax
  char v95; // al
  __int64 *v96; // rdi
  __int64 (*v97)(); // rax
  char v98; // al
  __int64 v99; // rdx
  __int64 v100; // rax
  char v101; // al
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 *v104; // rax
  __int64 v105; // rdi
  int v106; // ecx
  __int64 v107; // rax
  int v108; // edx
  __int64 v109; // r12
  __int128 v110; // rax
  int v111; // r9d
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  unsigned int *v115; // rdx
  __int64 v116; // r13
  __int64 *v117; // rdi
  __int64 v118; // rcx
  unsigned __int16 v119; // dx
  __int64 v120; // r8
  __int64 (__fastcall *v121)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 (*v122)(); // rax
  char v123; // al
  __int64 v124; // rax
  __int64 *v125; // rdx
  __int64 v126; // r10
  __int64 v127; // r11
  __int64 v128; // r14
  unsigned int v129; // edx
  __int64 v130; // r15
  int v131; // r9d
  __int64 v132; // rax
  unsigned int v133; // edx
  __int64 v134; // rax
  unsigned int v135; // edx
  __int64 v136; // rax
  unsigned int *v137; // rdx
  __int64 v138; // r15
  __int64 v139; // r13
  __int64 *v140; // rdi
  __int64 v141; // rcx
  unsigned __int16 v142; // dx
  __int64 v143; // r8
  __int64 (__fastcall *v144)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 (*v145)(); // rax
  char v146; // al
  __int64 *v147; // rax
  __int128 *v148; // rcx
  __int64 v149; // r9
  __int64 v150; // r8
  __int64 v151; // rsi
  __int64 v152; // rdx
  unsigned int *v153; // rdx
  __int64 v154; // r13
  __int64 v155; // r10
  __int64 *v156; // rdi
  __int64 v157; // rdx
  unsigned __int16 v158; // si
  __int64 (__fastcall *v159)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // r11
  __int64 (*v160)(); // rax
  char v161; // al
  __int64 v162; // r10
  __int64 *v163; // rdx
  char v164; // al
  __int64 v165; // rdx
  __int64 v166; // rax
  __int64 v167; // rax
  int v168; // edx
  __int64 v169; // r12
  __int128 v170; // rax
  int v171; // r9d
  unsigned int *v172; // rdx
  __int64 v173; // r13
  __int64 *v174; // rdi
  __int64 v175; // rcx
  unsigned __int16 v176; // dx
  __int64 v177; // r8
  __int64 (__fastcall *v178)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 (*v179)(); // rax
  char v180; // al
  unsigned int v181; // edx
  int v182; // r9d
  __int64 v183; // rax
  unsigned int v184; // edx
  unsigned int *v185; // rdx
  __int64 v186; // r13
  __int64 *v187; // rdi
  unsigned __int16 *v188; // rcx
  __int64 (__fastcall *v189)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 (*v190)(); // rax
  char v191; // al
  __int64 v192; // rax
  __int128 *v193; // rdx
  __int64 v194; // r14
  unsigned int v195; // edx
  __int64 v196; // r15
  int v197; // r9d
  __int64 v198; // rax
  unsigned int v199; // edx
  __int64 v200; // rax
  unsigned int v201; // edx
  int v202; // r9d
  __int128 v203; // rax
  int v204; // r9d
  int v205; // edx
  __int64 v206; // r12
  __int128 v207; // rax
  int v208; // r9d
  char v209; // al
  int v210; // ecx
  __int64 v211; // r13
  __int128 v212; // rax
  int v213; // r9d
  __int64 v214; // r15
  __int128 v215; // rax
  int v216; // r9d
  __int64 v217; // r13
  __int128 v218; // rax
  int v219; // r9d
  __int128 v220; // rax
  int v221; // r9d
  unsigned int *v222; // rdx
  __int64 v223; // r13
  __int64 *v224; // rdi
  __int64 v225; // rcx
  unsigned __int16 v226; // dx
  __int64 v227; // r8
  __int64 (__fastcall *v228)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 (*v229)(); // rax
  char v230; // al
  unsigned int v231; // edx
  int v232; // r9d
  __int64 v233; // rax
  unsigned int v234; // edx
  int v235; // ecx
  __int128 v236; // rax
  __int64 v237; // r13
  __int128 v238; // rax
  int v239; // r9d
  __int128 v240; // rax
  int v241; // r9d
  __int32 v242; // eax
  unsigned __int64 v243; // rt1
  __int32 v244; // eax
  unsigned __int64 v245; // rt2
  __int32 v246; // eax
  unsigned __int64 v247; // rtt
  unsigned __int64 v248; // rt0
  unsigned __int32 v249; // ecx
  __int64 v250; // rsi
  int v251; // edi
  unsigned __int64 v252; // r11
  __int64 v253; // rdx
  unsigned __int64 v254; // r10
  __int64 v255; // rdx
  __int64 **v256; // r13
  __int64 v257; // r12
  int v258; // r11d
  __int64 v259; // rcx
  __int32 v260; // r8d
  __int64 v261; // rcx
  unsigned __int64 v262; // rax
  __int64 v263; // rdi
  int v264; // ecx
  __int64 j; // rsi
  __int64 v266; // r15
  __int64 **v267; // r13
  int v268; // r12d
  __int64 *i; // rbx
  __int64 *v270; // rax
  __int64 v271; // rdi
  bool v272; // al
  unsigned int v273; // esi
  __int64 v274; // r11
  __int64 v275; // r13
  __int64 v276; // rdx
  __int64 v277; // rax
  __int64 v278; // rdx
  __int64 v279; // r15
  __int64 v280; // r14
  __int128 *v281; // rdx
  __int64 v282; // rax
  unsigned __int64 v283; // rdx
  __int64 v284; // rcx
  unsigned __int64 v285; // rax
  unsigned __int32 v286; // eax
  unsigned __int64 v287; // rt1
  char v288; // al
  __int128 v289; // [rsp-30h] [rbp-1C0h]
  __int128 v290; // [rsp-30h] [rbp-1C0h]
  __int128 v291; // [rsp-30h] [rbp-1C0h]
  __int128 v292; // [rsp-30h] [rbp-1C0h]
  __int128 v293; // [rsp-20h] [rbp-1B0h]
  __int128 v294; // [rsp-20h] [rbp-1B0h]
  __int128 v295; // [rsp-20h] [rbp-1B0h]
  __int128 v296; // [rsp-20h] [rbp-1B0h]
  __int128 v297; // [rsp-20h] [rbp-1B0h]
  __int128 v298; // [rsp-20h] [rbp-1B0h]
  __int64 v299; // [rsp-10h] [rbp-1A0h]
  __int128 v300; // [rsp-10h] [rbp-1A0h]
  __int64 v301; // [rsp-10h] [rbp-1A0h]
  __int128 v302; // [rsp-10h] [rbp-1A0h]
  __int128 v303; // [rsp-10h] [rbp-1A0h]
  __int128 v304; // [rsp-10h] [rbp-1A0h]
  __int128 v305; // [rsp-10h] [rbp-1A0h]
  __int128 v306; // [rsp-10h] [rbp-1A0h]
  __int128 v307; // [rsp-10h] [rbp-1A0h]
  __int128 v308; // [rsp-10h] [rbp-1A0h]
  __int128 v309; // [rsp-10h] [rbp-1A0h]
  __int64 v310; // [rsp-8h] [rbp-198h]
  __int64 *v311; // [rsp+10h] [rbp-180h]
  __int64 *v312; // [rsp+18h] [rbp-178h]
  unsigned __int8 v313; // [rsp+20h] [rbp-170h]
  _DWORD *v314; // [rsp+20h] [rbp-170h]
  bool v315; // [rsp+20h] [rbp-170h]
  __int32 v316; // [rsp+30h] [rbp-160h]
  __int64 v317; // [rsp+38h] [rbp-158h]
  __int32 v318; // [rsp+48h] [rbp-148h]
  unsigned __int64 v319; // [rsp+48h] [rbp-148h]
  __int64 v320; // [rsp+50h] [rbp-140h]
  unsigned __int64 v321; // [rsp+50h] [rbp-140h]
  __int64 v322; // [rsp+58h] [rbp-138h]
  char v323; // [rsp+58h] [rbp-138h]
  unsigned __int8 v324; // [rsp+58h] [rbp-138h]
  __int64 v325; // [rsp+60h] [rbp-130h]
  char v326; // [rsp+60h] [rbp-130h]
  unsigned __int8 v327; // [rsp+60h] [rbp-130h]
  __int128 v328; // [rsp+60h] [rbp-130h]
  unsigned __int64 v329; // [rsp+60h] [rbp-130h]
  __int16 v330; // [rsp+70h] [rbp-120h]
  __int64 v331; // [rsp+70h] [rbp-120h]
  __int64 v332; // [rsp+70h] [rbp-120h]
  __int64 v333; // [rsp+70h] [rbp-120h]
  int v334; // [rsp+70h] [rbp-120h]
  __int128 v335; // [rsp+70h] [rbp-120h]
  __int64 v336; // [rsp+70h] [rbp-120h]
  __int128 v337; // [rsp+70h] [rbp-120h]
  __int64 v338; // [rsp+70h] [rbp-120h]
  unsigned __int32 v339; // [rsp+70h] [rbp-120h]
  __int64 v340; // [rsp+78h] [rbp-118h]
  unsigned __int8 v341; // [rsp+80h] [rbp-110h]
  int v342; // [rsp+80h] [rbp-110h]
  __int128 v343; // [rsp+80h] [rbp-110h]
  __int128 v344; // [rsp+80h] [rbp-110h]
  int v345; // [rsp+80h] [rbp-110h]
  __int64 v346; // [rsp+88h] [rbp-108h]
  unsigned __int8 v347; // [rsp+90h] [rbp-100h]
  __int128 v348; // [rsp+90h] [rbp-100h]
  char v349; // [rsp+90h] [rbp-100h]
  __int128 v350; // [rsp+90h] [rbp-100h]
  __int128 v351; // [rsp+90h] [rbp-100h]
  char v352; // [rsp+90h] [rbp-100h]
  __int64 v353; // [rsp+90h] [rbp-100h]
  __int128 v354; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned int v355; // [rsp+BCh] [rbp-D4h] BYREF
  unsigned int v356; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v357; // [rsp+C8h] [rbp-C8h]
  __int64 v358; // [rsp+D0h] [rbp-C0h] BYREF
  int v359; // [rsp+D8h] [rbp-B8h]
  __int64 v360; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v361; // [rsp+E8h] [rbp-A8h]
  __int64 v362; // [rsp+F0h] [rbp-A0h] BYREF
  int v363; // [rsp+F8h] [rbp-98h]
  __int64 *v364; // [rsp+100h] [rbp-90h] BYREF
  int v365; // [rsp+108h] [rbp-88h]
  __int64 v366; // [rsp+110h] [rbp-80h]
  __int64 *v367; // [rsp+120h] [rbp-70h] BYREF
  __int64 *v368; // [rsp+128h] [rbp-68h]
  __int64 v369; // [rsp+130h] [rbp-60h]
  __m128i v370; // [rsp+140h] [rbp-50h] BYREF
  __int64 *v371; // [rsp+150h] [rbp-40h]
  __int64 *v372; // [rsp+158h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  v4 = *(const __m128i **)(a2 + 40);
  v5 = *a1;
  v6 = v4[2].m128i_i64[1];
  v7 = v4[3].m128i_i64[0];
  v325 = v4->m128i_i64[0];
  v8 = v4->m128i_i32[2];
  v354 = _mm_loadu_si128(v4);
  v316 = v8;
  v322 = v4[2].m128i_i64[1];
  v318 = v4[3].m128i_i32[0];
  v347 = sub_33E2470(v5, v354.m128i_i64[0], v354.m128i_i64[1]);
  v9 = sub_33E2470(*v3, v6, v7);
  v10 = *(_QWORD *)(a2 + 80);
  v341 = v9;
  v11 = *(__int16 **)(v2 + 48);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v358 = v10;
  v330 = v12;
  LOWORD(v356) = v12;
  v357 = v13;
  if ( v10 )
    sub_B96E90((__int64)&v358, v10, 1);
  v14 = *(_DWORD *)(v2 + 28);
  v359 = *(_DWORD *)(v2 + 72);
  v15 = *v3;
  v16 = (*v3)[128];
  v17 = **v3;
  v365 = v14;
  v364 = v15;
  v366 = v16;
  v15[128] = (__int64)&v364;
  v320 = v17;
  v18 = sub_33FE9E0((unsigned int)*v3, *(_DWORD *)(v2 + 24), v354.m128i_i32[0], v354.m128i_i32[2], v6, v7, v14);
  if ( v18 )
    goto LABEL_4;
  v22 = *v3;
  v23 = _mm_load_si128(&v354);
  v371 = (__int64 *)v6;
  v372 = (__int64 *)v7;
  v370 = v23;
  v18 = sub_3402EA0((_DWORD)v22, 96, (unsigned int)&v358, v356, v357, 0, (__int64)&v370, 2);
  v26 = v299;
  if ( v18 )
    goto LABEL_4;
  if ( ((v341 ^ 1) & v347) != 0 )
  {
    v29 = v356;
    v30 = 96;
    v31 = v357;
    v300 = (__int128)v354;
    v32 = (unsigned int)&v358;
    *((_QWORD *)&v293 + 1) = v7;
    *(_QWORD *)&v293 = v6;
LABEL_22:
    v33 = sub_3406EB0((unsigned int)*v3, v30, v32, v29, v31, v25, v293, v300);
LABEL_23:
    v19 = v33;
    goto LABEL_5;
  }
  if ( v330 )
  {
    if ( (unsigned __int16)(v330 - 17) > 0xD3u )
      goto LABEL_13;
  }
  else if ( !sub_30070B0((__int64)&v356) )
  {
    goto LABEL_13;
  }
  v18 = sub_3295970((__int64 *)v3, v2, (__int64)&v358, v26, v24);
  if ( v18 )
    goto LABEL_4;
LABEL_13:
  v27 = sub_33E1790(v6, v7, 1);
  if ( !v27 )
    goto LABEL_26;
  v331 = *(_QWORD *)(v27 + 96);
  if ( *(void **)(v331 + 24) == sub_C33340() )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(v331 + 32) + 20LL) & 7) != 3 )
      goto LABEL_26;
    v28 = *(_QWORD *)(v331 + 32);
  }
  else
  {
    if ( (*(_BYTE *)(v331 + 44) & 7) != 3 )
      goto LABEL_26;
    v28 = v331 + 24;
  }
  if ( (*(_BYTE *)(v28 + 20) & 8) != 0 || (*(_BYTE *)(v320 + 864) & 0x10) != 0 || (v14 & 0x80u) != 0 )
  {
    v19 = v354.m128i_i64[0];
    goto LABEL_5;
  }
LABEL_26:
  v18 = sub_329BF20((__int64 *)v3, v2);
  if ( v18 )
    goto LABEL_4;
  v34 = *((_BYTE *)v3 + 33);
  v35 = v3[1];
  if ( v34 )
  {
    v313 = *((_BYTE *)v3 + 33);
    v36 = sub_328A020((__int64)v35, 0x61u, v356, v357, 0);
    v34 = v313;
    if ( !v36 )
      goto LABEL_29;
  }
  v39 = sub_328F830(v35, v6, v7, (__int64)*v3, v34, *((_BYTE *)v3 + 35), 0);
  LODWORD(v25) = 0;
  if ( v39 )
  {
    *((_QWORD *)&v300 + 1) = v40;
    *(_QWORD *)&v300 = v39;
    v293 = (__int128)v354;
LABEL_41:
    v29 = v356;
    v31 = v357;
    v32 = (unsigned int)&v358;
LABEL_42:
    v30 = 97;
    goto LABEL_22;
  }
  v43 = *((_BYTE *)v3 + 33);
  if ( v43 )
  {
    v44 = v3[1];
    v43 = sub_328A020((__int64)v44, 0x61u, v356, v357, 0);
    if ( !v43 )
      goto LABEL_29;
  }
  else
  {
    v44 = v3[1];
  }
  v45 = sub_328F830(v44, v354.m128i_i64[0], v354.m128i_i64[1], (__int64)*v3, v43, *((_BYTE *)v3 + 35), 0);
  v25 = v301;
  if ( v45 )
  {
    *((_QWORD *)&v300 + 1) = v46;
    *(_QWORD *)&v300 = v45;
    *((_QWORD *)&v293 + 1) = v7;
    *(_QWORD *)&v293 = v6;
    goto LABEL_41;
  }
LABEL_29:
  v37 = *(_QWORD *)(v325 + 56);
  if ( v37 )
  {
    v38 = 1;
    do
    {
      if ( *(_DWORD *)(v37 + 8) == v316 )
      {
        if ( !v38 )
          goto LABEL_44;
        v37 = *(_QWORD *)(v37 + 32);
        if ( !v37 )
          goto LABEL_56;
        if ( *(_DWORD *)(v37 + 8) == v316 )
          goto LABEL_44;
        v38 = 0;
      }
      v37 = *(_QWORD *)(v37 + 32);
    }
    while ( v37 );
    if ( v38 == 1 )
      goto LABEL_44;
LABEL_56:
    if ( *(_DWORD *)(v325 + 24) == 98 )
    {
      v47 = sub_33E1790(*(_QWORD *)(*(_QWORD *)(v325 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v325 + 40) + 48LL), 1);
      if ( v47 )
      {
        if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v47 + 96) + 24LL), -2.0) )
        {
          v48 = sub_3406EB0(
                  (unsigned int)*v3,
                  96,
                  (unsigned int)&v358,
                  v356,
                  v357,
                  v25,
                  *(_OWORD *)*(_QWORD *)(v325 + 40),
                  *(_OWORD *)*(_QWORD *)(v325 + 40));
          v29 = v356;
          v31 = v357;
          *((_QWORD *)&v300 + 1) = v49;
          v32 = (unsigned int)&v358;
          *(_QWORD *)&v300 = v48;
          *((_QWORD *)&v293 + 1) = v7;
          *(_QWORD *)&v293 = v6;
          goto LABEL_42;
        }
      }
    }
  }
LABEL_44:
  v41 = 1;
  v42 = *(_QWORD *)(v322 + 56);
  if ( v42 )
  {
    do
    {
      if ( v318 == *(_DWORD *)(v42 + 8) )
      {
        if ( !v41 )
          goto LABEL_61;
        v42 = *(_QWORD *)(v42 + 32);
        if ( !v42 )
          goto LABEL_104;
        if ( v318 == *(_DWORD *)(v42 + 8) )
          goto LABEL_61;
        v41 = 0;
      }
      v42 = *(_QWORD *)(v42 + 32);
    }
    while ( v42 );
    if ( v41 == 1 )
      goto LABEL_61;
LABEL_104:
    if ( *(_DWORD *)(v322 + 24) == 98 )
    {
      v84 = sub_33E1790(*(_QWORD *)(*(_QWORD *)(v322 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v322 + 40) + 48LL), 1);
      if ( v84 )
      {
        v333 = *(_QWORD *)(v84 + 96);
        v314 = sub_C33320();
        sub_C3B1B0((__int64)&v370, -2.0);
        sub_C407B0(&v367, v370.m128i_i64, v314);
        sub_C338F0((__int64)&v370);
        sub_C41640((__int64 *)&v367, *(_DWORD **)(v333 + 24), 1, (bool *)v370.m128i_i8);
        v311 = *(__int64 **)(v333 + 24);
        v315 = 0;
        v312 = (__int64 *)sub_C33340();
        if ( v311 == v367 )
        {
          v271 = v333 + 24;
          if ( v312 == v311 )
            v272 = sub_C3E590(v271, (__int64)&v367);
          else
            v272 = sub_C33D00(v271, (__int64)&v367);
          v315 = v272;
        }
        if ( v312 == v367 )
        {
          if ( v368 )
          {
            v340 = v7;
            v266 = v2;
            v267 = v3;
            v268 = v14;
            for ( i = &v368[3 * *(v368 - 1)]; v368 != i; sub_91D830(i) )
              i -= 3;
            v270 = i;
            v14 = v268;
            v3 = v267;
            v2 = v266;
            v7 = v340;
            j_j_j___libc_free_0_0((unsigned __int64)(v270 - 1));
          }
        }
        else
        {
          sub_C338F0((__int64)&v367);
        }
        if ( v315 )
        {
          v85 = sub_3406EB0(
                  (unsigned int)*v3,
                  96,
                  (unsigned int)&v358,
                  v356,
                  v357,
                  v25,
                  *(_OWORD *)*(_QWORD *)(v322 + 40),
                  *(_OWORD *)*(_QWORD *)(v322 + 40));
          v29 = v356;
          v31 = v357;
          *((_QWORD *)&v300 + 1) = v86;
          v32 = (unsigned int)&v358;
          *(_QWORD *)&v300 = v85;
          v293 = (__int128)v354;
          goto LABEL_42;
        }
      }
    }
  }
LABEL_61:
  v50 = *((_DWORD *)v3 + 6);
  v51 = *(unsigned __int8 *)(v320 + 864);
  if ( (v51 & 4) != 0 || (v14 & 0x20) != 0 )
  {
    if ( v50 > 2 )
      goto LABEL_65;
    if ( *(_DWORD *)(v325 + 24) != 244
      || (v100 = *(_QWORD *)(v325 + 40), v322 != *(_QWORD *)v100)
      || *(_DWORD *)(v100 + 8) != v318 )
    {
      if ( *(_DWORD *)(v322 + 24) != 244
        || (v99 = *(_QWORD *)(v322 + 40), *(_QWORD *)v99 != v325)
        || *(_DWORD *)(v99 + 8) != v316 )
      {
        v51 &= 0x11u;
        if ( (_BYTE)v51 != 17 && (v14 & 0x880) != 0x880 )
          goto LABEL_65;
        goto LABEL_116;
      }
    }
    v18 = sub_33FE730(*v3, &v358, v356, v357, 0, 0.0);
LABEL_4:
    v19 = v18;
    goto LABEL_5;
  }
  v51 &= 0x11u;
  if ( (_BYTE)v51 != 17 && (v14 & 0x880) != 0x880 || v50 > 2 )
    goto LABEL_65;
LABEL_116:
  if ( v341 )
  {
    if ( *(_DWORD *)(v325 + 24) == 96
      && (unsigned __int8)sub_33E2470(
                            *v3,
                            *(_QWORD *)(*(_QWORD *)(v325 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(v325 + 40) + 48LL)) )
    {
      *((_QWORD *)&v306 + 1) = v7;
      *(_QWORD *)&v306 = v6;
      *(_QWORD *)&v203 = sub_3406EB0(
                           (unsigned int)*v3,
                           96,
                           (unsigned int)&v358,
                           v356,
                           v357,
                           v202,
                           *(_OWORD *)(*(_QWORD *)(v325 + 40) + 40LL),
                           v306);
      v33 = sub_3406EB0(
              (unsigned int)*v3,
              96,
              (unsigned int)&v358,
              v356,
              v357,
              v204,
              *(_OWORD *)*(_QWORD *)(v325 + 40),
              v203);
      goto LABEL_23;
    }
    goto LABEL_118;
  }
  v334 = v356;
  v342 = v357;
  v101 = sub_328A020((__int64)v3[1], 0x62u, v356, v357, 0);
  v88 = v342;
  v87 = v334;
  if ( v101 && !v347 )
  {
    if ( *(_DWORD *)(v325 + 24) == 98 )
    {
      v352 = sub_33E2470(*v3, **(_QWORD **)(v325 + 40), *(_QWORD *)(*(_QWORD *)(v325 + 40) + 8LL));
      v209 = sub_33E2470(*v3, *(_QWORD *)(*(_QWORD *)(v325 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v325 + 40) + 48LL));
      if ( v352 != 1 )
      {
        if ( v209 )
        {
          v166 = *(_QWORD *)(v325 + 40);
          if ( v322 == *(_QWORD *)v166 && *(_DWORD *)(v166 + 8) == v318 )
          {
            v217 = (__int64)*v3;
            *(_QWORD *)&v218 = sub_33FE730(*v3, &v358, v356, v357, 0, 1.0);
            *(_QWORD *)&v220 = sub_3406EB0(
                                 v217,
                                 96,
                                 (unsigned int)&v358,
                                 v356,
                                 v357,
                                 v219,
                                 *(_OWORD *)(*(_QWORD *)(v325 + 40) + 40LL),
                                 v218);
            *((_QWORD *)&v297 + 1) = v7;
            *(_QWORD *)&v297 = v6;
            v33 = sub_3406EB0((unsigned int)*v3, 98, (unsigned int)&v358, v356, v357, v221, v297, v220);
            goto LABEL_23;
          }
          if ( *(_DWORD *)(v322 + 24) == 96 )
          {
            v102 = *(_QWORD *)(v322 + 40);
            if ( *(_QWORD *)v102 == *(_QWORD *)(v102 + 40) )
            {
              v210 = *(_DWORD *)(v102 + 8);
              if ( v210 == *(_DWORD *)(v102 + 48) && *(_QWORD *)v166 == *(_QWORD *)v102 && v210 == *(_DWORD *)(v166 + 8) )
              {
                v211 = (__int64)*v3;
                *(_QWORD *)&v212 = sub_33FE730(*v3, &v358, v356, v357, 0, 2.0);
                v214 = v325;
                v307 = v212;
LABEL_251:
                *(_QWORD *)&v215 = sub_3406EB0(
                                     v211,
                                     96,
                                     (unsigned int)&v358,
                                     v356,
                                     v357,
                                     v213,
                                     *(_OWORD *)(*(_QWORD *)(v214 + 40) + 40LL),
                                     v307);
                v33 = sub_3406EB0(
                        (unsigned int)*v3,
                        98,
                        (unsigned int)&v358,
                        v356,
                        v357,
                        v216,
                        *(_OWORD *)*(_QWORD *)(v214 + 40),
                        v215);
                goto LABEL_23;
              }
            }
            if ( *(_DWORD *)(v325 + 24) != 96 )
              goto LABEL_155;
            goto LABEL_211;
          }
        }
      }
    }
    if ( *(_DWORD *)(v322 + 24) == 98
      && (v349 = sub_33E2470(*v3, **(_QWORD **)(v322 + 40), *(_QWORD *)(*(_QWORD *)(v322 + 40) + 8LL)),
          v164 = sub_33E2470(
                   *v3,
                   *(_QWORD *)(*(_QWORD *)(v322 + 40) + 40LL),
                   *(_QWORD *)(*(_QWORD *)(v322 + 40) + 48LL)),
          v349 != 1)
      && v164 )
    {
      v165 = *(_QWORD *)(v322 + 40);
      if ( *(_QWORD *)v165 == v325 && *(_DWORD *)(v165 + 8) == v316 )
      {
        v237 = (__int64)*v3;
        *(_QWORD *)&v238 = sub_33FE730(*v3, &v358, v356, v357, 0, 1.0);
        *(_QWORD *)&v240 = sub_3406EB0(
                             v237,
                             96,
                             (unsigned int)&v358,
                             v356,
                             v357,
                             v239,
                             *(_OWORD *)(*(_QWORD *)(v322 + 40) + 40LL),
                             v238);
        v33 = sub_3406EB0((unsigned int)*v3, 98, (unsigned int)&v358, v356, v357, v241, *(_OWORD *)&v354, v240);
        goto LABEL_23;
      }
      if ( *(_DWORD *)(v325 + 24) == 96 )
      {
        v166 = *(_QWORD *)(v325 + 40);
        if ( *(_QWORD *)v166 == *(_QWORD *)(v166 + 40) )
        {
          v235 = *(_DWORD *)(v166 + 8);
          if ( v235 == *(_DWORD *)(v166 + 48) && *(_QWORD *)v165 == *(_QWORD *)v166 && v235 == *(_DWORD *)(v165 + 8) )
          {
            v211 = (__int64)*v3;
            *(_QWORD *)&v236 = sub_33FE730(*v3, &v358, v356, v357, 0, 2.0);
            v214 = v322;
            v307 = v236;
            goto LABEL_251;
          }
        }
LABEL_211:
        if ( !(unsigned __int8)sub_33E2470(*v3, *(_QWORD *)v166, *(_QWORD *)(v166 + 8)) )
        {
          v167 = *(_QWORD *)(v325 + 40);
          if ( *(_QWORD *)v167 == *(_QWORD *)(v167 + 40) )
          {
            v168 = *(_DWORD *)(v167 + 8);
            if ( v322 == *(_QWORD *)v167 && *(_DWORD *)(v167 + 48) == v168 && v168 == v318 )
            {
              v169 = (__int64)*v3;
              *(_QWORD *)&v170 = sub_33FE730(v169, &v358, v356, v357, 0, 3.0);
              *((_QWORD *)&v295 + 1) = v7;
              *(_QWORD *)&v295 = v6;
              v112 = sub_3406EB0(v169, 98, (unsigned int)&v358, v356, v357, v171, v295, v170);
              goto LABEL_165;
            }
          }
        }
      }
    }
    else if ( *(_DWORD *)(v325 + 24) == 96 )
    {
      v166 = *(_QWORD *)(v325 + 40);
      goto LABEL_211;
    }
    if ( *(_DWORD *)(v322 + 24) != 96 )
      goto LABEL_118;
    v102 = *(_QWORD *)(v322 + 40);
LABEL_155:
    if ( !(unsigned __int8)sub_33E2470(*v3, *(_QWORD *)v102, *(_QWORD *)(v102 + 8)) )
    {
      v103 = *(_QWORD *)(v322 + 40);
      if ( *(_QWORD *)v103 == *(_QWORD *)(v103 + 40) )
      {
        v205 = *(_DWORD *)(v103 + 8);
        if ( *(_QWORD *)v103 == v325 && *(_DWORD *)(v103 + 48) == v205 && v205 == v316 )
        {
          v206 = (__int64)*v3;
          *(_QWORD *)&v207 = sub_33FE730(v206, &v358, v356, v357, 0, 3.0);
          v112 = sub_3406EB0(v206, 98, (unsigned int)&v358, v356, v357, v208, *(_OWORD *)&v354, v207);
          goto LABEL_165;
        }
      }
    }
    if ( *(_DWORD *)(v325 + 24) == 96 && *(_DWORD *)(v322 + 24) == 96 )
    {
      v104 = *(__int64 **)(v325 + 40);
      v105 = *v104;
      if ( *v104 == v104[5] )
      {
        v106 = *((_DWORD *)v104 + 2);
        if ( v106 == *((_DWORD *)v104 + 12) )
        {
          v107 = *(_QWORD *)(v322 + 40);
          if ( *(_QWORD *)v107 == *(_QWORD *)(v107 + 40) )
          {
            v108 = *(_DWORD *)(v107 + 8);
            if ( v105 == *(_QWORD *)v107 && *(_DWORD *)(v107 + 48) == v108 && v106 == v108 )
            {
              v109 = (__int64)*v3;
              *(_QWORD *)&v110 = sub_33FE730(v109, &v358, v356, v357, 0, 4.0);
              v112 = sub_3406EB0(
                       v109,
                       98,
                       (unsigned int)&v358,
                       v356,
                       v357,
                       v111,
                       *(_OWORD *)*(_QWORD *)(v325 + 40),
                       v110);
LABEL_165:
              v19 = v112;
              goto LABEL_5;
            }
          }
        }
      }
    }
LABEL_118:
    v87 = v356;
    v88 = v357;
  }
  v89 = sub_328C120((__int64 *)v3, 0x178u, 0x60u, (int)&v358, v87, v88, v325, v322, v14);
  if ( v89 )
  {
    v19 = v89;
    goto LABEL_5;
  }
LABEL_65:
  v52 = *(unsigned __int64 **)(v2 + 40);
  v53 = *(_QWORD *)(v2 + 80);
  v54 = _mm_loadu_si128((const __m128i *)v52);
  v55 = *v52;
  v56 = _mm_loadu_si128((const __m128i *)(v52 + 5));
  v354.m128i_i32[0] = *((_DWORD *)v52 + 2);
  v57 = *((unsigned int *)v52 + 12);
  v58 = v52[5];
  v59 = *(__int16 **)(v2 + 48);
  v346 = v56.m128i_i64[1];
  v60 = *v59;
  v61 = *((_QWORD *)v59 + 1);
  v362 = v53;
  LOWORD(v360) = v60;
  v361 = v61;
  if ( v53 )
    sub_B96E90((__int64)&v362, v53, 1);
  v62 = *(_DWORD *)(v2 + 72);
  v63 = v3[1];
  v369 = v2;
  v64 = *v3;
  v65 = *((unsigned __int8 *)v3 + 33);
  v363 = v62;
  v367 = v64;
  v368 = v63;
  v332 = *v64;
  v66 = *v63;
  if ( !(_BYTE)v65 )
  {
    v94 = *(__int64 (**)())(v66 + 1608);
    if ( v94 == sub_2FE3540 )
      goto LABEL_122;
    v95 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, __int64, __int64, __int64))v94)(
            v63,
            v64[5],
            (unsigned int)v360,
            v361,
            v65,
            v25);
    v65 = 0;
    if ( !v95 )
      goto LABEL_122;
    goto LABEL_131;
  }
  v67 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned int))(v66 + 1640);
  if ( v67 == sub_2FE40B0 )
  {
    v68 = **(unsigned __int16 **)(v2 + 48);
    v69 = 1;
    if ( (_WORD)v68 != 1 && (!(_WORD)v68 || (v69 = (unsigned __int16)v68, !v63[v68 + 14]))
      || *((_BYTE *)v63 + 500 * v69 + 6565) )
    {
      v65 = 0;
    }
  }
  else
  {
    v65 = ((unsigned int (__fastcall *)(__int64 *, __int64 *, __int64, __int64, __int64, __int64))v67)(
            v63,
            v64,
            v2,
            v51,
            v65,
            v25);
    if ( !*((_BYTE *)v3 + 33) )
      goto LABEL_136;
  }
  v70 = v368;
  v71 = 1;
  if ( (_WORD)v360 == 1 || (_WORD)v360 && (v71 = (unsigned __int16)v360, v368[(unsigned __int16)v360 + 14]) )
  {
    v72 = *((_BYTE *)v368 + 500 * v71 + 6564);
    if ( !v72 || v72 == 4 )
    {
LABEL_136:
      v96 = v3[1];
      v97 = *(__int64 (**)())(*v96 + 1608);
      if ( v97 == sub_2FE3540 )
        goto LABEL_79;
      v327 = v65;
      v98 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, __int64, __int64, __int64))v97)(
              v96,
              (*v3)[5],
              (unsigned int)v360,
              v361,
              v65,
              v25);
      v65 = v327;
      if ( !v98 )
        goto LABEL_79;
LABEL_131:
      if ( *(_DWORD *)(v332 + 952) )
      {
        v326 = v65 | *(_BYTE *)(v332 + 864) & 1;
        if ( !v326 )
        {
          v65 = 0;
          if ( (*(_BYTE *)(v2 + 29) & 2) == 0 )
            goto LABEL_122;
        }
      }
      else
      {
        v326 = 1;
      }
      goto LABEL_80;
    }
  }
LABEL_79:
  v326 = v65;
  if ( !(_BYTE)v65 )
    goto LABEL_122;
LABEL_80:
  if ( v58 == v55 && v354.m128i_i32[0] == (_DWORD)v57 )
    goto LABEL_122;
  v73 = v3[1];
  v74 = *(__int64 (**)())(*v73 + 1648);
  if ( v74 != sub_2FE3570 )
  {
    v324 = v65;
    v288 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64, __int64))v74)(
             v73,
             (unsigned int)v360,
             v361,
             *((unsigned int *)v3 + 7),
             v65,
             v25);
    v65 = v324;
    if ( v288 )
      goto LABEL_122;
    v73 = v3[1];
  }
  v355 = 151 - ((_BYTE)v65 == 0);
  v75 = *(__int64 (**)())(*v73 + 512);
  if ( v75 == sub_2FE30F0
    || (v323 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, __int64 *, __int64, __int64))v75)(
                 v73,
                 (unsigned int)v360,
                 v361,
                 v70,
                 v65,
                 v25)) == 0 )
  {
    v76 = *(_DWORD *)(v55 + 24);
    v323 = 0;
    if ( v76 == 98 )
      goto LABEL_84;
LABEL_88:
    v78 = *(_DWORD *)(v58 + 24);
    if ( v78 != 98 )
    {
      v246 = v57;
      v247 = v55;
      v55 = v58;
      v58 = v247;
      v57 = v354.m128i_u32[0];
      v354.m128i_i32[0] = v246;
      goto LABEL_93;
    }
    v79 = v57;
    v57 = v354.m128i_u32[0];
    v80 = v58;
    v58 = v55;
    v55 = v80;
    v354.m128i_i32[0] = v79;
    goto LABEL_90;
  }
  v76 = *(_DWORD *)(v55 + 24);
  if ( v76 != 98 )
    goto LABEL_88;
  if ( v326 )
  {
    if ( *(_DWORD *)(v58 + 24) != 98 )
    {
LABEL_330:
      v281 = *(__int128 **)(v55 + 40);
      v309 = __PAIR128__(v57 | v346 & 0xFFFFFFFF00000000LL, v58);
LABEL_331:
      v136 = sub_340F900(
               (_DWORD)v367,
               v355,
               (unsigned int)&v362,
               v360,
               v361,
               v25,
               *v281,
               *(__int128 *)((char *)v281 + 40),
               v309);
      goto LABEL_183;
    }
  }
  else
  {
    if ( (*(_BYTE *)(v55 + 29) & 2) == 0 )
      goto LABEL_273;
    if ( *(_DWORD *)(v58 + 24) != 98 || (*(_BYTE *)(v58 + 29) & 2) == 0 )
      goto LABEL_330;
  }
  v282 = *(_QWORD *)(v55 + 56);
  v283 = 0;
  while ( v282 )
  {
    v282 = *(_QWORD *)(v282 + 32);
    ++v283;
  }
  v284 = *(_QWORD *)(v58 + 56);
  v285 = 0;
  while ( v284 )
  {
    v284 = *(_QWORD *)(v284 + 32);
    ++v285;
  }
  if ( v283 <= v285 )
  {
    if ( v326 || (*(_BYTE *)(v55 + 29) & 2) != 0 )
      goto LABEL_330;
    goto LABEL_273;
  }
  v76 = *(_DWORD *)(v58 + 24);
  if ( v76 != 98 )
  {
LABEL_90:
    if ( v326 )
    {
LABEL_91:
      if ( v323 || (v81 = *(_QWORD *)(v55 + 56), v78 = 98, v81) && !*(_QWORD *)(v81 + 32) )
      {
        v281 = *(__int128 **)(v55 + 40);
        v309 = __PAIR128__(v57 | v54.m128i_i64[1] & 0xFFFFFFFF00000000LL, v58);
        goto LABEL_331;
      }
      goto LABEL_93;
    }
    goto LABEL_275;
  }
  v286 = v354.m128i_i32[0];
  v287 = v58;
  v58 = v55;
  v55 = v287;
  v354.m128i_i32[0] = v57;
  v57 = v286;
LABEL_84:
  if ( v326 || (*(_BYTE *)(v55 + 29) & 2) != 0 )
  {
    if ( v323 )
      goto LABEL_330;
    v77 = *(_QWORD *)(v55 + 56);
    v76 = 98;
    if ( v77 )
    {
      if ( !*(_QWORD *)(v77 + 32) )
        goto LABEL_330;
    }
    goto LABEL_88;
  }
LABEL_273:
  v76 = *(_DWORD *)(v58 + 24);
  if ( v76 != 98 )
  {
    if ( (*(_BYTE *)(v332 + 864) & 1) == 0 && (*(_BYTE *)(v2 + 29) & 8) == 0 )
    {
      v244 = v57;
      v245 = v58;
      v58 = v55;
      v55 = v245;
      v57 = v354.m128i_u32[0];
      v354.m128i_i32[0] = v244;
      goto LABEL_168;
    }
    v248 = v55;
    v55 = v58;
    v58 = v248;
    v78 = v76;
    v249 = v354.m128i_i32[0];
    v354.m128i_i32[0] = v57;
    v57 = v249;
    v76 = 98;
    goto LABEL_282;
  }
  v242 = v57;
  v57 = v354.m128i_u32[0];
  v243 = v55;
  v55 = v58;
  v58 = v243;
  v354.m128i_i32[0] = v242;
LABEL_275:
  if ( (*(_BYTE *)(v55 + 29) & 2) != 0 )
    goto LABEL_91;
  v78 = 98;
LABEL_93:
  if ( (*(_BYTE *)(v332 + 864) & 1) == 0 && (*(_BYTE *)(v2 + 29) & 8) == 0 )
    goto LABEL_167;
  if ( (unsigned int)(v76 - 150) <= 1 )
  {
    v82 = *(_QWORD *)(v58 + 56);
    v83 = 1;
    while ( v82 )
    {
      if ( *(_DWORD *)(v82 + 8) == (_DWORD)v57 )
      {
        if ( !v83 )
          goto LABEL_282;
        v83 = 0;
      }
      v82 = *(_QWORD *)(v82 + 32);
    }
    if ( v83 != 1 )
    {
      v252 = v58;
      v319 = v55;
      v339 = v57;
      v253 = v354.m128i_u32[0];
      goto LABEL_291;
    }
  }
LABEL_282:
  if ( (unsigned int)(v78 - 150) <= 1 )
  {
    v250 = *(_QWORD *)(v55 + 56);
    v251 = 1;
    while ( v250 )
    {
      if ( *(_DWORD *)(v250 + 8) == v354.m128i_i32[0] )
      {
        if ( !v251 )
          goto LABEL_167;
        v251 = 0;
      }
      v250 = *(_QWORD *)(v250 + 32);
    }
    if ( v251 != 1 )
    {
      v319 = v58;
      v252 = v55;
      v253 = (unsigned int)v57;
      v339 = v354.m128i_i32[0];
LABEL_291:
      v25 = v339;
      v254 = v252;
      v317 = v253;
      v255 = v2;
      v256 = v3;
      v257 = v252;
      v258 = v76;
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)(v254 + 24) - 150) > 1 )
          goto LABEL_166;
        v259 = *(_QWORD *)(v254 + 56);
        v260 = 1;
        while ( v259 )
        {
          if ( *(_DWORD *)(v259 + 8) == (_DWORD)v25 )
          {
            if ( !v260 )
              goto LABEL_166;
            v260 = 0;
          }
          v259 = *(_QWORD *)(v259 + 32);
        }
        if ( v260 == 1 )
        {
LABEL_166:
          v76 = v258;
          v3 = v256;
          goto LABEL_167;
        }
        v261 = *(_QWORD *)(v254 + 40);
        v25 = *(unsigned int *)(v261 + 88);
        v262 = *(_QWORD *)(v261 + 80);
        v321 = v262;
        if ( *(_DWORD *)(v262 + 24) == 98 )
          break;
LABEL_301:
        v254 = v262;
      }
      v263 = *(_QWORD *)(v262 + 56);
      v264 = 1;
      for ( j = v263; j; j = *(_QWORD *)(j + 32) )
      {
        if ( *(_DWORD *)(j + 8) == (_DWORD)v25 )
        {
          if ( !v264 )
            goto LABEL_318;
          v264 = v260;
        }
      }
      if ( v264 != 1 )
      {
        v354.m128i_i32[0] = v260;
        v274 = v257;
        v273 = v355;
        v3 = v256;
        v275 = v255;
        v276 = v317;
        goto LABEL_324;
      }
LABEL_318:
      v273 = v355;
      while ( v263 )
      {
        if ( v355 != *(_DWORD *)(*(_QWORD *)(v263 + 16) + 24LL) )
          goto LABEL_301;
        v263 = *(_QWORD *)(v263 + 32);
      }
      v274 = v257;
      v354.m128i_i32[0] = v260;
      v3 = v256;
      v275 = v255;
      v276 = v317;
LABEL_324:
      v353 = v274;
      v329 = v254;
      v345 = v25;
      *((_QWORD *)&v308 + 1) = v276;
      *(_QWORD *)&v308 = v319;
      v277 = sub_340F900(
               (unsigned int)*v3,
               v273,
               (unsigned int)&v362,
               v360,
               v361,
               v25,
               *(_OWORD *)*(_QWORD *)(v262 + 40),
               *(_OWORD *)(*(_QWORD *)(v262 + 40) + 40LL),
               v308);
      v279 = v278;
      v280 = v277;
      sub_33F9B80((unsigned int)*v3, v321, v345, v277, v278, 0, 0, 1);
      sub_34151B0(*v3, v321, v280);
      *((_QWORD *)&v298 + 1) = v279;
      *(_QWORD *)&v298 = v280;
      sub_33EC3B0(
        (unsigned int)*v3,
        v329,
        **(_QWORD **)(v329 + 40),
        *(_QWORD *)(*(_QWORD *)(v329 + 40) + 8LL),
        *(_QWORD *)(*(_QWORD *)(v329 + 40) + 40LL),
        *(_QWORD *)(*(_QWORD *)(v329 + 40) + 48LL),
        v298);
      v91 = v353;
      if ( !*(_DWORD *)(v353 + 24) )
        v91 = v275;
      v90 = v362;
      if ( !v362 )
        goto LABEL_126;
      goto LABEL_124;
    }
  }
LABEL_167:
  if ( v76 == 233 )
  {
    v172 = *(unsigned int **)(v58 + 40);
    v173 = *(_QWORD *)v172;
    if ( *(_DWORD *)(*(_QWORD *)v172 + 24LL) == 98 && (v326 || (*(_BYTE *)(v173 + 29) & 2) != 0) )
    {
      v174 = v3[1];
      v175 = *(_QWORD *)(v173 + 48) + 16LL * v172[2];
      v176 = *(_WORD *)v175;
      v177 = *(_QWORD *)(v175 + 8);
      v178 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v174 + 1576);
      if ( v178 == sub_2FE39B0 )
      {
        v179 = *(__int64 (**)())(*v174 + 1560);
        if ( v179 == sub_2D566A0 )
          goto LABEL_168;
        v180 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, __int64, __int64))v179)(
                 v174,
                 v360,
                 v361,
                 v176,
                 v177,
                 v25);
      }
      else
      {
        v180 = v178((__int64)v174, (__int64)*v3, v355, (unsigned int)v360, v361, v25, v176, v177);
      }
      if ( v180 )
      {
        *(_QWORD *)&v350 = sub_33FAF80(
                             (_DWORD)v367,
                             233,
                             (unsigned int)&v362,
                             v360,
                             v361,
                             v25,
                             *(_OWORD *)(*(_QWORD *)(v173 + 40) + 40LL));
        *((_QWORD *)&v350 + 1) = v181;
        v183 = sub_33FAF80((_DWORD)v367, 233, (unsigned int)&v362, v360, v361, v182, *(_OWORD *)*(_QWORD *)(v173 + 40));
        *((_QWORD *)&v290 + 1) = v184;
        *(_QWORD *)&v290 = v183;
        v136 = sub_340F900(
                 (_DWORD)v367,
                 v355,
                 (unsigned int)&v362,
                 v360,
                 v361,
                 v183,
                 v290,
                 v350,
                 __PAIR128__(v354.m128i_u32[0] | v346 & 0xFFFFFFFF00000000LL, v55));
        goto LABEL_183;
      }
    }
  }
LABEL_168:
  if ( *(_DWORD *)(v55 + 24) == 233 )
  {
    v222 = *(unsigned int **)(v55 + 40);
    v223 = *(_QWORD *)v222;
    if ( *(_DWORD *)(*(_QWORD *)v222 + 24LL) == 98 && (v326 || (*(_BYTE *)(v223 + 29) & 2) != 0) )
    {
      v224 = v3[1];
      v225 = *(_QWORD *)(v223 + 48) + 16LL * v222[2];
      v226 = *(_WORD *)v225;
      v227 = *(_QWORD *)(v225 + 8);
      v228 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v224 + 1576);
      if ( v228 == sub_2FE39B0 )
      {
        v229 = *(__int64 (**)())(*v224 + 1560);
        if ( v229 == sub_2D566A0 )
          goto LABEL_169;
        v230 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, __int64, __int64))v229)(
                 v224,
                 v360,
                 v361,
                 v226,
                 v227,
                 v25);
      }
      else
      {
        v230 = v228((__int64)v224, (__int64)*v3, v355, (unsigned int)v360, v361, v25, v226, v227);
      }
      if ( v230 )
      {
        v354.m128i_i64[0] = sub_33FAF80(
                              (_DWORD)v367,
                              233,
                              (unsigned int)&v362,
                              v360,
                              v361,
                              v25,
                              *(_OWORD *)(*(_QWORD *)(v223 + 40) + 40LL));
        v354.m128i_i64[1] = v231;
        v233 = sub_33FAF80((_DWORD)v367, 233, (unsigned int)&v362, v360, v361, v232, *(_OWORD *)*(_QWORD *)(v223 + 40));
        *((_QWORD *)&v292 + 1) = v234;
        *(_QWORD *)&v292 = v233;
        v136 = sub_340F900(
                 (_DWORD)v367,
                 v355,
                 (unsigned int)&v362,
                 v360,
                 v361,
                 v233,
                 v292,
                 *(_OWORD *)&v354,
                 __PAIR128__(v57 | v54.m128i_i64[1] & 0xFFFFFFFF00000000LL, v58));
        goto LABEL_183;
      }
    }
  }
LABEL_169:
  if ( !v323 )
    goto LABEL_122;
  if ( (unsigned int)(*(_DWORD *)(v58 + 24) - 150) <= 1 )
  {
    v113 = *(_QWORD *)(*(_QWORD *)(v58 + 40) + 80LL);
    if ( *(_DWORD *)(v113 + 24) == 233 )
    {
      v185 = *(unsigned int **)(v113 + 40);
      v186 = *(_QWORD *)v185;
      if ( *(_DWORD *)(*(_QWORD *)v185 + 24LL) == 98 && (v326 || (*(_BYTE *)(v186 + 29) & 2) != 0) )
      {
        v187 = v3[1];
        v25 = *v187;
        v189 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v187 + 1576);
        if ( v189 == sub_2FE39B0 )
        {
          v190 = *(__int64 (**)())(v25 + 1560);
          if ( v190 == sub_2D566A0 )
            goto LABEL_172;
          v188 = (unsigned __int16 *)(*(_QWORD *)(v186 + 48) + 16LL * v185[2]);
          v191 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _QWORD))v190)(
                   v187,
                   v360,
                   v361,
                   *v188,
                   *((_QWORD *)v188 + 1));
        }
        else
        {
          v191 = ((__int64 (__fastcall *)(__int64 *, __int64 *, _QWORD, _QWORD, __int64))v189)(
                   v187,
                   *v3,
                   v355,
                   (unsigned int)v360,
                   v361);
        }
        if ( v191 )
        {
          v192 = *(_QWORD *)(v58 + 40);
          v193 = *(__int128 **)(v186 + 40);
          *(_QWORD *)&v344 = v55;
          v337 = (__int128)_mm_loadu_si128((const __m128i *)v192);
          v351 = (__int128)_mm_loadu_si128((const __m128i *)(v192 + 40));
          v328 = *v193;
          *((_QWORD *)&v344 + 1) = v354.m128i_u32[0] | v346 & 0xFFFFFFFF00000000LL;
          v194 = sub_33FAF80((_DWORD)v367, 233, (unsigned int)&v362, v360, v361, v25, *(__int128 *)((char *)v193 + 40));
          v196 = v195;
          v198 = sub_33FAF80((_DWORD)v367, 233, (unsigned int)&v362, v360, v361, v197, v328);
          *((_QWORD *)&v296 + 1) = v196;
          *(_QWORD *)&v296 = v194;
          *((_QWORD *)&v291 + 1) = v199;
          *(_QWORD *)&v291 = v198;
          v200 = sub_340F900((_DWORD)v367, v355, (unsigned int)&v362, v360, v361, v198, v291, v296, v344);
          *((_QWORD *)&v305 + 1) = v201;
          *(_QWORD *)&v305 = v200;
          v136 = sub_340F900((_DWORD)v367, v355, (unsigned int)&v362, v360, v361, v200, v337, v351, v305);
          goto LABEL_183;
        }
      }
    }
  }
LABEL_172:
  v370.m128i_i64[0] = (__int64)&v367;
  v370.m128i_i64[1] = (__int64)&v355;
  v371 = &v362;
  v372 = &v360;
  if ( *(_DWORD *)(v58 + 24) == 233 )
  {
    v153 = *(unsigned int **)(v58 + 40);
    v154 = *(_QWORD *)v153;
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v153 + 24LL) - 150) <= 1 )
    {
      v155 = *(_QWORD *)(*(_QWORD *)(v154 + 40) + 80LL);
      if ( *(_DWORD *)(v155 + 24) == 98 && (v326 || (*(_BYTE *)(v155 + 29) & 2) != 0) )
      {
        v156 = v3[1];
        v157 = *(_QWORD *)(v154 + 48) + 16LL * v153[2];
        v158 = *(_WORD *)v157;
        v159 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v156 + 1576);
        if ( v159 == sub_2FE39B0 )
        {
          v160 = *(__int64 (**)())(*v156 + 1560);
          if ( v160 == sub_2D566A0 )
            goto LABEL_173;
          v336 = *(_QWORD *)(*(_QWORD *)(v154 + 40) + 80LL);
          v161 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _QWORD, __int64))v160)(
                   v156,
                   v360,
                   v361,
                   v158,
                   *(_QWORD *)(v157 + 8),
                   v25);
          v162 = v336;
        }
        else
        {
          v338 = *(_QWORD *)(*(_QWORD *)(v154 + 40) + 80LL);
          v161 = v159(
                   (__int64)v156,
                   (__int64)*v3,
                   v355,
                   (unsigned int)v360,
                   v361,
                   (__int64)*v3,
                   v158,
                   *(_QWORD *)(v157 + 8));
          v162 = v338;
          v25 = v310;
        }
        if ( v161 )
        {
          v163 = *(__int64 **)(v154 + 40);
          v148 = *(__int128 **)(v162 + 40);
          v149 = v163[5];
          v150 = v163[6];
          v151 = *v163;
          v152 = v163[1];
          v304 = __PAIR128__(v354.m128i_u32[0] | v346 & 0xFFFFFFFF00000000LL, v55);
          goto LABEL_196;
        }
      }
    }
  }
LABEL_173:
  if ( (unsigned int)(*(_DWORD *)(v55 + 24) - 150) <= 1 )
  {
    v114 = *(_QWORD *)(*(_QWORD *)(v55 + 40) + 80LL);
    if ( *(_DWORD *)(v114 + 24) != 233 )
      goto LABEL_122;
    v115 = *(unsigned int **)(v114 + 40);
    v116 = *(_QWORD *)v115;
    if ( *(_DWORD *)(*(_QWORD *)v115 + 24LL) != 98 || !v326 && (*(_BYTE *)(v116 + 29) & 2) == 0 )
      goto LABEL_122;
    v117 = v3[1];
    v118 = *(_QWORD *)(v116 + 48) + 16LL * v115[2];
    v119 = *(_WORD *)v118;
    v120 = *(_QWORD *)(v118 + 8);
    v121 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v117 + 1576);
    if ( v121 == sub_2FE39B0 )
    {
      v122 = *(__int64 (**)())(*v117 + 1560);
      if ( v122 == sub_2D566A0 )
        goto LABEL_122;
      v123 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, __int64, __int64))v122)(
               v117,
               v360,
               v361,
               v119,
               v120,
               v25);
    }
    else
    {
      v123 = v121((__int64)v117, (__int64)*v3, v355, (unsigned int)v360, v361, v25, v119, v120);
    }
    if ( v123 )
    {
      v124 = *(_QWORD *)(v55 + 40);
      v125 = *(__int64 **)(v116 + 40);
      *(_QWORD *)&v348 = v58;
      v126 = *v125;
      v127 = v125[1];
      v302 = *(_OWORD *)(v125 + 5);
      v343 = (__int128)_mm_loadu_si128((const __m128i *)v124);
      v354 = _mm_loadu_si128((const __m128i *)(v124 + 40));
      *(_QWORD *)&v335 = v126;
      *((_QWORD *)&v335 + 1) = v127;
      *((_QWORD *)&v348 + 1) = v57 | v54.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v128 = sub_33FAF80((_DWORD)v367, 233, (unsigned int)&v362, v360, v361, v25, v302);
      v130 = v129;
      v132 = sub_33FAF80((_DWORD)v367, 233, (unsigned int)&v362, v360, v361, v131, v335);
      *((_QWORD *)&v294 + 1) = v130;
      *(_QWORD *)&v294 = v128;
      *((_QWORD *)&v289 + 1) = v133;
      *(_QWORD *)&v289 = v132;
      v134 = sub_340F900((_DWORD)v367, v355, (unsigned int)&v362, v360, v361, v132, v289, v294, v348);
      *((_QWORD *)&v303 + 1) = v135;
      *(_QWORD *)&v303 = v134;
      v136 = sub_340F900((_DWORD)v367, v355, (unsigned int)&v362, v360, v361, v134, v343, *(_OWORD *)&v354, v303);
LABEL_183:
      v91 = v136;
      goto LABEL_184;
    }
  }
  if ( *(_DWORD *)(v55 + 24) != 233 )
    goto LABEL_122;
  v137 = *(unsigned int **)(v55 + 40);
  v138 = *(_QWORD *)v137;
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v137 + 24LL) - 150) > 1 )
    goto LABEL_122;
  v139 = *(_QWORD *)(*(_QWORD *)(v138 + 40) + 80LL);
  if ( *(_DWORD *)(v139 + 24) != 98 || !v326 && (*(_BYTE *)(v139 + 29) & 2) == 0 )
    goto LABEL_122;
  v140 = v3[1];
  v141 = *(_QWORD *)(v138 + 48) + 16LL * v137[2];
  v142 = *(_WORD *)v141;
  v143 = *(_QWORD *)(v141 + 8);
  v144 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v140 + 1576);
  if ( v144 == sub_2FE39B0 )
  {
    v145 = *(__int64 (**)())(*v140 + 1560);
    if ( v145 != sub_2D566A0 )
    {
      v146 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, __int64, __int64))v145)(
               v140,
               v360,
               v361,
               v142,
               v143,
               v25);
      goto LABEL_194;
    }
LABEL_122:
    v90 = v362;
    if ( !v362 )
    {
LABEL_139:
      v19 = 0;
      goto LABEL_5;
    }
    v91 = 0;
    goto LABEL_124;
  }
  v146 = v144((__int64)v140, (__int64)*v3, v355, (unsigned int)v360, v361, v25, v142, v143);
LABEL_194:
  if ( !v146 )
    goto LABEL_122;
  v147 = *(__int64 **)(v138 + 40);
  v148 = *(__int128 **)(v139 + 40);
  v149 = v147[5];
  v150 = v147[6];
  v151 = *v147;
  v152 = v147[1];
  v304 = __PAIR128__(v57 | v54.m128i_i64[1] & 0xFFFFFFFF00000000LL, v58);
LABEL_196:
  v91 = sub_325D670((__int64)&v370, v151, v152, v149, v150, v149, *v148, *(__int128 *)((char *)v148 + 40), v304);
LABEL_184:
  v90 = v362;
  if ( v362 )
  {
LABEL_124:
    v354.m128i_i64[0] = v91;
    sub_B91220((__int64)&v362, v90);
    v91 = v354.m128i_i64[0];
  }
  if ( !v91 )
    goto LABEL_139;
LABEL_126:
  if ( *(_DWORD *)(v91 + 24) )
  {
    v354.m128i_i64[0] = v91;
    sub_32B3E80((__int64)v3, v91, 1, 0, v92, v93);
    v91 = v354.m128i_i64[0];
  }
  v19 = v91;
LABEL_5:
  v20 = v358;
  v364[128] = v366;
  if ( v20 )
    sub_B91220((__int64)&v358, v20);
  return v19;
}
