// Function: sub_170F8E0
// Address: 0x170f8e0
//
__int64 __fastcall sub_170F8E0(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128i a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // rax
  _QWORD *v17; // rbx
  unsigned __int64 v18; // r12
  __int64 **v19; // rsi
  int v20; // edx
  _QWORD *v21; // rax
  __int64 **v22; // rax
  __int64 v23; // rdi
  __m128i v24; // xmm2
  __m128 v25; // xmm0
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rbx
  _QWORD *v29; // rax
  double v30; // xmm4_8
  double v31; // xmm5_8
  __int64 *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r12
  __int64 v36; // r9
  __int64 v37; // rax
  unsigned __int64 v38; // r14
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rbx
  __int64 *v41; // rax
  __int64 **v42; // r12
  __int64 v43; // r13
  __int64 v44; // r8
  unsigned __int64 v45; // rax
  bool v46; // al
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  _QWORD *v50; // r12
  char v51; // al
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rcx
  unsigned __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 *v62; // r11
  __int64 v63; // rdi
  __int64 *v64; // r12
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 *v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rsi
  __int64 v71; // rdx
  unsigned __int64 v72; // rax
  unsigned __int8 v73; // al
  _BYTE *i; // r12
  __int64 v75; // rbx
  unsigned int v76; // r13d
  __int64 v77; // r14
  __int64 v78; // rax
  bool v79; // al
  __int64 v80; // rax
  __int64 v81; // r13
  __int64 v82; // rsi
  unsigned __int8 *v83; // rsi
  __int64 v84; // rdx
  __int64 *v85; // rax
  __int64 v86; // rsi
  unsigned __int8 v87; // al
  __int64 v88; // r14
  __int64 *v89; // r14
  __int64 v90; // r12
  __int64 v91; // rcx
  __int64 v92; // rdi
  __int64 v93; // rbx
  unsigned __int64 v94; // rdi
  __int64 v95; // rax
  int v96; // ebx
  char v97; // dl
  char v98; // al
  __int64 v99; // rax
  unsigned int v100; // r12d
  __int64 v101; // r13
  __int64 v102; // rdi
  __int64 v103; // r12
  int v104; // r9d
  __int64 v105; // rax
  __int64 v106; // r14
  __int64 v107; // rbx
  _BYTE *v108; // r10
  __int64 *v109; // r13
  __int64 v110; // rdx
  unsigned __int64 v111; // r8
  __m128i *v112; // rax
  __int32 v113; // ecx
  bool v114; // al
  __int64 *v115; // r15
  unsigned __int8 *v116; // r12
  _QWORD *v117; // rax
  __int64 v118; // rdx
  unsigned __int64 v119; // rsi
  __int64 v120; // rdx
  __int64 v121; // rdx
  unsigned __int64 v122; // rsi
  __int64 v123; // rsi
  __int64 v124; // rdx
  unsigned __int64 v125; // rcx
  __int64 v126; // rdx
  __int64 *v127; // rax
  __int64 v128; // r12
  int v129; // ebx
  __int64 v130; // rsi
  unsigned __int64 v131; // rax
  __int64 v132; // rcx
  unsigned __int64 v133; // rbx
  int v134; // eax
  __int64 v135; // r12
  char v136; // al
  unsigned __int64 v137; // rax
  __int64 *v138; // rax
  __int64 v139; // r14
  __int64 *v140; // r9
  __int64 v141; // r8
  __int64 v142; // r15
  __int64 *v143; // r14
  int v144; // ebx
  __int64 v145; // r12
  __int64 v146; // rax
  __int64 v147; // rdx
  __int64 v148; // rdi
  unsigned __int64 v149; // r13
  int v150; // eax
  __int64 v151; // rax
  _QWORD *v152; // rsi
  _QWORD *v153; // rax
  __int64 v154; // r13
  unsigned __int64 v155; // rbx
  unsigned __int64 v156; // rax
  __int64 v157; // r14
  unsigned int v158; // eax
  __int64 v159; // rbx
  __int64 **v160; // rax
  __int64 v161; // rdx
  __int64 v162; // rcx
  __int64 *v163; // rax
  bool v164; // zf
  __int64 v165; // r14
  __m128i v166; // rax
  void *v167; // rdi
  __int64 v168; // rax
  int v169; // ecx
  __int64 v170; // rsi
  __int64 v171; // rdi
  int v172; // ecx
  unsigned int v173; // edx
  __int64 *v174; // rax
  __int64 v175; // r9
  __int64 v176; // r14
  __int64 v177; // r13
  __int64 v178; // r12
  __int64 v179; // rax
  __int64 v180; // rbx
  __m128i v181; // rax
  __int64 *v182; // r13
  bool v183; // al
  __int64 v184; // rax
  __int64 v185; // rcx
  _QWORD *v186; // rbx
  _QWORD *v187; // r13
  __int64 v188; // rdx
  __int64 v189; // rdi
  __int64 v190; // r14
  _BYTE *v191; // r12
  __int64 v192; // r13
  __int64 v193; // rbx
  char v194; // al
  int v195; // ebx
  __int64 v196; // rax
  __int64 v197; // rbx
  double v198; // xmm4_8
  double v199; // xmm5_8
  __int64 *v200; // rax
  __int64 *v201; // r13
  __int64 v202; // r12
  __int64 v203; // rdx
  __int64 *v204; // rdx
  __int64 v205; // r14
  __int64 v206; // rax
  unsigned __int8 *v207; // rax
  __int64 v208; // rax
  unsigned __int64 v209; // rax
  __int64 v210; // rbx
  __int64 v211; // rax
  unsigned __int8 *v212; // r12
  __int64 v213; // rbx
  __int64 v214; // rax
  int v215; // r8d
  int v216; // r9d
  __int64 v217; // rax
  int v218; // eax
  unsigned __int64 v219; // rbx
  __int64 **v220; // rax
  __int64 v221; // rdx
  __int64 v222; // rcx
  __int64 *v223; // rax
  int v224; // edx
  __int64 v225; // r13
  __m128i v226; // rax
  __int64 v227; // r12
  __int64 v228; // rdx
  unsigned __int64 v229; // rbx
  unsigned __int64 v230; // rax
  __int64 v231; // r14
  __int64 *v232; // rbx
  unsigned int v233; // eax
  __int64 v234; // r14
  __int64 v235; // r13
  __m128i v236; // rax
  __int64 v237; // rdi
  unsigned __int8 *v238; // r12
  __int64 v239; // rbx
  double v240; // xmm4_8
  double v241; // xmm5_8
  int v242; // eax
  __int64 *v243; // rax
  __int64 *v244; // r14
  __int64 v245; // r13
  __int64 v246; // rdx
  __int64 v247; // r12
  __int64 v248; // rax
  __int64 v249; // rdx
  _QWORD *v250; // r12
  bool v251; // al
  _QWORD *v252; // r12
  _QWORD *v253; // rax
  int v254; // r8d
  unsigned int v255; // eax
  _QWORD *v256; // rax
  signed __int64 v257; // rdx
  __int64 v258; // rdi
  unsigned __int8 *v259; // r12
  double v260; // xmm4_8
  double v261; // xmm5_8
  __int64 v262; // rbx
  int v263; // r8d
  unsigned int v264; // eax
  _QWORD *v265; // rax
  __int64 *v266; // r13
  unsigned __int8 v267; // al
  unsigned int v268; // ebx
  bool v269; // al
  __int64 v270; // rax
  __int64 v271; // rcx
  int v272; // edx
  __int64 v273; // r13
  __m128i v274; // rax
  __int64 v275; // rdi
  __int64 v276; // rdi
  int v277; // eax
  __int64 v278; // r13
  unsigned int v279; // ebx
  bool v280; // al
  __int64 v281; // rdx
  int v282; // eax
  __int64 ***v283; // r13
  int v284; // eax
  unsigned __int8 *v285; // rax
  __int64 v286; // r12
  unsigned __int8 *v287; // rbx
  __int64 v288; // rax
  __int64 v289; // rsi
  __int64 v290; // r13
  __int64 v291; // rax
  __int64 v292; // rdx
  __int64 v293; // rax
  __int64 *v294; // rbx
  __int64 v295; // rax
  __int64 v296; // rcx
  __m128i v297; // rax
  _QWORD *v298; // rax
  __int64 v299; // rax
  unsigned int v300; // ebx
  bool v301; // dl
  unsigned int v302; // r14d
  __int64 v303; // rax
  char v304; // dl
  unsigned int v305; // ebx
  int v306; // edx
  __int64 v307; // rdx
  int v308; // eax
  int v309; // r8d
  __m128i v310; // rax
  __int64 v311; // rax
  unsigned int v312; // ebx
  unsigned int v313; // r14d
  __int64 v314; // rax
  char v315; // dl
  unsigned int v316; // ebx
  int v318; // r12d
  __int64 v319; // rax
  __int64 v320; // rbx
  __int64 *v321; // rdi
  __int64 v322; // rsi
  __int64 v323; // r13
  __int64 v324; // r12
  __int64 *v325; // rcx
  int v326; // eax
  __int64 *v327; // r15
  __int64 v328; // rdx
  __int64 v329; // rdi
  __int64 v330; // rax
  __int64 v331; // rcx
  __int64 v332; // r8
  __int64 v333; // r9
  unsigned __int64 v334; // rbx
  __int64 v335; // rax
  __int64 v336; // r12
  __int64 v337; // rbx
  __int64 *v338; // rax
  __m128i v339; // rax
  __int64 v340; // r14
  __int64 *v341; // rax
  __int64 v342; // [rsp+0h] [rbp-1E0h]
  __int64 v343; // [rsp+8h] [rbp-1D8h]
  __int64 v344; // [rsp+18h] [rbp-1C8h]
  __int64 v345; // [rsp+20h] [rbp-1C0h]
  char v346; // [rsp+28h] [rbp-1B8h]
  __int64 *v347; // [rsp+28h] [rbp-1B8h]
  __int64 *v348; // [rsp+28h] [rbp-1B8h]
  __int64 v349; // [rsp+30h] [rbp-1B0h]
  __int64 *v350; // [rsp+40h] [rbp-1A0h]
  __int64 v351; // [rsp+40h] [rbp-1A0h]
  __int64 v352; // [rsp+48h] [rbp-198h]
  __int64 v353; // [rsp+48h] [rbp-198h]
  char v354; // [rsp+50h] [rbp-190h]
  _BYTE *v355; // [rsp+50h] [rbp-190h]
  _QWORD *v356; // [rsp+50h] [rbp-190h]
  __int64 *v357; // [rsp+50h] [rbp-190h]
  _QWORD *v358; // [rsp+60h] [rbp-180h]
  __int64 v359; // [rsp+60h] [rbp-180h]
  _BYTE *v360; // [rsp+60h] [rbp-180h]
  __int64 v361; // [rsp+60h] [rbp-180h]
  _BYTE *v363; // [rsp+68h] [rbp-178h]
  __int64 v364; // [rsp+68h] [rbp-178h]
  int v365; // [rsp+68h] [rbp-178h]
  __int64 *v366; // [rsp+70h] [rbp-170h]
  int v367; // [rsp+70h] [rbp-170h]
  __int64 v368; // [rsp+70h] [rbp-170h]
  __int64 *v369; // [rsp+70h] [rbp-170h]
  _BYTE *v370; // [rsp+70h] [rbp-170h]
  int v371; // [rsp+70h] [rbp-170h]
  int v372; // [rsp+70h] [rbp-170h]
  __int64 *v373; // [rsp+70h] [rbp-170h]
  __int64 v374; // [rsp+80h] [rbp-160h] BYREF
  __int64 ***v375; // [rsp+88h] [rbp-158h] BYREF
  __int64 v376; // [rsp+90h] [rbp-150h] BYREF
  __int64 v377; // [rsp+98h] [rbp-148h] BYREF
  __int64 v378; // [rsp+A0h] [rbp-140h] BYREF
  unsigned int v379; // [rsp+A8h] [rbp-138h]
  __int16 v380; // [rsp+B0h] [rbp-130h]
  unsigned __int64 v381; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v382; // [rsp+C8h] [rbp-118h]
  __int16 v383; // [rsp+D0h] [rbp-110h]
  __m128i v384; // [rsp+E0h] [rbp-100h] BYREF
  __m128 v385; // [rsp+F0h] [rbp-F0h]
  __int64 v386; // [rsp+100h] [rbp-E0h]
  _BYTE *v387; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v388; // [rsp+118h] [rbp-C8h]
  _BYTE v389[64]; // [rsp+120h] [rbp-C0h] BYREF
  __m128 v390; // [rsp+160h] [rbp-80h] BYREF
  __m128i v391; // [rsp+170h] [rbp-70h] BYREF
  __int64 v392; // [rsp+180h] [rbp-60h]

  v14 = a2;
  v15 = *(_DWORD *)(a2 + 20);
  v387 = v389;
  v388 = 0x800000000LL;
  v16 = 24LL * (v15 & 0xFFFFFFF);
  v17 = (_QWORD *)(a2 - v16);
  v18 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 3);
  if ( (unsigned __int64)v16 > 0xC0 )
  {
    sub_16CD150((__int64)&v387, v389, 0xAAAAAAAAAAAAAAABLL * (v16 >> 3), 8, a13, a14);
    v19 = (__int64 **)v387;
    v20 = v388;
    v21 = &v387[8 * (unsigned int)v388];
  }
  else
  {
    v19 = (__int64 **)v389;
    v20 = 0;
    v21 = v389;
  }
  if ( (_QWORD *)v14 != v17 )
  {
    do
    {
      if ( v21 )
        *v21 = *v17;
      v17 += 3;
      ++v21;
    }
    while ( (_QWORD *)v14 != v17 );
    v19 = (__int64 **)v387;
    v20 = v388;
  }
  v22 = *(__int64 ***)v14;
  v23 = *(_QWORD *)(v14 + 56);
  v392 = v14;
  LODWORD(v388) = v18 + v20;
  v343 = (__int64)v22;
  v342 = v23;
  v24 = _mm_loadu_si128((const __m128i *)(a1 + 2688));
  v25 = (__m128)_mm_loadu_si128((const __m128i *)(a1 + 2672));
  v349 = v14;
  v390 = v25;
  v391 = v24;
  v26 = sub_13E3340(v23, v19, (unsigned int)(v18 + v20), (__int64 *)&v390);
  if ( v26 )
  {
    v27 = *(_QWORD *)(v14 + 8);
    if ( v27 )
    {
      v28 = *(_QWORD *)a1;
      do
      {
        v29 = sub_1648700(v27);
        sub_170B990(v28, (__int64)v29);
        v27 = *(_QWORD *)(v27 + 8);
      }
      while ( v27 );
      if ( v14 == v26 )
        v26 = sub_1599EF0(*(__int64 ***)v14);
      sub_164D160(v14, v26, v25, a4, *(double *)v24.m128i_i64, a6, v30, v31, *(double *)a9.m128i_i64, a10);
      goto LABEL_15;
    }
    goto LABEL_22;
  }
  v33 = *(__int64 **)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
  v34 = *v33;
  v345 = (__int64)v33;
  if ( *(_BYTE *)(*v33 + 8) == 16 )
    v34 = **(_QWORD **)(v34 + 16);
  v350 = (__int64 *)sub_15A9730(*(_QWORD *)(a1 + 2664), v34);
  if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
    v35 = *(_QWORD *)(v14 - 8);
  else
    v35 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
  v366 = (__int64 *)(v35 + 24);
  v346 = 0;
  v36 = sub_16348C0(v14) | 4;
  v37 = v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
  if ( v14 != v37 )
  {
    v352 = v14;
    v38 = v37 + 8;
    while ( 1 )
    {
      v358 = (_QWORD *)(v38 - 8);
      v39 = v36 & 0xFFFFFFFFFFFFFFF8LL;
      v40 = v36 & 0xFFFFFFFFFFFFFFF8LL;
      v354 = (v36 >> 2) & 1;
      if ( !v354 )
      {
        v50 = (_QWORD *)(v38 + 16);
LABEL_67:
        v40 = sub_1643D30(v39, *v366);
        goto LABEL_51;
      }
      v41 = *(__int64 **)(v38 - 8);
      v42 = (__int64 **)v350;
      v43 = *v41;
      if ( *(_BYTE *)(*v41 + 8) == 16 )
        v42 = (__int64 **)sub_16463B0(v350, *(_DWORD *)(v43 + 32));
      v44 = v39;
      if ( !v39 )
        v44 = sub_1643D30(0, *v366);
      v45 = *(unsigned __int8 *)(v44 + 8);
      if ( (unsigned __int8)v45 <= 0xFu && (v52 = 35454, _bittest64(&v52, v45))
        || ((unsigned int)(v45 - 13) <= 1 || (_DWORD)v45 == 16)
        && (v344 = v44, v46 = sub_16435F0(v44, 0), v44 = v344, v46) )
      {
        v53 = v44;
        if ( !sub_12BE0A0(*(_QWORD *)(a1 + 2664), v44) )
        {
          v56 = *(_QWORD *)(v38 - 8);
          if ( *(_BYTE *)(v56 + 16) > 0x10u || !sub_1593BB0(v56, v53, v54, v55) )
          {
            v57 = sub_15A06D0(v42, v53, v54, v55);
            if ( *(_QWORD *)(v38 - 8) )
            {
              v58 = *(_QWORD *)v38;
              v59 = *(_QWORD *)(v38 + 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v59 = *(_QWORD *)v38;
              if ( v58 )
                *(_QWORD *)(v58 + 16) = *(_QWORD *)(v58 + 16) & 3LL | v59;
            }
            *(_QWORD *)(v38 - 8) = v57;
            v346 = v354;
            if ( v57 )
            {
              v60 = *(_QWORD *)(v57 + 8);
              *(_QWORD *)v38 = v60;
              if ( v60 )
                *(_QWORD *)(v60 + 16) = v38 | *(_QWORD *)(v60 + 16) & 3LL;
              *(_QWORD *)(v38 + 8) = (v57 + 8) | *(_QWORD *)(v38 + 8) & 3LL;
              *(_QWORD *)(v57 + 8) = v358;
              v346 = v354;
            }
          }
        }
      }
      if ( v42 != (__int64 **)v43 )
        break;
LABEL_50:
      v50 = (_QWORD *)(v38 + 16);
      if ( !v39 )
        goto LABEL_67;
LABEL_51:
      v51 = *(_BYTE *)(v40 + 8);
      if ( ((v51 - 14) & 0xFD) != 0 )
      {
        v36 = 0;
        if ( v51 == 13 )
          v36 = v40;
      }
      else
      {
        v36 = *(_QWORD *)(v40 + 24) | 4LL;
      }
      v366 += 3;
      v38 += 24LL;
      if ( (_QWORD *)v352 == v50 )
      {
        v14 = v352;
        if ( v346 )
          goto LABEL_15;
        goto LABEL_78;
      }
    }
    v380 = 257;
    v47 = *(_QWORD *)(v38 - 8);
    if ( v42 != *(__int64 ***)v47 )
    {
      v347 = *(__int64 **)(a1 + 8);
      if ( *(_BYTE *)(v47 + 16) > 0x10u )
      {
        v385.m128_i16[0] = 257;
        v61 = sub_15FE0A0((_QWORD *)v47, (__int64)v42, 1, (__int64)&v384, 0);
        v62 = v347;
        v47 = v61;
        v63 = v347[1];
        if ( v63 )
        {
          v64 = (__int64 *)v347[2];
          sub_157E9D0(v63 + 40, v61);
          v65 = *(_QWORD *)(v47 + 24);
          v62 = v347;
          v66 = *v64;
          *(_QWORD *)(v47 + 32) = v64;
          v66 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v47 + 24) = v66 | v65 & 7;
          *(_QWORD *)(v66 + 8) = v47 + 24;
          *v64 = *v64 & 7 | (v47 + 24);
        }
        v67 = &v378;
        v68 = v47;
        v348 = v62;
        sub_164B780(v47, &v378);
        v376 = v47;
        if ( !v348[10] )
          goto LABEL_435;
        ((void (__fastcall *)(__int64 *, __int64 *))v348[11])(v348 + 8, &v376);
        v70 = *v348;
        if ( *v348 )
        {
          v390.m128_u64[0] = *v348;
          sub_1623A60((__int64)&v390, v70, 2);
          v82 = *(_QWORD *)(v47 + 48);
          if ( v82 )
            sub_161E7C0(v47 + 48, v82);
          v83 = (unsigned __int8 *)v390.m128_u64[0];
          *(_QWORD *)(v47 + 48) = v390.m128_u64[0];
          if ( v83 )
            sub_1623210((__int64)&v390, v83, v47 + 48);
        }
      }
      else
      {
        v47 = sub_15A4750((__int64 ***)v47, v42, 1);
        v48 = sub_14DBA30(v47, v347[12], 0);
        if ( v48 )
        {
          if ( !*(_QWORD *)(v38 - 8) )
          {
            *(_QWORD *)(v38 - 8) = v48;
            v47 = v48;
            goto LABEL_46;
          }
          v47 = v48;
          goto LABEL_73;
        }
      }
      if ( !*(_QWORD *)(v38 - 8) )
        goto LABEL_75;
    }
LABEL_73:
    v71 = *(_QWORD *)v38;
    v72 = *(_QWORD *)(v38 + 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v72 = *(_QWORD *)v38;
    if ( v71 )
      *(_QWORD *)(v71 + 16) = *(_QWORD *)(v71 + 16) & 3LL | v72;
LABEL_75:
    *(_QWORD *)(v38 - 8) = v47;
    if ( !v47 )
    {
LABEL_49:
      v346 = v354;
      goto LABEL_50;
    }
LABEL_46:
    v49 = *(_QWORD *)(v47 + 8);
    *(_QWORD *)v38 = v49;
    if ( v49 )
      *(_QWORD *)(v49 + 16) = v38 | *(_QWORD *)(v49 + 16) & 3LL;
    *(_QWORD *)(v38 + 8) = (v47 + 8) | *(_QWORD *)(v38 + 8) & 3LL;
    *(_QWORD *)(v47 + 8) = v358;
    goto LABEL_49;
  }
LABEL_78:
  v73 = *(_BYTE *)(v345 + 16);
  if ( v73 != 77 )
    goto LABEL_79;
  if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
    v138 = *(__int64 **)(v345 - 8);
  else
    v138 = (__int64 *)(v345 - 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF));
  v139 = *v138;
  if ( *(_BYTE *)(*v138 + 16) != 56 || v14 == v139 )
    goto LABEL_22;
  v140 = &v138[3 * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF)];
  if ( v138 + 3 == v140 )
    goto LABEL_463;
  v141 = v14;
  v142 = *v138;
  v143 = v138 + 3;
  v144 = -1;
  do
  {
    v145 = *v143;
    if ( *(_BYTE *)(*v143 + 16) != 56 )
      goto LABEL_22;
    v146 = *(_DWORD *)(v142 + 20) & 0xFFFFFFF;
    v147 = *(_DWORD *)(v145 + 20) & 0xFFFFFFF;
    if ( v141 == v145 || (_DWORD)v147 != (_DWORD)v146 )
      goto LABEL_22;
    if ( (_DWORD)v146 )
    {
      v148 = 0;
      v149 = 0;
      v368 = (unsigned int)(v146 - 1);
      while ( 1 )
      {
        v152 = *(_QWORD **)(v142 + 24 * (v149 - v146));
        v153 = *(_QWORD **)(v145 + 24 * (v149 - v147));
        if ( !v153 )
          BUG();
        if ( *v153 != *v152 )
          goto LABEL_22;
        if ( v153 != v152 )
        {
          if ( v144 != -1 || v149 > 1 && *(_BYTE *)(v148 + 8) == 13 )
            goto LABEL_22;
          v144 = v149;
        }
        if ( v149 )
        {
          if ( (_DWORD)v149 == 1 )
          {
            v148 = *(_QWORD *)(v142 + 56);
          }
          else
          {
            v150 = *(unsigned __int8 *)(v148 + 8);
            if ( (unsigned int)(v150 - 13) <= 1 || v150 == 16 )
            {
              v359 = v141;
              v357 = v140;
              v151 = sub_1643D30(v148, (__int64)v152);
              v140 = v357;
              v141 = v359;
              v148 = v151;
            }
            else
            {
              v148 = 0;
            }
          }
        }
        if ( v149 == v368 )
          break;
        ++v149;
        v147 = *(_DWORD *)(v145 + 20) & 0xFFFFFFF;
        v146 = *(_DWORD *)(v142 + 20) & 0xFFFFFFF;
      }
    }
    v143 += 3;
  }
  while ( v143 != v140 );
  v139 = v142;
  v318 = v144;
  v14 = v141;
  if ( v144 == -1 )
  {
LABEL_463:
    v335 = sub_15F4880(v139);
    v336 = *(_QWORD *)(v14 + 40);
    v337 = v335;
    v338 = (__int64 *)sub_157EE30(v336);
    sub_14F2510(v336 + 40, v338, v337);
    goto LABEL_464;
  }
  v319 = *(_QWORD *)(v345 + 8);
  if ( !v319 || *(_QWORD *)(v319 + 8) )
    goto LABEL_22;
  v320 = sub_15F4880(v139);
  v321 = *(__int64 **)(a1 + 8);
  v390.m128_u64[0] = (unsigned __int64)v321;
  v390.m128_u64[1] = v321[1];
  v391.m128i_i64[0] = v321[2];
  v322 = *v321;
  v391.m128i_i64[1] = v322;
  if ( v322 )
  {
    sub_1623A60((__int64)&v391.m128i_i64[1], v322, 2);
    v321 = *(__int64 **)(a1 + 8);
  }
  v323 = v318;
  sub_17050D0(v321, v345);
  v385.m128_i16[0] = 257;
  v324 = sub_1709340(
           *(_QWORD *)(a1 + 8),
           **(_QWORD **)(v139 + 24 * (v318 - (unsigned __int64)(*(_DWORD *)(v139 + 20) & 0xFFFFFFF))),
           *(_DWORD *)(v345 + 20) & 0xFFFFFFF,
           v384.m128i_i64);
  sub_1705150((__int64)&v390);
  if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
  {
    v325 = *(__int64 **)(v345 - 8);
    v326 = *(_DWORD *)(v345 + 20);
  }
  else
  {
    v326 = *(_DWORD *)(v345 + 20);
    v325 = (__int64 *)(v345 - 24LL * (v326 & 0xFFFFFFF));
  }
  v353 = v14;
  v327 = v325;
  v361 = v320;
  v373 = &v325[3 * (v326 & 0xFFFFFFF)];
  while ( v327 != v373 )
  {
    if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
      v328 = *(_QWORD *)(v345 - 8);
    else
      v328 = v345 - 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF);
    v334 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v327 - v328) >> 3);
    v329 = *v327;
    v351 = v328;
    v327 += 3;
    v330 = sub_13CF970(v329);
    sub_1704F80(
      v324,
      *(_QWORD *)(v330 + 24 * v323),
      *(_QWORD *)(v351 + 24LL * *(unsigned int *)(v345 + 56) + 8LL * (unsigned int)v334 + 8),
      v331,
      v332,
      v333);
  }
  v337 = v361;
  v14 = v353;
  sub_1593B40((_QWORD *)(v361 + 24 * (v323 - (*(_DWORD *)(v361 + 20) & 0xFFFFFFF))), v324);
  v340 = *(_QWORD *)(v353 + 40);
  v341 = (__int64 *)sub_157EE30(v340);
  sub_14F2510(v340 + 40, v341, v361);
  sub_1593B40((_QWORD *)(v337 + 24 * (v323 - (*(_DWORD *)(v337 + 20) & 0xFFFFFFF))), v324);
LABEL_464:
  sub_1593B40((_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)), v337);
  v345 = v337;
  v73 = *(_BYTE *)(v337 + 16);
LABEL_79:
  if ( v73 <= 0x17u )
  {
    if ( v73 != 5 || *(_WORD *)(v345 + 18) != 32 )
      goto LABEL_81;
  }
  else if ( v73 != 56 )
  {
    goto LABEL_81;
  }
  if ( !sub_1704A60((__int64 *)v14, (__int64 *)v345) )
    goto LABEL_22;
  v84 = *(_QWORD *)(a1 + 2720);
  if ( v84 )
  {
    if ( (*(_DWORD *)(v345 + 20) & 0xFFFFFFF) == 2 && (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) == 2 )
    {
      v168 = *(_QWORD *)(v345 + 8);
      if ( v168 )
      {
        if ( !*(_QWORD *)(v168 + 8) )
        {
          v169 = *(_DWORD *)(v84 + 24);
          if ( v169 )
          {
            v170 = *(_QWORD *)(v14 + 40);
            v171 = *(_QWORD *)(v84 + 8);
            v172 = v169 - 1;
            v173 = v172 & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
            v174 = (__int64 *)(v171 + 16LL * v173);
            v175 = *v174;
            if ( v170 == *v174 )
            {
LABEL_233:
              v176 = v174[1];
              if ( v176 )
              {
                v177 = *(_QWORD *)(v14 - 24);
                v178 = *(_QWORD *)(sub_13CF970(v345) + 24);
                if ( sub_13FC1A0(v176, v177) && !sub_13FC1A0(v176, v178) )
                {
                  if ( *(_BYTE *)(v343 + 8) != 16
                    || (v179 = sub_13CF970(v345), v370 = *(_BYTE **)v179, *(_BYTE *)(**(_QWORD **)v179 + 8LL) == 16) )
                  {
                    if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
                      v307 = *(_QWORD *)(v345 - 8);
                    else
                      v307 = v345 - 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF);
                    sub_1593B40((_QWORD *)(v307 + 24), v177);
                    sub_1593B40((_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF))), v178);
                  }
                  else
                  {
                    sub_17050D0(*(__int64 **)(a1 + 8), v345);
                    v180 = *(_QWORD *)(a1 + 8);
                    v181.m128i_i64[0] = (__int64)sub_1649960(v345);
                    v391.m128i_i16[0] = 261;
                    v384 = v181;
                    v390.m128_u64[0] = (unsigned __int64)&v384;
                    v182 = sub_170A020(v180, 0, v370, v177, (__int64 *)&v390);
                    sub_15FA2E0((__int64)v182, (*(_BYTE *)(v345 + 17) & 2) != 0);
                    v391.m128i_i16[0] = 257;
                    v384.m128i_i64[0] = v178;
                    v349 = (__int64)sub_1704C00(0, (__int64)v182, v384.m128i_i64, 1, (__int64)&v390, 0);
                    v183 = sub_15FA300(v14);
                    sub_15FA2E0(v349, v183);
                  }
                  goto LABEL_15;
                }
              }
            }
            else
            {
              v308 = 1;
              while ( v175 != -8 )
              {
                v309 = v308 + 1;
                v173 = v172 & (v308 + v173);
                v174 = (__int64 *)(v171 + 16LL * v173);
                v175 = *v174;
                if ( v170 == *v174 )
                  goto LABEL_233;
                v308 = v309;
              }
            }
          }
        }
      }
    }
  }
  if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
    v85 = *(__int64 **)(v345 - 8);
  else
    v85 = (__int64 *)(v345 - 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF));
  v86 = *v85;
  v87 = *(_BYTE *)(*v85 + 16);
  if ( v87 <= 0x17u )
  {
    if ( v87 != 5 || *(_WORD *)(v86 + 18) != 32 )
      goto LABEL_111;
LABEL_20:
    if ( (*(_DWORD *)(v86 + 20) & 0xFFFFFFF) != 2 || !sub_1704A60((__int64 *)v345, (__int64 *)v86) )
      goto LABEL_111;
    goto LABEL_22;
  }
  if ( v87 == 56 )
    goto LABEL_20;
LABEL_111:
  v390.m128_u64[0] = (unsigned __int64)&v391;
  v390.m128_u64[1] = 0x800000000LL;
  if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
    v88 = *(_QWORD *)(v345 - 8);
  else
    v88 = v345 - 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF);
  v89 = (__int64 *)(v88 + 24);
  v90 = v345;
  v92 = sub_16348C0(v345) | 4;
  if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
    v90 = *(_QWORD *)(v345 - 8) + 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF);
  if ( v89 == (__int64 *)v90 )
    goto LABEL_246;
  do
  {
    v93 = v92;
    v94 = v92 & 0xFFFFFFFFFFFFFFF8LL;
    v95 = v94;
    v96 = (v93 >> 2) & 1;
    if ( !v96 || !v94 )
    {
      v86 = *v89;
      v95 = sub_1643D30(v94, *v89);
    }
    v97 = *(_BYTE *)(v95 + 8);
    v91 = (unsigned __int8)(v97 - 14) & 0xFD;
    if ( ((v97 - 14) & 0xFD) != 0 )
    {
      if ( v97 != 13 )
        v95 = 0;
      v92 = v95;
    }
    else
    {
      v92 = *(_QWORD *)(v95 + 24) | 4LL;
    }
    v89 += 3;
  }
  while ( (__int64 *)v90 != v89 );
  if ( !(_BYTE)v96 )
  {
LABEL_246:
    v188 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
    v189 = *(_QWORD *)(v14 + 24 * (1 - v188));
    if ( *(_BYTE *)(v189 + 16) <= 0x10u && sub_1593BB0(v189, v86, v188, v91) )
    {
      v210 = *(_DWORD *)(v345 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v210 != 1 )
      {
        v211 = sub_13CF970(v345);
        sub_13D6230((__int64)&v390, (char *)(v211 + 24), (char *)(v211 + 24 * v210));
        sub_13D6230((__int64)&v390, (char *)(v14 + 24 * (2LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF))), (char *)v14);
      }
    }
  }
  else
  {
    v184 = *(_DWORD *)(v345 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
      v185 = *(_QWORD *)(v345 - 8);
    else
      v185 = v345 - 24 * v184;
    v186 = *(_QWORD **)(v185 + 24LL * (unsigned int)(v184 - 1));
    v187 = *(_QWORD **)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
    if ( *v187 != *v186 )
      goto LABEL_245;
    v386 = v14;
    a9 = _mm_loadu_si128((const __m128i *)(a1 + 2688));
    v384 = _mm_loadu_si128((const __m128i *)(a1 + 2672));
    v385 = (__m128)a9;
    v212 = sub_13DEB20((__int64)v187, (__int64)v186, 0, 0, &v384);
    if ( !v212 )
    {
      if ( !byte_4FA2280
        || (v227 = *(_QWORD *)(a1 + 8),
            v381 = (unsigned __int64)sub_1649960(v345),
            v382 = v228,
            v385.m128_i16[0] = 773,
            v384.m128i_i64[0] = (__int64)&v381,
            v384.m128i_i64[1] = (__int64)".sum",
            (v212 = sub_17094A0(
                      v227,
                      (__int64)v186,
                      (__int64)v187,
                      v384.m128i_i64,
                      0,
                      0,
                      *(double *)v25.m128_u64,
                      a4,
                      *(double *)v24.m128i_i64)) == 0) )
      {
LABEL_245:
        v349 = 0;
        goto LABEL_171;
      }
    }
    v213 = *(_DWORD *)(v345 + 20) & 0xFFFFFFF;
    v214 = sub_13CF970(v345);
    if ( (_DWORD)v213 == 2 )
    {
      sub_1593B40((_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)), *(_QWORD *)v214);
      sub_1593B40((_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF))), (__int64)v212);
      goto LABEL_171;
    }
    sub_13D6230((__int64)&v390, (char *)(v214 + 24), (char *)(v214 + 24 * v213 - 24));
    v217 = v390.m128_u32[2];
    if ( v390.m128_i32[2] >= (unsigned __int32)v390.m128_i32[3] )
    {
      sub_16CD150((__int64)&v390, &v391, 0, 8, v215, v216);
      v217 = v390.m128_u32[2];
    }
    *(_QWORD *)(v390.m128_u64[0] + 8 * v217) = v212;
    v218 = *(_DWORD *)(v14 + 20);
    ++v390.m128_i32[2];
    sub_13D6230((__int64)&v390, (char *)(v14 - 24LL * (v218 & 0xFFFFFFF) + 48), (char *)v14);
  }
  if ( v390.m128_i32[2] )
  {
    if ( sub_15FA300(v14) && (*(_BYTE *)(v345 + 17) & 2) != 0 )
    {
      v243 = (__int64 *)sub_1649960(v14);
      v244 = (__int64 *)v390.m128_u64[0];
      v381 = (unsigned __int64)v243;
      v245 = v390.m128_u32[2];
      v382 = v246;
      v385.m128_i16[0] = 261;
      v384.m128i_i64[0] = (__int64)&v381;
      v247 = *(_QWORD *)sub_13CF970(v345);
      v248 = sub_16348C0(v345);
      v349 = (__int64)sub_1704C00(v248, v247, v244, v245, (__int64)&v384, 0);
      sub_15FA2E0(v349, 1);
    }
    else
    {
      v200 = (__int64 *)sub_1649960(v14);
      v201 = (__int64 *)v390.m128_u64[0];
      v202 = v390.m128_u32[2];
      v381 = (unsigned __int64)v200;
      v384.m128i_i64[0] = (__int64)&v381;
      v382 = v203;
      v385.m128_i16[0] = 261;
      if ( (*(_BYTE *)(v345 + 23) & 0x40) != 0 )
        v204 = *(__int64 **)(v345 - 8);
      else
        v204 = (__int64 *)(v345 - 24LL * (*(_DWORD *)(v345 + 20) & 0xFFFFFFF));
      v205 = *v204;
      v206 = sub_16348C0(v345);
      v349 = (__int64)sub_1704C00(v206, v205, v201, v202, (__int64)&v384, 0);
    }
    goto LABEL_171;
  }
  if ( (__m128i *)v390.m128_u64[0] != &v391 )
    _libc_free(v390.m128_u64[0]);
LABEL_81:
  if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) != 2 )
    goto LABEL_82;
  v127 = *(__int64 **)(v14 - 48);
  v128 = *v127;
  if ( *(_BYTE *)(*v127 + 8) == 16 )
    v128 = **(_QWORD **)(v128 + 16);
  v129 = sub_16431D0(**(_QWORD **)(v14 - 24));
  if ( v129 != 8 * (unsigned int)sub_15A95A0(*(_QWORD *)(a1 + 2664), *(_DWORD *)(v128 + 8) >> 8) )
    goto LABEL_82;
  v130 = v342;
  v131 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), v342);
  v375 = 0;
  v133 = v131;
  if ( v131 == 1 )
  {
    v135 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
    v375 = (__int64 ***)v135;
    goto LABEL_181;
  }
  v134 = *(_DWORD *)(v14 + 20);
  v390.m128_u64[0] = (unsigned __int64)&v375;
  v390.m128_u64[1] = (unsigned __int64)&v374;
  v130 = *(_QWORD *)(v14 + 24 * (1LL - (v134 & 0xFFFFFFF)));
  if ( !(unsigned __int8)sub_170B110((__int64)&v390, v130) )
  {
    v242 = *(_DWORD *)(v14 + 20);
    v390.m128_u64[0] = (unsigned __int64)&v375;
    v390.m128_u64[1] = (unsigned __int64)&v374;
    v130 = *(_QWORD *)(v14 + 24 * (1LL - (v242 & 0xFFFFFFF)));
    if ( !(unsigned __int8)sub_170B1F0((__int64)&v390, v130) || v374 != v133 )
      goto LABEL_82;
LABEL_180:
    v135 = (__int64)v375;
LABEL_181:
    v136 = *(_BYTE *)(v135 + 16);
    if ( v136 == 37 )
    {
      v266 = *(__int64 **)(v135 - 48);
      v267 = *((_BYTE *)v266 + 16);
      if ( v267 == 13 )
      {
        v268 = *((_DWORD *)v266 + 8);
        if ( v268 <= 0x40 )
          v269 = v266[3] == 0;
        else
          v269 = v268 == (unsigned int)sub_16A57B0((__int64)(v266 + 3));
        if ( !v269 )
          goto LABEL_184;
        v270 = v135;
      }
      else
      {
        if ( *(_BYTE *)(*v266 + 8) != 16 || v267 > 0x10u )
          goto LABEL_184;
        v299 = sub_15A1020(*(_BYTE **)(v135 - 48), v130, *v266, v132);
        if ( v299 && *(_BYTE *)(v299 + 16) == 13 )
        {
          v300 = *(_DWORD *)(v299 + 32);
          if ( v300 <= 0x40 )
            v301 = *(_QWORD *)(v299 + 24) == 0;
          else
            v301 = v300 == (unsigned int)sub_16A57B0(v299 + 24);
          v270 = (__int64)v375;
          if ( !v301 )
          {
            v135 = (__int64)v375;
            goto LABEL_184;
          }
        }
        else
        {
          v302 = 0;
          v371 = *(_DWORD *)(*v266 + 32);
          while ( v371 != v302 )
          {
            v303 = sub_15A0A60((__int64)v266, v302);
            if ( !v303 )
              goto LABEL_383;
            v304 = *(_BYTE *)(v303 + 16);
            if ( v304 != 9 )
            {
              if ( v304 != 13 )
                goto LABEL_383;
              v305 = *(_DWORD *)(v303 + 32);
              if ( v305 <= 0x40 )
              {
                if ( *(_QWORD *)(v303 + 24) )
                  goto LABEL_383;
              }
              else if ( v305 != (unsigned int)sub_16A57B0(v303 + 24) )
              {
                goto LABEL_383;
              }
            }
            ++v302;
          }
          v270 = (__int64)v375;
        }
      }
      v271 = *(_QWORD *)(v135 - 24);
      v272 = *(unsigned __int8 *)(v271 + 16);
      if ( (unsigned __int8)v272 > 0x17u )
      {
        v306 = v272 - 24;
      }
      else
      {
        if ( (_BYTE)v272 != 5 )
        {
          v135 = v270;
          goto LABEL_184;
        }
        v306 = *(unsigned __int16 *)(v271 + 18);
      }
      v135 = v270;
      if ( v306 == 45 )
      {
        v283 = v375;
        goto LABEL_377;
      }
LABEL_184:
      v137 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
      v390.m128_u64[0] = (unsigned __int64)&v384;
      v390.m128_u64[1] = v137;
      if ( !sub_170B2D0((__int64)&v390, v135) )
        goto LABEL_82;
      v391.m128i_i16[0] = 257;
      v349 = sub_15FDF90(v384.m128i_i64[0], v343, (__int64)&v390, 0);
      goto LABEL_15;
    }
    if ( v136 != 5 || *(_WORD *)(v135 + 18) != 13 )
      goto LABEL_184;
    v277 = *(_DWORD *)(v135 + 20);
    v278 = *(_QWORD *)(v135 - 24LL * (v277 & 0xFFFFFFF));
    if ( *(_BYTE *)(v278 + 16) == 13 )
    {
      v279 = *(_DWORD *)(v278 + 32);
      if ( v279 <= 0x40 )
      {
        if ( !*(_QWORD *)(v278 + 24) )
          goto LABEL_373;
        goto LABEL_383;
      }
      v280 = v279 == (unsigned int)sub_16A57B0(v278 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v278 + 8LL) != 16 )
        goto LABEL_184;
      v311 = sub_15A1020(
               *(_BYTE **)(v135 - 24LL * (*(_DWORD *)(v135 + 20) & 0xFFFFFFF)),
               v130,
               -24LL * (*(_DWORD *)(v135 + 20) & 0xFFFFFFF),
               v132);
      if ( !v311 || *(_BYTE *)(v311 + 16) != 13 )
      {
        v313 = 0;
        v372 = *(_DWORD *)(*(_QWORD *)v278 + 32LL);
        while ( v372 != v313 )
        {
          v314 = sub_15A0A60(v278, v313);
          if ( !v314 )
            goto LABEL_383;
          v315 = *(_BYTE *)(v314 + 16);
          if ( v315 != 9 )
          {
            if ( v315 != 13 )
              goto LABEL_383;
            v316 = *(_DWORD *)(v314 + 32);
            if ( !(v316 <= 0x40 ? *(_QWORD *)(v314 + 24) == 0 : v316 == (unsigned int)sub_16A57B0(v314 + 24)) )
              goto LABEL_383;
          }
          ++v313;
        }
        goto LABEL_372;
      }
      v312 = *(_DWORD *)(v311 + 32);
      if ( v312 <= 0x40 )
        v280 = *(_QWORD *)(v311 + 24) == 0;
      else
        v280 = v312 == (unsigned int)sub_16A57B0(v311 + 24);
    }
    if ( v280 )
    {
LABEL_372:
      v277 = *(_DWORD *)(v135 + 20);
LABEL_373:
      v281 = *(_QWORD *)(v135 + 24 * (1LL - (v277 & 0xFFFFFFF)));
      v135 = (__int64)v375;
      v282 = *(unsigned __int8 *)(v281 + 16);
      v283 = v375;
      if ( (unsigned __int8)v282 > 0x17u )
      {
        v284 = v282 - 24;
      }
      else
      {
        if ( (_BYTE)v282 != 5 )
          goto LABEL_184;
        v284 = *(unsigned __int16 *)(v281 + 18);
      }
      if ( v284 == 45 )
      {
LABEL_377:
        v391.m128i_i16[0] = 257;
        v285 = sub_1708970(*(_QWORD *)(a1 + 8), 45, v345, *v283, (__int64 *)&v390);
        v286 = *(_QWORD *)(a1 + 8);
        v287 = v285;
        v383 = 257;
        v288 = sub_13CF970((__int64)v283);
        v289 = *(_QWORD *)(v288 + 24);
        if ( v287[16] > 0x10u || *(_BYTE *)(v289 + 16) > 0x10u )
        {
          v292 = *(_QWORD *)(v288 + 24);
          v391.m128i_i16[0] = 257;
          v290 = sub_15FB440(13, (__int64 *)v287, v292, (__int64)&v390, 0);
          v293 = *(_QWORD *)(v286 + 8);
          if ( v293 )
          {
            v294 = *(__int64 **)(v286 + 16);
            sub_157E9D0(v293 + 40, v290);
            v295 = *(_QWORD *)(v290 + 24);
            v296 = *v294;
            *(_QWORD *)(v290 + 32) = v294;
            v296 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v290 + 24) = v296 | v295 & 7;
            *(_QWORD *)(v296 + 8) = v290 + 24;
            *v294 = *v294 & 7 | (v290 + 24);
          }
          v67 = (__int64 *)&v381;
          v68 = v290;
          sub_164B780(v290, (__int64 *)&v381);
          v164 = *(_QWORD *)(v286 + 80) == 0;
          v377 = v290;
          if ( v164 )
LABEL_435:
            sub_4263D6(v68, v67, v69);
          (*(void (__fastcall **)(__int64, __int64 *))(v286 + 88))(v286 + 64, &v377);
          sub_12A86E0((__int64 *)v286, v290);
        }
        else
        {
          v290 = sub_15A2B60((__int64 *)v287, v289, 0, 0, *(double *)v25.m128_u64, a4, *(double *)v24.m128i_i64);
          v291 = sub_14DBA30(v290, *(_QWORD *)(v286 + 96), 0);
          if ( v291 )
            v290 = v291;
        }
        v391.m128i_i16[0] = 257;
        v349 = sub_15FDBD0(46, v290, v343, (__int64)&v390, 0);
        goto LABEL_15;
      }
      goto LABEL_184;
    }
LABEL_383:
    v135 = (__int64)v375;
    goto LABEL_184;
  }
  v132 = v374;
  if ( 1LL << v374 == v133 )
    goto LABEL_180;
LABEL_82:
  if ( *(_BYTE *)(v343 + 8) == 16 )
    goto LABEL_22;
  for ( i = (_BYTE *)v345; i[16] == 71; i = (_BYTE *)*((_QWORD *)i - 3) )
    ;
  if ( (_BYTE *)v345 == i )
    goto LABEL_126;
  v75 = *(_QWORD *)i;
  v76 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
  v77 = v76;
  v78 = *(_QWORD *)(v14 + 24 * (1LL - v76));
  if ( *(_BYTE *)(v78 + 16) == 13 )
  {
    if ( *(_DWORD *)(v78 + 32) <= 0x40u )
    {
      v79 = *(_QWORD *)(v78 + 24) == 0;
    }
    else
    {
      v367 = *(_DWORD *)(v78 + 32);
      v79 = v367 == (unsigned int)sub_16A57B0(v78 + 24);
    }
    if ( v79 )
    {
      if ( *(_BYTE *)(v342 + 8) != 14 )
        goto LABEL_126;
      v80 = *(_QWORD *)(v342 + 24);
      v81 = *(_QWORD *)(v75 + 24);
      if ( v81 == v80 )
      {
        v390.m128_u64[0] = (unsigned __int64)&v391;
        v390.m128_u64[1] = 0x800000000LL;
        sub_13D6230((__int64)&v390, (char *)(v14 + 24 * (2 - v77)), (char *)v14);
        v381 = (unsigned __int64)sub_1649960(v14);
        v382 = v249;
        v385.m128_i16[0] = 261;
        v384.m128i_i64[0] = (__int64)&v381;
        v250 = sub_1704C00(
                 *(_QWORD *)(v75 + 24),
                 (__int64)i,
                 (__int64 *)v390.m128_u64[0],
                 v390.m128_u32[2],
                 (__int64)&v384,
                 0);
        v349 = (__int64)v250;
        v251 = sub_15FA300(v14);
        sub_15FA2E0((__int64)v250, v251);
        if ( (unsigned int)sub_1704F50(v14) != *(_DWORD *)(v75 + 8) >> 8 )
        {
          v383 = 257;
          v252 = sub_1709640(*(_QWORD *)(a1 + 8), v250, (__int64 *)&v381);
          v385.m128_i16[0] = 257;
          v253 = sub_1648A60(56, 1u);
          v349 = (__int64)v253;
          if ( v253 )
            sub_15FDB10((__int64)v253, (__int64)v252, v343, (__int64)&v384, 0);
        }
        goto LABEL_171;
      }
      if ( *(_BYTE *)(v81 + 8) == 14
        && v80 == *(_QWORD *)(v81 + 24)
        && (unsigned int)sub_1704F50(v14) == *(_DWORD *)(v75 + 8) >> 8 )
      {
        sub_1593B40((_QWORD *)(v14 - 24 * v77), (__int64)i);
        *(_QWORD *)(v14 + 56) = v81;
        goto LABEL_15;
      }
LABEL_126:
      v98 = *(_BYTE *)(v345 + 16);
      if ( v98 == 72 )
      {
        v190 = *(_QWORD *)(v345 - 24);
        if ( *(_BYTE *)(v190 + 16) == 71 )
          goto LABEL_253;
        goto LABEL_128;
      }
      if ( v98 != 71 )
        goto LABEL_128;
      v190 = v345;
LABEL_253:
      v191 = *(_BYTE **)(v190 - 24);
      v192 = *(_QWORD *)v191;
      if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) == 3 )
      {
        v193 = *(_QWORD *)(v192 + 24);
        v194 = *(_BYTE *)(v342 + 8);
        if ( v194 == 14 )
        {
          if ( *(_BYTE *)(v193 + 8) == 16
            && **(_QWORD **)(v342 + 16) == **(_QWORD **)(v193 + 16)
            && *(_QWORD *)(v342 + 32) == *(_DWORD *)(v193 + 32) )
          {
            goto LABEL_313;
          }
        }
        else if ( v194 == 16
               && *(_BYTE *)(v193 + 8) == 14
               && **(_QWORD **)(v193 + 16) == **(_QWORD **)(v342 + 16)
               && *(_QWORD *)(v193 + 32) == *(_DWORD *)(v342 + 32) )
        {
LABEL_313:
          if ( sub_15FA300(v14) )
          {
            v391.m128i_i16[0] = 257;
            v237 = *(_QWORD *)(a1 + 8);
            v384 = *(__m128i *)(v387 + 8);
            v238 = sub_1709730(v237, v193, v191, (__int64 **)&v384, 2u, (__int64 *)&v390);
          }
          else
          {
            v391.m128i_i16[0] = 257;
            v275 = *(_QWORD *)(a1 + 8);
            v384 = *(__m128i *)(v387 + 8);
            v238 = sub_1709A60(v275, v193, v191, (__int64 **)&v384, 2u, (__int64 *)&v390);
          }
          sub_164B7C0((__int64)v238, v14);
          v239 = *(_QWORD *)v238;
          if ( *(_BYTE *)(*(_QWORD *)v238 + 8LL) == 16 )
            v239 = **(_QWORD **)(v239 + 16);
          if ( (unsigned int)sub_1704F50(v14) == *(_DWORD *)(v239 + 8) >> 8 )
          {
            v349 = sub_170E100(
                     (__int64 *)a1,
                     v14,
                     (__int64)v238,
                     v25,
                     a4,
                     *(double *)v24.m128i_i64,
                     a6,
                     v240,
                     v241,
                     *(double *)a9.m128i_i64,
                     a10);
          }
          else
          {
            v391.m128i_i16[0] = 257;
            v349 = (__int64)sub_1648A60(56, 1u);
            if ( v349 )
              sub_15FDB10(v349, (__int64)v238, v343, (__int64)&v390, 0);
          }
          goto LABEL_15;
        }
      }
      LODWORD(v382) = sub_15A95F0(*(_QWORD *)(a1 + 2664), v343);
      if ( (unsigned int)v382 <= 0x40 )
      {
        v381 = 0;
        if ( v191[16] != 71 )
          goto LABEL_259;
LABEL_128:
        if ( sub_15FA300(v14) )
          goto LABEL_138;
        v99 = *(_QWORD *)v345;
        if ( *(_BYTE *)(*(_QWORD *)v345 + 8LL) == 16 )
          v99 = **(_QWORD **)(v99 + 16);
        v100 = 8 * sub_15A95A0(*(_QWORD *)(a1 + 2664), *(_DWORD *)(v99 + 8) >> 8);
        v379 = v100;
        if ( v100 <= 0x40 )
          v378 = 0;
        else
          sub_16A4EF0((__int64)&v378, 0, 0);
        v101 = sub_164A410(v345, *(_QWORD *)(a1 + 2664), (__int64)&v378);
        if ( *(_BYTE *)(v101 + 16) == 53 && (unsigned __int8)sub_15FA310(v14, *(_QWORD *)(a1 + 2664), (__int64)&v378) )
        {
          v102 = v378;
          v208 = 1LL << ((unsigned __int8)v379 - 1);
          if ( v379 <= 0x40 )
          {
            if ( (v208 & v378) == 0 )
              goto LABEL_281;
LABEL_138:
            v103 = *(_QWORD *)(a1 + 8);
            if ( (unsigned __int8)sub_15FA290(v14) )
            {
              v105 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
              v106 = *(_QWORD *)(v14 - 24 * v105);
              if ( *(_BYTE *)(v106 + 16) == 79 )
              {
                v107 = *(_QWORD *)(v106 - 72);
                if ( v107 )
                {
                  v108 = *(_BYTE **)(v106 - 48);
                  if ( v108[16] <= 0x10u )
                  {
                    v355 = *(_BYTE **)(v106 - 24);
                    if ( v355[16] <= 0x10u )
                    {
                      v390.m128_u64[1] = 0x400000000LL;
                      v390.m128_u64[0] = (unsigned __int64)&v391;
                      v109 = (__int64 *)(v14 + 24 * (1 - v105));
                      v110 = -24 * (1 - v105);
                      v111 = 0xAAAAAAAAAAAAAAABLL * (v110 >> 3);
                      v112 = &v391;
                      v113 = 0;
                      if ( (unsigned __int64)v110 > 0x60 )
                      {
                        v360 = v108;
                        v365 = -1431655765 * (v110 >> 3);
                        sub_16CD150((__int64)&v390, &v391, 0xAAAAAAAAAAAAAAABLL * (v110 >> 3), 8, v111, v104);
                        v113 = v390.m128_i32[2];
                        v108 = v360;
                        LODWORD(v111) = v365;
                        v112 = (__m128i *)(v390.m128_u64[0] + 8LL * v390.m128_u32[2]);
                      }
                      if ( (__int64 *)v14 != v109 )
                      {
                        do
                        {
                          if ( v112 )
                            v112->m128i_i64[0] = *v109;
                          v109 += 3;
                          v112 = (__m128i *)((char *)v112 + 8);
                        }
                        while ( (__int64 *)v14 != v109 );
                        v113 = v390.m128_i32[2];
                      }
                      v363 = v108;
                      v390.m128_i32[2] = v113 + v111;
                      v114 = sub_15FA300(v14);
                      v385.m128_i16[0] = 257;
                      if ( v114 )
                      {
                        v115 = (__int64 *)sub_1709730(
                                            v103,
                                            0,
                                            v363,
                                            (__int64 **)v390.m128_u64[0],
                                            v390.m128_u32[2],
                                            v384.m128i_i64);
                        v385.m128_i16[0] = 257;
                        v116 = sub_1709730(
                                 v103,
                                 0,
                                 v355,
                                 (__int64 **)v390.m128_u64[0],
                                 v390.m128_u32[2],
                                 v384.m128i_i64);
                      }
                      else
                      {
                        v207 = sub_1709A60(
                                 v103,
                                 0,
                                 v363,
                                 (__int64 **)v390.m128_u64[0],
                                 v390.m128_u32[2],
                                 v384.m128i_i64);
                        v385.m128_i16[0] = 257;
                        v115 = (__int64 *)v207;
                        v116 = sub_1709A60(
                                 v103,
                                 0,
                                 v355,
                                 (__int64 **)v390.m128_u64[0],
                                 v390.m128_u32[2],
                                 v384.m128i_i64);
                      }
                      v385.m128_i16[0] = 257;
                      v117 = sub_1648A60(56, 3u);
                      v349 = (__int64)v117;
                      if ( v117 )
                      {
                        v364 = (__int64)v117;
                        v356 = v117 - 9;
                        sub_15F1EA0((__int64)v117, *v115, 55, (__int64)(v117 - 9), 3, 0);
                        if ( *(_QWORD *)(v349 - 72) )
                        {
                          v118 = *(_QWORD *)(v349 - 64);
                          v119 = *(_QWORD *)(v349 - 56) & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v119 = v118;
                          if ( v118 )
                            *(_QWORD *)(v118 + 16) = v119 | *(_QWORD *)(v118 + 16) & 3LL;
                        }
                        *(_QWORD *)(v349 - 72) = v107;
                        v120 = *(_QWORD *)(v107 + 8);
                        *(_QWORD *)(v349 - 64) = v120;
                        if ( v120 )
                          *(_QWORD *)(v120 + 16) = (v349 - 64) | *(_QWORD *)(v120 + 16) & 3LL;
                        *(_QWORD *)(v349 - 56) = *(_QWORD *)(v349 - 56) & 3LL | (v107 + 8);
                        *(_QWORD *)(v107 + 8) = v356;
                        if ( *(_QWORD *)(v349 - 48) )
                        {
                          v121 = *(_QWORD *)(v349 - 40);
                          v122 = *(_QWORD *)(v349 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v122 = v121;
                          if ( v121 )
                            *(_QWORD *)(v121 + 16) = v122 | *(_QWORD *)(v121 + 16) & 3LL;
                        }
                        *(_QWORD *)(v349 - 48) = v115;
                        v123 = v115[1];
                        *(_QWORD *)(v349 - 40) = v123;
                        if ( v123 )
                          *(_QWORD *)(v123 + 16) = (v349 - 40) | *(_QWORD *)(v123 + 16) & 3LL;
                        *(_QWORD *)(v349 - 32) = (unsigned __int64)(v115 + 1) | *(_QWORD *)(v349 - 32) & 3LL;
                        v115[1] = v349 - 48;
                        if ( *(_QWORD *)(v349 - 24) )
                        {
                          v124 = *(_QWORD *)(v349 - 16);
                          v125 = *(_QWORD *)(v349 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v125 = v124;
                          if ( v124 )
                            *(_QWORD *)(v124 + 16) = v125 | *(_QWORD *)(v124 + 16) & 3LL;
                        }
                        *(_QWORD *)(v349 - 24) = v116;
                        if ( v116 )
                        {
                          v126 = *((_QWORD *)v116 + 1);
                          *(_QWORD *)(v349 - 16) = v126;
                          if ( v126 )
                            *(_QWORD *)(v126 + 16) = (v349 - 16) | *(_QWORD *)(v126 + 16) & 3LL;
                          *(_QWORD *)(v349 - 8) = (unsigned __int64)(v116 + 8) | *(_QWORD *)(v349 - 8) & 3LL;
                          *((_QWORD *)v116 + 1) = v349 - 24;
                        }
                        sub_164B780(v349, v384.m128i_i64);
                        sub_15F4370(v364, v106, 0, 0);
                      }
                      else
                      {
                        sub_15F4370(0, v106, 0, 0);
                      }
LABEL_171:
                      if ( (__m128i *)v390.m128_u64[0] != &v391 )
                        _libc_free(v390.m128_u64[0]);
                      goto LABEL_15;
                    }
                  }
                }
              }
            }
LABEL_22:
            v349 = 0;
            goto LABEL_15;
          }
          if ( (*(_QWORD *)(v378 + 8LL * ((v379 - 1) >> 6)) & v208) != 0 )
          {
LABEL_136:
            if ( v102 )
              j_j___libc_free_0_0(v102);
            goto LABEL_138;
          }
LABEL_281:
          v209 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), *(_QWORD *)(v101 + 56));
          LODWORD(v382) = v100;
          if ( v100 > 0x40 )
            sub_16A4EF0((__int64)&v381, v209, 0);
          else
            v381 = v209 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v100);
          if ( (int)sub_16A9900((__int64)&v378, &v381) <= 0 )
          {
            v297.m128i_i64[0] = (__int64)sub_1649960(v14);
            v384 = v297;
            v390.m128_u64[0] = (unsigned __int64)&v384;
            v391.m128i_i16[0] = 261;
            v349 = (__int64)sub_1704DA0(0, v345, (__int64 *)v387 + 1, (unsigned int)v388 - 1LL, (__int64)&v390, 0);
            sub_135E100((__int64 *)&v381);
            sub_135E100(&v378);
            goto LABEL_15;
          }
          sub_135E100((__int64 *)&v381);
        }
        if ( v379 <= 0x40 )
          goto LABEL_138;
        v102 = v378;
        goto LABEL_136;
      }
      sub_16A4EF0((__int64)&v381, 0, 0);
      if ( v191[16] == 71 )
      {
LABEL_286:
        if ( (unsigned int)v382 > 0x40 && v381 )
          j_j___libc_free_0_0(v381);
        goto LABEL_128;
      }
LABEL_259:
      if ( !(unsigned __int8)sub_15FA310(v14, *(_QWORD *)(a1 + 2664), (__int64)&v381) )
        goto LABEL_286;
      v195 = v382;
      if ( (unsigned int)v382 <= 0x40 )
      {
        if ( !v381 )
        {
LABEL_262:
          if ( (v191[16] == 53 || (unsigned __int8)sub_140AF60((__int64)v191, *(_QWORD **)(a1 + 2648), 0))
            && (v196 = sub_1755380(a1, v190), (v197 = v196) != 0) )
          {
            if ( v196 != v190 )
            {
              sub_164B7C0(v196, v190);
              sub_14F2510(*(_QWORD *)(v190 + 40) + 40LL, (__int64 *)(v190 + 24), v197);
              sub_170E100(
                (__int64 *)a1,
                v190,
                v197,
                v25,
                a4,
                *(double *)v24.m128i_i64,
                a6,
                v198,
                v199,
                *(double *)a9.m128i_i64,
                a10);
            }
          }
          else
          {
            if ( *(_BYTE *)(v192 + 8) == 16 )
              v192 = **(_QWORD **)(v192 + 16);
            v254 = sub_1704F50(v14);
            v255 = *(_DWORD *)(v192 + 8);
            v391.m128i_i16[0] = 257;
            if ( v254 == v255 >> 8 )
            {
              v298 = sub_1648A60(56, 1u);
              v349 = (__int64)v298;
              if ( v298 )
                sub_15FD590((__int64)v298, (__int64)v191, v343, (__int64)&v390, 0);
            }
            else
            {
              v256 = sub_1648A60(56, 1u);
              v349 = (__int64)v256;
              if ( v256 )
                sub_15FDB10((__int64)v256, (__int64)v191, v343, (__int64)&v390, 0);
            }
          }
LABEL_267:
          if ( (unsigned int)v382 > 0x40 && v381 )
            j_j___libc_free_0_0(v381);
          goto LABEL_15;
        }
        v390.m128_u64[0] = (unsigned __int64)&v391;
        v390.m128_u64[1] = 0x800000000LL;
        v257 = (__int64)(v381 << (64 - (unsigned __int8)v382)) >> (64 - (unsigned __int8)v382);
      }
      else
      {
        if ( v195 == (unsigned int)sub_16A57B0((__int64)&v381) )
          goto LABEL_262;
        v390.m128_u64[0] = (unsigned __int64)&v391;
        v390.m128_u64[1] = 0x800000000LL;
        v257 = *(_QWORD *)v381;
      }
      if ( !sub_1705540(a1, v192, v257, (__int64)&v390) )
      {
        if ( (__m128i *)v390.m128_u64[0] != &v391 )
          _libc_free(v390.m128_u64[0]);
        goto LABEL_286;
      }
      if ( sub_15FA300(v14) )
      {
        v258 = *(_QWORD *)(a1 + 8);
        v385.m128_i16[0] = 257;
        v259 = sub_1709730(v258, 0, v191, (__int64 **)v390.m128_u64[0], v390.m128_u32[2], v384.m128i_i64);
      }
      else
      {
        v276 = *(_QWORD *)(a1 + 8);
        v385.m128_i16[0] = 257;
        v259 = sub_1709A60(v276, 0, v191, (__int64 **)v390.m128_u64[0], v390.m128_u32[2], v384.m128i_i64);
      }
      if ( v343 == *(_QWORD *)v259 )
      {
        v349 = sub_170E100(
                 (__int64 *)a1,
                 v14,
                 (__int64)v259,
                 v25,
                 a4,
                 *(double *)v24.m128i_i64,
                 a6,
                 v260,
                 v261,
                 *(double *)a9.m128i_i64,
                 a10);
      }
      else
      {
        sub_164B7C0((__int64)v259, v14);
        v262 = *(_QWORD *)v259;
        if ( *(_BYTE *)(*(_QWORD *)v259 + 8LL) == 16 )
          v262 = **(_QWORD **)(v262 + 16);
        v263 = sub_1704F50(v14);
        v264 = *(_DWORD *)(v262 + 8);
        v385.m128_i16[0] = 257;
        if ( v263 == v264 >> 8 )
        {
          v349 = (__int64)sub_1648A60(56, 1u);
          if ( v349 )
            sub_15FD590(v349, (__int64)v259, v343, (__int64)&v384, 0);
        }
        else
        {
          v265 = sub_1648A60(56, 1u);
          v349 = (__int64)v265;
          if ( v265 )
            sub_15FDB10((__int64)v265, (__int64)v259, v343, (__int64)&v384, 0);
        }
      }
      if ( (__m128i *)v390.m128_u64[0] != &v391 )
        _libc_free(v390.m128_u64[0]);
      goto LABEL_267;
    }
  }
  if ( v76 != 2 )
    goto LABEL_126;
  v154 = *(_QWORD *)(v75 + 24);
  if ( *(_BYTE *)(v154 + 8) == 14
    && (v219 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), **(_QWORD **)(v154 + 16)),
        v219 == sub_12BE0A0(*(_QWORD *)(a1 + 2664), v342)) )
  {
    v220 = (__int64 **)sub_15A9730(*(_QWORD *)(a1 + 2664), v343);
    v223 = (__int64 *)sub_15A06D0(v220, v343, v221, v222);
    v224 = *(_DWORD *)(v14 + 20);
    v381 = (unsigned __int64)v223;
    v382 = *(_QWORD *)(v14 + 24 * (1LL - (v224 & 0xFFFFFFF)));
    if ( sub_15FA300(v14) )
    {
      v225 = *(_QWORD *)(a1 + 8);
      v226.m128i_i64[0] = (__int64)sub_1649960(v14);
      v384 = v226;
      v391.m128i_i16[0] = 261;
      v390.m128_u64[0] = (unsigned __int64)&v384;
      v167 = sub_1709730(v225, 0, i, (__int64 **)&v381, 2u, (__int64 *)&v390);
    }
    else
    {
      v273 = *(_QWORD *)(a1 + 8);
      v274.m128i_i64[0] = (__int64)sub_1649960(v14);
      v384 = v274;
      v391.m128i_i16[0] = 261;
      v390.m128_u64[0] = (unsigned __int64)&v384;
      v167 = sub_1709A60(v273, 0, i, (__int64 **)&v381, 2u, (__int64 *)&v390);
    }
  }
  else
  {
    if ( !sub_1704BC0(v342, 0) )
      goto LABEL_470;
    if ( !sub_1704BC0(v154, 0) )
      goto LABEL_470;
    v229 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), v342);
    v230 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), v154);
    if ( !v229 )
      goto LABEL_470;
    v231 = v230 / v229;
    if ( !(v230 % v229)
      && (v232 = *(__int64 **)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF))),
          v233 = sub_1643030(*v232),
          sub_135E0D0((__int64)&v390, v233, v231, 0),
          v234 = sub_170C190((__int64 *)a1, (__int64)v232, (unsigned __int64 *)&v390, (bool *)&v381),
          sub_135E100((__int64 *)&v390),
          v234) )
    {
      v164 = !sub_15FA300(v14);
      v235 = *(_QWORD *)(a1 + 8);
      if ( !v164 && (_BYTE)v381 )
      {
        v339.m128i_i64[0] = (__int64)sub_1649960(v14);
        v384 = v339;
        v391.m128i_i16[0] = 261;
        v390.m128_u64[0] = (unsigned __int64)&v384;
        v167 = sub_1709D80(v235, 0, i, v234, (__int64 *)&v390);
      }
      else
      {
        v236.m128i_i64[0] = (__int64)sub_1649960(v14);
        v384 = v236;
        v391.m128i_i16[0] = 261;
        v390.m128_u64[0] = (unsigned __int64)&v384;
        v167 = sub_170A020(v235, 0, i, v234, (__int64 *)&v390);
      }
    }
    else
    {
LABEL_470:
      if ( !sub_1704BC0(v342, 0) )
        goto LABEL_126;
      if ( !sub_1704BC0(v154, 0) )
        goto LABEL_126;
      if ( *(_BYTE *)(v154 + 8) != 14 )
        goto LABEL_126;
      v155 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), v342);
      v156 = sub_12BE0A0(*(_QWORD *)(a1 + 2664), **(_QWORD **)(v154 + 16));
      if ( !v155 )
        goto LABEL_126;
      v157 = v156 / v155;
      if ( v156 % v155 )
        goto LABEL_126;
      v369 = *(__int64 **)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
      v158 = sub_1643030(*v369);
      sub_135E0D0((__int64)&v390, v158, v157, 0);
      v159 = sub_170C190((__int64 *)a1, (__int64)v369, (unsigned __int64 *)&v390, (bool *)&v378);
      sub_135E100((__int64 *)&v390);
      if ( !v159 )
        goto LABEL_126;
      v160 = (__int64 **)sub_15A9730(*(_QWORD *)(a1 + 2664), v343);
      v163 = (__int64 *)sub_15A06D0(v160, v343, v161, v162);
      v382 = v159;
      v381 = (unsigned __int64)v163;
      v164 = !sub_15FA300(v14);
      v165 = *(_QWORD *)(a1 + 8);
      if ( !v164 && (_BYTE)v378 )
      {
        v310.m128i_i64[0] = (__int64)sub_1649960(v14);
        v384 = v310;
        v391.m128i_i16[0] = 261;
        v390.m128_u64[0] = (unsigned __int64)&v384;
        v167 = sub_1709730(v165, v154, i, (__int64 **)&v381, 2u, (__int64 *)&v390);
      }
      else
      {
        v166.m128i_i64[0] = (__int64)sub_1649960(v14);
        v384 = v166;
        v391.m128i_i16[0] = 261;
        v390.m128_u64[0] = (unsigned __int64)&v384;
        v167 = sub_1709A60(v165, v154, i, (__int64 **)&v381, 2u, (__int64 *)&v390);
      }
    }
  }
  v391.m128i_i16[0] = 257;
  v349 = sub_15FDF90((__int64)v167, v343, (__int64)&v390, 0);
LABEL_15:
  if ( v387 != v389 )
    _libc_free((unsigned __int64)v387);
  return v349;
}
