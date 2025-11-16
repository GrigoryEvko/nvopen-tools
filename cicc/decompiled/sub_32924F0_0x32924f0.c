// Function: sub_32924F0
// Address: 0x32924f0
//
__int64 __fastcall sub_32924F0(__int64 **a1, __int64 a2)
{
  __int64 v4; // rax
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __int16 *v7; // rax
  __int64 v8; // rsi
  __int16 v9; // dx
  __int64 v10; // rax
  int v11; // eax
  __int64 *v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // rdi
  __int64 (*v20)(); // r8
  __int64 v21; // r15
  __int64 v23; // r8
  unsigned __int8 *v24; // rsi
  unsigned __int16 *v25; // rax
  __int64 v26; // r10
  unsigned int v27; // ecx
  __int32 v28; // edx
  char v29; // al
  int v30; // r15d
  unsigned __int8 v31; // bl
  __int64 *v32; // rdi
  __int64 (*v33)(); // rax
  __int64 (*v34)(); // rcx
  char v35; // al
  bool v36; // zf
  char v37; // cl
  unsigned int v38; // r8d
  __int64 v39; // rax
  __m128i *v40; // rbx
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r15
  unsigned int v46; // ebx
  __int64 v47; // r9
  __int64 v48; // rdi
  __int64 v49; // rdx
  unsigned __int16 v50; // ax
  __int64 v51; // r8
  __int64 (__fastcall *v52)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // r11
  __int64 v53; // rcx
  __int64 (*v54)(); // rax
  __int64 v55; // rax
  __int64 v56; // r15
  unsigned int v57; // ebx
  __int64 v58; // rdi
  __int64 v59; // rcx
  __int64 v60; // r9
  unsigned __int16 v61; // dx
  __int64 (__fastcall *v62)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 (*v63)(); // rax
  __int64 v64; // rax
  __int64 v65; // r15
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rbx
  unsigned int v69; // r15d
  __int64 v70; // rax
  _DWORD *v71; // rax
  __int64 v72; // r15
  unsigned int v73; // ebx
  __int64 v74; // rax
  __int64 v75; // r12
  unsigned __int16 *v76; // rax
  __int64 v77; // r9
  __int64 *v78; // rax
  __int64 v79; // r12
  __int64 v80; // r13
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r15
  __int64 v84; // r14
  __int64 v85; // r9
  __int128 v86; // rax
  __int64 v87; // r9
  __int128 v88; // rax
  __int64 v89; // r9
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r15
  __int64 v93; // r14
  __int64 v94; // r9
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // r13
  __int64 v98; // r12
  __int64 v99; // r9
  __int128 v100; // rax
  __int64 v101; // r9
  __int128 v102; // rax
  __int64 v103; // r9
  __int64 v104; // rax
  __m128i *v105; // rbx
  unsigned __int64 v106; // rcx
  __int64 v107; // rdx
  __int64 v108; // rax
  unsigned __int64 v109; // rdx
  int v110; // r14d
  int v111; // ebx
  int v112; // eax
  __m128i v113; // xmm3
  int v114; // r9d
  __int64 v115; // r14
  __int32 v116; // edx
  __int32 v117; // ebx
  __int64 v118; // rax
  int v119; // eax
  __m128i v120; // xmm4
  __m128i v121; // xmm5
  int v122; // r9d
  unsigned int v123; // r14d
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // rax
  unsigned int v127; // r8d
  __int64 v128; // rax
  char v129; // al
  __int64 v130; // rax
  unsigned int v131; // r15d
  __int64 v132; // rax
  __int64 v133; // rax
  __m128i *v134; // rbx
  __int64 v135; // rax
  int v136; // eax
  __m128i v137; // xmm6
  __m128i v138; // xmm7
  int v139; // r9d
  __int64 v140; // rax
  __int32 v141; // edx
  __int32 v142; // r15d
  int v143; // eax
  __m128i v144; // xmm6
  __m128i v145; // xmm7
  __int64 v146; // rbx
  __int64 v147; // rax
  __int64 v148; // r15
  unsigned int v149; // ebx
  __int64 v150; // r9
  unsigned __int16 *v151; // rdx
  __int64 v152; // r9
  __int64 v153; // r12
  __int64 v154; // rdx
  __int64 v155; // r13
  __int64 v156; // r9
  __int128 v157; // rax
  __int64 v158; // r9
  __int128 v159; // rax
  __int64 v160; // r9
  int v161; // eax
  __int64 v162; // rbx
  __int64 v163; // r12
  __int64 v164; // rdx
  __int64 v165; // r13
  __int64 v166; // r9
  __int128 v167; // rax
  __int64 v168; // r9
  __int128 v169; // rax
  __int64 v170; // r9
  __int64 v171; // rax
  unsigned int v172; // r8d
  __int64 v173; // rax
  char v174; // al
  __int64 v175; // rax
  _DWORD *v176; // rbx
  __int64 *v177; // rax
  __int64 v178; // r14
  __int64 v179; // r15
  __int64 v180; // rax
  __int64 v181; // rdx
  __int64 v182; // r13
  __int64 v183; // r12
  __int64 v184; // r9
  __int128 v185; // rax
  __int64 v186; // r9
  __int64 v187; // rax
  char v188; // al
  __int64 v189; // r9
  __int128 v190; // rax
  __int64 v191; // r9
  __int64 v192; // r12
  __int64 v193; // rdx
  __int64 v194; // r13
  __int64 v195; // r9
  __int128 v196; // rax
  __int64 v197; // r9
  __int64 v198; // r9
  __int64 v199; // rax
  __int64 v200; // rax
  int v201; // r15d
  __int64 v202; // rax
  __int64 v203; // rdx
  __int64 v204; // rbx
  __int128 *v205; // rdx
  __int64 v206; // r9
  __int64 v207; // rax
  __int64 v208; // r9
  __int64 v209; // rdx
  __int128 *v210; // rdx
  __int128 v211; // rcx
  __int64 v212; // rbx
  __int64 v213; // rax
  __int64 v214; // rbx
  unsigned int v215; // r15d
  unsigned __int16 *v216; // rax
  __int64 v217; // r9
  __int64 v218; // r14
  __int64 v219; // rdx
  __int64 v220; // r15
  __int64 v221; // r9
  __int64 v222; // r12
  __int64 v223; // rdx
  __int64 v224; // r13
  __int64 v225; // r9
  __int128 v226; // rax
  __int64 v227; // r9
  __int64 v228; // rdx
  char v229; // al
  __int64 v230; // r9
  __int64 v231; // rdx
  __int64 v232; // r9
  __int128 v233; // rax
  __int64 v234; // r9
  __int64 v235; // r9
  __int64 v236; // rax
  int v237; // r15d
  __int64 v238; // rax
  __int64 v239; // r12
  __int64 v240; // r13
  __int128 v241; // rax
  __int64 v242; // r9
  __int64 v243; // rax
  __int64 v244; // r9
  __int128 *v245; // rbx
  __int64 v246; // r12
  __int64 v247; // rdx
  __int64 v248; // r13
  __int128 v249; // rax
  __int64 v250; // r9
  unsigned __int16 *v251; // rax
  char v252; // al
  __int64 v253; // r9
  __int64 v254; // r14
  __int64 v255; // rdx
  __int64 v256; // r15
  __int64 v257; // r9
  __int64 v258; // r12
  __int64 v259; // rdx
  __int64 v260; // r13
  __int64 v261; // r9
  __int128 v262; // rax
  __int64 v263; // r9
  __int64 v264; // r14
  __int64 v265; // rdx
  __int64 v266; // r15
  __int64 v267; // r9
  __int64 v268; // r12
  __int64 v269; // rdx
  __int64 v270; // r13
  __int64 v271; // r9
  __int128 v272; // rax
  __int64 v273; // r9
  __int64 v274; // rax
  __int64 v275; // r15
  unsigned int v276; // ebx
  __int64 v277; // r9
  __int64 *v278; // rax
  __int64 v279; // r14
  __int64 v280; // r15
  __int64 v281; // rax
  __int64 v282; // rdx
  __int64 v283; // r13
  __int64 v284; // r12
  __int64 v285; // r9
  __int128 v286; // rax
  __int64 v287; // r9
  __int128 v288; // rax
  __int64 v289; // r9
  __int64 v290; // rdx
  __int128 v291; // [rsp-30h] [rbp-220h]
  __int128 v292; // [rsp-30h] [rbp-220h]
  __int128 v293; // [rsp-30h] [rbp-220h]
  __int128 v294; // [rsp-30h] [rbp-220h]
  __int128 v295; // [rsp-30h] [rbp-220h]
  __int128 v296; // [rsp-30h] [rbp-220h]
  __int128 v297; // [rsp-30h] [rbp-220h]
  __int128 v298; // [rsp-20h] [rbp-210h]
  __int128 v299; // [rsp-20h] [rbp-210h]
  __int128 v300; // [rsp-20h] [rbp-210h]
  __int128 v301; // [rsp-20h] [rbp-210h]
  __int128 v302; // [rsp-20h] [rbp-210h]
  __int128 v303; // [rsp-20h] [rbp-210h]
  __int128 v304; // [rsp-20h] [rbp-210h]
  __int128 v305; // [rsp-20h] [rbp-210h]
  __int128 v306; // [rsp-20h] [rbp-210h]
  __int128 v307; // [rsp-20h] [rbp-210h]
  __int128 v308; // [rsp-20h] [rbp-210h]
  __int128 v309; // [rsp-20h] [rbp-210h]
  __int128 v310; // [rsp-20h] [rbp-210h]
  __int128 v311; // [rsp-20h] [rbp-210h]
  __int128 v312; // [rsp-10h] [rbp-200h]
  __int128 v313; // [rsp-10h] [rbp-200h]
  __int128 v314; // [rsp-10h] [rbp-200h]
  __int128 v315; // [rsp-10h] [rbp-200h]
  __int128 v316; // [rsp-10h] [rbp-200h]
  __int128 v317; // [rsp-10h] [rbp-200h]
  __int128 v318; // [rsp-10h] [rbp-200h]
  __int128 v319; // [rsp-10h] [rbp-200h]
  __int128 v320; // [rsp-10h] [rbp-200h]
  __int128 v321; // [rsp-10h] [rbp-200h]
  __int128 v322; // [rsp-10h] [rbp-200h]
  __int64 v323; // [rsp-8h] [rbp-1F8h]
  __int64 v324; // [rsp-8h] [rbp-1F8h]
  __int64 v325; // [rsp+0h] [rbp-1F0h]
  unsigned __int8 *v326; // [rsp+18h] [rbp-1D8h]
  __int32 v327; // [rsp+18h] [rbp-1D8h]
  int v328; // [rsp+20h] [rbp-1D0h]
  unsigned __int8 *v329; // [rsp+20h] [rbp-1D0h]
  __int64 v330; // [rsp+28h] [rbp-1C8h]
  int v331; // [rsp+28h] [rbp-1C8h]
  __int32 v332; // [rsp+30h] [rbp-1C0h]
  __int64 v333; // [rsp+30h] [rbp-1C0h]
  int v334; // [rsp+38h] [rbp-1B8h]
  int v335; // [rsp+38h] [rbp-1B8h]
  int v336; // [rsp+38h] [rbp-1B8h]
  unsigned int v337; // [rsp+40h] [rbp-1B0h]
  int v338; // [rsp+40h] [rbp-1B0h]
  unsigned int v339; // [rsp+40h] [rbp-1B0h]
  int v340; // [rsp+40h] [rbp-1B0h]
  int v341; // [rsp+40h] [rbp-1B0h]
  unsigned int v342; // [rsp+40h] [rbp-1B0h]
  unsigned int v343; // [rsp+48h] [rbp-1A8h]
  __int64 v344; // [rsp+70h] [rbp-180h]
  unsigned __int8 v345; // [rsp+70h] [rbp-180h]
  char v346; // [rsp+70h] [rbp-180h]
  __int64 v347; // [rsp+78h] [rbp-178h]
  __int64 v348; // [rsp+80h] [rbp-170h]
  __int64 v349; // [rsp+80h] [rbp-170h]
  __int128 v350; // [rsp+80h] [rbp-170h]
  __int64 v351; // [rsp+90h] [rbp-160h]
  __int128 v352; // [rsp+90h] [rbp-160h]
  __int128 v353; // [rsp+90h] [rbp-160h]
  __int128 v354; // [rsp+90h] [rbp-160h]
  __int64 v355; // [rsp+90h] [rbp-160h]
  char v356; // [rsp+BBh] [rbp-135h] BYREF
  unsigned int v357; // [rsp+BCh] [rbp-134h] BYREF
  __int64 v358; // [rsp+C0h] [rbp-130h] BYREF
  __m128i *v359; // [rsp+C8h] [rbp-128h] BYREF
  __int64 v360; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v361; // [rsp+D8h] [rbp-118h]
  __int64 v362; // [rsp+E0h] [rbp-110h] BYREF
  int v363; // [rsp+E8h] [rbp-108h]
  unsigned __int8 v364[8]; // [rsp+F0h] [rbp-100h] BYREF
  __m128i *v365; // [rsp+F8h] [rbp-F8h]
  _QWORD v366[6]; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v367; // [rsp+130h] [rbp-C0h] BYREF
  __m128i v368; // [rsp+140h] [rbp-B0h] BYREF
  __m128i v369; // [rsp+150h] [rbp-A0h] BYREF
  __int64 v370; // [rsp+160h] [rbp-90h]
  unsigned __int8 *v371; // [rsp+170h] [rbp-80h] BYREF
  __int64 *v372; // [rsp+178h] [rbp-78h]
  __m128i v373; // [rsp+180h] [rbp-70h]
  __m128i v374; // [rsp+190h] [rbp-60h]
  __m128i v375; // [rsp+1A0h] [rbp-50h]
  __m128i v376; // [rsp+1B0h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = _mm_loadu_si128((const __m128i *)v4);
  v6 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v351 = *(_QWORD *)v4;
  v347 = *(_QWORD *)(v4 + 40);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v362 = v8;
  LOWORD(v360) = v9;
  v361 = v10;
  if ( v8 )
    sub_B96E90((__int64)&v362, v8, 1);
  v11 = *(_DWORD *)(a2 + 72);
  v12 = *a1;
  v368.m128i_i64[0] = 0;
  v13 = *(unsigned int *)(a2 + 24);
  v368.m128i_i32[2] = 0;
  v363 = v11;
  v14 = (__int64)a1[1];
  v367.m128i_i64[0] = (__int64)v12;
  v367.m128i_i64[1] = v14;
  v369.m128i_i64[0] = 0;
  v369.m128i_i32[2] = 0;
  v370 = a2;
  v366[0] = sub_33CB160(v13);
  if ( BYTE4(v366[0]) )
  {
    v15 = *(_QWORD *)(v370 + 40) + 40LL * LODWORD(v366[0]);
    v368.m128i_i64[0] = *(_QWORD *)v15;
    v368.m128i_i32[2] = *(_DWORD *)(v15 + 8);
    v16 = *(unsigned int *)(v370 + 24);
  }
  else
  {
    v23 = v370;
    v16 = *(unsigned int *)(v370 + 24);
    if ( (_DWORD)v16 == 488 )
    {
      v24 = *(unsigned __int8 **)(v370 + 80);
      v25 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v370 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(v370 + 40) + 8LL));
      v26 = *((_QWORD *)v25 + 1);
      v27 = *v25;
      v371 = v24;
      if ( v24 )
      {
        v343 = v27;
        v344 = v26;
        v349 = v370;
        sub_B96E90((__int64)&v371, (__int64)v24, 1);
        v27 = v343;
        v26 = v344;
        v23 = v349;
      }
      LODWORD(v372) = *(_DWORD *)(v23 + 72);
      v368.m128i_i64[0] = sub_34015B0(v12, &v371, v27, v26, 0, 0);
      v368.m128i_i32[2] = v28;
      if ( v371 )
        sub_B91220((__int64)&v371, (__int64)v371);
      v16 = *(unsigned int *)(v370 + 24);
    }
  }
  v371 = (unsigned __int8 *)sub_33CB1F0(v16);
  if ( BYTE4(v371) )
  {
    v17 = *(_QWORD *)(v370 + 40) + 40LL * (unsigned int)v371;
    v369.m128i_i64[0] = *(_QWORD *)v17;
    v369.m128i_i32[2] = *(_DWORD *)(v17 + 8);
  }
  v18 = *a1;
  v348 = **a1;
  if ( *((_BYTE *)a1 + 33) )
  {
    v371 = (unsigned __int8 *)sub_33CB7C0(150);
    if ( (_WORD)v360 != 1 && (!(_WORD)v360 || !*(_QWORD *)(v367.m128i_i64[1] + 8LL * (unsigned __int16)v360 + 112)) )
      goto LABEL_9;
    if ( (unsigned int)v371 <= 0x1F3 )
    {
      v29 = *(_BYTE *)((unsigned int)v371 + 500LL * (unsigned __int16)v360 + v367.m128i_i64[1] + 6414);
      if ( v29 )
      {
        if ( v29 != 4 )
          goto LABEL_9;
      }
    }
    v18 = *a1;
  }
  v19 = a1[1];
  v20 = *(__int64 (**)())(*v19 + 1608);
  if ( v20 == sub_2FE3540 )
    goto LABEL_9;
  v345 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, __int64))v20)(v19, v18[5], (unsigned int)v360, v361);
  if ( !v345 )
    goto LABEL_9;
  v30 = *(_DWORD *)(a2 + 28);
  if ( !*(_DWORD *)(v348 + 952) || (*(_BYTE *)(v348 + 864) & 1) != 0 )
  {
    v31 = v345;
  }
  else
  {
    v31 = 0;
    if ( (v30 & 0x200) == 0 )
      goto LABEL_9;
  }
  v32 = a1[1];
  v33 = *(__int64 (**)())(*v32 + 1648);
  if ( v33 != sub_2FE3570 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD))v33)(
           v32,
           (unsigned int)v360,
           v361,
           *((unsigned int *)a1 + 7)) )
    {
      goto LABEL_9;
    }
    v32 = a1[1];
  }
  v357 = 150;
  v34 = *(__int64 (**)())(*v32 + 512);
  v35 = 0;
  if ( v34 != sub_2FE30F0 )
    v35 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v34)(v32, (unsigned int)v360, v361);
  v356 = v35;
  v364[0] = v31;
  v36 = (*(_BYTE *)(v348 + 864) & 0x10) == 0;
  v37 = v345;
  v366[2] = &v367;
  if ( v36 )
    v37 = (unsigned __int8)v30 >> 7;
  v365 = &v367;
  v366[0] = v364;
  v366[1] = &v356;
  v366[3] = &v357;
  v366[4] = &v362;
  v366[5] = &v360;
  v346 = v37;
  if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(v351 + 24)) )
  {
    v371 = (unsigned __int8 *)sub_33CB280(
                                *(unsigned int *)(v351 + 24),
                                ((unsigned __int8)(*(_DWORD *)(v351 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v371) )
      goto LABEL_95;
    v38 = *(_DWORD *)(v351 + 24);
    if ( (_DWORD)v371 != 98 )
    {
LABEL_40:
      v40 = v365;
      if ( (unsigned __int8)sub_33CB110(v38) )
      {
        v371 = (unsigned __int8 *)sub_33CB280(
                                    *(unsigned int *)(v351 + 24),
                                    ((unsigned __int8)(*(_DWORD *)(v351 + 28) >> 12) ^ 1) & 1);
        if ( !BYTE4(v371) || (_DWORD)v371 != 98 )
          goto LABEL_46;
        v123 = *(_DWORD *)(v351 + 24);
        v359 = (__m128i *)sub_33CB160(v123);
        if ( BYTE4(v359) )
        {
          v124 = *(_QWORD *)(v351 + 40) + 40LL * (unsigned int)v359;
          if ( (*(_QWORD *)v124 != v40[1].m128i_i64[0] || *(_DWORD *)(v124 + 8) != v40[1].m128i_i32[2])
            && !(unsigned __int8)sub_33D1720(*(_QWORD *)v124, 0) )
          {
            goto LABEL_46;
          }
        }
        v359 = (__m128i *)sub_33CB1F0(v123);
        if ( BYTE4(v359) )
        {
          v125 = *(_QWORD *)(v351 + 40) + 40LL * (unsigned int)v359;
          if ( v40[2].m128i_i64[0] != *(_QWORD *)v125 || v40[2].m128i_i32[2] != *(_DWORD *)(v125 + 8) )
            goto LABEL_46;
        }
      }
      else if ( *(_DWORD *)(v351 + 24) != 98 )
      {
        goto LABEL_46;
      }
      if ( (v364[0] || (*(_BYTE *)(v351 + 29) & 2) != 0)
        && (v356 || (v41 = *(_QWORD *)(v351 + 56)) != 0 && !*(_QWORD *)(v41 + 32)) )
      {
        v110 = v360;
        v111 = v361;
        v112 = sub_33CB7C0(244);
        LODWORD(v372) = v6.m128i_i32[2];
        *((_QWORD *)&v312 + 1) = 3;
        *(_QWORD *)&v312 = &v371;
        v113 = _mm_loadu_si128(&v369);
        v373 = _mm_loadu_si128(&v368);
        v374 = v113;
        v371 = (unsigned __int8 *)v347;
        v115 = sub_33FC220(v367.m128i_i32[0], v112, (unsigned int)&v362, v110, v111, v114, v312);
        v117 = v116;
        v338 = v360;
        v118 = *(_QWORD *)(v351 + 40);
        v334 = v361;
        v326 = *(unsigned __int8 **)v118;
        v328 = *(_DWORD *)(v118 + 8);
        v330 = *(_QWORD *)(v118 + 40);
        v332 = *(_DWORD *)(v118 + 48);
        v119 = sub_33CB7C0(v357);
        v374.m128i_i64[0] = v115;
        v374.m128i_i32[2] = v117;
        v373.m128i_i32[2] = v332;
        v120 = _mm_loadu_si128(&v368);
        *((_QWORD *)&v300 + 1) = 5;
        *(_QWORD *)&v300 = &v371;
        v373.m128i_i64[0] = v330;
        v121 = _mm_loadu_si128(&v369);
        v371 = v326;
        LODWORD(v372) = v328;
        v375 = v120;
        v376 = v121;
        v21 = sub_33FC220(v367.m128i_i32[0], v119, (unsigned int)&v362, v338, v334, v122, v300);
        if ( v21 )
          goto LABEL_10;
      }
LABEL_46:
      v21 = sub_3268760((__int64)v366, v5.m128i_i64[0], v5.m128i_i32[2], v347);
      if ( v21 )
        goto LABEL_10;
      goto LABEL_47;
    }
    v339 = *(_DWORD *)(v351 + 24);
    v126 = sub_33CB160(v38);
    v127 = v339;
    v359 = (__m128i *)v126;
    if ( BYTE4(v126) )
    {
      v128 = *(_QWORD *)(v351 + 40) + 40LL * (unsigned int)v359;
      if ( *(_QWORD *)v128 != v368.m128i_i64[0] || *(_DWORD *)(v128 + 8) != v368.m128i_i32[2] )
      {
        v129 = sub_33D1720(*(_QWORD *)v128, 0);
        v127 = v339;
        if ( !v129 )
          goto LABEL_95;
      }
    }
    v359 = (__m128i *)sub_33CB1F0(v127);
    if ( BYTE4(v359) )
    {
      v130 = *(_QWORD *)(v351 + 40) + 40LL * (unsigned int)v359;
      if ( v369.m128i_i64[0] != *(_QWORD *)v130 || v369.m128i_i32[2] != *(_DWORD *)(v130 + 8) )
        goto LABEL_95;
    }
  }
  else
  {
    v38 = *(_DWORD *)(v351 + 24);
    if ( v38 != 98 )
      goto LABEL_40;
  }
  if ( !v364[0] )
  {
    v39 = v351;
    if ( (*(_BYTE *)(v351 + 29) & 2) == 0 )
      goto LABEL_39;
  }
  v105 = v365;
  if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(v347 + 24)) )
  {
    v371 = (unsigned __int8 *)sub_33CB280(
                                *(unsigned int *)(v347 + 24),
                                ((unsigned __int8)(*(_DWORD *)(v347 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v371) || (_DWORD)v371 != 98 )
      goto LABEL_95;
    v131 = *(_DWORD *)(v347 + 24);
    v359 = (__m128i *)sub_33CB160(v131);
    if ( BYTE4(v359) )
    {
      v132 = *(_QWORD *)(v347 + 40) + 40LL * (unsigned int)v359;
      if ( (*(_QWORD *)v132 != v105[1].m128i_i64[0] || *(_DWORD *)(v132 + 8) != v105[1].m128i_i32[2])
        && !(unsigned __int8)sub_33D1720(*(_QWORD *)v132, 0) )
      {
        goto LABEL_95;
      }
    }
    v359 = (__m128i *)sub_33CB1F0(v131);
    if ( BYTE4(v359) )
    {
      v133 = *(_QWORD *)(v347 + 40) + 40LL * (unsigned int)v359;
      if ( v105[2].m128i_i64[0] != *(_QWORD *)v133 || v105[2].m128i_i32[2] != *(_DWORD *)(v133 + 8) )
        goto LABEL_95;
    }
  }
  else if ( *(_DWORD *)(v347 + 24) != 98 )
  {
    goto LABEL_95;
  }
  if ( !v364[0] && (*(_BYTE *)(v347 + 29) & 2) == 0 )
    goto LABEL_95;
  v106 = 0;
  v107 = *(_QWORD *)(v351 + 56);
  v108 = *(_QWORD *)(v347 + 56);
  if ( !v107 )
  {
    if ( !v108 )
      goto LABEL_95;
    goto LABEL_85;
  }
  do
  {
    v107 = *(_QWORD *)(v107 + 32);
    ++v106;
  }
  while ( v107 );
  if ( v108 )
  {
LABEL_85:
    v109 = 0;
    do
    {
      v108 = *(_QWORD *)(v108 + 32);
      ++v109;
    }
    while ( v108 );
    if ( v109 < v106 )
      goto LABEL_88;
LABEL_95:
    v39 = v351;
LABEL_39:
    v38 = *(_DWORD *)(v39 + 24);
    goto LABEL_40;
  }
LABEL_88:
  v21 = sub_3268760((__int64)v366, v5.m128i_i64[0], v5.m128i_i32[2], v347);
  if ( !v21 )
  {
    v134 = v365;
    if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(v351 + 24)) )
    {
      v371 = (unsigned __int8 *)sub_33CB280(
                                  *(unsigned int *)(v351 + 24),
                                  ((unsigned __int8)(*(_DWORD *)(v351 + 28) >> 12) ^ 1) & 1);
      if ( !BYTE4(v371) || (_DWORD)v371 != 98 )
        goto LABEL_47;
      v342 = *(_DWORD *)(v351 + 24);
      v171 = sub_33CB160(v342);
      v172 = v342;
      v359 = (__m128i *)v171;
      if ( BYTE4(v171) )
      {
        v173 = *(_QWORD *)(v351 + 40) + 40LL * (unsigned int)v359;
        if ( *(_QWORD *)v173 != v134[1].m128i_i64[0] || *(_DWORD *)(v173 + 8) != v134[1].m128i_i32[2] )
        {
          v174 = sub_33D1720(*(_QWORD *)v173, 0);
          v172 = v342;
          if ( !v174 )
            goto LABEL_47;
        }
      }
      v359 = (__m128i *)sub_33CB1F0(v172);
      if ( BYTE4(v359) )
      {
        v175 = *(_QWORD *)(v351 + 40) + 40LL * (unsigned int)v359;
        if ( v134[2].m128i_i64[0] != *(_QWORD *)v175 || v134[2].m128i_i32[2] != *(_DWORD *)(v175 + 8) )
          goto LABEL_47;
      }
    }
    else if ( *(_DWORD *)(v351 + 24) != 98 )
    {
      goto LABEL_47;
    }
    if ( (v364[0] || (*(_BYTE *)(v351 + 29) & 2) != 0)
      && (v356 || (v135 = *(_QWORD *)(v351 + 56)) != 0 && !*(_QWORD *)(v135 + 32)) )
    {
      v335 = v361;
      v340 = v360;
      v136 = sub_33CB7C0(244);
      v137 = _mm_loadu_si128(&v368);
      *((_QWORD *)&v313 + 1) = 3;
      v138 = _mm_loadu_si128(&v369);
      *(_QWORD *)&v313 = &v371;
      v371 = (unsigned __int8 *)v347;
      v373 = v137;
      v374 = v138;
      LODWORD(v372) = v6.m128i_i32[2];
      v325 = sub_33FC220(v367.m128i_i32[0], v136, (unsigned int)&v362, v340, v335, v139, v313);
      v341 = v360;
      v140 = *(_QWORD *)(v351 + 40);
      v327 = v141;
      v336 = v361;
      v142 = *(_DWORD *)(v140 + 48);
      v329 = *(unsigned __int8 **)v140;
      v331 = *(_DWORD *)(v140 + 8);
      v333 = *(_QWORD *)(v140 + 40);
      v143 = sub_33CB7C0(v357);
      v373.m128i_i32[2] = v142;
      v374.m128i_i32[2] = v327;
      v144 = _mm_loadu_si128(&v368);
      *((_QWORD *)&v301 + 1) = 5;
      v145 = _mm_loadu_si128(&v369);
      *(_QWORD *)&v301 = &v371;
      v371 = v329;
      LODWORD(v372) = v331;
      v373.m128i_i64[0] = v333;
      v374.m128i_i64[0] = v325;
      v375 = v144;
      v376 = v145;
      v21 = sub_33FC220(v367.m128i_i32[0], v143, (unsigned int)&v362, v341, v336, v325, v301);
      if ( v21 )
        goto LABEL_10;
    }
LABEL_47:
    if ( (unsigned __int8)sub_325F380((__int64)&v367, v351, 244)
      && (unsigned __int8)sub_32653B0(v364, **(_QWORD **)(v351 + 40)) )
    {
      if ( v356 )
      {
        v176 = *(_DWORD **)(v351 + 40);
LABEL_165:
        v177 = *(__int64 **)(*(_QWORD *)v176 + 40LL);
        v178 = *v177;
        v179 = v177[1];
        *((_QWORD *)&v316 + 1) = v6.m128i_i64[1];
        v353 = *(_OWORD *)(v177 + 5);
        *(_QWORD *)&v316 = v347;
        v180 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v42, v316);
        *((_QWORD *)&v304 + 1) = v179;
        *(_QWORD *)&v304 = v178;
        v182 = v181;
        v183 = v180;
        *(_QWORD *)&v185 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v184, v304);
        *((_QWORD *)&v317 + 1) = v182;
        *(_QWORD *)&v317 = v183;
        v187 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v186, v185, v353, v317);
LABEL_166:
        v21 = v187;
        goto LABEL_10;
      }
      v43 = *(_QWORD *)(v351 + 56);
      if ( v43 )
      {
        if ( !*(_QWORD *)(v43 + 32) )
        {
          v176 = *(_DWORD **)(v351 + 40);
          if ( (unsigned __int8)sub_3286E00(v176) )
            goto LABEL_165;
        }
      }
    }
    if ( !(unsigned __int8)sub_325F380((__int64)&v367, v351, 233) )
      goto LABEL_55;
    v44 = *(_QWORD *)(v351 + 40);
    v45 = *(_QWORD *)v44;
    v46 = *(_DWORD *)(v44 + 8);
    if ( !(unsigned __int8)sub_32653B0(v364, *(_QWORD *)v44) )
      goto LABEL_55;
    v48 = (__int64)a1[1];
    v49 = *(_QWORD *)(v45 + 48) + 16LL * v46;
    v50 = *(_WORD *)v49;
    v51 = *(_QWORD *)(v49 + 8);
    v52 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v48 + 1576LL);
    if ( v52 == sub_2FE39B0 )
    {
      v53 = v50;
      v54 = *(__int64 (**)())(*(_QWORD *)v48 + 1560LL);
      if ( v54 == sub_2D566A0 )
        goto LABEL_55;
      v188 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v54)(v48, v360, v361, v53, v51);
    }
    else
    {
      v188 = v52(v48, (__int64)*a1, v357, (unsigned int)v360, v361, v47, v50, v51);
    }
    if ( v188 )
    {
      *((_QWORD *)&v318 + 1) = v6.m128i_i64[1];
      *(_QWORD *)&v318 = v347;
      *(_QWORD *)&v190 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v189, v318);
      v354 = v190;
      v192 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v191, *(_OWORD *)(*(_QWORD *)(v45 + 40) + 40LL));
      v194 = v193;
      *(_QWORD *)&v196 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v195, *(_OWORD *)*(_QWORD *)(v45 + 40));
      v305 = v354;
      goto LABEL_173;
    }
LABEL_55:
    if ( !(unsigned __int8)sub_325F380((__int64)&v367, v347, 233)
      || (v55 = *(_QWORD *)(v347 + 40),
          v56 = *(_QWORD *)v55,
          v57 = *(_DWORD *)(v55 + 8),
          !(unsigned __int8)sub_32653B0(v364, *(_QWORD *)v55)) )
    {
LABEL_59:
      if ( (unsigned __int8)sub_325F380((__int64)&v367, v351, 233) )
      {
        v64 = *(_QWORD *)(v351 + 40);
        v65 = *(_QWORD *)v64;
        v337 = *(_DWORD *)(v64 + 8);
        if ( (unsigned __int8)sub_325F380((__int64)&v367, *(_QWORD *)v64, 244) )
        {
          v162 = **(_QWORD **)(v65 + 40);
          if ( (unsigned __int8)sub_32653B0(v364, v162) )
          {
            if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD))(*a1[1] + 1576))(
                   a1[1],
                   *a1,
                   v357,
                   (unsigned int)v360,
                   v361,
                   *(_QWORD *)(*a1[1] + 1576),
                   *(unsigned __int16 *)(*(_QWORD *)(v65 + 48) + 16LL * v337),
                   *(_QWORD *)(*(_QWORD *)(v65 + 48) + 16LL * v337 + 8)) )
            {
              v163 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v323, *(_OWORD *)(*(_QWORD *)(v162 + 40) + 40LL));
              v165 = v164;
              *(_QWORD *)&v167 = sub_328FBA0(
                                   &v367,
                                   0xE9u,
                                   (int)&v362,
                                   v360,
                                   v361,
                                   v166,
                                   *(_OWORD *)*(_QWORD *)(v162 + 40));
              *((_QWORD *)&v315 + 1) = v6.m128i_i64[1];
              *(_QWORD *)&v315 = v347;
              *((_QWORD *)&v303 + 1) = v165;
              *(_QWORD *)&v303 = v163;
              *(_QWORD *)&v169 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v168, v167, v303, v315);
              v21 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v170, v169);
              goto LABEL_10;
            }
          }
        }
      }
      if ( (unsigned __int8)sub_325F380((__int64)&v367, v351, 244) )
      {
        v146 = **(_QWORD **)(v351 + 40);
        if ( (unsigned __int8)sub_325F380((__int64)&v367, v146, 233) )
        {
          v147 = *(_QWORD *)(v146 + 40);
          v148 = *(_QWORD *)v147;
          v149 = *(_DWORD *)(v147 + 8);
          if ( (unsigned __int8)sub_32653B0(v364, *(_QWORD *)v147) )
          {
            v151 = (unsigned __int16 *)(*(_QWORD *)(v148 + 48) + 16LL * v149);
            if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, __int64, __int64, _QWORD, _QWORD))(*a1[1] + 1576))(
                   a1[1],
                   *a1,
                   v357,
                   (unsigned int)v360,
                   v361,
                   v150,
                   *v151,
                   *((_QWORD *)v151 + 1)) )
            {
              v153 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v152, *(_OWORD *)(*(_QWORD *)(v148 + 40) + 40LL));
              v155 = v154;
              *(_QWORD *)&v157 = sub_328FBA0(
                                   &v367,
                                   0xE9u,
                                   (int)&v362,
                                   v360,
                                   v361,
                                   v156,
                                   *(_OWORD *)*(_QWORD *)(v148 + 40));
              *((_QWORD *)&v314 + 1) = v6.m128i_i64[1];
              *(_QWORD *)&v314 = v347;
              *((_QWORD *)&v302 + 1) = v155;
              *(_QWORD *)&v302 = v153;
              *(_QWORD *)&v159 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v158, v157, v302, v314);
              v21 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v160, v159);
              goto LABEL_10;
            }
          }
        }
      }
      v358 = v348 + 856;
      v371 = v364;
      v372 = &v358;
      v359 = &v367;
      if ( v356 )
      {
        if ( (*(_BYTE *)(v348 + 864) & 1) != 0 )
          goto LABEL_208;
        v161 = *(_DWORD *)(a2 + 28);
        if ( (v161 & 0x800) == 0 )
          goto LABEL_9;
        if ( (v161 & 0x200) != 0 )
        {
LABEL_208:
          if ( (unsigned __int8)sub_326A1D0((__int64 *)&v359, v351) )
          {
            if ( (unsigned __int8)sub_32660C0((_QWORD **)&v371, *(_QWORD *)(*(_QWORD *)(v351 + 40) + 80LL)) )
            {
              v199 = *(_QWORD *)(v351 + 56);
              if ( v199 )
              {
                if ( !*(_QWORD *)(v199 + 32) )
                {
                  v200 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v351 + 40) + 80LL) + 56LL);
                  if ( v200 )
                  {
                    if ( !*(_QWORD *)(v200 + 32) )
                    {
                      *((_QWORD *)&v319 + 1) = v6.m128i_i64[1];
                      v201 = (unsigned int)&v362;
                      *(_QWORD *)&v319 = v347;
                      v202 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v198, v319);
                      v204 = v203;
                      v205 = *(__int128 **)(*(_QWORD *)(*(_QWORD *)(v351 + 40) + 80LL) + 40LL);
                      *((_QWORD *)&v306 + 1) = v204;
                      *(_QWORD *)&v306 = v202;
                      v207 = sub_3290460(
                               &v367,
                               v357,
                               (int)&v362,
                               v360,
                               v361,
                               v206,
                               *v205,
                               *(__int128 *)((char *)v205 + 40),
                               v306);
                      *((_QWORD *)&v211 + 1) = v209;
                      v210 = *(__int128 **)(v351 + 40);
                      *(_QWORD *)&v211 = v207;
LABEL_180:
                      v21 = sub_3290460(
                              &v367,
                              v357,
                              v201,
                              v360,
                              v361,
                              v208,
                              *v210,
                              *(__int128 *)((char *)v210 + 40),
                              v211);
                      goto LABEL_10;
                    }
                  }
                }
              }
            }
          }
          if ( (unsigned __int8)sub_326A1D0((__int64 *)&v359, v347) )
          {
            if ( (unsigned __int8)sub_32660C0((_QWORD **)&v371, *(_QWORD *)(*(_QWORD *)(v347 + 40) + 80LL)) )
            {
              v236 = *(_QWORD *)(v347 + 56);
              if ( v236 )
              {
                if ( !*(_QWORD *)(v236 + 32) && v346 )
                {
                  v237 = (unsigned int)&v362;
                  v238 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v347 + 40) + 80LL) + 40LL);
                  v239 = *(_QWORD *)(v238 + 40);
                  v240 = *(_QWORD *)(v238 + 48);
                  *(_QWORD *)&v241 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v235, *(_OWORD *)v238);
                  *((_QWORD *)&v295 + 1) = v240;
                  *(_QWORD *)&v295 = v239;
                  v243 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v242, v241, v295, *(_OWORD *)&v5);
                  v245 = *(__int128 **)(v347 + 40);
                  v246 = v243;
                  v248 = v247;
LABEL_196:
                  *(_QWORD *)&v249 = sub_328FBA0(&v367, 0xF4u, v237, v360, v361, v244, *v245);
                  *((_QWORD *)&v308 + 1) = v248;
                  *(_QWORD *)&v308 = v246;
                  v21 = sub_3290460(&v367, v357, v237, v360, v361, v250, v249, *(__int128 *)((char *)v245 + 40), v308);
                  goto LABEL_10;
                }
              }
            }
          }
        }
        if ( (unsigned __int8)sub_326A1D0((__int64 *)&v359, v351) )
        {
          v66 = *(_QWORD *)(v351 + 56);
          if ( v66 )
          {
            if ( !*(_QWORD *)(v66 + 32) )
            {
              v212 = *(_QWORD *)(*(_QWORD *)(v351 + 40) + 80LL);
              if ( (unsigned __int8)sub_325F380((__int64)&v367, v212, 233) )
              {
                v213 = *(_QWORD *)(v212 + 40);
                v214 = *(_QWORD *)v213;
                v215 = *(_DWORD *)(v213 + 8);
                if ( (unsigned __int8)sub_32660C0((_QWORD **)&v371, *(_QWORD *)v213) )
                {
                  v216 = (unsigned __int16 *)(*(_QWORD *)(v214 + 48) + 16LL * v215);
                  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD))(*a1[1] + 1576))(
                         a1[1],
                         *a1,
                         v357,
                         (unsigned int)v360,
                         v361,
                         *(_QWORD *)(*a1[1] + 1576),
                         *v216,
                         *((_QWORD *)v216 + 1)) )
                  {
                    *((_QWORD *)&v320 + 1) = v6.m128i_i64[1];
                    *(_QWORD *)&v320 = v347;
                    v218 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v217, v320);
                    v220 = v219;
                    v222 = sub_328FBA0(
                             &v367,
                             0xE9u,
                             (int)&v362,
                             v360,
                             v361,
                             v221,
                             *(_OWORD *)(*(_QWORD *)(v214 + 40) + 40LL));
                    v224 = v223;
                    *(_QWORD *)&v226 = sub_328FBA0(
                                         &v367,
                                         0xE9u,
                                         (int)&v362,
                                         v360,
                                         v361,
                                         v225,
                                         *(_OWORD *)*(_QWORD *)(v214 + 40));
                    *((_QWORD *)&v307 + 1) = v220;
                    v201 = (unsigned int)&v362;
                    *(_QWORD *)&v307 = v218;
                    *((_QWORD *)&v294 + 1) = v224;
                    *(_QWORD *)&v294 = v222;
                    *(_QWORD *)&v211 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v227, v226, v294, v307);
                    *((_QWORD *)&v211 + 1) = v228;
                    v210 = *(__int128 **)(v351 + 40);
                    goto LABEL_180;
                  }
                }
              }
            }
          }
        }
        if ( (unsigned __int8)sub_325F380((__int64)&v367, v351, 233) )
        {
          v67 = *(_QWORD *)(v351 + 40);
          v68 = *(_QWORD *)v67;
          v69 = *(_DWORD *)(v67 + 8);
          if ( (unsigned __int8)sub_326A1D0((__int64 *)&v359, *(_QWORD *)v67) )
          {
            v355 = *(_QWORD *)(*(_QWORD *)(v68 + 40) + 80LL);
            if ( (unsigned __int8)sub_32660C0((_QWORD **)&v371, v355) )
            {
              v251 = (unsigned __int16 *)(*(_QWORD *)(v68 + 48) + 16LL * v69);
              v252 = (*(__int64 (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD))(*a1[1] + 1576))(
                       a1[1],
                       *a1,
                       v357,
                       (unsigned int)v360,
                       v361,
                       *(_QWORD *)(*a1[1] + 1576),
                       *v251,
                       *((_QWORD *)v251 + 1));
              v253 = v324;
              if ( v252 )
              {
                *((_QWORD *)&v321 + 1) = v6.m128i_i64[1];
                *(_QWORD *)&v321 = v347;
                v254 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v253, v321);
                v256 = v255;
                v258 = sub_328FBA0(
                         &v367,
                         0xE9u,
                         (int)&v362,
                         v360,
                         v361,
                         v257,
                         *(_OWORD *)(*(_QWORD *)(v355 + 40) + 40LL));
                v260 = v259;
                *(_QWORD *)&v262 = sub_328FBA0(
                                     &v367,
                                     0xE9u,
                                     (int)&v362,
                                     v360,
                                     v361,
                                     v261,
                                     *(_OWORD *)*(_QWORD *)(v355 + 40));
                *((_QWORD *)&v309 + 1) = v256;
                *(_QWORD *)&v309 = v254;
                *((_QWORD *)&v296 + 1) = v260;
                *(_QWORD *)&v296 = v258;
                v264 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v263, v262, v296, v309);
                v266 = v265;
                v268 = sub_328FBA0(
                         &v367,
                         0xE9u,
                         (int)&v362,
                         v360,
                         v361,
                         v267,
                         *(_OWORD *)(*(_QWORD *)(v68 + 40) + 40LL));
                v270 = v269;
                *(_QWORD *)&v272 = sub_328FBA0(
                                     &v367,
                                     0xE9u,
                                     (int)&v362,
                                     v360,
                                     v361,
                                     v271,
                                     *(_OWORD *)*(_QWORD *)(v68 + 40));
                *((_QWORD *)&v322 + 1) = v266;
                *(_QWORD *)&v322 = v264;
                *((_QWORD *)&v310 + 1) = v270;
                *(_QWORD *)&v310 = v268;
                v187 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v273, v272, v310, v322);
                goto LABEL_166;
              }
            }
          }
        }
        if ( (unsigned __int8)sub_326A1D0((__int64 *)&v359, v347) )
        {
          if ( (unsigned __int8)sub_325F380((__int64)&v367, *(_QWORD *)(*(_QWORD *)(v347 + 40) + 80LL), 233) )
          {
            v70 = *(_QWORD *)(v347 + 56);
            if ( v70 )
            {
              if ( !*(_QWORD *)(v70 + 32) )
              {
                v274 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v347 + 40) + 80LL) + 40LL);
                v275 = *(_QWORD *)v274;
                v276 = *(_DWORD *)(v274 + 8);
                if ( (unsigned __int8)sub_32660C0((_QWORD **)&v371, *(_QWORD *)v274) )
                {
                  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD))(*a1[1] + 1576))(
                         a1[1],
                         *a1,
                         v357,
                         (unsigned int)v360,
                         v361,
                         *(_QWORD *)(*a1[1] + 1576),
                         *(unsigned __int16 *)(*(_QWORD *)(v275 + 48) + 16LL * v276),
                         *(_QWORD *)(*(_QWORD *)(v275 + 48) + 16LL * v276 + 8)) )
                  {
                    v278 = *(__int64 **)(v275 + 40);
                    v279 = *v278;
                    v280 = v278[1];
                    v281 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v277, *(_OWORD *)(v278 + 5));
                    *((_QWORD *)&v311 + 1) = v280;
                    v237 = (unsigned int)&v362;
                    *(_QWORD *)&v311 = v279;
                    v283 = v282;
                    v284 = v281;
                    *(_QWORD *)&v286 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v285, v311);
                    *(_QWORD *)&v288 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v287, v286);
                    *((_QWORD *)&v297 + 1) = v283;
                    *(_QWORD *)&v297 = v284;
                    v246 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v289, v288, v297, *(_OWORD *)&v5);
                    v248 = v290;
                    v245 = *(__int128 **)(v347 + 40);
                    goto LABEL_196;
                  }
                }
              }
            }
          }
        }
        if ( (unsigned __int8)sub_325F380((__int64)&v367, v347, 233) )
        {
          if ( (unsigned __int8)sub_326A1D0((__int64 *)&v359, **(_QWORD **)(v347 + 40)) )
          {
            v71 = *(_DWORD **)(v347 + 40);
            v72 = *(_QWORD *)v71;
            v73 = v71[2];
            v74 = *(_QWORD *)(*(_QWORD *)v71 + 40LL);
            v75 = *(_QWORD *)(v74 + 80);
            v352 = (__int128)_mm_loadu_si128((const __m128i *)v74);
            v350 = (__int128)_mm_loadu_si128((const __m128i *)(v74 + 40));
            if ( (unsigned __int8)sub_32660C0((_QWORD **)&v371, v75) )
            {
              v76 = (unsigned __int16 *)(*(_QWORD *)(v72 + 48) + 16LL * v73);
              if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD))(*a1[1] + 1576))(
                     a1[1],
                     *a1,
                     v357,
                     (unsigned int)v360,
                     v361,
                     *(_QWORD *)(*a1[1] + 1576),
                     *v76,
                     *((_QWORD *)v76 + 1)) )
              {
                v78 = *(__int64 **)(v75 + 40);
                v79 = *v78;
                v80 = v78[1];
                v81 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v77, *(_OWORD *)(v78 + 5));
                *((_QWORD *)&v298 + 1) = v80;
                *(_QWORD *)&v298 = v79;
                v83 = v82;
                v84 = v81;
                *(_QWORD *)&v86 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v85, v298);
                *(_QWORD *)&v88 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v87, v86);
                *((_QWORD *)&v291 + 1) = v83;
                *(_QWORD *)&v291 = v84;
                v90 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v89, v88, v291, *(_OWORD *)&v5);
                v92 = v91;
                v93 = v90;
                v95 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v94, v350);
                v97 = v96;
                v98 = v95;
                *(_QWORD *)&v100 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v99, v352);
                *(_QWORD *)&v102 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v101, v100);
                *((_QWORD *)&v299 + 1) = v92;
                *(_QWORD *)&v299 = v93;
                *((_QWORD *)&v292 + 1) = v97;
                *(_QWORD *)&v292 = v98;
                v104 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v103, v102, v292, v299);
LABEL_78:
                v21 = v104;
                goto LABEL_10;
              }
            }
          }
        }
      }
LABEL_9:
      v21 = 0;
      goto LABEL_10;
    }
    v58 = (__int64)a1[1];
    v59 = *(_QWORD *)(v56 + 48) + 16LL * v57;
    v60 = *(_QWORD *)v58;
    v61 = *(_WORD *)v59;
    v62 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v58 + 1576LL);
    if ( v62 == sub_2FE39B0 )
    {
      v63 = *(__int64 (**)())(v60 + 1560);
      if ( v63 == sub_2D566A0 )
        goto LABEL_59;
      v229 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, _QWORD))v63)(
               v58,
               v360,
               v361,
               v61,
               *(_QWORD *)(v59 + 8));
    }
    else
    {
      v229 = v62(v58, (__int64)*a1, v357, (unsigned int)v360, v361, v60, v61, *(_QWORD *)(v59 + 8));
    }
    if ( !v229 )
      goto LABEL_59;
    v192 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v230, *(_OWORD *)(*(_QWORD *)(v56 + 40) + 40LL));
    v194 = v231;
    *(_QWORD *)&v233 = sub_328FBA0(&v367, 0xE9u, (int)&v362, v360, v361, v232, *(_OWORD *)*(_QWORD *)(v56 + 40));
    *(_QWORD *)&v196 = sub_328FBA0(&v367, 0xF4u, (int)&v362, v360, v361, v234, v233);
    v305 = (__int128)v5;
LABEL_173:
    *((_QWORD *)&v293 + 1) = v194;
    *(_QWORD *)&v293 = v192;
    v104 = sub_3290460(&v367, v357, (int)&v362, v360, v361, v197, v196, v293, v305);
    goto LABEL_78;
  }
LABEL_10:
  if ( v362 )
    sub_B91220((__int64)&v362, v362);
  return v21;
}
