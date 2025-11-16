// Function: sub_32C7250
// Address: 0x32c7250
//
__int64 __fastcall sub_32C7250(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __int128 a7,
        __int128 a8,
        unsigned int a9,
        char a10)
{
  __int64 v11; // r13
  __int64 v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 *v16; // rdi
  __int64 (__fastcall *v17)(__int64, __int64, __int64, __int64, __int64); // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  __int64 v29; // rdi
  unsigned int v30; // ebx
  bool v31; // al
  __int64 result; // rax
  __int64 v33; // rdi
  __int64 (*v34)(); // rcx
  int v35; // eax
  int v36; // edx
  unsigned __int16 *v37; // rcx
  unsigned __int16 *v38; // rax
  unsigned __int64 v39; // rsi
  __int64 v40; // rdx
  __int16 v41; // cx
  __int64 v42; // rdx
  unsigned __int64 v43; // r13
  unsigned __int16 v44; // bx
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rax
  char v48; // r12
  char v49; // bl
  _DWORD *v50; // r12
  unsigned __int16 v51; // cx
  int v52; // eax
  __int64 v53; // rax
  unsigned int v54; // ebx
  bool v55; // al
  __int64 v56; // rax
  unsigned int v57; // ebx
  unsigned __int64 v58; // r12
  unsigned __int64 v59; // r12
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // r12
  __int64 v62; // rax
  __int64 v63; // r13
  bool v64; // al
  unsigned int v65; // r14d
  bool v66; // dl
  __int64 v67; // r12
  int v68; // eax
  __int64 v69; // r13
  __int128 v70; // rax
  int v71; // r9d
  __int64 v72; // rax
  __int64 v73; // r14
  __int128 *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r13
  __int64 v80; // rdx
  int v81; // ebx
  __int64 v82; // r12
  __int128 v83; // rax
  int v84; // r9d
  __int64 v85; // rdx
  __int64 v86; // rdi
  __int64 v87; // r9
  __int64 v88; // r8
  __int64 v89; // r10
  __int64 (*v90)(); // rax
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rax
  int v98; // edx
  __int64 v99; // rbx
  int v100; // eax
  __int64 v101; // r13
  unsigned int v102; // eax
  __int64 v103; // r12
  __int64 v104; // rdx
  __int64 v105; // r13
  __int64 v106; // rbx
  __int64 v107; // rsi
  int v108; // r9d
  __int64 v109; // r12
  __int64 v110; // rdx
  __int64 v111; // r13
  __int64 v112; // rbx
  __int64 v113; // rdx
  __int64 v114; // rbx
  int v115; // r9d
  __int64 v116; // r12
  __int64 v117; // rdx
  __int64 v118; // r13
  int v119; // r9d
  __int64 v120; // r13
  int v121; // r14d
  unsigned __int64 v122; // rbx
  __int64 v123; // rcx
  unsigned __int16 v124; // cx
  unsigned __int64 v125; // rsi
  bool v126; // zf
  unsigned __int16 v127; // ax
  __int64 v128; // rax
  unsigned int v129; // ecx
  __int64 v130; // rax
  unsigned __int64 v131; // rdx
  __int64 v132; // rdx
  __int64 v133; // rax
  unsigned int v134; // edx
  unsigned __int64 v135; // rax
  unsigned __int64 v136; // rcx
  int v137; // ecx
  int v138; // eax
  int v139; // eax
  __int64 v140; // rdi
  unsigned int v141; // ecx
  __int64 (*v142)(); // rax
  __int128 v143; // rax
  __int64 v144; // rdi
  __m128i v145; // rax
  __int64 v146; // r8
  __int64 v147; // r9
  char v148; // al
  int v149; // r9d
  __int64 v150; // r10
  __int64 v151; // r8
  __int64 v152; // rdi
  __int64 v153; // rax
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rdx
  char v157; // r8
  __int64 v158; // rax
  __int64 v159; // rcx
  __int64 v160; // rdx
  __int64 v161; // rax
  unsigned __int64 v162; // rdx
  char v163; // r13
  __int64 v164; // rdi
  __int64 (__fastcall *v165)(__int64); // rcx
  __int64 (*v166)(); // rax
  char v167; // al
  unsigned __int64 v168; // rsi
  char v169; // al
  __int64 (*v170)(); // rax
  __int64 v171; // rax
  unsigned int v172; // ebx
  __int64 v173; // rdi
  int v174; // eax
  __m128i v175; // rax
  int v176; // r9d
  int v177; // r13d
  __int64 v178; // rax
  unsigned int v179; // r12d
  int v180; // ebx
  __int64 v181; // rdi
  char v182; // al
  char v183; // r8
  __int64 v184; // rdx
  int v185; // eax
  __int64 v186; // rdi
  unsigned int v187; // ecx
  __int64 (*v188)(); // rax
  __int128 v189; // rax
  __int64 v190; // rdi
  __m128i v191; // rax
  __int64 v192; // r8
  __int64 v193; // r9
  char v194; // al
  __int64 v195; // rdi
  __int64 v196; // rdx
  unsigned int v197; // eax
  __int64 v198; // rcx
  unsigned __int64 v199; // rax
  __int64 *v200; // rbx
  __int64 v201; // r12
  __int64 **v202; // rax
  int v203; // r13d
  int v204; // ebx
  __int64 v205; // rax
  int v206; // eax
  unsigned __int16 v207; // ax
  __m128i v208; // rax
  __int64 v209; // rdi
  __int64 v210; // rdx
  __int64 v211; // r13
  char v212; // bl
  __int64 v213; // rax
  __int64 v214; // rdx
  unsigned int v215; // eax
  __int64 v216; // r12
  unsigned int v217; // ebx
  __int64 v218; // rdx
  __int64 v219; // rbx
  int v220; // eax
  int v221; // edx
  __int128 v222; // rax
  __int64 v223; // r12
  __int64 v224; // rdx
  __int64 v225; // rbx
  __int64 v226; // r8
  __int64 v227; // r9
  unsigned __int16 *v228; // rax
  __int64 v229; // r12
  __int64 v230; // rdx
  __int64 v231; // r13
  __int64 v232; // r8
  __int64 v233; // r9
  unsigned __int16 *v234; // rax
  int v235; // r9d
  __int64 v236; // r13
  int v237; // edx
  __int64 v238; // r8
  __int64 v239; // r9
  __int64 v240; // r12
  __int16 v241; // ax
  __int16 v242; // r14
  __int64 v243; // rax
  __int64 v244; // rdx
  int v245; // esi
  __int64 v246; // r12
  __int64 v247; // rdx
  unsigned __int64 v248; // r12
  __int64 v249; // r13
  __int64 v250; // rax
  unsigned int v251; // edx
  __int64 v252; // rbx
  __int64 v253; // r13
  __int64 v254; // r8
  __int64 v255; // r9
  __int64 v256; // r8
  __int64 v257; // r9
  __int64 v258; // rax
  unsigned int v259; // r14d
  __int64 v260; // rdi
  unsigned int v261; // r14d
  int v262; // eax
  __int64 v263; // rax
  char v264; // cl
  unsigned __int64 v265; // rax
  unsigned int v266; // ecx
  char v267; // di
  __int64 v268; // r13
  __int64 (__fastcall *v269)(__int64, _QWORD, __int64, _QWORD); // r14
  int v270; // eax
  __int64 v271; // rax
  unsigned int v272; // r12d
  int v273; // ebx
  int v274; // eax
  __int64 *v275; // rdi
  unsigned int v276; // r14d
  __int64 v277; // rax
  __int64 v278; // r15
  unsigned __int64 v279; // r13
  __int128 v280; // rax
  int v281; // r9d
  __int64 v282; // rdx
  __int64 v283; // rdx
  __int64 v284; // r12
  int v285; // r9d
  unsigned int v286; // edx
  __int64 v287; // rax
  char v288; // cl
  unsigned __int64 v289; // rax
  char v290; // al
  __int64 v291; // rdx
  __int64 v292; // r9
  __int64 v293; // rdx
  __int64 v294; // r9
  __int64 v295; // rdi
  __int64 v296; // rdx
  __int64 v297; // rdi
  __int128 v298; // [rsp-30h] [rbp-290h]
  __int128 v299; // [rsp-20h] [rbp-280h]
  __int128 v300; // [rsp-20h] [rbp-280h]
  __int128 v301; // [rsp-20h] [rbp-280h]
  __int128 v302; // [rsp-20h] [rbp-280h]
  __int128 v303; // [rsp-20h] [rbp-280h]
  __int128 v304; // [rsp-10h] [rbp-270h]
  __int128 v305; // [rsp-10h] [rbp-270h]
  __int128 v306; // [rsp-10h] [rbp-270h]
  __int128 v307; // [rsp-10h] [rbp-270h]
  __int64 v308; // [rsp-10h] [rbp-270h]
  __int128 v309; // [rsp+0h] [rbp-260h]
  __int128 v310; // [rsp+10h] [rbp-250h]
  __int64 v311; // [rsp+20h] [rbp-240h]
  unsigned __int32 v312; // [rsp+2Ch] [rbp-234h]
  int v313; // [rsp+38h] [rbp-228h]
  char v314; // [rsp+40h] [rbp-220h]
  __int64 (__fastcall *v315)(__int64, __int64, unsigned int); // [rsp+40h] [rbp-220h]
  __int64 v316; // [rsp+50h] [rbp-210h]
  int v317; // [rsp+50h] [rbp-210h]
  __int64 v318; // [rsp+58h] [rbp-208h]
  __int64 v319; // [rsp+60h] [rbp-200h]
  __int64 v320; // [rsp+68h] [rbp-1F8h]
  __int64 v321; // [rsp+70h] [rbp-1F0h]
  __int64 v322; // [rsp+78h] [rbp-1E8h]
  unsigned int v323; // [rsp+88h] [rbp-1D8h]
  __int64 v324; // [rsp+90h] [rbp-1D0h]
  char v325; // [rsp+9Bh] [rbp-1C5h]
  unsigned int v326; // [rsp+9Ch] [rbp-1C4h]
  __int64 v327; // [rsp+A0h] [rbp-1C0h]
  __int64 v328; // [rsp+A0h] [rbp-1C0h]
  unsigned __int16 v329; // [rsp+A0h] [rbp-1C0h]
  __int64 v330; // [rsp+A0h] [rbp-1C0h]
  _QWORD *v331; // [rsp+A0h] [rbp-1C0h]
  unsigned int v332; // [rsp+A0h] [rbp-1C0h]
  __int64 v333; // [rsp+A0h] [rbp-1C0h]
  __int64 v334; // [rsp+A0h] [rbp-1C0h]
  char v335; // [rsp+A8h] [rbp-1B8h]
  char v336; // [rsp+AFh] [rbp-1B1h]
  unsigned __int128 v337; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i si128; // [rsp+C0h] [rbp-1A0h]
  __int64 v339; // [rsp+D0h] [rbp-190h]
  unsigned __int64 v340; // [rsp+D8h] [rbp-188h]
  __int128 v341; // [rsp+E0h] [rbp-180h]
  __int64 v342; // [rsp+F0h] [rbp-170h]
  __int64 v343; // [rsp+F8h] [rbp-168h]
  __int64 v344; // [rsp+100h] [rbp-160h]
  __int64 v345; // [rsp+108h] [rbp-158h]
  __int128 v346; // [rsp+110h] [rbp-150h]
  __int64 v347; // [rsp+120h] [rbp-140h]
  __int64 v348; // [rsp+128h] [rbp-138h]
  __int64 v349; // [rsp+130h] [rbp-130h]
  __int64 v350; // [rsp+138h] [rbp-128h]
  __int64 v351; // [rsp+140h] [rbp-120h]
  __int64 v352; // [rsp+148h] [rbp-118h]
  __int64 v353; // [rsp+150h] [rbp-110h]
  __int64 v354; // [rsp+158h] [rbp-108h]
  __int64 v355; // [rsp+160h] [rbp-100h]
  __int64 v356; // [rsp+168h] [rbp-F8h]
  __int64 v357; // [rsp+170h] [rbp-F0h]
  __int64 v358; // [rsp+178h] [rbp-E8h]
  __m128i v359; // [rsp+180h] [rbp-E0h] BYREF
  unsigned int v360; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v361; // [rsp+198h] [rbp-C8h]
  unsigned __int64 v362; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v363; // [rsp+1A8h] [rbp-B8h]
  __int64 v364; // [rsp+1B0h] [rbp-B0h]
  __int64 v365; // [rsp+1B8h] [rbp-A8h]
  __int64 v366; // [rsp+1C0h] [rbp-A0h]
  __int64 v367; // [rsp+1C8h] [rbp-98h]
  __int64 v368; // [rsp+1D0h] [rbp-90h] BYREF
  __int64 v369; // [rsp+1D8h] [rbp-88h]
  unsigned __int64 v370; // [rsp+1E0h] [rbp-80h] BYREF
  __int64 v371; // [rsp+1E8h] [rbp-78h]
  __int128 v372; // [rsp+1F0h] [rbp-70h] BYREF
  __int64 v373; // [rsp+200h] [rbp-60h]
  __m128i v374; // [rsp+210h] [rbp-50h] BYREF
  __int64 v375; // [rsp+220h] [rbp-40h]
  __int64 v376; // [rsp+228h] [rbp-38h]

  v344 = a2;
  v337 = __PAIR128__(a4, a3);
  v335 = a10;
  v336 = a10;
  v339 = a5;
  v340 = a6;
  v345 = a3;
  v326 = a4;
  v343 = a5;
  *(_QWORD *)&v346 = a7;
  *(_QWORD *)&v341 = a8;
  LODWORD(v342) = DWORD2(a7);
  if ( (_QWORD)a8 == (_QWORD)a7 && DWORD2(a8) == DWORD2(a7) )
    return a8;
  v11 = a1[1];
  v327 = 16LL * (unsigned int)a4;
  v12 = *(_QWORD *)(v345 + 48) + v327;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v359.m128i_i16[0] = v13;
  v359.m128i_i64[1] = v14;
  v15 = v14;
  v16 = *(__int64 **)(*a1 + 40);
  v17 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v11 + 528LL);
  si128.m128i_i64[0] = *(_QWORD *)(*a1 + 64);
  v18 = sub_2E79000(v16);
  v323 = v17(v11, v18, si128.m128i_i64[0], v359.m128i_i64[0], v15);
  v324 = v19;
  v318 = (unsigned int)v342;
  v322 = 16LL * (unsigned int)v342;
  v20 = *(_QWORD *)(v346 + 48) + v322;
  LOWORD(v19) = *(_WORD *)v20;
  v361 = *(_QWORD *)(v20 + 8);
  LOWORD(v360) = v19;
  v21 = *(_DWORD *)(v343 + 24);
  if ( v21 == 35 || (v319 = 0, v21 == 11) )
    v319 = v343;
  v22 = *(_DWORD *)(v346 + 24);
  if ( v22 == 35 || (v321 = 0, v22 == 11) )
    v321 = v346;
  v23 = *(_DWORD *)(v341 + 24);
  if ( v23 == 35 || (v320 = 0, v23 == 11) )
    v320 = v341;
  v24 = sub_340F940(*a1, v323, v324, v337, DWORD2(v337), a9, v339, v340, v344);
  v25 = v24;
  if ( v24 && *(_DWORD *)(v24 + 24) != 328 )
  {
    v374.m128i_i64[0] = v24;
    sub_32B3B20((__int64)(a1 + 71), v374.m128i_i64);
    if ( *(int *)(v25 + 88) < 0 )
    {
      *(_DWORD *)(v25 + 88) = *((_DWORD *)a1 + 12);
      v95 = *((unsigned int *)a1 + 12);
      if ( v95 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v95 + 1, 8u, v26, v27);
        v95 = *((unsigned int *)a1 + 12);
      }
      *(_QWORD *)(a1[5] + 8 * v95) = v25;
      ++*((_DWORD *)a1 + 12);
    }
    v28 = *(_DWORD *)(v25 + 24);
    if ( v28 == 11 || v28 == 35 )
    {
      v29 = *(_QWORD *)(v25 + 96);
      v30 = *(_DWORD *)(v29 + 32);
      if ( v30 <= 0x40 )
        v31 = *(_QWORD *)(v29 + 24) == 0;
      else
        v31 = v30 == (unsigned int)sub_C444A0(v29 + 24);
      if ( v31 )
        return a8;
      else
        return a7;
    }
  }
  v33 = a1[1];
  v34 = *(__int64 (**)())(*(_QWORD *)v33 + 160LL);
  if ( v34 != sub_2FE2EB0
    && !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v34)(
          v33,
          *(unsigned __int16 *)(*(_QWORD *)(v345 + 48) + v327),
          *(_QWORD *)(*(_QWORD *)(v345 + 48) + v327 + 8)) )
  {
    v38 = (unsigned __int16 *)(*(_QWORD *)(v346 + 48) + v322);
    goto LABEL_25;
  }
  v35 = *(_DWORD *)(v346 + 24);
  v36 = *(_DWORD *)(v341 + 24);
  v37 = *(unsigned __int16 **)(v346 + 48);
  if ( v35 != 36 && v35 != 12 )
  {
    v38 = &v37[(unsigned __int64)v322 / 2];
    goto LABEL_25;
  }
  v38 = &v37[(unsigned __int64)v322 / 2];
  if ( v36 != 12 && v36 != 36 )
    goto LABEL_25;
  v85 = *v38;
  if ( !(_WORD)v85 )
    goto LABEL_25;
  v86 = a1[1];
  if ( !*(_QWORD *)(v86 + 8 * v85 + 112) || !*(_BYTE *)(v86 + 500LL * (unsigned __int16)v85 + 6426) )
    goto LABEL_25;
  v87 = a1[1];
  v88 = *((unsigned __int8 *)a1 + 35);
  v89 = *((_QWORD *)v37 + 1);
  v90 = *(__int64 (**)())(*(_QWORD *)v86 + 616LL);
  v91 = *v37;
  if ( v90 != sub_2FE3170 )
  {
    si128.m128i_i64[0] = *(_QWORD *)(v346 + 96);
    if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))v90)(
            v86,
            si128.m128i_i64[0] + 24,
            v91,
            v89,
            v88,
            v87) )
    {
      v87 = a1[1];
      v88 = *((unsigned __int8 *)a1 + 35);
      v90 = *(__int64 (**)())(*(_QWORD *)v87 + 616LL);
      goto LABEL_82;
    }
    v38 = (unsigned __int16 *)(*(_QWORD *)(v346 + 48) + v322);
LABEL_25:
    v39 = *((_QWORD *)&a8 + 1);
    v40 = *(_QWORD *)(v345 + 48) + v327;
    v41 = *(_WORD *)v40;
    v42 = *(_QWORD *)(v40 + 8);
    si128 = _mm_load_si128((const __m128i *)&v337);
    v43 = *((_QWORD *)&a7 + 1);
    LOWORD(v362) = v41;
    v363 = v42;
    v44 = *v38;
    v45 = *((_QWORD *)v38 + 1);
    if ( !(unsigned __int8)sub_33CF170(a8, *((_QWORD *)&a8 + 1)) )
      goto LABEL_26;
    v124 = v362;
    if ( v44 == (_WORD)v362 )
    {
      if ( v44 || v45 == v363 )
      {
LABEL_112:
        if ( a9 == 18 )
        {
          v164 = a1[1];
          v165 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v164 + 392LL);
          if ( v165 == sub_2FE3CB0 )
          {
            v166 = *(__int64 (**)())(*(_QWORD *)v164 + 384LL);
            v43 = v318 | v43 & 0xFFFFFFFF00000000LL;
            if ( v166 == sub_2FE3020 )
              goto LABEL_27;
            v39 = v346;
            v167 = ((__int64 (__fastcall *)(__int64, _QWORD))v166)(v164, v346);
          }
          else
          {
            v39 = v346;
            v43 = v318 | v43 & 0xFFFFFFFF00000000LL;
            v167 = ((__int64 (__fastcall *)(__int64, _QWORD))v165)(v164, v346);
          }
          if ( !v167 )
            goto LABEL_27;
          v168 = v340;
          if ( (unsigned __int8)sub_33CF460(v339, v340) )
          {
LABEL_115:
            v126 = *(_DWORD *)(v346 + 24) != 11 && *(_DWORD *)(v346 + 24) != 35;
            v127 = v362;
            LOBYTE(v342) = *(_DWORD *)(v346 + 24) == 11 || *(_DWORD *)(v346 + 24) == 35;
            if ( v126 )
            {
LABEL_206:
              if ( !v127 )
              {
                v370 = sub_3007260((__int64)&v362);
                v371 = v184;
LABEL_208:
                v374.m128i_i64[0] = v370;
                v374.m128i_i8[8] = v371;
                v185 = sub_CA1930(&v374);
                v186 = a1[1];
                v187 = v185 - 1;
                v188 = *(__int64 (**)())(*(_QWORD *)v186 + 1728LL);
                if ( v188 != sub_2FE3600 )
                {
                  LODWORD(v342) = v187;
                  v39 = (unsigned int)v362;
                  v290 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v188)(v186, (unsigned int)v362, v363);
                  v187 = v342;
                  if ( v290 )
                    goto LABEL_26;
                }
                *(_QWORD *)&v189 = sub_3400E40(*a1, v187, (unsigned int)v362, v363, v344);
                v190 = *a1;
                si128.m128i_i64[0] = v345;
                si128.m128i_i64[1] = si128.m128i_i64[1] & 0xFFFFFFFF00000000LL | v326;
                v191.m128i_i64[0] = sub_3406EB0(
                                      v190,
                                      191,
                                      v344,
                                      v362,
                                      v363,
                                      DWORD2(v189),
                                      __PAIR128__(si128.m128i_u64[1], v345),
                                      v189);
                si128 = v191;
                v330 = v191.m128i_i64[0];
                sub_32B3E80((__int64)a1, v191.m128i_i64[0], 1, 0, v192, v193);
                v342 = v44;
                v194 = sub_3280A00((__int64)&v362, v44, v45);
                v150 = v342;
                v151 = v330;
                if ( v194 )
                {
                  v333 = v342;
                  v353 = sub_33FAF80(*a1, 216, v344, v342, v45, v149, *(_OWORD *)&si128);
                  v354 = v291;
                  si128.m128i_i64[0] = v353;
                  v342 = v353;
                  si128.m128i_i64[1] = (unsigned int)v291 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                  sub_32B3E80((__int64)a1, v353, 1, 0, v353, v292);
                  v150 = v333;
                  v151 = v342;
                }
                if ( a9 == 18 )
                {
                  si128.m128i_i64[0] = v151;
                  v195 = *a1;
                  v342 = v150;
                  v151 = sub_34074A0(v195, v344, v151, si128.m128i_i64[1], (unsigned int)v150, v45);
                  v352 = v196;
                  v197 = v196;
                  v351 = v151;
LABEL_213:
                  si128.m128i_i64[0] = v151;
                  LODWORD(v150) = v342;
                  si128.m128i_i64[1] = v197 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                  goto LABEL_138;
                }
                goto LABEL_138;
              }
              if ( v127 != 1 && (unsigned __int16)(v127 - 504) > 7u )
              {
                v263 = 16LL * (v127 - 1);
                v264 = byte_444C4A0[v263 + 8];
                v265 = *(_QWORD *)&byte_444C4A0[v263];
                LOBYTE(v371) = v264;
                v370 = v265;
                goto LABEL_208;
              }
LABEL_289:
              BUG();
            }
            v128 = *(_QWORD *)(v346 + 96);
            LODWORD(v369) = *(_DWORD *)(v128 + 32);
            if ( (unsigned int)v369 > 0x40 )
              sub_C43780((__int64)&v368, (const void **)(v128 + 24));
            else
              v368 = *(_QWORD *)(v128 + 24);
            sub_C46F20((__int64)&v368, 1u);
            v129 = v369;
            LODWORD(v369) = 0;
            LODWORD(v371) = v129;
            v370 = v368;
            v130 = *(_QWORD *)(v346 + 96);
            if ( v129 > 0x40 )
            {
              sub_C43B90(&v370, (__int64 *)(v130 + 24));
              v266 = v371;
              v131 = v370;
              LODWORD(v371) = 0;
              v374.m128i_i32[2] = v266;
              v374.m128i_i64[0] = v370;
              if ( v266 > 0x40 )
              {
                v331 = (_QWORD *)v370;
                if ( v266 - (unsigned int)sub_C444A0((__int64)&v374) <= 0x40 )
                {
                  v267 = v342;
                  if ( *v331 )
                    v267 = 0;
                  LOBYTE(v342) = v267;
                }
                else
                {
                  LOBYTE(v342) = 0;
                }
                if ( v374.m128i_i64[0] )
                  j_j___libc_free_0_0(v374.m128i_u64[0]);
LABEL_121:
                if ( (unsigned int)v371 > 0x40 && v370 )
                  j_j___libc_free_0_0(v370);
                if ( (unsigned int)v369 > 0x40 && v368 )
                  j_j___libc_free_0_0(v368);
                v127 = v362;
                if ( (_BYTE)v342 )
                {
                  if ( (_WORD)v362 )
                  {
                    if ( (_WORD)v362 == 1 || (unsigned __int16)(v362 - 504) <= 7u )
                      goto LABEL_289;
                    v287 = 16LL * ((unsigned __int16)v362 - 1);
                    v288 = byte_444C4A0[v287 + 8];
                    v289 = *(_QWORD *)&byte_444C4A0[v287];
                    LOBYTE(v369) = v288;
                    v368 = v289;
                  }
                  else
                  {
                    v368 = sub_3007260((__int64)&v362);
                    v369 = v132;
                  }
                  v374.m128i_i64[0] = v368;
                  v374.m128i_i8[8] = v369;
                  v342 = sub_CA1930(&v374);
                  v133 = *(_QWORD *)(v346 + 96);
                  v134 = *(_DWORD *)(v133 + 32);
                  if ( v134 > 0x40 )
                  {
                    v332 = *(_DWORD *)(v133 + 32);
                    v139 = sub_C444A0(v133 + 24);
                    v134 = v332;
                  }
                  else
                  {
                    v135 = *(_QWORD *)(v133 + 24);
                    _BitScanReverse64(&v136, v135);
                    v137 = v136 ^ 0x3F;
                    v126 = v135 == 0;
                    v138 = 64;
                    if ( !v126 )
                      v138 = v137;
                    v139 = v134 + v138 - 64;
                  }
                  v140 = a1[1];
                  v141 = v139 + v342 - v134;
                  v142 = *(__int64 (**)())(*(_QWORD *)v140 + 1728LL);
                  if ( v142 == sub_2FE3600
                    || (LODWORD(v342) = v141,
                        v182 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v142)(v140, (unsigned int)v362, v363),
                        v141 = v342,
                        v183 = v182,
                        v127 = v362,
                        !v183) )
                  {
                    *(_QWORD *)&v143 = sub_3400E40(*a1, v141, (unsigned int)v362, v363, v344);
                    v144 = *a1;
                    si128.m128i_i64[0] = v345;
                    si128.m128i_i64[1] = si128.m128i_i64[1] & 0xFFFFFFFF00000000LL | v326;
                    v145.m128i_i64[0] = sub_3406EB0(
                                          v144,
                                          192,
                                          v344,
                                          v362,
                                          v363,
                                          DWORD2(v143),
                                          __PAIR128__(si128.m128i_u64[1], v345),
                                          v143);
                    si128 = v145;
                    v328 = v145.m128i_i64[0];
                    sub_32B3E80((__int64)a1, v145.m128i_i64[0], 1, 0, v146, v147);
                    v342 = v44;
                    v148 = sub_3280A00((__int64)&v362, v44, v45);
                    v150 = v342;
                    v151 = v328;
                    if ( v148 )
                    {
                      v334 = v342;
                      v357 = sub_33FAF80(*a1, 216, v344, v342, v45, v149, *(_OWORD *)&si128);
                      v358 = v293;
                      si128.m128i_i64[0] = v357;
                      v342 = v357;
                      si128.m128i_i64[1] = (unsigned int)v293 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                      sub_32B3E80((__int64)a1, v357, 1, 0, v357, v294);
                      v150 = v334;
                      v151 = v342;
                    }
                    if ( a9 == 18 )
                    {
                      si128.m128i_i64[0] = v151;
                      v295 = *a1;
                      v342 = v150;
                      v151 = sub_34074A0(v295, v344, v151, si128.m128i_i64[1], (unsigned int)v150, v45);
                      v356 = v296;
                      v197 = v296;
                      v355 = v151;
                      goto LABEL_213;
                    }
LABEL_138:
                    v152 = *a1;
                    v39 = 186;
                    si128.m128i_i64[0] = v151;
                    *((_QWORD *)&v306 + 1) = v43 & 0xFFFFFFFF00000000LL | v318;
                    *(_QWORD *)&v306 = v346;
                    result = sub_3406EB0(v152, 186, v344, v150, v45, v149, *(_OWORD *)&si128, v306);
                    if ( result )
                      return result;
                    goto LABEL_26;
                  }
                }
                goto LABEL_206;
              }
            }
            else
            {
              v131 = *(_QWORD *)(v130 + 24) & v368;
              v374.m128i_i32[2] = v129;
              v370 = v131;
              v374.m128i_i64[0] = v131;
              LODWORD(v371) = 0;
            }
            LOBYTE(v342) = v131 == 0;
            goto LABEL_121;
          }
          v169 = sub_33CF170(v339, v168);
          v39 = v345;
          if ( (_QWORD)v346 != v345 || v326 != (_DWORD)v342 )
            goto LABEL_27;
LABEL_176:
          if ( !v169 )
            goto LABEL_27;
          goto LABEL_115;
        }
        if ( a9 == 20 )
        {
          v125 = v340;
          if ( (unsigned __int8)sub_33CF170(v339, v340) )
            goto LABEL_115;
          v169 = sub_33CF4D0(v339, v125);
          v39 = (unsigned int)v342;
          if ( v326 != (_DWORD)v342 || (_QWORD)v346 != v345 )
            goto LABEL_27;
          goto LABEL_176;
        }
LABEL_26:
        if ( a9 == 17 && *(_DWORD *)(v345 + 24) == 186 )
        {
          v96 = *(_QWORD *)(v345 + 48);
          if ( *(_WORD *)v96 == (_WORD)v360 && (v361 == *(_QWORD *)(v96 + 8) || *(_WORD *)v96) )
          {
            v39 = v340;
            v339 = v343;
            if ( (unsigned __int8)sub_33CF170(v343, v340) )
            {
              v39 = *((_QWORD *)&a7 + 1);
              if ( (unsigned __int8)sub_33CF170(a7, *((_QWORD *)&a7 + 1)) )
              {
                v97 = *(_QWORD *)(*(_QWORD *)(v345 + 40) + 40LL);
                v98 = *(_DWORD *)(v97 + 24);
                si128 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v345 + 40));
                if ( v98 == 11 || v98 == 35 )
                {
                  v99 = *(_QWORD *)(v97 + 96);
                  v100 = *(_DWORD *)(v99 + 32) > 0x40u ? sub_C44630(v99 + 24) : sub_39FAC40(*(_QWORD *)(v99 + 24));
                  if ( v100 == 1 )
                  {
                    v39 = v360;
                    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)a1[1] + 1736LL))(
                           a1[1],
                           v360,
                           v361,
                           v99 + 24) )
                    {
                      v101 = *a1;
                      LODWORD(v346) = *(_DWORD *)(v99 + 32) - 1;
                      sub_3285E70((__int64)&v374, si128.m128i_i64[0]);
                      v102 = sub_9871A0(v99 + 24);
                      v103 = sub_3400E40(v101, v102, v360, v361, &v374);
                      v105 = v104;
                      sub_9C6650(&v374);
                      v106 = *a1;
                      *(_QWORD *)&v337 = v345;
                      v107 = v345;
                      v345 = *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL | v326;
                      sub_3285E70((__int64)&v374, v107);
                      *((_QWORD *)&v305 + 1) = v105;
                      *(_QWORD *)&v305 = v103;
                      v109 = sub_3406EB0(v106, 190, (unsigned int)&v374, v360, v361, v108, *(_OWORD *)&si128, v305);
                      v111 = v110;
                      sub_9C6650(&v374);
                      v112 = *a1;
                      sub_3285E70((__int64)&v374, v109);
                      *(_QWORD *)&v346 = sub_3400E40(v112, (unsigned int)v346, v360, v361, &v374);
                      *((_QWORD *)&v346 + 1) = v113;
                      sub_9C6650(&v374);
                      v114 = *a1;
                      sub_3285E70((__int64)&v374, v337);
                      *((_QWORD *)&v299 + 1) = v111;
                      *(_QWORD *)&v299 = v109;
                      v116 = sub_3406EB0(v114, 191, (unsigned int)&v374, v360, v361, v115, v299, v346);
                      v118 = v117;
                      sub_9C6650(&v374);
                      *((_QWORD *)&v300 + 1) = v118;
                      *(_QWORD *)&v300 = v116;
                      return sub_3406EB0(*a1, 186, v344, v360, v361, v119, v300, a8);
                    }
                  }
                }
              }
            }
          }
        }
LABEL_27:
        if ( v321 )
        {
          v39 = *((_QWORD *)&a8 + 1);
          if ( (unsigned __int8)sub_33CF170(a8, *((_QWORD *)&a8 + 1)) )
          {
            v46 = *(_QWORD *)(v321 + 96);
            if ( *(_DWORD *)(v46 + 32) > 0x40u )
            {
              if ( (unsigned int)sub_C44630(v46 + 24) == 1 )
                goto LABEL_32;
            }
            else
            {
              v47 = *(_QWORD *)(v46 + 24);
              if ( v47 && (v47 & (v47 - 1)) == 0 )
              {
LABEL_32:
                if ( !v320 || (v39 = *((_QWORD *)&a7 + 1), (v48 = sub_33CF170(a7, *((_QWORD *)&a7 + 1))) == 0) )
                {
LABEL_34:
                  v49 = 0;
                  goto LABEL_35;
                }
LABEL_143:
                v153 = *(_QWORD *)(v320 + 96);
                if ( *(_DWORD *)(v153 + 32) > 0x40u )
                {
                  v49 = 1;
                  if ( (unsigned int)sub_C44630(v153 + 24) == 1 )
                  {
LABEL_35:
                    v50 = (_DWORD *)a1[1];
                    v374 = _mm_loadu_si128(&v359);
                    if ( v359.m128i_i16[0] )
                    {
                      v51 = v359.m128i_i16[0] - 17;
                      if ( (unsigned __int16)(v359.m128i_i16[0] - 10) > 6u
                        && (unsigned __int16)(v359.m128i_i16[0] - 126) > 0x31u )
                      {
                        if ( v51 > 0xD3u )
                        {
LABEL_39:
                          v52 = v50[15];
                          goto LABEL_40;
                        }
                        goto LABEL_189;
                      }
                      if ( v51 <= 0xD3u )
                      {
LABEL_189:
                        v52 = v50[17];
LABEL_40:
                        if ( v52 == 1 )
                        {
                          if ( !*((_BYTE *)a1 + 33) || (v39 = 208, sub_328D6E0((__int64)v50, 0xD0u, v359.m128i_u16[0])) )
                          {
                            v170 = *(__int64 (**)())(*(_QWORD *)v50 + 1240LL);
                            if ( v170 != sub_2FE3380 )
                            {
                              v39 = v360;
                              if ( ((unsigned __int8 (__fastcall *)(_DWORD *, _QWORD, __int64))v170)(v50, v360, v361) )
                              {
                                if ( v49 )
                                {
                                  a9 = sub_33CBD40(a9, v359.m128i_u32[0], v359.m128i_i64[1]);
                                  v321 = v320;
                                }
                                if ( v335 )
                                {
                                  v171 = *(_QWORD *)(v321 + 96);
                                  v172 = *(_DWORD *)(v171 + 32);
                                  if ( v172 <= 0x40 )
                                  {
                                    if ( *(_QWORD *)(v171 + 24) == 1 )
                                      return 0;
                                  }
                                  else if ( (unsigned int)sub_C444A0(v171 + 24) == v172 - 1 )
                                  {
                                    return 0;
                                  }
                                }
                                v246 = *a1;
                                v341 = 0u;
                                if ( *((_BYTE *)a1 + 34) )
                                {
                                  *(_QWORD *)&v337 = v345;
                                  *((_QWORD *)&v337 + 1) = v326 | *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL;
                                  v339 = v343;
                                  v349 = sub_32889F0(
                                           v246,
                                           v344,
                                           v323,
                                           v324,
                                           v345,
                                           *((__int64 *)&v337 + 1),
                                           __PAIR128__(v340, v343),
                                           a9,
                                           0);
                                  *(_QWORD *)&v341 = v349;
                                  v343 = v349;
                                  v350 = v247;
                                  v248 = (unsigned int)v247 | *((_QWORD *)&v341 + 1) & 0xFFFFFFFF00000000LL;
                                  v249 = *a1;
                                  sub_3285E70((__int64)&v374, a7);
                                  v250 = sub_33FB310(v249, v341, v248, &v374, v360, v361);
                                  LODWORD(v345) = v251;
                                  v252 = v250;
                                  v253 = v251;
                                  sub_9C6650(&v374);
                                }
                                else
                                {
                                  si128 = 0u;
                                  *((_QWORD *)&v337 + 1) = v326 | *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL;
                                  *(_QWORD *)&v337 = v345;
                                  sub_3285E70((__int64)&v374, v345);
                                  v339 = v343;
                                  *((_QWORD *)&v303 + 1) = si128.m128i_i64[0];
                                  *(_QWORD *)&v303 = a9;
                                  v347 = sub_32889F0(
                                           v246,
                                           (int)&v374,
                                           2,
                                           0,
                                           v337,
                                           *((__int64 *)&v337 + 1),
                                           __PAIR128__(v340, v343),
                                           v303,
                                           si128.m128i_u64[1]);
                                  v348 = v283;
                                  *(_QWORD *)&v341 = v347;
                                  v343 = v347;
                                  *((_QWORD *)&v341 + 1) = (unsigned int)v283
                                                         | *((_QWORD *)&v341 + 1) & 0xFFFFFFFF00000000LL;
                                  sub_9C6650(&v374);
                                  v284 = *a1;
                                  sub_3285E70((__int64)&v374, a7);
                                  v252 = sub_33FAF80(v284, 214, (unsigned int)&v374, v360, v361, v285, v341);
                                  LODWORD(v345) = v286;
                                  v253 = v286;
                                  sub_9C6650(&v374);
                                  v255 = v308;
                                }
                                sub_32B3E80((__int64)a1, v343, 1, 0, v254, v255);
                                sub_32B3E80((__int64)a1, v252, 1, 0, v256, v257);
                                v258 = *(_QWORD *)(v321 + 96);
                                v259 = *(_DWORD *)(v258 + 32);
                                v260 = v258 + 24;
                                if ( v259 <= 0x40 )
                                {
                                  v261 = v259 - 1;
                                  if ( *(_QWORD *)(v258 + 24) == 1 )
                                    return v252;
                                }
                                else
                                {
                                  v343 = v258 + 24;
                                  v261 = v259 - 1;
                                  v262 = sub_C444A0(v260);
                                  v260 = v343;
                                  if ( v262 == v261 )
                                    return v252;
                                }
                                v274 = sub_9871A0(v260);
                                v275 = (__int64 *)a1[1];
                                v276 = v261 - v274;
                                v277 = *v275;
                                LODWORD(v343) = v276;
                                if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, _QWORD, __int64, _QWORD))(v277 + 1728))(
                                        v275,
                                        v360,
                                        v361,
                                        v276) )
                                {
                                  v278 = *a1;
                                  v279 = (unsigned int)v345 | v253 & 0xFFFFFFFF00000000LL;
                                  sub_3285E70((__int64)&v374, v252);
                                  *(_QWORD *)&v280 = sub_3400E40(
                                                       v278,
                                                       (unsigned int)v343,
                                                       *(unsigned __int16 *)(v322 + *(_QWORD *)(v346 + 48)),
                                                       *(_QWORD *)(v322 + *(_QWORD *)(v346 + 48) + 8),
                                                       &v374);
                                  *((_QWORD *)&v302 + 1) = v279;
                                  *(_QWORD *)&v302 = v252;
                                  v345 = sub_3406EB0(
                                           v278,
                                           190,
                                           v344,
                                           *(unsigned __int16 *)(*(_QWORD *)(v346 + 48) + v322),
                                           *(_QWORD *)(*(_QWORD *)(v346 + 48) + v322 + 8),
                                           v281,
                                           v302,
                                           v280);
                                  *(_QWORD *)&v346 = v282;
                                  sub_9C6650(&v374);
                                  return v345;
                                }
                                return 0;
                              }
                            }
                          }
                        }
LABEL_41:
                        if ( !v319 )
                          goto LABEL_104;
                        v53 = *(_QWORD *)(v319 + 96);
                        v54 = *(_DWORD *)(v53 + 32);
                        if ( v54 <= 0x40 )
                          v55 = *(_QWORD *)(v53 + 24) == 0;
                        else
                          v55 = v54 == (unsigned int)sub_C444A0(v53 + 24);
                        if ( v55 )
                        {
                          if ( a9 == 17 )
                          {
                            v173 = v346;
                            *(_QWORD *)&v346 = v341;
                            *(_QWORD *)&v341 = v173;
                          }
                          else if ( a9 != 22 )
                          {
                            goto LABEL_47;
                          }
                          v174 = *(_DWORD *)(v341 + 24);
                          if ( v174 == 35 || v174 == 11 )
                          {
                            v175.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v360);
                            v374 = v175;
                            v39 = sub_CA1930(&v374);
                            if ( sub_D94970(*(_QWORD *)(v341 + 96) + 24LL, (_QWORD *)v39) )
                            {
                              v177 = *(_DWORD *)(v346 + 24);
                              if ( v177 == 198 || v177 == 203 )
                              {
                                v271 = *(_QWORD *)(v346 + 40);
                                if ( *(_QWORD *)v271 == v345 && *(_DWORD *)(v271 + 8) == v326 )
                                {
                                  v272 = v360;
                                  v273 = v361;
                                  if ( !*((_BYTE *)a1 + 33) || (v39 = 198, sub_328D6E0(a1[1], 0xC6u, v360)) )
                                  {
                                    v297 = *a1;
                                    *(_QWORD *)&v337 = v345;
                                    return sub_33FAF80(
                                             v297,
                                             198,
                                             v344,
                                             v272,
                                             v273,
                                             v176,
                                             __PAIR128__(v326 | *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL, v345));
                                  }
                                }
                              }
                              else if ( v177 == 204 || v177 == 199 )
                              {
                                v178 = *(_QWORD *)(v346 + 40);
                                if ( *(_QWORD *)v178 == v345 && *(_DWORD *)(v178 + 8) == v326 )
                                {
                                  v179 = v360;
                                  v180 = v361;
                                  if ( !*((_BYTE *)a1 + 33) || (v39 = 199, sub_328D6E0(a1[1], 0xC7u, v360)) )
                                  {
                                    v181 = *a1;
                                    *(_QWORD *)&v337 = v345;
                                    return sub_33FAF80(
                                             v181,
                                             199,
                                             v344,
                                             v179,
                                             v180,
                                             v176,
                                             __PAIR128__(v326 | *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL, v345));
                                  }
                                }
                              }
                            }
                          }
                        }
LABEL_47:
                        if ( !v335 && v321 && v320 )
                        {
                          v56 = *(_QWORD *)(v320 + 96);
                          v57 = *(_DWORD *)(v56 + 32);
                          LODWORD(v363) = v57;
                          if ( v57 > 0x40 )
                          {
                            v39 = v56 + 24;
                            sub_C43780((__int64)&v362, (const void **)(v56 + 24));
                            v57 = v363;
                            if ( (unsigned int)v363 > 0x40 )
                            {
                              sub_C43D10((__int64)&v362);
                              v57 = v363;
                              v61 = v362;
                              goto LABEL_55;
                            }
                            v58 = v362;
                          }
                          else
                          {
                            v58 = *(_QWORD *)(v56 + 24);
                          }
                          v59 = ~v58;
                          v60 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v57;
                          if ( !v57 )
                            v60 = 0;
                          v61 = v60 & v59;
                          v362 = v61;
LABEL_55:
                          v374.m128i_i32[2] = v57;
                          v374.m128i_i64[0] = v61;
                          LODWORD(v363) = 0;
                          v62 = *(_QWORD *)(v321 + 96);
                          if ( *(_DWORD *)(v62 + 32) <= 0x40u )
                          {
                            if ( *(_QWORD *)(v62 + 24) != v61 )
                              goto LABEL_62;
                          }
                          else
                          {
                            v39 = (unsigned __int64)&v374;
                            v336 = sub_C43C50(v62 + 24, (const void **)&v374);
                            if ( !v336 )
                              goto LABEL_62;
                          }
                          v63 = *(_QWORD *)(v319 + 96);
                          *(_QWORD *)&v346 = v63 + 24;
                          v64 = sub_986760(v63 + 24);
                          if ( a9 == 18 && v64
                            || ((v65 = *(_DWORD *)(v63 + 32), v65 <= 0x40)
                              ? (v66 = *(_QWORD *)(v63 + 24) == 0)
                              : (v66 = v65 == (unsigned int)sub_C444A0(v346)),
                                (v336 = v66 && a9 == 20) != 0) )
                          {
                            v268 = a1[1];
                            v269 = *(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v268 + 1728LL);
                            v270 = sub_32844A0((unsigned __int16 *)&v359, v39);
                            v39 = v360;
                            v336 = v269(v268, v360, v361, (unsigned int)(v270 - 1)) ^ 1;
                          }
LABEL_62:
                          if ( v57 > 0x40 && v61 )
                            j_j___libc_free_0_0(v61);
                          if ( (unsigned int)v363 > 0x40 && v362 )
                            j_j___libc_free_0_0(v362);
                          v67 = *a1;
                          if ( v336 )
                          {
                            v68 = sub_32844A0((unsigned __int16 *)&v359, v39);
                            v69 = v344;
                            *(_QWORD *)&v70 = sub_3400BD0(
                                                v67,
                                                v68 - 1,
                                                v344,
                                                v359.m128i_i32[0],
                                                v359.m128i_i32[2],
                                                0,
                                                0);
                            *(_QWORD *)&v337 = v345;
                            *((_QWORD *)&v337 + 1) = v326 | *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL;
                            v72 = sub_3406EB0(
                                    v67,
                                    191,
                                    v69,
                                    v359.m128i_i32[0],
                                    v359.m128i_i32[2],
                                    v71,
                                    __PAIR128__(*((unsigned __int64 *)&v337 + 1), v345),
                                    v70);
                            v73 = *a1;
                            *(_QWORD *)&v346 = v72;
                            v74 = &a8;
                            v76 = v75;
                            if ( a9 != 20 )
                              v74 = &a7;
                            v344 = v69;
                            v77 = sub_33FB160(v73, *(_QWORD *)v74, *((_QWORD *)v74 + 1), v69, v360, v361);
                            v79 = v78;
                            v80 = v76;
                            v81 = v344;
                            v82 = v77;
                            *(_QWORD *)&v83 = sub_33FB160(*a1, v346, v80, v344, v360, v361);
                            *((_QWORD *)&v304 + 1) = v79;
                            *(_QWORD *)&v304 = v82;
                            return sub_3406EB0(v73, 188, v81, v360, v361, v84, v83, v304);
                          }
                          goto LABEL_105;
                        }
LABEL_104:
                        v67 = *a1;
LABEL_105:
                        v120 = a7;
                        *(_QWORD *)&v337 = v345;
                        v121 = DWORD2(a7);
                        v122 = *((_QWORD *)&v337 + 1) & 0xFFFFFFFF00000000LL | v326;
                        v339 = v343;
                        result = sub_3283940(
                                   v345,
                                   v326,
                                   v343,
                                   v340,
                                   a7,
                                   SDWORD2(a7),
                                   a8,
                                   *((__int64 *)&a8 + 1),
                                   a9,
                                   v67);
                        if ( !result )
                        {
                          result = sub_3286970(v337, v122, v339, v340, v120, v121, a8, *((__int64 *)&a8 + 1), a9, *a1);
                          if ( !result )
                          {
                            v123 = sub_32AD6C0(a1, v337, v122, v343, v340, a9, a7, a8, v344);
                            result = 0;
                            if ( v123 )
                              return v123;
                          }
                        }
                        return result;
                      }
                    }
                    else
                    {
                      v163 = sub_3007030((__int64)&v374);
                      if ( sub_30070B0((__int64)&v374) )
                        goto LABEL_189;
                      if ( !v163 )
                        goto LABEL_39;
                    }
                    v52 = v50[16];
                    goto LABEL_40;
                  }
                }
                else
                {
                  v154 = *(_QWORD *)(v153 + 24);
                  if ( v154 && (v154 & (v154 - 1)) == 0 )
                  {
                    v49 = 1;
                    goto LABEL_35;
                  }
                }
                if ( !v48 )
                  goto LABEL_41;
                goto LABEL_34;
              }
            }
          }
        }
        if ( !v320 )
          goto LABEL_41;
        v39 = *((_QWORD *)&a7 + 1);
        v48 = 0;
        if ( !(unsigned __int8)sub_33CF170(a7, *((_QWORD *)&a7 + 1)) )
          goto LABEL_41;
        goto LABEL_143;
      }
      v374.m128i_i64[1] = v45;
      v374.m128i_i16[0] = 0;
    }
    else
    {
      v374.m128i_i16[0] = v44;
      v374.m128i_i64[1] = v45;
      if ( v44 )
      {
        if ( v44 == 1 || (unsigned __int16)(v44 - 504) <= 7u )
          goto LABEL_289;
        v39 = *(_QWORD *)&byte_444C4A0[16 * v44 - 16];
        v157 = byte_444C4A0[16 * v44 - 8];
LABEL_149:
        if ( v124 )
        {
          if ( v124 == 1 || (unsigned __int16)(v124 - 504) <= 7u )
            goto LABEL_289;
          v162 = *(_QWORD *)&byte_444C4A0[16 * v124 - 16];
          LOBYTE(v161) = byte_444C4A0[16 * v124 - 8];
        }
        else
        {
          v314 = v157;
          v158 = sub_3007260((__int64)&v362);
          v157 = v314;
          v159 = v158;
          v161 = v160;
          v364 = v159;
          v162 = v159;
          v365 = v161;
        }
        if ( !(_BYTE)v161 && v157 || v162 < v39 )
          goto LABEL_26;
        goto LABEL_112;
      }
    }
    v329 = v362;
    v155 = sub_3007260((__int64)&v374);
    v124 = v329;
    v366 = v155;
    v39 = v155;
    v367 = v156;
    v157 = v156;
    goto LABEL_149;
  }
LABEL_82:
  v92 = **(unsigned __int16 **)(v341 + 48);
  if ( v90 != sub_2FE3170 )
  {
    v198 = *(_QWORD *)(*(_QWORD *)(v341 + 48) + 8LL);
    si128.m128i_i64[0] = *(_QWORD *)(v341 + 96);
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v90)(
           v87,
           si128.m128i_i64[0] + 24,
           v92,
           v198,
           v88) )
    {
      goto LABEL_85;
    }
  }
  v93 = *(_QWORD *)(v346 + 56);
  if ( !v93 || *(_QWORD *)(v93 + 32) )
  {
    v94 = *(_QWORD *)(v341 + 56);
    if ( !v94 || *(_QWORD *)(v94 + 32) )
      goto LABEL_85;
  }
  v199 = *(_QWORD *)(v341 + 96);
  v371 = *(_QWORD *)(v346 + 96);
  v370 = v199;
  v200 = *(__int64 **)(v199 + 8);
  v201 = sub_2E79000(*(__int64 **)(*a1 + 40));
  v202 = (__int64 **)sub_BCD420(v200, 2);
  v203 = sub_AD1300(v202, (__int64 *)&v370, 2);
  si128.m128i_i64[0] = *a1;
  v204 = (unsigned __int8)sub_AE5260(v201, (__int64)v200);
  v316 = a1[1];
  BYTE1(v204) = 1;
  v315 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v316 + 32LL);
  v205 = sub_2E79000(*(__int64 **)(*a1 + 40));
  if ( v315 == sub_2D42F30 )
  {
    v206 = sub_AE2980(v205, 0)[1];
    switch ( v206 )
    {
      case 1:
        v207 = 2;
        break;
      case 2:
        v207 = 3;
        break;
      case 4:
        v207 = 4;
        break;
      case 8:
        v207 = 5;
        break;
      case 16:
        v207 = 6;
        break;
      case 32:
        v207 = 7;
        break;
      case 64:
        v207 = 8;
        break;
      default:
        v207 = 9 * (v206 == 128);
        break;
    }
  }
  else
  {
    v207 = v315(v316, v205, 0);
  }
  v208.m128i_i64[0] = sub_33EE5B0(si128.m128i_i32[0], v203, v207, 0, v204, 0, 0, 0);
  v209 = *a1;
  si128 = v208;
  v311 = v208.m128i_i64[0];
  v312 = v208.m128i_u32[2];
  v325 = *(_BYTE *)(v208.m128i_i64[0] + 108);
  *(_QWORD *)&v310 = sub_3400D50(v209, 0, v344, 0);
  *((_QWORD *)&v310 + 1) = v210;
  v211 = *(_QWORD *)(v370 + 8);
  v212 = sub_AE5020(v201, v211);
  v213 = sub_9208B0(v201, v211);
  v374.m128i_i64[1] = v214;
  v374.m128i_i64[0] = ((1LL << v212) + ((unsigned __int64)(v213 + 7) >> 3) - 1) >> v212 << v212;
  v215 = sub_CA1930(&v374);
  v216 = *a1;
  v217 = v215;
  v374.m128i_i64[0] = *(_QWORD *)(v341 + 80);
  if ( v374.m128i_i64[0] )
    sub_325F5D0(v374.m128i_i64);
  v374.m128i_i32[2] = *(_DWORD *)(v341 + 72);
  *(_QWORD *)&v309 = sub_3400D50(v216, v217, &v374, 0);
  *((_QWORD *)&v309 + 1) = v218;
  if ( v374.m128i_i64[0] )
    sub_B91220((__int64)&v374, v374.m128i_i64[0]);
  v219 = *a1;
  v220 = sub_325F2E0(
           *a1,
           a1[1],
           *(unsigned __int16 *)(*(_QWORD *)(v345 + 48) + v327),
           *(_QWORD *)(*(_QWORD *)(v345 + 48) + v327 + 8));
  v313 = v221;
  v317 = v220;
  *(_QWORD *)&v222 = sub_33ED040(v219, a9);
  *((_QWORD *)&v301 + 1) = (unsigned int)v340;
  *(_QWORD *)&v301 = v343;
  *((_QWORD *)&v298 + 1) = v326;
  *(_QWORD *)&v298 = v345;
  v223 = sub_340F900(v219, 208, v344, v317, v313, v340, v298, v301, v222);
  v225 = v224;
  sub_32B3E80((__int64)a1, v223, 1, 0, v226, v227);
  v228 = (unsigned __int16 *)(*(_QWORD *)(v310 + 48) + 16LL * DWORD2(v310));
  v229 = sub_3288B20(*a1, v344, *v228, *((_QWORD *)v228 + 1), v223, v225, v309, v310, 0);
  v231 = v230;
  sub_32B3E80((__int64)a1, v229, 1, 0, v232, v233);
  v234 = (unsigned __int16 *)(*(_QWORD *)(v311 + 48) + 16LL * v312);
  *((_QWORD *)&v307 + 1) = v231;
  *(_QWORD *)&v307 = v229;
  v236 = sub_3406EB0(*a1, 56, v344, *v234, *((_QWORD *)v234 + 1), v235, *(_OWORD *)&si128, v307);
  LODWORD(v225) = v237;
  sub_32B3E80((__int64)a1, v236, 1, 0, v238, v239);
  v240 = *a1;
  v374 = 0u;
  v375 = 0;
  LOBYTE(v241) = v325;
  v376 = 0;
  HIBYTE(v241) = 1;
  v242 = v241;
  sub_2EAC2B0((__int64)&v372, *(_QWORD *)(v240 + 40));
  v243 = *a1;
  si128.m128i_i64[0] = v236;
  v244 = *(_QWORD *)(*(_QWORD *)(v346 + 48) + 8LL);
  v245 = **(unsigned __int16 **)(v346 + 48);
  si128.m128i_i64[1] = (unsigned int)v225 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  result = sub_33F1F00(
             v240,
             v245,
             v244,
             v344,
             (int)v243 + 288,
             0,
             v236,
             si128.m128i_i64[1],
             v372,
             v373,
             v242,
             0,
             (__int64)&v374,
             0);
  if ( !result )
  {
LABEL_85:
    v38 = (unsigned __int16 *)(*(_QWORD *)(v346 + 48) + v322);
    goto LABEL_25;
  }
  return result;
}
