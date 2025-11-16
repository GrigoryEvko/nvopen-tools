// Function: sub_32C3760
// Address: 0x32c3760
//
__int64 __fastcall sub_32C3760(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __m128i v7; // xmm0
  __int16 *v8; // rax
  unsigned __int16 v9; // r14
  __int64 v10; // rax
  int v11; // ebx
  __int64 v12; // r15
  int v13; // eax
  __int16 v14; // bx
  __int64 v15; // r14
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // r14
  __int128 v22; // rax
  __int64 v23; // rdi
  __int128 v24; // rax
  int v25; // r9d
  int v26; // esi
  __int64 result; // rax
  __int64 v28; // rsi
  unsigned __int16 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rcx
  unsigned __int16 v33; // ax
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  int v37; // edx
  unsigned __int16 *v38; // rax
  unsigned __int16 v39; // dx
  __int64 v40; // rbx
  unsigned int *v41; // rax
  __int64 v42; // r14
  __int64 v43; // rax
  __int16 v44; // cx
  __int64 v45; // rax
  __int64 v46; // rbx
  unsigned __int16 v47; // ax
  __int64 v48; // rax
  __int64 v49; // rcx
  int v50; // eax
  __int64 v51; // r14
  int v52; // r13d
  unsigned int *v53; // rax
  __int128 v54; // rcx
  __int64 v55; // rax
  __int16 v56; // dx
  __int64 v57; // rax
  bool v58; // al
  __int64 v59; // r12
  __int64 v60; // rdi
  __int64 v61; // r13
  unsigned int v62; // r13d
  __int64 v63; // rax
  __int64 v64; // rbx
  __int16 *v65; // rax
  __int16 v66; // dx
  __int64 v67; // rax
  unsigned int v68; // r13d
  __int64 v69; // rax
  __int16 *v70; // rax
  __int16 v71; // dx
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // edx
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rdx
  unsigned int v78; // eax
  __int64 v79; // rbx
  unsigned int v80; // r13d
  __int64 v81; // rax
  unsigned int v82; // eax
  int v83; // r13d
  __int64 v84; // rax
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 *v87; // r11
  _WORD *v88; // r10
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // rdx
  _BYTE *v92; // rcx
  int *v93; // rbx
  __int64 *v94; // rax
  int v95; // r12d
  int v96; // r14d
  __int64 v97; // rbx
  int v98; // esi
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rbx
  __int64 v105; // rdx
  __int64 v106; // rax
  unsigned __int16 v107; // dx
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rdx
  int v111; // eax
  unsigned int v112; // eax
  unsigned int v113; // ebx
  __int64 v114; // rdx
  __int64 v115; // rdx
  __int64 v116; // rdx
  unsigned __int64 v117; // rax
  _QWORD *v118; // rax
  unsigned int v119; // ebx
  unsigned int v120; // edx
  const __m128i *v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  bool v124; // al
  bool v125; // al
  __int64 v126; // rcx
  __int64 v127; // r8
  unsigned __int16 v128; // ax
  __int64 v129; // rdx
  __int64 v130; // r8
  __int64 v131; // rdi
  __int64 (*v132)(); // rcx
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // rax
  __int64 v136; // rdx
  __int128 v137; // rax
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 v140; // r10
  __int64 v141; // rax
  __int64 v142; // rdx
  unsigned int v143; // eax
  __int64 v144; // r13
  __int64 v145; // rdx
  __int64 v146; // rcx
  __int64 v147; // r8
  __int128 v148; // rax
  int v149; // r9d
  __int64 v150; // rdx
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 *v153; // r11
  __int64 v154; // rdx
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v158; // rdx
  __int64 v159; // rcx
  __int64 v160; // r8
  unsigned int v161; // eax
  unsigned int v162; // ebx
  __int64 v163; // rdx
  _QWORD *v164; // rdx
  __int64 v165; // rbx
  _BYTE *v166; // rax
  __int64 v167; // rax
  char v168; // dl
  _BYTE *v169; // r8
  __int64 v170; // rax
  __int64 v171; // rdx
  unsigned __int16 *v172; // rbx
  char v173; // al
  __int64 v174; // rdx
  _QWORD *v175; // rbx
  __int64 v176; // r14
  int v177; // eax
  __int64 v178; // r14
  unsigned __int16 v179; // bx
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // rdx
  __int64 v183; // rbx
  _QWORD *v184; // rbx
  __int64 v185; // rax
  __int64 v186; // rbx
  int v187; // eax
  __int64 v188; // rax
  __int64 v189; // rdx
  unsigned int v190; // eax
  __int64 *v191; // r10
  __int64 v192; // rdx
  __int64 v193; // rdi
  __int64 (*v194)(); // rcx
  __int64 v195; // rax
  unsigned __int16 v196; // ax
  __int64 v197; // rdx
  __int64 v198; // r8
  unsigned int v199; // edx
  __int64 v200; // rdx
  __int64 v201; // rcx
  __int64 v202; // r8
  __int64 v203; // rdx
  __int64 v204; // rcx
  __int64 (*v205)(); // rcx
  char v206; // al
  __int64 v207; // rdx
  unsigned int v208; // r9d
  __int128 v209; // rax
  unsigned int v210; // eax
  __int64 v211; // rdx
  __int64 v212; // r8
  __int64 v213; // rcx
  __int128 v214; // rax
  __int64 v215; // rbx
  __int64 v216; // r13
  __int64 v217; // r8
  __int64 v218; // r9
  __int64 v219; // rdx
  unsigned int v220; // eax
  int v221; // r9d
  __int64 *v222; // r10
  __int64 v223; // r13
  unsigned int v224; // edx
  __int64 v225; // r8
  __int64 v226; // r9
  __int64 v227; // rax
  __int64 v228; // rdx
  unsigned int v229; // eax
  __int128 v230; // rax
  int v231; // r9d
  unsigned int v232; // edx
  __int64 v233; // r8
  __int64 v234; // r9
  __int64 v235; // rax
  __int64 v236; // rdx
  __int64 v237; // r14
  __int64 v238; // r13
  __int64 v239; // rdx
  __int64 v240; // rcx
  __int64 v241; // r8
  __int128 v242; // rax
  __int64 v243; // rdx
  unsigned __int64 v244; // r14
  __int64 v245; // r8
  __int64 v246; // r9
  int v247; // r9d
  __int64 v248; // r12
  __int64 *v249; // r10
  __int64 v250; // rdx
  __int64 v251; // rbx
  unsigned int v252; // esi
  __int64 v253; // r11
  __int128 v254; // rax
  int v255; // r9d
  __int64 v256; // rax
  int v257; // edx
  __int64 v258; // r8
  __int64 v259; // r9
  __int64 v260; // r13
  int v261; // r9d
  unsigned int v262; // edx
  __int64 v263; // r8
  __int64 v264; // r9
  __int64 v265; // r12
  __int128 v266; // rax
  int v267; // r9d
  __int64 v268; // rax
  __int64 v269; // rdx
  unsigned __int64 v270; // rax
  __m128i v271; // rax
  __int64 v272; // r8
  __int64 v273; // r9
  __int64 v274; // r13
  __int64 v275; // rdx
  __int64 v276; // r14
  __int64 v277; // r8
  __int64 v278; // r9
  __int64 v279; // r13
  __int64 v280; // rdx
  __int64 v281; // r14
  __int64 v282; // r8
  __int64 v283; // r9
  unsigned __int8 *v284; // rax
  __int128 v285; // rax
  __int64 v286; // r13
  __int64 v287; // rdx
  __int64 v288; // r14
  __int64 v289; // r8
  __int64 v290; // r9
  __int64 v291; // r13
  __int128 v292; // rax
  __int64 v293; // r13
  __int64 v294; // rdx
  __int64 v295; // r14
  __int64 v296; // r8
  __int64 v297; // r9
  __int64 v298; // r13
  __int64 v299; // rdx
  __int64 v300; // r14
  __int64 v301; // r8
  __int64 v302; // r9
  int v303; // r9d
  __int64 v304; // r12
  _QWORD *v305; // r10
  __int64 v306; // rdx
  __int64 v307; // r15
  _QWORD *v308; // rax
  __int64 *v309; // rdx
  unsigned __int64 v310; // rax
  __m128i v311; // rax
  __int64 v312; // rbx
  __int64 v313; // r8
  __int64 v314; // r9
  __int64 v315; // r13
  unsigned __int8 *v316; // rax
  __int128 v317; // rax
  int v318; // r9d
  __int64 v319; // rdx
  __int64 v320; // r8
  __int64 v321; // r9
  __int64 v322; // r13
  int v323; // r9d
  __int64 v324; // rdx
  __int64 v325; // r15
  __int64 v326; // r8
  __int64 v327; // r9
  __int64 v328; // r13
  int v329; // r9d
  __int64 v330; // r14
  __int64 v331; // rdx
  __int64 v332; // r15
  __int64 v333; // r8
  __int64 v334; // r9
  __int64 v335; // rdi
  int v336; // r9d
  _BYTE *v337; // rbx
  __int64 v338; // r10
  __int64 v339; // r15
  __int64 v340; // r12
  __int64 v341; // r13
  __int64 v342; // r14
  int v343; // r8d
  __int64 v344; // r13
  __int64 v345; // rdx
  __int64 v346; // r12
  __int128 v347; // [rsp-30h] [rbp-1E0h]
  __int128 v348; // [rsp-30h] [rbp-1E0h]
  __int128 v349; // [rsp-20h] [rbp-1D0h]
  __int128 v350; // [rsp-20h] [rbp-1D0h]
  __int128 v351; // [rsp-20h] [rbp-1D0h]
  __int128 v352; // [rsp-20h] [rbp-1D0h]
  __int128 v353; // [rsp-20h] [rbp-1D0h]
  __int128 v354; // [rsp-10h] [rbp-1C0h]
  __int128 v355; // [rsp-10h] [rbp-1C0h]
  __int128 v356; // [rsp-10h] [rbp-1C0h]
  __int128 v357; // [rsp-10h] [rbp-1C0h]
  __int128 v358; // [rsp-10h] [rbp-1C0h]
  __int128 v359; // [rsp-10h] [rbp-1C0h]
  __int128 v360; // [rsp-10h] [rbp-1C0h]
  unsigned int v361; // [rsp-8h] [rbp-1B8h]
  _BYTE *v362; // [rsp+10h] [rbp-1A0h]
  __int64 *v363; // [rsp+18h] [rbp-198h]
  __int64 v364; // [rsp+20h] [rbp-190h]
  unsigned int v365; // [rsp+20h] [rbp-190h]
  __int64 *v366; // [rsp+28h] [rbp-188h]
  __int64 *v367; // [rsp+28h] [rbp-188h]
  __int64 v368; // [rsp+28h] [rbp-188h]
  _WORD *v369; // [rsp+30h] [rbp-180h]
  int *v370; // [rsp+30h] [rbp-180h]
  __int128 v371; // [rsp+30h] [rbp-180h]
  __int128 v372; // [rsp+30h] [rbp-180h]
  __int128 v373; // [rsp+30h] [rbp-180h]
  unsigned int v374; // [rsp+30h] [rbp-180h]
  _QWORD *v375; // [rsp+30h] [rbp-180h]
  int v376; // [rsp+30h] [rbp-180h]
  int v377; // [rsp+40h] [rbp-170h]
  __int64 v378; // [rsp+40h] [rbp-170h]
  _BYTE *v379; // [rsp+40h] [rbp-170h]
  __int64 *v380; // [rsp+40h] [rbp-170h]
  unsigned int v381; // [rsp+40h] [rbp-170h]
  unsigned int v382; // [rsp+40h] [rbp-170h]
  unsigned int v383; // [rsp+40h] [rbp-170h]
  unsigned int v384; // [rsp+40h] [rbp-170h]
  unsigned int v385; // [rsp+40h] [rbp-170h]
  __int64 *v386; // [rsp+40h] [rbp-170h]
  __int64 *v387; // [rsp+40h] [rbp-170h]
  __int64 *v388; // [rsp+40h] [rbp-170h]
  __int64 v389; // [rsp+40h] [rbp-170h]
  __int128 v390; // [rsp+40h] [rbp-170h]
  _QWORD *v391; // [rsp+40h] [rbp-170h]
  __int128 v392; // [rsp+50h] [rbp-160h]
  __int64 *v393; // [rsp+50h] [rbp-160h]
  __int64 v394; // [rsp+50h] [rbp-160h]
  unsigned int v395; // [rsp+50h] [rbp-160h]
  __int64 v396; // [rsp+50h] [rbp-160h]
  unsigned int v397; // [rsp+50h] [rbp-160h]
  unsigned int v398; // [rsp+50h] [rbp-160h]
  unsigned int v399; // [rsp+50h] [rbp-160h]
  unsigned int v400; // [rsp+50h] [rbp-160h]
  __int64 *v401; // [rsp+50h] [rbp-160h]
  __int64 *v402; // [rsp+50h] [rbp-160h]
  __int64 v403; // [rsp+50h] [rbp-160h]
  __int64 v404; // [rsp+50h] [rbp-160h]
  __int64 v405; // [rsp+50h] [rbp-160h]
  __int64 v406; // [rsp+50h] [rbp-160h]
  __int64 v407; // [rsp+50h] [rbp-160h]
  __int64 v408; // [rsp+50h] [rbp-160h]
  __int64 v409; // [rsp+50h] [rbp-160h]
  _QWORD *v410; // [rsp+50h] [rbp-160h]
  _QWORD *v411; // [rsp+50h] [rbp-160h]
  __int128 v412; // [rsp+50h] [rbp-160h]
  __int64 v413; // [rsp+58h] [rbp-158h]
  __int128 v414; // [rsp+60h] [rbp-150h] BYREF
  __int128 v415; // [rsp+70h] [rbp-140h]
  __int64 v416; // [rsp+80h] [rbp-130h]
  __int64 v417; // [rsp+88h] [rbp-128h]
  __int64 v418; // [rsp+90h] [rbp-120h]
  __int64 v419; // [rsp+98h] [rbp-118h]
  __m128i si128; // [rsp+A0h] [rbp-110h]
  __m128i v421; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v422; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v423; // [rsp+C8h] [rbp-E8h]
  unsigned int v424; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v425; // [rsp+D8h] [rbp-D8h]
  __int64 v426; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v427; // [rsp+E8h] [rbp-C8h]
  __int64 v428; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v429; // [rsp+F8h] [rbp-B8h]
  _QWORD *v430; // [rsp+100h] [rbp-B0h] BYREF
  __int64 *v431; // [rsp+108h] [rbp-A8h]
  _QWORD *v432; // [rsp+110h] [rbp-A0h] BYREF
  __int64 v433; // [rsp+118h] [rbp-98h]
  __int64 v434; // [rsp+120h] [rbp-90h]
  __int64 v435; // [rsp+128h] [rbp-88h]
  __int64 v436; // [rsp+130h] [rbp-80h]
  __int64 v437; // [rsp+138h] [rbp-78h]
  __int64 v438; // [rsp+140h] [rbp-70h] BYREF
  __int64 v439; // [rsp+148h] [rbp-68h]
  __int64 v440; // [rsp+150h] [rbp-60h] BYREF
  __int64 v441; // [rsp+158h] [rbp-58h]
  _BYTE v442[80]; // [rsp+160h] [rbp-50h] BYREF

  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v8 = *(__int16 **)(a2 + 48);
  v421 = v7;
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = *(_DWORD *)(v7.m128i_i64[0] + 24);
  LOWORD(v422) = v9;
  v423 = v10;
  if ( v11 == 51 )
  {
    v60 = *a1;
    v440 = 0;
    LODWORD(v441) = 0;
    v61 = sub_33F17F0(v60, 51, &v440, v422, v10);
    if ( v440 )
      sub_B91220((__int64)&v440, v440);
    return v61;
  }
  v12 = a2;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0xD3u )
      goto LABEL_4;
  }
  else if ( !sub_30070B0((__int64)&v422) )
  {
    goto LABEL_4;
  }
  if ( !*((_BYTE *)a1 + 34) )
    goto LABEL_45;
  if ( *((_BYTE *)a1 + 33) )
    goto LABEL_4;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 2) > 7u && (unsigned __int16)(v9 - 17) > 0x6Cu && (unsigned __int16)(v9 - 176) > 0x1Fu )
      goto LABEL_4;
  }
  else if ( !sub_3007070((__int64)&v422) )
  {
    goto LABEL_4;
  }
  v29 = (unsigned __int16 *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2]);
  v30 = *v29;
  v31 = *((_QWORD *)v29 + 1);
  LOWORD(v440) = v30;
  v441 = v31;
  if ( (_WORD)v30 )
  {
    if ( (unsigned __int16)(v30 - 2) > 7u && (unsigned __int16)(v30 - 17) > 0x6Cu )
    {
      LOWORD(v30) = v30 - 176;
      if ( (unsigned __int16)v30 > 0x1Fu )
        goto LABEL_4;
    }
  }
  else if ( !sub_3007070((__int64)&v440) )
  {
    goto LABEL_4;
  }
  v32 = a1[1];
  if ( v9 )
  {
    v33 = word_4456580[v9 - 1];
  }
  else
  {
    *(_QWORD *)&v415 = a1[1];
    v33 = sub_3009970((__int64)&v422, a2, v30, v32, a5);
    v11 = *(_DWORD *)(v7.m128i_i64[0] + 24);
    v32 = v415;
  }
  if ( v33 && *(_QWORD *)(v32 + 8LL * v33 + 112) )
  {
LABEL_45:
    if ( v11 == 156 )
    {
      v34 = *(_QWORD *)(v7.m128i_i64[0] + 56);
      if ( !v34 || *(_QWORD *)(v34 + 32) )
      {
        v13 = 156;
        goto LABEL_13;
      }
      if ( (unsigned __int8)sub_33E22F0(v7.m128i_i64[0]) )
      {
        v210 = sub_3281170(&v422, a2, v158, v159, v160);
        return sub_32C29C0(a1, v7.m128i_i64[0], v210, v211, v212);
      }
      v11 = *(_DWORD *)(v7.m128i_i64[0] + 24);
    }
  }
LABEL_4:
  if ( v11 == 11 )
  {
    if ( !*((_BYTE *)a1 + 33) )
      goto LABEL_88;
    goto LABEL_49;
  }
  v13 = v11;
  if ( (unsigned int)(v11 - 35) <= 1 || v11 == 12 )
  {
    if ( !*((_BYTE *)a1 + 33) )
      goto LABEL_88;
    if ( v11 != 35 )
    {
      if ( v11 != 36 && v11 != 12 )
        goto LABEL_11;
      if ( !(_WORD)v422 )
        goto LABEL_11;
      v47 = v422 - 17;
      if ( (unsigned __int16)(v422 - 2) > 7u && v47 > 0x6Cu && (unsigned __int16)(v422 - 176) > 0x1Fu )
        goto LABEL_11;
      if ( v47 <= 0xD3u )
        goto LABEL_11;
      v48 = a1[1];
      v49 = 1;
      if ( (_WORD)v422 != 1 )
      {
        v49 = (unsigned __int16)v422;
        if ( !*(_QWORD *)(v48 + 8LL * (unsigned __int16)v422 + 112) )
        {
          v13 = v11;
          goto LABEL_12;
        }
      }
      if ( *(_BYTE *)(v48 + 500 * v49 + 6425) )
      {
LABEL_11:
        v13 = v11;
        goto LABEL_12;
      }
LABEL_88:
      a2 = (unsigned int)v422;
      result = sub_33FB890(*a1, (unsigned int)v422, v423, v421.m128i_i64[0], v421.m128i_i64[1]);
      if ( v12 != result )
        return result;
      v11 = *(_DWORD *)(v7.m128i_i64[0] + 24);
      v13 = v11;
      goto LABEL_12;
    }
LABEL_49:
    if ( !(_WORD)v422
      || (unsigned __int16)(v422 - 10) > 6u
      && (unsigned __int16)(v422 - 126) > 0x31u
      && (unsigned __int16)(v422 - 208) > 0x14u
      || (unsigned __int16)(v422 - 17) <= 0xD3u
      || (a2 = a1[1], v35 = 1, (_WORD)v422 != 1)
      && (v35 = (unsigned __int16)v422, !*(_QWORD *)(a2 + 8LL * (unsigned __int16)v422 + 112)) )
    {
      v13 = v11;
      goto LABEL_13;
    }
    if ( *(_BYTE *)(a2 + 500 * v35 + 6426) )
    {
      v13 = v11;
      goto LABEL_12;
    }
    goto LABEL_88;
  }
LABEL_12:
  if ( v11 == 234 )
    return sub_33FB890(
             *a1,
             (unsigned int)v422,
             v423,
             **(_QWORD **)(v7.m128i_i64[0] + 40),
             *(_QWORD *)(*(_QWORD *)(v7.m128i_i64[0] + 40) + 8LL));
LABEL_13:
  if ( (unsigned int)(v13 - 186) > 2 )
    goto LABEL_59;
  v14 = v422;
  *(_QWORD *)&v415 = a1[1];
  if ( (_WORD)v422 )
  {
    if ( (unsigned __int16)(v422 - 2) > 7u
      && (unsigned __int16)(v422 - 17) > 0x6Cu
      && (unsigned __int16)(v422 - 176) > 0x1Fu )
    {
      goto LABEL_72;
    }
  }
  else if ( !sub_3007070((__int64)&v422) )
  {
    goto LABEL_72;
  }
  a2 = *(_QWORD *)(v7.m128i_i64[0] + 40);
  v15 = *(_QWORD *)a2;
  v16 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * *(unsigned int *)(a2 + 8));
  if ( (_WORD)v16 && *(_QWORD *)(v415 + 8 * v16 + 112) )
    goto LABEL_72;
  v17 = *(_DWORD *)(v15 + 24) == 234;
  *(_QWORD *)&v415 = v423;
  if ( v17 )
  {
    v18 = *(_QWORD *)(**(_QWORD **)(v15 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v15 + 40) + 8LL);
    if ( v14 == *(_WORD *)v18 && (*(_QWORD *)(v18 + 8) == v423 || v14) )
      goto LABEL_24;
  }
  if ( (unsigned __int8)sub_33CA6D0(v15) )
  {
    v157 = *(_QWORD *)(v15 + 56);
    if ( v157 )
    {
      if ( !*(_QWORD *)(v157 + 32) )
      {
        a2 = *(_QWORD *)(v7.m128i_i64[0] + 40);
LABEL_24:
        v19 = *(_QWORD *)(a2 + 40);
        if ( *(_DWORD *)(v19 + 24) == 234 )
        {
          v20 = *(_QWORD *)(**(_QWORD **)(v19 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v19 + 40) + 8LL);
          if ( v14 == *(_WORD *)v20 && (*(_QWORD *)(v20 + 8) == (_QWORD)v415 || v14) )
            goto LABEL_28;
        }
        if ( (unsigned __int8)sub_33CA6D0(*(_QWORD *)(a2 + 40)) )
        {
          v195 = *(_QWORD *)(v19 + 56);
          if ( v195 )
          {
            if ( !*(_QWORD *)(v195 + 32) )
            {
              a2 = *(_QWORD *)(v7.m128i_i64[0] + 40);
LABEL_28:
              v21 = *a1;
              *(_QWORD *)&v22 = sub_33FB890(*a1, (unsigned int)v422, v423, *(_QWORD *)(a2 + 40), *(_QWORD *)(a2 + 48));
              v23 = *a1;
              v415 = v22;
              *(_QWORD *)&v24 = sub_33FB890(
                                  v23,
                                  (unsigned int)v422,
                                  v423,
                                  **(_QWORD **)(v7.m128i_i64[0] + 40),
                                  *(_QWORD *)(*(_QWORD *)(v7.m128i_i64[0] + 40) + 8LL));
              v440 = *(_QWORD *)(v12 + 80);
              if ( v440 )
              {
                v392 = v24;
                *(_QWORD *)&v414 = &v440;
                sub_325F5D0(&v440);
                v24 = v392;
              }
              LODWORD(v441) = *(_DWORD *)(v12 + 72);
              v26 = *(_DWORD *)(v7.m128i_i64[0] + 24);
              v354 = v415;
              *(_QWORD *)&v415 = &v440;
              result = sub_3406EB0(v21, v26, (unsigned int)&v440, v422, v423, v25, v24, v354);
              v28 = v440;
              if ( !v440 )
                return result;
LABEL_104:
              *(_QWORD *)&v414 = result;
              sub_B91220(v415, v28);
              return v414;
            }
          }
        }
      }
    }
  }
  v11 = *(_DWORD *)(v7.m128i_i64[0] + 24);
LABEL_59:
  if ( v11 != 298 )
    goto LABEL_71;
  if ( (*(_BYTE *)(v7.m128i_i64[0] + 33) & 0xC) != 0 )
    goto LABEL_71;
  if ( (*(_WORD *)(v7.m128i_i64[0] + 32) & 0x380) != 0 )
    goto LABEL_71;
  v36 = *(_QWORD *)(v7.m128i_i64[0] + 56);
  if ( !v36 )
    goto LABEL_71;
  v37 = 1;
  do
  {
    if ( v421.m128i_i32[2] == *(_DWORD *)(v36 + 8) )
    {
      if ( !v37 )
        goto LABEL_71;
      v36 = *(_QWORD *)(v36 + 32);
      if ( !v36 )
        goto LABEL_203;
      if ( *(_DWORD *)(v36 + 8) == v421.m128i_i32[2] )
        goto LABEL_71;
      v37 = 0;
    }
    v36 = *(_QWORD *)(v36 + 32);
  }
  while ( v36 );
  if ( v37 == 1 )
    goto LABEL_71;
LABEL_203:
  v165 = 16LL * v421.m128i_u32[2];
  v166 = (_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40));
  if ( *(_WORD *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + v165) == 16 || *v166 == 1 )
  {
    v167 = sub_2E79000(*(__int64 **)(*a1 + 40));
    v168 = 1;
    v169 = (_BYTE *)v167;
    v170 = (unsigned __int16)v422;
    if ( *v169 )
      goto LABEL_207;
  }
  else
  {
    v168 = *(_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40));
    if ( v168 )
      goto LABEL_71;
    v170 = (unsigned __int16)v422;
  }
  if ( v168 != ((_WORD)v170 == 16) )
  {
LABEL_71:
    *(_QWORD *)&v415 = a1[1];
    goto LABEL_72;
  }
LABEL_207:
  if ( !*((_BYTE *)a1 + 33) )
  {
    v171 = *(_QWORD *)(v7.m128i_i64[0] + 112);
    if ( (*(_BYTE *)(v171 + 37) & 0xF) == 0 && (*(_BYTE *)(v7.m128i_i64[0] + 32) & 8) == 0 )
    {
      *(_QWORD *)&v415 = a1[1];
      goto LABEL_211;
    }
  }
  v199 = 1;
  *(_QWORD *)&v415 = a1[1];
  if ( (_WORD)v170 == 1 || (_WORD)v170 && (v199 = (unsigned __int16)v170, *(_QWORD *)(v415 + 8 * v170 + 112)) )
  {
    if ( !*(_BYTE *)(v415 + 500LL * v199 + 6712) )
    {
      v171 = *(_QWORD *)(v7.m128i_i64[0] + 112);
LABEL_211:
      v172 = (unsigned __int16 *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + v165);
      a2 = *v172;
      v173 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v415 + 248LL))(
               v415,
               a2,
               *((_QWORD *)v172 + 1),
               (unsigned int)v422,
               v423,
               *a1,
               v171);
      a6 = v361;
      if ( v173 )
      {
        v174 = *(_QWORD *)(v7.m128i_i64[0] + 112);
        v175 = *(_QWORD **)(v7.m128i_i64[0] + 40);
        v176 = *a1;
        v440 = *(_QWORD *)(v12 + 80);
        if ( v440 )
        {
          *(_QWORD *)&v414 = v174;
          *(_QWORD *)&v415 = &v440;
          sub_325F5D0(&v440);
          v174 = v414;
        }
        v177 = *(_DWORD *)(v12 + 72);
        *(_QWORD *)&v415 = &v440;
        LODWORD(v441) = v177;
        v178 = sub_33F1A60(v176, v422, v423, (unsigned int)&v440, *v175, v175[1], v175[5], v175[6], v174);
        sub_9C6650(&v440);
        sub_34161C0(*a1, v7.m128i_i64[0], 1, v178, 1);
        return v178;
      }
      goto LABEL_71;
    }
  }
LABEL_72:
  *(_QWORD *)&v414 = *a1;
  v38 = *(unsigned __int16 **)(v12 + 48);
  v39 = *v38;
  v40 = *((_QWORD *)v38 + 1);
  v41 = *(unsigned int **)(v12 + 40);
  LOWORD(v426) = v39;
  v427 = v40;
  v42 = *(_QWORD *)v41;
  v43 = *(_QWORD *)(*(_QWORD *)v41 + 48LL) + 16LL * v41[2];
  v44 = *(_WORD *)v43;
  v45 = *(_QWORD *)(v43 + 8);
  LOWORD(v428) = v44;
  v429 = v45;
  if ( v39 )
  {
    if ( (unsigned __int16)(v39 - 10) > 6u
      && (unsigned __int16)(v39 - 126) > 0x31u
      && (unsigned __int16)(v39 - 208) > 0x14u )
    {
      goto LABEL_91;
    }
    if ( (unsigned __int16)(v39 - 17) > 0xD3u )
    {
      LOWORD(v438) = v39;
      v439 = v40;
      goto LABEL_78;
    }
    v39 = word_4456580[v39 - 1];
    v156 = 0;
  }
  else
  {
    if ( !(unsigned __int8)sub_3007030((__int64)&v426) )
      goto LABEL_91;
    if ( !sub_30070B0((__int64)&v426) )
    {
      v439 = v40;
      LOWORD(v438) = 0;
LABEL_136:
      v102 = sub_3007260((__int64)&v438);
      v104 = v103;
      v105 = v102;
      v106 = v104;
      v434 = v105;
      v46 = v105;
      v435 = v106;
      goto LABEL_137;
    }
    v196 = sub_3009970((__int64)&v426, a2, v99, v100, v101);
    v198 = v197;
    v39 = v196;
    v156 = v198;
  }
  LOWORD(v438) = v39;
  v439 = v156;
  if ( !v39 )
    goto LABEL_136;
LABEL_78:
  if ( v39 == 1 || (unsigned __int16)(v39 - 504) <= 7u )
    goto LABEL_328;
  v46 = *(_QWORD *)&byte_444C4A0[16 * v39 - 16];
LABEL_137:
  v107 = v428;
  if ( (_WORD)v428 )
  {
    if ( (unsigned __int16)(v428 - 17) > 0xD3u )
    {
LABEL_139:
      v108 = v429;
      goto LABEL_140;
    }
    v107 = word_4456580[(unsigned __int16)v428 - 1];
    v108 = 0;
  }
  else
  {
    v125 = sub_30070B0((__int64)&v428);
    v107 = 0;
    if ( !v125 )
      goto LABEL_139;
    v128 = sub_3009970((__int64)&v428, a2, 0, v126, v127);
    v130 = v129;
    v107 = v128;
    v108 = v130;
  }
LABEL_140:
  LOWORD(v432) = v107;
  v433 = v108;
  if ( v107 )
  {
    if ( v107 == 1 || (unsigned __int16)(v107 - 504) <= 7u )
      goto LABEL_328;
    v109 = *(_QWORD *)&byte_444C4A0[16 * v107 - 16];
  }
  else
  {
    v109 = sub_3007260((__int64)&v432);
    v436 = v109;
    v437 = v110;
  }
  if ( v46 != v109 )
    goto LABEL_91;
  LODWORD(v431) = 1;
  v430 = 0;
  v111 = *(_DWORD *)(v42 + 24);
  if ( v111 != 187 )
  {
    if ( v111 != 188 )
    {
      if ( v111 != 186 || (unsigned int)(*(_DWORD *)(*(_QWORD *)v414 + 544LL) - 42) <= 1 )
        goto LABEL_91;
      v112 = sub_32844A0((unsigned __int16 *)&v428, a2);
      LODWORD(v441) = v112;
      v113 = v112;
      a6 = v112;
      if ( v112 > 0x40 )
      {
        a2 = 0;
        sub_C43690((__int64)&v440, 0, 0);
        a6 = v441;
        v114 = 1LL << ((unsigned __int8)v113 - 1);
        if ( (unsigned int)v441 > 0x40 )
        {
          *(_QWORD *)(v440 + 8LL * ((v113 - 1) >> 6)) |= v114;
          a6 = v441;
          if ( (unsigned int)v441 > 0x40 )
          {
            sub_C43D10((__int64)&v440);
            a6 = v441;
            v118 = (_QWORD *)v440;
LABEL_153:
            v430 = v118;
            v119 = 245;
            LODWORD(v431) = a6;
            goto LABEL_154;
          }
          v115 = v440;
LABEL_150:
          v116 = ~v115;
          v117 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a6;
          if ( !a6 )
            v117 = 0;
          v118 = (_QWORD *)(v116 & v117);
          goto LABEL_153;
        }
      }
      else
      {
        v440 = 0;
        v114 = 1LL << ((unsigned __int8)v112 - 1);
      }
      v115 = v440 | v114;
      goto LABEL_150;
    }
    v161 = sub_32844A0((unsigned __int16 *)&v428, a2);
    LODWORD(v441) = v161;
    v162 = v161;
    a6 = v161;
    if ( v161 > 0x40 )
    {
      a2 = 0;
      sub_C43690((__int64)&v440, 0, 0);
      a6 = v441;
      v163 = 1LL << ((unsigned __int8)v162 - 1);
      if ( (unsigned int)v441 > 0x40 )
      {
        *(_QWORD *)(v440 + 8LL * ((v162 - 1) >> 6)) |= v163;
        v164 = (_QWORD *)v440;
        a6 = v441;
        goto LABEL_202;
      }
    }
    else
    {
      v440 = 0;
      v163 = 1LL << ((unsigned __int8)v161 - 1);
    }
    v164 = (_QWORD *)(v440 | v163);
LABEL_202:
    v430 = v164;
    v119 = 244;
    LODWORD(v431) = a6;
    goto LABEL_154;
  }
  v179 = v428;
  if ( (_WORD)v428 )
  {
    if ( (unsigned __int16)(v428 - 17) > 0xD3u )
    {
LABEL_217:
      v180 = v429;
      goto LABEL_218;
    }
    v179 = word_4456580[(unsigned __int16)v428 - 1];
    v180 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v428) )
      goto LABEL_217;
    v179 = sub_3009970((__int64)&v428, a2, v200, v201, v202);
    v180 = v203;
  }
LABEL_218:
  LOWORD(v440) = v179;
  v441 = v180;
  if ( !v179 )
  {
    v181 = sub_3007260((__int64)&v440);
    v438 = v181;
    v439 = v182;
    goto LABEL_220;
  }
  if ( v179 == 1 || (unsigned __int16)(v179 - 504) <= 7u )
LABEL_328:
    BUG();
  v181 = *(_QWORD *)&byte_444C4A0[16 * v179 - 16];
LABEL_220:
  LODWORD(v441) = v181;
  a6 = v181;
  v183 = 1LL << ((unsigned __int8)v181 - 1);
  if ( (unsigned int)v181 > 0x40 )
  {
    a2 = 0;
    v398 = v181 - 1;
    sub_C43690((__int64)&v440, 0, 0);
    a6 = v441;
    if ( (unsigned int)v441 > 0x40 )
    {
      *(_QWORD *)(v440 + 8LL * (v398 >> 6)) |= v183;
      v184 = (_QWORD *)v440;
      a6 = v441;
      goto LABEL_223;
    }
  }
  else
  {
    v440 = 0;
  }
  v184 = (_QWORD *)(v440 | v183);
LABEL_223:
  v430 = v184;
  v119 = 245;
  LODWORD(v431) = a6;
LABEL_154:
  if ( *((_BYTE *)a1 + 33) )
  {
    v120 = 1;
    if ( (_WORD)v426 != 1 )
    {
      if ( !(_WORD)v426 )
        goto LABEL_160;
      v120 = (unsigned __int16)v426;
      if ( !*(_QWORD *)(v415 + 8LL * (unsigned __int16)v426 + 112) )
        goto LABEL_160;
    }
    if ( *(_BYTE *)(v119 + v415 + 500LL * v120 + 6414) )
      goto LABEL_160;
  }
  v121 = *(const __m128i **)(v42 + 40);
  v395 = a6;
  a2 = v121[3].m128i_i64[0];
  v378 = v121->m128i_i64[0];
  v371 = (__int128)_mm_loadu_si128(v121);
  v122 = sub_33DFBC0(v121[2].m128i_i64[1], a2, 1, 0);
  a6 = v395;
  if ( !v122 )
    goto LABEL_160;
  v123 = *(_QWORD *)(v122 + 96);
  if ( *(_DWORD *)(v123 + 32) <= 0x40u )
  {
    if ( *(_QWORD **)(v123 + 24) != v430 )
      goto LABEL_160;
  }
  else
  {
    a2 = (__int64)&v430;
    v124 = sub_C43C50(v123 + 24, (const void **)&v430);
    a6 = v395;
    if ( !v124 )
    {
LABEL_160:
      if ( a6 <= 0x40 )
        goto LABEL_91;
      result = 0;
      goto LABEL_162;
    }
  }
  a2 = (unsigned __int16)v426;
  if ( *(_DWORD *)(v378 + 24) != 234
    || (v213 = *(_QWORD *)(**(_QWORD **)(v378 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v378 + 40) + 8LL),
        (_WORD)v426 != *(_WORD *)v213)
    || *(_QWORD *)(v213 + 8) != v427 && !(_WORD)v426 )
  {
    v204 = *(_QWORD *)v415;
    if ( v119 == 245 )
    {
      v205 = *(__int64 (**)())(v204 + 1600);
      if ( v205 == sub_2D566B0 )
        goto LABEL_160;
    }
    else
    {
      v205 = *(__int64 (**)())(v204 + 1592);
      if ( v205 == sub_2FE3530 )
        goto LABEL_160;
    }
    v399 = a6;
    a2 = v426;
    v206 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64))v205)(v415, v426, v427);
    a6 = v399;
    if ( !v206 )
      goto LABEL_160;
  }
  v440 = *(_QWORD *)(v12 + 80);
  if ( v440 )
  {
    v400 = a6;
    *(_QWORD *)&v415 = &v440;
    sub_325F5D0(&v440);
    a6 = v400;
  }
  v381 = a6;
  LODWORD(v441) = *(_DWORD *)(v12 + 72);
  *(_QWORD *)&v415 = sub_33FAF80(v414, 234, (unsigned int)&v440, v426, v427, a6, v371);
  *((_QWORD *)&v415 + 1) = v207;
  v208 = v381;
  if ( v440 )
  {
    sub_B91220((__int64)&v440, v440);
    v208 = v381;
  }
  v440 = *(_QWORD *)(v12 + 80);
  if ( v440 )
  {
    v382 = v208;
    sub_325F5D0(&v440);
    v208 = v382;
  }
  v383 = v208;
  LODWORD(v441) = *(_DWORD *)(v12 + 72);
  *(_QWORD *)&v209 = sub_33FAF80(v414, v119, (unsigned int)&v440, v426, v427, v208, v415);
  a2 = v440;
  v415 = v209;
  a6 = v383;
  if ( v440 )
  {
    sub_B91220((__int64)&v440, v440);
    a6 = v383;
  }
  if ( *(_DWORD *)(v42 + 24) == 187 )
  {
    v440 = *(_QWORD *)(v12 + 80);
    if ( v440 )
    {
      v384 = a6;
      sub_325F5D0(&v440);
      a6 = v384;
    }
    a2 = 244;
    v385 = a6;
    LODWORD(v441) = *(_DWORD *)(v12 + 72);
    *(_QWORD *)&v415 = sub_33FAF80(v414, 244, (unsigned int)&v440, v426, v427, a6, v415);
    sub_9C6650(&v440);
    a6 = v385;
    result = v415;
  }
  else
  {
    result = v415;
  }
  if ( a6 > 0x40 )
  {
LABEL_162:
    if ( v430 )
    {
      *(_QWORD *)&v415 = result;
      j_j___libc_free_0_0((unsigned __int64)v430);
      result = v415;
    }
  }
  if ( result )
    return result;
LABEL_91:
  v50 = *(_DWORD *)(v7.m128i_i64[0] + 24);
  if ( v50 != 244 )
  {
LABEL_92:
    if ( v50 != 245 )
      goto LABEL_93;
    v193 = a1[1];
    v194 = *(__int64 (**)())(*(_QWORD *)v193 + 1600LL);
    a2 = *(unsigned __int16 *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2]);
    if ( v194 != sub_2D566B0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v194)(
             v193,
             a2,
             *(_QWORD *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2] + 8)) )
      {
        goto LABEL_93;
      }
    }
    goto LABEL_176;
  }
  v131 = a1[1];
  v132 = *(__int64 (**)())(*(_QWORD *)v131 + 1592LL);
  a2 = *(unsigned __int16 *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2]);
  if ( v132 != sub_2FE3530
    && ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v132)(
         v131,
         a2,
         *(_QWORD *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2] + 8)) )
  {
    v50 = *(_DWORD *)(v7.m128i_i64[0] + 24);
    goto LABEL_92;
  }
LABEL_176:
  v133 = *(_QWORD *)(v7.m128i_i64[0] + 56);
  if ( v133 )
  {
    if ( !*(_QWORD *)(v133 + 32) )
    {
      *(_QWORD *)&v414 = &v422;
      if ( sub_3280180((__int64)&v422) && !sub_32801E0((__int64)&v422) )
      {
        v134 = *(_QWORD *)(v7.m128i_i64[0] + 48);
        *(_QWORD *)&v415 = &v440;
        v135 = 16LL * v421.m128i_u32[2] + v134;
        v136 = *(_QWORD *)(v135 + 8);
        LOWORD(v135) = *(_WORD *)v135;
        v441 = v136;
        LOWORD(v440) = v135;
        if ( !sub_32801E0((__int64)&v440) )
        {
          v396 = v415;
          *(_QWORD *)&v137 = sub_33FB890(
                               *a1,
                               (unsigned int)v422,
                               v423,
                               **(_QWORD **)(v7.m128i_i64[0] + 40),
                               *(_QWORD *)(*(_QWORD *)(v7.m128i_i64[0] + 40) + 8LL));
          v415 = v137;
          sub_32B3E80((__int64)a1, v137, 1, 0, v138, v139);
          v140 = v396;
          v428 = *(_QWORD *)(v12 + 80);
          if ( v428 )
          {
            sub_325F5D0(&v428);
            v140 = v396;
          }
          LODWORD(v429) = *(_DWORD *)(v12 + 72);
          if ( *(_WORD *)(*(_QWORD *)(v7.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2]) == 16 && !*((_BYTE *)a1 + 34) )
          {
            v411 = (_QWORD *)v140;
            v307 = *a1;
            sub_3285E70(v140, v421.m128i_i64[0]);
            v308 = (_QWORD *)sub_2D5B750((unsigned __int16 *)v414);
            v431 = v309;
            v430 = v308;
            v310 = sub_CA1930(&v430);
            sub_986680((__int64)&v432, v310 >> 1);
            v311.m128i_i64[0] = sub_34007B0(v307, (unsigned int)&v432, (_DWORD)v411, 8, 0, 0, 0);
            v414 = (__int128)v311;
            v312 = v311.m128i_i64[0];
            sub_969240((__int64 *)&v432);
            sub_9C6650(v411);
            if ( *(_DWORD *)(v7.m128i_i64[0] + 24) == 244 )
            {
              v391 = v411;
              si128 = _mm_load_si128((const __m128i *)&v414);
              v325 = si128.m128i_u32[2];
              sub_32B3E80((__int64)a1, v414, 1, 0, v313, v314);
            }
            else
            {
              v315 = *a1;
              sub_3285E70((__int64)v411, v415);
              v316 = (unsigned __int8 *)sub_2E79000(*(__int64 **)(*a1 + 40));
              v391 = v411;
              *(_QWORD *)&v317 = sub_3400D50(v315, *v316, v411, 0);
              v412 = v317;
              sub_3285E70((__int64)&v432, v415);
              *(_QWORD *)&v412 = sub_3406EB0(v315, 53, (unsigned int)&v432, 8, 0, v318, v415, v412);
              *((_QWORD *)&v412 + 1) = v319;
              sub_9C6650(&v432);
              sub_9C6650(v391);
              sub_32B3E80((__int64)a1, v412, 1, 0, v320, v321);
              v322 = *a1;
              sub_3285E70((__int64)v391, v421.m128i_i64[0]);
              v418 = sub_3406EB0(v322, 186, (_DWORD)v391, 8, 0, v323, v412, v414);
              v312 = v418;
              v419 = v324;
              v325 = (unsigned int)v324;
              sub_9C6650(v391);
              sub_32B3E80((__int64)a1, v418, 1, 0, v326, v327);
            }
            *(_QWORD *)&v414 = v391;
            v328 = *a1;
            sub_3285E70((__int64)v391, v421.m128i_i64[0]);
            *((_QWORD *)&v359 + 1) = v325;
            *(_QWORD *)&v359 = v312;
            *((_QWORD *)&v352 + 1) = v325;
            *(_QWORD *)&v352 = v312;
            v330 = sub_3406EB0(v328, 54, v414, v422, v423, v329, v352, v359);
            v332 = v331;
            sub_9C6650((_QWORD *)v414);
            sub_32B3E80((__int64)a1, v330, 1, 0, v333, v334);
            *((_QWORD *)&v360 + 1) = v332;
            *(_QWORD *)&v360 = v330;
            v353 = v415;
            v335 = *a1;
            *(_QWORD *)&v415 = &v428;
            v155 = sub_3406EB0(v335, 188, (unsigned int)&v428, v422, v423, v336, v353, v360);
            v153 = (__int64 *)v415;
          }
          else
          {
            v379 = (_BYTE *)v140;
            v141 = sub_2D5B750((unsigned __int16 *)v414);
            v441 = v142;
            *(_QWORD *)&v414 = v379;
            v440 = v141;
            v143 = sub_CA1930(v379);
            sub_986680((__int64)&v430, v143);
            if ( *(_DWORD *)(v7.m128i_i64[0] + 24) == 244 )
            {
              v265 = *a1;
              *(_QWORD *)&v266 = sub_34007B0(v265, (unsigned int)&v430, (unsigned int)&v428, v422, v423, 0, 0);
              v151 = sub_3406EB0(v265, 188, (unsigned int)&v428, v422, v423, v267, v415, v266);
            }
            else
            {
              v144 = *a1;
              sub_9865C0((__int64)&v432, (__int64)&v430);
              sub_987160((__int64)&v432, (__int64)&v430, v145, v146, v147);
              LODWORD(v441) = v433;
              v380 = (__int64 *)v414;
              LODWORD(v433) = 0;
              v440 = (__int64)v432;
              *(_QWORD *)&v148 = sub_34007B0(v144, v414, (unsigned int)&v428, v422, v423, 0, 0);
              *(_QWORD *)&v414 = sub_3406EB0(v144, 186, (unsigned int)&v428, v422, v423, v149, v415, v148);
              *(_QWORD *)&v415 = v150;
              sub_969240(v380);
              sub_969240((__int64 *)&v432);
              v151 = v414;
              v152 = v415;
            }
            *(_QWORD *)&v414 = v152;
            *(_QWORD *)&v415 = v151;
            sub_969240((__int64 *)&v430);
            v153 = &v428;
            v154 = v414;
            v155 = v415;
          }
          *(_QWORD *)&v414 = v154;
          *(_QWORD *)&v415 = v155;
          sub_9C6650(v153);
          return v415;
        }
      }
    }
  }
LABEL_93:
  v51 = v421.m128i_i64[0];
  v52 = *(_DWORD *)(v421.m128i_i64[0] + 24);
  if ( v52 == 152 )
  {
    v185 = *(_QWORD *)(v421.m128i_i64[0] + 56);
    if ( !v185 )
      goto LABEL_112;
    if ( *(_QWORD *)(v185 + 32) )
      goto LABEL_112;
    v186 = *(_QWORD *)(v421.m128i_i64[0] + 40);
    v187 = *(_DWORD *)(*(_QWORD *)v186 + 24LL);
    if ( v187 != 36 && v187 != 12 )
      goto LABEL_112;
    *(_QWORD *)&v414 = &v422;
    if ( !sub_3280180((__int64)&v422) || sub_32801E0(v414) )
      goto LABEL_112;
    v188 = sub_3262090(*(_QWORD *)(v186 + 40), *(_DWORD *)(v186 + 48));
    *(_QWORD *)&v415 = &v440;
    v441 = v189;
    v440 = v188;
    v397 = sub_CA1930(&v440);
    v190 = sub_327FC40(*(_QWORD **)(*a1 + 64), v397);
    v17 = *((_BYTE *)a1 + 34) == 0;
    v191 = &v440;
    v424 = v190;
    v425 = v192;
    if ( v17 || (a2 = (__int64)&v424, (unsigned __int8)sub_325E6A0(a1[1], (unsigned __int16 *)&v424)) )
    {
      v386 = v191;
      *(_QWORD *)&v214 = sub_33FB890(
                           *a1,
                           v424,
                           v425,
                           *(_QWORD *)(*(_QWORD *)(v421.m128i_i64[0] + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v421.m128i_i64[0] + 40) + 48LL));
      v215 = DWORD2(v214);
      v216 = v214;
      v415 = v214;
      sub_32B3E80((__int64)a1, v214, 1, 0, v217, v218);
      v440 = sub_2D5B750((unsigned __int16 *)v414);
      v441 = v219;
      v220 = sub_CA1930(v386);
      v222 = v386;
      if ( v397 >= v220 )
      {
        if ( v397 > v220 )
        {
          v251 = 16 * v215;
          v374 = v220;
          sub_3285E70((__int64)&v432, v415);
          v252 = v397;
          v406 = *a1;
          *(_QWORD *)&v254 = sub_3400BD0(
                               *a1,
                               v252 - v374,
                               (unsigned int)&v432,
                               *(unsigned __int16 *)(v251 + *(_QWORD *)(v216 + 48)),
                               *(_QWORD *)(v251 + *(_QWORD *)(v216 + 48) + 8),
                               0,
                               0,
                               v253);
          v256 = sub_3406EB0(
                   v406,
                   192,
                   (unsigned int)&v432,
                   *(unsigned __int16 *)(*(_QWORD *)(v216 + 48) + v251),
                   *(_QWORD *)(*(_QWORD *)(v216 + 48) + v251 + 8),
                   v255,
                   v415,
                   v254);
          LODWORD(v251) = v257;
          v407 = v256;
          sub_32B3E80((__int64)a1, v256, 1, 0, v258, v259);
          v260 = *a1;
          *(_QWORD *)&v415 = v407;
          *((_QWORD *)&v415 + 1) = (unsigned int)v251 | *((_QWORD *)&v415 + 1) & 0xFFFFFFFF00000000LL;
          sub_3285E70((__int64)v386, v407);
          v216 = sub_33FAF80(
                   v260,
                   216,
                   (_DWORD)v386,
                   v422,
                   v423,
                   v261,
                   __PAIR128__(*((unsigned __int64 *)&v415 + 1), v407));
          v215 = v262;
          sub_9C6650(v386);
          sub_32B3E80((__int64)a1, v216, 1, 0, v263, v264);
          sub_9C6650(&v432);
          v222 = v386;
        }
      }
      else
      {
        v223 = *a1;
        v440 = *(_QWORD *)(v12 + 80);
        if ( v440 )
        {
          sub_325F5D0(v386);
          v222 = v386;
        }
        v401 = v222;
        LODWORD(v441) = *(_DWORD *)(v12 + 72);
        v216 = sub_33FAF80(v223, 213, (_DWORD)v222, v422, v423, v221, v415);
        v215 = v224;
        sub_9C6650(v401);
        sub_32B3E80((__int64)a1, v216, 1, 0, v225, v226);
        v222 = v401;
      }
      if ( *(_WORD *)(*(_QWORD *)(v421.m128i_i64[0] + 48) + 16LL * v421.m128i_u32[2]) == 16 && !*((_BYTE *)a1 + 34) )
      {
        *(_QWORD *)&v415 = v222;
        v268 = sub_2D5B750((unsigned __int16 *)v414);
        v441 = v269;
        v440 = v268;
        v270 = sub_CA1930((_BYTE *)v415);
        sub_986680((__int64)&v430, v270 >> 1);
        v271.m128i_i64[0] = sub_33FB890(
                              *a1,
                              (unsigned int)v422,
                              v423,
                              **(_QWORD **)(v421.m128i_i64[0] + 40),
                              *(_QWORD *)(*(_QWORD *)(v421.m128i_i64[0] + 40) + 8LL));
        v414 = (__int128)v271;
        sub_32B3E80((__int64)a1, v271.m128i_i64[0], 1, 0, v272, v273);
        v274 = sub_33FB890(
                 *a1,
                 (unsigned int)v422,
                 v423,
                 *(_QWORD *)(*(_QWORD *)(v421.m128i_i64[0] + 40) + 40LL),
                 *(_QWORD *)(*(_QWORD *)(v421.m128i_i64[0] + 40) + 48LL));
        v276 = v275;
        sub_32B3E80((__int64)a1, v274, 1, 0, v277, v278);
        v408 = *a1;
        sub_3285E70(v415, v421.m128i_i64[0]);
        *((_QWORD *)&v356 + 1) = v276;
        *(_QWORD *)&v356 = v274;
        v279 = sub_3406EB0(v408, 188, v415, v422, v423, v408, v414, v356);
        v281 = v280;
        sub_9C6650((_QWORD *)v415);
        sub_32B3E80((__int64)a1, v279, 1, 0, v282, v283);
        v389 = *a1;
        sub_3285E70(v415, v279);
        v284 = (unsigned __int8 *)sub_2E79000(*(__int64 **)(*a1 + 40));
        v375 = (_QWORD *)v415;
        *(_QWORD *)&v285 = sub_3400D50(v389, *v284, v415, 0);
        v415 = v285;
        sub_3285E70((__int64)&v432, v279);
        *((_QWORD *)&v350 + 1) = v281;
        *(_QWORD *)&v350 = v279;
        v286 = sub_3406EB0(v389, 53, (unsigned int)&v432, 8, 0, v389, v350, v415);
        v288 = v287;
        sub_9C6650(&v432);
        sub_9C6650(v375);
        *(_QWORD *)&v415 = v286;
        *((_QWORD *)&v415 + 1) = v288;
        sub_32B3E80((__int64)a1, v286, 1, 0, v289, v290);
        v291 = *a1;
        sub_3285E70((__int64)v375, v415);
        *(_QWORD *)&v292 = sub_34007B0(v291, (unsigned int)&v430, (_DWORD)v375, 8, 0, 0, 0);
        v390 = v292;
        sub_3285E70((__int64)&v432, v415);
        *((_QWORD *)&v348 + 1) = v288;
        *(_QWORD *)&v348 = v415;
        v293 = sub_3406EB0(v291, 186, (unsigned int)&v432, 8, 0, DWORD2(v390), v348, v390);
        v295 = v294;
        sub_9C6650(&v432);
        *(_QWORD *)&v415 = v375;
        sub_9C6650(v375);
        sub_32B3E80((__int64)a1, v293, 1, 0, v296, v297);
        v409 = *a1;
        sub_3285E70((__int64)v375, v421.m128i_i64[0]);
        *((_QWORD *)&v357 + 1) = v295;
        *(_QWORD *)&v357 = v293;
        *((_QWORD *)&v351 + 1) = v295;
        *(_QWORD *)&v351 = v293;
        v298 = sub_3406EB0(v409, 54, (_DWORD)v375, v422, v423, v409, v351, v357);
        v300 = v299;
        sub_9C6650(v375);
        sub_32B3E80((__int64)a1, v298, 1, 0, v301, v302);
        v304 = *a1;
        v305 = v375;
        v440 = *(_QWORD *)(v12 + 80);
        if ( v440 )
        {
          sub_325F5D0((__int64 *)v415);
          v305 = (_QWORD *)v415;
        }
        *((_QWORD *)&v358 + 1) = v300;
        *(_QWORD *)&v358 = v298;
        v410 = v305;
        LODWORD(v441) = *(_DWORD *)(v12 + 72);
        *(_QWORD *)&v414 = sub_3406EB0(v304, 188, (_DWORD)v305, v422, v423, v303, v414, v358);
        *(_QWORD *)&v415 = v306;
        sub_9C6650(v410);
        sub_969240((__int64 *)&v430);
        return v414;
      }
      else
      {
        v402 = v222;
        v227 = sub_2D5B750((unsigned __int16 *)v414);
        v441 = v228;
        v387 = v402;
        v440 = v227;
        v229 = sub_CA1930(v402);
        *(_QWORD *)&v414 = &v426;
        sub_986680((__int64)&v426, v229);
        *(_QWORD *)&v415 = v216;
        v403 = *a1;
        *((_QWORD *)&v415 + 1) = v215 | *((_QWORD *)&v415 + 1) & 0xFFFFFFFF00000000LL;
        sub_3285E70((__int64)v387, v216);
        *(_QWORD *)&v230 = sub_34007B0(v403, (unsigned int)&v426, (_DWORD)v387, v422, v423, 0, 0);
        v372 = v230;
        sub_3285E70((__int64)&v432, v216);
        v368 = sub_3406EB0(
                 v403,
                 186,
                 (unsigned int)&v432,
                 v422,
                 v423,
                 v231,
                 __PAIR128__(*((unsigned __int64 *)&v415 + 1), v216),
                 v372);
        v365 = v232;
        sub_9C6650(&v432);
        sub_9C6650(v387);
        sub_32B3E80((__int64)a1, v368, 1, 0, v233, v234);
        v235 = sub_33FB890(
                 *a1,
                 (unsigned int)v422,
                 v423,
                 **(_QWORD **)(v421.m128i_i64[0] + 40),
                 *(_QWORD *)(*(_QWORD *)(v421.m128i_i64[0] + 40) + 8LL));
        v237 = v236;
        v238 = v235;
        v404 = *a1;
        sub_3285E70((__int64)v387, v235);
        sub_9865C0((__int64)&v430, (__int64)&v426);
        sub_987160((__int64)&v430, (__int64)&v426, v239, v240, v241);
        LODWORD(v433) = (_DWORD)v431;
        v432 = v430;
        LODWORD(v431) = 0;
        *(_QWORD *)&v242 = sub_34007B0(v404, (unsigned int)&v432, (_DWORD)v387, v422, v423, 0, 0);
        v373 = v242;
        sub_3285E70((__int64)&v428, v238);
        *((_QWORD *)&v347 + 1) = v237;
        *(_QWORD *)&v347 = v238;
        v416 = sub_3406EB0(v404, 186, (unsigned int)&v428, v422, v423, (unsigned int)&v428, v347, v373);
        v417 = v243;
        v244 = (unsigned int)v243 | v237 & 0xFFFFFFFF00000000LL;
        sub_9C6650(&v428);
        sub_969240((__int64 *)&v432);
        sub_969240((__int64 *)&v430);
        sub_9C6650(v387);
        sub_32B3E80((__int64)a1, v416, 1, 0, v245, v246);
        v248 = *a1;
        v249 = v387;
        v440 = *(_QWORD *)(v12 + 80);
        if ( v440 )
        {
          sub_325F5D0(v387);
          v249 = v387;
        }
        *((_QWORD *)&v355 + 1) = v244;
        *(_QWORD *)&v355 = v416;
        LODWORD(v441) = *(_DWORD *)(v12 + 72);
        v388 = v249;
        *((_QWORD *)&v349 + 1) = v365 | *((_QWORD *)&v415 + 1) & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v349 = v368;
        v405 = sub_3406EB0(v248, 187, (_DWORD)v249, v422, v423, v247, v349, v355);
        *(_QWORD *)&v415 = v250;
        sub_9C6650(v388);
        sub_969240((__int64 *)v414);
        return v405;
      }
    }
    v52 = *(_DWORD *)(v421.m128i_i64[0] + 24);
  }
  if ( v52 == 54 )
  {
    a2 = v421.m128i_i64[0];
    result = sub_328E660(a1, v421.m128i_i64[0], v422, v423);
    if ( result )
      return result;
    v52 = *(_DWORD *)(v421.m128i_i64[0] + 24);
  }
  if ( v52 != 167 )
    goto LABEL_112;
  if ( (_WORD)v422 )
  {
    if ( (unsigned __int16)(v422 - 2) > 7u )
      return 0;
  }
  else if ( !sub_30070A0((__int64)&v422) )
  {
    return 0;
  }
  v53 = *(unsigned int **)(v421.m128i_i64[0] + 40);
  v54 = *(_OWORD *)v53;
  v55 = *(_QWORD *)(*(_QWORD *)v53 + 48LL) + 16LL * v53[2];
  v56 = *(_WORD *)v55;
  v57 = *(_QWORD *)(v55 + 8);
  LOWORD(v440) = v56;
  v441 = v57;
  if ( v56 )
  {
    v58 = (unsigned __int16)(v56 - 2) <= 7u;
  }
  else
  {
    v415 = v54;
    v58 = sub_30070A0((__int64)&v440);
    *(_QWORD *)&v54 = v415;
  }
  if ( !v58 )
  {
LABEL_112:
    if ( *((int *)a1 + 6) <= 2 )
    {
      if ( (_WORD)v422 )
      {
        if ( *(_QWORD *)(a1[1] + 8LL * (unsigned __int16)v422 + 112) )
        {
          if ( (unsigned __int16)(v422 - 17) <= 0xD3u && v52 == 165 )
          {
            if ( (unsigned __int8)sub_3286E00(&v421) )
            {
              *(_QWORD *)&v414 = &v422;
              v62 = sub_3281500(&v422, a2);
              v63 = *(_QWORD *)(v51 + 48);
              *(_QWORD *)&v415 = &v432;
              v64 = 16LL * v421.m128i_u32[2];
              v65 = (__int16 *)(v64 + v63);
              v66 = *v65;
              v67 = *((_QWORD *)v65 + 1);
              LOWORD(v432) = v66;
              v433 = v67;
              if ( v62 >= (unsigned int)sub_3281500(&v432, a2) )
              {
                v393 = (__int64 *)v415;
                v68 = sub_3281500((_WORD *)v414, a2);
                v69 = *(_QWORD *)(v51 + 48);
                *(_QWORD *)&v415 = &v440;
                v70 = (__int16 *)(v64 + v69);
                v71 = *v70;
                v72 = *((_QWORD *)v70 + 1);
                LOWORD(v440) = v71;
                v441 = v72;
                if ( !(v68 % (unsigned int)sub_3281500(&v440, a2)) )
                {
                  v431 = a1;
                  v369 = (_WORD *)v415;
                  v430 = (_QWORD *)v414;
                  v366 = v393;
                  *(_QWORD *)&v415 = sub_326AC40(
                                       &v430,
                                       **(_QWORD **)(v51 + 40),
                                       *(_QWORD *)(*(_QWORD *)(v51 + 40) + 8LL));
                  v73 = *(_QWORD *)(v51 + 40);
                  v377 = v74;
                  v75 = *(_QWORD *)(v73 + 40);
                  v76 = sub_326AC40(&v430, v75, *(_QWORD *)(v73 + 48));
                  v394 = v76;
                  v413 = v77;
                  if ( (_QWORD)v415 )
                  {
                    if ( v76 )
                    {
                      v78 = sub_3281500((_WORD *)v414, v75);
                      v79 = *(_QWORD *)(v51 + 48) + v64;
                      v80 = v78;
                      v81 = *(_QWORD *)(v79 + 8);
                      LOWORD(v440) = *(_WORD *)v79;
                      v441 = v81;
                      v82 = sub_3281500(v369, v75);
                      v440 = (__int64)v442;
                      v83 = v80 / v82;
                      v441 = 0x800000000LL;
                      v84 = sub_3288400(v51, v75);
                      v87 = v366;
                      v88 = v369;
                      v90 = v84 + 4 * v89;
                      v91 = v12;
                      *(_QWORD *)&v414 = v90;
                      v92 = v442;
                      v93 = (int *)v84;
                      v94 = a1;
                      while ( (int *)v414 != v93 )
                      {
                        v95 = *v93;
                        if ( v83 )
                        {
                          v370 = v93;
                          v96 = 0;
                          v97 = (__int64)v88;
                          do
                          {
                            v367 = v94;
                            v98 = v96 + v83 * v95;
                            v362 = v92;
                            if ( v95 < 0 )
                              v98 = -1;
                            ++v96;
                            v363 = v87;
                            v364 = v91;
                            sub_10E2270(v97, v98, v91, (__int64)v92, v85, v86);
                            v94 = v367;
                            v91 = v364;
                            v87 = v363;
                            v92 = v362;
                          }
                          while ( v83 != v96 );
                          v88 = (_WORD *)v97;
                          v93 = v370;
                        }
                        ++v93;
                      }
                      v337 = v92;
                      v338 = v94[1];
                      v339 = v91;
                      v340 = *v94;
                      v341 = v440;
                      v432 = *(_QWORD **)(v91 + 80);
                      v342 = (unsigned int)v441;
                      if ( v432 )
                      {
                        v376 = v338;
                        *(_QWORD *)&v414 = v87;
                        sub_325F5D0(v87);
                        LODWORD(v338) = v376;
                        v87 = (__int64 *)v414;
                      }
                      v343 = v415;
                      LODWORD(v433) = *(_DWORD *)(v339 + 72);
                      *(_QWORD *)&v415 = v87;
                      v344 = sub_3449A00(v338, v422, v423, (_DWORD)v87, v343, v377, v394, v413, v341, v342, v340);
                      v346 = v345;
                      sub_9C6650((_QWORD *)v415);
                      if ( v344 )
                      {
                        result = v344;
                        if ( (_BYTE *)v440 != v337 )
                        {
                          *(_QWORD *)&v414 = v346;
                          _libc_free(v440);
                          return v344;
                        }
                        return result;
                      }
                      if ( (_BYTE *)v440 != v337 )
                        _libc_free(v440);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  v59 = *a1;
  v440 = *(_QWORD *)(v12 + 80);
  if ( v440 )
  {
    v414 = v54;
    *(_QWORD *)&v415 = &v440;
    sub_325F5D0(&v440);
    v54 = v414;
  }
  LODWORD(v441) = *(_DWORD *)(v12 + 72);
  *(_QWORD *)&v415 = &v440;
  result = sub_33FAF80(v59, 215, (unsigned int)&v440, v422, v423, a6, v54);
  v28 = v440;
  if ( v440 )
    goto LABEL_104;
  return result;
}
