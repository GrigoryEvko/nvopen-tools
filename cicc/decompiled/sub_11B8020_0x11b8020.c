// Function: sub_11B8020
// Address: 0x11b8020
//
unsigned __int8 *__fastcall sub_11B8020(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  unsigned __int8 *v4; // rbx
  __int64 v5; // r15
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm3
  __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // r9
  unsigned __int8 *v12; // r10
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  int v28; // edx
  __int64 v29; // rsi
  char v30; // al
  _BYTE *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // ebx
  char *v35; // rax
  unsigned __int8 *v36; // r15
  __int64 v37; // rax
  unsigned __int8 *v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // rcx
  unsigned __int64 v41; // r11
  _QWORD *v42; // rax
  unsigned int **v43; // rdi
  __int64 v44; // r12
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // r8
  char v49; // al
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // r15
  unsigned int v53; // r9d
  __int64 **v54; // r8
  _BYTE *v55; // r15
  __int64 v56; // rcx
  unsigned int v57; // r9d
  __int64 v58; // rsi
  _BYTE *v59; // rbx
  __int64 v60; // rax
  int v61; // eax
  bool v62; // al
  _QWORD *v63; // rdi
  unsigned __int64 v64; // rax
  _QWORD *v65; // r15
  __int64 v66; // r8
  unsigned int v67; // edx
  char *v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rbx
  unsigned int v72; // r15d
  __int64 *v73; // r15
  __int64 v74; // rax
  __int64 *v75; // rax
  __int64 v76; // r12
  _BYTE *v77; // rbx
  __int64 v78; // r13
  __int64 v79; // rsi
  char v80; // al
  unsigned __int64 *v81; // r11
  char v82; // r13
  char v83; // al
  _BYTE *v84; // rdi
  __int64 v85; // r13
  _BYTE *v86; // rax
  int v87; // ebx
  int v88; // eax
  __int64 v89; // rax
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 v94; // rbx
  _QWORD *v95; // rax
  int v96; // edx
  unsigned __int8 *v97; // rdx
  __int64 v98; // rsi
  int v99; // eax
  int v100; // eax
  unsigned __int8 *v101; // rdx
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rcx
  __int64 v106; // rdi
  unsigned int **v107; // rdi
  __int64 v108; // rax
  __int64 v109; // r12
  __int64 v110; // r13
  int v111; // eax
  unsigned __int8 *v112; // rdx
  unsigned __int8 *v113; // r10
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rdx
  int v118; // eax
  int v119; // eax
  __int64 *v120; // rdx
  __int64 v121; // rdi
  unsigned __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rax
  bool v125; // zf
  __int64 **v126; // rdi
  __int64 v127; // r9
  __int64 *v128; // rbx
  __int64 v129; // r13
  __int64 v130; // r12
  _QWORD *v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rbx
  unsigned int v134; // r15d
  __int64 v135; // rax
  unsigned int v136; // r10d
  __int64 v137; // rbx
  unsigned int v138; // r15d
  __int64 v139; // rax
  unsigned __int64 v140; // rdx
  __m128i *v141; // rax
  __m128i *v142; // rcx
  char *v143; // rax
  char *k; // rcx
  __int64 *v145; // rdi
  __int64 m; // rax
  __int64 v147; // rcx
  __int64 *v148; // rsi
  __int64 v149; // rbx
  __int64 v150; // r14
  __int64 v151; // r15
  __int64 v152; // r13
  unsigned __int64 v153; // r12
  __int64 v154; // rax
  _QWORD *v155; // rax
  __int64 v156; // rax
  __int64 v157; // rbx
  __int64 v158; // r15
  __int64 v159; // rdx
  unsigned int v160; // esi
  unsigned __int8 *v161; // rbx
  __int64 v162; // rdx
  unsigned int v163; // r15d
  __int64 v164; // rdx
  unsigned int v165; // eax
  int v166; // esi
  __int64 v167; // r9
  __int64 v168; // rcx
  int v169; // edx
  unsigned __int64 v170; // rdx
  __m128i *v171; // r10
  __m128i *v172; // rcx
  __int64 v173; // rsi
  char *v174; // rdx
  char *j; // rcx
  __int64 v176; // r13
  __int64 v177; // r12
  __int64 *v178; // rdi
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 *v181; // rbx
  __int64 v182; // rax
  __int64 v183; // r10
  _BYTE *v184; // rax
  _BYTE *v185; // rax
  _BYTE *v186; // rdx
  _BYTE *v187; // rcx
  __int64 v188; // r15
  _QWORD *v189; // rax
  __int64 v190; // r9
  __int64 v191; // rdx
  int v192; // eax
  __int64 v193; // rbx
  __int64 v194; // r8
  void *v195; // rcx
  __int64 v196; // rdx
  int v197; // eax
  __int64 v198; // rbx
  __int64 **v199; // r15
  _QWORD *v200; // rax
  __int64 v201; // rax
  __int64 v202; // r8
  __int64 v203; // r9
  __int64 v204; // r15
  __int64 v205; // rax
  int v206; // eax
  bool v207; // al
  _QWORD *v208; // rax
  __int64 v209; // r9
  __int64 v210; // r10
  void *v211; // r12
  __int64 v212; // rsi
  __int64 v213; // rbx
  void *v214; // rdi
  int v215; // eax
  int v216; // eax
  __int64 v217; // r8
  __int64 v218; // rbx
  unsigned int v219; // r15d
  __int64 v220; // rbx
  __int64 v221; // r15
  unsigned __int8 **v222; // rdx
  __int64 v223; // r8
  __int64 v224; // rdx
  unsigned __int8 *v225; // rcx
  __int64 v226; // rdx
  unsigned int v227; // r15d
  int v228; // eax
  unsigned __int64 v229; // rax
  int v230; // r9d
  char *v231; // rax
  char *ii; // rdx
  __int64 v233; // rax
  int v234; // edx
  __int64 v235; // rcx
  unsigned int v236; // edx
  unsigned __int64 v237; // rax
  _QWORD *v238; // rax
  char v239; // cl
  __int64 v240; // r9
  _BYTE *v241; // rax
  __int64 v242; // rbx
  __int64 v243; // rdx
  __int64 v244; // rax
  __int64 v245; // rcx
  __int64 v246; // rax
  __int64 v247; // rdx
  __int64 v248; // r15
  __int64 v249; // rdx
  unsigned int v250; // edx
  unsigned int v251; // edi
  _QWORD *v252; // rcx
  int v253; // edx
  _QWORD *v254; // rax
  __int64 v255; // r12
  __int64 v256; // r13
  __int64 v257; // rdx
  unsigned int v258; // esi
  __int64 v259; // rdx
  int v260; // ebx
  __int64 v261; // r14
  __int64 v262; // r13
  __int64 kk; // rbx
  __int64 v264; // rdx
  unsigned int v265; // esi
  _QWORD *v266; // rax
  __int64 v267; // r14
  __int64 v268; // r12
  __int64 i; // rbx
  __int64 v270; // rdx
  unsigned int v271; // esi
  __int64 v272; // rax
  _QWORD *v273; // rax
  __int64 v274; // r9
  __int64 v275; // r14
  __int64 v276; // r12
  __int64 jj; // rbx
  __int64 v278; // rdx
  unsigned int v279; // esi
  __int64 v280; // r15
  _BYTE *v281; // rax
  unsigned int v282; // r15d
  int v283; // eax
  int v284; // eax
  bool v285; // r15
  unsigned int n; // ebx
  __int64 v287; // rax
  unsigned int v288; // r15d
  _QWORD *v289; // rax
  __int64 v290; // r9
  __int64 v291; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 *v292; // [rsp+10h] [rbp-1D0h]
  unsigned int v293; // [rsp+10h] [rbp-1D0h]
  unsigned int v294; // [rsp+10h] [rbp-1D0h]
  unsigned int v295; // [rsp+18h] [rbp-1C8h]
  char v296; // [rsp+18h] [rbp-1C8h]
  __int64 v297; // [rsp+18h] [rbp-1C8h]
  __int64 v298; // [rsp+18h] [rbp-1C8h]
  __int64 v299; // [rsp+18h] [rbp-1C8h]
  unsigned __int64 v300; // [rsp+20h] [rbp-1C0h]
  __int64 v301; // [rsp+20h] [rbp-1C0h]
  __int64 v302; // [rsp+20h] [rbp-1C0h]
  __int64 v303; // [rsp+20h] [rbp-1C0h]
  unsigned __int8 *v304; // [rsp+28h] [rbp-1B8h]
  unsigned int v305; // [rsp+28h] [rbp-1B8h]
  unsigned __int64 v306; // [rsp+28h] [rbp-1B8h]
  int v307; // [rsp+28h] [rbp-1B8h]
  __int64 v308; // [rsp+28h] [rbp-1B8h]
  __int64 v309; // [rsp+28h] [rbp-1B8h]
  __int64 v310; // [rsp+28h] [rbp-1B8h]
  unsigned __int64 v311; // [rsp+28h] [rbp-1B8h]
  __int64 v312; // [rsp+28h] [rbp-1B8h]
  unsigned __int8 *v313; // [rsp+28h] [rbp-1B8h]
  __int64 **v314; // [rsp+30h] [rbp-1B0h]
  __int64 v315; // [rsp+30h] [rbp-1B0h]
  _BYTE *v316; // [rsp+30h] [rbp-1B0h]
  __int64 v317; // [rsp+30h] [rbp-1B0h]
  __int64 v318; // [rsp+30h] [rbp-1B0h]
  unsigned int v319; // [rsp+30h] [rbp-1B0h]
  int v320; // [rsp+30h] [rbp-1B0h]
  _BYTE *v321; // [rsp+30h] [rbp-1B0h]
  __int64 v322; // [rsp+30h] [rbp-1B0h]
  __int64 v323; // [rsp+30h] [rbp-1B0h]
  unsigned int v324; // [rsp+30h] [rbp-1B0h]
  __int64 v325; // [rsp+30h] [rbp-1B0h]
  __int64 v326; // [rsp+30h] [rbp-1B0h]
  int v327; // [rsp+30h] [rbp-1B0h]
  unsigned __int8 *v328; // [rsp+40h] [rbp-1A0h]
  __int64 v329; // [rsp+40h] [rbp-1A0h]
  int v330; // [rsp+40h] [rbp-1A0h]
  int v331; // [rsp+40h] [rbp-1A0h]
  __int64 *v332; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v333; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v334; // [rsp+40h] [rbp-1A0h]
  __int64 v335; // [rsp+40h] [rbp-1A0h]
  __int64 v336; // [rsp+40h] [rbp-1A0h]
  unsigned int v337; // [rsp+40h] [rbp-1A0h]
  void *v338; // [rsp+40h] [rbp-1A0h]
  __int64 v339; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v340; // [rsp+40h] [rbp-1A0h]
  unsigned int v341; // [rsp+40h] [rbp-1A0h]
  __int64 v342; // [rsp+40h] [rbp-1A0h]
  unsigned int v343; // [rsp+40h] [rbp-1A0h]
  unsigned __int64 v344; // [rsp+40h] [rbp-1A0h]
  __int64 v345; // [rsp+40h] [rbp-1A0h]
  __int64 v346; // [rsp+40h] [rbp-1A0h]
  __int64 v347; // [rsp+40h] [rbp-1A0h]
  void *v348; // [rsp+40h] [rbp-1A0h]
  __int64 v349; // [rsp+40h] [rbp-1A0h]
  __int64 v350; // [rsp+50h] [rbp-190h]
  unsigned int v351; // [rsp+50h] [rbp-190h]
  __int64 v352; // [rsp+50h] [rbp-190h]
  __int64 v353; // [rsp+50h] [rbp-190h]
  __int64 v354; // [rsp+50h] [rbp-190h]
  __int64 v355; // [rsp+50h] [rbp-190h]
  __int64 v356; // [rsp+50h] [rbp-190h]
  void *v357; // [rsp+50h] [rbp-190h]
  __int64 v358; // [rsp+50h] [rbp-190h]
  __int64 v359; // [rsp+50h] [rbp-190h]
  unsigned int v360; // [rsp+50h] [rbp-190h]
  __int64 v361; // [rsp+50h] [rbp-190h]
  __int64 v362; // [rsp+50h] [rbp-190h]
  __int64 v363; // [rsp+50h] [rbp-190h]
  void *v364; // [rsp+50h] [rbp-190h]
  unsigned __int8 *v365; // [rsp+50h] [rbp-190h]
  int v366; // [rsp+50h] [rbp-190h]
  int v367; // [rsp+50h] [rbp-190h]
  unsigned __int64 v368; // [rsp+50h] [rbp-190h]
  __int64 v369; // [rsp+60h] [rbp-180h]
  char v370; // [rsp+60h] [rbp-180h]
  __int64 v371; // [rsp+60h] [rbp-180h]
  __int64 v372; // [rsp+60h] [rbp-180h]
  _BYTE *v373; // [rsp+60h] [rbp-180h]
  __int64 v374; // [rsp+60h] [rbp-180h]
  unsigned int v375; // [rsp+60h] [rbp-180h]
  __int64 v376; // [rsp+60h] [rbp-180h]
  __int64 v377; // [rsp+60h] [rbp-180h]
  _BYTE *v378; // [rsp+60h] [rbp-180h]
  _QWORD *v379; // [rsp+60h] [rbp-180h]
  __int64 v380; // [rsp+60h] [rbp-180h]
  __int64 v381; // [rsp+60h] [rbp-180h]
  __int64 v382; // [rsp+60h] [rbp-180h]
  __int64 v383; // [rsp+60h] [rbp-180h]
  __int64 v384; // [rsp+60h] [rbp-180h]
  __int64 v385; // [rsp+60h] [rbp-180h]
  __int64 v386; // [rsp+60h] [rbp-180h]
  __int64 v387; // [rsp+60h] [rbp-180h]
  __int64 v388; // [rsp+60h] [rbp-180h]
  __int64 v389; // [rsp+60h] [rbp-180h]
  __int64 v390; // [rsp+60h] [rbp-180h]
  __int64 v391; // [rsp+60h] [rbp-180h]
  int v392; // [rsp+60h] [rbp-180h]
  unsigned __int64 v393; // [rsp+60h] [rbp-180h]
  unsigned __int64 v394; // [rsp+60h] [rbp-180h]
  __int64 v395; // [rsp+60h] [rbp-180h]
  __int64 v396; // [rsp+60h] [rbp-180h]
  __int64 v397; // [rsp+60h] [rbp-180h]
  __int64 v398; // [rsp+60h] [rbp-180h]
  unsigned __int64 v399; // [rsp+70h] [rbp-170h] BYREF
  __int64 v400; // [rsp+78h] [rbp-168h] BYREF
  _QWORD v401[2]; // [rsp+80h] [rbp-160h]
  _QWORD v402[2]; // [rsp+90h] [rbp-150h]
  unsigned __int64 v403; // [rsp+A0h] [rbp-140h] BYREF
  const void **v404; // [rsp+A8h] [rbp-138h]
  __int16 v405; // [rsp+C0h] [rbp-120h]
  char *v406; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v407; // [rsp+D8h] [rbp-108h]
  _BYTE v408[16]; // [rsp+E0h] [rbp-100h] BYREF
  __int16 v409; // [rsp+F0h] [rbp-F0h]
  void *s[2]; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v411; // [rsp+130h] [rbp-B0h] BYREF
  _QWORD v412[2]; // [rsp+140h] [rbp-A0h] BYREF
  __m128i v413; // [rsp+150h] [rbp-90h]
  __int64 v414; // [rsp+160h] [rbp-80h]

  v2 = (__int64)a1;
  v3 = a2;
  v4 = *(unsigned __int8 **)(a2 - 96);
  v350 = a2;
  v5 = *(_QWORD *)(a2 - 32);
  v6 = _mm_loadu_si128(a1 + 6);
  v7 = _mm_loadu_si128(a1 + 7);
  v369 = *(_QWORD *)(a2 - 64);
  v8 = _mm_loadu_si128(a1 + 9);
  v9 = a1[10].m128i_i64[0];
  v412[0] = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v412[1] = a2;
  v414 = v9;
  *(__m128i *)s = v6;
  v411 = v7;
  v413 = v8;
  v10 = sub_10031B0((__int64)v4, v369, v5, (__int64)s);
  v12 = (unsigned __int8 *)v369;
  if ( v10 )
    return sub_F162A0((__int64)a1, a2, v10);
  if ( *(_BYTE *)v5 != 17 )
  {
    v28 = *v4;
    goto LABEL_16;
  }
  v14 = sub_11AEFE0(v5);
  v12 = (unsigned __int8 *)v369;
  v16 = v14;
  if ( v14 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v17 = *(_QWORD *)(a2 - 8);
    else
      v17 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v18 = *(_QWORD **)(v17 + 64);
    if ( v18 )
    {
      v19 = *(_QWORD *)(v17 + 72);
      **(_QWORD **)(v17 + 80) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v17 + 80);
    }
    *(_QWORD *)(v17 + 64) = v16;
    v20 = *(_QWORD *)(v16 + 16);
    *(_QWORD *)(v17 + 72) = v20;
    if ( v20 )
    {
      v15 = v17 + 72;
      *(_QWORD *)(v20 + 16) = v17 + 72;
    }
    *(_QWORD *)(v17 + 80) = v16 + 16;
    *(_QWORD *)(v16 + 16) = v17 + 64;
    if ( *(_BYTE *)v18 > 0x1Cu )
    {
      v21 = a1[2].m128i_i64[1];
      s[0] = v18;
      v22 = v21 + 2096;
      sub_11B4E60(v22, (__int64 *)s, v20, v16, v15, (__int64)v11);
      v27 = v18[2];
      if ( v27 )
      {
        if ( !*(_QWORD *)(v27 + 8) )
        {
          s[0] = *(void **)(v27 + 24);
          sub_11B4E60(v22, (__int64 *)s, v23, v24, v25, v26);
        }
      }
    }
    return (unsigned __int8 *)v350;
  }
  v37 = *((_QWORD *)v4 + 2);
  v28 = *v4;
  if ( v37 && !*(_QWORD *)(v37 + 8) && (_BYTE)v28 == 91 )
  {
    if ( (v4[7] & 0x40) != 0 )
    {
      v38 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
      v39 = *(_BYTE **)v38;
      if ( !*(_QWORD *)v38 )
        goto LABEL_16;
    }
    else
    {
      v38 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
      v39 = *(_BYTE **)v38;
      if ( !*(_QWORD *)v38 )
        goto LABEL_16;
    }
    v11 = (_BYTE *)*((_QWORD *)v38 + 4);
    if ( v11 )
    {
      v40 = *((_QWORD *)v38 + 8);
      if ( *(_BYTE *)v40 == 17 )
      {
        v295 = *(_DWORD *)(v40 + 32);
        if ( v295 > 0x40 )
        {
          v340 = (unsigned __int8 *)v369;
          v321 = (_BYTE *)*((_QWORD *)v38 + 4);
          v389 = *((_QWORD *)v38 + 8);
          v216 = sub_C444A0(v40 + 24);
          v12 = v340;
          v28 = 91;
          if ( v295 - v216 > 0x40 )
            goto LABEL_16;
          v11 = v321;
          v41 = **(_QWORD **)(v389 + 24);
        }
        else
        {
          v41 = *(_QWORD *)(v40 + 24);
        }
        if ( *v11 > 0x15u )
        {
          v42 = *(_QWORD **)(v5 + 24);
          if ( *(_DWORD *)(v5 + 32) > 0x40u )
            v42 = (_QWORD *)*v42;
          if ( (unsigned __int64)v42 < v41 )
          {
            v43 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v412[0]) = 257;
            v329 = (__int64)v11;
            v371 = v41;
            v44 = sub_A83A20(v43, v39, v12, (_BYTE *)v5, (__int64)s);
            v45 = *(_QWORD *)(v2 + 32);
            LOWORD(v412[0]) = 257;
            v46 = sub_BCB2E0(*(_QWORD **)(v45 + 72));
            v47 = sub_ACD640(v46, v371, 0);
            v350 = (__int64)sub_BD2C40(72, 3u);
            if ( v350 )
              sub_B4DFA0(v350, v44, v329, v47, (__int64)s, v329, 0, 0);
            return (unsigned __int8 *)v350;
          }
        }
      }
    }
  }
LABEL_16:
  if ( (unsigned __int8)(v28 - 12) <= 1u )
    goto LABEL_515;
  if ( (unsigned __int8)(v28 - 9) > 2u )
    goto LABEL_32;
  v29 = (__int64)v4;
  v304 = v12;
  s[1] = v412;
  v406 = v408;
  s[0] = 0;
  v411.m128i_i64[0] = 8;
  v411.m128i_i32[2] = 0;
  v411.m128i_i8[12] = 1;
  v407 = 0x800000000LL;
  v403 = (unsigned __int64)s;
  v404 = (const void **)&v406;
  v30 = sub_AA8FD0(&v403, (__int64)v4);
  v12 = v304;
  v370 = v30;
  if ( v30 )
  {
    do
    {
      v31 = v406;
      if ( !(_DWORD)v407 )
      {
        v12 = v304;
        goto LABEL_25;
      }
      v29 = *(_QWORD *)&v406[8 * (unsigned int)v407 - 8];
      LODWORD(v407) = v407 - 1;
    }
    while ( (unsigned __int8)sub_AA8FD0(&v403, v29) );
    v12 = v304;
  }
  v370 = 0;
  v31 = v406;
LABEL_25:
  if ( v31 != v408 )
  {
    v328 = v12;
    _libc_free(v31, v29);
    v12 = v328;
  }
  if ( !v411.m128i_i8[12] )
  {
    v334 = v12;
    _libc_free(s[1], v29);
    v12 = v334;
  }
  if ( v370 )
  {
LABEL_515:
    v32 = *((_QWORD *)v12 + 2);
    if ( v32 )
    {
      if ( !*(_QWORD *)(v32 + 8) )
      {
        v118 = *v12;
        if ( (unsigned __int8)v118 > 0x1Cu )
        {
          v119 = v118 - 29;
          goto LABEL_198;
        }
        if ( (_BYTE)v118 == 5 )
        {
          v119 = *((unsigned __int16 *)v12 + 1);
LABEL_198:
          if ( v119 == 49 )
          {
            v120 = (v12[7] & 0x40) != 0
                 ? (__int64 *)*((_QWORD *)v12 - 1)
                 : (__int64 *)&v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
            v376 = *v120;
            if ( *v120 )
            {
              if ( (v121 = *(_QWORD *)(v376 + 8), v122 = *(unsigned __int8 *)(v121 + 8), (unsigned __int8)v122 <= 0xCu)
                && (v123 = 4143, _bittest64(&v123, v122))
                || (v122 & 0xFD) == 4 )
              {
                v124 = *(_QWORD *)(v3 + 8);
                v125 = *(_BYTE *)(v124 + 8) == 18;
                LODWORD(v124) = *(_DWORD *)(v124 + 32);
                BYTE4(v403) = v125;
                LODWORD(v403) = v124;
                v126 = (__int64 **)sub_BCE1B0((__int64 *)v121, v403);
                if ( *v4 == 13 )
                  v127 = sub_ACADE0(v126);
                else
                  v127 = sub_ACA8A0(v126);
                v128 = *(__int64 **)(v2 + 32);
                v409 = 257;
                v353 = v127;
                v129 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v128[10] + 104LL))(
                         v128[10],
                         v127,
                         v376,
                         v5);
                if ( !v129 )
                {
                  LOWORD(v412[0]) = 257;
                  v155 = sub_BD2C40(72, 3u);
                  v129 = (__int64)v155;
                  if ( v155 )
                    sub_B4DFA0((__int64)v155, v353, v376, v5, (__int64)s, v353, 0, 0);
                  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v128[11] + 16LL))(
                    v128[11],
                    v129,
                    &v406,
                    v128[7],
                    v128[8]);
                  v156 = 16LL * *((unsigned int *)v128 + 2);
                  v157 = *v128;
                  v158 = v157 + v156;
                  while ( v158 != v157 )
                  {
                    v159 = *(_QWORD *)(v157 + 8);
                    v160 = *(_DWORD *)v157;
                    v157 += 16;
                    sub_B99FD0(v129, v160, v159);
                  }
                }
                v130 = *(_QWORD *)(v3 + 8);
                LOWORD(v412[0]) = 257;
                v131 = sub_BD2C40(72, unk_3F10A14);
                v350 = (__int64)v131;
                if ( v131 )
                  sub_B51BF0((__int64)v131, v129, v130, (__int64)s, 0, 0);
                return (unsigned __int8 *)v350;
              }
            }
          }
        }
      }
    }
  }
  v28 = *v4;
LABEL_32:
  if ( (unsigned __int8)v28 > 0x1Cu )
  {
    v96 = v28 - 29;
  }
  else
  {
    if ( (_BYTE)v28 != 5 )
      goto LABEL_34;
    v96 = *((unsigned __int16 *)v4 + 1);
  }
  if ( v96 == 49 )
  {
    v97 = (v4[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v4 - 1) : &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
    v98 = *(_QWORD *)v97;
    if ( *(_QWORD *)v97 )
    {
      v99 = *v12;
      if ( (unsigned __int8)v99 <= 0x1Cu )
      {
        if ( (_BYTE)v99 != 5 )
          goto LABEL_34;
        v100 = *((unsigned __int16 *)v12 + 1);
      }
      else
      {
        v100 = v99 - 29;
      }
      if ( v100 == 49 )
      {
        v101 = (v12[7] & 0x40) != 0
             ? (unsigned __int8 *)*((_QWORD *)v12 - 1)
             : &v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
        v102 = *(_QWORD *)v101;
        if ( v102 )
        {
          if ( (v103 = *((_QWORD *)v4 + 2)) != 0 && !*(_QWORD *)(v103 + 8)
            || (v104 = *((_QWORD *)v12 + 2)) != 0 && !*(_QWORD *)(v104 + 8) )
          {
            v105 = *(_QWORD *)(v98 + 8);
            if ( (unsigned int)*(unsigned __int8 *)(v105 + 8) - 17 <= 1 )
            {
              v106 = *(_QWORD *)(v102 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v106 + 8) - 17 > 1 && v106 == *(_QWORD *)(v105 + 24) )
              {
                v107 = *(unsigned int ***)(v2 + 32);
                LOWORD(v412[0]) = 257;
                v108 = sub_A83A20(v107, (_BYTE *)v98, (_BYTE *)v102, (_BYTE *)v5, (__int64)s);
                v109 = *(_QWORD *)(v3 + 8);
                LOWORD(v412[0]) = 257;
                v110 = v108;
                v350 = (__int64)sub_BD2C40(72, unk_3F10A14);
                if ( v350 )
                  sub_B51BF0(v350, v110, v109, (__int64)s, 0, 0);
                return (unsigned __int8 *)v350;
              }
            }
          }
        }
      }
    }
  }
LABEL_34:
  if ( *(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL) == 17 && *(_BYTE *)v5 == 17 )
  {
    v375 = *(_DWORD *)(v5 + 32);
    if ( v375 <= 0x40 || (v333 = v12, v111 = sub_C444A0(v5 + 24), v12 = v333, v375 - v111 <= 0x40) )
    {
      s[0] = &v400;
      s[1] = &v399;
      if ( *v12 == 90 )
      {
        v112 = (v12[7] & 0x40) != 0
             ? (unsigned __int8 *)*((_QWORD *)v12 - 1)
             : &v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v112 )
        {
          v400 = *(_QWORD *)v112;
          v113 = (v12[7] & 0x40) != 0
               ? (unsigned __int8 *)*((_QWORD *)v12 - 1)
               : &v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
          if ( (unsigned __int8)sub_11B1B00((_QWORD **)&s[1], *((_QWORD *)v113 + 4)) )
          {
            v114 = *(_QWORD *)(v400 + 8);
            if ( *(_BYTE *)(v114 + 8) == 17 && *(unsigned int *)(v114 + 32) > v399 )
            {
              v115 = *(_QWORD *)(v3 + 16);
              if ( !v115 || *(_QWORD *)(v115 + 8) || **(_BYTE **)(v115 + 24) != 91 )
              {
                while ( 1 )
                {
                  LOBYTE(v403) = 0;
                  s[0] = &v411;
                  s[1] = (void *)0x1000000000LL;
                  v116 = sub_11B54A0(v3, (__int64)s, 0, v2, (__int64)&v403, (__int64)v11);
                  if ( v3 != v117 && v3 != v116 )
                    break;
                  if ( s[0] != &v411 )
                    _libc_free(s[0], s);
                  if ( !(_BYTE)v403 )
                    goto LABEL_35;
                }
                v210 = v116;
                if ( !v117 )
                {
                  v396 = v116;
                  v272 = sub_ACADE0(*(__int64 ***)(v116 + 8));
                  v210 = v396;
                  v117 = v272;
                }
                v339 = v210;
                v387 = v117;
                v211 = s[0];
                v212 = unk_3F1FE60;
                v213 = LODWORD(s[1]);
                v409 = 257;
                v350 = (__int64)sub_BD2C40(112, unk_3F1FE60);
                if ( v350 )
                {
                  v212 = v339;
                  sub_B4E9E0(v350, v339, v387, v211, v213, (__int64)&v406, 0, 0);
                }
                v214 = s[0];
                if ( s[0] == &v411 )
                  return (unsigned __int8 *)v350;
LABEL_349:
                _libc_free(v214, v212);
                return (unsigned __int8 *)v350;
              }
            }
          }
        }
      }
    }
  }
LABEL_35:
  v33 = *((_QWORD *)v4 + 1);
  if ( *(_BYTE *)(v33 + 8) == 17 )
  {
    v34 = *(_DWORD *)(v33 + 32);
    LODWORD(v404) = v34;
    if ( v34 > 0x40 )
    {
      sub_C43690((__int64)&v403, 0, 0);
      LODWORD(v407) = v34;
      sub_C43690((__int64)&v406, -1, 1);
      LODWORD(s[1]) = v407;
      if ( (unsigned int)v407 > 0x40 )
      {
        sub_C43780((__int64)s, (const void **)&v406);
LABEL_41:
        v36 = sub_11A3F30(v2, (unsigned __int8 *)v3, (__int64)s, (__int64 *)&v403, 0, 0);
        if ( LODWORD(s[1]) > 0x40 && s[0] )
          j_j___libc_free_0_0(s[0]);
        if ( v36 )
        {
          if ( (unsigned __int8 *)v3 != v36 )
            v350 = (__int64)sub_F162A0(v2, v3, (__int64)v36);
          if ( (unsigned int)v407 > 0x40 && v406 )
            j_j___libc_free_0_0(v406);
          if ( (unsigned int)v404 > 0x40 && v403 )
            j_j___libc_free_0_0(v403);
          return (unsigned __int8 *)v350;
        }
        if ( (unsigned int)v407 > 0x40 && v406 )
          j_j___libc_free_0_0(v406);
        if ( (unsigned int)v404 > 0x40 && v403 )
          j_j___libc_free_0_0(v403);
        goto LABEL_74;
      }
    }
    else
    {
      v403 = 0;
      LODWORD(v407) = v34;
      v35 = (char *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v34);
      LODWORD(s[1]) = v34;
      if ( !v34 )
        v35 = 0;
      v406 = v35;
    }
    s[0] = v406;
    goto LABEL_41;
  }
LABEL_74:
  v48 = *(_QWORD *)(v3 - 96);
  v49 = *(_BYTE *)v48;
  if ( *(_BYTE *)v48 <= 0x1Cu )
    goto LABEL_76;
  v50 = *(_QWORD *)(v48 + 16);
  if ( !v50 || *(_QWORD *)(v50 + 8) )
    goto LABEL_76;
  if ( v49 == 92 )
  {
    v161 = *(unsigned __int8 **)(v48 - 32);
    if ( *v161 > 0x15u )
      goto LABEL_306;
    v316 = *(_BYTE **)(v3 - 64);
    if ( *v316 > 0x15u )
      goto LABEL_306;
    v162 = *(_QWORD *)(v3 - 32);
    if ( *(_BYTE *)v162 != 17 )
      goto LABEL_306;
    v163 = *(_DWORD *)(v162 + 32);
    if ( v163 > 0x40 )
    {
      v358 = *(_QWORD *)(v3 - 96);
      v385 = *(_QWORD *)(v3 - 32);
      v197 = sub_C444A0(v162 + 24);
      v48 = v358;
      if ( v163 - v197 > 0x40 )
        goto LABEL_306;
      v377 = **(_QWORD **)(v385 + 24);
    }
    else
    {
      v377 = *(_QWORD *)(v162 + 24);
    }
    v164 = *(_QWORD *)(*(_QWORD *)(v48 - 64) + 8LL);
    if ( *(_BYTE *)(v164 + 8) != 18 )
    {
      v165 = *(_DWORD *)(v48 + 80);
      v166 = *(_DWORD *)(v164 + 32);
      if ( v165 == v166 )
      {
        if ( !v165 )
        {
          s[0] = &v411;
          s[1] = (void *)0x1000000000LL;
          v178 = (__int64 *)&v411;
          v406 = v408;
          v407 = 0x1000000000LL;
LABEL_428:
          v193 = *(_QWORD *)(v48 - 64);
          v325 = sub_AD3730(v178, LODWORD(s[1]));
          v364 = v406;
          v173 = unk_3F1FE60;
          v395 = (unsigned int)v407;
          v405 = 257;
          v179 = (__int64)sub_BD2C40(112, unk_3F1FE60);
          if ( !v179 )
            goto LABEL_319;
          v195 = v364;
          v194 = v395;
          v196 = v325;
LABEL_318:
          v173 = v193;
          v382 = v179;
          sub_B4E9E0(v179, v193, v196, v195, v194, (__int64)&v403, 0, 0);
          v179 = v382;
LABEL_319:
          if ( v406 != v408 )
          {
            v383 = v179;
            _libc_free(v406, v173);
            v179 = v383;
          }
          if ( s[0] != &v411 )
          {
            v384 = v179;
            _libc_free(s[0], v173);
            v179 = v384;
          }
          if ( v179 )
            return (unsigned __int8 *)v179;
          v48 = *(_QWORD *)(v3 - 96);
          v49 = *(_BYTE *)v48;
LABEL_292:
          v181 = *(__int64 **)(v2 + 32);
          if ( v49 == 91 )
          {
            v182 = *(_QWORD *)(v48 + 16);
            if ( v182 )
            {
              if ( !*(_QWORD *)(v182 + 8) )
              {
                v183 = *(_QWORD *)(v48 - 96);
                if ( v183 )
                {
                  v184 = *(_BYTE **)(v48 - 64);
                  v378 = v184;
                  if ( v184 )
                  {
                    if ( *v184 > 0x15u )
                    {
                      v185 = *(_BYTE **)(v48 - 32);
                      v355 = (__int64)v185;
                      if ( *v185 == 17 )
                      {
                        v186 = *(_BYTE **)(v3 - 64);
                        if ( *v186 <= 0x15u )
                        {
                          v187 = *(_BYTE **)(v3 - 32);
                          if ( !v187 )
                            BUG();
                          if ( *v187 == 17 && v185 != v187 )
                          {
                            v409 = 257;
                            v308 = (__int64)v187;
                            v317 = (__int64)v186;
                            v335 = v183;
                            v188 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v181[10] + 104LL))(
                                     v181[10],
                                     v183);
                            if ( !v188 )
                            {
                              LOWORD(v412[0]) = 257;
                              v266 = sub_BD2C40(72, 3u);
                              v188 = (__int64)v266;
                              if ( v266 )
                                sub_B4DFA0((__int64)v266, v335, v317, v308, (__int64)s, 0, 0, 0);
                              (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v181[11]
                                                                                                  + 16LL))(
                                v181[11],
                                v188,
                                &v406,
                                v181[7],
                                v181[8]);
                              v267 = v3;
                              v268 = *v181 + 16LL * *((unsigned int *)v181 + 2);
                              for ( i = *v181; v268 != i; i += 16 )
                              {
                                v270 = *(_QWORD *)(i + 8);
                                v271 = *(_DWORD *)i;
                                sub_B99FD0(v188, v271, v270);
                              }
                              v3 = v267;
                            }
                            LOWORD(v412[0]) = 257;
                            v189 = sub_BD2C40(72, 3u);
                            if ( v189 )
                            {
                              v191 = (__int64)v378;
                              v379 = v189;
                              sub_B4DFA0((__int64)v189, v188, v191, v355, (__int64)s, v190, 0, 0);
                              return (unsigned __int8 *)v379;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          goto LABEL_76;
        }
        v167 = *(_QWORD *)(v48 + 72);
        v168 = 0;
        v354 = v165 - 1;
        while ( 1 )
        {
          v169 = *(_DWORD *)(v167 + 4 * v168);
          if ( v169 != -1 && v169 != (_DWORD)v168 && v169 != v166 + (_DWORD)v168 )
            goto LABEL_306;
          if ( v165 - 1 == v168 )
            break;
          ++v168;
        }
        v170 = v165;
        s[0] = &v411;
        s[1] = (void *)0x1000000000LL;
        if ( v165 > 0x10uLL )
        {
          v293 = v165;
          v298 = v167;
          v302 = v48;
          v310 = 8LL * v165;
          v344 = v165;
          sub_C8D5F0((__int64)s, &v411, v165, 8u, v48, v167);
          v170 = v344;
          v172 = (__m128i *)((char *)s[0] + 8 * LODWORD(s[1]));
          v48 = v302;
          v167 = v298;
          v171 = (__m128i *)((char *)s[0] + v310);
          v165 = v293;
          if ( v172 == (__m128i *)((char *)s[0] + v310) )
          {
            LODWORD(s[1]) = v293;
            v406 = v408;
            v407 = 0x1000000000LL;
            goto LABEL_426;
          }
        }
        else
        {
          v171 = (__m128i *)((char *)&v411 + 8 * v165);
          v172 = &v411;
        }
        do
        {
          if ( v172 )
            v172->m128i_i64[0] = 0;
          v172 = (__m128i *)((char *)v172 + 8);
        }
        while ( v171 != v172 );
        LODWORD(s[1]) = v165;
        v406 = v408;
        v407 = 0x1000000000LL;
        if ( v170 <= 0x10 )
        {
LABEL_280:
          v173 = (__int64)v406;
          v174 = &v406[4 * v170];
          for ( j = &v406[4 * (unsigned int)v407]; v174 != j; j += 4 )
          {
            if ( j )
              *(_DWORD *)j = 0;
          }
          LODWORD(v407) = v165;
          v301 = v2;
          v176 = 0;
          v297 = v3;
          v177 = v167;
          v307 = v377 + v165;
          v291 = v48;
          while ( 1 )
          {
            if ( v377 == v176 )
            {
              *((_QWORD *)s[0] + v176) = v316;
              *(_DWORD *)&v406[4 * v176] = v307;
            }
            else
            {
              v180 = sub_AD69F0(v161, (unsigned int)v176);
              *((_QWORD *)s[0] + v176) = v180;
              v173 = *(unsigned int *)(v177 + 4 * v176);
              *(_DWORD *)&v406[4 * v176] = v173;
            }
            v178 = (__int64 *)s[0];
            v179 = *((_QWORD *)s[0] + v176);
            if ( !v179 )
            {
              v2 = v301;
              v3 = v297;
              goto LABEL_319;
            }
            if ( v354 == v176 )
              break;
            ++v176;
          }
          v48 = v291;
          v2 = v301;
          v3 = v297;
          goto LABEL_428;
        }
LABEL_426:
        v294 = v165;
        v299 = v167;
        v303 = v48;
        v311 = v170;
        sub_C8D5F0((__int64)&v406, v408, v170, 4u, v48, v167);
        v165 = v294;
        v167 = v299;
        v48 = v303;
        v170 = v311;
        goto LABEL_280;
      }
    }
LABEL_306:
    v49 = 92;
    goto LABEL_292;
  }
  if ( v49 == 91 )
  {
    v132 = *(_QWORD *)(v3 + 8);
    if ( *(_BYTE *)(v132 + 8) == 18 )
      goto LABEL_292;
    v133 = *(_QWORD *)(v3 - 32);
    if ( *(_BYTE *)v133 != 17 )
      goto LABEL_291;
    v134 = *(_DWORD *)(v133 + 32);
    if ( v134 > 0x40 )
    {
      v380 = *(_QWORD *)(v3 - 96);
      v356 = *(_QWORD *)(v3 + 8);
      v192 = sub_C444A0(v133 + 24);
      v48 = v380;
      if ( v134 - v192 > 0x40 )
        goto LABEL_291;
      v132 = v356;
      v135 = **(_QWORD **)(v133 + 24);
    }
    else
    {
      v135 = *(_QWORD *)(v133 + 24);
    }
    v401[0] = v135;
    v136 = *(_DWORD *)(v132 + 32);
    if ( **(_BYTE **)(v3 - 64) > 0x15u )
      goto LABEL_291;
    v137 = *(_QWORD *)(v48 - 32);
    v402[0] = *(_QWORD *)(v3 - 64);
    if ( *(_BYTE *)v137 != 17 )
      goto LABEL_291;
    v138 = *(_DWORD *)(v137 + 32);
    if ( v138 > 0x40 )
    {
      v388 = v48;
      v360 = v136;
      v215 = sub_C444A0(v137 + 24);
      v48 = v388;
      if ( v138 - v215 > 0x40 )
        goto LABEL_291;
      v136 = v360;
      v139 = **(_QWORD **)(v137 + 24);
    }
    else
    {
      v139 = *(_QWORD *)(v137 + 24);
    }
    v401[1] = v139;
    if ( **(_BYTE **)(v48 - 64) <= 0x15u )
    {
      v140 = v136;
      v402[1] = *(_QWORD *)(v48 - 64);
      s[0] = &v411;
      s[1] = (void *)0x1000000000LL;
      if ( !v136 )
      {
        v407 = 0x1000000000LL;
        v406 = v408;
        goto LABEL_237;
      }
      if ( v136 > 0x10uLL )
      {
        v323 = 8LL * v136;
        v343 = v136;
        v362 = v48;
        v393 = v136;
        sub_C8D5F0((__int64)s, &v411, v136, 8u, v48, (__int64)v11);
        v140 = v393;
        v142 = (__m128i *)((char *)s[0] + 8 * LODWORD(s[1]));
        v48 = v362;
        v136 = v343;
        v141 = (__m128i *)((char *)s[0] + v323);
        if ( v142 == (__m128i *)((char *)s[0] + v323) )
        {
          LODWORD(s[1]) = v343;
          v406 = v408;
          v407 = 0x1000000000LL;
LABEL_423:
          v324 = v136;
          v363 = v48;
          v394 = v140;
          sub_C8D5F0((__int64)&v406, v408, v140, 4u, v48, (__int64)v11);
          v136 = v324;
          v48 = v363;
          v140 = v394;
LABEL_232:
          v143 = &v406[4 * (unsigned int)v407];
          for ( k = &v406[4 * v140]; k != v143; v143 += 4 )
          {
            if ( v143 )
              *(_DWORD *)v143 = 0;
          }
          LODWORD(v407) = v136;
LABEL_237:
          v145 = (__int64 *)s[0];
          for ( m = 0; m != 2; ++m )
          {
            v147 = v401[m];
            v148 = &v145[v147];
            if ( !*v148 )
            {
              *v148 = v402[m];
              *(_DWORD *)&v406[4 * v147] = v136 + v147;
              v145 = (__int64 *)s[0];
            }
          }
          v149 = 0;
          if ( v136 )
          {
            v150 = v48;
            v151 = v2;
            v152 = v3;
            v153 = v140;
            do
            {
              if ( !v145[v149] )
              {
                v154 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)(v152 + 8) + 24LL));
                *((_QWORD *)s[0] + v149) = v154;
                *(_DWORD *)&v406[4 * v149] = v149;
                v145 = (__int64 *)s[0];
              }
              ++v149;
            }
            while ( v153 != v149 );
            v48 = v150;
            v3 = v152;
            v2 = v151;
          }
          v193 = *(_QWORD *)(v48 - 96);
          v318 = sub_AD3730(v145, LODWORD(s[1]));
          v357 = v406;
          v173 = unk_3F1FE60;
          v381 = (unsigned int)v407;
          v405 = 257;
          v179 = (__int64)sub_BD2C40(112, unk_3F1FE60);
          v194 = v381;
          v195 = v357;
          v196 = v318;
          if ( !v179 )
            goto LABEL_319;
          goto LABEL_318;
        }
      }
      else
      {
        v141 = (__m128i *)((char *)&v411 + 8 * v136);
        v142 = &v411;
      }
      do
      {
        if ( v142 )
          v142->m128i_i64[0] = 0;
        v142 = (__m128i *)((char *)v142 + 8);
      }
      while ( v142 != v141 );
      LODWORD(s[1]) = v136;
      v406 = v408;
      v407 = 0x1000000000LL;
      if ( v140 <= 0x10 )
        goto LABEL_232;
      goto LABEL_423;
    }
LABEL_291:
    v49 = 91;
    goto LABEL_292;
  }
LABEL_76:
  v51 = *(_QWORD *)(v3 + 16);
  if ( !v51 || *(_QWORD *)(v51 + 8) || **(_BYTE **)(v51 + 24) != 91 )
  {
    v52 = *(_QWORD *)(v3 + 8);
    if ( *(_BYTE *)(v52 + 8) != 18 )
    {
      v53 = *(_DWORD *)(v52 + 32);
      if ( v53 != 1 )
      {
        v351 = *(_DWORD *)(v52 + 32);
        v372 = *(_QWORD *)(v3 - 64);
        sub_B48880((__int64 *)&v403, v53, 0);
        v54 = (__int64 **)v52;
        v55 = (_BYTE *)v3;
        v56 = *(_QWORD *)(v3 - 32);
        v57 = v351;
        v58 = 1;
        if ( *(_BYTE *)v56 == 17 )
        {
          while ( 1 )
          {
            if ( v372 != *((_QWORD *)v55 - 8) )
              goto LABEL_95;
            v59 = (_BYTE *)*((_QWORD *)v55 - 12);
            if ( *v59 != 91 )
              v59 = 0;
            if ( (_BYTE *)v3 != v55 )
            {
              v60 = *((_QWORD *)v55 + 2);
              if ( !v60 || *(_QWORD *)(v60 + 8) )
              {
                if ( v59 )
                  goto LABEL_95;
                if ( *(_DWORD *)(v56 + 32) <= 0x40u )
                {
                  v62 = *(_QWORD *)(v56 + 24) == 0;
                }
                else
                {
                  v305 = v57;
                  v314 = v54;
                  v330 = *(_DWORD *)(v56 + 32);
                  v352 = v56;
                  v61 = sub_C444A0(v56 + 24);
                  v56 = v352;
                  v58 = 1;
                  v54 = v314;
                  v57 = v305;
                  v62 = v330 == v61;
                }
                if ( !v62 )
                  goto LABEL_95;
              }
            }
            v63 = *(_QWORD **)(v56 + 24);
            if ( *(_DWORD *)(v56 + 32) > 0x40u )
              v63 = (_QWORD *)*v63;
            if ( (v403 & 1) != 0 )
              v403 = 2
                   * ((v403 >> 58 << 57)
                    | ~(-1LL << (v403 >> 58)) & (~(-1LL << (v403 >> 58)) & (v403 >> 1) | (1LL << (char)v63)))
                   + 1;
            else
              *(_QWORD *)(*(_QWORD *)v403 + 8LL * ((unsigned int)v63 >> 6)) |= 1LL << (char)v63;
            if ( !v59 )
              break;
            v55 = v59;
            v56 = *((_QWORD *)v59 - 4);
            if ( *(_BYTE *)v56 != 17 )
              goto LABEL_95;
          }
          v198 = (__int64)v55;
          v199 = v54;
          if ( v3 == v198 )
            goto LABEL_95;
          if ( **(_BYTE **)(v198 - 96) != 13 )
          {
            v64 = v403;
            if ( (v403 & 1) != 0 )
            {
              if ( (~(-1LL << (v403 >> 58)) & (v403 >> 1)) != (1LL << (v403 >> 58)) - 1 )
                goto LABEL_102;
            }
            else
            {
              v250 = *(_DWORD *)(v403 + 64);
              v251 = v250 >> 6;
              if ( v250 >> 6 )
              {
                v252 = *(_QWORD **)v403;
                v58 = *(_QWORD *)v403 + 8LL * (v251 - 1) + 8;
                while ( *v252 == -1 )
                {
                  if ( (_QWORD *)v58 == ++v252 )
                    goto LABEL_417;
                }
                goto LABEL_96;
              }
LABEL_417:
              v253 = v250 & 0x3F;
              if ( v253 )
              {
                v58 = 1LL << v253;
                if ( *(_QWORD *)(*(_QWORD *)v403 + 8LL * v251) != (1LL << v253) - 1 )
                  goto LABEL_96;
              }
            }
          }
          v319 = v57;
          v200 = (_QWORD *)sub_BD5C60(v3);
          v336 = sub_BCB2E0(v200);
          v359 = sub_ACADE0(v199);
          v201 = sub_AD64C0(v336, 0, 0);
          v203 = v319;
          v204 = v201;
          v205 = *(_QWORD *)(v198 - 32);
          if ( *(_DWORD *)(v205 + 32) <= 0x40u )
          {
            v207 = *(_QWORD *)(v205 + 24) == 0;
          }
          else
          {
            v320 = *(_DWORD *)(v205 + 32);
            v337 = v203;
            v206 = sub_C444A0(v205 + 24);
            v203 = v337;
            v207 = v320 == v206;
          }
          if ( !v207 )
          {
            v341 = v203;
            LOWORD(v412[0]) = 257;
            v238 = sub_BD2C40(72, 3u);
            v203 = v341;
            v198 = (__int64)v238;
            if ( v238 )
            {
              sub_B4DFA0((__int64)v238, v359, v372, v204, (__int64)s, v341, v3 + 24, 0);
              v203 = v341;
            }
          }
          s[0] = &v411;
          s[1] = (void *)0x1000000000LL;
          if ( (unsigned int)v203 > 0x10 )
          {
            v367 = v203;
            v397 = (unsigned int)v203;
            sub_C8D5F0((__int64)s, &v411, (unsigned int)v203, 4u, v202, v203);
            memset(s[0], 0, 4 * v397);
            LODWORD(v203) = v367;
            LODWORD(s[1]) = v367;
          }
          else
          {
            if ( !(_DWORD)v203 )
            {
              LODWORD(s[1]) = 0;
              goto LABEL_337;
            }
            memset(&v411, 0, (unsigned int)(4 * v203));
            LODWORD(s[1]) = v203;
          }
          v236 = 0;
          do
          {
            if ( (v403 & 1) != 0 )
              v237 = (((v403 >> 1) & ~(-1LL << (v403 >> 58))) >> v236) & 1;
            else
              v237 = (*(_QWORD *)(*(_QWORD *)v403 + 8LL * (v236 >> 6)) >> v236) & 1LL;
            if ( !(_BYTE)v237 )
              *((_DWORD *)s[0] + v236) = -1;
            ++v236;
          }
          while ( (_DWORD)v203 != v236 );
LABEL_337:
          v409 = 257;
          v58 = unk_3F1FE60;
          v386 = LODWORD(s[1]);
          v338 = s[0];
          v208 = sub_BD2C40(112, unk_3F1FE60);
          v350 = (__int64)v208;
          if ( v208 )
          {
            v58 = v198;
            sub_B4EB40((__int64)v208, v198, v338, v386, (__int64)&v406, v209, 0);
          }
          if ( s[0] != &v411 )
            _libc_free(s[0], v58);
          if ( (v403 & 1) != 0 )
            goto LABEL_101;
          v65 = (_QWORD *)v403;
          if ( !v403 )
            goto LABEL_101;
          goto LABEL_98;
        }
LABEL_95:
        v64 = v403;
        if ( (v403 & 1) == 0 )
        {
LABEL_96:
          if ( v64 )
          {
            v350 = 0;
            v65 = (_QWORD *)v64;
LABEL_98:
            if ( (_QWORD *)*v65 != v65 + 2 )
              _libc_free(*v65, v58);
            j_j___libc_free_0(v65, 72);
LABEL_101:
            if ( v350 )
              return (unsigned __int8 *)v350;
          }
        }
      }
    }
  }
LABEL_102:
  v66 = *(_QWORD *)(v3 - 96);
  if ( *(_BYTE *)v66 == 92 )
  {
    v67 = *(_DWORD *)(v66 + 80);
    if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v66 - 64) + 8LL) + 32LL) == v67 )
    {
      v390 = *(_QWORD *)(v3 - 96);
      if ( (unsigned __int8)sub_B4EE20(*(int **)(v66 + 72), v67, v67) )
      {
        v217 = v390;
        if ( *(_BYTE *)(*(_QWORD *)(v390 + 8) + 8LL) != 18 )
        {
          v218 = *(_QWORD *)(v3 - 32);
          if ( *(_BYTE *)v218 == 17 )
          {
            v219 = *(_DWORD *)(v218 + 32);
            if ( v219 > 0x40 )
            {
              if ( v219 - (unsigned int)sub_C444A0(v218 + 24) > 0x40 )
                goto LABEL_104;
              v217 = v390;
              v391 = **(_QWORD **)(v218 + 24);
            }
            else
            {
              v391 = *(_QWORD *)(v218 + 24);
            }
            v220 = *(_QWORD *)(v217 - 64);
            v221 = *(_QWORD *)(v3 - 64);
            if ( *(_BYTE *)v220 == 91 )
            {
              v222 = (*(_BYTE *)(v220 + 7) & 0x40) != 0
                   ? *(unsigned __int8 ***)(v220 - 8)
                   : (unsigned __int8 **)(v220 - 32LL * (*(_DWORD *)(v220 + 4) & 0x7FFFFFF));
              v361 = v217;
              if ( (unsigned __int8)sub_AC2BE0(*v222) )
              {
                v223 = v361;
                v224 = (*(_BYTE *)(v220 + 7) & 0x40) != 0
                     ? *(_QWORD *)(v220 - 8)
                     : v220 - 32LL * (*(_DWORD *)(v220 + 4) & 0x7FFFFFF);
                if ( v221 == *(_QWORD *)(v224 + 32) )
                {
                  v225 = *(unsigned __int8 **)(v224 + 64);
                  v226 = *v225;
                  if ( (_BYTE)v226 == 17 )
                  {
                    v227 = *((_DWORD *)v225 + 8);
                    if ( v227 <= 0x40 )
                    {
                      if ( *((_QWORD *)v225 + 3) )
                        goto LABEL_104;
                    }
                    else
                    {
                      v228 = sub_C444A0((__int64)(v225 + 24));
                      v223 = v361;
                      if ( v227 != v228 )
                        goto LABEL_104;
                    }
                  }
                  else
                  {
                    v280 = *((_QWORD *)v225 + 1);
                    v347 = v361;
                    if ( (unsigned int)*(unsigned __int8 *)(v280 + 8) - 17 > 1 || (unsigned __int8)v226 > 0x15u )
                      goto LABEL_104;
                    v365 = v225;
                    v281 = sub_AD7630((__int64)v225, 0, v226);
                    v223 = v347;
                    if ( v281 && *v281 == 17 )
                    {
                      v282 = *((_DWORD *)v281 + 8);
                      if ( v282 <= 0x40 )
                      {
                        if ( *((_QWORD *)v281 + 3) )
                          goto LABEL_104;
                      }
                      else
                      {
                        v283 = sub_C444A0((__int64)(v281 + 24));
                        v223 = v347;
                        if ( v282 != v283 )
                          goto LABEL_104;
                      }
                    }
                    else
                    {
                      if ( *(_BYTE *)(v280 + 8) != 17 )
                        goto LABEL_104;
                      v284 = *(_DWORD *)(v280 + 32);
                      v285 = 0;
                      v313 = v365;
                      v366 = v284;
                      v326 = v220;
                      for ( n = 0; v366 != n; ++n )
                      {
                        v287 = sub_AD69F0(v313, n);
                        if ( !v287 )
                          goto LABEL_104;
                        if ( *(_BYTE *)v287 != 13 )
                        {
                          if ( *(_BYTE *)v287 != 17 )
                            goto LABEL_104;
                          v288 = *(_DWORD *)(v287 + 32);
                          v285 = v288 <= 0x40
                               ? *(_QWORD *)(v287 + 24) == 0
                               : v288 == (unsigned int)sub_C444A0(v287 + 24);
                          if ( !v285 )
                            goto LABEL_104;
                        }
                      }
                      v223 = v347;
                      v220 = v326;
                      if ( !v285 )
                        goto LABEL_104;
                    }
                  }
                  v229 = *(unsigned int *)(*(_QWORD *)(v223 + 8) + 32LL);
                  s[0] = &v411;
                  s[1] = (void *)0x1000000000LL;
                  v230 = v229;
                  if ( v229 )
                  {
                    if ( v229 > 0x10 )
                    {
                      v327 = v229;
                      v349 = v223;
                      v368 = v229;
                      sub_C8D5F0((__int64)s, &v411, v229, 4u, v223, v229);
                      v230 = v327;
                      v223 = v349;
                      v229 = v368;
                    }
                    v231 = (char *)s[0] + 4 * v229;
                    for ( ii = (char *)s[0] + 4 * LODWORD(s[1]); v231 != ii; ii += 4 )
                    {
                      if ( ii )
                        *(_DWORD *)ii = 0;
                    }
                    LODWORD(s[1]) = v230;
                    v233 = 0;
                    do
                    {
                      v235 = 4 * v233;
                      if ( v391 == v233 )
                        v234 = 0;
                      else
                        v234 = *(_DWORD *)(*(_QWORD *)(v223 + 72) + 4 * v233);
                      ++v233;
                      *(_DWORD *)((char *)s[0] + v235) = v234;
                    }
                    while ( v230 != (_DWORD)v233 );
                  }
                  v409 = 257;
                  v398 = LODWORD(s[1]);
                  v348 = s[0];
                  v289 = sub_BD2C40(112, unk_3F1FE60);
                  v350 = (__int64)v289;
                  if ( v289 )
                  {
                    v212 = v220;
                    sub_B4EB40((__int64)v289, v220, v348, v398, (__int64)&v406, v290, 0);
                    v214 = s[0];
                    if ( s[0] == &v411 )
                      return (unsigned __int8 *)v350;
                    goto LABEL_349;
                  }
                  if ( s[0] != &v411 )
                    _libc_free(s[0], unk_3F1FE60);
                }
              }
            }
          }
        }
      }
    }
  }
LABEL_104:
  v350 = (__int64)sub_11AF6F0(v3);
  if ( v350 )
    return (unsigned __int8 *)v350;
  v68 = *(char **)(v3 - 96);
  v69 = *((_QWORD *)v68 + 2);
  if ( v69 )
  {
    if ( !*(_QWORD *)(v69 + 8) )
    {
      v239 = *v68;
      if ( (unsigned __int8)*v68 > 0x1Cu )
      {
        v240 = *(_QWORD *)(v2 + 32);
        v241 = *(_BYTE **)(v3 - 64);
        switch ( v239 )
        {
          case 'K':
            v242 = *((_QWORD *)v68 - 4);
            if ( !v242 )
              goto LABEL_106;
            if ( *v241 != 75 )
              goto LABEL_106;
            v243 = *((_QWORD *)v241 - 4);
            if ( !v243 )
              goto LABEL_106;
            v392 = 46;
            break;
          case 'E':
            v242 = *((_QWORD *)v68 - 4);
            if ( !v242 )
              goto LABEL_106;
            if ( *v241 != 69 )
              goto LABEL_106;
            v243 = *((_QWORD *)v241 - 4);
            if ( !v243 )
              goto LABEL_106;
            v392 = 40;
            break;
          case 'D':
            v242 = *((_QWORD *)v68 - 4);
            if ( !v242 )
              goto LABEL_106;
            if ( *v241 != 68 )
              goto LABEL_106;
            v243 = *((_QWORD *)v241 - 4);
            if ( !v243 )
              goto LABEL_106;
            v392 = 39;
            break;
          default:
            goto LABEL_106;
        }
        v244 = *(_QWORD *)(v242 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v244 + 8) - 17 <= 1 )
          v244 = **(_QWORD **)(v244 + 16);
        if ( *(_QWORD *)(v243 + 8) == v244 )
        {
          v245 = *(_QWORD *)(v3 - 32);
          v309 = *(_QWORD *)(v2 + 32);
          v409 = 257;
          v322 = v245;
          v342 = v243;
          v246 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v240 + 80) + 104LL))(
                   *(_QWORD *)(v240 + 80),
                   v242);
          v247 = v342;
          v248 = v246;
          if ( !v246 )
          {
            v345 = v309;
            v312 = v247;
            LOWORD(v412[0]) = 257;
            v273 = sub_BD2C40(72, 3u);
            v274 = v345;
            v248 = (__int64)v273;
            if ( v273 )
            {
              sub_B4DFA0((__int64)v273, v242, v312, v322, (__int64)s, v345, 0, 0);
              v274 = v345;
            }
            v346 = v274;
            (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v274 + 88) + 16LL))(
              *(_QWORD *)(v274 + 88),
              v248,
              &v406,
              *(_QWORD *)(v274 + 56),
              *(_QWORD *)(v274 + 64));
            v275 = v3;
            v276 = *(_QWORD *)v346 + 16LL * *(unsigned int *)(v346 + 8);
            for ( jj = *(_QWORD *)v346; v276 != jj; jj += 16 )
            {
              v278 = *(_QWORD *)(jj + 8);
              v279 = *(_DWORD *)jj;
              sub_B99FD0(v248, v279, v278);
            }
            v3 = v275;
          }
          v249 = *(_QWORD *)(v3 + 8);
          LOWORD(v412[0]) = 257;
          v179 = sub_B51D30(v392, v248, v249, (__int64)s, 0, 0);
          if ( v179 )
            return (unsigned __int8 *)v179;
        }
      }
    }
  }
LABEL_106:
  v70 = *(_QWORD *)(v3 + 8);
  v315 = v70;
  if ( *(_BYTE *)(v70 + 8) != 17 )
    return (unsigned __int8 *)v350;
  if ( (*(_BYTE *)(v70 + 32) & 1) != 0 )
    return (unsigned __int8 *)v350;
  v71 = *(_QWORD *)(v3 - 32);
  if ( *(_BYTE *)v71 != 17 )
    return (unsigned __int8 *)v350;
  v72 = *(_DWORD *)(v71 + 32);
  if ( v72 > 0x40 )
  {
    if ( v72 - (unsigned int)sub_C444A0(v71 + 24) > 0x40 )
      return (unsigned __int8 *)v350;
    v306 = **(_QWORD **)(v71 + 24);
  }
  else
  {
    v306 = *(_QWORD *)(v71 + 24);
  }
  v73 = *(__int64 **)(v2 + 32);
  v373 = *(_BYTE **)(v3 - 64);
  v296 = **(_BYTE **)(v2 + 88);
  v74 = *(_QWORD *)(v3 - 96);
  if ( *(_BYTE *)v74 == 91 )
  {
    if ( (*(_BYTE *)(v74 + 7) & 0x40) == 0 )
    {
      v75 = (__int64 *)(v74 - 32LL * (*(_DWORD *)(v74 + 4) & 0x7FFFFFF));
      v76 = *v75;
      if ( !*v75 )
        return (unsigned __int8 *)v350;
LABEL_114:
      v77 = (_BYTE *)v75[4];
      if ( !v77 )
        return (unsigned __int8 *)v350;
      v78 = v75[8];
      if ( *(_BYTE *)v78 != 17 )
        return (unsigned __int8 *)v350;
      if ( *(_DWORD *)(v78 + 32) > 0x40u )
      {
        v331 = *(_DWORD *)(v78 + 32);
        if ( v331 - (unsigned int)sub_C444A0(v78 + 24) > 0x40 )
          return (unsigned __int8 *)v350;
        v300 = **(_QWORD **)(v78 + 24);
      }
      else
      {
        v300 = *(_QWORD *)(v78 + 24);
      }
      if ( (unsigned __int8)(*(_BYTE *)v76 - 12) > 1u )
      {
        if ( (unsigned __int8)(*(_BYTE *)v76 - 9) > 2u )
          return (unsigned __int8 *)v350;
        v79 = v76;
        s[0] = 0;
        v406 = v408;
        s[1] = v412;
        v411.m128i_i64[0] = 8;
        v411.m128i_i32[2] = 0;
        v411.m128i_i8[12] = 1;
        v407 = 0x800000000LL;
        v403 = (unsigned __int64)s;
        v404 = (const void **)&v406;
        v80 = sub_AA8FD0(&v403, v76);
        v81 = &v403;
        v82 = v80;
        if ( v80 )
        {
          while ( 1 )
          {
            v84 = v406;
            if ( !(_DWORD)v407 )
              break;
            v292 = v81;
            v79 = *(_QWORD *)&v406[8 * (unsigned int)v407 - 8];
            LODWORD(v407) = v407 - 1;
            v83 = sub_AA8FD0(v81, v79);
            v81 = v292;
            if ( !v83 )
              goto LABEL_467;
          }
        }
        else
        {
LABEL_467:
          v84 = v406;
          v82 = 0;
        }
        if ( v84 != v408 )
          _libc_free(v84, v79);
        if ( !v411.m128i_i8[12] )
          _libc_free(s[1], v79);
        if ( !v82 )
          return (unsigned __int8 *)v350;
      }
      if ( v300 + 1 != v306 || (v300 & 1) != 0 )
        return (unsigned __int8 *)v350;
      if ( v296 )
      {
        if ( *v373 != 67 )
          return (unsigned __int8 *)v350;
        v85 = *((_QWORD *)v373 - 4);
        if ( !v85 )
          return (unsigned __int8 *)v350;
        s[0] = *((void **)v373 - 4);
        s[1] = &v403;
        if ( *v77 != 67 )
          return (unsigned __int8 *)v350;
        v86 = (_BYTE *)*((_QWORD *)v77 - 4);
        if ( *v86 != 55 )
          return (unsigned __int8 *)v350;
      }
      else
      {
        if ( *v77 != 67 )
          return (unsigned __int8 *)v350;
        v85 = *((_QWORD *)v77 - 4);
        if ( !v85 )
          return (unsigned __int8 *)v350;
        s[0] = *((void **)v77 - 4);
        s[1] = &v403;
        if ( *v373 != 67 )
          return (unsigned __int8 *)v350;
        v86 = (_BYTE *)*((_QWORD *)v373 - 4);
        if ( *v86 != 55 )
          return (unsigned __int8 *)v350;
      }
      if ( v85 == *((_QWORD *)v86 - 8) )
      {
        if ( (unsigned __int8)sub_11B1B00((_QWORD **)&s[1], *((_QWORD *)v86 - 4)) )
        {
          v332 = *(__int64 **)(v85 + 8);
          v87 = sub_BCB060((__int64)v332);
          v88 = sub_BCB060(v315);
          if ( v87 == 2 * v88 && v88 == v403 )
          {
            v89 = sub_BCDA70(v332, *(_DWORD *)(v315 + 32) >> 1);
            v409 = 257;
            v90 = v89;
            if ( v89 != *(_QWORD *)(v76 + 8) )
            {
              v91 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v73[10] + 120LL))(
                      v73[10],
                      49,
                      v76,
                      v89);
              if ( v91 )
              {
                v76 = v91;
              }
              else
              {
                LOWORD(v412[0]) = 257;
                v76 = sub_B51D30(49, v76, v90, (__int64)s, 0, 0);
                if ( (unsigned __int8)sub_920620(v76) )
                {
                  v259 = v73[12];
                  v260 = *((_DWORD *)v73 + 26);
                  if ( v259 )
                    sub_B99FD0(v76, 3u, v259);
                  sub_B45150(v76, v260);
                }
                (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v73[11] + 16LL))(
                  v73[11],
                  v76,
                  &v406,
                  v73[7],
                  v73[8]);
                v261 = v85;
                v262 = *v73 + 16LL * *((unsigned int *)v73 + 2);
                for ( kk = *v73; kk != v262; kk += 16 )
                {
                  v264 = *(_QWORD *)(kk + 8);
                  v265 = *(_DWORD *)kk;
                  sub_B99FD0(v76, v265, v264);
                }
                v85 = v261;
              }
            }
            if ( v296 )
              v92 = v306 >> 1;
            else
              v92 = v300 >> 1;
            v409 = 257;
            v93 = sub_BCB2E0((_QWORD *)v73[9]);
            v374 = sub_ACD640(v93, v92, 0);
            v94 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v73[10] + 104LL))(v73[10], v76, v85);
            if ( !v94 )
            {
              LOWORD(v412[0]) = 257;
              v254 = sub_BD2C40(72, 3u);
              v94 = (__int64)v254;
              if ( v254 )
                sub_B4DFA0((__int64)v254, v76, v85, v374, (__int64)s, 0, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v73[11] + 16LL))(
                v73[11],
                v94,
                &v406,
                v73[7],
                v73[8]);
              v255 = *v73;
              v256 = *v73 + 16LL * *((unsigned int *)v73 + 2);
              while ( v255 != v256 )
              {
                v257 = *(_QWORD *)(v255 + 8);
                v258 = *(_DWORD *)v255;
                v255 += 16;
                sub_B99FD0(v94, v258, v257);
              }
            }
            LOWORD(v412[0]) = 257;
            v95 = sub_BD2C40(72, unk_3F10A14);
            v350 = (__int64)v95;
            if ( v95 )
              sub_B51BF0((__int64)v95, v94, v315, (__int64)s, 0, 0);
          }
        }
      }
      return (unsigned __int8 *)v350;
    }
    v75 = *(__int64 **)(v74 - 8);
    v76 = *v75;
    if ( *v75 )
      goto LABEL_114;
  }
  return (unsigned __int8 *)v350;
}
