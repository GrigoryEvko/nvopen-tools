// Function: sub_1900BB0
// Address: 0x1900bb0
//
__int64 __fastcall sub_1900BB0(
        const __m128i *a1,
        __m128 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  const __m128i *v9; // r15
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdx
  __int32 v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rbx
  int v27; // r12d
  __int64 v28; // r14
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdx
  _QWORD *v35; // rdi
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rbx
  unsigned __int64 v39; // rax
  __int64 v40; // r12
  __int64 v41; // r14
  __int64 v42; // r13
  __int64 v43; // r12
  char v44; // bl
  __int64 v45; // r8
  __int64 v46; // rax
  int v47; // eax
  __int64 v48; // rbx
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r12
  __int64 v53; // rdx
  __int64 v54; // rax
  __int32 v55; // eax
  __int64 v56; // rdi
  __int64 *v57; // rbx
  __int64 *v58; // r12
  __int64 v59; // rdi
  __int64 v61; // rsi
  double v62; // xmm4_8
  double v63; // xmm5_8
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rbx
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // r10
  __int64 v71; // rdi
  unsigned __int8 v72; // al
  unsigned __int8 v73; // al
  __int64 v74; // rax
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int32 v77; // ebx
  __int64 v78; // r9
  int v79; // eax
  char v80; // al
  __int32 v81; // ebx
  __int64 *v82; // rax
  __int64 v83; // rax
  __int64 v84; // rsi
  unsigned int v85; // eax
  bool v86; // r10
  __int64 v87; // r11
  __int64 v88; // rdi
  unsigned __int8 v89; // al
  unsigned int v90; // esi
  __int64 v91; // rcx
  unsigned int v92; // edx
  __int64 *v93; // rax
  __int64 v94; // r11
  __int64 v95; // rdx
  __int64 v96; // r9
  unsigned __int8 v97; // bl
  int v98; // eax
  int v99; // r10d
  bool v100; // r8
  __int32 v101; // r11d
  __int64 v102; // rax
  __int64 v103; // r13
  unsigned int v104; // edx
  __int64 *v105; // rbx
  __int64 v106; // rdi
  __int64 v107; // r9
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // rdx
  unsigned __int64 v111; // rax
  __int64 v112; // rcx
  __int64 *v113; // r13
  double v114; // xmm4_8
  double v115; // xmm5_8
  __int64 v116; // rax
  __int64 v117; // r13
  _QWORD *v118; // r12
  __int64 v119; // rbx
  __int64 *v120; // rdx
  unsigned int v121; // r13d
  __int64 *v122; // r15
  __int64 v123; // r13
  int v124; // eax
  int v125; // esi
  __int64 *v126; // r11
  int v127; // edx
  unsigned int m; // ecx
  __int64 v129; // rdi
  __int64 v130; // rsi
  bool v131; // r8
  bool v132; // r9
  int v133; // eax
  __int64 v134; // rax
  unsigned int v135; // r13d
  int v136; // eax
  __int64 v137; // rcx
  unsigned int j; // r8d
  __int64 v139; // rdi
  __int64 v140; // rsi
  bool v141; // r9
  bool v142; // r10
  _QWORD *v143; // r12
  __int64 v144; // rdi
  unsigned int v145; // ecx
  _QWORD *v146; // rax
  __int64 v147; // r9
  __int64 v148; // rbx
  __int64 v149; // rdx
  __int64 v150; // r8
  unsigned int v151; // esi
  __int64 v152; // rcx
  unsigned int v153; // esi
  unsigned int v154; // edx
  __int64 *v155; // rax
  __int64 v156; // rdi
  unsigned int v157; // r13d
  int v158; // eax
  __int64 v159; // rcx
  int v160; // r8d
  unsigned int i; // edx
  __int64 v162; // rdi
  __int64 *v163; // r15
  __int64 v164; // rsi
  bool v165; // r9
  bool v166; // r10
  __int64 v167; // rax
  __int64 v168; // rdx
  __int64 v169; // rcx
  __int64 v170; // rdx
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // rdx
  __int64 v174; // rbx
  __int64 v175; // rax
  __int64 v176; // rsi
  __int64 v177; // rdx
  __int64 v178; // r10
  unsigned int v179; // edi
  __int64 *v180; // rcx
  __int64 v181; // r11
  __int64 v182; // rdx
  __int64 v183; // rcx
  int v184; // ebx
  __int64 v185; // rax
  bool v186; // zf
  int v187; // ebx
  __int64 v188; // rdi
  __int64 v189; // rdi
  char v190; // bl
  __int64 v191; // rax
  int v192; // ecx
  unsigned int v193; // esi
  __int64 v194; // r10
  __int64 v195; // r11
  __int64 v196; // r9
  __int64 *v197; // r8
  __int64 v198; // rdx
  __int64 v199; // r13
  __int64 v200; // rax
  __int64 v201; // r9
  __int64 v202; // rdx
  char v203; // al
  unsigned int v204; // r8d
  unsigned __int64 v205; // rax
  __int64 v206; // rsi
  __int64 *v207; // rax
  __int64 v208; // rax
  unsigned __int64 v209; // rax
  __int64 v210; // rsi
  int v211; // r13d
  _QWORD *v212; // r11
  int v213; // ecx
  int v214; // ecx
  __int64 v215; // rdx
  char v216; // al
  __int64 v217; // rax
  int v218; // eax
  __int64 v219; // rax
  __int64 v220; // rdx
  __int64 v221; // rbx
  __int64 v222; // rax
  __int64 v223; // rax
  char v224; // al
  __int64 *v225; // rbx
  __int64 v226; // rcx
  unsigned int v227; // ebx
  int v228; // eax
  _QWORD *v229; // rax
  __int64 v230; // rsi
  unsigned __int64 v231; // rdx
  __int64 v232; // rsi
  __int32 v233; // eax
  __int64 v234; // rbx
  char v235; // al
  __int64 *v236; // r13
  __int64 v237; // rax
  __int64 v238; // rdx
  __int64 v239; // rax
  char v240; // al
  unsigned int v241; // edx
  int v242; // eax
  __int64 v243; // rsi
  int v244; // r8d
  __int64 v245; // rdi
  unsigned int v246; // edx
  __int64 v247; // r9
  int v248; // r11d
  _QWORD *v249; // r10
  __int64 v250; // rax
  int v251; // eax
  __int64 v252; // r13
  int v253; // eax
  int v254; // esi
  int v255; // edx
  unsigned int k; // ecx
  __int64 v257; // rdi
  __int64 v258; // rsi
  bool v259; // r8
  bool v260; // r9
  unsigned int v261; // ecx
  int v262; // eax
  __int64 v263; // rsi
  int v264; // r8d
  __int64 v265; // rdi
  int v266; // r11d
  unsigned int v267; // edx
  __int64 v268; // r9
  __int64 v269; // r11
  unsigned __int8 v270; // dl
  __int64 v271; // rdx
  int v272; // eax
  __int64 v273; // r13
  __int64 v274; // rdx
  int v275; // eax
  int v276; // r9d
  __int64 v277; // rax
  int v278; // eax
  unsigned int v279; // esi
  __int32 v280; // eax
  int v281; // eax
  __int64 v282; // rax
  char v283; // al
  unsigned int v284; // eax
  __int64 v285; // rax
  int v286; // eax
  int v287; // r8d
  __int32 v288; // ecx
  int v289; // edx
  char v290; // al
  unsigned int v291; // ecx
  bool v292; // al
  __int64 v293; // rax
  double v294; // xmm4_8
  double v295; // xmm5_8
  unsigned __int8 v296; // al
  int v297; // ecx
  int v298; // r8d
  __int16 v299; // ax
  __int64 v300; // rdi
  char v301; // al
  unsigned __int16 v302; // ax
  __int16 v303; // ax
  unsigned __int8 v304; // si
  __int64 v305; // rsi
  int v306; // eax
  char v307; // al
  char v308; // al
  char v309; // al
  char v310; // al
  __int64 v311; // rax
  __int64 *v312; // rdi
  __int32 v313; // edi
  int v314; // edx
  char v315; // dl
  __int32 v316; // [rsp+8h] [rbp-178h]
  int v317; // [rsp+Ch] [rbp-174h]
  __int64 v318; // [rsp+10h] [rbp-170h]
  __int32 v319; // [rsp+18h] [rbp-168h]
  __int64 v320; // [rsp+18h] [rbp-168h]
  __int32 v321; // [rsp+18h] [rbp-168h]
  __int64 v322; // [rsp+18h] [rbp-168h]
  __int64 v323; // [rsp+20h] [rbp-160h]
  int v324; // [rsp+20h] [rbp-160h]
  unsigned int v325; // [rsp+20h] [rbp-160h]
  __int32 v326; // [rsp+20h] [rbp-160h]
  __int64 v327; // [rsp+20h] [rbp-160h]
  __int32 v328; // [rsp+20h] [rbp-160h]
  __int32 v329; // [rsp+20h] [rbp-160h]
  __int64 *v330; // [rsp+20h] [rbp-160h]
  __int64 v331; // [rsp+28h] [rbp-158h]
  __int64 v332; // [rsp+28h] [rbp-158h]
  __int64 *v333; // [rsp+28h] [rbp-158h]
  __int64 v334; // [rsp+28h] [rbp-158h]
  __int64 v335; // [rsp+28h] [rbp-158h]
  bool v336; // [rsp+28h] [rbp-158h]
  int v337; // [rsp+28h] [rbp-158h]
  __int64 v338; // [rsp+28h] [rbp-158h]
  __int64 v339; // [rsp+28h] [rbp-158h]
  __int64 v340; // [rsp+28h] [rbp-158h]
  bool v341; // [rsp+28h] [rbp-158h]
  bool v342; // [rsp+28h] [rbp-158h]
  __int64 v343; // [rsp+28h] [rbp-158h]
  __int64 v344; // [rsp+28h] [rbp-158h]
  int v345; // [rsp+28h] [rbp-158h]
  __int64 v346; // [rsp+28h] [rbp-158h]
  __int64 v347; // [rsp+28h] [rbp-158h]
  __int64 v348; // [rsp+30h] [rbp-150h]
  __int64 v349; // [rsp+30h] [rbp-150h]
  unsigned int v350; // [rsp+30h] [rbp-150h]
  unsigned int v351; // [rsp+30h] [rbp-150h]
  const __m128i *v352; // [rsp+30h] [rbp-150h]
  __int64 v353; // [rsp+30h] [rbp-150h]
  unsigned int v354; // [rsp+30h] [rbp-150h]
  __int64 v355; // [rsp+30h] [rbp-150h]
  __int64 v356; // [rsp+30h] [rbp-150h]
  int v357; // [rsp+30h] [rbp-150h]
  __int64 *v358; // [rsp+30h] [rbp-150h]
  int v359; // [rsp+30h] [rbp-150h]
  int v360; // [rsp+30h] [rbp-150h]
  int v361; // [rsp+30h] [rbp-150h]
  int v362; // [rsp+30h] [rbp-150h]
  int v363; // [rsp+30h] [rbp-150h]
  __int32 v364; // [rsp+38h] [rbp-148h]
  unsigned __int8 v365; // [rsp+3Fh] [rbp-141h]
  __int64 v366; // [rsp+40h] [rbp-140h]
  __int64 v367; // [rsp+50h] [rbp-130h]
  __int64 v368; // [rsp+50h] [rbp-130h]
  __int64 m128i_i64; // [rsp+50h] [rbp-130h]
  int v370; // [rsp+50h] [rbp-130h]
  bool v371; // [rsp+50h] [rbp-130h]
  __int64 v372; // [rsp+50h] [rbp-130h]
  __int64 *v373; // [rsp+50h] [rbp-130h]
  __int64 *v374; // [rsp+50h] [rbp-130h]
  __int64 v375; // [rsp+58h] [rbp-128h]
  __int64 *v376; // [rsp+58h] [rbp-128h]
  __int64 v377; // [rsp+58h] [rbp-128h]
  unsigned int v378; // [rsp+58h] [rbp-128h]
  unsigned int v379; // [rsp+58h] [rbp-128h]
  char v380; // [rsp+60h] [rbp-120h]
  __int64 *v381; // [rsp+60h] [rbp-120h]
  __int64 v382; // [rsp+60h] [rbp-120h]
  int v383; // [rsp+60h] [rbp-120h]
  __int64 *v384; // [rsp+60h] [rbp-120h]
  int v385; // [rsp+60h] [rbp-120h]
  int v386; // [rsp+60h] [rbp-120h]
  __int64 v387; // [rsp+68h] [rbp-118h]
  __int64 v388; // [rsp+68h] [rbp-118h]
  int v389; // [rsp+68h] [rbp-118h]
  int v390; // [rsp+68h] [rbp-118h]
  unsigned int v391; // [rsp+68h] [rbp-118h]
  unsigned int v392; // [rsp+68h] [rbp-118h]
  int v393; // [rsp+68h] [rbp-118h]
  int v394; // [rsp+68h] [rbp-118h]
  _BOOL8 v395; // [rsp+70h] [rbp-110h] BYREF
  __int64 v396; // [rsp+78h] [rbp-108h]
  __int64 v397; // [rsp+80h] [rbp-100h]
  __int64 v398; // [rsp+88h] [rbp-F8h]
  __int64 v399; // [rsp+90h] [rbp-F0h]
  __m128i v400[2]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v401; // [rsp+C0h] [rbp-C0h]
  __m128i v402; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v403; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v404; // [rsp+F0h] [rbp-90h]
  __int64 v405; // [rsp+100h] [rbp-80h] BYREF
  __int64 v406; // [rsp+108h] [rbp-78h]
  _QWORD *v407; // [rsp+110h] [rbp-70h]
  __int64 v408; // [rsp+118h] [rbp-68h]
  __int64 v409; // [rsp+120h] [rbp-60h]
  unsigned __int64 v410; // [rsp+128h] [rbp-58h]
  _QWORD *v411; // [rsp+130h] [rbp-50h]
  _QWORD *v412; // [rsp+138h] [rbp-48h]
  __int64 v413; // [rsp+140h] [rbp-40h]
  _QWORD *v414; // [rsp+148h] [rbp-38h]

  v9 = a1;
  v409 = 0;
  v413 = 0;
  v414 = 0;
  v406 = 8;
  v405 = sub_22077B0(64);
  v10 = (_QWORD *)(v405 + 24);
  v11 = sub_22077B0(512);
  v410 = v405 + 24;
  v412 = (_QWORD *)v11;
  v12 = v11 + 512;
  v411 = (_QWORD *)v11;
  *(_QWORD *)(v405 + 24) = v11;
  v408 = v11;
  v407 = (_QWORD *)v11;
  v13 = a1[1].m128i_i64[0];
  v413 = v12;
  v414 = v10;
  v14 = *(_QWORD *)(v13 + 56);
  v409 = v12;
  v15 = *(_QWORD *)(v14 + 24);
  v16 = *(_QWORD *)(v14 + 32);
  v17 = sub_22077B0(136);
  if ( v17 )
  {
    v18 = a1[37].m128i_i32[0];
    *(_QWORD *)(v17 + 8) = v14;
    *(_QWORD *)(v17 + 16) = v15;
    *(_DWORD *)v17 = v18;
    *(_DWORD *)(v17 + 4) = v18;
    *(_QWORD *)(v17 + 32) = (char *)a1 + 88;
    v19 = a1[7].m128i_i64[1];
    *(_QWORD *)(v17 + 24) = v16;
    *(_QWORD *)(v17 + 40) = v19;
    a1[7].m128i_i64[1] = v17 + 32;
    *(_QWORD *)(v17 + 56) = a1 + 15;
    v20 = a1[17].m128i_i64[0];
    *(_QWORD *)(v17 + 48) = 0;
    *(_QWORD *)(v17 + 64) = v20;
    a1[17].m128i_i64[0] = v17 + 56;
    *(_QWORD *)(v17 + 80) = (char *)a1 + 392;
    v21 = a1[26].m128i_i64[1];
    *(_QWORD *)(v17 + 72) = 0;
    *(_QWORD *)(v17 + 88) = v21;
    a1[26].m128i_i64[1] = v17 + 80;
    *(_QWORD *)(v17 + 104) = a1 + 34;
    v22 = a1[36].m128i_i64[0];
    *(_QWORD *)(v17 + 96) = 0;
    *(_QWORD *)(v17 + 112) = v22;
    a1[36].m128i_i64[0] = v17 + 104;
    *(_QWORD *)(v17 + 120) = 0;
    *(_BYTE *)(v17 + 128) = 0;
  }
  v23 = v411;
  v402.m128i_i64[0] = v17;
  if ( v411 == (_QWORD *)(v413 - 8) )
  {
    sub_18FC260(&v405, &v402);
    v24 = (__int64)v411;
  }
  else
  {
    if ( v411 )
    {
      *v411 = v17;
      v23 = v411;
    }
    v24 = (__int64)(v23 + 1);
    v411 = (_QWORD *)v24;
  }
  v365 = 0;
  v364 = a1[37].m128i_i32[0];
  if ( (_QWORD *)v24 != v407 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( v412 == (_QWORD *)v24 )
          v24 = *(v414 - 1) + 512LL;
        v36 = *(_QWORD *)(v24 - 8);
        v9[37].m128i_i32[0] = *(_DWORD *)v36;
        v380 = *(_BYTE *)(v36 + 128);
        if ( v380 )
          break;
        v375 = **(_QWORD **)(v36 + 8);
        if ( !sub_157F0B0(v375) )
          ++v9[37].m128i_i32[0];
        v37 = sub_157F0B0(v375);
        v38 = v37;
        if ( v37 )
        {
          v39 = sub_157EBA0(v37);
          v40 = v39;
          if ( *(_BYTE *)(v39 + 16) == 26 && (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) == 3 )
          {
            v273 = *(_QWORD *)(v39 - 72);
            if ( *(_BYTE *)(v273 + 16) > 0x17u )
            {
              if ( (unsigned __int8)sub_18FCE30(*(_QWORD *)(v39 - 72)) )
                v380 = sub_18FF290((__int64)v9, v273, v40, v375, v38);
            }
          }
        }
        v367 = 0;
        v387 = v375 + 40;
        if ( *(_QWORD *)(v375 + 48) != v375 + 40 )
        {
          v366 = v36;
          v41 = *(_QWORD *)(v375 + 48);
          while ( 1 )
          {
            v42 = v41;
            v41 = *(_QWORD *)(v41 + 8);
            v43 = v42 - 24;
            v44 = sub_1AE9990(v42 - 24, v9->m128i_i64[0]);
            if ( v44 )
            {
              sub_1AEAA40(v42 - 24);
              goto LABEL_76;
            }
            if ( *(_BYTE *)(v42 - 8) != 78 )
              goto LABEL_49;
            v46 = *(_QWORD *)(v42 - 48);
            if ( *(_BYTE *)(v46 + 16) )
              goto LABEL_49;
            v47 = *(_DWORD *)(v46 + 36);
            switch ( v47 )
            {
              case 4:
                v48 = *(_QWORD *)(v42 - 24LL * (*(_DWORD *)(v42 - 4) & 0xFFFFFFF) - 24);
                v49 = *(unsigned __int8 *)(v48 + 16);
                if ( (unsigned __int8)v49 > 0x17u )
                {
                  if ( (_BYTE)v49 == 78 )
                  {
                    if ( !(unsigned __int8)sub_1560260((_QWORD *)(v48 + 56), -1, 36) )
                    {
                      if ( *(char *)(v48 + 23) < 0 )
                      {
                        v50 = sub_1648A40(v48);
                        v52 = v50 + v51;
                        v53 = 0;
                        if ( *(char *)(v48 + 23) < 0 )
                          v53 = sub_1648A40(v48);
                        if ( (unsigned int)((v52 - v53) >> 4) )
                          goto LABEL_40;
                      }
                      v54 = *(_QWORD *)(v48 - 24);
                      if ( *(_BYTE *)(v54 + 16) )
                        goto LABEL_40;
                      v402.m128i_i64[0] = *(_QWORD *)(v54 + 112);
                      if ( !(unsigned __int8)sub_1560260(&v402, -1, 36) )
                        goto LABEL_40;
                    }
                    if ( !*(_BYTE *)(*(_QWORD *)v48 + 8LL) )
                      goto LABEL_40;
LABEL_230:
                    v207 = (__int64 *)sub_157E9C0(v375);
                    v208 = sub_159C4F0(v207);
                    v402.m128i_i64[0] = v48;
                    v400[0].m128i_i64[0] = v208;
                    goto LABEL_81;
                  }
                  v205 = (unsigned int)(v49 - 35);
                  if ( (unsigned __int8)v205 <= 0x34u )
                  {
                    v206 = 0x1F133FFE23FFFFLL;
                    if ( _bittest64(&v206, v205) )
                      goto LABEL_230;
                  }
                }
LABEL_40:
                if ( v387 == v41 )
                  goto LABEL_41;
                break;
              case 191:
                goto LABEL_40;
              case 114:
                if ( *(_QWORD *)(v42 - 16) )
                  goto LABEL_40;
                sub_141F820(&v402, v43 | 4, 1u, v9->m128i_i64[0]);
                if ( (unsigned __int8)sub_18FEB70((__int64)&v9[24].m128i_i64[1], v402.m128i_i64, (__int64 **)v400) )
                  goto LABEL_40;
                sub_18FFC60((__int64)&v9[24].m128i_i64[1], v9[26].m128i_i64[1], &v402, v9[37].m128i_i32);
                if ( v387 == v41 )
                  goto LABEL_41;
                break;
              case 79:
                v78 = *(_QWORD *)(v43 - 24LL * (*(_DWORD *)(v42 - 4) & 0xFFFFFFF));
                v79 = *(unsigned __int8 *)(v78 + 16);
                if ( (unsigned __int8)v79 <= 0x17u )
                  goto LABEL_122;
                if ( (_BYTE)v79 == 78 )
                {
                  v349 = *(_QWORD *)(v43 - 24LL * (*(_DWORD *)(v42 - 4) & 0xFFFFFFF));
                  v80 = sub_1560260((_QWORD *)(v78 + 56), -1, 36);
                  v78 = v349;
                  if ( v80 )
                    goto LABEL_94;
                  if ( *(char *)(v349 + 23) >= 0 )
                    goto LABEL_568;
                  v219 = sub_1648A40(v349);
                  v78 = v349;
                  v221 = v219 + v220;
                  v222 = 0;
                  if ( *(char *)(v349 + 23) < 0 )
                  {
                    v222 = sub_1648A40(v349);
                    v78 = v349;
                  }
                  if ( !(unsigned int)((v221 - v222) >> 4) )
                  {
LABEL_568:
                    v223 = *(_QWORD *)(v78 - 24);
                    if ( !*(_BYTE *)(v223 + 16) )
                    {
                      v355 = v78;
                      v402.m128i_i64[0] = *(_QWORD *)(v223 + 112);
                      v224 = sub_1560260(&v402, -1, 36);
                      v78 = v355;
                      if ( !v224 )
                      {
                        v367 = 0;
                        goto LABEL_123;
                      }
LABEL_94:
                      if ( *(_BYTE *)(*(_QWORD *)v78 + 8LL) )
                        goto LABEL_95;
                      goto LABEL_122;
                    }
                  }
                  goto LABEL_122;
                }
                v209 = (unsigned int)(v79 - 35);
                if ( (unsigned __int8)v209 > 0x34u )
                  goto LABEL_122;
                v210 = 0x1F133FFE23FFFFLL;
                if ( !_bittest64(&v210, v209) )
                  goto LABEL_122;
LABEL_95:
                v81 = v9[7].m128i_i32[0];
                if ( !v81 )
                  goto LABEL_96;
                v318 = v78;
                v320 = v9[6].m128i_i64[0];
                v317 = 1;
                v316 = v81 - 1;
                v325 = (v81 - 1) & sub_18FDEE0(v78);
                while ( 1 )
                {
                  v225 = (__int64 *)(v320 + 16LL * v325);
                  if ( (unsigned __int8)sub_18FB980(v318, *v225) )
                    break;
                  if ( *v225 == -8 )
                  {
                    v78 = v318;
                    goto LABEL_96;
                  }
                  v325 = v316 & (v317 + v325);
                  ++v317;
                }
                v78 = v318;
                v43 = v42 - 24;
                if ( v225 == (__int64 *)(v9[6].m128i_i64[0] + 16LL * v9[7].m128i_u32[0]) )
                  goto LABEL_96;
                v226 = *(_QWORD *)(v225[1] + 24);
                if ( !v226 )
                  goto LABEL_96;
                if ( *(_BYTE *)(v226 + 16) == 13
                  && ((v227 = *(_DWORD *)(v226 + 32), v227 <= 0x40)
                    ? (v44 = *(_QWORD *)(v226 + 24) == 1)
                    : (v334 = v226, v228 = sub_16A57B0(v226 + 24), v226 = v334, v78 = v318, v44 = v227 - 1 == v228),
                      v44) )
                {
LABEL_76:
                  sub_19003A0((__int64)v9, v43);
                  sub_15F20C0((_QWORD *)v43);
                  v380 = v44;
                  if ( v387 == v41 )
                    goto LABEL_41;
                }
                else
                {
                  v229 = (_QWORD *)(v43 - 24LL * (*(_DWORD *)(v42 - 4) & 0xFFFFFFF));
                  if ( *v229 )
                  {
                    v230 = v229[1];
                    v231 = v229[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v231 = v230;
                    if ( v230 )
                      *(_QWORD *)(v230 + 16) = *(_QWORD *)(v230 + 16) & 3LL | v231;
                  }
                  *v229 = v226;
                  v232 = *(_QWORD *)(v226 + 8);
                  v229[1] = v232;
                  if ( v232 )
                    *(_QWORD *)(v232 + 16) = (unsigned __int64)(v229 + 1) | *(_QWORD *)(v232 + 16) & 3LL;
                  v229[2] = v229[2] & 3LL | (v226 + 8);
                  *(_QWORD *)(v226 + 8) = v229;
LABEL_96:
                  v368 = v78;
                  v82 = (__int64 *)sub_157E9C0(v375);
                  v83 = sub_159C4F0(v82);
                  v84 = v9[7].m128i_i64[1];
                  v400[0].m128i_i64[0] = v83;
                  v402.m128i_i64[0] = v368;
                  sub_18FEF10((__int64)&v9[5].m128i_i64[1], v84, v402.m128i_i64, v400);
                  v367 = 0;
                  if ( v387 == v41 )
                    goto LABEL_41;
                }
                break;
              default:
LABEL_49:
                v61 = sub_13E3350(v42 - 24, v9 + 2, 0, 1, v45);
                if ( v61 )
                {
                  if ( *(_QWORD *)(v42 - 16) )
                  {
                    sub_164D160(v42 - 24, v61, a2, *(double *)a3.m128i_i64, a4, a5, v62, v63, a8, a9);
                    v380 = 1;
                  }
                  v44 = sub_1AE9990(v42 - 24, v9->m128i_i64[0]);
                  if ( v44 )
                    goto LABEL_76;
                }
                v64 = *(unsigned __int8 *)(v42 - 8);
                if ( (_BYTE)v64 == 78 )
                {
                  if ( (unsigned __int8)sub_1560260((_QWORD *)(v42 + 32), -1, 36) )
                    goto LABEL_569;
                  if ( *(char *)(v42 - 1) >= 0 )
                    goto LABEL_570;
                  v65 = sub_1648A40(v42 - 24);
                  v67 = v65 + v66;
                  v68 = 0;
                  if ( *(char *)(v42 - 1) < 0 )
                    v68 = sub_1648A40(v42 - 24);
                  if ( !(unsigned int)((v67 - v68) >> 4) )
                  {
LABEL_570:
                    v69 = *(_QWORD *)(v42 - 48);
                    if ( !*(_BYTE *)(v69 + 16) )
                    {
                      v402.m128i_i64[0] = *(_QWORD *)(v69 + 112);
                      if ( (unsigned __int8)sub_1560260(&v402, -1, 36) )
                      {
LABEL_569:
                        if ( *(_BYTE *)(*(_QWORD *)(v42 - 24) + 8LL) )
                        {
                          v77 = v9[7].m128i_i32[0];
                          if ( !v77 )
                            goto LABEL_80;
                          goto LABEL_128;
                        }
                      }
                    }
                  }
LABEL_61:
                  v395 = 0;
                  v70 = v9->m128i_i64[1];
                  v71 = v42 - 24;
                  v396 = 0;
                  v397 = 0;
                  v398 = 0;
                  v399 = v42 - 24;
                  if ( *(_BYTE *)(v42 - 8) == 78 )
                  {
                    v250 = *(_QWORD *)(v42 - 48);
                    if ( !*(_BYTE *)(v250 + 16) && (*(_BYTE *)(v250 + 33) & 0x20) != 0 )
                    {
                      if ( (unsigned __int8)sub_14A36E0(v70) )
                      {
                        LOBYTE(v395) = 1;
LABEL_309:
                        if ( v396 && BYTE6(v397) )
                        {
                          v71 = v399;
                          if ( !(_BYTE)v398 && (unsigned int)v397 <= 1 )
                          {
LABEL_101:
                            if ( *(_BYTE *)(v71 + 16) == 54 )
                              goto LABEL_102;
LABEL_106:
                            v86 = v395;
                            if ( !v395 )
                            {
                              v87 = v399;
                              v88 = 0;
                              v89 = *(_BYTE *)(v399 + 16);
                              if ( v89 <= 0x17u )
                                goto LABEL_108;
                              if ( v89 != 54 )
                              {
                                if ( v89 == 55 )
                                {
                                  v88 = *(_QWORD *)(v399 - 24);
                                }
                                else if ( v89 == 78 )
                                {
                                  v277 = *(_QWORD *)(v399 - 24);
                                  if ( !*(_BYTE *)(v277 + 16) )
                                  {
                                    v278 = *(_DWORD *)(v277 + 36);
                                    if ( v278 == 4057 || v278 == 4085 )
                                    {
                                      v88 = *(_QWORD *)(v399 + 24 * (1LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF)));
                                    }
                                    else
                                    {
                                      v86 = v278 == 4492 || v278 == 4503;
                                      if ( v86 )
                                      {
                                        v86 = 0;
                                        v88 = *(_QWORD *)(v399 + 24 * (2LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF)));
                                      }
                                    }
                                  }
                                }
LABEL_108:
                                v90 = v9[16].m128i_u32[2];
                                if ( !v90 )
                                  goto LABEL_338;
                                v91 = v9[15].m128i_i64[1];
                                v92 = (v90 - 1) & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
                                v93 = (__int64 *)(v91 + 16LL * v92);
                                v94 = *v93;
                                if ( v88 != *v93 )
                                {
                                  v286 = 1;
                                  while ( v94 != -8 )
                                  {
                                    v287 = v286 + 1;
                                    v92 = (v90 - 1) & (v286 + v92);
                                    v93 = (__int64 *)(v91 + 16LL * v92);
                                    v94 = *v93;
                                    if ( v88 == *v93 )
                                      goto LABEL_110;
                                    v286 = v287;
                                  }
                                  goto LABEL_338;
                                }
LABEL_110:
                                if ( v93 == (__int64 *)(v91 + 16LL * v90) )
                                  goto LABEL_338;
                                v95 = v93[1];
                                v96 = *(_QWORD *)(v95 + 24);
                                v97 = *(_BYTE *)(v95 + 40);
                                v350 = *(_DWORD *)(v95 + 32);
                                v98 = *(_DWORD *)(v95 + 36);
                                if ( !v96 )
                                  goto LABEL_338;
                                if ( v86 )
                                {
                                  v99 = WORD2(v397);
                                  if ( v98 != WORD2(v397) )
                                  {
                                    v100 = (_DWORD)v397 != 0;
                                    goto LABEL_115;
                                  }
                                  if ( (_BYTE)v398 || (unsigned int)v397 > 1 )
                                  {
LABEL_338:
                                    if ( !v395 )
                                    {
                                      v269 = v399;
                                      goto LABEL_340;
                                    }
                                    v99 = WORD2(v397);
                                    v91 = v9[15].m128i_i64[1];
                                    v90 = v9[16].m128i_u32[2];
                                    v100 = (_DWORD)v397 != 0;
LABEL_115:
                                    v101 = v9[37].m128i_i32[0];
                                    v102 = v396;
                                    goto LABEL_116;
                                  }
LABEL_399:
                                  if ( v395 )
                                  {
                                    v100 = (_DWORD)v397 != 0;
                                    if ( v97 < (unsigned __int8)((_DWORD)v397 != 0) )
                                    {
                                      v99 = WORD2(v397);
                                      goto LABEL_115;
                                    }
                                  }
                                  else
                                  {
                                    v338 = *(_QWORD *)(v95 + 24);
                                    v292 = sub_15F32D0(v399);
                                    v96 = v338;
                                    if ( v97 < (unsigned __int8)v292 )
                                      goto LABEL_338;
                                  }
                                  if ( *(_BYTE *)(v42 - 8) == 54 && (*(_QWORD *)(v42 + 24) || *(__int16 *)(v42 - 6) < 0) )
                                  {
                                    v339 = v96;
                                    v293 = sub_1625790(v42 - 24, 6);
                                    v96 = v339;
                                    if ( v293 )
                                      goto LABEL_436;
                                  }
                                  v343 = v96;
                                  v308 = sub_18FECC0((__int64)v9, v42 - 24, v350);
                                  v96 = v343;
                                  if ( v308
                                    || (v309 = sub_18FBB40((__int64)v9, v350, v9[37].m128i_i32[0], v343, v42 - 24),
                                        v96 = v343,
                                        v309) )
                                  {
LABEL_436:
                                    v296 = *(_BYTE *)(v96 + 16);
                                    if ( v296 <= 0x17u )
                                      goto LABEL_439;
                                    if ( v296 == 54 )
                                      goto LABEL_440;
                                    if ( v296 == 55 )
                                    {
                                      v96 = *(_QWORD *)(v96 - 48);
                                      if ( v96 )
                                      {
LABEL_440:
                                        if ( *(_QWORD *)(v42 - 16) )
                                          sub_164D160(
                                            v42 - 24,
                                            v96,
                                            a2,
                                            *(double *)a3.m128i_i64,
                                            a4,
                                            a5,
                                            v294,
                                            v295,
                                            a8,
                                            a9);
                                        sub_19003A0((__int64)v9, v42 - 24);
                                        sub_15F20C0((_QWORD *)(v42 - 24));
                                        v380 = 1;
                                        goto LABEL_40;
                                      }
                                    }
                                    else
                                    {
LABEL_439:
                                      v96 = sub_14A3740(v9->m128i_i64[1]);
                                      if ( v96 )
                                        goto LABEL_440;
                                    }
                                  }
                                  goto LABEL_338;
                                }
                                v269 = v399;
                                if ( v98 == -1 )
                                {
                                  v283 = *(_BYTE *)(v399 + 16);
                                  if ( v283 != 54 && v283 != 55 )
                                    goto LABEL_338;
                                  v284 = *(unsigned __int16 *)(v399 + 18);
                                  if ( (v284 & 1) != 0 || ((v284 >> 7) & 6) != 0 )
                                    goto LABEL_338;
                                  goto LABEL_399;
                                }
LABEL_340:
                                v100 = sub_15F32D0(v269);
                                if ( v395 )
                                {
                                  v99 = WORD2(v397);
                                  v91 = v9[15].m128i_i64[1];
                                  v90 = v9[16].m128i_u32[2];
                                  goto LABEL_115;
                                }
                                v101 = v9[37].m128i_i32[0];
                                v91 = v9[15].m128i_i64[1];
                                v90 = v9[16].m128i_u32[2];
                                v270 = *(_BYTE *)(v399 + 16);
                                if ( v270 <= 0x17u )
                                {
LABEL_365:
                                  v99 = -1;
                                  v102 = 0;
                                  goto LABEL_116;
                                }
                                if ( v270 == 54 || v270 == 55 )
                                {
                                  v102 = *(_QWORD *)(v399 - 24);
                                  v99 = -1;
                                  goto LABEL_116;
                                }
                                v99 = -1;
                                v102 = 0;
                                if ( v270 == 78 )
                                {
                                  v271 = *(_QWORD *)(v399 - 24);
                                  if ( !*(_BYTE *)(v271 + 16) )
                                  {
                                    v272 = *(_DWORD *)(v271 + 36);
                                    if ( v272 == 4085 || v272 == 4057 )
                                    {
                                      v99 = -1;
                                      v102 = *(_QWORD *)(v399 + 24 * (1LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF)));
                                      goto LABEL_116;
                                    }
                                    if ( v272 == 4503 || v272 == 4492 )
                                    {
                                      v99 = -1;
                                      v102 = *(_QWORD *)(v399 + 24 * (2LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF)));
                                      goto LABEL_116;
                                    }
                                    goto LABEL_365;
                                  }
                                }
LABEL_116:
                                v400[0].m128i_i64[0] = v102;
                                v103 = v9[17].m128i_i64[0];
                                m128i_i64 = (__int64)v9[15].m128i_i64;
                                if ( v90 )
                                {
                                  v104 = (v90 - 1) & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
                                  v105 = (__int64 *)(v91 + 16LL * v104);
                                  v106 = *v105;
                                  if ( v102 == *v105 )
                                  {
LABEL_118:
                                    v107 = v105[1];
LABEL_119:
                                    v108 = v9[17].m128i_i64[1];
                                    v109 = *(_QWORD *)(v103 + 16);
                                    if ( v108 )
                                    {
                                      v9[17].m128i_i64[1] = *(_QWORD *)v108;
                                    }
                                    else
                                    {
                                      v321 = v101;
                                      v327 = *(_QWORD *)(v103 + 16);
                                      v336 = v100;
                                      v357 = v99;
                                      v372 = v107;
                                      v108 = sub_145CBF0(v9[18].m128i_i64, 48, 8);
                                      v107 = v372;
                                      v99 = v357;
                                      v100 = v336;
                                      v109 = v327;
                                      v101 = v321;
                                    }
                                    v110 = v400[0].m128i_i64[0];
                                    *(_QWORD *)(v108 + 24) = v43;
                                    *(_DWORD *)(v108 + 32) = v101;
                                    *(_QWORD *)(v108 + 16) = v110;
                                    *(_DWORD *)(v108 + 36) = v99;
                                    *(_BYTE *)(v108 + 40) = v100;
                                    *(_QWORD *)v108 = v109;
                                    *(_QWORD *)(v108 + 8) = v107;
                                    v105[1] = v108;
                                    *(_QWORD *)(v103 + 16) = v108;
                                    goto LABEL_122;
                                  }
                                  v337 = 1;
                                  v358 = 0;
                                  while ( v106 != -8 )
                                  {
                                    if ( v106 != -16 || v358 )
                                      v105 = v358;
                                    v104 = (v90 - 1) & (v337 + v104);
                                    v106 = *(_QWORD *)(v91 + 16LL * v104);
                                    if ( v102 == v106 )
                                    {
                                      v105 = (__int64 *)(v91 + 16LL * v104);
                                      goto LABEL_118;
                                    }
                                    v358 = v105;
                                    v105 = (__int64 *)(v91 + 16LL * v104);
                                    ++v337;
                                  }
                                  if ( v358 )
                                    v105 = v358;
                                  v288 = v9[16].m128i_i32[0];
                                  ++v9[15].m128i_i64[0];
                                  v289 = v288 + 1;
                                  if ( 4 * (v288 + 1) < 3 * v90 )
                                  {
                                    if ( v90 - (v289 + v9[16].m128i_i32[1]) <= v90 >> 3 )
                                    {
                                      v329 = v101;
                                      v342 = v100;
                                      v360 = v99;
                                      sub_18FCC80(m128i_i64, v90);
                                      sub_18FB8D0(m128i_i64, v400[0].m128i_i64, &v402);
                                      v105 = (__int64 *)v402.m128i_i64[0];
                                      v102 = v400[0].m128i_i64[0];
                                      v101 = v329;
                                      v100 = v342;
                                      v99 = v360;
                                      v289 = v9[16].m128i_i32[0] + 1;
                                    }
                                    goto LABEL_424;
                                  }
                                }
                                else
                                {
                                  ++v9[15].m128i_i64[0];
                                }
                                v328 = v101;
                                v341 = v100;
                                v359 = v99;
                                sub_18FCC80(m128i_i64, 2 * v90);
                                sub_18FB8D0(m128i_i64, v400[0].m128i_i64, &v402);
                                v105 = (__int64 *)v402.m128i_i64[0];
                                v102 = v400[0].m128i_i64[0];
                                v99 = v359;
                                v100 = v341;
                                v101 = v328;
                                v289 = v9[16].m128i_i32[0] + 1;
LABEL_424:
                                v9[16].m128i_i32[0] = v289;
                                if ( *v105 != -8 )
                                  --v9[16].m128i_i32[1];
                                *v105 = v102;
                                v107 = 0;
                                v105[1] = 0;
                                goto LABEL_119;
                              }
LABEL_385:
                              v88 = *(_QWORD *)(v87 - 24);
                              v86 = 0;
                              goto LABEL_108;
                            }
LABEL_363:
                            v88 = v396;
                            v86 = 1;
                            goto LABEL_108;
                          }
                          goto LABEL_100;
                        }
                        goto LABEL_63;
                      }
                      if ( v395 )
                        goto LABEL_309;
                      v71 = v399;
                    }
                  }
                  if ( *(_BYTE *)(v71 + 16) == 54 && *(_QWORD *)(v71 - 24) )
                  {
                    v85 = *(unsigned __int16 *)(v71 + 18);
                    if ( (v85 & 1) == 0 && ((v85 >> 7) & 6) == 0 )
                    {
LABEL_102:
                      if ( *(_QWORD *)(v71 + 48) || *(__int16 *)(v71 + 18) < 0 )
                      {
                        if ( sub_1625790(v71, 6) )
                        {
                          sub_18FCF20(&v402, v42 - 24);
                          a2 = (__m128)_mm_loadu_si128(&v402);
                          a3 = _mm_loadu_si128(&v403);
                          v400[0] = (__m128i)a2;
                          v401 = v404;
                          v400[1] = a3;
                          if ( !(unsigned __int8)sub_18FEB70(
                                                   (__int64)&v9[24].m128i_i64[1],
                                                   v400[0].m128i_i64,
                                                   (__int64 **)&v402) )
                            sub_18FFC60((__int64)&v9[24].m128i_i64[1], v9[26].m128i_i64[1], v400, v9[37].m128i_i32);
                        }
                        goto LABEL_106;
                      }
                      v87 = v399;
                      if ( !v395 )
                        goto LABEL_385;
                      goto LABEL_363;
                    }
LABEL_100:
                    ++v9[37].m128i_i32[0];
                    v367 = 0;
                    goto LABEL_101;
                  }
LABEL_63:
                  if ( (unsigned __int8)sub_15F2ED0(v42 - 24) )
                  {
                    if ( !v395 )
                      goto LABEL_65;
                  }
                  else
                  {
                    if ( !sub_15F3330(v42 - 24) )
                      goto LABEL_67;
                    if ( !v395 )
                    {
LABEL_65:
                      v72 = *(_BYTE *)(v399 + 16);
                      if ( v72 <= 0x17u )
                        goto LABEL_66;
                      if ( v72 == 54 || v72 == 55 )
                      {
                        if ( !*(_QWORD *)(v399 - 24) )
                          goto LABEL_66;
                      }
                      else
                      {
                        if ( v72 != 78 )
                          goto LABEL_66;
                        v217 = *(_QWORD *)(v399 - 24);
                        if ( *(_BYTE *)(v217 + 16) )
                          goto LABEL_66;
                        v218 = *(_DWORD *)(v217 + 36);
                        if ( v218 == 4085 || v218 == 4057 )
                        {
                          if ( !*(_QWORD *)(v399 + 24 * (1LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF))) )
                            goto LABEL_66;
                        }
                        else if ( v218 != 4503 && v218 != 4492
                               || !*(_QWORD *)(v399 + 24 * (2LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF))) )
                        {
LABEL_66:
                          v367 = 0;
                          goto LABEL_67;
                        }
                      }
                      v216 = sub_15F2ED0(v399);
                      goto LABEL_250;
                    }
                  }
                  if ( !v396 )
                    goto LABEL_66;
                  v216 = BYTE6(v397);
LABEL_250:
                  if ( v216 )
                    goto LABEL_66;
LABEL_67:
                  v73 = *(_BYTE *)(v42 - 8);
                  if ( *(_BYTE *)(*(_QWORD *)(v42 - 24) + 8LL) && v73 == 78 )
                  {
                    if ( !(unsigned __int8)sub_1560260((_QWORD *)(v42 + 32), -1, 36) )
                    {
                      if ( *(char *)(v42 - 1) < 0 )
                      {
                        v167 = sub_1648A40(v42 - 24);
                        v169 = v167 + v168;
                        v170 = 0;
                        if ( *(char *)(v42 - 1) < 0 )
                        {
                          v353 = v169;
                          v171 = sub_1648A40(v42 - 24);
                          v169 = v353;
                          v170 = v171;
                        }
                        if ( (unsigned int)((v169 - v170) >> 4) )
                          goto LABEL_571;
                      }
                      v239 = *(_QWORD *)(v42 - 48);
                      if ( *(_BYTE *)(v239 + 16)
                        || (v402.m128i_i64[0] = *(_QWORD *)(v239 + 112), !(unsigned __int8)sub_1560260(&v402, -1, 36)) )
                      {
LABEL_571:
                        if ( !(unsigned __int8)sub_1560260((_QWORD *)(v42 + 32), -1, 37) )
                        {
                          if ( *(char *)(v42 - 1) < 0 )
                          {
                            v172 = sub_1648A40(v42 - 24);
                            v174 = v172 + v173;
                            v175 = *(char *)(v42 - 1) >= 0 ? 0LL : sub_1648A40(v42 - 24);
                            if ( v175 != v174 )
                            {
                              while ( *(_DWORD *)(*(_QWORD *)v175 + 8LL) <= 1u )
                              {
                                v175 += 16;
                                if ( v174 == v175 )
                                  goto LABEL_405;
                              }
LABEL_184:
                              v73 = *(_BYTE *)(v42 - 8);
                              goto LABEL_185;
                            }
                          }
LABEL_405:
                          v285 = *(_QWORD *)(v42 - 48);
                          if ( *(_BYTE *)(v285 + 16) )
                            goto LABEL_184;
                          v402.m128i_i64[0] = *(_QWORD *)(v285 + 112);
                          if ( !(unsigned __int8)sub_1560260(&v402, -1, 37) )
                            goto LABEL_184;
                        }
                      }
                    }
                    v400[0].m128i_i64[0] = v42 - 24;
                    v348 = (__int64)v9[34].m128i_i64;
                    if ( (unsigned __int8)sub_18FE7E0((__int64)v9[34].m128i_i64, v400[0].m128i_i64, (__int64 **)&v402) )
                    {
                      if ( v402.m128i_i64[0] != v9[34].m128i_i64[1] + 16LL * v9[35].m128i_u32[2] )
                      {
                        v74 = *(_QWORD *)(v402.m128i_i64[0] + 8);
                        if ( *(_QWORD *)(v74 + 24) )
                        {
                          v323 = *(_QWORD *)(v74 + 24);
                          v44 = sub_18FBB40((__int64)v9, *(_DWORD *)(v74 + 32), v9[37].m128i_i32[0], v323, v42 - 24);
                          if ( v44 )
                          {
                            if ( *(_QWORD *)(v42 - 16) )
                              sub_164D160(v42 - 24, v323, a2, *(double *)a3.m128i_i64, a4, a5, v75, v76, a8, a9);
                            goto LABEL_76;
                          }
                        }
                      }
                    }
                    v233 = v9[37].m128i_i32[0];
                    v400[0].m128i_i64[0] = v42 - 24;
                    v326 = v233;
                    v234 = v9[36].m128i_i64[0];
                    v235 = sub_18FE7E0(v348, v400[0].m128i_i64, (__int64 **)&v402);
                    v236 = (__int64 *)v402.m128i_i64[0];
                    if ( !v235 )
                    {
                      v279 = v9[35].m128i_u32[2];
                      v280 = v9[35].m128i_i32[0];
                      ++v9[34].m128i_i64[0];
                      v281 = v280 + 1;
                      if ( 4 * v281 >= 3 * v279 )
                      {
                        v279 *= 2;
                      }
                      else if ( v279 - v9[35].m128i_i32[1] - v281 > v279 >> 3 )
                      {
LABEL_380:
                        v9[35].m128i_i32[0] = v281;
                        if ( *v236 != -8 )
                          --v9[35].m128i_i32[1];
                        v282 = v400[0].m128i_i64[0];
                        v335 = 0;
                        v236[1] = 0;
                        *v236 = v282;
                        goto LABEL_285;
                      }
                      sub_18FE910(v348, v279);
                      sub_18FE7E0(v348, v400[0].m128i_i64, (__int64 **)&v402);
                      v236 = (__int64 *)v402.m128i_i64[0];
                      v281 = v9[35].m128i_i32[0] + 1;
                      goto LABEL_380;
                    }
                    v335 = *(_QWORD *)(v402.m128i_i64[0] + 8);
LABEL_285:
                    v356 = *(_QWORD *)(v234 + 16);
                    v237 = malloc(0x28u);
                    if ( !v237 )
                    {
                      sub_16BD1C0("Allocation failed", 1u);
                      v237 = 0;
                    }
                    v238 = v400[0].m128i_i64[0];
                    *(_QWORD *)(v237 + 24) = v43;
                    *(_DWORD *)(v237 + 32) = v326;
                    *(_QWORD *)(v237 + 16) = v238;
                    *(_QWORD *)v237 = v356;
                    *(_QWORD *)(v237 + 8) = v335;
                    v236[1] = v237;
                    *(_QWORD *)(v234 + 16) = v237;
                    if ( v387 == v41 )
                    {
LABEL_41:
                      v36 = v366;
                      goto LABEL_42;
                    }
                  }
                  else
                  {
LABEL_185:
                    if ( v73 == 57 && ((*(unsigned __int16 *)(v42 - 6) >> 1) & 0x7FFFBFFF) == 5 )
                      goto LABEL_40;
                    if ( v395 )
                    {
                      v176 = v396;
                      if ( !v396 )
                        goto LABEL_201;
                      if ( !HIBYTE(v397) )
                      {
                        if ( !(unsigned __int8)sub_15F3040(v42 - 24) )
                          goto LABEL_40;
                        goto LABEL_202;
                      }
                    }
                    else
                    {
                      if ( *(_BYTE *)(v399 + 16) != 55 )
                        goto LABEL_201;
                      v176 = *(_QWORD *)(v399 - 24);
                      if ( !v176 )
                        goto LABEL_201;
                    }
                    v177 = v9[16].m128i_u32[2];
                    if ( !(_DWORD)v177 )
                      goto LABEL_201;
                    v178 = v9[15].m128i_i64[1];
                    v179 = (v177 - 1) & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
                    v180 = (__int64 *)(v178 + 16LL * v179);
                    v181 = *v180;
                    if ( v176 != *v180 )
                    {
                      v297 = 1;
                      while ( v181 != -8 )
                      {
                        v298 = v297 + 1;
                        v179 = (v177 - 1) & (v297 + v179);
                        v180 = (__int64 *)(v178 + 16LL * v179);
                        v181 = *v180;
                        if ( v176 == *v180 )
                          goto LABEL_192;
                        v297 = v298;
                      }
                      goto LABEL_201;
                    }
LABEL_192:
                    if ( v180 == (__int64 *)(v178 + 16 * v177) )
                      goto LABEL_201;
                    v182 = v180[1];
                    v183 = *(_QWORD *)(v182 + 24);
                    v184 = *(_DWORD *)(v182 + 36);
                    v354 = *(_DWORD *)(v182 + 32);
                    if ( !v183 )
                      goto LABEL_201;
                    if ( v73 <= 0x17u )
                      goto LABEL_197;
                    if ( v73 != 54 )
                    {
                      if ( v73 == 55 )
                      {
                        v185 = *(_QWORD *)(v42 - 72);
                        if ( !v185 )
                          goto LABEL_201;
                        goto LABEL_198;
                      }
LABEL_197:
                      v332 = *(_QWORD *)(v182 + 24);
                      v185 = sub_14A3740(v9->m128i_i64[1]);
                      v183 = v332;
                      goto LABEL_198;
                    }
                    v185 = v42 - 24;
LABEL_198:
                    if ( v183 == v185 )
                    {
                      if ( v395 )
                      {
                        if ( WORD2(v397) == v184 )
                        {
                          LOBYTE(v299) = v398;
LABEL_452:
                          if ( !(_BYTE)v299 )
                          {
                            v340 = v183;
                            v44 = sub_18FBCB0((unsigned __int8 *)&v395);
                            if ( v44 )
                            {
                              if ( (unsigned __int8)sub_18FEEA0((__int64)v9, v42 - 24, v354)
                                || sub_18FBB40((__int64)v9, v354, v9[37].m128i_i32[0], v340, v42 - 24) )
                              {
                                goto LABEL_76;
                              }
                            }
                          }
                        }
                      }
                      else if ( v184 == -1 )
                      {
                        v315 = *(_BYTE *)(v399 + 16);
                        if ( v315 == 54 || v315 == 55 )
                        {
                          v299 = *(_WORD *)(v399 + 18) & 1;
                          goto LABEL_452;
                        }
                      }
                    }
LABEL_201:
                    if ( !(unsigned __int8)sub_15F3040(v42 - 24) )
                      goto LABEL_40;
LABEL_202:
                    v186 = !v395;
                    v187 = v9[37].m128i_i32[0] + 1;
                    v9[37].m128i_i32[0] = v187;
                    if ( v186 )
                    {
                      v188 = v399;
                      if ( *(_BYTE *)(v399 + 16) != 55 || !*(_QWORD *)(v399 - 24) )
                        goto LABEL_40;
                      if ( !v367 )
                        goto LABEL_468;
                    }
                    else
                    {
                      v191 = v396;
                      if ( !v396 || !HIBYTE(v397) )
                        goto LABEL_40;
                      if ( !v367 )
                        goto LABEL_211;
                    }
                    v189 = v9->m128i_i64[1];
                    v402 = 0u;
                    v403 = 0u;
                    v404 = v367;
                    if ( *(_BYTE *)(v367 + 16) == 78 )
                    {
                      v311 = *(_QWORD *)(v367 - 24);
                      if ( !*(_BYTE *)(v311 + 16)
                        && (*(_BYTE *)(v311 + 33) & 0x20) != 0
                        && (unsigned __int8)sub_14A36E0(v189) )
                      {
                        v402.m128i_i8[0] = 1;
                      }
                    }
                    v190 = sub_18FC460(v402.m128i_i8, (__int64)&v395);
                    if ( v190 )
                    {
                      sub_19003A0((__int64)v9, v367);
                      sub_15F20C0((_QWORD *)v367);
                      v380 = v190;
                    }
                    if ( v395 )
                    {
                      v191 = v396;
                      v187 = v9[37].m128i_i32[0];
LABEL_211:
                      v371 = (_DWORD)v397 != 0;
LABEL_212:
                      v192 = WORD2(v397);
                      goto LABEL_213;
                    }
                    v188 = v399;
LABEL_468:
                    v371 = sub_15F32D0(v188);
                    if ( v395 )
                    {
                      v191 = v396;
                      v187 = v9[37].m128i_i32[0];
                      goto LABEL_212;
                    }
                    v187 = v9[37].m128i_i32[0];
                    v304 = *(_BYTE *)(v399 + 16);
                    if ( v304 <= 0x17u )
                      goto LABEL_482;
                    if ( v304 == 54 || v304 == 55 )
                    {
                      v191 = *(_QWORD *)(v399 - 24);
                      v192 = -1;
                    }
                    else
                    {
                      v192 = -1;
                      v191 = 0;
                      if ( v304 == 78 )
                      {
                        v305 = *(_QWORD *)(v399 - 24);
                        if ( !*(_BYTE *)(v305 + 16) )
                        {
                          v306 = *(_DWORD *)(v305 + 36);
                          if ( v306 != 4085 && v306 != 4057 )
                          {
                            if ( v306 == 4503 || v306 == 4492 )
                            {
                              v192 = -1;
                              v191 = *(_QWORD *)(v399 + 24 * (2LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF)));
                              goto LABEL_213;
                            }
LABEL_482:
                            v192 = -1;
                            v191 = 0;
                            goto LABEL_213;
                          }
                          v192 = -1;
                          v191 = *(_QWORD *)(v399 + 24 * (1LL - (*(_DWORD *)(v399 + 20) & 0xFFFFFFF)));
                        }
                      }
                    }
LABEL_213:
                    v193 = v9[16].m128i_u32[2];
                    v400[0].m128i_i64[0] = v191;
                    v194 = v9[17].m128i_i64[0];
                    if ( !v193 )
                    {
                      ++v9[15].m128i_i64[0];
                      goto LABEL_537;
                    }
                    v195 = v9[15].m128i_i64[1];
                    v196 = (v193 - 1) & (((unsigned int)v191 >> 9) ^ ((unsigned int)v191 >> 4));
                    v197 = (__int64 *)(v195 + 16 * v196);
                    v198 = *v197;
                    if ( v191 != *v197 )
                    {
                      v345 = 1;
                      v312 = 0;
                      while ( v198 != -8 )
                      {
                        if ( !v312 && v198 == -16 )
                          v312 = v197;
                        LODWORD(v196) = (v193 - 1) & (v345 + v196);
                        v197 = (__int64 *)(v195 + 16LL * (unsigned int)v196);
                        v198 = *v197;
                        if ( v191 == *v197 )
                          goto LABEL_215;
                        ++v345;
                      }
                      if ( v312 )
                        v197 = v312;
                      v313 = v9[16].m128i_i32[0];
                      ++v9[15].m128i_i64[0];
                      v314 = v313 + 1;
                      if ( 4 * (v313 + 1) < 3 * v193 )
                      {
                        if ( v193 - v9[16].m128i_i32[1] - v314 <= v193 >> 3 )
                        {
                          v347 = v194;
                          v363 = v192;
                          sub_18FCC80((__int64)v9[15].m128i_i64, v193);
                          sub_18FB8D0((__int64)v9[15].m128i_i64, v400[0].m128i_i64, &v402);
                          v197 = (__int64 *)v402.m128i_i64[0];
                          v191 = v400[0].m128i_i64[0];
                          v194 = v347;
                          v192 = v363;
                          v314 = v9[16].m128i_i32[0] + 1;
                        }
                        goto LABEL_533;
                      }
LABEL_537:
                      v346 = v194;
                      v362 = v192;
                      sub_18FCC80((__int64)v9[15].m128i_i64, 2 * v193);
                      sub_18FB8D0((__int64)v9[15].m128i_i64, v400[0].m128i_i64, &v402);
                      v197 = (__int64 *)v402.m128i_i64[0];
                      v191 = v400[0].m128i_i64[0];
                      v192 = v362;
                      v194 = v346;
                      v314 = v9[16].m128i_i32[0] + 1;
LABEL_533:
                      v9[16].m128i_i32[0] = v314;
                      if ( *v197 != -8 )
                        --v9[16].m128i_i32[1];
                      *v197 = v191;
                      v199 = 0;
                      v197[1] = 0;
                      goto LABEL_216;
                    }
LABEL_215:
                    v199 = v197[1];
LABEL_216:
                    v200 = v9[17].m128i_i64[1];
                    v201 = *(_QWORD *)(v194 + 16);
                    if ( v200 )
                    {
                      v9[17].m128i_i64[1] = *(_QWORD *)v200;
                    }
                    else
                    {
                      v322 = *(_QWORD *)(v194 + 16);
                      v330 = v197;
                      v344 = v194;
                      v361 = v192;
                      v200 = sub_145CBF0(v9[18].m128i_i64, 48, 8);
                      v192 = v361;
                      v194 = v344;
                      v197 = v330;
                      v201 = v322;
                    }
                    v202 = v400[0].m128i_i64[0];
                    *(_QWORD *)v200 = v201;
                    v186 = !v395;
                    *(_QWORD *)(v200 + 24) = v43;
                    *(_QWORD *)(v200 + 16) = v202;
                    *(_DWORD *)(v200 + 32) = v187;
                    *(_DWORD *)(v200 + 36) = v192;
                    *(_BYTE *)(v200 + 40) = v371;
                    *(_QWORD *)(v200 + 8) = v199;
                    v197[1] = v200;
                    *(_QWORD *)(v194 + 16) = v200;
                    if ( !v186 )
                    {
                      v367 = 0;
                      if ( (unsigned int)v397 <= 1 )
                      {
                        if ( (_BYTE)v398 )
                          v43 = 0;
                        v367 = v43;
                      }
                      goto LABEL_40;
                    }
                    v300 = v399;
                    v301 = *(_BYTE *)(v399 + 16);
                    if ( v301 == 54 || v301 == 55 )
                    {
                      v367 = 0;
                      v302 = *(_WORD *)(v399 + 18);
                      if ( ((v302 >> 7) & 6) != 0 || (v302 & 1) != 0 )
                        goto LABEL_40;
LABEL_464:
                      v303 = *(_WORD *)(v300 + 18) & 1;
LABEL_465:
                      if ( !(_BYTE)v303 )
                      {
                        v367 = v43;
                        goto LABEL_40;
                      }
                      goto LABEL_122;
                    }
                    v367 = 0;
                    if ( sub_15F32D0(v399) )
                      goto LABEL_40;
                    if ( v395 )
                    {
                      LOBYTE(v303) = v398;
                      goto LABEL_465;
                    }
                    v300 = v399;
                    v310 = *(_BYTE *)(v399 + 16);
                    if ( v310 == 54 || v310 == 55 )
                      goto LABEL_464;
LABEL_122:
                    v367 = 0;
LABEL_123:
                    if ( v387 == v41 )
                      goto LABEL_41;
                  }
                }
                else
                {
                  v111 = (unsigned int)(v64 - 35);
                  if ( (unsigned __int8)v111 > 0x34u )
                    goto LABEL_61;
                  v112 = 0x1F133FFE23FFFFLL;
                  if ( !_bittest64(&v112, v111) )
                    goto LABEL_61;
                  v77 = v9[7].m128i_i32[0];
                  if ( !v77 )
                    goto LABEL_80;
LABEL_128:
                  v331 = v9[6].m128i_i64[0];
                  v324 = 1;
                  v319 = v77 - 1;
                  v351 = (v77 - 1) & sub_18FDEE0(v42 - 24);
                  while ( 1 )
                  {
                    v113 = (__int64 *)(v331 + 16LL * v351);
                    v44 = sub_18FB980(v43, *v113);
                    if ( v44 )
                      break;
                    if ( *v113 == -8 )
                      goto LABEL_80;
                    v351 = v319 & (v324 + v351);
                    ++v324;
                  }
                  if ( v113 != (__int64 *)(v9[6].m128i_i64[0] + 16LL * v9[7].m128i_u32[0]) )
                  {
                    v116 = v113[1];
                    v117 = *(_QWORD *)(v116 + 24);
                    if ( v117 )
                    {
                      if ( *(_BYTE *)(v117 + 16) > 0x17u )
                        sub_15F2780(*(unsigned __int8 **)(v116 + 24), v43);
                      sub_164D160(v43, v117, a2, *(double *)a3.m128i_i64, a4, a5, v114, v115, a8, a9);
                      goto LABEL_76;
                    }
                  }
LABEL_80:
                  v400[0].m128i_i64[0] = v43;
                  v402.m128i_i64[0] = v43;
LABEL_81:
                  sub_18FEF10((__int64)&v9[5].m128i_i64[1], v9[7].m128i_i64[1], v402.m128i_i64, v400);
                  if ( v387 == v41 )
                    goto LABEL_41;
                }
                break;
            }
          }
        }
LABEL_42:
        v55 = v9[37].m128i_i32[0];
        *(_BYTE *)(v36 + 128) = 1;
        v365 |= v380;
        *(_DWORD *)(v36 + 4) = v55;
        v24 = (__int64)v411;
        if ( v407 == v411 )
          goto LABEL_43;
      }
      v25 = *(__int64 **)(v36 + 16);
      if ( v25 == *(__int64 **)(v36 + 24) )
        break;
      v26 = *v25;
      v27 = *(_DWORD *)(v36 + 4);
      *(_QWORD *)(v36 + 16) = v25 + 1;
      v28 = *(_QWORD *)(v26 + 24);
      v29 = *(_QWORD *)(v26 + 32);
      v30 = sub_22077B0(136);
      if ( v30 )
      {
        *(_DWORD *)v30 = v27;
        *(_QWORD *)(v30 + 32) = (char *)v9 + 88;
        v31 = v9[7].m128i_i64[1];
        *(_DWORD *)(v30 + 4) = v27;
        *(_QWORD *)(v30 + 40) = v31;
        v9[7].m128i_i64[1] = v30 + 32;
        *(_QWORD *)(v30 + 56) = v9 + 15;
        v32 = v9[17].m128i_i64[0];
        *(_QWORD *)(v30 + 8) = v26;
        *(_QWORD *)(v30 + 64) = v32;
        v9[17].m128i_i64[0] = v30 + 56;
        *(_QWORD *)(v30 + 80) = (char *)v9 + 392;
        v33 = v9[26].m128i_i64[1];
        *(_QWORD *)(v30 + 16) = v28;
        *(_QWORD *)(v30 + 88) = v33;
        v9[26].m128i_i64[1] = v30 + 80;
        *(_QWORD *)(v30 + 104) = v9 + 34;
        v34 = v9[36].m128i_i64[0];
        *(_QWORD *)(v30 + 24) = v29;
        *(_QWORD *)(v30 + 112) = v34;
        *(_QWORD *)(v30 + 48) = 0;
        *(_QWORD *)(v30 + 72) = 0;
        *(_QWORD *)(v30 + 96) = 0;
        v9[36].m128i_i64[0] = v30 + 104;
        *(_QWORD *)(v30 + 120) = 0;
        *(_BYTE *)(v30 + 128) = 0;
      }
      v35 = v411;
      v402.m128i_i64[0] = v30;
      if ( v411 == (_QWORD *)(v413 - 8) )
      {
        sub_18FC260(&v405, &v402);
        v24 = (__int64)v411;
      }
      else
      {
        if ( v411 )
        {
          *v411 = v30;
          v35 = v411;
        }
        v24 = (__int64)(v35 + 1);
        v411 = v35 + 1;
      }
LABEL_16:
      if ( v407 == (_QWORD *)v24 )
        goto LABEL_43;
    }
    *(_QWORD *)(*(_QWORD *)(v36 + 104) + 32LL) = *(_QWORD *)(v36 + 112);
    if ( !*(_QWORD *)(v36 + 120) )
      goto LABEL_156;
    v352 = v9;
    v118 = *(_QWORD **)(v36 + 120);
LABEL_137:
    v119 = *(_QWORD *)(v36 + 104);
    v120 = v118 + 2;
    v121 = *(_DWORD *)(v119 + 24);
    v388 = *(_QWORD *)(v119 + 8);
    if ( !v118[1] )
    {
      if ( !v121 )
        goto LABEL_154;
      v157 = v121 - 1;
      v158 = sub_18FE780(*v120);
      v159 = v388;
      v160 = 1;
      for ( i = v157 & v158; ; i = v157 & v241 )
      {
        v162 = v118[2];
        v163 = (__int64 *)(v159 + 16LL * i);
        v164 = *v163;
        v165 = *v163 == -8;
        v166 = *v163 == -16;
        if ( v165 || v162 == -16 || v162 == -8 || *v163 == -16 )
        {
          if ( v162 == v164 )
            goto LABEL_171;
        }
        else
        {
          v377 = v159;
          v383 = v160;
          v392 = i;
          v240 = sub_15F41F0(v162, v164);
          i = v392;
          v160 = v383;
          v159 = v377;
          if ( v240 )
          {
LABEL_171:
            *v163 = -16;
            --*(_DWORD *)(v119 + 16);
            ++*(_DWORD *)(v119 + 20);
            goto LABEL_154;
          }
          v164 = *v163;
          v165 = *v163 == -8;
          v166 = *v163 == -16;
        }
        if ( (v165 || v166) && v164 == -8 )
          goto LABEL_154;
        v241 = v160 + i;
        ++v160;
      }
    }
    if ( !v121 )
    {
      ++*(_QWORD *)v119;
      goto LABEL_140;
    }
    v135 = v121 - 1;
    v136 = sub_18FE780(*v120);
    v137 = v388;
    v370 = 1;
    v376 = 0;
    v120 = v118 + 2;
    for ( j = v135 & v136; ; j = v135 & v204 )
    {
      v139 = v118[2];
      v122 = (__int64 *)(v137 + 16LL * j);
      v140 = *v122;
      v141 = *v122 == -8;
      v142 = *v122 == -16;
      if ( v141 || v139 == -16 || v139 == -8 || *v122 == -16 )
      {
        if ( v139 == v140 )
          goto LABEL_153;
      }
      else
      {
        v333 = v120;
        v382 = v137;
        v391 = j;
        v203 = sub_15F41F0(v139, v140);
        j = v391;
        v137 = v382;
        v120 = v333;
        if ( v203 )
          goto LABEL_153;
        v140 = *v122;
        v142 = *v122 == -16;
        v141 = *v122 == -8;
      }
      if ( v141 || v142 )
      {
        if ( v140 == -8 )
        {
          v121 = *(_DWORD *)(v119 + 24);
          if ( v376 )
            v122 = v376;
          v251 = *(_DWORD *)(v119 + 16);
          ++*(_QWORD *)v119;
          v133 = v251 + 1;
          if ( 4 * v133 < 3 * v121 )
          {
            if ( v121 - (v133 + *(_DWORD *)(v119 + 20)) <= v121 >> 3 )
            {
              v384 = v120;
              v122 = 0;
              sub_18FE910(v119, v121);
              v393 = *(_DWORD *)(v119 + 24);
              if ( !v393 )
                goto LABEL_145;
              v252 = *(_QWORD *)(v119 + 8);
              v253 = sub_18FE780(*v384);
              v254 = v393;
              v126 = 0;
              v394 = 1;
              v255 = v254 - 1;
              for ( k = (v254 - 1) & v253; ; k = v255 & v261 )
              {
                v257 = v118[2];
                v122 = (__int64 *)(v252 + 16LL * k);
                v258 = *v122;
                v259 = *v122 == -8;
                v260 = *v122 == -16;
                if ( v259 || v257 == -8 || v257 == -16 || *v122 == -16 )
                {
                  if ( v257 == v258 )
                    goto LABEL_145;
                }
                else
                {
                  v374 = v126;
                  v379 = k;
                  v386 = v255;
                  v307 = sub_15F41F0(v257, v258);
                  v255 = v386;
                  k = v379;
                  v126 = v374;
                  if ( v307 )
                    goto LABEL_145;
                  v258 = *v122;
                  v259 = *v122 == -8;
                  v260 = *v122 == -16;
                }
                if ( v260 || v259 )
                {
                  if ( v258 == -8 )
                  {
LABEL_513:
                    if ( v126 )
                      v122 = v126;
                    goto LABEL_145;
                  }
                  if ( !v126 && v260 )
                    v126 = v122;
                }
                v261 = v394 + k;
                ++v394;
              }
            }
            goto LABEL_146;
          }
LABEL_140:
          v381 = v120;
          v122 = 0;
          sub_18FE910(v119, 2 * v121);
          v389 = *(_DWORD *)(v119 + 24);
          if ( v389 )
          {
            v123 = *(_QWORD *)(v119 + 8);
            v124 = sub_18FE780(*v381);
            v125 = v389;
            v126 = 0;
            v390 = 1;
            v127 = v125 - 1;
            for ( m = (v125 - 1) & v124; ; m = v127 & v291 )
            {
              v129 = v118[2];
              v122 = (__int64 *)(v123 + 16LL * m);
              v130 = *v122;
              v131 = *v122 == -8;
              v132 = *v122 == -16;
              if ( v131 || v129 == -16 || v129 == -8 || *v122 == -16 )
              {
                if ( v129 == v130 )
                  break;
              }
              else
              {
                v373 = v126;
                v378 = m;
                v385 = v127;
                v290 = sub_15F41F0(v129, v130);
                v127 = v385;
                m = v378;
                v126 = v373;
                if ( v290 )
                  break;
                v130 = *v122;
                v131 = *v122 == -8;
                v132 = *v122 == -16;
              }
              if ( v131 || v132 )
              {
                if ( v130 == -8 )
                  goto LABEL_513;
                if ( !v126 && v132 )
                  v126 = v122;
              }
              v291 = v390 + m;
              ++v390;
            }
          }
LABEL_145:
          v133 = *(_DWORD *)(v119 + 16) + 1;
LABEL_146:
          *(_DWORD *)(v119 + 16) = v133;
          if ( *v122 != -8 )
            --*(_DWORD *)(v119 + 20);
          v134 = v118[2];
          v122[1] = 0;
          *v122 = v134;
LABEL_153:
          v122[1] = v118[1];
LABEL_154:
          *(_QWORD *)(v36 + 120) = *v118;
          _libc_free((unsigned __int64)v118);
          v118 = *(_QWORD **)(v36 + 120);
          if ( !v118 )
          {
            v9 = v352;
LABEL_156:
            sub_18FFE20((__int64 *)(v36 + 80));
            *(_QWORD *)(*(_QWORD *)(v36 + 56) + 32LL) = *(_QWORD *)(v36 + 64);
            v143 = *(_QWORD **)(v36 + 72);
            if ( !v143 )
            {
LABEL_172:
              sub_18FE3A0((__int64 *)(v36 + 32));
              j_j___libc_free_0(v36, 136);
              if ( v411 == v412 )
              {
                j_j___libc_free_0(v411, 512);
                v274 = *--v414 + 512LL;
                v412 = (_QWORD *)*v414;
                v24 = (__int64)(v412 + 63);
                v413 = v274;
                v411 = v412 + 63;
              }
              else
              {
                v24 = (__int64)--v411;
              }
              goto LABEL_16;
            }
            while ( 2 )
            {
              v148 = *(_QWORD *)(v36 + 56);
              v149 = v143[1];
              v150 = *(_QWORD *)(v148 + 8);
              v151 = *(_DWORD *)(v148 + 24);
              if ( !v149 )
              {
                if ( v151 )
                {
                  v152 = v143[2];
                  v153 = v151 - 1;
                  v154 = v153 & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
                  v155 = (__int64 *)(v150 + 16LL * v154);
                  v156 = *v155;
                  if ( *v155 == v152 )
                  {
LABEL_165:
                    *v155 = -16;
                    --*(_DWORD *)(v148 + 16);
                    ++*(_DWORD *)(v148 + 20);
                    v148 = *(_QWORD *)(v36 + 56);
                  }
                  else
                  {
                    v275 = 1;
                    while ( v156 != -8 )
                    {
                      v276 = v275 + 1;
                      v154 = v153 & (v275 + v154);
                      v155 = (__int64 *)(v150 + 16LL * v154);
                      v156 = *v155;
                      if ( v152 == *v155 )
                        goto LABEL_165;
                      v275 = v276;
                    }
                  }
                }
LABEL_161:
                *(_QWORD *)(v36 + 72) = *v143;
                *v143 = *(_QWORD *)(v148 + 40);
                *(_QWORD *)(v148 + 40) = v143;
                v143 = *(_QWORD **)(v36 + 72);
                if ( !v143 )
                  goto LABEL_172;
                continue;
              }
              break;
            }
            if ( v151 )
            {
              v144 = v143[2];
              v145 = (v151 - 1) & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
              v146 = (_QWORD *)(v150 + 16LL * v145);
              v147 = *v146;
              if ( *v146 == v144 )
              {
LABEL_160:
                v146[1] = v149;
                v148 = *(_QWORD *)(v36 + 56);
                goto LABEL_161;
              }
              v211 = 1;
              v212 = 0;
              while ( v147 != -8 )
              {
                if ( v147 == -16 && !v212 )
                  v212 = v146;
                v145 = (v151 - 1) & (v211 + v145);
                v146 = (_QWORD *)(v150 + 16LL * v145);
                v147 = *v146;
                if ( v144 == *v146 )
                  goto LABEL_160;
                ++v211;
              }
              v213 = *(_DWORD *)(v148 + 16);
              if ( v212 )
                v146 = v212;
              ++*(_QWORD *)v148;
              v214 = v213 + 1;
              if ( 4 * v214 < 3 * v151 )
              {
                if ( v151 - *(_DWORD *)(v148 + 20) - v214 <= v151 >> 3 )
                {
                  sub_18FCC80(v148, v151);
                  v262 = *(_DWORD *)(v148 + 24);
                  if ( !v262 )
                  {
LABEL_565:
                    ++*(_DWORD *)(v148 + 16);
                    BUG();
                  }
                  v263 = v143[2];
                  v264 = v262 - 1;
                  v265 = *(_QWORD *)(v148 + 8);
                  v249 = 0;
                  v266 = 1;
                  v214 = *(_DWORD *)(v148 + 16) + 1;
                  v267 = (v262 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
                  v146 = (_QWORD *)(v265 + 16LL * v267);
                  v268 = *v146;
                  if ( v263 != *v146 )
                  {
                    while ( v268 != -8 )
                    {
                      if ( !v249 && v268 == -16 )
                        v249 = v146;
                      v267 = v264 & (v266 + v267);
                      v146 = (_QWORD *)(v265 + 16LL * v267);
                      v268 = *v146;
                      if ( v263 == *v146 )
                        goto LABEL_243;
                      ++v266;
                    }
                    goto LABEL_302;
                  }
                }
                goto LABEL_243;
              }
            }
            else
            {
              ++*(_QWORD *)v148;
            }
            sub_18FCC80(v148, 2 * v151);
            v242 = *(_DWORD *)(v148 + 24);
            if ( !v242 )
              goto LABEL_565;
            v243 = v143[2];
            v244 = v242 - 1;
            v245 = *(_QWORD *)(v148 + 8);
            v214 = *(_DWORD *)(v148 + 16) + 1;
            v246 = (v242 - 1) & (((unsigned int)v243 >> 9) ^ ((unsigned int)v243 >> 4));
            v146 = (_QWORD *)(v245 + 16LL * v246);
            v247 = *v146;
            if ( v243 != *v146 )
            {
              v248 = 1;
              v249 = 0;
              while ( v247 != -8 )
              {
                if ( !v249 && v247 == -16 )
                  v249 = v146;
                v246 = v244 & (v248 + v246);
                v146 = (_QWORD *)(v245 + 16LL * v246);
                v247 = *v146;
                if ( v243 == *v146 )
                  goto LABEL_243;
                ++v248;
              }
LABEL_302:
              if ( v249 )
                v146 = v249;
            }
LABEL_243:
            *(_DWORD *)(v148 + 16) = v214;
            if ( *v146 != -8 )
              --*(_DWORD *)(v148 + 20);
            v215 = v143[2];
            v146[1] = 0;
            *v146 = v215;
            v149 = v143[1];
            goto LABEL_160;
          }
          goto LABEL_137;
        }
        if ( !v376 )
        {
          if ( !v142 )
            v122 = 0;
          v376 = v122;
        }
      }
      v204 = v370 + j;
      ++v370;
    }
  }
LABEL_43:
  v56 = v405;
  v9[37].m128i_i32[0] = v364;
  if ( v56 )
  {
    v57 = (__int64 *)v410;
    v58 = v414 + 1;
    if ( (unsigned __int64)(v414 + 1) > v410 )
    {
      do
      {
        v59 = *v57++;
        j_j___libc_free_0(v59, 512);
      }
      while ( v58 > v57 );
      v56 = v405;
    }
    j_j___libc_free_0(v56, 8 * v406);
  }
  return v365;
}
