// Function: sub_795660
// Address: 0x795660
//
__int64 __fastcall sub_795660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  FILE *v7; // r13
  __int64 v8; // rcx
  _QWORD *v9; // r12
  __int64 v10; // r14
  unsigned __int64 v11; // rbx
  char m; // al
  __int64 v13; // r13
  size_t v14; // rdx
  __int64 v15; // rax
  char *v16; // r8
  _WORD *v17; // r13
  __int64 result; // rax
  _QWORD *v19; // rdx
  __m128i *v20; // r8
  char *v21; // rcx
  unsigned int v22; // esi
  int v23; // r12d
  __int64 v24; // r14
  __int64 *v25; // rbx
  int v26; // r11d
  unsigned int v27; // edx
  __int64 v28; // rdi
  int *v29; // rax
  int v30; // r9d
  int v31; // eax
  unsigned int v32; // eax
  bool v33; // zf
  unsigned int v34; // edi
  int v35; // esi
  __int64 v36; // rcx
  __int64 *v37; // r13
  unsigned int v38; // edx
  _DWORD *i14; // rax
  unsigned __int64 *v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rsi
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 *v46; // rbx
  unsigned __int64 v47; // r14
  char nn; // al
  __int64 v49; // rbx
  size_t v50; // r8
  __int64 v51; // r12
  char *v52; // rcx
  char *v53; // r12
  unsigned __int64 *v55; // r15
  __int64 v56; // rcx
  char v57; // dl
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  unsigned int v60; // ecx
  __int64 v61; // rsi
  char *IO_backup_base; // rbx
  int v63; // edx
  unsigned int v64; // eax
  int *v65; // r8
  int v66; // edi
  int v67; // eax
  __int64 v68; // r9
  unsigned __int64 v69; // r14
  unsigned __int64 v70; // rsi
  __int64 v71; // rbx
  FILE *v72; // rcx
  __int64 v73; // r13
  FILE *v74; // r15
  _QWORD *v75; // rax
  unsigned int v76; // edx
  int *v77; // rdx
  unsigned __int64 *v78; // rdi
  __int64 v79; // rax
  __int64 *v80; // rbx
  unsigned __int64 v81; // r14
  char i10; // al
  __int64 v83; // r12
  size_t v84; // r8
  __int64 v85; // rax
  char *v86; // rdi
  _QWORD *v87; // rcx
  char v88; // al
  unsigned int v89; // r9d
  unsigned int v90; // edx
  unsigned int v91; // eax
  __int64 v92; // r12
  size_t v93; // rdx
  unsigned int v94; // ecx
  char *v95; // rdi
  void *v96; // rax
  __int64 v97; // r8
  __int64 v98; // r9
  unsigned __int64 v99; // r12
  __int16 *v100; // r14
  int v101; // eax
  const __m128i *v102; // rsi
  __int8 v103; // al
  const __m128i *v104; // rsi
  __int8 v105; // al
  unsigned __int64 v106; // r12
  _QWORD *v107; // rdx
  _BYTE *v108; // rbx
  unsigned __int64 v109; // r13
  char i; // al
  int v111; // r9d
  unsigned int v112; // edi
  __int64 v113; // rcx
  __m128i v114; // xmm2
  int v115; // eax
  __m128i v116; // xmm1
  int v117; // edx
  unsigned int v118; // eax
  int *v119; // rsi
  int v120; // r8d
  int v121; // eax
  unsigned int v122; // eax
  unsigned int v123; // r12d
  __int64 v124; // r14
  size_t v125; // rdx
  __int64 v126; // r12
  int v127; // ecx
  __int64 v128; // rax
  char *v129; // rcx
  char *v130; // r14
  unsigned int v131; // edi
  int v132; // esi
  __int64 v133; // rcx
  __int64 *v134; // r12
  unsigned int v135; // edx
  _DWORD *k; // rax
  __m128i v137; // xmm3
  __m128i v138; // xmm4
  __int64 v139; // rax
  __int64 v140; // rcx
  __int64 v141; // rsi
  unsigned int v142; // eax
  __int64 v143; // rdx
  __int64 *v144; // rbx
  __int64 *v145; // r14
  __int64 v146; // rax
  unsigned __int64 v147; // r10
  char *v148; // r8
  unsigned int v149; // ecx
  __int64 v150; // rsi
  __int64 v151; // r12
  __int64 *v152; // rbx
  int v153; // edx
  unsigned int v154; // eax
  int *v155; // rdi
  int v156; // r9d
  int v157; // eax
  unsigned int v158; // eax
  unsigned int v159; // edi
  int v160; // esi
  __int64 v161; // rcx
  __int64 *v162; // r13
  unsigned int v163; // edx
  _DWORD *ii; // rax
  unsigned __int64 *v165; // rax
  __int64 v166; // rcx
  __int64 v167; // rsi
  unsigned int v168; // eax
  __int64 v169; // rdx
  __int64 *v170; // rbx
  unsigned __int64 v171; // r14
  char n; // al
  __int64 v173; // r12
  size_t v174; // rbx
  __int64 v175; // rax
  char *v176; // rdi
  _WORD *v177; // r12
  __int64 v178; // rbx
  unsigned __int64 *v179; // rcx
  unsigned __int64 v180; // rax
  bool v181; // r11
  char v182; // al
  _DWORD *v183; // rdx
  int *v184; // rax
  int *v185; // rdx
  __int64 v186; // rax
  int v187; // r12d
  unsigned int v188; // edx
  unsigned int v189; // eax
  unsigned int v190; // r12d
  __int64 v191; // rax
  __int64 v192; // rdx
  __int64 v193; // rsi
  __int64 *v194; // rax
  __int64 i15; // rdx
  unsigned int v196; // eax
  __int64 v197; // rax
  int v198; // edx
  int v199; // esi
  unsigned int v200; // edx
  unsigned int v201; // eax
  __int64 v202; // rax
  int v203; // esi
  unsigned int v204; // ecx
  int v205; // edi
  __int64 v206; // rsi
  __int64 *v207; // r12
  unsigned int v208; // edx
  _DWORD *i9; // rax
  __int64 v210; // rcx
  int v211; // edi
  __int64 v212; // rsi
  unsigned int v213; // eax
  __int64 v214; // rdx
  __int64 *v215; // rbx
  __int64 v216; // rbx
  __int64 v217; // r14
  FILE *v218; // r15
  __int64 v219; // rcx
  char v220; // al
  __int64 v221; // rax
  unsigned int v222; // eax
  __int64 v223; // rax
  unsigned int v224; // ecx
  __int64 v225; // rsi
  __int64 *v226; // rbx
  int v227; // edx
  unsigned int v228; // eax
  int *v229; // r9
  int v230; // edi
  int v231; // eax
  unsigned int v232; // eax
  unsigned int v233; // edi
  int v234; // esi
  __int64 v235; // rcx
  __int64 *v236; // r9
  unsigned int v237; // edx
  _DWORD *i1; // rax
  __int64 v239; // rcx
  __int64 v240; // rsi
  unsigned int v241; // eax
  __int64 v242; // rdx
  __int64 *v243; // rbx
  int *v244; // rdx
  unsigned int v245; // r9d
  __int64 v246; // r9
  __int64 v247; // rsi
  __int64 v248; // rdx
  _QWORD *v249; // rcx
  _QWORD *i2; // rax
  __int64 v251; // rax
  FILE *v252; // rax
  FILE *v253; // r13
  __int64 v254; // rsi
  __int64 v255; // r12
  __int64 v256; // r13
  __int64 v257; // rax
  unsigned __int64 v258; // r15
  unsigned __int64 v259; // r14
  unsigned __int64 v260; // r12
  unsigned __int64 *v261; // rbx
  unsigned __int64 v262; // rax
  _QWORD *v263; // r12
  unsigned __int64 i8; // rsi
  unsigned int v265; // r12d
  __int64 v266; // rax
  __int64 v267; // rcx
  __int64 v268; // rax
  bool v269; // bl
  unsigned __int64 *v270; // rbx
  unsigned __int64 v271; // r12
  char i3; // al
  unsigned __int64 v273; // rdx
  int v274; // edx
  size_t v275; // r8
  __int64 v276; // r9
  unsigned int v277; // ebx
  unsigned int v278; // ebx
  char *v279; // rcx
  _QWORD *v280; // rbx
  char v281; // al
  int v282; // edx
  size_t v283; // r8
  __int64 v284; // r9
  unsigned int v285; // esi
  unsigned int v286; // esi
  char *v287; // rcx
  _QWORD *v288; // rcx
  char v289; // al
  char v290; // al
  __int64 v291; // r13
  char v292; // al
  __int64 v293; // rbx
  unsigned int v294; // esi
  __int64 v295; // rdi
  int v296; // edx
  unsigned int v297; // eax
  int *v298; // r10
  int v299; // r11d
  int v300; // eax
  unsigned __int64 *v301; // rsi
  int v302; // eax
  unsigned int v303; // r11d
  int v304; // edi
  __int64 v305; // rsi
  _QWORD *v306; // r10
  unsigned int v307; // edx
  _DWORD *i12; // rax
  _QWORD *v309; // rax
  __int64 v310; // rcx
  int v311; // edi
  __int64 v312; // rsi
  unsigned int v313; // eax
  __int64 v314; // rdx
  __int64 v315; // rsi
  int *v316; // rdx
  __int64 v317; // rax
  __int64 v318; // rdi
  __int64 v319; // rdx
  _QWORD *v320; // rsi
  _QWORD *i13; // rax
  __int64 v322; // rax
  __int64 jj; // rax
  __int64 kk; // rdx
  unsigned int v325; // eax
  __int64 v326; // rdx
  char v327; // al
  __int64 v328; // rax
  int v329; // esi
  unsigned int v330; // edx
  unsigned int v331; // eax
  __int64 v332; // rax
  int v333; // eax
  __int64 v334; // rsi
  __int32 v335; // eax
  int v336; // eax
  __int64 v337; // rsi
  __int64 v338; // rdx
  _QWORD *v339; // rcx
  _QWORD *j; // rax
  __int64 v341; // rax
  __int32 v342; // eax
  __int64 v343; // rax
  __int64 v344; // rcx
  __int64 v345; // rax
  char v346; // bl
  unsigned int v347; // ecx
  __int64 v348; // rsi
  __int64 *v349; // rbx
  int v350; // edx
  unsigned int v351; // eax
  int *v352; // r8
  int v353; // edi
  int v354; // eax
  unsigned int v355; // eax
  unsigned int v356; // edi
  int v357; // esi
  __int64 v358; // rcx
  __int64 *v359; // r8
  unsigned int v360; // edx
  _DWORD *i4; // rax
  __int64 v362; // rcx
  int v363; // edi
  __int64 v364; // rsi
  unsigned int v365; // eax
  __int64 v366; // rdx
  __int64 *v367; // rbx
  int *v368; // rdx
  unsigned int v369; // r8d
  __int64 v370; // rax
  __int64 v371; // rsi
  __int64 v372; // rdx
  _QWORD *v373; // rcx
  _QWORD *i5; // rax
  __int64 v375; // rax
  __int32 v376; // edx
  __int64 v377; // rdx
  char v378; // al
  unsigned int v379; // ecx
  __int64 v380; // rdi
  int v381; // edx
  unsigned int v382; // eax
  int *v383; // rsi
  int v384; // r8d
  int v385; // eax
  unsigned int v386; // eax
  unsigned int v387; // edi
  int v388; // esi
  __int64 v389; // rcx
  __int64 *v390; // rbx
  unsigned int v391; // edx
  _DWORD *i6; // rax
  __int64 v393; // rcx
  int v394; // r8d
  __int64 *v395; // rdi
  __int64 v396; // rsi
  unsigned int v397; // eax
  __int64 v398; // rdx
  int *v399; // rdx
  unsigned int v400; // ebx
  __int64 v401; // rax
  __int64 v402; // rcx
  __int64 v403; // rax
  __int64 mm; // rax
  FILE *v405; // rsi
  char *IO_read_base; // rdx
  unsigned int v407; // ecx
  unsigned int v408; // eax
  unsigned int v409; // eax
  __int64 v410; // rcx
  __int64 v411; // rdx
  _QWORD *v412; // rsi
  _QWORD *i7; // rax
  __int64 v414; // rax
  unsigned int v415; // eax
  unsigned __int64 v416; // [rsp+8h] [rbp-148h]
  FILE *v417; // [rsp+10h] [rbp-140h]
  unsigned __int64 v418; // [rsp+18h] [rbp-138h]
  __int64 v419; // [rsp+20h] [rbp-130h]
  unsigned __int64 *v420; // [rsp+28h] [rbp-128h]
  __int64 *v421; // [rsp+30h] [rbp-120h]
  __int64 v422; // [rsp+30h] [rbp-120h]
  unsigned __int64 v423; // [rsp+38h] [rbp-118h]
  __int64 v424; // [rsp+40h] [rbp-110h]
  int v425; // [rsp+40h] [rbp-110h]
  __int64 v426; // [rsp+48h] [rbp-108h]
  int v427; // [rsp+48h] [rbp-108h]
  __int64 *v428; // [rsp+48h] [rbp-108h]
  __int64 v429; // [rsp+48h] [rbp-108h]
  size_t v430; // [rsp+50h] [rbp-100h]
  __int64 v431; // [rsp+50h] [rbp-100h]
  __int64 v432; // [rsp+50h] [rbp-100h]
  __int64 v433; // [rsp+50h] [rbp-100h]
  __int64 v434; // [rsp+50h] [rbp-100h]
  size_t v435; // [rsp+50h] [rbp-100h]
  __int64 v436; // [rsp+58h] [rbp-F8h]
  _QWORD *v437; // [rsp+58h] [rbp-F8h]
  __int64 v438; // [rsp+58h] [rbp-F8h]
  __int64 v439; // [rsp+58h] [rbp-F8h]
  size_t v440; // [rsp+58h] [rbp-F8h]
  size_t v441; // [rsp+58h] [rbp-F8h]
  __int64 v442; // [rsp+60h] [rbp-F0h]
  int v443; // [rsp+68h] [rbp-E8h]
  __int64 *v444; // [rsp+68h] [rbp-E8h]
  __int64 v445; // [rsp+68h] [rbp-E8h]
  __int64 v446; // [rsp+70h] [rbp-E0h]
  __int64 *v447; // [rsp+70h] [rbp-E0h]
  size_t v448; // [rsp+70h] [rbp-E0h]
  int v449; // [rsp+70h] [rbp-E0h]
  size_t v450; // [rsp+70h] [rbp-E0h]
  size_t v451; // [rsp+70h] [rbp-E0h]
  size_t v452; // [rsp+70h] [rbp-E0h]
  unsigned __int64 v453; // [rsp+78h] [rbp-D8h]
  __int64 v454; // [rsp+78h] [rbp-D8h]
  _BOOL4 v455; // [rsp+78h] [rbp-D8h]
  _QWORD *v456; // [rsp+78h] [rbp-D8h]
  unsigned int v457; // [rsp+78h] [rbp-D8h]
  size_t v458; // [rsp+78h] [rbp-D8h]
  size_t v459; // [rsp+78h] [rbp-D8h]
  unsigned int v460; // [rsp+78h] [rbp-D8h]
  char *v461; // [rsp+78h] [rbp-D8h]
  __int64 v462; // [rsp+80h] [rbp-D0h]
  char v463; // [rsp+80h] [rbp-D0h]
  _BOOL4 v464; // [rsp+80h] [rbp-D0h]
  __int64 v465; // [rsp+80h] [rbp-D0h]
  char *v466; // [rsp+80h] [rbp-D0h]
  int v467; // [rsp+80h] [rbp-D0h]
  unsigned int v468; // [rsp+80h] [rbp-D0h]
  unsigned __int64 v469; // [rsp+80h] [rbp-D0h]
  __int64 v470; // [rsp+88h] [rbp-C8h]
  char *v471; // [rsp+88h] [rbp-C8h]
  int v472; // [rsp+88h] [rbp-C8h]
  unsigned int v473; // [rsp+88h] [rbp-C8h]
  __int64 v474; // [rsp+88h] [rbp-C8h]
  __int64 v475; // [rsp+88h] [rbp-C8h]
  __int64 v476; // [rsp+88h] [rbp-C8h]
  __int64 v477; // [rsp+88h] [rbp-C8h]
  __int64 v478; // [rsp+88h] [rbp-C8h]
  __m128i *v479; // [rsp+88h] [rbp-C8h]
  unsigned int v480; // [rsp+88h] [rbp-C8h]
  unsigned int v481; // [rsp+88h] [rbp-C8h]
  _QWORD *v482; // [rsp+90h] [rbp-C0h] BYREF
  unsigned __int64 *i11; // [rsp+98h] [rbp-B8h]
  unsigned int v484; // [rsp+A4h] [rbp-ACh] BYREF
  unsigned int v485; // [rsp+A8h] [rbp-A8h] BYREF
  int v486; // [rsp+ACh] [rbp-A4h] BYREF
  _QWORD v487[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v488; // [rsp+C0h] [rbp-90h]
  unsigned __int64 v489; // [rsp+C8h] [rbp-88h]
  _QWORD v490[4]; // [rsp+D0h] [rbp-80h] BYREF
  __m128i v491; // [rsp+F0h] [rbp-60h] BYREF
  __m128i v492; // [rsp+100h] [rbp-50h] BYREF
  __int64 v493; // [rsp+110h] [rbp-40h]

  v6 = a1;
  v7 = (FILE *)a2;
  v8 = *(unsigned __int8 *)(a2 + 40);
  v484 = 1;
  switch ( (char)v8 )
  {
    case 0:
      v108 = *(_BYTE **)(a2 + 48);
      v109 = *(_QWORD *)v108;
      for ( i = *(_BYTE *)(*(_QWORD *)v108 + 140LL); i == 12; i = *(_BYTE *)(v109 + 140) )
        v109 = *(_QWORD *)(v109 + 160);
      v111 = 32;
      if ( (v108[25] & 3) == 0 )
      {
        v111 = 16;
        if ( (unsigned __int8)(i - 2) > 1u )
          v111 = sub_7764B0(a1, v109, &v484);
      }
      v112 = *(_DWORD *)(a1 + 64);
      v113 = *(_QWORD *)(v6 + 56);
      v114 = _mm_loadu_si128((const __m128i *)(v6 + 32));
      v493 = *(_QWORD *)(v6 + 48);
      v115 = *(_DWORD *)(v6 + 128);
      v116 = _mm_loadu_si128((const __m128i *)(v6 + 16));
      v492 = v114;
      v117 = v115 + 1;
      *(_DWORD *)(v6 + 128) = v115 + 1;
      v118 = v112 & (v115 + 1);
      *(_DWORD *)(v6 + 40) = v117;
      v491 = v116;
      v119 = (int *)(v113 + 4LL * (v112 & v117));
      v120 = *v119;
      *v119 = v117;
      if ( v120 )
      {
        do
        {
          v118 = v112 & (v118 + 1);
          v183 = (_DWORD *)(v113 + 4LL * v118);
        }
        while ( *v183 );
        *v183 = v120;
      }
      v121 = *(_DWORD *)(v6 + 68) + 1;
      *(_DWORD *)(v6 + 68) = v121;
      if ( 2 * v121 > v112 )
      {
        LODWORD(i11) = v111;
        sub_7702C0(v6 + 56);
        v111 = (int)i11;
      }
      *(_QWORD *)(v6 + 48) = 0;
      if ( (unsigned __int8)(*(_BYTE *)(v109 + 140) - 8) > 3u )
      {
        v125 = 8;
        v124 = 16;
        v123 = 16;
      }
      else
      {
        v122 = (unsigned int)(v111 + 7) >> 3;
        v123 = v122 + 9;
        if ( (((_BYTE)v122 + 9) & 7) != 0 )
          v123 = v122 + 17 - (((_BYTE)v122 + 9) & 7);
        v124 = v123;
        v125 = v123 - 8LL;
      }
      v126 = v111 + v123;
      if ( (unsigned int)v126 > 0x400 )
      {
        i11 = (unsigned __int64 *)v125;
        v265 = v126 + 16;
        v266 = sub_822B10(v265);
        v267 = *(_QWORD *)(v6 + 32);
        v125 = (size_t)i11;
        *(_DWORD *)(v266 + 8) = v265;
        *(_QWORD *)v266 = v267;
        *(_DWORD *)(v266 + 12) = *(_DWORD *)(v6 + 40);
        v129 = (char *)(v266 + 16);
        *(_QWORD *)(v6 + 32) = v266;
      }
      else
      {
        v127 = v126 & 7;
        v128 = (unsigned int)(v126 + 8 - v127);
        v33 = v127 == 0;
        v129 = *(char **)(v6 + 16);
        if ( !v33 )
          v126 = v128;
        if ( 0x10000 - ((int)v129 - *(_DWORD *)(v6 + 24)) < (unsigned int)v126 )
        {
          i11 = (unsigned __int64 *)v125;
          sub_772E70((_QWORD *)(v6 + 16));
          v129 = *(char **)(v6 + 16);
          v125 = (size_t)i11;
        }
        *(_QWORD *)(v6 + 16) = &v129[v126];
      }
      v130 = (char *)memset(v129, 0, v125) + v124;
      *((_QWORD *)v130 - 1) = v109;
      if ( (unsigned __int8)(*(_BYTE *)(v109 + 140) - 9) <= 2u )
        *(_QWORD *)v130 = 0;
      if ( v484 )
      {
        if ( (unsigned int)sub_786210(v6, (_QWORD **)v108, (unsigned __int64)v130, v130) )
        {
          if ( ((v108[25] & 3) != 0 || *(_BYTE *)(v109 + 140) == 6) && (v130[8] & 4) != 0 )
          {
            v337 = *((_QWORD *)v130 + 2);
            v338 = 2;
            v339 = *(_QWORD **)v337;
            for ( j = **(_QWORD ***)v337; j; ++v338 )
            {
              v339 = j;
              j = (_QWORD *)*j;
            }
            *v339 = qword_4F08088;
            v130[8] &= ~4u;
            v341 = *(_QWORD *)(v337 + 24);
            qword_4F08080 += v338;
            qword_4F08088 = v337;
            *((_QWORD *)v130 + 2) = v341;
          }
          if ( *(_QWORD *)(v6 + 48) && v484 )
            v484 = sub_799890(v6);
        }
        else
        {
          v484 = 0;
        }
      }
      v131 = *(_DWORD *)(v6 + 40);
      v132 = *(_DWORD *)(v6 + 64);
      v133 = *(_QWORD *)(v6 + 56);
      v134 = *(__int64 **)(v6 + 32);
      v135 = v132 & v131;
      for ( k = (_DWORD *)(v133 + 4LL * (v132 & v131)); v131 != *k; k = (_DWORD *)(v133 + 4LL * v135) )
        v135 = v132 & (v135 + 1);
      *k = 0;
      if ( *(_DWORD *)(v133 + 4LL * ((v135 + 1) & v132)) )
        sub_771390(*(_QWORD *)(v6 + 56), *(_DWORD *)(v6 + 64), v135);
      v137 = _mm_loadu_si128(&v491);
      v138 = _mm_loadu_si128(&v492);
      v139 = v493;
      --*(_DWORD *)(v6 + 68);
      *(__m128i *)(v6 + 16) = v137;
      *(_QWORD *)(v6 + 48) = v139;
      *(__m128i *)(v6 + 32) = v138;
      if ( !v134 || (__int64 *)v492.m128i_i64[0] == v134 )
        return v484;
      while ( 1 )
      {
        v140 = *((unsigned int *)v134 + 3);
        v141 = *(_QWORD *)(v6 + 56);
        v142 = v140 & *(_DWORD *)(v6 + 64);
        v143 = *(unsigned int *)(v141 + 4LL * v142);
        if ( !(_DWORD)v140 || (_DWORD)v143 == (_DWORD)v140 )
          goto LABEL_316;
        while ( (_DWORD)v143 )
        {
          v142 = *(_DWORD *)(v6 + 64) & (v142 + 1);
          v143 = *(unsigned int *)(v141 + 4LL * v142);
          if ( (_DWORD)v140 == (_DWORD)v143 )
            goto LABEL_316;
        }
        v144 = (__int64 *)*v134;
        sub_822B90(v134, *((unsigned int *)v134 + 2), v143, v140);
        if ( !v144 )
          break;
        v134 = v144;
      }
      v134 = 0;
LABEL_316:
      *(_QWORD *)(v6 + 32) = v134;
      return v484;
    case 1:
    case 2:
    case 3:
    case 4:
      v9 = *(_QWORD **)(a2 + 72);
      if ( (_BYTE)v8 == 2 )
      {
        v186 = v9[1];
        v9 = (_QWORD *)*v9;
        v470 = v186;
      }
      else
      {
        v470 = *(_QWORD *)(a2 + 80);
      }
      v10 = *(_QWORD *)(a2 + 48);
      if ( !v10 )
      {
        if ( (*(_BYTE *)(a1 + 132) & 1) != 0 )
        {
          if ( (_BYTE)v8 == 3 )
          {
            v346 = *(_BYTE *)(a1 + 133);
            *(_BYTE *)(a1 + 133) = v346 | 0x10;
            v269 = (v346 & 0x10) != 0;
            result = sub_795660(a1, v9);
          }
          else if ( v470 )
          {
            v269 = 0;
            result = sub_795660(a1, v470);
          }
          else
          {
            result = 1;
            v269 = 0;
          }
        }
        else
        {
          result = 0;
          v269 = 0;
        }
        *(_BYTE *)(a1 + 133) = *(_BYTE *)(a1 + 133) & 0xEF | (16 * v269);
        return result;
      }
      LOBYTE(i11) = *(_BYTE *)(v10 + 24);
      LODWORD(v482) = (_BYTE)i11 == 9;
      if ( (_BYTE)i11 == 9 && !(unsigned int)sub_77A4E0((const __m128i *)a1, *(_QWORD *)(v10 + 56), &v491) )
        return 0;
      v11 = *(_QWORD *)v10;
      for ( m = *(_BYTE *)(*(_QWORD *)v10 + 140LL); m == 12; m = *(_BYTE *)(v11 + 140) )
        v11 = *(_QWORD *)(v11 + 160);
      if ( (unsigned __int8)(m - 2) > 1u )
      {
        v329 = sub_7764B0(a1, v11, &v484);
        if ( (unsigned __int8)(*(_BYTE *)(v11 + 140) - 8) > 3u )
        {
          v15 = (unsigned int)(v329 + 16);
          v14 = 8;
          v13 = 16;
        }
        else
        {
          v330 = (unsigned int)(v329 + 7) >> 3;
          v331 = v330 + 9;
          if ( (((_BYTE)v330 + 9) & 7) != 0 )
            v331 = v330 + 17 - (((_BYTE)v330 + 9) & 7);
          v13 = v331;
          v15 = v329 + v331;
          v14 = v13 - 8;
        }
        if ( (unsigned int)v15 > 0x400 )
        {
          v458 = v14;
          v467 = v15 + 16;
          v332 = sub_822B10((unsigned int)(v15 + 16));
          v14 = v458;
          v16 = (char *)(v332 + 16);
          *(_QWORD *)v332 = *(_QWORD *)(a1 + 32);
          *(_DWORD *)(v332 + 8) = v467;
          *(_DWORD *)(v332 + 12) = *(_DWORD *)(a1 + 40);
          *(_QWORD *)(a1 + 32) = v332;
          goto LABEL_13;
        }
        if ( (v15 & 7) != 0 )
          v15 = (_DWORD)v15 + 8 - (unsigned int)(v15 & 7);
      }
      else
      {
        v13 = 16;
        v14 = 8;
        v15 = 32;
      }
      v16 = *(char **)(a1 + 16);
      if ( 0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24)) < (unsigned int)v15 )
      {
        v459 = v14;
        v468 = v15;
        sub_772E70((_QWORD *)(a1 + 16));
        v16 = *(char **)(a1 + 16);
        v14 = v459;
        v15 = v468;
      }
      *(_QWORD *)(a1 + 16) = &v16[v15];
LABEL_13:
      v17 = (char *)memset(v16, 0, v14) + v13;
      *((_QWORD *)v17 - 1) = v11;
      if ( (unsigned __int8)(*(_BYTE *)(v11 + 140) - 9) <= 2u )
        *(_QWORD *)v17 = 0;
      if ( !(unsigned int)sub_7A0A10((unsigned int)v482, a1, v10, v11, v17) )
      {
        v484 = 0;
        goto LABEL_17;
      }
      v268 = sub_620EE0(v17, byte_4B6DF90[*(unsigned __int8 *)(v11 + 160)], v490);
      if ( LODWORD(v490[0]) )
      {
        result = v484;
        if ( !v484 )
        {
          if ( (_BYTE)i11 != 9 )
            return result;
          goto LABEL_347;
        }
      }
      else
      {
        if ( !v484 )
          goto LABEL_17;
        if ( !v268 )
        {
          if ( v470 )
            v484 = sub_795660(a1, v470);
LABEL_17:
          if ( (_BYTE)i11 != 9 )
            return v484;
LABEL_347:
          sub_7762B0(a1, *(_QWORD *)(v10 + 56));
          sub_7999E0(a1, *(_QWORD *)(v10 + 56), &v491, &v484);
          return v484;
        }
      }
      v484 = sub_795660(a1, v9);
      goto LABEL_17;
    case 5:
      v475 = *(_QWORD *)(a2 + 48);
      LOBYTE(v482) = *(_BYTE *)(v475 + 24);
      v464 = (_BYTE)v482 == 9;
      if ( (_BYTE)v482 == 9 && !(unsigned int)sub_77A4E0((const __m128i *)a1, *(_QWORD *)(v475 + 56), &v491) )
        return 0;
      v171 = *(_QWORD *)v475;
      for ( n = *(_BYTE *)(*(_QWORD *)v475 + 140LL); n == 12; n = *(_BYTE *)(v171 + 140) )
        v171 = *(_QWORD *)(v171 + 160);
      if ( (unsigned __int8)(n - 2) > 1u )
      {
        v199 = sub_7764B0(a1, v171, &v484);
        if ( (unsigned __int8)(*(_BYTE *)(v171 + 140) - 8) > 3u )
        {
          v175 = (unsigned int)(v199 + 16);
          v174 = 8;
          v173 = 16;
        }
        else
        {
          v200 = (unsigned int)(v199 + 7) >> 3;
          v201 = v200 + 9;
          if ( (((_BYTE)v200 + 9) & 7) != 0 )
            v201 = v200 + 17 - (((_BYTE)v200 + 9) & 7);
          v173 = v201;
          v175 = v199 + v201;
          v174 = v173 - 8;
        }
        if ( (unsigned int)v175 > 0x400 )
        {
          LODWORD(i11) = v175 + 16;
          v202 = sub_822B10((unsigned int)(v175 + 16));
          v203 = (int)i11;
          v176 = (char *)(v202 + 16);
          *(_QWORD *)v202 = *(_QWORD *)(v6 + 32);
          *(_DWORD *)(v202 + 8) = v203;
          *(_DWORD *)(v202 + 12) = *(_DWORD *)(v6 + 40);
          *(_QWORD *)(v6 + 32) = v202;
          goto LABEL_188;
        }
        if ( (v175 & 7) != 0 )
          v175 = (_DWORD)v175 + 8 - (unsigned int)(v175 & 7);
      }
      else
      {
        v173 = 16;
        v174 = 8;
        v175 = 32;
      }
      v176 = *(char **)(a1 + 16);
      if ( 0x10000 - (*(_DWORD *)(v6 + 16) - *(_DWORD *)(v6 + 24)) < (unsigned int)v175 )
      {
        LODWORD(i11) = v175;
        sub_772E70((_QWORD *)(v6 + 16));
        v176 = *(char **)(v6 + 16);
        v175 = (unsigned int)i11;
      }
      *(_QWORD *)(v6 + 16) = &v176[v175];
LABEL_188:
      v177 = (char *)memset(v176, 0, v174) + v173;
      *((_QWORD *)v177 - 1) = v171;
      if ( (unsigned __int8)(*(_BYTE *)(v171 + 140) - 9) <= 2u )
        *(_QWORD *)v177 = 0;
      v178 = 0;
      i11 = &qword_4D042E0;
      while ( 1 )
      {
        v179 = i11;
        v180 = *(_QWORD *)(v6 + 120) + 1LL;
        *(_QWORD *)(v6 + 120) = v180;
        if ( v180 <= *v179 )
        {
          v222 = sub_7A0A10(v464, v6, v475, v171, v177);
          ++*(_QWORD *)(v6 + 120);
          v484 = v222;
          if ( !v222
            || (v223 = sub_620EE0(v177, byte_4B6DF90[*(unsigned __int8 *)(v171 + 160)], v490),
                v178 = v223,
                LODWORD(v490[0])) )
          {
            v181 = v178 == 0;
          }
          else
          {
            if ( v223 )
            {
              v325 = sub_795660(v6, v7->_IO_save_base);
              v181 = 0;
              v484 = v325;
              if ( !v325 )
                goto LABEL_193;
              v326 = *(_QWORD *)(v6 + 72);
              v327 = *(_BYTE *)(v326 + 48);
              if ( (v327 & 1) == 0 )
              {
                if ( (v327 & 2) == 0 )
                {
                  if ( (v327 & 4) != 0 )
                    *(_BYTE *)(v326 + 48) = v327 & 0xFB;
                  goto LABEL_193;
                }
                *(_BYTE *)(v326 + 48) = v327 & 0xFD;
              }
            }
            v181 = 1;
            v178 = 0;
          }
        }
        else
        {
          sub_6855B0(0x97Fu, (FILE *)(v6 + 112), (_QWORD *)(v6 + 96));
          v484 = 0;
          v181 = v178 == 0;
        }
LABEL_193:
        if ( (_BYTE)v482 == 9 )
          sub_7762B0(v6, *(_QWORD *)(v475 + 56));
        result = v484;
        if ( !v484 || v181 )
        {
          if ( (_BYTE)v482 == 9 )
          {
            sub_7999E0(v6, *(_QWORD *)(v475 + 56), &v491, &v484);
            return v484;
          }
          return result;
        }
      }
    case 6:
      v182 = *(_BYTE *)(*(_QWORD *)(a2 + 72) + 120LL);
      if ( (v182 & 4) != 0 )
      {
        *(_BYTE *)(*(_QWORD *)(a1 + 72) + 48LL) |= 8u;
        return 1;
      }
      if ( (v182 & 2) != 0 )
      {
        *(_BYTE *)(*(_QWORD *)(a1 + 72) + 48LL) |= 2u;
        return 1;
      }
      if ( (v182 & 8) != 0 )
      {
        *(_BYTE *)(*(_QWORD *)(a1 + 72) + 48LL) |= 4u;
        return 1;
      }
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        return 0;
      sub_6855B0(0xA87u, (FILE *)a2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    case 7:
    case 15:
    case 24:
      return 1;
    case 8:
      v145 = *(__int64 **)(a1 + 72);
      while ( 1 )
      {
        v146 = v145[1];
        if ( v146 )
          break;
        v145 = (__int64 *)*v145;
        if ( !v145 )
        {
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            sub_6855B0(0xB8Bu, (FILE *)a2, (_QWORD *)(a1 + 96));
            sub_770D30(a1);
            return 0;
          }
          return 0;
        }
      }
      v147 = v145[3];
      v148 = (char *)v145[4];
      if ( *(_QWORD *)(a2 + 48) )
      {
        v149 = *(_DWORD *)(a1 + 64);
        v150 = *(_QWORD *)(a1 + 56);
        v151 = *(_QWORD *)(a1 + 24);
        i11 = *(unsigned __int64 **)(a1 + 16);
        v152 = *(__int64 **)(a1 + 32);
        LODWORD(v482) = *(_DWORD *)(a1 + 40);
        v474 = *(_QWORD *)(a1 + 48);
        v153 = *(_DWORD *)(a1 + 128) + 1;
        *(_DWORD *)(a1 + 128) = v153;
        v154 = v149 & v153;
        *(_DWORD *)(a1 + 40) = v153;
        v155 = (int *)(v150 + 4LL * (v149 & v153));
        v156 = *v155;
        *v155 = v153;
        if ( v156 )
        {
          do
          {
            v154 = v149 & (v154 + 1);
            v185 = (int *)(v150 + 4LL * v154);
          }
          while ( *v185 );
          *v185 = v156;
        }
        v157 = *(_DWORD *)(v6 + 68) + 1;
        *(_DWORD *)(v6 + 68) = v157;
        if ( 2 * v157 > v149 )
        {
          v461 = v148;
          v469 = v147;
          sub_7702C0(v6 + 56);
          v148 = v461;
          v147 = v469;
        }
        *(_QWORD *)(v6 + 48) = 0;
        v158 = sub_786210(v6, (_QWORD **)v7->_IO_write_end, v147, v148);
        v33 = *(_QWORD *)(v6 + 48) == 0;
        v484 = v158;
        if ( !v33 && v158 )
          v484 = sub_799890(v6);
        v159 = *(_DWORD *)(v6 + 40);
        v160 = *(_DWORD *)(v6 + 64);
        v161 = *(_QWORD *)(v6 + 56);
        v162 = *(__int64 **)(v6 + 32);
        v163 = v160 & v159;
        for ( ii = (_DWORD *)(v161 + 4LL * (v160 & v159)); v159 != *ii; ii = (_DWORD *)(v161 + 4LL * v163) )
          v163 = v160 & (v163 + 1);
        *ii = 0;
        if ( *(_DWORD *)(v161 + 4LL * ((v163 + 1) & v160)) )
          sub_771390(*(_QWORD *)(v6 + 56), *(_DWORD *)(v6 + 64), v163);
        v165 = i11;
        --*(_DWORD *)(v6 + 68);
        *(_QWORD *)(v6 + 24) = v151;
        *(_QWORD *)(v6 + 16) = v165;
        LODWORD(v165) = (_DWORD)v482;
        *(_QWORD *)(v6 + 32) = v152;
        *(_DWORD *)(v6 + 40) = (_DWORD)v165;
        *(_QWORD *)(v6 + 48) = v474;
        if ( v162 && v162 != v152 )
        {
          while ( 1 )
          {
            v166 = *((unsigned int *)v162 + 3);
            v167 = *(_QWORD *)(v6 + 56);
            v168 = v166 & *(_DWORD *)(v6 + 64);
            v169 = *(unsigned int *)(v167 + 4LL * v168);
            if ( !(_DWORD)v166 || (_DWORD)v169 == (_DWORD)v166 )
              goto LABEL_477;
            while ( (_DWORD)v169 )
            {
              v168 = *(_DWORD *)(v6 + 64) & (v168 + 1);
              v169 = *(unsigned int *)(v167 + 4LL * v168);
              if ( (_DWORD)v166 == (_DWORD)v169 )
                goto LABEL_477;
            }
            v170 = (__int64 *)*v162;
            sub_822B90(v162, *((unsigned int *)v162 + 2), v169, v166);
            if ( !v170 )
              break;
            v162 = v170;
          }
          v162 = 0;
LABEL_477:
          *(_QWORD *)(v6 + 32) = v162;
          result = v484;
          goto LABEL_450;
        }
      }
      else
      {
        v334 = *(_QWORD *)(a2 + 72);
        if ( v334 )
        {
          if ( *(_BYTE *)(v334 + 48) != 1 )
          {
            v335 = *(_DWORD *)(a1 + 40);
            v491.m128i_i64[0] = v145[3];
            v491.m128i_i32[2] = 0;
            v491.m128i_i32[3] = v335;
            v492.m128i_i64[1] = (__int64)v148;
            if ( *v145 )
            {
              v336 = *(_DWORD *)(*v145 + 44);
              if ( v336 )
                v491.m128i_i32[3] = v336 - 1;
            }
            else
            {
              v491.m128i_i32[3] = 1;
            }
            result = sub_79B7D0(a1, v334, v7, &v491, 0, 0);
            goto LABEL_450;
          }
          for ( jj = *(_QWORD *)(v146 + 152); *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
            ;
          for ( kk = *(_QWORD *)(jj + 160); *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
            ;
          sub_7790A0(a1, (__m128i *)v145[3], kk, v145[4]);
        }
        else
        {
          for ( mm = *(_QWORD *)(v146 + 152); *(_BYTE *)(mm + 140) == 12; mm = *(_QWORD *)(mm + 160) )
            ;
          result = sub_8D2600(*(_QWORD *)(mm + 160));
          if ( !(_DWORD)result )
          {
            v405 = v7;
            if ( !v7->_flags )
            {
              IO_read_base = v7->_IO_read_base;
              while ( IO_read_base[40] != 11 || !**((_DWORD **)IO_read_base + 10) )
              {
                IO_read_base = (char *)*((_QWORD *)IO_read_base + 3);
                if ( !IO_read_base )
                  goto LABEL_584;
              }
              v405 = (FILE *)*((_QWORD *)IO_read_base + 10);
            }
LABEL_584:
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              LODWORD(i11) = 0;
              sub_6855B0(0xA88u, v405, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
              result = 0;
            }
LABEL_450:
            *((_BYTE *)v145 + 48) |= 1u;
            return result;
          }
        }
      }
      result = v484;
      goto LABEL_450;
    case 11:
      return sub_7987E0(a1, *(_QWORD *)(a2 + 72), *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8LL));
    case 12:
      v47 = **(_QWORD **)(a2 + 48);
      i11 = *(unsigned __int64 **)(a2 + 48);
      for ( nn = *(_BYTE *)(v47 + 140); nn == 12; nn = *(_BYTE *)(v47 + 140) )
        v47 = *(_QWORD *)(v47 + 160);
      if ( (unsigned __int8)(nn - 2) > 1u )
      {
        v187 = sub_7764B0(a1, v47, &v484);
        if ( (unsigned __int8)(*(_BYTE *)(v47 + 140) - 8) > 3u )
        {
          v51 = (unsigned int)(v187 + 16);
          v50 = 8;
          v49 = 16;
        }
        else
        {
          v188 = (unsigned int)(v187 + 7) >> 3;
          v189 = v188 + 9;
          if ( (((_BYTE)v188 + 9) & 7) != 0 )
            v189 = v188 + 17 - (((_BYTE)v188 + 9) & 7);
          v49 = v189;
          v51 = v189 + v187;
          v50 = v189 - 8LL;
        }
        if ( (unsigned int)v51 > 0x400 )
        {
          v482 = (_QWORD *)v50;
          v190 = v51 + 16;
          v191 = sub_822B10(v190);
          v192 = *(_QWORD *)(a1 + 32);
          v50 = (size_t)v482;
          *(_DWORD *)(v191 + 8) = v190;
          v52 = (char *)(v191 + 16);
          *(_QWORD *)v191 = v192;
          *(_DWORD *)(v191 + 12) = *(_DWORD *)(a1 + 40);
          *(_QWORD *)(a1 + 32) = v191;
          goto LABEL_53;
        }
        if ( (v51 & 7) != 0 )
          v51 = (_DWORD)v51 + 8 - (unsigned int)(v51 & 7);
      }
      else
      {
        v49 = 16;
        v50 = 8;
        v51 = 32;
      }
      v52 = *(char **)(a1 + 16);
      if ( 0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24)) < (unsigned int)v51 )
      {
        v482 = (_QWORD *)v50;
        sub_772E70((_QWORD *)(a1 + 16));
        v52 = *(char **)(a1 + 16);
        v50 = (size_t)v482;
      }
      *(_QWORD *)(a1 + 16) = &v52[v51];
LABEL_53:
      v53 = (char *)memset(v52, 0, v50) + v49;
      *((_QWORD *)v53 - 1) = v47;
      if ( (unsigned __int8)(*(_BYTE *)(v47 + 140) - 9) <= 2u )
        *(_QWORD *)v53 = 0;
      v453 = v47;
      v55 = i11;
      v446 = a1 + 56;
      while ( 2 )
      {
        result = sub_795660(a1, v7->_IO_save_base);
        v484 = result;
        if ( !(_DWORD)result )
          return result;
        v56 = *(_QWORD *)(a1 + 72);
        v57 = *(_BYTE *)(v56 + 48);
        if ( (v57 & 1) != 0 )
          return result;
        if ( (v57 & 2) != 0 )
        {
          *(_BYTE *)(v56 + 48) = v57 & 0xFD;
          return result;
        }
        if ( (v57 & 4) != 0 )
          *(_BYTE *)(v56 + 48) = v57 & 0xFB;
        v58 = *(_QWORD *)(a1 + 120) + 1LL;
        *(_QWORD *)(a1 + 120) = v58;
        if ( v58 > qword_4D042E0 )
        {
          sub_6855B0(0x97Fu, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
          return 0;
        }
        v224 = *(_DWORD *)(a1 + 64);
        v225 = *(_QWORD *)(a1 + 56);
        v226 = *(__int64 **)(a1 + 32);
        v476 = *(_QWORD *)(a1 + 16);
        v465 = *(_QWORD *)(a1 + 24);
        LODWORD(v482) = *(_DWORD *)(a1 + 40);
        i11 = *(unsigned __int64 **)(a1 + 48);
        v227 = *(_DWORD *)(a1 + 128) + 1;
        *(_DWORD *)(a1 + 128) = v227;
        v228 = v224 & v227;
        *(_DWORD *)(a1 + 40) = v227;
        v229 = (int *)(v225 + 4LL * (v224 & v227));
        v230 = *v229;
        *v229 = v227;
        if ( v230 )
        {
          do
          {
            v228 = v224 & (v228 + 1);
            v244 = (int *)(v225 + 4LL * v228);
          }
          while ( *v244 );
          *v244 = v230;
        }
        v231 = *(_DWORD *)(a1 + 68) + 1;
        *(_DWORD *)(a1 + 68) = v231;
        if ( 2 * v231 > v224 )
          sub_7702C0(v446);
        *(_QWORD *)(a1 + 48) = 0;
        v232 = sub_786210(a1, (_QWORD **)v55, (unsigned __int64)v53, v53);
        v33 = *(_QWORD *)(a1 + 48) == 0;
        v484 = v232;
        if ( !v33 && v232 )
          v484 = sub_799890(a1);
        v233 = *(_DWORD *)(a1 + 40);
        v234 = *(_DWORD *)(a1 + 64);
        v235 = *(_QWORD *)(a1 + 56);
        v236 = *(__int64 **)(a1 + 32);
        v237 = v234 & v233;
        for ( i1 = (_DWORD *)(v235 + 4LL * (v234 & v233)); v233 != *i1; i1 = (_DWORD *)(v235 + 4LL * v237) )
          v237 = v234 & (v237 + 1);
        *i1 = 0;
        if ( *(_DWORD *)(v235 + 4LL * ((v237 + 1) & v234)) )
        {
          v444 = v236;
          sub_771390(*(_QWORD *)(a1 + 56), *(_DWORD *)(a1 + 64), v237);
          v236 = v444;
        }
        --*(_DWORD *)(a1 + 68);
        *(_QWORD *)(a1 + 32) = v226;
        *(_QWORD *)(a1 + 16) = v476;
        *(_QWORD *)(a1 + 24) = v465;
        *(_DWORD *)(a1 + 40) = (_DWORD)v482;
        *(_QWORD *)(a1 + 48) = i11;
        if ( !v236 || v236 == v226 )
          goto LABEL_296;
        while ( 1 )
        {
          v239 = *((unsigned int *)v236 + 3);
          v240 = *(_QWORD *)(a1 + 56);
          v241 = v239 & *(_DWORD *)(a1 + 64);
          v242 = *(unsigned int *)(v240 + 4LL * v241);
          if ( (_DWORD)v242 == (_DWORD)v239 || !(_DWORD)v239 )
            goto LABEL_295;
          while ( (_DWORD)v242 )
          {
            v241 = *(_DWORD *)(a1 + 64) & (v241 + 1);
            v242 = *(unsigned int *)(v240 + 4LL * v241);
            if ( (_DWORD)v239 == (_DWORD)v242 )
              goto LABEL_295;
          }
          v243 = (__int64 *)*v236;
          sub_822B90(v236, *((unsigned int *)v236 + 2), v242, v239);
          if ( !v243 )
            break;
          v236 = v243;
        }
        v236 = 0;
LABEL_295:
        *(_QWORD *)(a1 + 32) = v236;
LABEL_296:
        if ( ((*((_BYTE *)v55 + 25) & 3) != 0 || *(_BYTE *)(v453 + 140) == 6) && (v53[8] & 4) != 0 )
        {
          v247 = *((_QWORD *)v53 + 2);
          v248 = 2;
          v249 = *(_QWORD **)v247;
          for ( i2 = **(_QWORD ***)v247; i2; ++v248 )
          {
            v249 = i2;
            i2 = (_QWORD *)*i2;
          }
          *v249 = qword_4F08088;
          v53[8] &= ~4u;
          v251 = *(_QWORD *)(v247 + 24);
          qword_4F08080 += v248;
          qword_4F08088 = v247;
          *((_QWORD *)v53 + 2) = v251;
        }
        v245 = v484;
        ++*(_QWORD *)(a1 + 120);
        if ( v245 )
        {
          v246 = sub_620EE0(v53, byte_4B6DF90[*(unsigned __int8 *)(v453 + 160)], v490);
          result = v484;
          if ( !v246 || !v484 )
            return result;
          continue;
        }
        return 0;
      }
    case 13:
      return sub_7A0E60(a1, a2, 0, v8, a5, a6);
    case 14:
      v59 = *(_QWORD *)(a1 + 16);
      v60 = *(_DWORD *)(a1 + 64);
      v485 = 1;
      v61 = *(_QWORD *)(a1 + 56);
      IO_backup_base = v7->_IO_backup_base;
      v462 = v59;
      v454 = *(_QWORD *)(a1 + 24);
      v447 = *(__int64 **)(a1 + 32);
      v443 = *(_DWORD *)(a1 + 40);
      v442 = *(_QWORD *)(a1 + 48);
      v63 = *(_DWORD *)(a1 + 128) + 1;
      *(_DWORD *)(a1 + 128) = v63;
      v64 = v60 & v63;
      *(_DWORD *)(a1 + 40) = v63;
      v65 = (int *)(v61 + 4LL * (v60 & v63));
      v66 = *v65;
      *v65 = v63;
      if ( v66 )
      {
        do
        {
          v64 = v60 & (v64 + 1);
          v77 = (int *)(v61 + 4LL * v64);
        }
        while ( *v77 );
        *v77 = v66;
      }
      v67 = *(_DWORD *)(v6 + 68) + 1;
      *(_DWORD *)(v6 + 68) = v67;
      if ( v60 < 2 * v67 )
        sub_7702C0(v6 + 56);
      *(_QWORD *)(v6 + 48) = 0;
      v487[0] = *((_QWORD *)IO_backup_base + 1);
      v68 = *((_QWORD *)IO_backup_base + 2);
      i11 = (unsigned __int64 *)v487[0];
      v487[1] = v68;
      v488 = *((_QWORD *)IO_backup_base + 5);
      v69 = *((_QWORD *)IO_backup_base + 6);
      v489 = v69;
      if ( v488 == 0 || v68 == 0 || v487[0] == 0 || !v69 )
      {
        v485 = 0;
        goto LABEL_236;
      }
      v436 = v68;
      v70 = (unsigned __int64)i11;
      v482 = v487;
      v471 = IO_backup_base;
      v71 = 0;
      v72 = v7;
      v73 = v6;
      v74 = v72;
      while ( 1 )
      {
        v75 = sub_77A250(v73, v70, &v485);
        v76 = v485;
        v490[v71] = v75;
        if ( v76 )
          *((_BYTE *)v75 - 9) |= 1u;
        if ( ++v71 == 4 )
          break;
        v70 = v482[v71];
      }
      v252 = v74;
      v6 = v73;
      v253 = v252;
      if ( !v485 )
        goto LABEL_329;
      v254 = v436;
      v255 = 1;
      while ( 2 )
      {
        if ( !(unsigned int)sub_7A0470(v6, v254, v490[v255], v253) )
          goto LABEL_386;
        if ( ++v255 != 4 )
        {
          v254 = v482[v255];
          continue;
        }
        break;
      }
      if ( !v485 )
        goto LABEL_329;
      v270 = (unsigned __int64 *)*((_QWORD *)v471 + 8);
      v271 = **((_QWORD **)v471 + 7);
      v477 = *((_QWORD *)v471 + 7);
      v420 = v270;
      for ( i3 = *(_BYTE *)(v271 + 140); i3 == 12; i3 = *(_BYTE *)(v271 + 140) )
        v271 = *(_QWORD *)(v271 + 160);
      v273 = *v270;
      v423 = *v270;
      if ( *(_BYTE *)(*v270 + 140) == 12 )
      {
        do
          v273 = *(_QWORD *)(v273 + 160);
        while ( *(_BYTE *)(v273 + 140) == 12 );
        v423 = v273;
      }
      if ( (*(_BYTE *)(v477 + 25) & 3) != 0 )
      {
        v274 = 32;
LABEL_363:
        if ( (unsigned __int8)(*(_BYTE *)(v271 + 140) - 8) <= 3u )
        {
          v276 = ((unsigned int)(v274 + 7) >> 3) + 17 - (((unsigned __int8)((unsigned int)(v274 + 7) >> 3) + 9) & 7);
          v278 = v274 + v276;
          v275 = v276 - 8;
          goto LABEL_366;
        }
        goto LABEL_364;
      }
      v274 = 16;
      if ( (unsigned __int8)(i3 - 2) <= 1u )
        goto LABEL_363;
      v274 = sub_7764B0(v6, v271, &v485);
      if ( (unsigned __int8)(*(_BYTE *)(v271 + 140) - 8) > 3u )
      {
LABEL_364:
        v275 = 8;
        v276 = 16;
        v277 = 16;
        goto LABEL_365;
      }
      v408 = (unsigned int)(v274 + 7) >> 3;
      v277 = v408 + 9;
      if ( (((_BYTE)v408 + 9) & 7) != 0 )
        v277 = v408 + 17 - (((_BYTE)v408 + 9) & 7);
      v276 = v277;
      v275 = v277 - 8LL;
LABEL_365:
      v278 = v274 + v277;
      if ( v278 > 0x400 )
      {
        v433 = v276;
        v440 = v275;
        v400 = v278 + 16;
        v401 = sub_822B10(v400);
        v275 = v440;
        v276 = v433;
        v402 = v401;
        v403 = *(_QWORD *)(v6 + 32);
        *(_DWORD *)(v402 + 8) = v400;
        *(_QWORD *)v402 = v403;
        *(_DWORD *)(v402 + 12) = *(_DWORD *)(v6 + 40);
        *(_QWORD *)(v6 + 32) = v402;
        v279 = (char *)(v402 + 16);
        goto LABEL_371;
      }
LABEL_366:
      v279 = *(char **)(v6 + 16);
      if ( (v278 & 7) != 0 )
        v278 = v278 + 8 - (v278 & 7);
      if ( 0x10000 - (*(_DWORD *)(v6 + 16) - *(_DWORD *)(v6 + 24)) < v278 )
      {
        v434 = v276;
        v441 = v275;
        sub_772E70((_QWORD *)(v6 + 16));
        v279 = *(char **)(v6 + 16);
        v276 = v434;
        v275 = v441;
      }
      *(_QWORD *)(v6 + 16) = &v279[v278];
LABEL_371:
      v280 = (char *)memset(v279, 0, v275) + v276;
      *(v280 - 1) = v271;
      if ( (unsigned __int8)(*(_BYTE *)(v271 + 140) - 9) <= 2u )
        *v280 = 0;
      v281 = *(_BYTE *)(v423 + 140);
      if ( (*((_BYTE *)v420 + 25) & 3) != 0 )
      {
        v282 = 32;
      }
      else
      {
        v282 = 16;
        if ( (unsigned __int8)(v281 - 2) > 1u )
        {
          v282 = sub_7764B0(v6, v423, &v485);
          if ( (unsigned __int8)(*(_BYTE *)(v423 + 140) - 8) <= 3u )
          {
            v409 = (unsigned int)(v282 + 7) >> 3;
            v285 = v409 + 9;
            if ( (((_BYTE)v409 + 9) & 7) != 0 )
            {
              v415 = v409 + 17 - (((_BYTE)v409 + 9) & 7);
              v284 = v415;
              v285 = v415;
              v283 = v415 - 8LL;
            }
            else
            {
              v284 = v285;
              v283 = v285 - 8LL;
            }
LABEL_377:
            v286 = v282 + v285;
            if ( v286 > 0x400 )
            {
              v426 = v284;
              v430 = v283;
              v343 = sub_822B10(v286 + 16);
              v283 = v430;
              v344 = v343;
              v345 = *(_QWORD *)(v6 + 32);
              v284 = v426;
              *(_DWORD *)(v344 + 8) = v286 + 16;
              *(_QWORD *)v344 = v345;
              *(_DWORD *)(v344 + 12) = *(_DWORD *)(v6 + 40);
              *(_QWORD *)(v6 + 32) = v344;
              v287 = (char *)(v344 + 16);
              goto LABEL_383;
            }
LABEL_378:
            v287 = *(char **)(v6 + 16);
            if ( (v286 & 7) != 0 )
              v286 = v286 + 8 - (v286 & 7);
            if ( 0x10000 - (*(_DWORD *)(v6 + 16) - *(_DWORD *)(v6 + 24)) < v286 )
            {
              v429 = v284;
              v435 = v283;
              sub_772E70((_QWORD *)(v6 + 16));
              v287 = *(char **)(v6 + 16);
              v284 = v429;
              v283 = v435;
            }
            *(_QWORD *)(v6 + 16) = &v287[v286];
LABEL_383:
            v288 = (char *)memset(v287, 0, v283) + v284;
            *(v288 - 1) = v423;
            v418 = (unsigned __int64)v288;
            if ( (unsigned __int8)(*(_BYTE *)(v423 + 140) - 9) <= 2u )
              *v288 = 0;
            if ( *((_BYTE *)i11 + 177) == 2 && (v416 = i11[23]) != 0 )
            {
              if ( v485 )
              {
                v417 = v253;
                v256 = v6;
                i11 = &qword_4D042E0;
                v257 = v6 + 56;
                v258 = v69;
                v259 = v271;
                v419 = v257;
                v260 = (unsigned __int64)v280;
                while ( 1 )
                {
                  v261 = i11;
                  v262 = *(_QWORD *)(v256 + 120) + 1LL;
                  *(_QWORD *)(v256 + 120) = v262;
                  if ( v262 > *v261 )
                    break;
                  v347 = *(_DWORD *)(v256 + 64);
                  v348 = *(_QWORD *)(v256 + 56);
                  v349 = *(__int64 **)(v256 + 32);
                  v438 = *(_QWORD *)(v256 + 16);
                  v431 = *(_QWORD *)(v256 + 24);
                  v427 = *(_DWORD *)(v256 + 40);
                  v424 = *(_QWORD *)(v256 + 48);
                  v350 = *(_DWORD *)(v256 + 128) + 1;
                  *(_DWORD *)(v256 + 128) = v350;
                  v351 = v347 & v350;
                  *(_DWORD *)(v256 + 40) = v350;
                  v352 = (int *)(v348 + 4LL * (v347 & v350));
                  v353 = *v352;
                  *v352 = v350;
                  if ( v353 )
                  {
                    do
                    {
                      v351 = v347 & (v351 + 1);
                      v368 = (int *)(v348 + 4LL * v351);
                    }
                    while ( *v368 );
                    *v368 = v353;
                  }
                  v354 = *(_DWORD *)(v256 + 68) + 1;
                  *(_DWORD *)(v256 + 68) = v354;
                  if ( v347 < 2 * v354 )
                    sub_7702C0(v419);
                  *(_QWORD *)(v256 + 48) = 0;
                  v355 = sub_786210(v256, (_QWORD **)v477, v260, (char *)v260);
                  v33 = *(_QWORD *)(v256 + 48) == 0;
                  v485 = v355;
                  if ( !v33 && v355 )
                    v485 = sub_799890(v256);
                  v356 = *(_DWORD *)(v256 + 40);
                  v357 = *(_DWORD *)(v256 + 64);
                  v358 = *(_QWORD *)(v256 + 56);
                  v359 = *(__int64 **)(v256 + 32);
                  v360 = v357 & v356;
                  for ( i4 = (_DWORD *)(v358 + 4LL * (v357 & v356)); v356 != *i4; i4 = (_DWORD *)(v358 + 4LL * v360) )
                    v360 = v357 & (v360 + 1);
                  *i4 = 0;
                  if ( *(_DWORD *)(v358 + 4LL * ((v360 + 1) & v357)) )
                  {
                    v421 = v359;
                    sub_771390(*(_QWORD *)(v256 + 56), *(_DWORD *)(v256 + 64), v360);
                    v359 = v421;
                  }
                  --*(_DWORD *)(v256 + 68);
                  *(_QWORD *)(v256 + 32) = v349;
                  *(_QWORD *)(v256 + 16) = v438;
                  *(_QWORD *)(v256 + 24) = v431;
                  *(_DWORD *)(v256 + 40) = v427;
                  *(_QWORD *)(v256 + 48) = v424;
                  if ( v349 != v359 && v359 )
                  {
                    while ( 1 )
                    {
                      v362 = *((unsigned int *)v359 + 3);
                      v363 = *(_DWORD *)(v256 + 64);
                      v364 = *(_QWORD *)(v256 + 56);
                      v365 = v363 & *((_DWORD *)v359 + 3);
                      v366 = *(unsigned int *)(v364 + 4LL * v365);
                      if ( (_DWORD)v362 == (_DWORD)v366 || !(_DWORD)v362 )
                        goto LABEL_527;
                      while ( (_DWORD)v366 )
                      {
                        v365 = v363 & (v365 + 1);
                        v366 = *(unsigned int *)(v364 + 4LL * v365);
                        if ( (_DWORD)v362 == (_DWORD)v366 )
                          goto LABEL_527;
                      }
                      v367 = (__int64 *)*v359;
                      sub_822B90(v359, *((unsigned int *)v359 + 2), v366, v362);
                      if ( !v367 )
                        break;
                      v359 = v367;
                    }
                    v359 = 0;
LABEL_527:
                    *(_QWORD *)(v256 + 32) = v359;
                  }
                  if ( ((*(_BYTE *)(v477 + 25) & 3) != 0 || *(_BYTE *)(v259 + 140) == 6)
                    && (*(_BYTE *)(v260 + 8) & 4) != 0 )
                  {
                    v371 = *(_QWORD *)(v260 + 16);
                    v372 = 2;
                    v373 = *(_QWORD **)v371;
                    for ( i5 = **(_QWORD ***)v371; i5; ++v372 )
                    {
                      v373 = i5;
                      i5 = (_QWORD *)*i5;
                    }
                    *v373 = qword_4F08088;
                    *(_BYTE *)(v260 + 8) &= ~4u;
                    v375 = *(_QWORD *)(v371 + 24);
                    qword_4F08080 += v372;
                    qword_4F08088 = v371;
                    *(_QWORD *)(v260 + 16) = v375;
                  }
                  v369 = v485;
                  ++*(_QWORD *)(v256 + 120);
                  if ( !v369 )
                  {
LABEL_533:
                    v69 = v258;
                    v6 = v256;
                    goto LABEL_329;
                  }
                  v370 = sub_620EE0((_WORD *)v260, byte_4B6DF90[*(unsigned __int8 *)(v259 + 160)], &v486);
                  if ( v486 )
                  {
                    if ( !v485 || !v370 )
                      goto LABEL_533;
                  }
                  else
                  {
                    if ( !v370 )
                      goto LABEL_533;
                    v376 = *(_DWORD *)(v256 + 40);
                    v491.m128i_i32[2] = 0;
                    v491.m128i_i32[3] = v376;
                    v491.m128i_i64[0] = v490[0];
                    v492.m128i_i64[1] = v490[0];
                    if ( !(unsigned int)sub_79B7D0(v256, v416, v417, &v491, 0, 0) )
                    {
                      v69 = v258;
                      v6 = v256;
                      goto LABEL_386;
                    }
                    v485 = sub_795660(v256, v417->_IO_save_base);
                    if ( !v485 )
                      goto LABEL_533;
                    v377 = *(_QWORD *)(v256 + 72);
                    v378 = *(_BYTE *)(v377 + 48);
                    if ( (v378 & 1) != 0 )
                      goto LABEL_533;
                    if ( (v378 & 2) != 0 )
                    {
                      v69 = v258;
                      v6 = v256;
                      *(_BYTE *)(v377 + 48) = v378 & 0xFD;
                      goto LABEL_329;
                    }
                    if ( (v378 & 4) != 0 )
                      *(_BYTE *)(v377 + 48) = v378 & 0xFB;
                    v379 = *(_DWORD *)(v256 + 64);
                    v380 = *(_QWORD *)(v256 + 56);
                    v439 = *(_QWORD *)(v256 + 16);
                    v432 = *(_QWORD *)(v256 + 24);
                    v428 = *(__int64 **)(v256 + 32);
                    v425 = *(_DWORD *)(v256 + 40);
                    v422 = *(_QWORD *)(v256 + 48);
                    v381 = *(_DWORD *)(v256 + 128) + 1;
                    *(_DWORD *)(v256 + 128) = v381;
                    v382 = v379 & v381;
                    *(_DWORD *)(v256 + 40) = v381;
                    v383 = (int *)(v380 + 4LL * (v379 & v381));
                    v384 = *v383;
                    *v383 = v381;
                    if ( v384 )
                    {
                      do
                      {
                        v382 = v379 & (v382 + 1);
                        v399 = (int *)(v380 + 4LL * v382);
                      }
                      while ( *v399 );
                      *v399 = v384;
                    }
                    v385 = *(_DWORD *)(v256 + 68) + 1;
                    *(_DWORD *)(v256 + 68) = v385;
                    if ( v379 < 2 * v385 )
                      sub_7702C0(v419);
                    *(_QWORD *)(v256 + 48) = 0;
                    v386 = sub_786210(v256, (_QWORD **)v420, v418, (char *)v418);
                    v33 = *(_QWORD *)(v256 + 48) == 0;
                    v485 = v386;
                    if ( !v33 && v386 )
                      v485 = sub_799890(v256);
                    v387 = *(_DWORD *)(v256 + 40);
                    v388 = *(_DWORD *)(v256 + 64);
                    v389 = *(_QWORD *)(v256 + 56);
                    v390 = *(__int64 **)(v256 + 32);
                    v391 = v388 & v387;
                    for ( i6 = (_DWORD *)(v389 + 4LL * (v388 & v387)); v387 != *i6; i6 = (_DWORD *)(v389 + 4LL * v391) )
                      v391 = v388 & (v391 + 1);
                    *i6 = 0;
                    if ( *(_DWORD *)(v389 + 4LL * ((v391 + 1) & v388)) )
                      sub_771390(*(_QWORD *)(v256 + 56), *(_DWORD *)(v256 + 64), v391);
                    --*(_DWORD *)(v256 + 68);
                    *(_QWORD *)(v256 + 16) = v439;
                    *(_DWORD *)(v256 + 40) = v425;
                    *(_QWORD *)(v256 + 24) = v432;
                    *(_QWORD *)(v256 + 48) = v422;
                    *(_QWORD *)(v256 + 32) = v428;
                    if ( v390 && v428 != v390 )
                    {
                      do
                      {
                        v393 = *((unsigned int *)v390 + 3);
                        v394 = *(_DWORD *)(v256 + 64);
                        v395 = v390;
                        v396 = *(_QWORD *)(v256 + 56);
                        v397 = v394 & *((_DWORD *)v390 + 3);
                        v398 = *(unsigned int *)(v396 + 4LL * v397);
                        if ( (_DWORD)v393 == (_DWORD)v398 || !(_DWORD)v393 )
                          goto LABEL_566;
                        while ( (_DWORD)v398 )
                        {
                          v397 = v394 & (v397 + 1);
                          v398 = *(unsigned int *)(v396 + 4LL * v397);
                          if ( (_DWORD)v393 == (_DWORD)v398 )
                            goto LABEL_566;
                        }
                        v390 = (__int64 *)*v390;
                        sub_822B90(v395, *((unsigned int *)v395 + 2), v398, v393);
                      }
                      while ( v390 );
                      v390 = 0;
LABEL_566:
                      *(_QWORD *)(v256 + 32) = v390;
                    }
                    if ( ((*((_BYTE *)v420 + 25) & 3) != 0 || *(_BYTE *)(v423 + 140) == 6)
                      && (*(_BYTE *)(v418 + 8) & 4) != 0 )
                    {
                      v410 = *(_QWORD *)(v418 + 16);
                      v411 = 2;
                      v412 = *(_QWORD **)v410;
                      for ( i7 = **(_QWORD ***)v410; i7; ++v411 )
                      {
                        v412 = i7;
                        i7 = (_QWORD *)*i7;
                      }
                      *v412 = qword_4F08088;
                      *(_BYTE *)(v418 + 8) &= ~4u;
                      v414 = *(_QWORD *)(v410 + 24);
                      qword_4F08080 += v411;
                      qword_4F08088 = v410;
                      *(_QWORD *)(v418 + 16) = v414;
                    }
                    if ( !v485 )
                      goto LABEL_533;
                  }
                }
                v69 = v258;
                sub_6855B0(0x97Fu, (FILE *)(v256 + 112), (_QWORD *)(v256 + 96));
                v6 = v256;
                v485 = 0;
              }
            }
            else
            {
LABEL_386:
              v485 = 0;
            }
LABEL_329:
            v263 = v482;
            for ( i8 = v69; ; i8 = v263[3] )
            {
              --v263;
              sub_77A750(v6, i8);
              if ( v263 == &v482 )
                break;
            }
            if ( *(_QWORD *)(v6 + 48) && v485 )
              v485 = sub_799890(v6);
LABEL_236:
            v204 = *(_DWORD *)(v6 + 40);
            v205 = *(_DWORD *)(v6 + 64);
            v206 = *(_QWORD *)(v6 + 56);
            v207 = *(__int64 **)(v6 + 32);
            v208 = v205 & v204;
            for ( i9 = (_DWORD *)(v206 + 4LL * (v205 & v204)); v204 != *i9; i9 = (_DWORD *)(v206 + 4LL * v208) )
              v208 = v205 & (v208 + 1);
            *i9 = 0;
            if ( *(_DWORD *)(v206 + 4LL * ((v208 + 1) & v205)) )
              sub_771390(*(_QWORD *)(v6 + 56), *(_DWORD *)(v6 + 64), v208);
            --*(_DWORD *)(v6 + 68);
            *(_QWORD *)(v6 + 16) = v462;
            *(_DWORD *)(v6 + 40) = v443;
            *(_QWORD *)(v6 + 24) = v454;
            *(_QWORD *)(v6 + 48) = v442;
            *(_QWORD *)(v6 + 32) = v447;
            if ( v207 && v447 != v207 )
            {
              while ( 1 )
              {
                v210 = *((unsigned int *)v207 + 3);
                v211 = *(_DWORD *)(v6 + 64);
                v212 = *(_QWORD *)(v6 + 56);
                v213 = v211 & *((_DWORD *)v207 + 3);
                v214 = *(unsigned int *)(v212 + 4LL * v213);
                if ( (_DWORD)v210 == (_DWORD)v214 || !(_DWORD)v210 )
                  goto LABEL_313;
                while ( (_DWORD)v214 )
                {
                  v213 = v211 & (v213 + 1);
                  v214 = *(unsigned int *)(v212 + 4LL * v213);
                  if ( (_DWORD)v210 == (_DWORD)v214 )
                    goto LABEL_313;
                }
                v215 = (__int64 *)*v207;
                sub_822B90(v207, *((unsigned int *)v207 + 2), v214, v210);
                if ( !v215 )
                  break;
                v207 = v215;
              }
              v207 = 0;
LABEL_313:
              *(_QWORD *)(v6 + 32) = v207;
            }
            return v485;
          }
LABEL_376:
          v283 = 8;
          v284 = 16;
          v285 = 16;
          goto LABEL_377;
        }
      }
      if ( (unsigned __int8)(v281 - 8) <= 3u )
      {
        v407 = ((unsigned int)(v282 + 7) >> 3) + 17 - (((unsigned __int8)((unsigned int)(v282 + 7) >> 3) + 9) & 7);
        v284 = v407;
        v286 = v282 + v407;
        v283 = v407 - 8LL;
        goto LABEL_378;
      }
      goto LABEL_376;
    case 16:
      v78 = *(unsigned __int64 **)(a2 + 48);
      v79 = *(_QWORD *)(a2 + 80);
      LODWORD(v487[0]) = 1;
      v80 = *(__int64 **)(v79 + 16);
      LOBYTE(v79) = *((_BYTE *)v78 + 24);
      i11 = v78;
      v463 = v79;
      v455 = (_BYTE)v79 == 9;
      if ( (_BYTE)v79 == 9 && !(unsigned int)sub_77A4E0((const __m128i *)v6, i11[7], &v491) )
        return LODWORD(v487[0]);
      v81 = *i11;
      for ( i10 = *(_BYTE *)(*i11 + 140); i10 == 12; i10 = *(_BYTE *)(v81 + 140) )
        v81 = *(_QWORD *)(v81 + 160);
      if ( (unsigned __int8)(i10 - 2) > 1u )
      {
        v472 = sub_7764B0(v6, v81, v487);
        if ( (unsigned __int8)(*(_BYTE *)(v81 + 140) - 8) > 3u )
        {
          v84 = 8;
          v83 = 16;
          v85 = (unsigned int)(v472 + 16);
        }
        else
        {
          v196 = (unsigned int)(v472 + 7) >> 3;
          v83 = v196 + 9;
          if ( (((_BYTE)v196 + 9) & 7) != 0 )
            v83 = v196 + 17 - (((_BYTE)v196 + 9) & 7);
          v85 = (unsigned int)(v472 + v83);
          v84 = v83 - 8;
        }
        if ( (unsigned int)v85 > 0x400 )
        {
          v448 = v84;
          LODWORD(v482) = v85 + 16;
          v197 = sub_822B10((unsigned int)(v85 + 16));
          v198 = (int)v482;
          v84 = v448;
          v86 = (char *)(v197 + 16);
          *(_QWORD *)v197 = *(_QWORD *)(v6 + 32);
          *(_DWORD *)(v197 + 8) = v198;
          *(_DWORD *)(v197 + 12) = *(_DWORD *)(v6 + 40);
          *(_QWORD *)(v6 + 32) = v197;
          goto LABEL_84;
        }
        if ( (v85 & 7) != 0 )
          v85 = (_DWORD)v85 + 8 - (unsigned int)(v85 & 7);
      }
      else
      {
        v472 = 16;
        v83 = 16;
        v84 = 8;
        v85 = 32;
      }
      v86 = *(char **)(v6 + 16);
      if ( 0x10000 - (*(_DWORD *)(v6 + 16) - *(_DWORD *)(v6 + 24)) < (unsigned int)v85 )
      {
        v450 = v84;
        LODWORD(v482) = v85;
        sub_772E70((_QWORD *)(v6 + 16));
        v86 = *(char **)(v6 + 16);
        v84 = v450;
        v85 = (unsigned int)v482;
      }
      *(_QWORD *)(v6 + 16) = &v86[v85];
LABEL_84:
      v87 = (char *)memset(v86, 0, v84) + v83;
      *(v87 - 1) = v81;
      v88 = *(_BYTE *)(v81 + 140);
      v482 = v87;
      if ( (unsigned __int8)(v88 - 9) <= 2u )
        *v87 = 0;
      LODWORD(v487[0]) = sub_7A0A10(v455, v6, i11, v81, v482);
      if ( !LODWORD(v487[0]) )
        goto LABEL_265;
      v89 = byte_4B6DF90[*(unsigned __int8 *)(v81 + 160)];
      if ( (unsigned __int8)(*(_BYTE *)(v81 + 140) - 8) > 3u )
      {
        v93 = 8;
        v92 = 16;
        v91 = 16;
      }
      else
      {
        v90 = (unsigned int)(v472 + 7) >> 3;
        v91 = v90 + 9;
        if ( (((_BYTE)v90 + 9) & 7) != 0 )
          v91 = v90 + 17 - (((_BYTE)v90 + 9) & 7);
        v92 = v91;
        v93 = v91 - 8LL;
      }
      v94 = v91 + v472;
      if ( v91 + v472 > 0x400 )
      {
        v451 = v93;
        v457 = byte_4B6DF90[*(unsigned __int8 *)(v81 + 160)];
        v480 = v94 + 16;
        v328 = sub_822B10(v94 + 16);
        v89 = v457;
        v93 = v451;
        v95 = (char *)(v328 + 16);
        *(_QWORD *)v328 = *(_QWORD *)(v6 + 32);
        *(_DWORD *)(v328 + 8) = v480;
        *(_DWORD *)(v328 + 12) = *(_DWORD *)(v6 + 40);
        *(_QWORD *)(v6 + 32) = v328;
      }
      else
      {
        v95 = *(char **)(v6 + 16);
        if ( (v94 & 7) != 0 )
          v94 = v94 + 8 - (v94 & 7);
        if ( 0x10000 - (*(_DWORD *)(v6 + 16) - *(_DWORD *)(v6 + 24)) < v94 )
        {
          v452 = v93;
          v460 = v94;
          v481 = byte_4B6DF90[*(unsigned __int8 *)(v81 + 160)];
          sub_772E70((_QWORD *)(v6 + 16));
          v95 = *(char **)(v6 + 16);
          v93 = v452;
          v94 = v460;
          v89 = v481;
        }
        *(_QWORD *)(v6 + 16) = &v95[v94];
      }
      v473 = v89;
      v96 = memset(v95, 0, v93);
      v98 = v473;
      v99 = (unsigned __int64)v96 + v92;
      *(_QWORD *)(v99 - 8) = v81;
      if ( (unsigned __int8)(*(_BYTE *)(v81 + 140) - 9) <= 2u )
        *(_QWORD *)v99 = 0;
      if ( !v80 )
        goto LABEL_254;
      v100 = (__int16 *)v482;
      v482 = (_QWORD *)a2;
      while ( 2 )
      {
        v104 = (const __m128i *)v80[1];
        v105 = v104[10].m128i_i8[13];
        if ( v105 == 1 )
        {
          if ( (v104[10].m128i_i8[8] & 8) == 0 )
          {
            if ( (v104[10].m128i_i8[11] & 4) != 0 )
              goto LABEL_264;
            goto LABEL_103;
          }
        }
        else if ( v105 == 3 )
        {
          if ( (v104[10].m128i_i8[11] & 4) != 0 )
            goto LABEL_264;
LABEL_103:
          *(__m128i *)v99 = _mm_loadu_si128(v104 + 11);
          LODWORD(v487[0]) = 1;
          goto LABEL_104;
        }
        LODWORD(v487[0]) = sub_79CCD0(v6, v104, v99, v99, 0);
        if ( !LODWORD(v487[0]) )
          goto LABEL_265;
LABEL_104:
        v101 = sub_621000(v100, v473, (__int16 *)v99, v473);
        if ( !v101 )
          goto LABEL_475;
        if ( v101 >= 0 )
        {
          v102 = (const __m128i *)v80[2];
          if ( !v102 )
            goto LABEL_112;
          v103 = v102[10].m128i_i8[13];
          if ( v103 != 1 )
          {
            if ( v103 == 3 )
            {
              if ( (v102[10].m128i_i8[11] & 4) != 0 )
                goto LABEL_264;
              goto LABEL_110;
            }
LABEL_303:
            LODWORD(v487[0]) = sub_79CCD0(v6, v102, v99, v99, 0);
            if ( !LODWORD(v487[0]) )
              goto LABEL_265;
            goto LABEL_111;
          }
          if ( (v102[10].m128i_i8[8] & 8) != 0 )
            goto LABEL_303;
          if ( (v102[10].m128i_i8[11] & 4) == 0 )
          {
LABEL_110:
            *(__m128i *)v99 = _mm_loadu_si128(v102 + 11);
            LODWORD(v487[0]) = 1;
LABEL_111:
            if ( (int)sub_621000(v100, v473, (__int16 *)v99, v473) > 0 )
            {
LABEL_112:
              v80 = (__int64 *)v80[4];
              if ( !v80 )
                break;
              continue;
            }
LABEL_475:
            v7 = (FILE *)v482;
            goto LABEL_255;
          }
LABEL_264:
          LODWORD(v487[0]) = 0;
          goto LABEL_265;
        }
        break;
      }
      v7 = (FILE *)v482;
LABEL_254:
      v80 = (__int64 *)*((_QWORD *)v7->_IO_backup_base + 1);
      if ( v80 )
      {
LABEL_255:
        v216 = *v80;
        v217 = v6;
        v218 = v7;
        while ( 1 )
        {
          v219 = *(_QWORD *)(v217 + 72);
          v220 = *(_BYTE *)(v219 + 48);
          if ( (v220 & 8) != 0 )
            break;
          if ( (v220 & 2) != 0 )
          {
            while ( 1 )
            {
              v216 = *(_QWORD *)(v216 + 24);
              if ( v218 == (FILE *)v216 )
                goto LABEL_392;
              v289 = *(_BYTE *)(v216 + 40);
              if ( (unsigned __int8)(v289 - 12) <= 1u || v289 == 5 )
              {
                *(_BYTE *)(v219 + 48) &= ~2u;
                v221 = v216;
                goto LABEL_261;
              }
            }
          }
          if ( (v220 & 4) != 0 )
          {
            while ( 1 )
            {
              v216 = *(_QWORD *)(v216 + 24);
              if ( v218 == (FILE *)v216 )
                break;
              v290 = *(_BYTE *)(v216 + 40);
              if ( (unsigned __int8)(v290 - 12) <= 1u || v290 == 5 )
              {
                *(_BYTE *)(v219 + 48) &= ~4u;
                if ( *(_BYTE *)(v216 + 40) == 13 )
                  v333 = sub_7A0E60(v217, v216, 1, v219, v97, v98);
                else
                  v333 = sub_795660(v217, v216) == 0;
                LODWORD(v487[0]) = v333;
                if ( v333 )
                  goto LABEL_260;
                break;
              }
            }
LABEL_392:
            v6 = v217;
            goto LABEL_265;
          }
          if ( (v220 & 1) != 0 )
            goto LABEL_392;
LABEL_260:
          v221 = v216;
LABEL_261:
          v216 = *(_QWORD *)(v216 + 16);
          if ( !v216 )
          {
            v291 = *(_QWORD *)(v221 + 24);
            if ( v218 == (FILE *)v291 )
              goto LABEL_392;
            while ( 1 )
            {
              v292 = *(_BYTE *)(v291 + 40);
              if ( v292 == 5 )
              {
LABEL_438:
                v216 = v291;
                goto LABEL_262;
              }
              if ( v292 == 12 )
              {
                v293 = **(_QWORD **)(v291 + 48);
                for ( i11 = *(unsigned __int64 **)(v291 + 48); *(_BYTE *)(v293 + 140) == 12; v293 = *(_QWORD *)(v293 + 160) )
                  ;
                v294 = *(_DWORD *)(v217 + 64);
                v295 = *(_QWORD *)(v217 + 56);
                v482 = *(_QWORD **)(v217 + 16);
                v478 = *(_QWORD *)(v217 + 24);
                v456 = *(_QWORD **)(v217 + 32);
                v449 = *(_DWORD *)(v217 + 40);
                v445 = *(_QWORD *)(v217 + 48);
                v296 = *(_DWORD *)(v217 + 128) + 1;
                *(_DWORD *)(v217 + 128) = v296;
                v297 = v294 & v296;
                *(_DWORD *)(v217 + 40) = v296;
                v298 = (int *)(v295 + 4LL * (v294 & v296));
                v299 = *v298;
                *v298 = v296;
                if ( v299 )
                {
                  do
                  {
                    v297 = v294 & (v297 + 1);
                    v316 = (int *)(v295 + 4LL * v297);
                  }
                  while ( *v316 );
                  *v316 = v299;
                }
                v300 = *(_DWORD *)(v217 + 68) + 1;
                *(_DWORD *)(v217 + 68) = v300;
                if ( v294 < 2 * v300 )
                  sub_7702C0(v217 + 56);
                v301 = i11;
                *(_QWORD *)(v217 + 48) = 0;
                v302 = sub_786210(v217, (_QWORD **)v301, v99, (char *)v99);
                v33 = *(_QWORD *)(v217 + 48) == 0;
                LODWORD(v487[0]) = v302;
                if ( !v33 && v302 )
                  LODWORD(v487[0]) = sub_799890(v217);
                v303 = *(_DWORD *)(v217 + 40);
                v304 = *(_DWORD *)(v217 + 64);
                v305 = *(_QWORD *)(v217 + 56);
                v306 = *(_QWORD **)(v217 + 32);
                v307 = v304 & v303;
                for ( i12 = (_DWORD *)(v305 + 4LL * (v304 & v303)); v303 != *i12; i12 = (_DWORD *)(v305 + 4LL * v307) )
                  v307 = v304 & (v307 + 1);
                *i12 = 0;
                if ( *(_DWORD *)(v305 + 4LL * (v304 & (v307 + 1))) )
                {
                  v437 = v306;
                  sub_771390(*(_QWORD *)(v217 + 56), *(_DWORD *)(v217 + 64), v307);
                  v306 = v437;
                }
                v309 = v482;
                --*(_DWORD *)(v217 + 68);
                *(_QWORD *)(v217 + 16) = v309;
                *(_DWORD *)(v217 + 40) = v449;
                *(_QWORD *)(v217 + 24) = v478;
                *(_QWORD *)(v217 + 48) = v445;
                *(_QWORD *)(v217 + 32) = v456;
                if ( v306 && v456 != v306 )
                {
                  while ( 1 )
                  {
                    v310 = *((unsigned int *)v306 + 3);
                    v311 = *(_DWORD *)(v217 + 64);
                    v312 = *(_QWORD *)(v217 + 56);
                    v313 = v311 & *((_DWORD *)v306 + 3);
                    v314 = *(unsigned int *)(v312 + 4LL * v313);
                    if ( (_DWORD)v310 == (_DWORD)v314 || !(_DWORD)v310 )
                      break;
                    while ( (_DWORD)v314 )
                    {
                      v313 = v311 & (v313 + 1);
                      v314 = *(unsigned int *)(v312 + 4LL * v313);
                      if ( (_DWORD)v310 == (_DWORD)v314 )
                        goto LABEL_432;
                    }
                    v315 = *((unsigned int *)v306 + 2);
                    v482 = (_QWORD *)*v306;
                    sub_822B90(v306, v315, v314, v310);
                    if ( !v482 )
                    {
                      v306 = 0;
                      break;
                    }
                    v306 = v482;
                  }
LABEL_432:
                  *(_QWORD *)(v217 + 32) = v306;
                }
                if ( ((*((_BYTE *)i11 + 25) & 3) != 0 || *(_BYTE *)(v293 + 140) == 6) && (*(_BYTE *)(v99 + 8) & 4) != 0 )
                {
                  v318 = *(_QWORD *)(v99 + 16);
                  v319 = 2;
                  v320 = *(_QWORD **)v318;
                  for ( i13 = **(_QWORD ***)v318; i13; ++v319 )
                  {
                    v320 = i13;
                    i13 = (_QWORD *)*i13;
                  }
                  *v320 = qword_4F08088;
                  *(_BYTE *)(v99 + 8) &= ~4u;
                  v322 = *(_QWORD *)(v318 + 24);
                  qword_4F08080 += v319;
                  qword_4F08088 = v318;
                  *(_QWORD *)(v99 + 16) = v322;
                }
                if ( !LODWORD(v487[0]) )
                  goto LABEL_392;
                v317 = sub_620EE0((_WORD *)v99, byte_4B6DF90[*(unsigned __int8 *)(v293 + 160)], v490);
                if ( LODWORD(v490[0]) || v317 )
                  goto LABEL_438;
              }
              else if ( v292 == 13 && !(unsigned int)sub_7A0E60(v217, v291, 1, v219, v97, v98) )
              {
                goto LABEL_263;
              }
              v216 = *(_QWORD *)(v291 + 16);
              if ( v216 )
                break;
LABEL_404:
              v291 = *(_QWORD *)(v291 + 24);
              if ( v218 == (FILE *)v291 )
                goto LABEL_392;
            }
            while ( *(_BYTE *)(v216 + 40) == 7 )
            {
              v216 = *(_QWORD *)(v216 + 16);
              if ( !v216 )
                goto LABEL_404;
            }
          }
LABEL_262:
          if ( !(unsigned int)sub_795660(v217, v216) )
          {
LABEL_263:
            v6 = v217;
            goto LABEL_264;
          }
        }
        v6 = v217;
        *(_BYTE *)(v219 + 48) = v220 & 0xF7;
      }
LABEL_265:
      if ( v463 == 9 )
      {
        sub_7762B0(v6, i11[7]);
        sub_7999E0(v6, i11[7], &v491, v487);
      }
      return LODWORD(v487[0]);
    case 17:
      v106 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 8LL);
      v107 = sub_77A250(a1, v106, &v484);
      result = v484;
      if ( v484 )
        return sub_7A0470(a1, v106, v107, a2);
      return result;
    case 19:
      return sub_7987E0(
               a1,
               *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 72) + 8LL) + 72LL),
               *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 72) + 8LL) + 80LL) + 8LL));
    case 20:
      result = 1;
      if ( (*(_BYTE *)(a2 + 80) & 1) == 0 )
        return result;
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        return 0;
      sub_6855B0(0xC88u, (FILE *)a2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    case 21:
    case 22:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        return 0;
      sub_6855B0(0xA90u, (FILE *)a2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    case 25:
      v19 = *(_QWORD **)(a1 + 72);
      v20 = (__m128i *)v19[3];
      v21 = (char *)v19[4];
      if ( *(_QWORD *)(a2 + 48) )
      {
        v22 = *(_DWORD *)(a1 + 64);
        v23 = *(_DWORD *)(a1 + 40);
        v24 = *(_QWORD *)(a1 + 16);
        i11 = *(unsigned __int64 **)(a1 + 24);
        v25 = *(__int64 **)(a1 + 32);
        v482 = *(_QWORD **)(a1 + 48);
        v26 = *(_DWORD *)(a1 + 128) + 1;
        *(_DWORD *)(a1 + 128) = v26;
        *(_DWORD *)(a1 + 40) = v26;
        v27 = v22 & v26;
        v28 = *(_QWORD *)(a1 + 56);
        v29 = (int *)(v28 + 4LL * (v22 & v26));
        v30 = *v29;
        *v29 = v26;
        if ( v30 )
        {
          do
          {
            v27 = v22 & (v27 + 1);
            v184 = (int *)(v28 + 4LL * v27);
          }
          while ( *v184 );
          *v184 = v30;
        }
        v31 = *(_DWORD *)(v6 + 68) + 1;
        *(_DWORD *)(v6 + 68) = v31;
        if ( 2 * v31 > v22 )
        {
          v466 = v21;
          v479 = v20;
          sub_7702C0(v6 + 56);
          v21 = v466;
          v20 = v479;
        }
        *(_QWORD *)(v6 + 48) = 0;
        v32 = sub_786210(v6, (_QWORD **)v7->_IO_write_end, (unsigned __int64)v20, v21);
        v33 = *(_QWORD *)(v6 + 48) == 0;
        v484 = v32;
        if ( !v33 && v32 )
          v484 = sub_799890(v6);
        v34 = *(_DWORD *)(v6 + 40);
        v35 = *(_DWORD *)(v6 + 64);
        v36 = *(_QWORD *)(v6 + 56);
        v37 = *(__int64 **)(v6 + 32);
        v38 = v35 & v34;
        for ( i14 = (_DWORD *)(v36 + 4LL * (v35 & v34)); v34 != *i14; i14 = (_DWORD *)(v36 + 4LL * v38) )
          v38 = v35 & (v38 + 1);
        *i14 = 0;
        if ( *(_DWORD *)(v36 + 4LL * (v35 & (v38 + 1))) )
          sub_771390(*(_QWORD *)(v6 + 56), *(_DWORD *)(v6 + 64), v38);
        v40 = i11;
        --*(_DWORD *)(v6 + 68);
        *(_QWORD *)(v6 + 16) = v24;
        *(_QWORD *)(v6 + 24) = v40;
        v41 = v482;
        *(_QWORD *)(v6 + 32) = v25;
        *(_DWORD *)(v6 + 40) = v23;
        *(_QWORD *)(v6 + 48) = v41;
        if ( v25 == v37 || !v37 )
          return v484;
        while ( 1 )
        {
          v42 = *((unsigned int *)v37 + 3);
          v43 = *(_QWORD *)(v6 + 56);
          v44 = v42 & *(_DWORD *)(v6 + 64);
          v45 = *(unsigned int *)(v43 + 4LL * v44);
          if ( (_DWORD)v45 == (_DWORD)v42 || !(_DWORD)v42 )
            goto LABEL_336;
          while ( (_DWORD)v45 )
          {
            v44 = *(_DWORD *)(v6 + 64) & (v44 + 1);
            v45 = *(unsigned int *)(v43 + 4LL * v44);
            if ( (_DWORD)v42 == (_DWORD)v45 )
              goto LABEL_336;
          }
          v46 = (__int64 *)*v37;
          sub_822B90(v37, *((unsigned int *)v37 + 2), v45, v42);
          if ( !v46 )
            break;
          v37 = v46;
        }
        v37 = 0;
LABEL_336:
        *(_QWORD *)(v6 + 32) = v37;
        return v484;
      }
      v193 = *(_QWORD *)(a2 + 72);
      result = 1;
      if ( v193 )
      {
        if ( *(_BYTE *)(v193 + 48) == 1 )
        {
          v194 = (__int64 *)v19[2];
          for ( i15 = *v194; *(_BYTE *)(i15 + 140) == 12; i15 = *(_QWORD *)(i15 + 160) )
            ;
          sub_7790A0(a1, v20, i15, (__int64)v21);
          return v484;
        }
        else
        {
          v342 = *(_DWORD *)(a1 + 40);
          v491.m128i_i64[0] = v19[3];
          v492.m128i_i64[1] = (__int64)v21;
          v491.m128i_i32[2] = 0;
          v491.m128i_i32[3] = v342;
          return sub_79B7D0(a1, v193, v7, &v491, 0, 0);
        }
      }
      return result;
    default:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        return 0;
      sub_6855B0(0xAA2u, (FILE *)a2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
  }
}
