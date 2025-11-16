// Function: sub_1F27020
// Address: 0x1f27020
//
_BOOL8 __fastcall sub_1F27020(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned int v14; // ecx
  int v15; // r8d
  int v16; // r9d
  int v17; // r14d
  _QWORD *v18; // r12
  _QWORD *v19; // r13
  unsigned int v20; // eax
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *j; // rdx
  __int64 v25; // r14
  __int64 v26; // r12
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // r15
  __int64 v29; // rbx
  __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 *v32; // r13
  unsigned __int64 *v33; // r12
  __int64 v34; // r13
  unsigned __int64 *v35; // r12
  unsigned __int64 *v36; // r13
  unsigned __int64 v37; // rdi
  __int64 v38; // rax
  _QWORD *v39; // r12
  __int64 v40; // rdx
  unsigned __int64 *v41; // r13
  unsigned __int64 *v42; // r12
  unsigned __int64 v43; // rdi
  __int64 m; // rdx
  unsigned int v45; // eax
  unsigned __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rax
  unsigned int v49; // r8d
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned int v53; // eax
  unsigned int v54; // ecx
  int v55; // edi
  __int64 v56; // rdx
  int v57; // r13d
  __int64 *v58; // rbx
  __int64 *v59; // r12
  __int64 v60; // rdi
  _QWORD *v62; // rdi
  unsigned int v63; // eax
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rax
  int v67; // r13d
  __int64 v68; // r12
  _QWORD *v69; // rax
  __int64 v70; // rdx
  _QWORD *k; // rdx
  __int64 v72; // rsi
  unsigned __int64 *v73; // r13
  unsigned __int64 *v74; // r12
  int v75; // edx
  __int64 v76; // r12
  unsigned int v77; // eax
  _QWORD *v78; // rdi
  unsigned __int64 v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // rdx
  _QWORD *i; // rdx
  int v84; // ebx
  __int64 v85; // rax
  __int64 v86; // r15
  int v87; // edx
  unsigned __int64 v88; // r14
  __int64 v89; // rax
  unsigned __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rdx
  unsigned int v93; // eax
  __int64 *v94; // rdx
  __int64 v95; // rax
  __int64 v96; // r14
  unsigned __int64 v97; // rdi
  __int64 *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rcx
  int v103; // r9d
  _DWORD *v104; // r11
  __int64 v105; // rdi
  _QWORD *v106; // r14
  _QWORD *v107; // rbx
  _QWORD *v108; // r12
  __int64 v109; // r11
  _QWORD *v110; // r8
  int v111; // r13d
  unsigned int v112; // edx
  __int64 *v113; // r15
  __int64 v114; // r9
  __int64 v115; // r13
  unsigned int v116; // r10d
  unsigned int v117; // r9d
  __int64 v118; // rbx
  __int64 v119; // rsi
  __int64 n; // rax
  __int64 v121; // rax
  _DWORD *v122; // r11
  unsigned int v123; // r8d
  char *v124; // r14
  unsigned int v125; // r8d
  __int64 v126; // rsi
  __int64 ii; // rax
  unsigned int v128; // r9d
  unsigned int v129; // ebx
  __int64 v130; // r15
  __int64 jj; // rax
  unsigned int v132; // edi
  unsigned int v133; // eax
  __int64 v134; // rsi
  __int64 v135; // rdx
  __int64 v136; // rax
  __int64 v137; // rcx
  _QWORD *v138; // rdx
  unsigned int v139; // r8d
  unsigned int v140; // ecx
  __int64 v141; // rdx
  __int64 v142; // rax
  __int64 v143; // rcx
  _QWORD *v144; // rdx
  __int64 v145; // r15
  __int64 v146; // rax
  __int64 v147; // r13
  unsigned int v148; // eax
  unsigned __int64 v149; // rdx
  __int64 v150; // rax
  __int64 *v151; // rax
  __int64 *kk; // rdx
  __int64 v153; // rax
  _BYTE *v154; // rax
  _BYTE *mm; // rdx
  unsigned int v156; // esi
  unsigned int v157; // edx
  unsigned __int64 v158; // rbx
  __int64 v159; // rax
  int v160; // ecx
  __int64 v161; // rax
  int nn; // eax
  __int64 v165; // rcx
  unsigned int v166; // eax
  int v167; // edx
  unsigned int v168; // esi
  int v169; // ecx
  unsigned __int64 v170; // r10
  __int64 v171; // rdx
  unsigned __int64 v172; // r10
  __int64 v175; // rbx
  __int64 *v176; // r14
  __int64 v177; // rsi
  char *v178; // r12
  unsigned __int64 *v179; // r13
  unsigned __int64 v180; // rbx
  unsigned __int64 v181; // rax
  __int64 v182; // rdi
  __int64 v183; // rdx
  int v184; // ecx
  __int64 v185; // rax
  char *v186; // rax
  unsigned __int64 v187; // rdx
  unsigned __int64 v188; // r10
  char *v189; // rdi
  __int64 v190; // rax
  char *v191; // rax
  unsigned __int64 v192; // rdx
  __int64 v193; // rcx
  unsigned __int64 v194; // rdx
  __int64 v195; // rdi
  __int64 v196; // rcx
  __int64 *v197; // rax
  __int64 v198; // r10
  unsigned __int64 v199; // r10
  int *v200; // rbx
  int *v201; // r12
  __int64 v202; // r13
  __int64 v203; // r14
  __int64 *v204; // rax
  __int64 v205; // r15
  __int64 v206; // rax
  __int64 v207; // rax
  __int64 v208; // rdx
  __int64 v209; // rdi
  __int64 v210; // rdi
  __int64 v211; // rdx
  unsigned __int64 v212; // rdx
  int v213; // eax
  char *v214; // r12
  __int64 v215; // rax
  __int64 v216; // rax
  char *v217; // r15
  __int64 v218; // rbx
  __int64 v219; // r14
  char *v220; // rax
  char *v221; // r13
  double v222; // xmm4_8
  double v223; // xmm5_8
  __int64 v224; // r12
  __int64 i1; // rbx
  __int64 v226; // r14
  __int64 *v227; // r13
  unsigned __int64 v228; // rax
  __int64 *v229; // r14
  __int64 v230; // r9
  __int64 *v231; // rdx
  unsigned __int64 v232; // rsi
  __int64 v233; // r8
  __int64 v234; // rax
  __int64 *v235; // rdi
  _DWORD *v236; // rsi
  _QWORD *v237; // r9
  _QWORD *v238; // r15
  __int64 v239; // rbx
  unsigned int v240; // r12d
  unsigned int v241; // r13d
  __int64 v242; // rdx
  char v243; // al
  __int64 v244; // r11
  __int64 v245; // rcx
  __int64 v246; // r8
  _QWORD *v247; // r9
  __int64 v248; // r9
  unsigned int v249; // r10d
  __int64 *v250; // rax
  size_t v251; // r11
  unsigned __int64 v252; // r8
  __int64 v253; // rdx
  char *v254; // rdi
  int v255; // edx
  unsigned int v256; // edx
  unsigned int *v257; // rax
  unsigned int v258; // ecx
  __int64 v259; // rdi
  int v260; // ecx
  __int64 v261; // rdx
  __int64 v262; // rax
  __int64 v263; // rdx
  unsigned int v264; // esi
  unsigned int v265; // esi
  int v266; // r15d
  unsigned int i2; // ecx
  __int64 v268; // rdx
  int v269; // r9d
  int v270; // r12d
  unsigned int v271; // r8d
  unsigned int v272; // r11d
  _DWORD *v273; // rax
  int v274; // edi
  int v275; // ebx
  int v276; // ebx
  int v277; // r13d
  __int64 v278; // r8
  _DWORD *v279; // rax
  int v280; // edi
  unsigned int v281; // r8d
  _DWORD *v282; // rax
  int v283; // edi
  __int64 *v284; // rbx
  __int64 *v285; // r12
  int v286; // r13d
  __int64 v287; // rdi
  __int64 v288; // rdi
  __int64 v289; // rax
  __int64 v290; // rbx
  char *v291; // r15
  unsigned __int16 *v292; // rcx
  int v293; // edx
  __int64 v294; // r12
  __int16 v295; // ax
  __int64 v296; // rax
  __int64 v297; // rax
  __int16 v298; // dx
  bool v299; // al
  __int64 v300; // r15
  __int64 v301; // rbx
  __int64 v302; // rax
  __int64 v303; // r14
  __int64 v304; // rsi
  unsigned __int64 v305; // rdx
  __int64 v306; // r8
  unsigned int v307; // esi
  unsigned int v308; // edi
  __int64 *v309; // rax
  __int64 v310; // r11
  int v311; // edi
  unsigned __int64 v312; // rcx
  int v313; // eax
  int v314; // eax
  int v315; // r8d
  __int64 v316; // r10
  unsigned int v317; // edx
  __int64 v318; // rdi
  int v319; // esi
  unsigned __int64 v320; // rcx
  int v321; // r10d
  unsigned int v322; // r11d
  int i3; // r9d
  int v324; // r10d
  _DWORD *v325; // r9
  int v326; // edx
  unsigned int v327; // r9d
  int v328; // edx
  int v329; // edi
  int v330; // esi
  _DWORD *v331; // rcx
  int v332; // eax
  int v333; // ecx
  __int64 v334; // rcx
  int v335; // edx
  unsigned int v336; // r9d
  int v337; // r10d
  _DWORD *v338; // r9
  int v339; // edx
  int v340; // r10d
  unsigned int *v341; // rdi
  unsigned int *v342; // rcx
  __int64 v343; // r8
  int v344; // esi
  unsigned int v345; // edi
  int v346; // esi
  int v347; // esi
  int v348; // ecx
  unsigned int v349; // r12d
  unsigned __int64 v350; // rdx
  __int64 v351; // rdi
  int v352; // ecx
  _DWORD *v353; // rdx
  unsigned int v354; // r9d
  int v355; // esi
  int v356; // edi
  _DWORD *v357; // rcx
  unsigned int v358; // r11d
  int v359; // esi
  int v360; // edi
  unsigned int v361; // r11d
  int v362; // edi
  int v363; // esi
  _QWORD *v364; // rax
  _DWORD *v365; // rcx
  unsigned int v366; // r13d
  int v367; // esi
  int v368; // edi
  unsigned int v369; // r13d
  int v370; // edi
  int v371; // esi
  _QWORD *v372; // rax
  __int64 v373; // rax
  __int64 v374; // rax
  __int64 v375; // rax
  int v376; // edi
  unsigned int *v377; // rsi
  __int128 v378; // [rsp-20h] [rbp-210h]
  char v379; // [rsp+10h] [rbp-1E0h]
  __int64 v380; // [rsp+18h] [rbp-1D8h]
  _QWORD *v381; // [rsp+20h] [rbp-1D0h]
  unsigned int v382; // [rsp+28h] [rbp-1C8h]
  __int64 v383; // [rsp+28h] [rbp-1C8h]
  _DWORD *v384; // [rsp+28h] [rbp-1C8h]
  bool v385; // [rsp+30h] [rbp-1C0h]
  __int64 v386; // [rsp+30h] [rbp-1C0h]
  int v387; // [rsp+30h] [rbp-1C0h]
  __int64 v388; // [rsp+30h] [rbp-1C0h]
  _QWORD *v389; // [rsp+30h] [rbp-1C0h]
  __int64 v390; // [rsp+30h] [rbp-1C0h]
  unsigned int v391; // [rsp+30h] [rbp-1C0h]
  _DWORD *v392; // [rsp+30h] [rbp-1C0h]
  _DWORD *v393; // [rsp+30h] [rbp-1C0h]
  __int64 v394; // [rsp+38h] [rbp-1B8h]
  unsigned int v395; // [rsp+38h] [rbp-1B8h]
  _QWORD *v396; // [rsp+38h] [rbp-1B8h]
  unsigned int v397; // [rsp+38h] [rbp-1B8h]
  _QWORD *v398; // [rsp+38h] [rbp-1B8h]
  unsigned int v399; // [rsp+38h] [rbp-1B8h]
  unsigned int v400; // [rsp+38h] [rbp-1B8h]
  unsigned int v401; // [rsp+38h] [rbp-1B8h]
  __int64 v402; // [rsp+38h] [rbp-1B8h]
  __int64 v403; // [rsp+38h] [rbp-1B8h]
  int v404; // [rsp+40h] [rbp-1B0h]
  unsigned int v405; // [rsp+44h] [rbp-1ACh]
  __int64 v406; // [rsp+48h] [rbp-1A8h]
  __int64 v407; // [rsp+48h] [rbp-1A8h]
  unsigned int v408; // [rsp+48h] [rbp-1A8h]
  int v409; // [rsp+48h] [rbp-1A8h]
  unsigned int v410; // [rsp+48h] [rbp-1A8h]
  unsigned int v411; // [rsp+48h] [rbp-1A8h]
  unsigned int v412; // [rsp+48h] [rbp-1A8h]
  __int64 v413; // [rsp+48h] [rbp-1A8h]
  unsigned __int64 v414; // [rsp+50h] [rbp-1A0h]
  __int64 *v415; // [rsp+50h] [rbp-1A0h]
  __int64 *v416; // [rsp+58h] [rbp-198h]
  __int64 v417; // [rsp+58h] [rbp-198h]
  __int64 v418; // [rsp+58h] [rbp-198h]
  __int64 v419; // [rsp+58h] [rbp-198h]
  int v420; // [rsp+58h] [rbp-198h]
  unsigned int v421; // [rsp+58h] [rbp-198h]
  _DWORD *v422; // [rsp+60h] [rbp-190h]
  __int64 v423; // [rsp+60h] [rbp-190h]
  _DWORD *v424; // [rsp+60h] [rbp-190h]
  unsigned __int64 v425; // [rsp+60h] [rbp-190h]
  unsigned __int64 v426; // [rsp+60h] [rbp-190h]
  __int64 *v427; // [rsp+60h] [rbp-190h]
  __int64 v428; // [rsp+60h] [rbp-190h]
  unsigned int v429; // [rsp+60h] [rbp-190h]
  int v430; // [rsp+60h] [rbp-190h]
  unsigned __int64 v431; // [rsp+68h] [rbp-188h]
  int v432; // [rsp+68h] [rbp-188h]
  unsigned int v433; // [rsp+68h] [rbp-188h]
  int v434; // [rsp+68h] [rbp-188h]
  unsigned int v435; // [rsp+68h] [rbp-188h]
  unsigned __int64 *v436; // [rsp+68h] [rbp-188h]
  __int64 v437; // [rsp+68h] [rbp-188h]
  __int64 v438; // [rsp+68h] [rbp-188h]
  __int64 v439; // [rsp+68h] [rbp-188h]
  _DWORD *v440; // [rsp+70h] [rbp-180h]
  unsigned __int64 v441; // [rsp+70h] [rbp-180h]
  unsigned int v442; // [rsp+70h] [rbp-180h]
  unsigned int v443; // [rsp+70h] [rbp-180h]
  unsigned __int64 v444; // [rsp+70h] [rbp-180h]
  unsigned int v445; // [rsp+70h] [rbp-180h]
  char *v446; // [rsp+70h] [rbp-180h]
  unsigned int v447; // [rsp+70h] [rbp-180h]
  char *v448; // [rsp+70h] [rbp-180h]
  _DWORD *v449; // [rsp+70h] [rbp-180h]
  __int64 v450; // [rsp+78h] [rbp-178h]
  __int64 *v451; // [rsp+78h] [rbp-178h]
  __int64 v452; // [rsp+78h] [rbp-178h]
  __int64 v453; // [rsp+78h] [rbp-178h]
  _QWORD *v454; // [rsp+78h] [rbp-178h]
  int srca; // [rsp+80h] [rbp-170h]
  unsigned __int64 srcb; // [rsp+80h] [rbp-170h]
  unsigned int srcc; // [rsp+80h] [rbp-170h]
  unsigned int srci; // [rsp+80h] [rbp-170h]
  void *srcd; // [rsp+80h] [rbp-170h]
  int srce; // [rsp+80h] [rbp-170h]
  unsigned int srcl; // [rsp+80h] [rbp-170h]
  _DWORD *srck; // [rsp+80h] [rbp-170h]
  _DWORD *srcj; // [rsp+80h] [rbp-170h]
  __int64 *srcf; // [rsp+80h] [rbp-170h]
  void *srch; // [rsp+80h] [rbp-170h]
  _QWORD *srcg; // [rsp+80h] [rbp-170h]
  unsigned int srcm; // [rsp+80h] [rbp-170h]
  char v470; // [rsp+9Fh] [rbp-151h] BYREF
  __int64 v471; // [rsp+A0h] [rbp-150h]
  __int64 v472; // [rsp+A8h] [rbp-148h]
  __int64 v473; // [rsp+B0h] [rbp-140h]
  _BYTE *v474; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v475; // [rsp+C8h] [rbp-128h]
  _BYTE v476[16]; // [rsp+D0h] [rbp-120h] BYREF
  int *v477; // [rsp+E0h] [rbp-110h] BYREF
  unsigned __int64 v478; // [rsp+E8h] [rbp-108h]
  __int64 v479; // [rsp+F0h] [rbp-100h] BYREF
  void *v480; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v481; // [rsp+108h] [rbp-E8h]
  _BYTE v482[32]; // [rsp+110h] [rbp-E0h] BYREF
  __int64 *v483; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v484; // [rsp+138h] [rbp-B8h]
  __int64 v485; // [rsp+140h] [rbp-B0h] BYREF
  unsigned int v486; // [rsp+148h] [rbp-A8h]

  *(_QWORD *)(a1 + 240) = a2;
  v10 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 232) = a2[7];
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_640:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4FCA82C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_640;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4FCA82C);
  v17 = *(_DWORD *)(a1 + 264);
  ++*(_QWORD *)(a1 + 248);
  *(_QWORD *)(a1 + 1424) = v13;
  if ( v17 || (v16 = *(_DWORD *)(a1 + 268)) != 0 )
  {
    v18 = *(_QWORD **)(a1 + 256);
    v14 = 64;
    v19 = &v18[13 * *(unsigned int *)(a1 + 272)];
    v20 = 4 * v17;
    if ( (unsigned int)(4 * v17) < 0x40 )
      v20 = 64;
    if ( *(_DWORD *)(a1 + 272) <= v20 )
    {
      while ( v18 != v19 )
      {
        if ( *v18 != -8 )
        {
          if ( *v18 != -16 )
          {
            _libc_free(v18[10]);
            _libc_free(v18[7]);
            _libc_free(v18[4]);
            _libc_free(v18[1]);
          }
          *v18 = -8;
        }
        v18 += 13;
      }
    }
    else
    {
      do
      {
        if ( *v18 != -8 && *v18 != -16 )
        {
          _libc_free(v18[10]);
          _libc_free(v18[7]);
          _libc_free(v18[4]);
          _libc_free(v18[1]);
        }
        v18 += 13;
      }
      while ( v18 != v19 );
      v75 = *(_DWORD *)(a1 + 272);
      if ( v17 )
      {
        v76 = 64;
        if ( v17 != 1 )
        {
          _BitScanReverse(&v77, v17 - 1);
          v14 = 33 - (v77 ^ 0x1F);
          v76 = (unsigned int)(1 << v14);
          if ( (int)v76 < 64 )
            v76 = 64;
        }
        v78 = *(_QWORD **)(a1 + 256);
        if ( (_DWORD)v76 == v75 )
        {
          *(_QWORD *)(a1 + 264) = 0;
          v372 = &v78[13 * v76];
          do
          {
            if ( v78 )
              *v78 = -8;
            v78 += 13;
          }
          while ( v372 != v78 );
        }
        else
        {
          j___libc_free_0(v78);
          v79 = (((((((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v76 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v76 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 8)
              | (((((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v76 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 4)
              | (((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
              | (4 * (int)v76 / 3u + 1)
              | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1);
          v80 = ((v79 >> 16) | v79) + 1;
          *(_DWORD *)(a1 + 272) = v80;
          v81 = (_QWORD *)sub_22077B0(104 * v80);
          v82 = *(unsigned int *)(a1 + 272);
          *(_QWORD *)(a1 + 264) = 0;
          *(_QWORD *)(a1 + 256) = v81;
          v14 = 3 * v82;
          for ( i = &v81[13 * v82]; i != v81; v81 += 13 )
          {
            if ( v81 )
              *v81 = -8;
          }
        }
        goto LABEL_18;
      }
      if ( v75 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 256));
        *(_QWORD *)(a1 + 256) = 0;
        *(_QWORD *)(a1 + 264) = 0;
        *(_DWORD *)(a1 + 272) = 0;
        goto LABEL_18;
      }
    }
    *(_QWORD *)(a1 + 264) = 0;
  }
LABEL_18:
  v21 = *(_DWORD *)(a1 + 296);
  ++*(_QWORD *)(a1 + 280);
  if ( !v21 )
  {
    v15 = *(_DWORD *)(a1 + 300);
    if ( !v15 )
      goto LABEL_24;
    v22 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)v22 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 288));
      *(_QWORD *)(a1 + 288) = 0;
      *(_QWORD *)(a1 + 296) = 0;
      *(_DWORD *)(a1 + 304) = 0;
      goto LABEL_24;
    }
    goto LABEL_21;
  }
  v14 = 4 * v21;
  v22 = *(unsigned int *)(a1 + 304);
  if ( (unsigned int)(4 * v21) < 0x40 )
    v14 = 64;
  if ( v14 >= (unsigned int)v22 )
  {
LABEL_21:
    v23 = *(_QWORD **)(a1 + 288);
    for ( j = &v23[2 * v22]; j != v23; v23 += 2 )
      *v23 = -8;
    *(_QWORD *)(a1 + 296) = 0;
    goto LABEL_24;
  }
  v62 = *(_QWORD **)(a1 + 288);
  v63 = v21 - 1;
  if ( !v63 )
  {
    v68 = 2048;
    v67 = 128;
LABEL_85:
    j___libc_free_0(v62);
    *(_DWORD *)(a1 + 304) = v67;
    v69 = (_QWORD *)sub_22077B0(v68);
    v70 = *(unsigned int *)(a1 + 304);
    *(_QWORD *)(a1 + 296) = 0;
    *(_QWORD *)(a1 + 288) = v69;
    for ( k = &v69[2 * v70]; k != v69; v69 += 2 )
    {
      if ( v69 )
        *v69 = -8;
    }
    goto LABEL_24;
  }
  _BitScanReverse(&v63, v63);
  v14 = 33 - (v63 ^ 0x1F);
  v64 = (unsigned int)(1 << (33 - (v63 ^ 0x1F)));
  if ( (int)v64 < 64 )
    v64 = 64;
  if ( (_DWORD)v64 != (_DWORD)v22 )
  {
    v65 = (4 * (int)v64 / 3u + 1) | ((unsigned __int64)(4 * (int)v64 / 3u + 1) >> 1);
    v66 = ((v65 | (v65 >> 2)) >> 4) | v65 | (v65 >> 2) | ((((v65 | (v65 >> 2)) >> 4) | v65 | (v65 >> 2)) >> 8);
    v67 = (v66 | (v66 >> 16)) + 1;
    v68 = 16 * ((v66 | (v66 >> 16)) + 1);
    goto LABEL_85;
  }
  v364 = &v62[2 * v64];
  *(_QWORD *)(a1 + 296) = 0;
  do
  {
    if ( v62 )
      *v62 = -8;
    v62 += 2;
  }
  while ( v364 != v62 );
LABEL_24:
  v25 = *(_QWORD *)(a1 + 392);
  *(_DWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 1440) = 0;
  v26 = v25 + 8LL * *(unsigned int *)(a1 + 400);
  while ( v25 != v26 )
  {
    while ( 1 )
    {
      v27 = *(unsigned __int64 **)(v26 - 8);
      v26 -= 8;
      if ( !v27 )
        break;
      sub_1DB4CE0((__int64)v27);
      v28 = v27[12];
      if ( v28 )
      {
        v29 = *(_QWORD *)(v28 + 16);
        while ( v29 )
        {
          sub_1F21070(*(_QWORD *)(v29 + 24));
          v30 = v29;
          v29 = *(_QWORD *)(v29 + 16);
          j_j___libc_free_0(v30, 56);
        }
        j_j___libc_free_0(v28, 48);
      }
      v31 = v27[8];
      if ( (unsigned __int64 *)v31 != v27 + 10 )
        _libc_free(v31);
      if ( (unsigned __int64 *)*v27 != v27 + 2 )
        _libc_free(*v27);
      j_j___libc_free_0(v27, 120);
      if ( v25 == v26 )
        goto LABEL_36;
    }
  }
LABEL_36:
  v32 = *(unsigned __int64 **)(a1 + 536);
  *(_DWORD *)(a1 + 400) = 0;
  v33 = &v32[6 * *(unsigned int *)(a1 + 544)];
  while ( v32 != v33 )
  {
    v33 -= 6;
    if ( (unsigned __int64 *)*v33 != v33 + 2 )
      _libc_free(*v33);
  }
  v34 = *(unsigned int *)(a1 + 1392);
  v35 = *(unsigned __int64 **)(a1 + 1384);
  *(_DWORD *)(a1 + 544) = 0;
  v36 = &v35[2 * v34];
  while ( v36 != v35 )
  {
    v37 = *v35;
    v35 += 2;
    _libc_free(v37);
  }
  *(_DWORD *)(a1 + 1392) = 0;
  v38 = *(unsigned int *)(a1 + 1344);
  if ( (_DWORD)v38 )
  {
    v39 = *(_QWORD **)(a1 + 1336);
    *(_QWORD *)(a1 + 1400) = 0;
    v40 = *v39;
    v41 = &v39[v38];
    v42 = v39 + 1;
    *(_QWORD *)(a1 + 1320) = v40;
    *(_QWORD *)(a1 + 1328) = v40 + 4096;
    while ( v41 != v42 )
    {
      v43 = *v42++;
      _libc_free(v43);
    }
    *(_DWORD *)(a1 + 1344) = 1;
  }
  m = *(_QWORD *)(a1 + 232);
  v45 = -858993459 * ((__int64)(*(_QWORD *)(m + 16) - *(_QWORD *)(m + 8)) >> 3) - *(_DWORD *)(m + 32);
  v405 = v45;
  if ( !v45 )
    return 0;
  v481 = 0x800000000LL;
  v480 = v482;
  v414 = v45;
  if ( v45 > 8 )
    sub_16CD150((__int64)&v480, v482, v45, 4, v15, v16);
  v450 = a1 + 392;
  if ( v414 > *(unsigned int *)(a1 + 404) )
    sub_1F232A0(v450, v414);
  v46 = *(unsigned int *)(a1 + 544);
  if ( v414 < v46 )
  {
    v72 = *(_QWORD *)(a1 + 536);
    v73 = (unsigned __int64 *)(v72 + 48 * v46);
    v74 = (unsigned __int64 *)(v72 + 48 * v414);
    while ( v74 != v73 )
    {
      v73 -= 6;
      if ( (unsigned __int64 *)*v73 != v73 + 2 )
        _libc_free(*v73);
    }
    *(_DWORD *)(a1 + 544) = v405;
  }
  else if ( v414 > v46 )
  {
    if ( v414 > *(unsigned int *)(a1 + 548) )
    {
      sub_1F234A0(a1 + 536, v414);
      v46 = *(unsigned int *)(a1 + 544);
    }
    v47 = *(_QWORD *)(a1 + 536);
    v48 = v47 + 48 * v46;
    for ( m = v47 + 48 * v414; m != v48; v48 += 48 )
    {
      if ( v48 )
      {
        v14 = v48 + 16;
        *(_DWORD *)(v48 + 8) = 0;
        *(_QWORD *)v48 = v48 + 16;
        *(_DWORD *)(v48 + 12) = 4;
      }
    }
    *(_DWORD *)(a1 + 544) = v405;
  }
  v49 = sub_1F23E90(a1, v405, m, v14, v15, v16);
  v50 = *(_QWORD *)(a1 + 232);
  v51 = *(_QWORD *)(v50 + 8);
  v52 = *(_QWORD *)(v50 + 16);
  v53 = *(_DWORD *)(v50 + 32);
  v54 = 0;
  v55 = -858993459 * ((v52 - v51) >> 3);
  if ( (int)(v55 - v53) <= 0 )
    goto LABEL_65;
  do
  {
    v56 = v53++;
    v54 += *(_DWORD *)(v51 + 40 * v56 + 8);
  }
  while ( v55 != v53 );
  if ( v49 <= 1 || v54 <= 0xF || byte_4FCAAA0 || (unsigned __int8)sub_1636880(a1, *a2) )
  {
LABEL_65:
    v57 = 0;
    v58 = *(__int64 **)(a1 + 1432);
    v59 = &v58[*(unsigned int *)(a1 + 1440)];
    if ( v58 == v59 )
    {
      v385 = 0;
    }
    else
    {
      do
      {
        v60 = *v58++;
        ++v57;
        sub_1E16240(v60);
      }
      while ( v59 != v58 );
      v385 = v57 != 0;
    }
    *(_DWORD *)(a1 + 1440) = 0;
    goto LABEL_69;
  }
  v84 = 0;
  do
  {
    v85 = sub_22077B0(120);
    v86 = v85;
    if ( v85 )
    {
      *(_QWORD *)(v85 + 8) = 0x200000000LL;
      v87 = 0;
      *(_QWORD *)v85 = v85 + 16;
      *(_QWORD *)(v85 + 64) = v85 + 80;
      *(_QWORD *)(v85 + 72) = 0x200000000LL;
      *(_QWORD *)(v85 + 96) = 0;
      *(_QWORD *)(v85 + 104) = 0;
      *(_DWORD *)(v85 + 112) = v84;
      *(_DWORD *)(v85 + 116) = 0;
    }
    else
    {
      v87 = MEMORY[0x48];
    }
    srca = v87;
    v88 = *(_QWORD *)(*(_QWORD *)(a1 + 1424) + 344LL) & 0xFFFFFFFFFFFFFFF9LL;
    v89 = sub_145CDC0(0x10u, (__int64 *)(a1 + 1320));
    if ( v89 )
    {
      *(_QWORD *)(v89 + 8) = v88;
      *(_DWORD *)v89 = srca;
    }
    v92 = *(unsigned int *)(v86 + 72);
    if ( (unsigned int)v92 >= *(_DWORD *)(v86 + 76) )
    {
      srch = (void *)v89;
      sub_16CD150(v86 + 64, (const void *)(v86 + 80), 0, 8, v90, v91);
      v92 = *(unsigned int *)(v86 + 72);
      v89 = (__int64)srch;
    }
    *(_QWORD *)(*(_QWORD *)(v86 + 64) + 8 * v92) = v89;
    v93 = *(_DWORD *)(a1 + 400);
    ++*(_DWORD *)(v86 + 72);
    if ( v93 >= *(_DWORD *)(a1 + 404) )
    {
      sub_1F232A0(v450, 0);
      v93 = *(_DWORD *)(a1 + 400);
    }
    v94 = (__int64 *)(*(_QWORD *)(a1 + 392) + 8LL * v93);
    if ( v94 )
    {
      *v94 = v86;
      v93 = *(_DWORD *)(a1 + 400);
      v86 = 0;
    }
    *(_DWORD *)(a1 + 400) = v93 + 1;
    v95 = (unsigned int)v481;
    if ( (unsigned int)v481 >= HIDWORD(v481) )
    {
      sub_16CD150((__int64)&v480, v482, 0, 4, v90, v91);
      v95 = (unsigned int)v481;
    }
    *((_DWORD *)v480 + v95) = v84;
    LODWORD(v481) = v481 + 1;
    if ( v86 )
    {
      sub_1DB4CE0(v86);
      v96 = *(_QWORD *)(v86 + 96);
      if ( v96 )
      {
        sub_1F21070(*(_QWORD *)(v96 + 16));
        j_j___libc_free_0(v96, 48);
      }
      v97 = *(_QWORD *)(v86 + 64);
      if ( v97 != v86 + 80 )
        _libc_free(v97);
      if ( *(_QWORD *)v86 != v86 + 16 )
        _libc_free(*(_QWORD *)v86);
      j_j___libc_free_0(v86, 120);
    }
    ++v84;
  }
  while ( v84 != v405 );
  v404 = 0;
  do
  {
    ++v404;
    v98 = *(__int64 **)(a1 + 312);
    v416 = &v98[*(unsigned int *)(a1 + 320)];
    if ( v98 == v416 )
      break;
    v451 = *(__int64 **)(a1 + 312);
    v379 = 0;
    do
    {
      v99 = *v451;
      v100 = *(_QWORD *)(a1 + 256);
      v101 = *(unsigned int *)(a1 + 272);
      if ( !(_DWORD)v101 )
        goto LABEL_252;
      LODWORD(v102) = (v101 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
      v103 = 1;
      v104 = (_DWORD *)(v100 + 104LL * (unsigned int)v102);
      v105 = *(_QWORD *)v104;
      if ( v99 != *(_QWORD *)v104 )
      {
        while ( v105 != -8 )
        {
          LODWORD(v102) = (v101 - 1) & (v103 + v102);
          v104 = (_DWORD *)(v100 + 104LL * (unsigned int)v102);
          v105 = *(_QWORD *)v104;
          if ( v99 == *(_QWORD *)v104 )
            goto LABEL_134;
          ++v103;
        }
LABEL_252:
        v106 = *(_QWORD **)(v99 + 64);
        v107 = *(_QWORD **)(v99 + 72);
        LODWORD(v102) = 13 * v101;
        v104 = (_DWORD *)(v100 + 104LL * (unsigned int)v101);
        if ( v106 == v107 )
        {
LABEL_253:
          LODWORD(v115) = 0;
          v108 = 0;
LABEL_254:
          v431 = 0;
          v125 = 0;
          v124 = 0;
          goto LABEL_156;
        }
        goto LABEL_135;
      }
LABEL_134:
      v106 = *(_QWORD **)(v99 + 64);
      v107 = *(_QWORD **)(v99 + 72);
      if ( v106 == v107 )
        goto LABEL_253;
LABEL_135:
      v422 = v104;
      v108 = 0;
      v109 = a1;
      srcb = 0;
      LODWORD(v102) = 0;
      v110 = v107;
      while ( (_DWORD)v101 )
      {
        v111 = 1;
        v112 = (v101 - 1) & (((unsigned int)*v106 >> 9) ^ ((unsigned int)*v106 >> 4));
        v113 = (__int64 *)(v100 + 104LL * v112);
        v114 = *v113;
        if ( *v106 != *v113 )
        {
          while ( v114 != -8 )
          {
            v112 = (v101 - 1) & (v111 + v112);
            v113 = (__int64 *)(v100 + 104LL * v112);
            v114 = *v113;
            if ( *v106 == *v113 )
              goto LABEL_138;
            ++v111;
          }
          break;
        }
LABEL_138:
        v115 = (unsigned int)(v102 + 63) >> 6;
        if ( v113 != (__int64 *)(v100 + 104 * v101) )
        {
          v116 = *((_DWORD *)v113 + 24);
          v117 = (v116 + 63) >> 6;
          v118 = v117;
          if ( (unsigned int)v102 >= v116 )
          {
LABEL_140:
            if ( v118 )
            {
              v119 = v113[10];
              for ( n = 0; n != v118; ++n )
                v108[n] |= *(_QWORD *)(v119 + 8 * n);
            }
            goto LABEL_143;
          }
          v184 = v102 & 0x3F;
          if ( v116 > srcb << 6 )
          {
            v380 = v109;
            v381 = v110;
            v185 = 2 * srcb;
            v382 = *((_DWORD *)v113 + 24);
            v387 = v184;
            if ( 2 * srcb < v117 )
              v185 = (v116 + 63) >> 6;
            v395 = (v116 + 63) >> 6;
            v441 = v185;
            v407 = 8 * v185;
            v186 = realloc((unsigned __int64)v108, 8 * v185, 8 * (int)v185, v184, (int)v110, v117);
            v117 = v395;
            v184 = v387;
            v116 = v382;
            v108 = v186;
            v110 = v381;
            v109 = v380;
            if ( !v186 )
            {
              if ( v407
                || (v374 = malloc(1u),
                    v117 = v395,
                    v184 = v387,
                    v116 = v382,
                    v108 = (_QWORD *)v374,
                    v110 = v381,
                    v109 = v380,
                    !v374) )
              {
                sub_16BD1C0("Allocation failed", 1u);
                v117 = v395;
                v184 = v387;
                v116 = v382;
                v110 = v381;
                v109 = v380;
                v118 = (unsigned int)(*((_DWORD *)v113 + 24) + 63) >> 6;
              }
            }
            if ( (unsigned int)v115 < v441 )
            {
              v383 = v109;
              v389 = v110;
              v397 = v116;
              v409 = v184;
              v433 = v117;
              memset(&v108[(unsigned int)v115], 0, 8 * (v441 - (unsigned int)v115));
              v117 = v433;
              v184 = v409;
              v116 = v397;
              v110 = v389;
              v109 = v383;
            }
            if ( v184 )
              v108[(unsigned int)(v115 - 1)] &= ~(-1LL << v184);
            v187 = v441 - (unsigned int)srcb;
            if ( v441 != (unsigned int)srcb )
            {
              v189 = (char *)&v108[(unsigned int)srcb];
              v434 = v184;
              v390 = v109;
              v398 = v110;
              v410 = v116;
              srcl = v117;
              memset(v189, 0, 8 * v187);
              v109 = v390;
              v110 = v398;
              v116 = v410;
              v184 = v434;
              v117 = srcl;
            }
            srcb = v441;
            if ( v441 <= (unsigned int)v115 )
              goto LABEL_238;
          }
          else if ( srcb <= (unsigned int)v115 )
          {
LABEL_238:
            if ( v184 )
              v108[(unsigned int)(v115 - 1)] &= ~(-1LL << v184);
            v115 = v117;
            LODWORD(v102) = v116;
            goto LABEL_140;
          }
          v388 = v109;
          v396 = v110;
          v408 = v116;
          v432 = v184;
          v442 = v117;
          memset(&v108[(unsigned int)v115], 0, 8 * (srcb - (unsigned int)v115));
          v117 = v442;
          v184 = v432;
          v116 = v408;
          v110 = v396;
          v109 = v388;
          goto LABEL_238;
        }
LABEL_143:
        if ( v110 == ++v106 )
          goto LABEL_149;
LABEL_144:
        v100 = *(_QWORD *)(v109 + 256);
        v101 = *(unsigned int *)(v109 + 272);
      }
      ++v106;
      v115 = (unsigned int)(v102 + 63) >> 6;
      if ( v110 != v106 )
        goto LABEL_144;
LABEL_149:
      srcc = v102;
      v104 = v422;
      if ( !(_DWORD)v102 )
        goto LABEL_254;
      v431 = (unsigned int)v115;
      v121 = malloc(8LL * (unsigned int)v115);
      v122 = v422;
      v123 = srcc;
      v124 = (char *)v121;
      if ( !v121 )
      {
        if ( 8LL * (unsigned int)v115 || (v375 = malloc(1u), v123 = srcc, v122 = v422, !v375) )
        {
          v449 = v122;
          srcm = v123;
          sub_16BD1C0("Allocation failed", 1u);
          v123 = srcm;
          v122 = v449;
        }
        else
        {
          v124 = (char *)v375;
        }
      }
      v440 = v122;
      srci = v123;
      memcpy(v124, v108, 8LL * (unsigned int)v115);
      v104 = v440;
      v125 = srci;
      v102 = (unsigned int)(v440[12] + 63) >> 6;
      if ( (unsigned int)v102 > (unsigned int)v115 )
        v102 = v115;
      if ( (_DWORD)v102 )
      {
        v126 = *((_QWORD *)v440 + 4);
        for ( ii = 0; ii != v102; ++ii )
          *(_QWORD *)&v124[8 * ii] &= ~*(_QWORD *)(v126 + 8 * ii);
      }
LABEL_156:
      v128 = v104[6];
      v129 = (v128 + 63) >> 6;
      v130 = v129;
      if ( v125 >= v128 )
      {
        v128 = v125;
        v129 = (v125 + 63) >> 6;
        goto LABEL_158;
      }
      v188 = (unsigned int)v115;
      srce = v125 & 0x3F;
      if ( v128 > v431 << 6 )
      {
        v384 = v104;
        v391 = v104[6];
        v190 = 2 * v431;
        v399 = v125;
        if ( 2 * v431 < v129 )
          v190 = (v128 + 63) >> 6;
        v444 = v190;
        v423 = 8 * v190;
        v191 = realloc((unsigned __int64)v124, 8 * v190, 8 * (int)v190, v102, v125, v128);
        v188 = (unsigned int)v115;
        v125 = v399;
        v124 = v191;
        v128 = v391;
        v104 = v384;
        if ( !v191 )
        {
          if ( v423 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v104 = v384;
            v128 = v391;
            v125 = v399;
            v188 = (unsigned int)v115;
            v130 = (unsigned int)(v384[6] + 63) >> 6;
          }
          else
          {
            v373 = malloc(1u);
            v188 = (unsigned int)v115;
            v125 = v399;
            v128 = v391;
            v104 = v384;
            v124 = (char *)v373;
            if ( !v373 )
            {
              sub_16BD1C0("Allocation failed", 1u);
              v104 = v384;
              v188 = (unsigned int)v115;
              v125 = v399;
              v128 = v391;
              v130 = (unsigned int)(v384[6] + 63) >> 6;
            }
          }
        }
        if ( v188 < v444 && v444 != v188 )
        {
          v393 = v104;
          v401 = v128;
          v412 = v125;
          v426 = v188;
          memset(&v124[8 * v188], 0, 8 * (v444 - v188));
          v104 = v393;
          v128 = v401;
          v125 = v412;
          v188 = v426;
        }
        if ( srce )
        {
          LODWORD(v102) = srce;
          *(_QWORD *)&v124[8 * (unsigned int)(v115 - 1)] &= ~(-1LL << srce);
        }
        if ( v444 != v431 )
        {
          v392 = v104;
          v400 = v128;
          v411 = v125;
          v425 = v188;
          memset(&v124[8 * v431], 0, 8 * (v444 - v431));
          v104 = v392;
          v128 = v400;
          v125 = v411;
          v188 = v425;
        }
        v431 = v444;
        if ( v188 >= v444 )
          goto LABEL_268;
LABEL_279:
        v192 = v431 - v188;
        if ( v431 != v188 )
        {
          v424 = v104;
          v435 = v128;
          v445 = v125;
          memset(&v124[8 * v188], 0, 8 * v192);
          v104 = v424;
          v128 = v435;
          v125 = v445;
        }
        goto LABEL_268;
      }
      if ( (unsigned int)v115 < v431 )
        goto LABEL_279;
LABEL_268:
      if ( srce )
      {
        LODWORD(v102) = srce;
        *(_QWORD *)&v124[8 * (unsigned int)(v115 - 1)] &= ~(-1LL << srce);
      }
LABEL_158:
      if ( v130 )
      {
        v102 = *((_QWORD *)v104 + 1);
        for ( jj = 0; jj != v130; ++jj )
          *(_QWORD *)&v124[8 * jj] |= *(_QWORD *)(v102 + 8 * jj);
      }
      v132 = v104[18];
      v133 = (v132 + 63) >> 6;
      if ( v133 > (unsigned int)v115 )
        v133 = v115;
      if ( v133 )
      {
        v102 = *((_QWORD *)v104 + 7);
        v134 = v133 + 1;
        v135 = 1;
        while ( (v108[v135 - 1] & ~*(_QWORD *)(v102 + 8 * v135 - 8)) == 0 )
        {
          v133 = v135++;
          if ( v134 == v135 )
            goto LABEL_243;
        }
LABEL_167:
        if ( v125 > v132 )
        {
          v443 = v128;
          srcj = v104;
          sub_13A49F0((__int64)(v104 + 14), v125, 0, v102, v125, v128);
          v128 = v443;
          v104 = srcj;
        }
        v136 = 0;
        if ( (_DWORD)v115 )
        {
          do
          {
            v137 = v108[v136];
            v138 = (_QWORD *)(v136 * 8 + *((_QWORD *)v104 + 7));
            ++v136;
            *v138 |= v137;
          }
          while ( (unsigned int)v115 != v136 );
        }
        v379 = 1;
      }
      else
      {
LABEL_243:
        while ( (_DWORD)v115 != v133 )
        {
          if ( v108[v133] )
            goto LABEL_167;
          ++v133;
        }
      }
      v139 = v104[24];
      v140 = (v139 + 63) >> 6;
      if ( v140 > v129 )
        v140 = v129;
      if ( v140 )
      {
        v141 = 0;
        while ( (*(_QWORD *)&v124[8 * v141] & ~*(_QWORD *)(*((_QWORD *)v104 + 10) + 8 * v141)) == 0 )
        {
          if ( v140 == ++v141 )
            goto LABEL_247;
        }
LABEL_178:
        if ( v139 < v128 )
        {
          srck = v104;
          sub_13A49F0((__int64)(v104 + 20), v128, 0, v140, v139, v128);
          v104 = srck;
        }
        v142 = 0;
        if ( v129 )
        {
          do
          {
            v143 = *(_QWORD *)&v124[v142];
            v144 = (_QWORD *)(v142 + *((_QWORD *)v104 + 10));
            v142 += 8;
            *v144 |= v143;
          }
          while ( 8LL * v129 != v142 );
        }
        v379 = 1;
      }
      else
      {
LABEL_247:
        while ( v129 != v140 )
        {
          if ( *(_QWORD *)&v124[8 * v140] )
            goto LABEL_178;
          ++v140;
        }
      }
      _libc_free((unsigned __int64)v124);
      _libc_free((unsigned __int64)v108);
      ++v451;
    }
    while ( v416 != v451 );
  }
  while ( v379 );
  v145 = a1;
  HIDWORD(v484) = 16;
  v475 = 0x1000000000LL;
  v146 = *(_QWORD *)(a1 + 240);
  *(_DWORD *)(a1 + 1560) = v404;
  v483 = &v485;
  v147 = *(_QWORD *)(v146 + 328);
  v474 = v476;
  v406 = v146 + 320;
  if ( v147 == v146 + 320 )
    goto LABEL_312;
  v386 = a1 + 248;
  v417 = 8 * v414;
  v148 = 16;
  while ( 2 )
  {
    LODWORD(v484) = 0;
    v149 = v148;
    v150 = 0;
    if ( v414 > v149 )
    {
      sub_16CD150((__int64)&v483, &v485, v414, 8, v90, v91);
      v150 = (unsigned int)v484;
    }
    v151 = &v483[v150];
    for ( kk = &v483[(unsigned __int64)v417 / 8]; kk != v151; ++v151 )
    {
      if ( v151 )
        *v151 = 0;
    }
    LODWORD(v475) = 0;
    LODWORD(v484) = v405;
    if ( v414 > HIDWORD(v475) )
    {
      sub_16CD150((__int64)&v474, v476, v414, 1, v90, v91);
      v153 = (unsigned int)v475;
    }
    else
    {
      v153 = 0;
    }
    v154 = &v474[v153];
    for ( mm = &v474[v414]; mm != v154; ++v154 )
    {
      if ( v154 )
        *v154 = 0;
    }
    v156 = *(_DWORD *)(v145 + 272);
    LODWORD(v475) = v405;
    if ( !v156 )
    {
      ++*(_QWORD *)(v145 + 248);
      goto LABEL_425;
    }
    v90 = *(_QWORD *)(v145 + 256);
    v157 = (v156 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
    v158 = v90 + 104LL * v157;
    v159 = *(_QWORD *)v158;
    if ( v147 != *(_QWORD *)v158 )
    {
      v311 = 1;
      v312 = 0;
      while ( v159 != -8 )
      {
        if ( !v312 && v159 == -16 )
          v312 = v158;
        v91 = (unsigned int)(v311 + 1);
        v157 = (v156 - 1) & (v157 + v311);
        v158 = v90 + 104LL * v157;
        v159 = *(_QWORD *)v158;
        if ( v147 == *(_QWORD *)v158 )
          goto LABEL_201;
        ++v311;
      }
      v313 = *(_DWORD *)(v145 + 264);
      if ( v312 )
        v158 = v312;
      ++*(_QWORD *)(v145 + 248);
      v314 = v313 + 1;
      if ( 4 * v314 < 3 * v156 )
      {
        if ( v156 - *(_DWORD *)(v145 + 268) - v314 <= v156 >> 3 )
        {
          sub_1F21E90(v386, v156);
          v346 = *(_DWORD *)(v145 + 272);
          if ( !v346 )
          {
LABEL_637:
            ++*(_DWORD *)(a1 + 264);
            BUG();
          }
          v347 = v346 - 1;
          v90 = *(_QWORD *)(v145 + 256);
          v348 = 1;
          v349 = v347 & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
          v350 = 0;
          v158 = v90 + 104LL * v349;
          v351 = *(_QWORD *)v158;
          v314 = *(_DWORD *)(v145 + 264) + 1;
          if ( v147 != *(_QWORD *)v158 )
          {
            while ( v351 != -8 )
            {
              if ( !v350 && v351 == -16 )
                v350 = v158;
              v91 = (unsigned int)(v348 + 1);
              v349 = v347 & (v348 + v349);
              v158 = v90 + 104LL * v349;
              v351 = *(_QWORD *)v158;
              if ( v147 == *(_QWORD *)v158 )
                goto LABEL_421;
              ++v348;
            }
            if ( v350 )
              v158 = v350;
          }
        }
        goto LABEL_421;
      }
LABEL_425:
      sub_1F21E90(v386, 2 * v156);
      v315 = *(_DWORD *)(v145 + 272);
      if ( !v315 )
        goto LABEL_637;
      v90 = (unsigned int)(v315 - 1);
      v316 = *(_QWORD *)(v145 + 256);
      v317 = v90 & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
      v158 = v316 + 104LL * v317;
      v318 = *(_QWORD *)v158;
      v314 = *(_DWORD *)(v145 + 264) + 1;
      if ( v147 != *(_QWORD *)v158 )
      {
        v319 = 1;
        v320 = 0;
        while ( v318 != -8 )
        {
          if ( !v320 && v318 == -16 )
            v320 = v158;
          v91 = (unsigned int)(v319 + 1);
          v317 = v90 & (v319 + v317);
          v158 = v316 + 104LL * v317;
          v318 = *(_QWORD *)v158;
          if ( v147 == *(_QWORD *)v158 )
            goto LABEL_421;
          ++v319;
        }
        if ( v320 )
          v158 = v320;
      }
LABEL_421:
      *(_DWORD *)(v145 + 264) = v314;
      if ( *(_QWORD *)v158 != -8 )
        --*(_DWORD *)(v145 + 268);
      *(_QWORD *)v158 = v147;
      memset((void *)(v158 + 8), 0, 0x60u);
      goto LABEL_220;
    }
LABEL_201:
    v160 = *(_DWORD *)(v158 + 72);
    if ( v160 )
    {
      v90 = (unsigned int)(v160 - 1) >> 6;
      v161 = 0;
      while ( 1 )
      {
        _RDX = *(_QWORD *)(*(_QWORD *)(v158 + 56) + 8 * v161);
        if ( v90 == v161 )
          break;
        if ( _RDX )
          goto LABEL_207;
        if ( (_DWORD)v90 + 1 == ++v161 )
          goto LABEL_220;
      }
      _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v160;
      if ( !_RDX )
        goto LABEL_220;
LABEL_207:
      __asm { tzcnt   rdx, rdx }
      for ( nn = _RDX + ((_DWORD)v161 << 6); nn != -1; nn = ((_DWORD)v171 << 6) + _RAX )
      {
        v165 = nn;
        v166 = nn + 1;
        v483[v165] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v145 + 1424) + 392LL) + 16LL * *(unsigned int *)(v147 + 48));
        v167 = *(_DWORD *)(v158 + 72);
        if ( v167 == v166 )
          break;
        v90 = v166 >> 6;
        v168 = (unsigned int)(v167 - 1) >> 6;
        if ( (unsigned int)v90 > v168 )
          break;
        v169 = 64 - (v166 & 0x3F);
        v91 = *(_QWORD *)(v158 + 56);
        v170 = 0xFFFFFFFFFFFFFFFFLL >> v169;
        v171 = (unsigned int)v90;
        if ( v169 == 64 )
          v170 = 0;
        v172 = ~v170;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v91 + 8 * v171);
          if ( (_DWORD)v90 == (_DWORD)v171 )
            _RAX = v172 & *(_QWORD *)(v91 + 8 * v171);
          if ( v168 == (_DWORD)v171 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v158 + 72);
          if ( _RAX )
            break;
          if ( v168 < (unsigned int)++v171 )
            goto LABEL_220;
        }
        __asm { tzcnt   rax, rax }
      }
    }
LABEL_220:
    v175 = *(_QWORD *)(v147 + 32);
    srcd = (void *)(v147 + 24);
    v176 = &v479;
    v177 = (__int64)&v470;
    if ( v175 != v147 + 24 )
    {
      v394 = v147;
      v178 = &v470;
      v179 = (unsigned __int64 *)&v477;
      while ( 1 )
      {
        v177 = v175;
        v477 = (int *)v176;
        v478 = 0x400000000LL;
        v470 = 0;
        if ( (unsigned __int8)sub_1F216F0(v145, v175, (__int64)v179, v178, v90, v91) )
          break;
        if ( v477 != (int *)v176 )
          _libc_free((unsigned __int64)v477);
        if ( !v175 )
          BUG();
LABEL_227:
        if ( (*(_BYTE *)v175 & 4) != 0 )
        {
          v175 = *(_QWORD *)(v175 + 8);
          if ( srcd == (void *)v175 )
            goto LABEL_229;
        }
        else
        {
          while ( (*(_BYTE *)(v175 + 46) & 8) != 0 )
            v175 = *(_QWORD *)(v175 + 8);
          v175 = *(_QWORD *)(v175 + 8);
          if ( srcd == (void *)v175 )
          {
LABEL_229:
            v147 = v394;
            goto LABEL_230;
          }
        }
      }
      v193 = *(_QWORD *)(v145 + 1424);
      v194 = v175;
      if ( (*(_BYTE *)(v175 + 46) & 4) != 0 )
      {
        do
          v194 = *(_QWORD *)v194 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v194 + 46) & 4) != 0 );
      }
      v195 = *(_QWORD *)(v193 + 368);
      v196 = *(unsigned int *)(v193 + 384);
      if ( (_DWORD)v196 )
      {
        v177 = ((_DWORD)v196 - 1) & (((unsigned int)v194 >> 9) ^ ((unsigned int)v194 >> 4));
        v197 = (__int64 *)(v195 + 16 * v177);
        v198 = *v197;
        if ( *v197 == v194 )
          goto LABEL_290;
        v213 = 1;
        while ( v198 != -8 )
        {
          v91 = (unsigned int)(v213 + 1);
          v177 = ((_DWORD)v196 - 1) & (unsigned int)(v213 + v177);
          v197 = (__int64 *)(v195 + 16LL * (unsigned int)v177);
          v198 = *v197;
          if ( *v197 == v194 )
            goto LABEL_290;
          v213 = v91;
        }
      }
      v197 = (__int64 *)(v195 + 16LL * (unsigned int)v196);
LABEL_290:
      v90 = (unsigned __int64)v477;
      v199 = (unsigned __int64)&v477[(unsigned int)v478];
      if ( v477 == (int *)v199 )
      {
LABEL_300:
        if ( (__int64 *)v199 != v176 )
          _libc_free(v199);
        goto LABEL_227;
      }
      v452 = v175;
      v200 = v477;
      v446 = v178;
      v201 = &v477[(unsigned int)v478];
      v436 = v179;
      v202 = v197[1];
      v427 = v176;
      v203 = v145;
      while ( 2 )
      {
        while ( 1 )
        {
          v205 = *v200;
          v206 = v205;
          if ( v470 )
            break;
          v207 = v483[v205];
          if ( (v207 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_295;
          v208 = *(_QWORD *)(v203 + 392);
          ++v200;
          v472 = v202;
          v471 = v207;
          v209 = *(_QWORD *)(v208 + 8 * v205);
          v473 = **(_QWORD **)(v209 + 64);
          *((_QWORD *)&v378 + 1) = v202;
          *(_QWORD *)&v378 = v207;
          sub_1DB8610(v209, v177, v473, v196, v90, v91, v378, v473);
          v483[v205] = 0;
          v474[v205] = 0;
          if ( v201 == v200 )
          {
LABEL_299:
            v145 = v203;
            v175 = v452;
            v178 = v446;
            v179 = v436;
            v176 = v427;
            v199 = (unsigned __int64)v477;
            goto LABEL_300;
          }
        }
        if ( !v474[v205] )
        {
          v210 = *(_QWORD *)(v203 + 536) + 48 * v205;
          v211 = *(unsigned int *)(v210 + 8);
          if ( (unsigned int)v211 >= *(_DWORD *)(v210 + 12) )
          {
            v177 = v210 + 16;
            sub_16CD150(v210, (const void *)(v210 + 16), 0, 8, v90, v91);
            v206 = v205;
            v211 = *(unsigned int *)(v210 + 8);
          }
          v196 = *(_QWORD *)v210;
          *(_QWORD *)(*(_QWORD *)v210 + 8 * v211) = v202;
          v212 = (unsigned __int64)v474;
          ++*(_DWORD *)(v210 + 8);
          *(_BYTE *)(v212 + v205) = 1;
        }
        v204 = &v483[v206];
        if ( (*v204 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          *v204 = v202;
LABEL_295:
        if ( v201 == ++v200 )
          goto LABEL_299;
        continue;
      }
    }
LABEL_230:
    v180 = 0;
    do
    {
      v181 = v483[v180 / 8];
      if ( (v181 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v182 = *(_QWORD *)(*(_QWORD *)(v145 + 392) + v180);
        v183 = **(_QWORD **)(v182 + 64);
        v478 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v145 + 1424) + 392LL) + 16LL * *(unsigned int *)(v147 + 48) + 8);
        v477 = (int *)v181;
        v479 = v183;
        sub_1DB8610(v182, v177, v183, v478, v90, v91, __PAIR128__(v478, v181), v183);
      }
      v180 += 8LL;
    }
    while ( v180 != v417 );
    v147 = *(_QWORD *)(v147 + 8);
    if ( v406 != v147 )
    {
      v148 = HIDWORD(v484);
      continue;
    }
    break;
  }
  if ( v474 != v476 )
    _libc_free((unsigned __int64)v474);
  if ( v483 != &v485 )
    _libc_free((unsigned __int64)v483);
LABEL_312:
  if ( byte_4FCA9C0 )
  {
    v289 = *(_QWORD *)(a1 + 240);
    v454 = (_QWORD *)(v289 + 320);
    srcg = *(_QWORD **)(v289 + 328);
    if ( srcg != (_QWORD *)(v289 + 320) )
    {
      while ( 1 )
      {
        v290 = srcg[4];
        v291 = (char *)(srcg + 3);
        if ( (_QWORD *)v290 != srcg + 3 )
          break;
LABEL_380:
        srcg = (_QWORD *)srcg[1];
        if ( v454 == srcg )
          goto LABEL_313;
      }
      while ( 1 )
      {
        v292 = *(unsigned __int16 **)(v290 + 16);
        v293 = *v292;
        if ( (unsigned int)(v293 - 17) > 1 && (unsigned __int16)(v293 - 12) > 1u )
        {
          if ( (_WORD)v293 == 1 )
          {
            v294 = *(_QWORD *)(v290 + 32);
            if ( (*(_BYTE *)(v294 + 64) & 8) != 0 )
              goto LABEL_398;
          }
          v295 = *(_WORD *)(v290 + 46);
          if ( (v295 & 4) != 0 || (v295 & 8) == 0 )
            v296 = (*((_QWORD *)v292 + 1) >> 16) & 1LL;
          else
            LOBYTE(v296) = sub_1E15D00(v290, 0x10000u, 1);
          if ( (_BYTE)v296 )
            break;
          v297 = *(_QWORD *)(v290 + 16);
          if ( *(_WORD *)v297 == 1 )
          {
            v294 = *(_QWORD *)(v290 + 32);
            if ( (*(_BYTE *)(v294 + 64) & 0x10) != 0 )
              goto LABEL_398;
          }
          v298 = *(_WORD *)(v290 + 46);
          if ( (v298 & 4) != 0 || (v298 & 8) == 0 )
            v299 = (*(_QWORD *)(v297 + 8) & 0x20000LL) != 0;
          else
            v299 = sub_1E15D00(v290, 0x20000u, 1);
          if ( v299 )
            break;
        }
LABEL_378:
        if ( (*(_BYTE *)v290 & 4) != 0 )
        {
          v290 = *(_QWORD *)(v290 + 8);
          if ( (char *)v290 == v291 )
            goto LABEL_380;
        }
        else
        {
          while ( (*(_BYTE *)(v290 + 46) & 8) != 0 )
            v290 = *(_QWORD *)(v290 + 8);
          v290 = *(_QWORD *)(v290 + 8);
          if ( (char *)v290 == v291 )
            goto LABEL_380;
        }
      }
      v294 = *(_QWORD *)(v290 + 32);
LABEL_398:
      if ( v294 + 40LL * *(unsigned int *)(v290 + 40) == v294 )
        goto LABEL_378;
      v448 = v291;
      v300 = v290;
      v301 = v294 + 40LL * *(unsigned int *)(v290 + 40);
      while ( 2 )
      {
        if ( *(_BYTE *)v294 == 5 )
        {
          v302 = *(int *)(v294 + 24);
          if ( (int)v302 >= 0 )
          {
            v303 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8 * v302);
            if ( *(_DWORD *)(v303 + 8) )
            {
              v304 = *(_QWORD *)(a1 + 1424);
              v305 = v300;
              if ( (*(_BYTE *)(v300 + 46) & 4) != 0 )
              {
                do
                  v305 = *(_QWORD *)v305 & 0xFFFFFFFFFFFFFFF8LL;
                while ( (*(_BYTE *)(v305 + 46) & 4) != 0 );
              }
              v306 = *(_QWORD *)(v304 + 368);
              v307 = *(_DWORD *)(v304 + 384);
              if ( v307 )
              {
                v308 = (v307 - 1) & (((unsigned int)v305 >> 9) ^ ((unsigned int)v305 >> 4));
                v309 = (__int64 *)(v306 + 16LL * v308);
                v310 = *v309;
                if ( *v309 == v305 )
                {
LABEL_407:
                  if ( sub_1DB3C70((__int64 *)v303, v309[1]) == *(_QWORD *)v303 + 24LL * *(unsigned int *)(v303 + 8) )
                  {
                    *(_DWORD *)(v303 + 72) = 0;
                    *(_DWORD *)(v303 + 8) = 0;
                  }
                  goto LABEL_409;
                }
                v332 = 1;
                while ( v310 != -8 )
                {
                  v333 = v332 + 1;
                  v308 = (v307 - 1) & (v332 + v308);
                  v309 = (__int64 *)(v306 + 16LL * v308);
                  v310 = *v309;
                  if ( *v309 == v305 )
                    goto LABEL_407;
                  v332 = v333;
                }
              }
              v309 = (__int64 *)(v306 + 16LL * v307);
              goto LABEL_407;
            }
          }
        }
LABEL_409:
        v294 += 40;
        if ( v301 == v294 )
        {
          v290 = v300;
          v291 = v448;
          goto LABEL_378;
        }
        continue;
      }
    }
  }
LABEL_313:
  v483 = 0;
  v214 = (char *)v480;
  v484 = 0;
  v485 = 0;
  v486 = 0;
  v215 = 0;
  do
  {
    if ( !*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 392) + 8LL * *(int *)&v214[v215]) + 8LL) )
    {
      *(_DWORD *)&v214[v215] = -1;
      v214 = (char *)v480;
    }
    v215 += 4;
  }
  while ( 4 * v414 != v215 );
  v216 = 4LL * (unsigned int)v481;
  v217 = &v214[v216];
  v218 = v216 >> 2;
  if ( v216 )
  {
    while ( 1 )
    {
      v219 = 4 * v218;
      v220 = (char *)sub_2207800(4 * v218, &unk_435FF63);
      v221 = v220;
      if ( v220 )
        break;
      v218 >>= 1;
      if ( !v218 )
        goto LABEL_551;
    }
    sub_1F225E0(v214, v217, v220, v218, a1);
  }
  else
  {
LABEL_551:
    v219 = 0;
    v221 = 0;
    sub_1F21DF0(v214, v217, a1);
  }
  j_j___libc_free_0(v221, v219);
  v224 = *(_QWORD *)(a1 + 536);
  for ( i1 = v224 + 48LL * *(unsigned int *)(a1 + 544); i1 != v224; v224 += 48 )
  {
    v226 = 8LL * *(unsigned int *)(v224 + 8);
    v227 = (__int64 *)(*(_QWORD *)v224 + v226);
    if ( v227 != *(__int64 **)v224 )
    {
      srcf = *(__int64 **)v224;
      _BitScanReverse64(&v228, v226 >> 3);
      sub_1F22B90(*(__int64 **)v224, (__int64 *)(*(_QWORD *)v224 + v226), 2LL * (int)(63 - (v228 ^ 0x3F)));
      if ( (unsigned __int64)v226 <= 0x80 )
      {
        sub_1F20F60(srcf, v227);
      }
      else
      {
        v229 = srcf + 16;
        sub_1F20F60(srcf, srcf + 16);
        if ( v227 != srcf + 16 )
        {
          do
          {
            v230 = *v229;
            v231 = v229 - 1;
            v232 = *v229 & 0xFFFFFFFFFFFFFFF8LL;
            v233 = (*v229 >> 1) & 3;
            if ( (*(_DWORD *)((*(v229 - 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(v229 - 1) >> 1) & 3) <= (*(_DWORD *)(v232 + 24) | (unsigned int)v233) )
            {
              v235 = v229;
            }
            else
            {
              do
              {
                v234 = *v231;
                v235 = v231--;
                v231[2] = v234;
              }
              while ( ((unsigned int)v233 | *(_DWORD *)(v232 + 24)) < (*(_DWORD *)((*v231 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                     | (unsigned int)(*v231 >> 1) & 3) );
            }
            ++v229;
            *v235 = v230;
          }
          while ( v227 != v229 );
        }
      }
    }
  }
  v236 = v480;
  v237 = (_QWORD *)a1;
  while ( 2 )
  {
    v385 = 0;
    v238 = v237;
    v447 = 0;
    v413 = 0;
LABEL_330:
    while ( 1 )
    {
      ++v447;
      if ( v236[v413] != -1 )
        break;
      if ( v405 <= v447 )
        goto LABEL_350;
      v413 = v447;
    }
    if ( v405 > v447 )
    {
      v453 = v413;
      v413 = v447;
      v239 = v447;
      while ( 1 )
      {
        v240 = v236[v239];
        if ( v240 != -1 )
        {
          v241 = v236[v453];
          if ( ((*(_QWORD *)(v238[192] + 8LL * (v240 >> 6)) & (1LL << v240)) != 0) == ((*(_QWORD *)(v238[192]
                                                                                                  + 8LL * (v241 >> 6))
                                                                                      & (1LL << v241)) != 0) )
          {
            v242 = v238[49];
            v418 = v238[67];
            v402 = *(_QWORD *)(v242 + 8LL * (int)v240);
            v428 = v418 + 48LL * (int)v240;
            v437 = *(_QWORD *)(v242 + 8LL * (int)v241);
            v243 = sub_1DB4AC0(v437, *(__int64 **)v428, *(unsigned int *)(v428 + 8));
            v244 = v418;
            if ( !v243 )
            {
              v419 = v437;
              v438 = v244 + 48LL * (int)v241;
              if ( !(unsigned __int8)sub_1DB4AC0(v402, *(__int64 **)v438, *(unsigned int *)(v438 + 8)) )
              {
                sub_1DB9380(v419, v402, **(_QWORD **)(v419 + 64), v245, v246, v247);
                v248 = v438;
                v249 = *(_DWORD *)(v438 + 8);
                v250 = *(__int64 **)v428;
                v251 = 8LL * *(unsigned int *)(v428 + 8);
                v252 = *(unsigned int *)(v428 + 8);
                v253 = v249;
                if ( v252 > *(unsigned int *)(v438 + 12) - (unsigned __int64)v249 )
                {
                  v403 = 8LL * *(unsigned int *)(v428 + 8);
                  v415 = *(__int64 **)v428;
                  v421 = *(_DWORD *)(v438 + 8);
                  v430 = *(_DWORD *)(v428 + 8);
                  sub_16CD150(v438, (const void *)(v438 + 16), v252 + v249, 8, v252, v438);
                  v248 = v438;
                  v251 = v403;
                  v250 = v415;
                  v249 = v421;
                  v253 = *(unsigned int *)(v438 + 8);
                  LODWORD(v252) = v430;
                }
                v254 = *(char **)v248;
                if ( v251 )
                {
                  v420 = v252;
                  v429 = v249;
                  v439 = v248;
                  memcpy(&v254[8 * v253], v250, v251);
                  v248 = v439;
                  LODWORD(v252) = v420;
                  v249 = v429;
                  v254 = *(char **)v439;
                  LODWORD(v253) = *(_DWORD *)(v439 + 8);
                }
                v255 = v252 + v253;
                *(_DWORD *)(v248 + 8) = v255;
                sub_1F23700(v254, &v254[8 * v249], (__int64)&v254[8 * v255]);
                if ( v486 )
                {
                  v256 = (v486 - 1) & (37 * v240);
                  v257 = (unsigned int *)(v484 + 8LL * v256);
                  v258 = *v257;
                  if ( v240 == *v257 )
                  {
LABEL_345:
                    v257[1] = v241;
                    *(_DWORD *)((char *)v480 + v239 * 4) = -1;
                    v259 = v238[29];
                    v260 = *(_DWORD *)(v259 + 32);
                    v261 = *(_QWORD *)(v259 + 8);
                    v262 = v261 + 40LL * (v241 + v260);
                    v263 = v261 + 40LL * (v240 + v260);
                    v264 = *(_DWORD *)(v262 + 16);
                    if ( *(_DWORD *)(v263 + 16) >= v264 )
                      v264 = *(_DWORD *)(v263 + 16);
                    *(_DWORD *)(v262 + 16) = v264;
                    sub_1E08740(v259, v264);
                    v385 = 1;
                    v236 = v480;
                    *(_QWORD *)(*(_QWORD *)(v238[29] + 8LL) + 40LL * (*(_DWORD *)(v238[29] + 32LL) + v240) + 8) = -1;
                    goto LABEL_333;
                  }
                  v340 = 1;
                  v341 = 0;
                  while ( v258 != 0x7FFFFFFF )
                  {
                    if ( !v341 && v258 == 0x80000000 )
                      v341 = v257;
                    v256 = (v486 - 1) & (v340 + v256);
                    v257 = (unsigned int *)(v484 + 8LL * v256);
                    v258 = *v257;
                    if ( v240 == *v257 )
                      goto LABEL_345;
                    ++v340;
                  }
                  if ( v341 )
                    v257 = v341;
                  v483 = (__int64 *)((char *)v483 + 1);
                  v335 = v485 + 1;
                  if ( 4 * ((int)v485 + 1) < 3 * v486 )
                  {
                    if ( v486 - HIDWORD(v485) - v335 <= v486 >> 3 )
                    {
                      sub_1E4B4F0((__int64)&v483, v486);
                      if ( !v486 )
                      {
LABEL_636:
                        LODWORD(v485) = v485 + 1;
                        BUG();
                      }
                      v342 = 0;
                      LODWORD(v343) = (v486 - 1) & (37 * v240);
                      v335 = v485 + 1;
                      v344 = 1;
                      v257 = (unsigned int *)(v484 + 8LL * (unsigned int)v343);
                      v345 = *v257;
                      if ( v240 != *v257 )
                      {
                        while ( v345 != 0x7FFFFFFF )
                        {
                          if ( !v342 && v345 == 0x80000000 )
                            v342 = v257;
                          v343 = (v486 - 1) & ((_DWORD)v343 + v344);
                          v257 = (unsigned int *)(v484 + 8 * v343);
                          v345 = *v257;
                          if ( v240 == *v257 )
                            goto LABEL_461;
                          ++v344;
                        }
                        if ( v342 )
                          v257 = v342;
                      }
                    }
                    goto LABEL_461;
                  }
                }
                else
                {
                  v483 = (__int64 *)((char *)v483 + 1);
                }
                sub_1E4B4F0((__int64)&v483, 2 * v486);
                if ( !v486 )
                  goto LABEL_636;
                LODWORD(v334) = (v486 - 1) & (37 * v240);
                v335 = v485 + 1;
                v257 = (unsigned int *)(v484 + 8LL * (unsigned int)v334);
                v336 = *v257;
                if ( v240 != *v257 )
                {
                  v376 = 1;
                  v377 = 0;
                  while ( v336 != 0x7FFFFFFF )
                  {
                    if ( !v377 && v336 == 0x80000000 )
                      v377 = v257;
                    v334 = (v486 - 1) & ((_DWORD)v334 + v376);
                    v257 = (unsigned int *)(v484 + 8 * v334);
                    v336 = *v257;
                    if ( v240 == *v257 )
                      goto LABEL_461;
                    ++v376;
                  }
                  if ( v377 )
                    v257 = v377;
                }
LABEL_461:
                LODWORD(v485) = v335;
                if ( *v257 != 0x7FFFFFFF )
                  --HIDWORD(v485);
                *v257 = v240;
                v257[1] = 0;
                goto LABEL_345;
              }
            }
            v236 = v480;
          }
        }
LABEL_333:
        if ( ++v239 == v405 - 1 - v447 + (unsigned __int64)v447 + 1 )
          goto LABEL_330;
      }
    }
LABEL_350:
    v237 = v238;
    if ( v385 )
      continue;
    break;
  }
  v265 = v486;
  v266 = 0;
  while ( 2 )
  {
    if ( v265 )
    {
      i2 = v265 - 1;
      v268 = v484;
      v269 = 1;
      v270 = 37 * v266;
      v271 = (37 * v266) & (v265 - 1);
      v272 = v271;
      v273 = (_DWORD *)(v484 + 8LL * v271);
      v274 = *v273;
      v275 = *v273;
      if ( *v273 == v266 )
      {
LABEL_354:
        v276 = v273[1];
        goto LABEL_355;
      }
      while ( 1 )
      {
        if ( v275 == 0x7FFFFFFF )
          goto LABEL_360;
        v272 = i2 & (v269 + v272);
        v275 = *(_DWORD *)(v484 + 8LL * v272);
        if ( v266 == v275 )
          break;
        ++v269;
      }
      v337 = 1;
      v338 = 0;
      while ( v274 != 0x7FFFFFFF )
      {
        if ( v274 == 0x80000000 && !v338 )
          v338 = v273;
        v271 = i2 & (v337 + v271);
        v273 = (_DWORD *)(v484 + 8LL * v271);
        v274 = *v273;
        if ( v275 == *v273 )
          goto LABEL_354;
        ++v337;
      }
      if ( v338 )
        v273 = v338;
      v483 = (__int64 *)((char *)v483 + 1);
      v339 = v485 + 1;
      if ( 4 * ((int)v485 + 1) >= 3 * v265 )
      {
        sub_1E4B4F0((__int64)&v483, 2 * v265);
        if ( !v486 )
        {
LABEL_638:
          LODWORD(v485) = v485 + 1;
          BUG();
        }
        v361 = v270 & (v486 - 1);
        v339 = v485 + 1;
        v273 = (_DWORD *)(v484 + 8LL * v361);
        v362 = *v273;
        if ( *v273 != v275 )
        {
          v363 = 1;
          v357 = 0;
          while ( v362 != 0x7FFFFFFF )
          {
            if ( !v357 && v362 == 0x80000000 )
              v357 = v273;
            v361 = (v486 - 1) & (v363 + v361);
            v273 = (_DWORD *)(v484 + 8LL * v361);
            v362 = *v273;
            if ( v275 == *v273 )
              goto LABEL_473;
            ++v363;
          }
LABEL_513:
          if ( v357 )
            v273 = v357;
        }
      }
      else if ( v265 - HIDWORD(v485) - v339 <= v265 >> 3 )
      {
        sub_1E4B4F0((__int64)&v483, v265);
        if ( !v486 )
          goto LABEL_638;
        v357 = 0;
        v358 = v270 & (v486 - 1);
        v339 = v485 + 1;
        v359 = 1;
        v273 = (_DWORD *)(v484 + 8LL * v358);
        v360 = *v273;
        if ( *v273 != v275 )
        {
          while ( v360 != 0x7FFFFFFF )
          {
            if ( v360 == 0x80000000 && !v357 )
              v357 = v273;
            v358 = (v486 - 1) & (v359 + v358);
            v273 = (_DWORD *)(v484 + 8LL * v358);
            v360 = *v273;
            if ( v275 == *v273 )
              goto LABEL_473;
            ++v359;
          }
          goto LABEL_513;
        }
      }
LABEL_473:
      LODWORD(v485) = v339;
      if ( *v273 != 0x7FFFFFFF )
        --HIDWORD(v485);
      *v273 = v275;
      v273[1] = 0;
      v265 = v486;
      if ( v486 )
      {
        v268 = v484;
        v276 = 0;
        for ( i2 = v486 - 1; ; i2 = v486 - 1 )
        {
LABEL_355:
          v277 = 37 * v276;
          LODWORD(v278) = i2 & (37 * v276);
          v279 = (_DWORD *)(v268 + 8LL * (unsigned int)v278);
          v280 = *v279;
          if ( v276 == *v279 )
          {
LABEL_356:
            v276 = v279[1];
            if ( !v265 )
              goto LABEL_444;
            goto LABEL_357;
          }
          v321 = *v279;
          v322 = i2 & (37 * v276);
          for ( i3 = 1; ; ++i3 )
          {
            if ( v321 == 0x7FFFFFFF )
              goto LABEL_360;
            v322 = i2 & (i3 + v322);
            v321 = *(_DWORD *)(v268 + 8LL * v322);
            if ( v321 == v276 )
              break;
          }
          v324 = 1;
          v325 = 0;
          while ( v280 != 0x7FFFFFFF )
          {
            if ( v280 == 0x80000000 && !v325 )
              v325 = v279;
            v278 = i2 & ((_DWORD)v278 + v324);
            v279 = (_DWORD *)(v268 + 8 * v278);
            v280 = *v279;
            if ( v276 == *v279 )
              goto LABEL_356;
            ++v324;
          }
          if ( v325 )
            v279 = v325;
          v483 = (__int64 *)((char *)v483 + 1);
          v326 = v485 + 1;
          if ( 4 * ((int)v485 + 1) >= 3 * v265 )
          {
            sub_1E4B4F0((__int64)&v483, 2 * v265);
            if ( !v486 )
            {
LABEL_639:
              LODWORD(v485) = v485 + 1;
              BUG();
            }
            v369 = (v486 - 1) & v277;
            v326 = v485 + 1;
            v279 = (_DWORD *)(v484 + 8LL * v369);
            v370 = *v279;
            if ( v276 != *v279 )
            {
              v371 = 1;
              v365 = 0;
              while ( v370 != 0x7FFFFFFF )
              {
                if ( !v365 && v370 == 0x80000000 )
                  v365 = v279;
                v369 = (v486 - 1) & (v371 + v369);
                v279 = (_DWORD *)(v484 + 8LL * v369);
                v370 = *v279;
                if ( v276 == *v279 )
                  goto LABEL_441;
                ++v371;
              }
              goto LABEL_533;
            }
          }
          else if ( v265 - (v326 + HIDWORD(v485)) <= v265 >> 3 )
          {
            sub_1E4B4F0((__int64)&v483, v265);
            if ( !v486 )
              goto LABEL_639;
            v365 = 0;
            v366 = (v486 - 1) & v277;
            v326 = v485 + 1;
            v367 = 1;
            v279 = (_DWORD *)(v484 + 8LL * v366);
            v368 = *v279;
            if ( v276 != *v279 )
            {
              while ( v368 != 0x7FFFFFFF )
              {
                if ( v368 == 0x80000000 && !v365 )
                  v365 = v279;
                v366 = (v486 - 1) & (v367 + v366);
                v279 = (_DWORD *)(v484 + 8LL * v366);
                v368 = *v279;
                if ( v276 == *v279 )
                  goto LABEL_441;
                ++v367;
              }
LABEL_533:
              if ( v365 )
                v279 = v365;
            }
          }
LABEL_441:
          LODWORD(v485) = v326;
          if ( *v279 != 0x7FFFFFFF )
            --HIDWORD(v485);
          *v279 = v276;
          v276 = 0;
          v279[1] = 0;
          v265 = v486;
          if ( !v486 )
          {
LABEL_444:
            v483 = (__int64 *)((char *)v483 + 1);
            goto LABEL_445;
          }
LABEL_357:
          v281 = v270 & (v265 - 1);
          v282 = (_DWORD *)(v484 + 8LL * v281);
          v283 = *v282;
          if ( *v282 == v266 )
            goto LABEL_358;
          v352 = 1;
          v353 = 0;
          while ( v283 != 0x7FFFFFFF )
          {
            if ( !v353 && v283 == 0x80000000 )
              v353 = v282;
            v281 = (v265 - 1) & (v352 + v281);
            v282 = (_DWORD *)(v484 + 8LL * v281);
            v283 = *v282;
            if ( v266 == *v282 )
              goto LABEL_358;
            ++v352;
          }
          if ( v353 )
            v282 = v353;
          v483 = (__int64 *)((char *)v483 + 1);
          v328 = v485 + 1;
          if ( 4 * ((int)v485 + 1) < 3 * v265 )
          {
            if ( v265 - HIDWORD(v485) - v328 > v265 >> 3 )
              goto LABEL_501;
            sub_1E4B4F0((__int64)&v483, v265);
            if ( !v486 )
            {
LABEL_635:
              LODWORD(v485) = v485 + 1;
              BUG();
            }
            v331 = 0;
            v354 = v270 & (v486 - 1);
            v328 = v485 + 1;
            v355 = 1;
            v282 = (_DWORD *)(v484 + 8LL * v354);
            v356 = *v282;
            if ( *v282 == v266 )
              goto LABEL_501;
            while ( v356 != 0x7FFFFFFF )
            {
              if ( !v331 && v356 == 0x80000000 )
                v331 = v282;
              v354 = (v486 - 1) & (v355 + v354);
              v282 = (_DWORD *)(v484 + 8LL * v354);
              v356 = *v282;
              if ( v266 == *v282 )
                goto LABEL_501;
              ++v355;
            }
            goto LABEL_507;
          }
LABEL_445:
          sub_1E4B4F0((__int64)&v483, 2 * v265);
          if ( !v486 )
            goto LABEL_635;
          v327 = v270 & (v486 - 1);
          v328 = v485 + 1;
          v282 = (_DWORD *)(v484 + 8LL * v327);
          v329 = *v282;
          if ( *v282 == v266 )
            goto LABEL_501;
          v330 = 1;
          v331 = 0;
          while ( v329 != 0x7FFFFFFF )
          {
            if ( v329 == 0x80000000 && !v331 )
              v331 = v282;
            v327 = (v486 - 1) & (v330 + v327);
            v282 = (_DWORD *)(v484 + 8LL * v327);
            v329 = *v282;
            if ( v266 == *v282 )
              goto LABEL_501;
            ++v330;
          }
LABEL_507:
          if ( v331 )
            v282 = v331;
LABEL_501:
          LODWORD(v485) = v328;
          if ( *v282 != 0x7FFFFFFF )
            --HIDWORD(v485);
          *v282 = v266;
          v282[1] = 0;
LABEL_358:
          v282[1] = v276;
          v265 = v486;
          if ( !v486 )
            break;
          v268 = v484;
        }
      }
    }
LABEL_360:
    if ( v405 > ++v266 )
      continue;
    break;
  }
  sub_1F259B0((_QWORD *)a1, (__int64)&v483, a3, a4, a5, a6, v222, v223, a9, a10);
  v284 = *(__int64 **)(a1 + 1432);
  v285 = &v284[*(unsigned int *)(a1 + 1440)];
  if ( v284 != v285 )
  {
    v286 = 0;
    do
    {
      v287 = *v284++;
      ++v286;
      sub_1E16240(v287);
    }
    while ( v285 != v284 );
    v385 = v286 != 0;
  }
  v288 = v484;
  *(_DWORD *)(a1 + 1440) = 0;
  j___libc_free_0(v288);
LABEL_69:
  if ( v480 != v482 )
    _libc_free((unsigned __int64)v480);
  return v385;
}
