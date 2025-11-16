// Function: sub_EB11F0
// Address: 0xeb11f0
//
__int64 __fastcall sub_EB11F0(
        __int64 a1,
        _QWORD *a2,
        unsigned int *a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        _QWORD *a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // r9
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // edx
  char *v18; // rax
  signed __int64 v19; // rsi
  unsigned int v20; // r12d
  __int64 v21; // rax
  __int64 (*v22)(); // rdx
  __int64 v23; // r13
  __int64 (*v24)(); // rbx
  unsigned int v25; // eax
  int v26; // ebx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  _QWORD *v31; // r14
  __int64 (*v32)(); // rax
  __int64 v33; // rdx
  int v34; // ebx
  __int64 (*v35)(); // rax
  __int64 v36; // r15
  char *v37; // r13
  bool v38; // zf
  __int64 (*v39)(); // rdx
  unsigned __int16 *v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 (*v44)(); // rax
  __int64 v45; // r8
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  char v48; // r15
  __int64 (*v49)(); // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __m128i *v52; // r13
  char **v53; // rsi
  int v54; // edx
  __m128i *v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rdi
  char *v58; // r9
  int v59; // edx
  char *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rbx
  __int64 v63; // r13
  __int64 v64; // r15
  unsigned __int64 v65; // r8
  unsigned __int16 *v66; // rax
  unsigned __int16 *v67; // r12
  _DWORD *v68; // rcx
  __int64 v69; // rdx
  _QWORD *v70; // rbx
  _QWORD *v71; // r12
  __int64 v72; // rdi
  char *v73; // r8
  __int64 v74; // rcx
  __int64 v75; // rsi
  char *v76; // rsi
  char *v77; // rbx
  char *v78; // rdx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // r15
  __int64 v82; // rdx
  __int64 v83; // rdx
  unsigned int v84; // edx
  unsigned int v85; // eax
  unsigned __int64 v86; // r13
  unsigned int v87; // r14d
  unsigned __int64 v88; // rax
  __int64 v89; // rax
  __int64 i; // rdx
  unsigned __int64 v91; // rax
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 j; // rbx
  unsigned int v95; // r13d
  unsigned int v96; // r12d
  __int64 v97; // rdi
  char v98; // cl
  __int64 v99; // rax
  __int64 *v100; // rsi
  unsigned int v101; // r12d
  __int64 v102; // rsi
  __int64 v103; // rdi
  char v104; // cl
  __int64 v105; // rdx
  char **v106; // rsi
  char *v107; // r13
  __int64 v108; // rax
  unsigned __int8 *v109; // rbx
  char *v110; // r14
  __int64 v111; // r15
  int v112; // ebx
  unsigned int v113; // edx
  __int64 v114; // rax
  unsigned __int64 v115; // rdx
  char v116; // r15
  __int64 (*v117)(); // rax
  __int64 v118; // rax
  __int64 v119; // r9
  __int64 v120; // rax
  unsigned __int64 v121; // r8
  int v122; // esi
  __int64 *v123; // rdx
  __m128i *v124; // rcx
  __m128i *v125; // rax
  __int64 v126; // rsi
  __int64 v127; // rcx
  char *v128; // rax
  char v129; // al
  _QWORD *v130; // rbx
  __int64 v131; // rdi
  char *v132; // r13
  unsigned __int64 v133; // rdx
  const __m128i *v134; // rbx
  __m128i *v135; // rcx
  __m128i v136; // xmm3
  __int64 *v137; // r14
  __int64 v138; // rsi
  unsigned __int64 v139; // r8
  char *v140; // rdx
  const __m128i *v141; // rbx
  __m128i *v142; // rax
  __m128i v143; // xmm5
  __int64 v144; // r15
  __int64 v145; // r14
  __int64 v146; // rax
  __int64 *v147; // r8
  char *v148; // rdx
  __int64 *v149; // rdi
  __int64 v150; // rcx
  __int64 v151; // rdx
  __int64 v152; // r12
  __int64 v153; // rbx
  __int64 v154; // rdi
  _QWORD *v155; // rbx
  __int64 v156; // rax
  _QWORD *v157; // r12
  __int64 v158; // rdi
  unsigned int v159; // r12d
  __int64 *v160; // rbx
  __int64 *v161; // r13
  char **v162; // rbx
  char **v163; // r13
  unsigned __int64 v165; // rdx
  const __m128i *v166; // rbx
  __m128i *v167; // r9
  __m128i v168; // xmm3
  unsigned __int64 v169; // rdx
  const __m128i *v170; // rbx
  __m128i *v171; // rdi
  __m128i v172; // xmm3
  char *v173; // rdi
  __int64 v174; // rax
  unsigned __int8 *v175; // rsi
  size_t v176; // rdx
  size_t v177; // rax
  void **v178; // rbx
  size_t v179; // rdx
  unsigned __int8 *v180; // rsi
  __int64 v181; // rax
  size_t v182; // rdx
  size_t v183; // rdx
  _BYTE *v184; // rdi
  __int64 v185; // rcx
  _DWORD *v186; // rax
  void *v187; // rdi
  char *v188; // rdx
  __int64 v189; // rax
  _WORD *v190; // rax
  unsigned __int64 v191; // rbx
  size_t v192; // rbx
  _WORD *v193; // rdi
  __int64 v194; // rax
  void **v195; // r9
  void *v196; // rsi
  size_t v197; // r10
  char *v198; // rdx
  unsigned __int8 *v199; // rbx
  char *v200; // rsi
  __int64 v201; // rax
  __int64 v202; // rdi
  unsigned __int8 **v203; // r9
  char *v204; // rax
  char *v205; // rax
  void **v206; // rdi
  unsigned int v207; // ebx
  unsigned __int8 **v208; // r9
  __int64 v209; // rax
  char *v210; // rbx
  void **v211; // rdi
  unsigned int v212; // ebx
  __int64 v213; // rax
  __int64 v214; // rdx
  __int64 v215; // rdi
  _BYTE *v216; // rax
  void **v217; // rdi
  unsigned int v218; // ebx
  __int64 v219; // rdi
  __int64 v220; // rax
  _BYTE *v221; // rax
  void **v222; // rdi
  __int64 v223; // rax
  _BYTE *v224; // rdx
  _WORD *v225; // rdx
  __int64 v226; // rbx
  _BYTE *v227; // rax
  void **v228; // rdi
  _BYTE *v229; // rdx
  signed __int64 v230; // rbx
  signed __int64 v231; // rbx
  size_t v232; // rax
  _BYTE *v233; // rsi
  size_t v234; // rdx
  void **v235; // rdi
  _QWORD *v236; // rax
  _QWORD *v237; // rax
  _WORD *v238; // rax
  signed __int64 v239; // rbx
  void **v240; // rdi
  void **v241; // rdi
  _BYTE *v242; // rax
  void **v243; // rdi
  signed __int64 v244; // rbx
  size_t v245; // rdx
  unsigned __int8 *v246; // rsi
  __int64 v247; // rax
  _QWORD *v248; // r13
  _QWORD *v249; // rbx
  unsigned int v250; // eax
  __int64 v251; // rcx
  void **v252; // rdi
  __int64 v253; // rax
  __int64 v254; // rdi
  __int64 v255; // rax
  _QWORD *v256; // rax
  _WORD *v257; // rax
  _QWORD *v258; // rax
  __int64 v259; // rax
  void **v260; // r9
  char *v261; // rdi
  __int64 v262; // rax
  __int64 v263; // rax
  unsigned __int8 v269; // [rsp+90h] [rbp-560h]
  char v270; // [rsp+90h] [rbp-560h]
  unsigned __int8 v271; // [rsp+90h] [rbp-560h]
  unsigned __int8 v272; // [rsp+90h] [rbp-560h]
  unsigned __int8 v273; // [rsp+90h] [rbp-560h]
  void **v274; // [rsp+90h] [rbp-560h]
  unsigned int v275; // [rsp+9Ch] [rbp-554h]
  __int64 v276; // [rsp+A8h] [rbp-548h]
  __int64 v277; // [rsp+A8h] [rbp-548h]
  void **v278; // [rsp+A8h] [rbp-548h]
  unsigned __int8 **v279; // [rsp+A8h] [rbp-548h]
  __int64 v280; // [rsp+A8h] [rbp-548h]
  size_t v281; // [rsp+A8h] [rbp-548h]
  unsigned __int8 **v282; // [rsp+A8h] [rbp-548h]
  unsigned __int8 **v283; // [rsp+A8h] [rbp-548h]
  size_t v284; // [rsp+A8h] [rbp-548h]
  unsigned __int8 **v285; // [rsp+A8h] [rbp-548h]
  size_t v286; // [rsp+A8h] [rbp-548h]
  unsigned __int16 *v287; // [rsp+B0h] [rbp-540h]
  unsigned int v288; // [rsp+B0h] [rbp-540h]
  __int64 v289; // [rsp+B8h] [rbp-538h]
  __int64 v290; // [rsp+B8h] [rbp-538h]
  __int64 v291; // [rsp+B8h] [rbp-538h]
  __int64 v292; // [rsp+C0h] [rbp-530h]
  unsigned __int8 v293; // [rsp+C0h] [rbp-530h]
  __int64 v294; // [rsp+C0h] [rbp-530h]
  unsigned __int8 *v295; // [rsp+C0h] [rbp-530h]
  unsigned __int8 v296; // [rsp+C0h] [rbp-530h]
  __int64 v297; // [rsp+C0h] [rbp-530h]
  int v298; // [rsp+C8h] [rbp-528h]
  __int64 *v299; // [rsp+C8h] [rbp-528h]
  unsigned int v300; // [rsp+C8h] [rbp-528h]
  char *v301; // [rsp+D0h] [rbp-520h] BYREF
  __int64 v302; // [rsp+D8h] [rbp-518h]
  unsigned __int64 v303; // [rsp+E0h] [rbp-510h]
  char v304[8]; // [rsp+E8h] [rbp-508h] BYREF
  char *v305; // [rsp+F0h] [rbp-500h] BYREF
  __int64 v306; // [rsp+F8h] [rbp-4F8h]
  unsigned __int64 v307; // [rsp+100h] [rbp-4F0h]
  char v308[8]; // [rsp+108h] [rbp-4E8h] BYREF
  void *base; // [rsp+110h] [rbp-4E0h] BYREF
  __int64 v310; // [rsp+118h] [rbp-4D8h]
  _BYTE v311[16]; // [rsp+120h] [rbp-4D0h] BYREF
  __int64 v312[2]; // [rsp+130h] [rbp-4C0h] BYREF
  __int64 v313; // [rsp+140h] [rbp-4B0h] BYREF
  _BYTE *v314; // [rsp+150h] [rbp-4A0h] BYREF
  __int64 v315; // [rsp+158h] [rbp-498h]
  _BYTE v316[32]; // [rsp+160h] [rbp-490h] BYREF
  _BYTE *v317; // [rsp+180h] [rbp-470h] BYREF
  __int64 v318; // [rsp+188h] [rbp-468h]
  _BYTE v319[32]; // [rsp+190h] [rbp-460h] BYREF
  _QWORD *v320; // [rsp+1B0h] [rbp-440h] BYREF
  size_t n; // [rsp+1B8h] [rbp-438h]
  _QWORD v322[8]; // [rsp+1C0h] [rbp-430h] BYREF
  unsigned int v323; // [rsp+200h] [rbp-3F0h]
  char v324; // [rsp+204h] [rbp-3ECh]
  void **v325; // [rsp+208h] [rbp-3E8h]
  char *v326; // [rsp+210h] [rbp-3E0h] BYREF
  __int64 v327; // [rsp+218h] [rbp-3D8h]
  char *v328; // [rsp+220h] [rbp-3D0h] BYREF
  char *v329; // [rsp+228h] [rbp-3C8h]
  void *dest; // [rsp+230h] [rbp-3C0h]
  __int64 v331; // [rsp+238h] [rbp-3B8h]
  _QWORD *v332; // [rsp+240h] [rbp-3B0h]
  __int64 v333; // [rsp+248h] [rbp-3A8h]
  __int64 v334; // [rsp+250h] [rbp-3A0h]
  __int64 v335; // [rsp+258h] [rbp-398h]
  __int64 v336; // [rsp+260h] [rbp-390h]
  __int64 v337; // [rsp+268h] [rbp-388h]
  __int64 v338; // [rsp+270h] [rbp-380h]
  __int64 v339; // [rsp+278h] [rbp-378h]
  int v340; // [rsp+280h] [rbp-370h]
  char v341; // [rsp+288h] [rbp-368h]
  char **v342; // [rsp+290h] [rbp-360h] BYREF
  __int64 v343; // [rsp+298h] [rbp-358h]
  _BYTE v344[128]; // [rsp+2A0h] [rbp-350h] BYREF
  __int64 *v345; // [rsp+320h] [rbp-2D0h] BYREF
  __int64 v346; // [rsp+328h] [rbp-2C8h]
  _BYTE v347[128]; // [rsp+330h] [rbp-2C0h] BYREF
  void *v348; // [rsp+3B0h] [rbp-240h] BYREF
  __int64 v349; // [rsp+3B8h] [rbp-238h]
  _BYTE v350[560]; // [rsp+3C0h] [rbp-230h] BYREF

  v10 = a1;
  v11 = a1;
  v317 = v319;
  v301 = v304;
  v305 = v308;
  v342 = (char **)v344;
  v314 = v316;
  v345 = (__int64 *)v347;
  v315 = 0x400000000LL;
  v318 = 0x400000000LL;
  v343 = 0x400000000LL;
  v346 = 0x400000000LL;
  v302 = 0;
  v303 = 4;
  v306 = 0;
  v307 = 4;
  base = v311;
  v310 = 0x400000000LL;
  v348 = v350;
  v349 = 0x400000000LL;
  sub_EABFE0(a1);
  v275 = 0;
  v13 = **(_DWORD **)(a1 + 48);
  if ( !v13 )
  {
LABEL_72:
    v73 = (char *)base;
    *a3 = v318;
    v74 = (__int64)a4;
    *a4 = v315;
    v75 = 4LL * (unsigned int)v310;
    if ( (unsigned int)v310 > 1uLL )
    {
      qsort(v73, v75 >> 2, 4u, (__compar_fn_t)sub_EA2430);
      v73 = (char *)base;
      v75 = 4LL * (unsigned int)v310;
    }
    v76 = &v73[v75];
    v77 = v73;
    if ( v73 != v76 )
    {
      while ( 1 )
      {
        v78 = v77;
        v77 += 4;
        if ( v76 == v77 )
          break;
        v74 = *((unsigned int *)v77 - 1);
        if ( (_DWORD)v74 == *(_DWORD *)v77 )
        {
          if ( v76 == v78 )
          {
            v77 = v76;
          }
          else
          {
            v186 = v78 + 8;
            if ( v76 != v78 + 8 )
            {
              while ( 1 )
              {
                if ( (_DWORD)v74 != *v186 )
                {
                  v74 = (unsigned int)*v186;
                  v78 += 4;
                  *(_DWORD *)v78 = v74;
                }
                if ( v76 == (char *)++v186 )
                  break;
                v74 = *(unsigned int *)v78;
              }
              v73 = (char *)base;
              v187 = v78 + 4;
              v188 = (char *)((_BYTE *)base + 4 * (unsigned int)v310 - v76);
              v77 = &v188[(_QWORD)v187];
              if ( (char *)base + 4 * (unsigned int)v310 != v76 )
              {
                memmove(v187, v76, (size_t)v188);
                v73 = (char *)base;
              }
            }
          }
          break;
        }
      }
    }
    v326 = (char *)&v328;
    LODWORD(v310) = (v77 - v73) >> 2;
    v327 = 0;
    LOBYTE(v328) = 0;
    sub_EA88D0((__int64)a7, (unsigned int)v310, (__int64 *)&v326, v74, (__int64)v73, v12);
    if ( v326 != (char *)&v328 )
      j_j___libc_free_0(v326, v328 + 1);
    if ( (_DWORD)v310 )
    {
      v290 = v10;
      v294 = 4LL * (unsigned int)v310;
      v81 = 0;
      do
      {
        v327 = 0;
        v82 = *a7;
        v328 = 0;
        v329 = 0;
        dest = 0;
        v332 = (_QWORD *)(v82 + 8 * v81);
        v331 = 0x100000000LL;
        v326 = (char *)&unk_49DD210;
        sub_CB5980((__int64)&v326, 0, 0, 0);
        v83 = *(unsigned int *)((char *)base + v81);
        v81 += 4;
        (*(void (__fastcall **)(__int64, char **, __int64))(*(_QWORD *)a9 + 40LL))(a9, &v326, v83);
        v326 = (char *)&unk_49DD210;
        sub_CB5840((__int64)&v326);
      }
      while ( v294 != v81 );
      v10 = v290;
    }
    v84 = *a3;
    v85 = *a4;
    if ( *a4 | *a3 )
    {
      v86 = v84 + v85;
      v87 = v84 + v85;
      v88 = *(unsigned int *)(a5 + 8);
      if ( v86 != v88 )
      {
        if ( v86 >= v88 )
        {
          if ( v86 > *(unsigned int *)(a5 + 12) )
          {
            sub_C8D5F0(a5, (const void *)(a5 + 16), v86, 0x10u, v79, v80);
            v88 = *(unsigned int *)(a5 + 8);
          }
          v89 = *(_QWORD *)a5 + 16 * v88;
          for ( i = *(_QWORD *)a5 + 16 * v86; i != v89; v89 += 16 )
          {
            if ( v89 )
            {
              *(_QWORD *)v89 = 0;
              *(_BYTE *)(v89 + 8) = 0;
            }
          }
        }
        *(_DWORD *)(a5 + 8) = v86;
      }
      v91 = *(unsigned int *)(a6 + 8);
      if ( v86 != v91 )
      {
        v92 = 32 * v86;
        if ( v86 < v91 )
        {
          v248 = (_QWORD *)(*(_QWORD *)a6 + 32 * v91);
          v249 = (_QWORD *)(*(_QWORD *)a6 + v92);
          while ( v249 != v248 )
          {
            v248 -= 4;
            if ( (_QWORD *)*v248 != v248 + 2 )
              j_j___libc_free_0(*v248, v248[2] + 1LL);
          }
        }
        else
        {
          if ( v86 > *(unsigned int *)(a6 + 12) )
          {
            sub_95D880(a6, v86);
            v91 = *(unsigned int *)(a6 + 8);
          }
          v93 = *(_QWORD *)a6 + 32 * v91;
          for ( j = *(_QWORD *)a6 + v92; j != v93; v93 += 32 )
          {
            if ( v93 )
            {
              *(_QWORD *)(v93 + 8) = 0;
              *(_QWORD *)v93 = v93 + 16;
              *(_BYTE *)(v93 + 16) = 0;
            }
          }
        }
        *(_DWORD *)(a6 + 8) = v87;
      }
      v95 = *a3;
      if ( *a3 )
      {
        v96 = 0;
        do
        {
          v97 = v96++;
          v98 = v305[v97];
          v99 = *(_QWORD *)a5 + 16 * v97;
          *(_QWORD *)v99 = *(_QWORD *)&v317[8 * v97];
          v100 = v345;
          *(_BYTE *)(v99 + 8) = v98;
          sub_2240AE0(*(_QWORD *)a6 + 32 * v97, &v100[4 * v97]);
        }
        while ( *a3 > v96 );
        v95 = *a3;
      }
      if ( *a4 )
      {
        v101 = 0;
        do
        {
          v102 = v101;
          v103 = v95 + v101++;
          v104 = v301[v102];
          v105 = *(_QWORD *)a5 + 16 * v103;
          *(_QWORD *)v105 = *(_QWORD *)&v314[8 * v102];
          v106 = &v342[4 * v102];
          *(_BYTE *)(v105 + 8) = v104;
          sub_2240AE0(*(_QWORD *)a6 + 32 * v103, v106);
        }
        while ( *a4 > v101 );
      }
    }
    n = 0;
    v320 = v322;
    v331 = 0x100000000LL;
    LOBYTE(v322[0]) = 0;
    v327 = 0;
    v326 = (char *)&unk_49DD210;
    v328 = 0;
    v332 = &v320;
    v329 = 0;
    dest = 0;
    sub_CB5980((__int64)&v326, 0, 0, 0);
    v107 = (char *)v348;
    v108 = ***(_QWORD ***)(v10 + 248);
    v109 = *(unsigned __int8 **)(v108 + 8);
    v295 = *(unsigned __int8 **)(v108 + 16);
    v19 = (unsigned __int64)(unsigned int)v349 << 7;
    if ( (unsigned int)v349 > 1uLL )
    {
      qsort(v348, v19 >> 7, 0x80u, (__compar_fn_t)sub_EA2940);
      v107 = (char *)v348;
      v19 = (unsigned __int64)(unsigned int)v349 << 7;
    }
    v110 = &v107[v19];
    if ( &v107[v19] != v107 )
    {
      v288 = 0;
      v19 = (signed __int64)v109;
      v291 = v10;
      while ( v107[20] )
      {
LABEL_222:
        v107 += 128;
        if ( v107 == v110 )
        {
          v109 = (unsigned __int8 *)v19;
          goto LABEL_224;
        }
      }
      v111 = *((_QWORD *)v107 + 1);
      v112 = *(_DWORD *)v107;
      v113 = v111 - v19;
      if ( (_DWORD)v111 != (_DWORD)v19 )
      {
        if ( v113 > (unsigned __int64)(v329 - (_BYTE *)dest) )
        {
          sub_CB6200((__int64)&v326, (unsigned __int8 *)v19, v113);
        }
        else if ( v113 )
        {
          v277 = v113;
          memcpy(dest, (const void *)v19, v113);
          dest = (char *)dest + v277;
          if ( v112 != 9 )
          {
LABEL_117:
            switch ( v112 )
            {
              case 0:
                v225 = dest;
                if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 5 )
                {
                  sub_CB6200((__int64)&v326, (unsigned __int8 *)".align", 6u);
                }
                else
                {
                  *(_DWORD *)dest = 1768710446;
                  v225[2] = 28263;
                  dest = (char *)dest + 6;
                }
                v181 = 0;
                if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v291 + 224) + 152LL) + 260LL) )
                {
                  v226 = *((_QWORD *)v107 + 3);
                  v227 = dest;
                  if ( dest >= v329 )
                  {
                    v228 = (void **)sub_CB5D20((__int64)&v326, 32);
                  }
                  else
                  {
                    v228 = (void **)&v326;
                    dest = (char *)dest + 1;
                    *v227 = 32;
                  }
                  sub_CB59D0((__int64)v228, (unsigned int)v226);
                  v181 = 2;
                  if ( (unsigned int)v226 > 3 )
                    v181 = ((unsigned int)v226 > 6) + 3LL;
                }
                goto LABEL_221;
              case 1:
                v224 = dest;
                if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 4 )
                {
                  sub_CB6200((__int64)&v326, ".even", 5u);
                }
                else
                {
                  *(_DWORD *)dest = 1702257966;
                  v224[4] = 110;
                  dest = (char *)dest + 5;
                }
                goto LABEL_220;
              case 2:
                v229 = dest;
                if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 4 )
                {
                  sub_CB6200((__int64)&v326, ".byte", 5u);
                }
                else
                {
                  *(_DWORD *)dest = 1954112046;
                  v229[4] = 101;
                  dest = (char *)dest + 5;
                }
                goto LABEL_220;
              case 3:
                if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 1 )
                {
                  v211 = (void **)sub_CB6200((__int64)&v326, (unsigned __int8 *)"${", 2u);
                }
                else
                {
                  v211 = (void **)&v326;
                  *(_WORD *)dest = 31524;
                  dest = (char *)dest + 2;
                }
                v212 = v275 + 1;
                v213 = sub_CB59D0((__int64)v211, v275);
                v214 = *(_QWORD *)(v213 + 32);
                v215 = v213;
                if ( (unsigned __int64)(*(_QWORD *)(v213 + 24) - v214) <= 2 )
                  goto LABEL_362;
                *(_BYTE *)(v214 + 2) = 125;
                *(_WORD *)v214 = 20538;
                *(_QWORD *)(v213 + 32) += 3LL;
                goto LABEL_295;
              case 4:
                v221 = dest;
                if ( v107[120] )
                {
                  if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 1 )
                  {
                    v222 = (void **)sub_CB6200((__int64)&v326, (unsigned __int8 *)"${", 2u);
                  }
                  else
                  {
                    v222 = (void **)&v326;
                    *(_WORD *)dest = 31524;
                    dest = (char *)dest + 2;
                  }
                  v212 = v275 + 1;
                  v215 = sub_CB59D0((__int64)v222, v275);
                  v223 = *(_QWORD *)(v215 + 32);
                  if ( (unsigned __int64)(*(_QWORD *)(v215 + 24) - v223) <= 2 )
                  {
LABEL_362:
                    sub_CB6200(v215, ":P}", 3u);
                  }
                  else
                  {
                    *(_BYTE *)(v223 + 2) = 125;
                    *(_WORD *)v223 = 20538;
                    *(_QWORD *)(v215 + 32) += 3LL;
                  }
                }
                else
                {
                  if ( v329 <= dest )
                  {
                    v241 = (void **)sub_CB5D20((__int64)&v326, 36);
                  }
                  else
                  {
                    v241 = (void **)&v326;
                    dest = (char *)dest + 1;
                    *v221 = 36;
                  }
                  v212 = v275 + 1;
                  sub_CB59D0((__int64)v241, v275);
                }
LABEL_295:
                v275 = v212;
                v181 = 0;
                goto LABEL_221;
              case 5:
                v216 = dest;
                if ( v107[120] )
                {
                  if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 1 )
                  {
                    v217 = (void **)sub_CB6200((__int64)&v326, (unsigned __int8 *)"${", 2u);
                  }
                  else
                  {
                    v217 = (void **)&v326;
                    *(_WORD *)dest = 31524;
                    dest = (char *)dest + 2;
                  }
                  v218 = v288 + 1;
                  v219 = sub_CB59D0((__int64)v217, v288);
                  v220 = *(_QWORD *)(v219 + 32);
                  if ( (unsigned __int64)(*(_QWORD *)(v219 + 24) - v220) <= 2 )
                  {
                    sub_CB6200(v219, ":P}", 3u);
                  }
                  else
                  {
                    *(_BYTE *)(v220 + 2) = 125;
                    *(_WORD *)v220 = 20538;
                    *(_QWORD *)(v219 + 32) += 3LL;
                  }
                }
                else
                {
                  if ( v329 <= dest )
                  {
                    v240 = (void **)sub_CB5D20((__int64)&v326, 36);
                  }
                  else
                  {
                    v240 = (void **)&v326;
                    dest = (char *)dest + 1;
                    *v216 = 36;
                  }
                  v218 = v288 + 1;
                  sub_CB59D0((__int64)v240, v288);
                }
                v288 = v218;
                v181 = 0;
                goto LABEL_221;
              case 6:
                v189 = *((_QWORD *)v107 + 3);
                if ( v189 == 64 )
                {
                  v257 = dest;
                  if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 9 )
                  {
                    sub_CB6200((__int64)&v326, "qword ptr ", 0xAu);
                  }
                  else
                  {
                    *(_QWORD *)dest = 0x74702064726F7771LL;
                    v257[4] = 8306;
                    dest = (char *)dest + 10;
                  }
                }
                else if ( v189 > 64 )
                {
                  switch ( v189 )
                  {
                    case 128LL:
                      v258 = dest;
                      if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 0xB )
                      {
                        sub_CB6200((__int64)&v326, "xmmword ptr ", 0xCu);
                      }
                      else
                      {
                        *((_DWORD *)dest + 2) = 544371824;
                        *v258 = 0x2064726F776D6D78LL;
                        dest = (char *)dest + 12;
                      }
                      break;
                    case 256LL:
                      v237 = dest;
                      if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 0xB )
                      {
                        sub_CB6200((__int64)&v326, "ymmword ptr ", 0xCu);
                      }
                      else
                      {
                        *((_DWORD *)dest + 2) = 544371824;
                        *v237 = 0x2064726F776D6D79LL;
                        dest = (char *)dest + 12;
                      }
                      break;
                    case 80LL:
                      v238 = dest;
                      if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 9 )
                      {
                        sub_CB6200((__int64)&v326, "xword ptr ", 0xAu);
                      }
                      else
                      {
                        *(_QWORD *)dest = 0x74702064726F7778LL;
                        v238[4] = 8306;
                        dest = (char *)dest + 10;
                      }
                      break;
                  }
                }
                else
                {
                  switch ( v189 )
                  {
                    case 16LL:
                      v256 = dest;
                      if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 8 )
                      {
                        sub_CB6200((__int64)&v326, (unsigned __int8 *)"word ptr ", 9u);
                      }
                      else
                      {
                        *((_BYTE *)dest + 8) = 32;
                        *v256 = 0x7274702064726F77LL;
                        dest = (char *)dest + 9;
                      }
                      break;
                    case 32LL:
                      v190 = dest;
                      if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 9 )
                      {
                        sub_CB6200((__int64)&v326, "dword ptr ", 0xAu);
                      }
                      else
                      {
                        *(_QWORD *)dest = 0x74702064726F7764LL;
                        v190[4] = 8306;
                        dest = (char *)dest + 10;
                      }
                      break;
                    case 8LL:
                      v236 = dest;
                      if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 8 )
                      {
                        sub_CB6200((__int64)&v326, "byte ptr ", 9u);
                      }
                      else
                      {
                        *((_BYTE *)dest + 8) = 32;
                        *v236 = 0x7274702065747962LL;
                        dest = (char *)dest + 9;
                      }
                      break;
                  }
                }
                goto LABEL_220;
              case 7:
                v173 = (char *)dest;
                v174 = *(_QWORD *)(*(_QWORD *)(v291 + 224) + 152LL);
                v175 = *(unsigned __int8 **)(v174 + 104);
                v176 = *(_QWORD *)(v174 + 112);
                v177 = v329 - (_BYTE *)dest;
                if ( v176 > v329 - (_BYTE *)dest )
                {
                  v247 = sub_CB6200((__int64)&v326, v175, v176);
                  v173 = *(char **)(v247 + 32);
                  v178 = (void **)v247;
                  v177 = *(_QWORD *)(v247 + 24) - (_QWORD)v173;
                }
                else
                {
                  v178 = (void **)&v326;
                  if ( v176 )
                  {
                    v281 = v176;
                    memcpy(dest, v175, v176);
                    v173 = (char *)dest + v281;
                    dest = v173;
                    v177 = v329 - v173;
                  }
                }
                v179 = *((_QWORD *)v107 + 5);
                v180 = (unsigned __int8 *)*((_QWORD *)v107 + 4);
                if ( v179 > v177 )
                {
                  sub_CB6200((__int64)v178, v180, v179);
                }
                else if ( v179 )
                {
                  v280 = *((_QWORD *)v107 + 5);
                  memcpy(v173, v180, v179);
                  v178[4] = (char *)v178[4] + v280;
                }
                goto LABEL_220;
              case 8:
                if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 1 )
                {
                  sub_CB6200((__int64)&v326, (unsigned __int8 *)"\n\t", 2u);
                }
                else
                {
                  *(_WORD *)dest = 2314;
                  dest = (char *)dest + 2;
                }
                goto LABEL_220;
              case 10:
                if ( v107[48] )
                {
                  if ( v329 == dest )
                  {
                    sub_CB6200((__int64)&v326, (unsigned __int8 *)"[", 1u);
                  }
                  else
                  {
                    *(_BYTE *)dest = 91;
                    dest = (char *)dest + 1;
                  }
                }
                v191 = *((_QWORD *)v107 + 9);
                if ( v191 )
                {
                  v245 = *((_QWORD *)v107 + 9);
                  v246 = (unsigned __int8 *)*((_QWORD *)v107 + 8);
                  if ( v191 > v329 - (_BYTE *)dest )
                  {
                    sub_CB6200((__int64)&v326, v246, v245);
                  }
                  else
                  {
                    memcpy(dest, v246, v245);
                    dest = (char *)dest + v191;
                  }
                }
                v192 = *((_QWORD *)v107 + 11);
                if ( !v192 )
                  goto LABEL_263;
                v193 = dest;
                if ( *((_QWORD *)v107 + 9) )
                {
                  if ( (unsigned __int64)(v329 - (_BYTE *)dest) > 2 )
                  {
                    *((_BYTE *)dest + 2) = 32;
                    v195 = (void **)&v326;
                    *v193 = 11040;
                    v193 = (char *)dest + 3;
                    dest = (char *)dest + 3;
                    v192 = *((_QWORD *)v107 + 11);
                  }
                  else
                  {
                    v194 = sub_CB6200((__int64)&v326, " + ", 3u);
                    v192 = *((_QWORD *)v107 + 11);
                    v193 = *(_WORD **)(v194 + 32);
                    v195 = (void **)v194;
                  }
                  v196 = (void *)*((_QWORD *)v107 + 10);
                  if ( (_BYTE *)v195[3] - (_BYTE *)v193 >= v192 )
                  {
                    if ( !v192 )
                      goto LABEL_263;
LABEL_262:
                    v278 = v195;
                    memcpy(v193, v196, v192);
                    v278[4] = (char *)v278[4] + v192;
                    goto LABEL_263;
                  }
                }
                else
                {
                  v196 = (void *)*((_QWORD *)v107 + 10);
                  v195 = (void **)&v326;
                  if ( v192 <= v329 - (_BYTE *)dest )
                    goto LABEL_262;
                }
                sub_CB6200((__int64)v195, (unsigned __int8 *)v196, v192);
LABEL_263:
                if ( *((_DWORD *)v107 + 28) > 1u )
                {
                  v242 = dest;
                  if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 4 )
                  {
                    v243 = (void **)sub_CB6200((__int64)&v326, " * $$", 5u);
                  }
                  else
                  {
                    *(_DWORD *)dest = 606087712;
                    v243 = (void **)&v326;
                    v242[4] = 36;
                    dest = (char *)dest + 5;
                  }
                  sub_CB59D0((__int64)v243, *((unsigned int *)v107 + 28));
                }
                v197 = *((_QWORD *)v107 + 13);
                if ( !v197 )
                  goto LABEL_282;
                v198 = (char *)dest;
                if ( *((_QWORD *)v107 + 9) || *((_QWORD *)v107 + 11) )
                {
                  if ( (unsigned __int64)(v329 - (_BYTE *)dest) <= 2 )
                  {
                    sub_CB6200((__int64)&v326, " + ", 3u);
                    v198 = (char *)dest;
                  }
                  else
                  {
                    *((_BYTE *)dest + 2) = 32;
                    *(_WORD *)v198 = 11040;
                    v198 = (char *)dest + 3;
                    dest = (char *)dest + 3;
                  }
                  v197 = *((_QWORD *)v107 + 13);
                }
                v199 = (unsigned __int8 *)*((_QWORD *)v107 + 12);
                v200 = (char *)v348 + 128 * (unsigned __int64)(unsigned int)v349;
                v201 = (v200 - v107) >> 9;
                v202 = (v200 - v107) >> 7;
                if ( v201 > 0 )
                {
                  v203 = (unsigned __int8 **)v107;
                  v204 = &v107[512 * v201];
                  while ( v199 != v203[1] || *((_DWORD *)v203 + 4) != v197 || (unsigned int)(*(_DWORD *)v203 - 3) > 1 )
                  {
                    if ( v199 == v203[17]
                      && *((_DWORD *)v203 + 36) == v197
                      && (unsigned int)(*((_DWORD *)v203 + 32) - 3) <= 1 )
                    {
                      v203 += 16;
                      goto LABEL_276;
                    }
                    if ( v199 == v203[33]
                      && *((_DWORD *)v203 + 68) == v197
                      && (unsigned int)(*((_DWORD *)v203 + 64) - 3) <= 1 )
                    {
                      v203 += 32;
                      goto LABEL_276;
                    }
                    if ( v199 == v203[49]
                      && *((_DWORD *)v203 + 100) == v197
                      && (unsigned int)(*((_DWORD *)v203 + 96) - 3) <= 1 )
                    {
                      v203 += 48;
                      goto LABEL_276;
                    }
                    v203 += 64;
                    if ( v204 == (char *)v203 )
                    {
                      v202 = (v200 - (char *)v203) >> 7;
                      goto LABEL_435;
                    }
                  }
                  goto LABEL_276;
                }
                v203 = (unsigned __int8 **)v107;
LABEL_435:
                switch ( v202 )
                {
                  case 2LL:
                    goto LABEL_448;
                  case 3LL:
                    if ( v199 == v203[1] && *((_DWORD *)v203 + 4) == v197 && (unsigned int)(*(_DWORD *)v203 - 3) <= 1 )
                      goto LABEL_276;
                    v203 += 16;
LABEL_448:
                    if ( v199 == v203[1] && *((_DWORD *)v203 + 4) == v197 && (unsigned int)(*(_DWORD *)v203 - 3) <= 1 )
                      goto LABEL_276;
                    v203 += 16;
                    break;
                  case 1LL:
                    break;
                  default:
                    v205 = v329;
                    goto LABEL_439;
                }
                v205 = v329;
                if ( v199 != v203[1] || *((_DWORD *)v203 + 4) != v197 || (unsigned int)(*(_DWORD *)v203 - 3) > 1 )
                  goto LABEL_439;
LABEL_276:
                v205 = v329;
                if ( v200 != (char *)v203 )
                {
                  if ( *(_DWORD *)v203 == 3 )
                  {
                    if ( (unsigned __int64)(v329 - v198) <= 1 )
                    {
                      v285 = v203;
                      v262 = sub_CB6200((__int64)&v326, (unsigned __int8 *)"${", 2u);
                      v203 = v285;
                      v252 = (void **)v262;
                    }
                    else
                    {
                      *(_WORD *)v198 = 31524;
                      v252 = (void **)&v326;
                      dest = (char *)dest + 2;
                    }
                    v282 = v203;
                    v207 = v275 + 1;
                    v253 = sub_CB59D0((__int64)v252, v275);
                    v208 = v282;
                    v254 = v253;
                    v255 = *(_QWORD *)(v253 + 32);
                    if ( (unsigned __int64)(*(_QWORD *)(v254 + 24) - v255) <= 2 )
                    {
                      sub_CB6200(v254, ":P}", 3u);
                      v208 = v282;
                    }
                    else
                    {
                      *(_BYTE *)(v255 + 2) = 125;
                      *(_WORD *)v255 = 20538;
                      *(_QWORD *)(v254 + 32) += 3LL;
                    }
                  }
                  else
                  {
                    if ( v198 >= v329 )
                    {
                      v283 = v203;
                      v259 = sub_CB5D20((__int64)&v326, 36);
                      v203 = v283;
                      v206 = (void **)v259;
                    }
                    else
                    {
                      v206 = (void **)&v326;
                      dest = v198 + 1;
                      *v198 = 36;
                    }
                    v279 = v203;
                    v207 = v275 + 1;
                    sub_CB59D0((__int64)v206, v275);
                    v208 = v279;
                  }
                  *((_BYTE *)v208 + 20) = 1;
                  v275 = v207;
                  goto LABEL_282;
                }
LABEL_439:
                if ( (unsigned __int64)(v205 - v198) <= 6 )
                {
                  v286 = v197;
                  v263 = sub_CB6200((__int64)&v326, (unsigned __int8 *)"offset ", 7u);
                  v197 = v286;
                  v261 = *(char **)(v263 + 32);
                  v260 = (void **)v263;
                }
                else
                {
                  *(_DWORD *)v198 = 1936090735;
                  v260 = (void **)&v326;
                  *((_WORD *)v198 + 2) = 29797;
                  v198[6] = 32;
                  v261 = (char *)dest + 7;
                  dest = (char *)dest + 7;
                }
                if ( (_BYTE *)v260[3] - v261 < v197 )
                {
                  sub_CB6200((__int64)v260, v199, v197);
                }
                else if ( v197 )
                {
                  v274 = v260;
                  v284 = v197;
                  memcpy(v261, v199, v197);
                  v274[4] = (char *)v274[4] + v284;
                }
LABEL_282:
                v209 = *((_QWORD *)v107 + 9);
                if ( *((_QWORD *)v107 + 7) )
                {
                  v210 = " + $$";
                  if ( v209 || *((_QWORD *)v107 + 11) )
                    goto LABEL_331;
                }
                else if ( v209 || *((_QWORD *)v107 + 11) || *((_QWORD *)v107 + 13) )
                {
                  goto LABEL_334;
                }
                v210 = " + $$";
                if ( !*((_QWORD *)v107 + 13) )
                  v210 = "$$";
LABEL_331:
                v232 = strlen(v210);
                v233 = dest;
                v234 = v232;
                if ( v329 - (_BYTE *)dest >= v232 )
                {
                  if ( (_DWORD)v232 )
                  {
                    v250 = 0;
                    do
                    {
                      v251 = v250++;
                      v233[v251] = v210[v251];
                    }
                    while ( v250 < (unsigned int)v234 );
                  }
                  dest = (char *)dest + v234;
                  v235 = (void **)&v326;
                }
                else
                {
                  v235 = (void **)sub_CB6200((__int64)&v326, (unsigned __int8 *)v210, v232);
                }
                sub_CB59F0((__int64)v235, *((_QWORD *)v107 + 7));
LABEL_334:
                v181 = 0;
                if ( v107[48] )
                {
                  if ( v329 == dest )
                  {
                    sub_CB6200((__int64)&v326, (unsigned __int8 *)"]", 1u);
                  }
                  else
                  {
                    *(_BYTE *)dest = 93;
                    dest = (char *)dest + 1;
                  }
LABEL_220:
                  v181 = 0;
                }
LABEL_221:
                v19 = v111 + v181 + *((unsigned int *)v107 + 4);
                break;
              default:
                goto LABEL_220;
            }
            goto LABEL_222;
          }
          goto LABEL_237;
        }
      }
      if ( v112 != 9 )
        goto LABEL_117;
LABEL_237:
      v19 = v111 + *((unsigned int *)v107 + 4);
      goto LABEL_222;
    }
LABEL_224:
    if ( v295 != v109 )
    {
      v19 = (signed __int64)v109;
      v182 = v295 - v109;
      if ( v295 - v109 > (unsigned __int64)(v329 - (_BYTE *)dest) )
      {
        sub_CB6200((__int64)&v326, v109, v182);
      }
      else
      {
        memcpy(dest, v109, v182);
        dest = (char *)dest + v295 - v109;
      }
    }
    v183 = n;
    v184 = (_BYTE *)*a2;
    if ( v320 == v322 )
    {
      if ( n )
      {
        if ( n == 1 )
        {
          *v184 = v322[0];
        }
        else
        {
          v19 = (signed __int64)v322;
          memcpy(v184, v322, n);
        }
        v183 = n;
        v184 = (_BYTE *)*a2;
      }
      a2[1] = v183;
      v184[v183] = 0;
      v184 = v320;
      goto LABEL_231;
    }
    v19 = v322[0];
    if ( v184 == (_BYTE *)(a2 + 2) )
    {
      *a2 = v320;
      a2[1] = v183;
      a2[2] = v19;
    }
    else
    {
      v185 = a2[2];
      *a2 = v320;
      a2[1] = v183;
      a2[2] = v19;
      if ( v184 )
      {
        v320 = v184;
        v322[0] = v185;
LABEL_231:
        n = 0;
        *v184 = 0;
        v326 = (char *)&unk_49DD210;
        sub_CB5840((__int64)&v326);
        if ( v320 != v322 )
        {
          v19 = v322[0] + 1LL;
          j_j___libc_free_0(v320, v322[0] + 1LL);
        }
        v159 = 0;
        goto LABEL_183;
      }
    }
    v320 = v322;
    v184 = v322;
    goto LABEL_231;
  }
  while ( 1 )
  {
    if ( (unsigned int)(v13 - 21) <= 1 )
    {
      v14 = sub_ECD690(v11 + 40);
      sub_EABFE0(v11);
      if ( **(_DWORD **)(v11 + 48) == 9 )
        sub_EABFE0(v11);
      v15 = sub_ECD690(v11 + 40) - v14;
      v16 = (unsigned int)v349;
      v17 = v349;
      if ( (unsigned int)v349 >= (unsigned __int64)HIDWORD(v349) )
      {
        v139 = (unsigned int)v349 + 1LL;
        v327 = v14;
        v140 = (char *)v348;
        v141 = (const __m128i *)&v326;
        LODWORD(v326) = 9;
        LODWORD(v328) = v15;
        BYTE4(v328) = 0;
        v329 = 0;
        dest = 0;
        v331 = 0;
        LOBYTE(v332) = 0;
        v333 = 0;
        v334 = 0;
        v335 = 0;
        v336 = 0;
        v337 = 0;
        v338 = 0;
        v339 = 0;
        v340 = 1;
        v341 = 0;
        if ( HIDWORD(v349) < v139 )
        {
          if ( v348 > &v326 || &v326 >= (char **)((char *)v348 + 128 * (unsigned __int64)(unsigned int)v349) )
          {
            sub_C8D5F0((__int64)&v348, v350, v139, 0x80u, v139, v12);
            v140 = (char *)v348;
            v16 = (unsigned int)v349;
          }
          else
          {
            v244 = (char *)&v326 - (_BYTE *)v348;
            sub_C8D5F0((__int64)&v348, v350, v139, 0x80u, v139, v12);
            v140 = (char *)v348;
            v16 = (unsigned int)v349;
            v141 = (const __m128i *)((char *)v348 + v244);
          }
        }
        v142 = (__m128i *)&v140[128 * v16];
        *v142 = _mm_loadu_si128(v141);
        v143 = _mm_loadu_si128(v141 + 1);
        LODWORD(v349) = v349 + 1;
        v142[1] = v143;
        v142[2] = _mm_loadu_si128(v141 + 2);
        v142[3] = _mm_loadu_si128(v141 + 3);
        v142[4] = _mm_loadu_si128(v141 + 4);
        v142[5] = _mm_loadu_si128(v141 + 5);
        v142[6] = _mm_loadu_si128(v141 + 6);
        v142[7] = _mm_loadu_si128(v141 + 7);
      }
      else
      {
        v18 = (char *)v348 + 128 * (unsigned __int64)(unsigned int)v349;
        if ( v18 )
        {
          *((_DWORD *)v18 + 4) = v15;
          v18[20] = 0;
          *((_QWORD *)v18 + 3) = 0;
          *((_QWORD *)v18 + 4) = 0;
          *((_QWORD *)v18 + 5) = 0;
          v18[48] = 0;
          *((_QWORD *)v18 + 7) = 0;
          *((_QWORD *)v18 + 8) = 0;
          *((_QWORD *)v18 + 9) = 0;
          *((_QWORD *)v18 + 10) = 0;
          *((_QWORD *)v18 + 11) = 0;
          *((_QWORD *)v18 + 12) = 0;
          *((_QWORD *)v18 + 13) = 0;
          *((_DWORD *)v18 + 28) = 1;
          v18[120] = 0;
          *(_DWORD *)v18 = 9;
          *((_QWORD *)v18 + 1) = v14;
          v17 = v349;
        }
        LODWORD(v349) = v17 + 1;
      }
      goto LABEL_9;
    }
    v324 = 0;
    v323 = -1;
    v320 = v322;
    v19 = (signed __int64)&v320;
    n = 0x800000000LL;
    v325 = &v348;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD **, __int64))(*(_QWORD *)v11 + 272LL))(v11, &v320, a10)
      || v324 )
    {
      break;
    }
    v298 = n;
    if ( v323 != -1 )
    {
      v287 = (unsigned __int16 *)(*a8 - 40LL * v323);
      if ( (_DWORD)n == 1 )
        goto LABEL_57;
      v276 = v11;
      v20 = 1;
      while ( 1 )
      {
        while ( 1 )
        {
          v31 = (_QWORD *)v320[v20];
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *))(*v31 + 48LL))(v31) )
            goto LABEL_25;
          v21 = *v31;
          v22 = *(__int64 (**)())(*v31 + 96LL);
          if ( v22 != sub_EA2160 )
            break;
LABEL_17:
          v23 = *(_QWORD *)(v276 + 8);
          v24 = *(__int64 (**)())(*(_QWORD *)v23 + 88LL);
          v25 = (*(__int64 (__fastcall **)(_QWORD *))(v21 + 56))(v31);
          if ( v24 == sub_EA21A0 || (v19 = v25, !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v24)(v23, v25)) )
          {
            if ( *((_BYTE *)v287 + 4) && *((_DWORD *)v31 + 2) < (unsigned int)*((unsigned __int8 *)v287 + 4) )
            {
              v26 = (*(__int64 (__fastcall **)(_QWORD *))(*v31 + 56LL))(v31);
              v29 = (unsigned int)v310;
              v30 = (unsigned int)v310 + 1LL;
              if ( v30 > HIDWORD(v310) )
              {
                v19 = (signed __int64)v311;
                sub_C8D5F0((__int64)&base, v311, v30, 4u, v27, v28);
                v29 = (unsigned int)v310;
              }
              *((_DWORD *)base + v29) = v26;
              LODWORD(v310) = v310 + 1;
            }
            goto LABEL_23;
          }
          v32 = *(__int64 (**)())(*v31 + 16LL);
          if ( v32 != sub_EA2130 )
            goto LABEL_26;
LABEL_23:
          if ( ++v20 == v298 )
            goto LABEL_56;
        }
        if ( !((unsigned __int8 (__fastcall *)(_QWORD *))v22)(v31) )
        {
          v21 = *v31;
          goto LABEL_17;
        }
LABEL_25:
        v32 = *(__int64 (**)())(*v31 + 16LL);
        if ( v32 == sub_EA2130 )
          goto LABEL_23;
LABEL_26:
        v289 = ((__int64 (__fastcall *)(_QWORD *))v32)(v31);
        v34 = v33;
        if ( !v33 )
          goto LABEL_23;
        v35 = *(__int64 (**)())(*v31 + 24LL);
        if ( v35 == sub_EA2140 )
          goto LABEL_23;
        v36 = ((__int64 (__fastcall *)(_QWORD *))v35)(v31);
        if ( !v36 )
          goto LABEL_23;
        v37 = (char *)v31[2];
        v292 = v31[3];
        v38 = (*(unsigned __int8 (__fastcall **)(_QWORD *))(*v31 + 40LL))(v31) == 0;
        v43 = *v31;
        if ( !v38 )
        {
          v39 = *(__int64 (**)())(v43 + 104);
          if ( v39 == sub_EA2170 || (v38 = ((unsigned __int8 (__fastcall *)(_QWORD *))v39)(v31) == 0, v43 = *v31, v38) )
          {
            v292 = 1;
            v37 = "i";
          }
          else
          {
            v292 = 1;
            v37 = "r";
          }
        }
        v44 = *(__int64 (**)())(v43 + 72);
        if ( v20 == 1 )
        {
          v40 = v287;
          if ( (v287[13] & 0x10) != 0 )
          {
            if ( v44 == sub_EA2150 )
              v270 = 0;
            else
              v270 = ((__int64 (__fastcall *)(_QWORD *))v44)(v31);
            v114 = (unsigned int)v318;
            ++v275;
            v115 = (unsigned int)v318 + 1LL;
            if ( v115 > HIDWORD(v318) )
            {
              sub_C8D5F0((__int64)&v317, v319, v115, 8u, v41, v42);
              v114 = (unsigned int)v318;
            }
            *(_QWORD *)&v317[8 * v114] = v36;
            v116 = 0;
            LODWORD(v318) = v318 + 1;
            v117 = *(__int64 (**)())(*v31 + 96LL);
            if ( v117 != sub_EA2160 )
              v116 = ((__int64 (__fastcall *)(_QWORD *))v117)(v31);
            v118 = v306;
            if ( v306 + 1 > v307 )
            {
              sub_C8D290((__int64)&v305, v308, v306 + 1, 1u, v41, v42);
              v118 = v306;
            }
            v305[v118] = v116;
            v326 = "=";
            v328 = v37;
            v329 = (char *)v292;
            ++v306;
            LOWORD(dest) = 1283;
            sub_CA0F50(v312, (void **)&v326);
            v120 = (unsigned int)v346;
            v121 = (unsigned int)v346 + 1LL;
            v122 = v346;
            if ( v121 > HIDWORD(v346) )
            {
              v137 = v345;
              v138 = (unsigned int)v346 + 1LL;
              if ( v345 > v312 || v312 >= &v345[4 * (unsigned int)v346] )
              {
                sub_95D880((__int64)&v345, v138);
                v120 = (unsigned int)v346;
                v123 = v345;
                v124 = (__m128i *)v312;
                v122 = v346;
              }
              else
              {
                sub_95D880((__int64)&v345, v138);
                v123 = v345;
                v120 = (unsigned int)v346;
                v124 = (__m128i *)((char *)v345 + (char *)v312 - (char *)v137);
                v122 = v346;
              }
            }
            else
            {
              v123 = v345;
              v124 = (__m128i *)v312;
            }
            v125 = (__m128i *)&v123[4 * v120];
            if ( v125 )
            {
              v125->m128i_i64[0] = (__int64)v125[1].m128i_i64;
              if ( (__m128i *)v124->m128i_i64[0] == &v124[1] )
              {
                v125[1] = _mm_loadu_si128(v124 + 1);
              }
              else
              {
                v125->m128i_i64[0] = v124->m128i_i64[0];
                v125[1].m128i_i64[0] = v124[1].m128i_i64[0];
              }
              v126 = v124->m128i_i64[1];
              v124->m128i_i64[0] = (__int64)v124[1].m128i_i64;
              v124->m128i_i64[1] = 0;
              v125->m128i_i64[1] = v126;
              v122 = v346;
              v124[1].m128i_i8[0] = 0;
            }
            LODWORD(v346) = v122 + 1;
            if ( (__int64 *)v312[0] != &v313 )
              j_j___libc_free_0(v312[0], v313 + 1);
            v127 = (unsigned int)v349;
            v19 = (signed __int64)v348;
            v59 = v349;
            v128 = (char *)v348 + 128 * (unsigned __int64)(unsigned int)v349;
            if ( (unsigned int)v349 < (unsigned __int64)HIDWORD(v349) )
            {
              if ( v128 )
              {
                *((_DWORD *)v128 + 4) = v34;
                *(_DWORD *)v128 = 5;
                *((_QWORD *)v128 + 1) = v289;
                v59 = v349;
                v128[20] = 0;
                *((_QWORD *)v128 + 3) = 0;
                *((_QWORD *)v128 + 4) = 0;
                *((_QWORD *)v128 + 5) = 0;
                v128[48] = 0;
                *((_QWORD *)v128 + 7) = 0;
                *((_QWORD *)v128 + 8) = 0;
                *((_QWORD *)v128 + 9) = 0;
                *((_QWORD *)v128 + 10) = 0;
                *((_QWORD *)v128 + 11) = 0;
                *((_QWORD *)v128 + 12) = 0;
                *((_QWORD *)v128 + 13) = 0;
                *((_DWORD *)v128 + 28) = 1;
                v128[120] = v270;
              }
              goto LABEL_55;
            }
            LODWORD(v328) = v34;
            LODWORD(v326) = 5;
            v327 = v289;
            v133 = (unsigned int)v349 + 1LL;
            v341 = v270;
            v134 = (const __m128i *)&v326;
            BYTE4(v328) = 0;
            v329 = 0;
            dest = 0;
            v331 = 0;
            LOBYTE(v332) = 0;
            v333 = 0;
            v334 = 0;
            v335 = 0;
            v336 = 0;
            v337 = 0;
            v338 = 0;
            v339 = 0;
            v340 = 1;
            if ( HIDWORD(v349) < v133 )
            {
              if ( v348 > &v326 || v128 <= (char *)&v326 )
              {
                v134 = (const __m128i *)&v326;
                sub_C8D5F0((__int64)&v348, v350, v133, 0x80u, v121, v119);
                v19 = (signed __int64)v348;
                v127 = (unsigned int)v349;
              }
              else
              {
                v239 = (char *)&v326 - (_BYTE *)v348;
                sub_C8D5F0((__int64)&v348, v350, v133, 0x80u, v121, v119);
                v19 = (signed __int64)v348;
                v127 = (unsigned int)v349;
                v134 = (const __m128i *)((char *)v348 + v239);
              }
            }
            v135 = (__m128i *)(v19 + (v127 << 7));
            *v135 = _mm_loadu_si128(v134);
            v136 = _mm_loadu_si128(v134 + 1);
            LODWORD(v349) = v349 + 1;
            v135[1] = v136;
            v135[2] = _mm_loadu_si128(v134 + 2);
            v135[3] = _mm_loadu_si128(v134 + 3);
            v135[4] = _mm_loadu_si128(v134 + 4);
            v135[5] = _mm_loadu_si128(v134 + 5);
            v135[6] = _mm_loadu_si128(v134 + 6);
            v135[7] = _mm_loadu_si128(v134 + 7);
            goto LABEL_23;
          }
        }
        v45 = 0;
        if ( v44 != sub_EA2150 )
          v45 = ((unsigned int (__fastcall *)(_QWORD *, signed __int64, __int64 (*)(), unsigned __int16 *, _QWORD))v44)(
                  v31,
                  v19,
                  v39,
                  v40,
                  0);
        v46 = (unsigned int)v315;
        v47 = (unsigned int)v315 + 1LL;
        if ( v47 > HIDWORD(v315) )
        {
          v272 = v45;
          sub_C8D5F0((__int64)&v314, v316, v47, 8u, v45, v42);
          v46 = (unsigned int)v315;
          v45 = v272;
        }
        *(_QWORD *)&v314[8 * v46] = v36;
        v48 = 0;
        LODWORD(v315) = v315 + 1;
        v49 = *(__int64 (**)())(*v31 + 96LL);
        if ( v49 != sub_EA2160 )
        {
          v271 = v45;
          v129 = ((__int64 (__fastcall *)(_QWORD *))v49)(v31);
          v45 = v271;
          v48 = v129;
        }
        v50 = v302;
        if ( v302 + 1 > v303 )
        {
          v273 = v45;
          sub_C8D290((__int64)&v301, v304, v302 + 1, 1u, v45, v42);
          v50 = v302;
          v45 = v273;
        }
        v301[v50] = v48;
        ++v302;
        if ( v37 )
        {
          v269 = v45;
          v326 = (char *)&v328;
          sub_EA2A30((__int64 *)&v326, v37, (__int64)&v37[v292]);
          v45 = v269;
        }
        else
        {
          LOBYTE(v328) = 0;
          v326 = (char *)&v328;
          v327 = 0;
        }
        v51 = (unsigned int)v343;
        v52 = (__m128i *)&v326;
        v53 = v342;
        v54 = v343;
        if ( (unsigned __int64)(unsigned int)v343 + 1 > HIDWORD(v343) )
        {
          v296 = v45;
          if ( v342 > &v326 || &v326 >= &v342[4 * (unsigned int)v343] )
          {
            v52 = (__m128i *)&v326;
            sub_95D880((__int64)&v342, (unsigned int)v343 + 1LL);
            v51 = (unsigned int)v343;
            v53 = v342;
            v45 = v296;
            v54 = v343;
          }
          else
          {
            v132 = (char *)((char *)&v326 - (char *)v342);
            sub_95D880((__int64)&v342, (unsigned int)v343 + 1LL);
            v53 = v342;
            v51 = (unsigned int)v343;
            v45 = v296;
            v52 = (__m128i *)&v132[(_QWORD)v342];
            v54 = v343;
          }
        }
        v55 = (__m128i *)&v53[4 * v51];
        if ( v55 )
        {
          v55->m128i_i64[0] = (__int64)v55[1].m128i_i64;
          if ( (__m128i *)v52->m128i_i64[0] == &v52[1] )
          {
            v55[1] = _mm_loadu_si128(v52 + 1);
          }
          else
          {
            v55->m128i_i64[0] = v52->m128i_i64[0];
            v55[1].m128i_i64[0] = v52[1].m128i_i64[0];
          }
          v56 = v52->m128i_i64[1];
          v52->m128i_i64[0] = (__int64)v52[1].m128i_i64;
          v52->m128i_i64[1] = 0;
          v55->m128i_i64[1] = v56;
          v54 = v343;
          v52[1].m128i_i8[0] = 0;
        }
        LODWORD(v343) = v54 + 1;
        if ( v326 != (char *)&v328 )
        {
          v293 = v45;
          j_j___libc_free_0(v326, v328 + 1);
          v45 = v293;
        }
        v57 = (unsigned int)v349;
        v58 = (char *)v348;
        v59 = v349;
        v60 = (char *)v348 + 128 * (unsigned __int64)(unsigned int)v349;
        v19 = 6LL * v287[8] + 8 * (5LL * *v287 + 5);
        if ( (*((_BYTE *)&v287[3 * v20 - 2] + v19) & 8) != 0 )
        {
          if ( (unsigned int)v349 >= (unsigned __int64)HIDWORD(v349) )
          {
            v165 = (unsigned int)v349 + 1LL;
            LODWORD(v328) = v34;
            v166 = (const __m128i *)&v326;
            LODWORD(v326) = 3;
            v327 = v289;
            BYTE4(v328) = 0;
            v329 = 0;
            dest = 0;
            v331 = 0;
            LOBYTE(v332) = 0;
            v333 = 0;
            v334 = 0;
            v335 = 0;
            v336 = 0;
            v337 = 0;
            v338 = 0;
            v339 = 0;
            v340 = 1;
            v341 = v45;
            if ( v165 > HIDWORD(v349) )
            {
              if ( v348 > &v326 || v60 <= (char *)&v326 )
              {
                v19 = (signed __int64)v350;
                v166 = (const __m128i *)&v326;
                sub_C8D5F0((__int64)&v348, v350, v165, 0x80u, v45, (__int64)v348);
                v58 = (char *)v348;
                v57 = (unsigned int)v349;
              }
              else
              {
                v19 = (signed __int64)v350;
                v231 = (char *)&v326 - (_BYTE *)v348;
                sub_C8D5F0((__int64)&v348, v350, v165, 0x80u, v45, (__int64)v348);
                v58 = (char *)v348;
                v57 = (unsigned int)v349;
                v166 = (const __m128i *)((char *)v348 + v231);
              }
            }
            v167 = (__m128i *)&v58[128 * v57];
            *v167 = _mm_loadu_si128(v166);
            v168 = _mm_loadu_si128(v166 + 1);
            LODWORD(v349) = v349 + 1;
            v167[1] = v168;
            v167[2] = _mm_loadu_si128(v166 + 2);
            v167[3] = _mm_loadu_si128(v166 + 3);
            v167[4] = _mm_loadu_si128(v166 + 4);
            v167[5] = _mm_loadu_si128(v166 + 5);
            v167[6] = _mm_loadu_si128(v166 + 6);
            v167[7] = _mm_loadu_si128(v166 + 7);
            goto LABEL_23;
          }
          if ( v60 )
          {
            *(_DWORD *)v60 = 3;
LABEL_54:
            *((_DWORD *)v60 + 4) = v34;
            v60[20] = 0;
            *((_QWORD *)v60 + 1) = v289;
            *((_QWORD *)v60 + 3) = 0;
            *((_QWORD *)v60 + 4) = 0;
            *((_QWORD *)v60 + 5) = 0;
            v60[48] = 0;
            *((_QWORD *)v60 + 7) = 0;
            *((_QWORD *)v60 + 8) = 0;
            *((_QWORD *)v60 + 9) = 0;
            *((_QWORD *)v60 + 10) = 0;
            *((_QWORD *)v60 + 11) = 0;
            *((_QWORD *)v60 + 12) = 0;
            *((_QWORD *)v60 + 13) = 0;
            *((_DWORD *)v60 + 28) = 1;
            v60[120] = v45;
            v59 = v349;
          }
        }
        else
        {
          if ( (unsigned int)v349 >= (unsigned __int64)HIDWORD(v349) )
          {
            v169 = (unsigned int)v349 + 1LL;
            LODWORD(v328) = v34;
            v170 = (const __m128i *)&v326;
            LODWORD(v326) = 4;
            v327 = v289;
            BYTE4(v328) = 0;
            v329 = 0;
            dest = 0;
            v331 = 0;
            LOBYTE(v332) = 0;
            v333 = 0;
            v334 = 0;
            v335 = 0;
            v336 = 0;
            v337 = 0;
            v338 = 0;
            v339 = 0;
            v340 = 1;
            v341 = v45;
            if ( v169 > HIDWORD(v349) )
            {
              if ( v348 > &v326 || v60 <= (char *)&v326 )
              {
                v19 = (signed __int64)v350;
                v170 = (const __m128i *)&v326;
                sub_C8D5F0((__int64)&v348, v350, v169, 0x80u, v45, (__int64)v348);
                v58 = (char *)v348;
                v57 = (unsigned int)v349;
              }
              else
              {
                v19 = (signed __int64)v350;
                v230 = (char *)&v326 - (_BYTE *)v348;
                sub_C8D5F0((__int64)&v348, v350, v169, 0x80u, v45, (__int64)v348);
                v58 = (char *)v348;
                v57 = (unsigned int)v349;
                v170 = (const __m128i *)((char *)v348 + v230);
              }
            }
            v171 = (__m128i *)&v58[128 * v57];
            *v171 = _mm_loadu_si128(v170);
            v172 = _mm_loadu_si128(v170 + 1);
            LODWORD(v349) = v349 + 1;
            v171[1] = v172;
            v171[2] = _mm_loadu_si128(v170 + 2);
            v171[3] = _mm_loadu_si128(v170 + 3);
            v171[4] = _mm_loadu_si128(v170 + 4);
            v171[5] = _mm_loadu_si128(v170 + 5);
            v171[6] = _mm_loadu_si128(v170 + 6);
            v171[7] = _mm_loadu_si128(v170 + 7);
            goto LABEL_23;
          }
          if ( v60 )
          {
            *(_DWORD *)v60 = 4;
            goto LABEL_54;
          }
        }
LABEL_55:
        ++v20;
        LODWORD(v349) = v59 + 1;
        if ( v20 == v298 )
        {
LABEL_56:
          v11 = v276;
LABEL_57:
          v61 = (unsigned int)v310;
          v62 = *((unsigned __int8 *)v287 + 9);
          v12 = *v287;
          v63 = *((unsigned int *)v287 + 3);
          v64 = *((unsigned __int8 *)v287 + 8);
          v65 = (unsigned int)v310 + v62;
          if ( v65 > HIDWORD(v310) )
          {
            v19 = (signed __int64)v311;
            v300 = *v287;
            sub_C8D5F0((__int64)&base, v311, (unsigned int)v310 + v62, 4u, v65, v12);
            v61 = (unsigned int)v310;
            v12 = v300;
          }
          v66 = &v287[20 * (unsigned __int16)v12 + 20 + v64 + v63];
          v67 = &v66[v62];
          v68 = (char *)base + 4 * v61;
          if ( v67 != v66 )
          {
            do
            {
              if ( v68 )
                *v68 = *v66;
              ++v66;
              ++v68;
            }
            while ( v67 != v66 );
            v61 = (unsigned int)v310;
          }
          v69 = v62 + v61;
          v70 = v320;
          LODWORD(v310) = v69;
          v71 = &v320[(unsigned int)n];
          if ( v320 != v71 )
          {
            do
            {
              v72 = *--v71;
              if ( v72 )
                (*(void (__fastcall **)(__int64, signed __int64, __int64, _DWORD *))(*(_QWORD *)v72 + 8LL))(
                  v72,
                  v19,
                  v69,
                  v68);
            }
            while ( v70 != v71 );
            goto LABEL_68;
          }
          goto LABEL_69;
        }
      }
    }
    v130 = v320;
    v71 = &v320[(unsigned int)n];
    if ( v320 != v71 )
    {
      do
      {
        v131 = *--v71;
        if ( v131 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v131 + 8LL))(v131);
      }
      while ( v130 != v71 );
LABEL_68:
      v71 = v320;
    }
LABEL_69:
    if ( v71 == v322 )
    {
LABEL_9:
      v13 = **(_DWORD **)(v11 + 48);
      if ( !v13 )
        goto LABEL_71;
    }
    else
    {
      _libc_free(v71, v19);
      v13 = **(_DWORD **)(v11 + 48);
      if ( !v13 )
      {
LABEL_71:
        v10 = v11;
        goto LABEL_72;
      }
    }
  }
  v144 = v11;
  v145 = *(_QWORD *)(v11 + 16);
  v146 = *(unsigned int *)(v144 + 24);
  if ( v145 != v145 + 112 * v146 )
  {
    v297 = v145 + 112 * v146;
    v147 = v312;
    do
    {
      LOWORD(dest) = 261;
      v148 = *(char **)(v145 + 8);
      v145 += 112;
      v149 = *(__int64 **)(v144 + 248);
      v299 = v147;
      v326 = v148;
      v327 = *(_QWORD *)(v145 - 96);
      v19 = *(_QWORD *)(v145 - 112);
      v150 = *(_QWORD *)(v145 - 16);
      v151 = *(_QWORD *)(v145 - 8);
      *(_BYTE *)(v144 + 32) = 1;
      v312[0] = v150;
      v312[1] = v151;
      sub_C91CB0(v149, v19, 0, (__int64)&v326, (__int64)v147, 1, 0, 0, 1u);
      sub_EA2AE0((_QWORD *)v144);
      v147 = v299;
    }
    while ( v297 != v145 );
    v152 = *(_QWORD *)(v144 + 16);
    v153 = v152 + 112LL * *(unsigned int *)(v144 + 24);
    while ( v152 != v153 )
    {
      while ( 1 )
      {
        v153 -= 112;
        v154 = *(_QWORD *)(v153 + 8);
        if ( v154 == v153 + 32 )
          break;
        _libc_free(v154, v19);
        if ( v152 == v153 )
          goto LABEL_175;
      }
    }
  }
LABEL_175:
  v155 = v320;
  v156 = (unsigned int)n;
  *(_DWORD *)(v144 + 24) = 0;
  v157 = &v155[v156];
  if ( v155 != v157 )
  {
    do
    {
      v158 = *--v157;
      if ( v158 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v158 + 8LL))(v158);
    }
    while ( v155 != v157 );
    v157 = v320;
  }
  if ( v157 != v322 )
    _libc_free(v157, v19);
  v159 = 1;
LABEL_183:
  if ( v348 != v350 )
    _libc_free(v348, v19);
  if ( base != v311 )
    _libc_free(base, v19);
  v160 = v345;
  v161 = &v345[4 * (unsigned int)v346];
  if ( v345 != v161 )
  {
    do
    {
      v161 -= 4;
      if ( (__int64 *)*v161 != v161 + 2 )
      {
        v19 = v161[2] + 1;
        j_j___libc_free_0(*v161, v19);
      }
    }
    while ( v160 != v161 );
    v161 = v345;
  }
  if ( v161 != (__int64 *)v347 )
    _libc_free(v161, v19);
  v162 = v342;
  v163 = &v342[4 * (unsigned int)v343];
  if ( v342 != v163 )
  {
    do
    {
      v163 -= 4;
      if ( *v163 != (char *)(v163 + 2) )
      {
        v19 = (signed __int64)(v163[2] + 1);
        j_j___libc_free_0(*v163, v19);
      }
    }
    while ( v162 != v163 );
    v163 = v342;
  }
  if ( v163 != (char **)v344 )
    _libc_free(v163, v19);
  if ( v305 != v308 )
    _libc_free(v305, v19);
  if ( v301 != v304 )
    _libc_free(v301, v19);
  if ( v317 != v319 )
    _libc_free(v317, v19);
  if ( v314 != v316 )
    _libc_free(v314, v19);
  return v159;
}
