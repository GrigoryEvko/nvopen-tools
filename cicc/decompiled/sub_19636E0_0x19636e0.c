// Function: sub_19636E0
// Address: 0x19636e0
//
__int64 __fastcall sub_19636E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v15; // rax
  __int64 v16; // rbx
  char *v17; // rbx
  char *v18; // r15
  char *v19; // rdi
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  _QWORD *v22; // rdi
  __int64 *v23; // r14
  __int64 *v24; // r15
  __int64 *v25; // r12
  __int64 *v26; // r15
  _QWORD *v27; // rax
  __int64 *v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // r8
  int v33; // r9d
  unsigned int v34; // ecx
  __int64 *v35; // rdx
  __int64 v36; // rdi
  __int64 *v37; // rax
  char v38; // cl
  __int64 v39; // rsi
  __int64 v40; // rdi
  unsigned int v41; // esi
  __int64 *v42; // rdx
  __int64 v43; // r10
  __int64 *v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // r12
  unsigned __int64 v49; // rax
  __int64 v50; // rcx
  _QWORD *v51; // rax
  _QWORD *v53; // rax
  __int64 v54; // r15
  __int64 i; // r12
  unsigned int **v56; // rax
  unsigned int *v57; // r8
  void *v58; // r8
  unsigned int v59; // esi
  int v60; // edx
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rax
  int v64; // edx
  _QWORD *v65; // rsi
  int v66; // edi
  unsigned int v67; // eax
  void **v68; // rcx
  void *v69; // r8
  int v70; // edx
  __int64 v71; // rax
  char *v72; // rcx
  __int64 v73; // rsi
  __int64 v74; // rax
  void **v75; // rdi
  void **v76; // rax
  char *v77; // rsi
  int v78; // edx
  char *v79; // rax
  __int64 v80; // rsi
  __int64 **v81; // rax
  __int64 *v82; // r14
  char v83; // bl
  _QWORD *v84; // rax
  int v85; // edx
  unsigned int v86; // edi
  __int64 *v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r13
  _QWORD *v90; // r12
  double v91; // xmm4_8
  double v92; // xmm5_8
  char v93; // si
  unsigned int v94; // edx
  unsigned int v95; // esi
  int v96; // r9d
  __int64 v97; // r8
  unsigned int v98; // ecx
  __int64 *v99; // r12
  __int64 v100; // rdi
  _QWORD *v101; // r11
  void *v102; // rcx
  __int64 v103; // rdi
  _QWORD *v104; // rdx
  __int64 v105; // r10
  __int64 v106; // rdi
  __int64 v107; // rax
  __int64 v108; // rcx
  unsigned __int64 v109; // rsi
  int v110; // eax
  _QWORD *v111; // rax
  int v112; // edi
  int v113; // r8d
  _QWORD *v114; // rax
  __int64 v115; // rax
  __int64 *v116; // rdi
  __int64 *v117; // rsi
  unsigned int v118; // esi
  int v119; // r11d
  int v120; // ecx
  int v121; // r9d
  int v122; // edx
  __int64 v123; // rax
  __int64 v124; // rcx
  unsigned __int64 v125; // rdx
  __int64 v126; // rcx
  int v127; // edx
  __int64 v128; // rax
  __int64 v129; // rcx
  unsigned __int64 v130; // rsi
  __int64 v131; // rcx
  __int64 v132; // rcx
  unsigned __int64 v133; // rax
  __int64 *v134; // rax
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // rax
  unsigned __int64 v138; // rdi
  unsigned __int64 v139; // r12
  __int64 v140; // rax
  unsigned int v141; // r9d
  __int64 v142; // r8
  __int64 v143; // r15
  bool v144; // sf
  __int64 v145; // rax
  __int64 v146; // rdx
  __int64 v147; // rbx
  __int64 v148; // rax
  __int64 v149; // r14
  __int64 v150; // rbx
  __int64 v151; // rax
  unsigned int *v152; // rax
  _BYTE *v153; // rdx
  __int64 v154; // r15
  __int64 v155; // r12
  __int64 v156; // r13
  unsigned int v157; // eax
  __int64 *v158; // rcx
  __int64 v159; // r11
  _QWORD *v160; // rcx
  __m128i *v161; // rdi
  __int64 v162; // rax
  __int64 v163; // rax
  char *v164; // r13
  char *v165; // r12
  __int64 v166; // rax
  char *v167; // r15
  unsigned __int64 v168; // rax
  unsigned int v169; // eax
  __int64 v170; // rax
  __int64 *v171; // rbx
  __int64 *v172; // r15
  __int64 v173; // rdi
  __int64 *v174; // rbx
  __int64 v175; // rcx
  __int64 v176; // rax
  __int64 v177; // rdi
  __int64 v178; // rdx
  __int64 v179; // rax
  __int64 *v180; // r10
  __int64 v181; // r14
  _QWORD *v182; // r15
  __int64 *v183; // r13
  __int64 *v184; // r12
  int v185; // ecx
  __int64 v186; // rsi
  int v187; // ecx
  __int64 v188; // rdi
  unsigned int v189; // edx
  __int64 *v190; // rax
  __int64 v191; // r9
  __int64 v192; // rax
  _QWORD *v193; // r8
  int v194; // ecx
  unsigned int v195; // edx
  __int64 v196; // r10
  unsigned int v197; // eax
  unsigned int v198; // ecx
  __int64 v199; // rsi
  unsigned int v200; // edx
  __int64 *v201; // rax
  __int64 v202; // r9
  __int64 v203; // rdx
  unsigned __int64 **v204; // rax
  unsigned __int64 v205; // rdi
  __int64 v206; // rbx
  unsigned __int64 v207; // rax
  __int64 v208; // rdx
  __m128 *v209; // r13
  __int64 *v210; // rax
  __int64 v211; // r9
  size_t v212; // rdx
  __int64 v213; // rsi
  __int64 v214; // rax
  __int64 v215; // rbx
  __int64 v216; // r9
  __int64 v217; // rdx
  __int64 v218; // r8
  __int64 *v219; // r14
  unsigned __int64 v220; // r13
  unsigned __int64 v221; // r12
  _QWORD *v222; // rcx
  __int64 v223; // rcx
  int v224; // eax
  __int64 v225; // rax
  int v226; // esi
  __int64 v227; // rsi
  __int64 **v228; // rax
  __int64 *v229; // rdi
  unsigned __int64 v230; // rsi
  __int64 v231; // rsi
  __int64 v232; // rsi
  __int64 v233; // rax
  __int64 v234; // rdi
  __int64 v235; // rdx
  unsigned __int64 v236; // rcx
  __int64 v237; // rdx
  size_t v238; // rdx
  int v239; // ecx
  void *v240; // rax
  void **v241; // r9
  int v242; // eax
  int v243; // r8d
  char *v244; // r10
  size_t v245; // rdx
  unsigned __int64 v246; // rsi
  bool v247; // cf
  unsigned __int64 v248; // rax
  __int64 v249; // rsi
  __int64 v250; // rax
  char *v251; // r11
  char *v252; // rax
  char *v253; // rdx
  char *v254; // rax
  char *v255; // rsi
  char *v256; // rdi
  __int64 v257; // r12
  char *v258; // rax
  char *v259; // r12
  size_t v260; // rdx
  __int64 v261; // rax
  int v262; // r8d
  int v263; // r9d
  __int64 v264; // rsi
  __int64 *v265; // r10
  int v266; // edx
  int v267; // eax
  void *v268; // rsi
  __int64 v269; // rdi
  unsigned int v270; // r9d
  int v271; // r10d
  _QWORD *v272; // rdx
  int v273; // r10d
  int v274; // r10d
  __int64 v275; // r9
  __int64 *v276; // rcx
  int v277; // r8d
  unsigned int v278; // r11d
  __int64 v279; // rsi
  int v280; // r11d
  int v281; // r11d
  __int64 v282; // r10
  unsigned int v283; // ecx
  __int64 v284; // rdi
  int v285; // r9d
  __int64 *v286; // rsi
  int v287; // eax
  __int64 v288; // rdi
  int v289; // r10d
  unsigned int v290; // r9d
  unsigned __int64 **v291; // rax
  int v292; // r10d
  int v293; // r11d
  __int64 *v294; // r10
  __int64 *v295; // rcx
  int v296; // edi
  _QWORD *v297; // r8
  int v298; // ecx
  unsigned int v299; // ebx
  __int64 *v300; // r9
  int v301; // esi
  __int64 *v302; // rdx
  __int64 v303; // rdi
  int v304; // r9d
  int v305; // r13d
  unsigned int v306; // edi
  __int64 *v307; // r9
  __int64 *v308; // rsi
  int v309; // edi
  int v310; // eax
  int v311; // r8d
  _QWORD *v312; // [rsp+8h] [rbp-588h]
  char *v313; // [rsp+8h] [rbp-588h]
  __int64 *v314; // [rsp+10h] [rbp-580h]
  __int64 v315; // [rsp+10h] [rbp-580h]
  _QWORD *v316; // [rsp+18h] [rbp-578h]
  __int64 v317; // [rsp+18h] [rbp-578h]
  __int64 *v318; // [rsp+18h] [rbp-578h]
  __int64 v319; // [rsp+20h] [rbp-570h]
  __m128i *v320; // [rsp+28h] [rbp-568h]
  __int64 *v321; // [rsp+28h] [rbp-568h]
  size_t v322; // [rsp+28h] [rbp-568h]
  _QWORD *v323; // [rsp+28h] [rbp-568h]
  char *v324; // [rsp+28h] [rbp-568h]
  _QWORD *v325; // [rsp+30h] [rbp-560h]
  __int64 v326; // [rsp+30h] [rbp-560h]
  __int64 v327; // [rsp+30h] [rbp-560h]
  _QWORD *v328; // [rsp+30h] [rbp-560h]
  char *v329; // [rsp+30h] [rbp-560h]
  char *v330; // [rsp+30h] [rbp-560h]
  _QWORD *v331; // [rsp+30h] [rbp-560h]
  _QWORD *srce; // [rsp+38h] [rbp-558h]
  int src; // [rsp+38h] [rbp-558h]
  void *srcf; // [rsp+38h] [rbp-558h]
  void *srca; // [rsp+38h] [rbp-558h]
  _BYTE *srcd; // [rsp+38h] [rbp-558h]
  int srcb; // [rsp+38h] [rbp-558h]
  unsigned int srcc; // [rsp+38h] [rbp-558h]
  int v339; // [rsp+40h] [rbp-550h]
  int v340; // [rsp+40h] [rbp-550h]
  _QWORD *v341; // [rsp+40h] [rbp-550h]
  __int64 v342; // [rsp+40h] [rbp-550h]
  __int64 v343; // [rsp+40h] [rbp-550h]
  _QWORD *v344; // [rsp+40h] [rbp-550h]
  __int64 v345; // [rsp+48h] [rbp-548h]
  unsigned __int64 v346; // [rsp+48h] [rbp-548h]
  int v347; // [rsp+48h] [rbp-548h]
  int v348; // [rsp+48h] [rbp-548h]
  __int64 v349; // [rsp+48h] [rbp-548h]
  __int64 *v350; // [rsp+50h] [rbp-540h]
  unsigned __int8 v352; // [rsp+60h] [rbp-530h]
  __int64 v353; // [rsp+60h] [rbp-530h]
  __int64 *v354; // [rsp+60h] [rbp-530h]
  __int64 v357; // [rsp+70h] [rbp-520h]
  __int64 *v358; // [rsp+78h] [rbp-518h]
  __int64 v360; // [rsp+80h] [rbp-510h]
  __int64 v361; // [rsp+88h] [rbp-508h]
  __m128i *v362; // [rsp+90h] [rbp-500h] BYREF
  size_t n; // [rsp+98h] [rbp-4F8h]
  __m128i v364; // [rsp+A0h] [rbp-4F0h] BYREF
  __int64 *v365; // [rsp+B0h] [rbp-4E0h] BYREF
  const char *v366; // [rsp+B8h] [rbp-4D8h]
  _WORD v367[32]; // [rsp+C0h] [rbp-4D0h] BYREF
  unsigned __int64 *v368; // [rsp+100h] [rbp-490h] BYREF
  __int64 *v369; // [rsp+108h] [rbp-488h]
  __int64 *v370; // [rsp+110h] [rbp-480h] BYREF
  __int64 v371; // [rsp+118h] [rbp-478h]
  __int64 *v372; // [rsp+120h] [rbp-470h]
  char v373; // [rsp+128h] [rbp-468h] BYREF
  __int64 v374; // [rsp+130h] [rbp-460h] BYREF
  void *v375; // [rsp+170h] [rbp-420h] BYREF
  __int64 v376; // [rsp+178h] [rbp-418h]
  __int64 v377; // [rsp+180h] [rbp-410h] BYREF
  __m128 v378; // [rsp+188h] [rbp-408h]
  __int64 v379; // [rsp+198h] [rbp-3F8h]
  __int64 v380; // [rsp+1A0h] [rbp-3F0h]
  __m128 v381; // [rsp+1A8h] [rbp-3E8h]
  __int64 v382; // [rsp+1B8h] [rbp-3D8h]
  __int64 *v383; // [rsp+1C0h] [rbp-3D0h] BYREF
  __int64 v384; // [rsp+1C8h] [rbp-3C8h] BYREF
  __int64 v385; // [rsp+1D0h] [rbp-3C0h] BYREF
  _BYTE v386[356]; // [rsp+1D8h] [rbp-3B8h] BYREF
  int v387; // [rsp+33Ch] [rbp-254h]
  __int64 v388; // [rsp+340h] [rbp-250h]
  char *v389; // [rsp+350h] [rbp-240h] BYREF
  __int64 v390; // [rsp+358h] [rbp-238h]
  _QWORD *v391; // [rsp+360h] [rbp-230h] BYREF
  unsigned int v392; // [rsp+368h] [rbp-228h]
  unsigned int *v393; // [rsp+3A0h] [rbp-1F0h] BYREF
  __int64 v394; // [rsp+3A8h] [rbp-1E8h]
  unsigned int v395; // [rsp+3B0h] [rbp-1E0h] BYREF
  char v396; // [rsp+3B8h] [rbp-1D8h] BYREF
  char v397; // [rsp+560h] [rbp-30h] BYREF

  v361 = a1;
  v15 = sub_15E0530(*a6);
  if ( sub_1602790(v15)
    || (v135 = sub_15E0530(*a6),
        v136 = sub_16033E0(v135),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v136 + 48LL))(v136)) )
  {
    sub_15CA3B0((__int64)&v389, (__int64)"licm", (__int64)"InstSunk", 8, v361);
    sub_15CAB20((__int64)&v389, "sinking ", 8u);
    sub_15C9340((__int64)&v368, "Inst", 4u, v361);
    v16 = sub_17C2270((__int64)&v389, (__int64)&v368);
    LODWORD(v376) = *(_DWORD *)(v16 + 8);
    BYTE4(v376) = *(_BYTE *)(v16 + 12);
    v377 = *(_QWORD *)(v16 + 16);
    a7 = (__m128)_mm_loadu_si128((const __m128i *)(v16 + 24));
    v378 = a7;
    v379 = *(_QWORD *)(v16 + 40);
    v375 = &unk_49ECF68;
    v380 = *(_QWORD *)(v16 + 48);
    a8 = _mm_loadu_si128((const __m128i *)(v16 + 56));
    v381 = (__m128)a8;
    LOBYTE(v383) = *(_BYTE *)(v16 + 80);
    if ( (_BYTE)v383 )
      v382 = *(_QWORD *)(v16 + 72);
    v384 = (__int64)v386;
    v385 = 0x400000000LL;
    if ( *(_DWORD *)(v16 + 96) )
      sub_195ED40((__int64)&v384, v16 + 88);
    v386[352] = *(_BYTE *)(v16 + 456);
    v387 = *(_DWORD *)(v16 + 460);
    v388 = *(_QWORD *)(v16 + 464);
    v375 = &unk_49ECF98;
    if ( v372 != &v374 )
      j_j___libc_free_0(v372, v374 + 1);
    if ( v368 != (unsigned __int64 *)&v370 )
      j_j___libc_free_0(v368, (char *)v370 + 1);
    v17 = (char *)v394;
    v389 = (char *)&unk_49ECF68;
    v18 = (char *)(v394 + 88LL * v395);
    if ( (char *)v394 != v18 )
    {
      do
      {
        v18 -= 88;
        v19 = (char *)*((_QWORD *)v18 + 4);
        if ( v19 != v18 + 48 )
          j_j___libc_free_0(v19, *((_QWORD *)v18 + 6) + 1LL);
        if ( *(char **)v18 != v18 + 16 )
          j_j___libc_free_0(*(_QWORD *)v18, *((_QWORD *)v18 + 2) + 1LL);
      }
      while ( v17 != v18 );
      v18 = (char *)v394;
    }
    if ( v18 != &v396 )
      _libc_free((unsigned __int64)v18);
    sub_143AA50(a6, (__int64)&v375);
    v20 = (_QWORD *)v384;
    v375 = &unk_49ECF68;
    v21 = (_QWORD *)(v384 + 88LL * (unsigned int)v385);
    if ( (_QWORD *)v384 != v21 )
    {
      do
      {
        v21 -= 11;
        v22 = (_QWORD *)v21[4];
        if ( v22 != v21 + 6 )
          j_j___libc_free_0(v22, v21[6] + 1LL);
        if ( (_QWORD *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21, v21[2] + 1LL);
      }
      while ( v20 != v21 );
      v21 = (_QWORD *)v384;
    }
    if ( v21 != (_QWORD *)v386 )
      _libc_free((unsigned __int64)v21);
  }
  v23 = (__int64 *)&v373;
  v368 = 0;
  v369 = (__int64 *)&v373;
  v24 = (__int64 *)&v373;
  v25 = *(__int64 **)(v361 + 8);
  v370 = (__int64 *)&v373;
  v371 = 8;
  LODWORD(v372) = 0;
  v352 = 0;
  if ( v25 )
  {
    while ( 1 )
    {
      v27 = sub_1648700((__int64)v25);
      v28 = (__int64 *)v25[1];
      v29 = (__int64)v27;
      if ( v24 == v23 )
      {
        v26 = &v23[HIDWORD(v371)];
        if ( v26 == v23 )
        {
          v134 = v23;
        }
        else
        {
          do
          {
            if ( v27 == (_QWORD *)*v23 )
              break;
            ++v23;
          }
          while ( v26 != v23 );
          v134 = v26;
        }
LABEL_44:
        while ( v134 != v23 )
        {
          if ( (unsigned __int64)*v23 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_33;
          ++v23;
        }
        if ( v26 != v23 )
          goto LABEL_34;
      }
      else
      {
        v26 = &v24[(unsigned int)v371];
        v23 = sub_16CC9F0((__int64)&v368, (__int64)v27);
        if ( v29 == *v23 )
        {
          if ( v370 == v369 )
            v134 = &v370[HIDWORD(v371)];
          else
            v134 = &v370[(unsigned int)v371];
          goto LABEL_44;
        }
        if ( v370 == v369 )
        {
          v23 = &v370[HIDWORD(v371)];
          v134 = v23;
          goto LABEL_44;
        }
        v23 = &v370[(unsigned int)v371];
LABEL_33:
        if ( v26 != v23 )
          goto LABEL_34;
      }
      if ( sub_1377F70(a4 + 56, *(_QWORD *)(v29 + 40)) )
        goto LABEL_34;
      v30 = *(unsigned int *)(a3 + 48);
      if ( !(_DWORD)v30 )
        goto LABEL_175;
      v31 = *(_QWORD *)(v29 + 40);
      v32 = *(_QWORD *)(a3 + 32);
      v33 = v30 - 1;
      v34 = (v30 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( v31 != *v35 )
      {
        v122 = 1;
        while ( v36 != -8 )
        {
          v292 = v122 + 1;
          v34 = v33 & (v122 + v34);
          v35 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v35;
          if ( v31 == *v35 )
            goto LABEL_49;
          v122 = v292;
        }
LABEL_175:
        v123 = sub_1599EF0(*(__int64 ***)v361);
        if ( *v25 )
        {
          v124 = v25[1];
          v125 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v125 = v124;
          if ( v124 )
            *(_QWORD *)(v124 + 16) = *(_QWORD *)(v124 + 16) & 3LL | v125;
        }
        *v25 = v123;
        v352 = 1;
        if ( v123 )
        {
          v126 = *(_QWORD *)(v123 + 8);
          v25[1] = v126;
          if ( v126 )
            *(_QWORD *)(v126 + 16) = (unsigned __int64)(v25 + 1) | *(_QWORD *)(v126 + 16) & 3LL;
          v25[2] = v25[2] & 3 | (v123 + 8);
          *(_QWORD *)(v123 + 8) = v25;
          v352 = 1;
        }
LABEL_34:
        if ( !v28 )
          goto LABEL_110;
        goto LABEL_35;
      }
LABEL_49:
      v37 = (__int64 *)(v32 + 16 * v30);
      if ( v37 == v35 || !v35[1] )
        goto LABEL_175;
      v38 = *(_BYTE *)(v29 + 23) & 0x40;
      if ( v38 )
        v39 = *(_QWORD *)(v29 - 8);
      else
        v39 = v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF);
      v40 = *(_QWORD *)(v39
                      + 0xFFFFFFFD55555558LL * (unsigned int)(((__int64)v25 - v39) >> 3)
                      + 24LL * *(unsigned int *)(v29 + 56)
                      + 8);
      v41 = v33 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v42 = (__int64 *)(v32 + 16LL * v41);
      v43 = *v42;
      if ( v40 != *v42 )
      {
        v127 = 1;
        while ( v43 != -8 )
        {
          v293 = v127 + 1;
          v41 = v33 & (v41 + v127);
          v42 = (__int64 *)(v32 + 16LL * v41);
          v43 = *v42;
          if ( v40 == *v42 )
            goto LABEL_54;
          v127 = v293;
        }
LABEL_190:
        v128 = sub_1599EF0(*(__int64 ***)v361);
        if ( *v25 )
        {
          v129 = v25[1];
          v130 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v130 = v129;
          if ( v129 )
            *(_QWORD *)(v129 + 16) = v130 | *(_QWORD *)(v129 + 16) & 3LL;
        }
        *v25 = v128;
        v352 = 1;
        if ( v128 )
        {
          v131 = *(_QWORD *)(v128 + 8);
          v25[1] = v131;
          if ( v131 )
            *(_QWORD *)(v131 + 16) = (unsigned __int64)(v25 + 1) | *(_QWORD *)(v131 + 16) & 3LL;
          v25[2] = (v128 + 8) | v25[2] & 3;
          *(_QWORD *)(v128 + 8) = v25;
          v352 = 1;
        }
        goto LABEL_34;
      }
LABEL_54:
      if ( v37 == v42 || !v42[1] )
        goto LABEL_190;
      v44 = v369;
      if ( v370 != v369 )
        break;
      v116 = &v369[HIDWORD(v371)];
      if ( v369 == v116 )
      {
LABEL_274:
        if ( HIDWORD(v371) >= (unsigned int)v371 )
          break;
        ++HIDWORD(v371);
        *v116 = v29;
        v368 = (unsigned __int64 *)((char *)v368 + 1);
        v38 = *(_BYTE *)(v29 + 23) & 0x40;
      }
      else
      {
        v117 = 0;
        while ( v29 != *v44 )
        {
          if ( *v44 == -2 )
            v117 = v44;
          if ( v116 == ++v44 )
          {
            if ( !v117 )
              goto LABEL_274;
            *v117 = v29;
            LODWORD(v372) = (_DWORD)v372 - 1;
            v368 = (unsigned __int64 *)((char *)v368 + 1);
            v38 = *(_BYTE *)(v29 + 23) & 0x40;
            break;
          }
        }
      }
LABEL_58:
      v45 = 3LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF);
      if ( v38 )
      {
        v46 = *(_QWORD **)(v29 - 8);
        v47 = &v46[v45];
      }
      else
      {
        v46 = (_QWORD *)(v29 - v45 * 8);
        v47 = (_QWORD *)v29;
      }
      if ( v46 == v47 )
        goto LABEL_34;
      while ( *v46 == v361 )
      {
        v46 += 3;
        if ( v47 == v46 )
          goto LABEL_34;
      }
      v48 = *(_QWORD *)(v29 + 40);
      if ( !sub_157F5F0(v48) )
        goto LABEL_71;
      if ( *(_DWORD *)(a5 + 24) )
      {
        v49 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v48) + 16) - 34;
        if ( (unsigned int)v49 <= 0x36 )
        {
          v50 = 0x40018000000001LL;
          if ( _bittest64(&v50, v49) )
            goto LABEL_71;
        }
      }
      do
      {
        v48 = *(_QWORD *)(v48 + 8);
        if ( !v48 )
          goto LABEL_77;
        v51 = sub_1648700(v48);
      }
      while ( (unsigned __int8)(*((_BYTE *)v51 + 16) - 25) > 9u );
      if ( *(_BYTE *)(sub_157EBA0(v51[5]) + 16) == 28 )
        goto LABEL_71;
      while ( 1 )
      {
        v48 = *(_QWORD *)(v48 + 8);
        if ( !v48 )
          break;
        v53 = sub_1648700(v48);
        if ( (unsigned __int8)(*((_BYTE *)v53 + 16) - 25) <= 9u && *(_BYTE *)(sub_157EBA0(v53[5]) + 16) == 28 )
          goto LABEL_71;
      }
LABEL_77:
      v54 = *(_QWORD *)(v29 + 40);
      for ( i = *(_QWORD *)(v54 + 8); i; i = *(_QWORD *)(i + 8) )
      {
        if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
          break;
      }
      v56 = (unsigned int **)&v391;
      v389 = 0;
      v390 = 1;
      do
        *v56++ = (unsigned int *)-8LL;
      while ( v56 != &v393 );
      v393 = &v395;
      v394 = 0x800000000LL;
      sub_1962C70((__int64)&v389, i, 0);
      v57 = v393;
      if ( (_DWORD)v394 )
      {
        while ( 1 )
        {
          v58 = *(void **)v57;
          v375 = v58;
          v59 = *(_DWORD *)(v29 + 20) & 0xFFFFFFF;
          if ( !v59 )
            goto LABEL_90;
          v60 = 0;
          v61 = 24LL * *(unsigned int *)(v29 + 56) + 8;
          while ( 1 )
          {
            v62 = v29 - 24LL * v59;
            if ( (*(_BYTE *)(v29 + 23) & 0x40) != 0 )
              v62 = *(_QWORD *)(v29 - 8);
            if ( v58 == *(void **)(v62 + v61) )
              break;
            ++v60;
            v61 += 8;
            if ( v59 == v60 )
              goto LABEL_90;
          }
          if ( v60 < 0 )
            goto LABEL_90;
          v63 = sub_1AAB350(v54, &v375, 1, ".split.loop.exit", a3, a2, 1);
          v64 = *(_DWORD *)(a5 + 24);
          if ( !v64 )
            goto LABEL_90;
          v95 = *(_DWORD *)(a5 + 32);
          v345 = a5 + 8;
          if ( !v95 )
            break;
          v96 = v95 - 1;
          v97 = *(_QWORD *)(a5 + 16);
          v98 = (v95 - 1) & (((unsigned int)v63 >> 4) ^ ((unsigned int)v63 >> 9));
          v99 = (__int64 *)(v97 + 16LL * v98);
          v100 = *v99;
          if ( *v99 == v63 )
          {
LABEL_138:
            v101 = v99 + 1;
            goto LABEL_139;
          }
          v340 = 1;
          v265 = 0;
          srcb = *(_DWORD *)(a5 + 24);
          while ( v100 != -8 )
          {
            if ( !v265 && v100 == -16 )
              v265 = v99;
            v98 = v96 & (v340 + v98);
            v99 = (__int64 *)(v97 + 16LL * v98);
            v100 = *v99;
            if ( v63 == *v99 )
              goto LABEL_138;
            ++v340;
          }
          if ( v265 )
            v99 = v265;
          ++*(_QWORD *)(a5 + 8);
          v266 = v64 + 1;
          if ( 4 * (srcb + 1) >= 3 * v95 )
            goto LABEL_408;
          if ( v95 - *(_DWORD *)(a5 + 28) - v266 <= v95 >> 3 )
          {
            srcc = ((unsigned int)v63 >> 4) ^ ((unsigned int)v63 >> 9);
            v342 = v63;
            sub_14DDDA0(v345, v95);
            v273 = *(_DWORD *)(a5 + 32);
            if ( !v273 )
              goto LABEL_511;
            v274 = v273 - 1;
            v275 = *(_QWORD *)(a5 + 16);
            v276 = 0;
            v277 = 1;
            v266 = *(_DWORD *)(a5 + 24) + 1;
            v63 = v342;
            v278 = v274 & srcc;
            v99 = (__int64 *)(v275 + 16LL * (v274 & srcc));
            v279 = *v99;
            if ( v342 != *v99 )
            {
              while ( v279 != -8 )
              {
                if ( !v276 && v279 == -16 )
                  v276 = v99;
                v278 = v274 & (v277 + v278);
                v99 = (__int64 *)(v275 + 16LL * v278);
                v279 = *v99;
                if ( v342 == *v99 )
                  goto LABEL_387;
                ++v277;
              }
              if ( v276 )
                v99 = v276;
            }
          }
LABEL_387:
          *(_DWORD *)(a5 + 24) = v266;
          if ( *v99 != -8 )
            --*(_DWORD *)(a5 + 28);
          *v99 = v63;
          v101 = v99 + 1;
          v99[1] = 0;
          v95 = *(_DWORD *)(a5 + 32);
          if ( !v95 )
          {
            ++*(_QWORD *)(a5 + 8);
            goto LABEL_391;
          }
          v97 = *(_QWORD *)(a5 + 16);
          v96 = v95 - 1;
LABEL_139:
          v102 = v375;
          v103 = v96 & (((unsigned int)v375 >> 9) ^ ((unsigned int)v375 >> 4));
          v104 = (_QWORD *)(v97 + 16 * v103);
          v105 = *v104;
          if ( v375 != (void *)*v104 )
          {
            v339 = 1;
            v111 = 0;
            while ( v105 != -8 )
            {
              if ( !v111 && v105 == -16 )
                v111 = v104;
              LODWORD(v103) = v96 & (v339 + v103);
              v104 = (_QWORD *)(v97 + 16LL * (unsigned int)v103);
              v105 = *v104;
              if ( v375 == (void *)*v104 )
                goto LABEL_140;
              ++v339;
            }
            v112 = *(_DWORD *)(a5 + 24);
            if ( !v111 )
              v111 = v104;
            ++*(_QWORD *)(a5 + 8);
            v113 = v112 + 1;
            if ( 4 * (v112 + 1) >= 3 * v95 )
            {
LABEL_391:
              v341 = v101;
              sub_14DDDA0(v345, 2 * v95);
              v267 = *(_DWORD *)(a5 + 32);
              if ( !v267 )
                goto LABEL_511;
              v268 = v375;
              v269 = *(_QWORD *)(a5 + 16);
              v347 = v267 - 1;
              v101 = v341;
              v113 = *(_DWORD *)(a5 + 24) + 1;
              v270 = (v267 - 1) & (((unsigned int)v375 >> 9) ^ ((unsigned int)v375 >> 4));
              v111 = (_QWORD *)(v269 + 16LL * v270);
              v102 = (void *)*v111;
              if ( v375 != (void *)*v111 )
              {
                v271 = 1;
                v272 = 0;
                while ( v102 != (void *)-8LL )
                {
                  if ( v102 == (void *)-16LL && !v272 )
                    v272 = v111;
                  v270 = v347 & (v271 + v270);
                  v111 = (_QWORD *)(v269 + 16LL * v270);
                  v102 = (void *)*v111;
                  if ( v375 == (void *)*v111 )
                    goto LABEL_152;
                  ++v271;
                }
LABEL_418:
                v102 = v268;
                if ( v272 )
                  v111 = v272;
              }
            }
            else if ( v95 - (v113 + *(_DWORD *)(a5 + 28)) <= v95 >> 3 )
            {
              v344 = v101;
              sub_14DDDA0(v345, v95);
              v287 = *(_DWORD *)(a5 + 32);
              if ( !v287 )
              {
LABEL_511:
                ++*(_DWORD *)(a5 + 24);
                BUG();
              }
              v268 = v375;
              v288 = *(_QWORD *)(a5 + 16);
              v289 = 1;
              v348 = v287 - 1;
              v101 = v344;
              v113 = *(_DWORD *)(a5 + 24) + 1;
              v272 = 0;
              v290 = (v287 - 1) & (((unsigned int)v375 >> 9) ^ ((unsigned int)v375 >> 4));
              v111 = (_QWORD *)(v288 + 16LL * v290);
              v102 = (void *)*v111;
              if ( v375 != (void *)*v111 )
              {
                while ( v102 != (void *)-8LL )
                {
                  if ( !v272 && v102 == (void *)-16LL )
                    v272 = v111;
                  v290 = v348 & (v289 + v290);
                  v111 = (_QWORD *)(v288 + 16LL * v290);
                  v102 = (void *)*v111;
                  if ( v375 == (void *)*v111 )
                    goto LABEL_152;
                  ++v289;
                }
                goto LABEL_418;
              }
            }
LABEL_152:
            *(_DWORD *)(a5 + 24) = v113;
            if ( *v111 != -8 )
              --*(_DWORD *)(a5 + 28);
            *v111 = v102;
            v114 = v111 + 1;
            *v114 = 0;
            if ( v101 != v114 )
              goto LABEL_155;
            goto LABEL_90;
          }
LABEL_140:
          if ( v101 == v104 + 1 )
            goto LABEL_90;
          v106 = v99[1];
          v107 = v104[1];
          v108 = (v106 >> 2) & 1;
          v109 = v107 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v107 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          {
LABEL_155:
            v115 = v99[1];
            if ( (v115 & 4) != 0 )
            {
              v133 = v115 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v133 )
                *(_DWORD *)(v133 + 8) = 0;
            }
            else
            {
              v99[1] = 0;
            }
            goto LABEL_90;
          }
          if ( (v107 & 4) == 0 )
          {
            if ( !(_BYTE)v108 )
              goto LABEL_302;
            *(_DWORD *)((v106 & 0xFFFFFFFFFFFFFFF8LL) + 8) = 0;
            v137 = v104[1];
            v138 = v99[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v137 & 4) != 0 )
            {
              v291 = (unsigned __int64 **)(v137 & 0xFFFFFFFFFFFFFFF8LL);
            }
            else
            {
              v139 = v137 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v137 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
LABEL_216:
                v140 = *(unsigned int *)(v138 + 8);
                if ( (unsigned int)v140 >= *(_DWORD *)(v138 + 12) )
                {
                  sub_16CD150(v138, (const void *)(v138 + 16), 0, 8, v97, v96);
                  v140 = *(unsigned int *)(v138 + 8);
                }
                *(_QWORD *)(*(_QWORD *)v138 + 8 * v140) = v139;
                ++*(_DWORD *)(v138 + 8);
                goto LABEL_90;
              }
              v291 = 0;
            }
            v139 = **v291;
            goto LABEL_216;
          }
          v110 = *(_DWORD *)(v109 + 8);
          if ( !v110 )
            goto LABEL_155;
          if ( (_BYTE)v108 )
          {
            sub_195E100(v106 & 0xFFFFFFFFFFFFFFF8LL, v109, (__int64)v104, v108, v97, v96);
            goto LABEL_90;
          }
          if ( v110 == 1 )
          {
            v109 = **(_QWORD **)v109;
LABEL_302:
            v99[1] = v109;
            goto LABEL_90;
          }
          v346 = v104[1] & 0xFFFFFFFFFFFFFFF8LL;
          v261 = sub_22077B0(48);
          if ( v261 )
          {
            v264 = v346;
            *(_QWORD *)v261 = v261 + 16;
            *(_QWORD *)(v261 + 8) = 0x400000000LL;
            if ( *(_DWORD *)(v346 + 8) )
            {
              v349 = v261;
              sub_195E100(v261, v264, v261 + 16, 0x400000000LL, v262, v263);
              v261 = v349;
            }
          }
          v99[1] = v261 | 4;
LABEL_90:
          if ( (v390 & 1) != 0 )
          {
            v65 = &v391;
            v66 = 7;
          }
          else
          {
            v65 = v391;
            v66 = v392 - 1;
            if ( !v392 )
              goto LABEL_129;
          }
          v67 = v66 & (((unsigned int)v375 >> 9) ^ ((unsigned int)v375 >> 4));
          v68 = (void **)&v65[v67];
          v69 = *v68;
          if ( v375 == *v68 )
          {
LABEL_93:
            *v68 = (void *)-16LL;
            v57 = v393;
            ++HIDWORD(v390);
            LODWORD(v390) = (2 * ((unsigned int)v390 >> 1) - 2) | v390 & 1;
            v70 = v394;
            v71 = 8LL * (unsigned int)v394;
            v72 = (char *)&v393[(unsigned __int64)v71 / 4];
            v73 = v71 >> 3;
            v74 = v71 >> 5;
            if ( v74 )
            {
              v75 = (void **)v393;
              v76 = (void **)&v393[8 * v74];
              while ( *v75 != v375 )
              {
                if ( v375 == v75[1] )
                {
                  v77 = (char *)(++v75 + 1);
                  goto LABEL_101;
                }
                if ( v375 == v75[2] )
                {
                  v75 += 2;
                  v77 = (char *)(v75 + 1);
                  goto LABEL_101;
                }
                if ( v375 == v75[3] )
                {
                  v75 += 3;
                  break;
                }
                v75 += 4;
                if ( v76 == v75 )
                {
                  v73 = (v72 - (char *)v75) >> 3;
                  if ( v73 == 2 )
                    goto LABEL_343;
                  goto LABEL_208;
                }
              }
LABEL_100:
              v77 = (char *)(v75 + 1);
LABEL_101:
              if ( v77 != v72 )
              {
                memmove(v75, v77, v72 - v77);
                v70 = v394;
                v57 = v393;
              }
              v78 = v70 - 1;
              LODWORD(v394) = v78;
              goto LABEL_104;
            }
            v75 = (void **)v393;
            if ( v73 == 2 )
            {
LABEL_343:
              v240 = v375;
              v241 = v75;
LABEL_344:
              v75 = v241 + 1;
              if ( *v241 == v240 )
              {
                v75 = v241;
                v77 = (char *)(v241 + 1);
                goto LABEL_101;
              }
LABEL_345:
              if ( *v75 == v240 )
                goto LABEL_100;
            }
            else
            {
LABEL_208:
              if ( v73 == 3 )
              {
                v77 = (char *)(v75 + 1);
                v240 = v375;
                v241 = v75 + 1;
                if ( *v75 == v375 )
                  goto LABEL_101;
                goto LABEL_344;
              }
              if ( v73 == 1 )
              {
                v240 = v375;
                goto LABEL_345;
              }
            }
            v75 = (void **)v72;
            v77 = v72 + 8;
            goto LABEL_101;
          }
          v120 = 1;
          while ( v69 != (void *)-8LL )
          {
            v121 = v120 + 1;
            v67 = v66 & (v120 + v67);
            v68 = (void **)&v65[v67];
            v69 = *v68;
            if ( v375 == *v68 )
              goto LABEL_93;
            v120 = v121;
          }
LABEL_129:
          v78 = v394;
          v57 = v393;
LABEL_104:
          if ( !v78 )
            goto LABEL_105;
        }
        ++*(_QWORD *)(a5 + 8);
LABEL_408:
        v343 = v63;
        sub_14DDDA0(v345, 2 * v95);
        v280 = *(_DWORD *)(a5 + 32);
        if ( !v280 )
          goto LABEL_511;
        v63 = v343;
        v281 = v280 - 1;
        v282 = *(_QWORD *)(a5 + 16);
        v266 = *(_DWORD *)(a5 + 24) + 1;
        v283 = v281 & (((unsigned int)v343 >> 9) ^ ((unsigned int)v343 >> 4));
        v99 = (__int64 *)(v282 + 16LL * v283);
        v284 = *v99;
        if ( v343 != *v99 )
        {
          v285 = 1;
          v286 = 0;
          while ( v284 != -8 )
          {
            if ( v284 == -16 && !v286 )
              v286 = v99;
            v283 = v281 & (v285 + v283);
            v99 = (__int64 *)(v282 + 16LL * v283);
            v284 = *v99;
            if ( v343 == *v99 )
              goto LABEL_387;
            ++v285;
          }
          if ( v286 )
            v99 = v286;
        }
        goto LABEL_387;
      }
LABEL_105:
      if ( v57 != &v395 )
        _libc_free((unsigned __int64)v57);
      if ( (v390 & 1) == 0 )
        j___libc_free_0(v391);
      v28 = *(__int64 **)(v361 + 8);
      if ( !v28 )
      {
LABEL_110:
        if ( HIDWORD(v371) == (_DWORD)v372 )
          goto LABEL_71;
        v79 = (char *)&v391;
        v389 = 0;
        v390 = 1;
        do
        {
          *(_QWORD *)v79 = -8;
          v79 += 16;
        }
        while ( v79 != &v397 );
        v375 = 0;
        v376 = 1;
        v80 = *(_QWORD *)(v361 + 8);
        v81 = (__int64 **)&v377;
        do
          *v81++ = (__int64 *)-8LL;
        while ( v81 != &v383 );
        v383 = &v385;
        v384 = 0x800000000LL;
        sub_1963390((__int64)&v375, v80, 0);
        v358 = &v383[(unsigned int)v384];
        if ( v383 == v358 )
        {
LABEL_183:
          if ( v358 != &v385 )
            _libc_free((unsigned __int64)v358);
          if ( (v376 & 1) != 0 )
          {
            if ( (v390 & 1) != 0 )
              goto LABEL_71;
          }
          else
          {
            j___libc_free_0(v377);
            if ( (v390 & 1) != 0 )
            {
LABEL_71:
              if ( v370 != v369 )
                _libc_free((unsigned __int64)v370);
              return v352;
            }
          }
          j___libc_free_0(v391);
          goto LABEL_71;
        }
        v82 = v383;
        v357 = a4 + 56;
        v83 = v352;
        while ( 2 )
        {
          v90 = (_QWORD *)*v82;
          if ( sub_1377F70(v357, *(_QWORD *)(*v82 + 40)) )
            goto LABEL_123;
          v360 = v90[5];
          v93 = v390 & 1;
          if ( (v390 & 1) != 0 )
          {
            v84 = &v391;
            v85 = 31;
            goto LABEL_118;
          }
          v94 = v392;
          v84 = v391;
          if ( v392 )
          {
            v85 = v392 - 1;
LABEL_118:
            v86 = v85 & (((unsigned int)v360 >> 9) ^ ((unsigned int)v360 >> 4));
            v87 = &v84[2 * v86];
            v88 = *v87;
            if ( v360 == *v87 )
            {
LABEL_119:
              if ( !v93 )
              {
                v118 = v392;
                if ( v87 != &v84[2 * v392] )
                {
LABEL_121:
                  v89 = v87[1];
                  goto LABEL_122;
                }
                if ( v392 )
                {
                  v119 = v392 - 1;
                  goto LABEL_220;
                }
                ++v389;
                goto LABEL_277;
              }
              if ( v87 != v84 + 64 )
                goto LABEL_121;
              v119 = 31;
              v118 = 32;
LABEL_220:
              v141 = v119 & (((unsigned int)v360 >> 4) ^ ((unsigned int)v360 >> 9));
              v142 = v84[2 * v141];
              v350 = &v84[2 * v141];
              if ( v360 == v142 )
                goto LABEL_221;
              v294 = &v84[2 * (v119 & (((unsigned int)v360 >> 4) ^ ((unsigned int)v360 >> 9)))];
              v295 = 0;
              v296 = 1;
LABEL_432:
              if ( v142 == -8 )
              {
                if ( !v295 )
                  v295 = v294;
                ++v389;
                v197 = ((unsigned int)v390 >> 1) + 1;
                v350 = v295;
                if ( 4 * v197 < 3 * v118 )
                {
                  if ( v118 - (v197 + HIDWORD(v390)) <= v118 >> 3 )
                  {
                    sub_19624B0((__int64)&v389, v118);
                    if ( (v390 & 1) != 0 )
                    {
                      v297 = &v391;
                      v298 = 31;
                    }
                    else
                    {
                      v297 = v391;
                      if ( !v392 )
                      {
LABEL_512:
                        LODWORD(v390) = (2 * ((unsigned int)v390 >> 1) + 2) | v390 & 1;
                        BUG();
                      }
                      v298 = v392 - 1;
                    }
                    v299 = v298 & (((unsigned int)v360 >> 4) ^ ((unsigned int)v360 >> 9));
                    v300 = 0;
                    v301 = 1;
                    v302 = &v297[2 * v299];
                    v197 = ((unsigned int)v390 >> 1) + 1;
                    v350 = v302;
                    v303 = *v302;
                    if ( *v302 != v360 )
                    {
                      while ( v303 != -8 )
                      {
                        if ( v303 == -16 && !v300 )
                          v300 = v302;
                        v299 = v298 & (v301 + v299);
                        v302 = &v297[2 * v299];
                        v303 = *v302;
                        if ( v360 == *v302 )
                        {
                          v350 = &v297[2 * v299];
                          goto LABEL_281;
                        }
                        ++v301;
                      }
                      if ( v300 )
                        v302 = v300;
                      v350 = v302;
                    }
                  }
LABEL_281:
                  LODWORD(v390) = v390 & 1 | (2 * v197);
                  if ( *v350 != -8 )
                    --HIDWORD(v390);
                  v143 = v361;
                  *v350 = v360;
                  v350[1] = 0;
                  if ( *(_BYTE *)(v143 + 16) == 78 )
                    goto LABEL_222;
LABEL_284:
                  v89 = sub_15F4880(v361);
LABEL_260:
                  v174 = (__int64 *)sub_157EE30(v360);
                  sub_157E9D0(v360 + 40, v89);
                  v175 = *v174;
                  v176 = *(_QWORD *)(v89 + 24);
                  *(_QWORD *)(v89 + 32) = v174;
                  v177 = v361;
                  v175 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v89 + 24) = v175 | v176 & 7;
                  *(_QWORD *)(v175 + 8) = v89 + 24;
                  *v174 = *v174 & 7 | (v89 + 24);
                  sub_1649960(v177);
                  v354 = (__int64 *)v89;
                  if ( v178 )
                  {
                    v362 = (__m128i *)sub_1649960(v361);
                    n = v260;
                    v365 = (__int64 *)&v362;
                    v367[0] = 773;
                    v366 = ".le";
                    sub_164B780(v89, (__int64 *)&v365);
                  }
                  v179 = 3LL * (*(_DWORD *)(v89 + 20) & 0xFFFFFFF);
                  v180 = (__int64 *)(v89 - v179 * 8);
                  if ( (*(_BYTE *)(v89 + 23) & 0x40) != 0 )
                  {
                    v180 = *(__int64 **)(v89 - 8);
                    v354 = &v180[v179];
                  }
                  if ( v354 != v180 )
                  {
                    v321 = v82;
                    v181 = a2;
                    v182 = v90;
                    v319 = v89;
                    v183 = v180;
                    while ( 2 )
                    {
                      v184 = (__int64 *)*v183;
                      if ( *(_BYTE *)(*v183 + 16) <= 0x17u )
                        goto LABEL_271;
                      v185 = *(_DWORD *)(v181 + 24);
                      if ( !v185 )
                        goto LABEL_271;
                      v186 = v184[5];
                      v187 = v185 - 1;
                      v188 = *(_QWORD *)(v181 + 8);
                      v189 = v187 & (((unsigned int)v186 >> 9) ^ ((unsigned int)v186 >> 4));
                      v190 = (__int64 *)(v188 + 16LL * v189);
                      v191 = *v190;
                      if ( *v190 != v186 )
                      {
                        v242 = 1;
                        while ( v191 != -8 )
                        {
                          v243 = v242 + 1;
                          v189 = v187 & (v242 + v189);
                          v190 = (__int64 *)(v188 + 16LL * v189);
                          v191 = *v190;
                          if ( v186 == *v190 )
                            goto LABEL_269;
                          v242 = v243;
                        }
                        goto LABEL_271;
                      }
LABEL_269:
                      v192 = v190[1];
                      if ( !v192 || sub_1377F70(v192 + 56, v182[5]) )
                      {
LABEL_271:
                        v183 += 3;
                        if ( v354 == v183 )
                        {
                          v82 = v321;
                          v89 = v319;
                          v90 = v182;
                          goto LABEL_273;
                        }
                        continue;
                      }
                      break;
                    }
                    v211 = *(_QWORD *)(v360 + 48);
                    if ( v211 )
                      v211 -= 24;
                    v317 = v211;
                    v362 = (__m128i *)sub_1649960((__int64)v184);
                    v367[0] = 773;
                    v365 = (__int64 *)&v362;
                    n = v212;
                    v366 = ".lcssa";
                    v213 = *v184;
                    src = *((_DWORD *)v182 + 5) & 0xFFFFFFF;
                    v326 = *v184;
                    v214 = sub_1648B60(64);
                    v215 = v214;
                    if ( v214 )
                    {
                      sub_15F1EA0(v214, v326, 53, 0, 0, v317);
                      *(_DWORD *)(v215 + 56) = src;
                      sub_164B780(v215, (__int64 *)&v365);
                      v213 = *(unsigned int *)(v215 + 56);
                      sub_1648880(v215, v213, 1);
                    }
                    if ( (*((_DWORD *)v182 + 5) & 0xFFFFFFF) != 0 )
                    {
                      v216 = (__int64)(v184 + 1);
                      v217 = 0;
                      v318 = v183;
                      v315 = v181;
                      v218 = 8LL * (*((_DWORD *)v182 + 5) & 0xFFFFFFF);
                      v219 = v184;
                      v220 = (unsigned __int64)(v184 + 1);
                      v221 = 0;
                      do
                      {
                        if ( (*((_BYTE *)v182 + 23) & 0x40) != 0 )
                          v222 = (_QWORD *)*(v182 - 1);
                        else
                          v222 = &v182[-3 * (*((_DWORD *)v182 + 5) & 0xFFFFFFF)];
                        v223 = v222[3 * *((unsigned int *)v182 + 14) + 1 + v221 / 8];
                        v224 = *(_DWORD *)(v215 + 20) & 0xFFFFFFF;
                        if ( v224 == *(_DWORD *)(v215 + 56) )
                        {
                          v327 = v218;
                          srcf = (void *)v223;
                          sub_15F55D0(v215, v213, v217, v223, v218, v216);
                          v218 = v327;
                          v223 = (__int64)srcf;
                          v224 = *(_DWORD *)(v215 + 20) & 0xFFFFFFF;
                        }
                        v225 = (v224 + 1) & 0xFFFFFFF;
                        v226 = v225 | *(_DWORD *)(v215 + 20) & 0xF0000000;
                        *(_DWORD *)(v215 + 20) = v226;
                        if ( (v226 & 0x40000000) != 0 )
                          v227 = *(_QWORD *)(v215 - 8);
                        else
                          v227 = v215 - 24 * v225;
                        v228 = (__int64 **)(v227 + 24LL * (unsigned int)(v225 - 1));
                        if ( *v228 )
                        {
                          v229 = v228[1];
                          v230 = (unsigned __int64)v228[2] & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v230 = v229;
                          if ( v229 )
                            v229[2] = v229[2] & 3 | v230;
                        }
                        *v228 = v219;
                        v231 = v219[1];
                        v228[1] = (__int64 *)v231;
                        if ( v231 )
                          *(_QWORD *)(v231 + 16) = (unsigned __int64)(v228 + 1) | *(_QWORD *)(v231 + 16) & 3LL;
                        v228[2] = (__int64 *)(v220 | (unsigned __int64)v228[2] & 3);
                        v219[1] = (__int64)v228;
                        v232 = *(_DWORD *)(v215 + 20) & 0xFFFFFFF;
                        v233 = (unsigned int)(v232 - 1);
                        if ( (*(_BYTE *)(v215 + 23) & 0x40) != 0 )
                          v234 = *(_QWORD *)(v215 - 8);
                        else
                          v234 = v215 - 24 * v232;
                        v221 += 8LL;
                        v213 = 3LL * *(unsigned int *)(v215 + 56);
                        *(_QWORD *)(v234 + 8 * v233 + 24LL * *(unsigned int *)(v215 + 56) + 8) = v223;
                      }
                      while ( v218 != v221 );
                      v183 = v318;
                      v181 = v315;
                      if ( *v318 )
                      {
LABEL_327:
                        v235 = v183[1];
                        v236 = v183[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v236 = v235;
                        if ( v235 )
                          *(_QWORD *)(v235 + 16) = v236 | *(_QWORD *)(v235 + 16) & 3LL;
                        goto LABEL_329;
                      }
                      *v318 = v215;
                    }
                    else
                    {
                      if ( *v183 )
                        goto LABEL_327;
LABEL_329:
                      *v183 = v215;
                      if ( !v215 )
                        goto LABEL_271;
                    }
                    v237 = *(_QWORD *)(v215 + 8);
                    v183[1] = v237;
                    if ( v237 )
                      *(_QWORD *)(v237 + 16) = (unsigned __int64)(v183 + 1) | *(_QWORD *)(v237 + 16) & 3LL;
                    v183[2] = (v215 + 8) | v183[2] & 3;
                    *(_QWORD *)(v215 + 8) = v183;
                    goto LABEL_271;
                  }
LABEL_273:
                  v350[1] = v89;
LABEL_122:
                  v83 = 1;
                  sub_164D160(
                    (__int64)v90,
                    v89,
                    a7,
                    *(double *)a8.m128i_i64,
                    *(double *)a9.m128i_i64,
                    a10,
                    v91,
                    v92,
                    a13,
                    a14);
                  sub_15F20C0(v90);
LABEL_123:
                  if ( v358 == ++v82 )
                  {
                    v352 = v83;
                    v358 = v383;
                    goto LABEL_183;
                  }
                  continue;
                }
                v118 *= 2;
LABEL_277:
                sub_19624B0((__int64)&v389, v118);
                if ( (v390 & 1) != 0 )
                {
                  v193 = &v391;
                  v194 = 31;
                }
                else
                {
                  v193 = v391;
                  if ( !v392 )
                    goto LABEL_512;
                  v194 = v392 - 1;
                }
                v195 = v194 & (((unsigned int)v360 >> 9) ^ ((unsigned int)v360 >> 4));
                v350 = &v193[2 * v195];
                v196 = *v350;
                v197 = ((unsigned int)v390 >> 1) + 1;
                if ( *v350 != v360 )
                {
                  v307 = &v193[2 * (v194 & (((unsigned int)v360 >> 9) ^ ((unsigned int)v360 >> 4)))];
                  v308 = 0;
                  v309 = 1;
                  while ( v196 != -8 )
                  {
                    if ( v196 == -16 && !v308 )
                      v308 = v307;
                    v195 = v194 & (v309 + v195);
                    v307 = &v193[2 * v195];
                    v196 = *v307;
                    if ( v360 == *v307 )
                    {
                      v350 = &v193[2 * v195];
                      goto LABEL_281;
                    }
                    ++v309;
                  }
                  if ( !v308 )
                    v308 = v307;
                  v350 = v308;
                }
                goto LABEL_281;
              }
              if ( v142 == -16 && !v295 )
                v295 = v294;
              v305 = v296 + 1;
              v306 = v119 & (v141 + v296);
              v141 = v306;
              v294 = &v84[2 * v306];
              v142 = *v294;
              if ( v360 != *v294 )
              {
                v296 = v305;
                goto LABEL_432;
              }
              v350 = &v84[2 * v306];
LABEL_221:
              v143 = v361;
              if ( *(_BYTE *)(v361 + 16) != 78 )
                goto LABEL_284;
LABEL_222:
              v144 = *(char *)(v143 + 23) < 0;
              v365 = (__int64 *)v367;
              v366 = (const char *)0x100000000LL;
              if ( v144 )
              {
                v145 = sub_1648A40(v143);
                v147 = v145 + v146;
                if ( *(char *)(v143 + 23) >= 0 )
                  v148 = v147 >> 4;
                else
                  LODWORD(v148) = (v147 - sub_1648A40(v361)) >> 4;
                v353 = 16LL * (unsigned int)v148;
                if ( (_DWORD)v148 )
                {
                  v316 = v90;
                  v314 = v82;
                  v149 = 0;
                  v150 = v361;
                  while ( 2 )
                  {
                    v151 = 0;
                    if ( *(char *)(v150 + 23) < 0 )
                      v151 = sub_1648A40(v150);
                    v152 = (unsigned int *)(v149 + v151);
                    v153 = *(_BYTE **)v152;
                    v154 = v152[2];
                    v155 = *(_DWORD *)(v150 + 20) & 0xFFFFFFF;
                    v156 = v152[3];
                    if ( *(_DWORD *)(*(_QWORD *)v152 + 8LL) == 1 )
                      goto LABEL_247;
                    v157 = (unsigned int)v366;
                    if ( (unsigned int)v366 >= HIDWORD(v366) )
                    {
                      srcd = v153;
                      sub_1740340((__int64)&v365, 0);
                      v157 = (unsigned int)v366;
                      v153 = srcd;
                    }
                    v158 = &v365[7 * v157];
                    if ( v158 )
                    {
                      *((_BYTE *)v158 + 16) = 0;
                      *v158 = (__int64)(v158 + 2);
                      v158[1] = 0;
                      v158[4] = 0;
                      v158[5] = 0;
                      v158[6] = 0;
                      v159 = *(_QWORD *)v153;
                      v320 = (__m128i *)(v158 + 2);
                      v325 = v158;
                      v362 = &v364;
                      sub_195E050((__int64 *)&v362, v153 + 16, (__int64)&v153[v159 + 16]);
                      v160 = v325;
                      v161 = (__m128i *)*v325;
                      if ( v362 == &v364 )
                      {
                        v238 = n;
                        if ( n )
                        {
                          if ( n == 1 )
                          {
                            v161->m128i_i8[0] = v364.m128i_i8[0];
                          }
                          else
                          {
                            memcpy(v161, &v364, n);
                            v160 = v325;
                          }
                          v238 = n;
                          v161 = (__m128i *)*v325;
                        }
                        v160[1] = v238;
                        v161->m128i_i8[v238] = 0;
                        v161 = v362;
                      }
                      else
                      {
                        if ( v320 == v161 )
                        {
                          *v325 = v362;
                          v325[1] = n;
                          v325[2] = v364.m128i_i64[0];
                        }
                        else
                        {
                          *v325 = v362;
                          v162 = v325[2];
                          v325[1] = n;
                          v325[2] = v364.m128i_i64[0];
                          if ( v161 )
                          {
                            v362 = v161;
                            v364.m128i_i64[0] = v162;
                            goto LABEL_237;
                          }
                        }
                        v362 = &v364;
                        v161 = &v364;
                      }
LABEL_237:
                      n = 0;
                      v161->m128i_i8[0] = 0;
                      if ( v362 != &v364 )
                      {
                        srce = v160;
                        j_j___libc_free_0(v362, v364.m128i_i64[0] + 1);
                        v160 = srce;
                      }
                      v163 = 3 * v156;
                      v164 = (char *)v160[5];
                      v165 = (char *)(v150 + 24 * v154 - 24 * v155);
                      v166 = 8 * v163 - 24 * v154;
                      v167 = &v165[v166];
                      if ( v165 != &v165[v166] )
                      {
                        v168 = 0xAAAAAAAAAAAAAAABLL * (v166 >> 3);
                        if ( v168 <= (__int64)(v160[6] - (_QWORD)v164) >> 3 )
                        {
                          do
                          {
                            if ( v164 )
                              *(_QWORD *)v164 = *(_QWORD *)v165;
                            v165 += 24;
                            v164 += 8;
                          }
                          while ( v167 != v165 );
                          v160[5] += 8 * v168;
                          goto LABEL_245;
                        }
                        v244 = (char *)v160[4];
                        v245 = v164 - v244;
                        v246 = (v164 - v244) >> 3;
                        if ( v168 > 0xFFFFFFFFFFFFFFFLL - v246 )
                          sub_4262D8((__int64)"vector::_M_range_insert");
                        if ( v168 < v246 )
                          v168 = (v164 - v244) >> 3;
                        v247 = __CFADD__(v246, v168);
                        v248 = v246 + v168;
                        if ( v247 )
                        {
                          v249 = 0x7FFFFFFFFFFFFFF8LL;
                          goto LABEL_359;
                        }
                        if ( v248 )
                        {
                          if ( v248 > 0xFFFFFFFFFFFFFFFLL )
                            v248 = 0xFFFFFFFFFFFFFFFLL;
                          v249 = 8 * v248;
LABEL_359:
                          v328 = v160;
                          v250 = sub_22077B0(v249);
                          v160 = v328;
                          v251 = (char *)v250;
                          v244 = (char *)v328[4];
                          srca = (void *)(v249 + v250);
                          v245 = v164 - v244;
                        }
                        else
                        {
                          srca = 0;
                          v251 = 0;
                        }
                        if ( v244 != v164 )
                        {
                          v312 = v160;
                          v322 = v245;
                          v329 = v244;
                          v252 = (char *)memmove(v251, v244, v245);
                          v160 = v312;
                          v245 = v322;
                          v244 = v329;
                          v251 = v252;
                        }
                        v253 = &v251[v245];
                        v254 = v165;
                        v255 = v253;
                        do
                        {
                          if ( v255 )
                            *(_QWORD *)v255 = *(_QWORD *)v254;
                          v254 += 24;
                          v255 += 8;
                        }
                        while ( v167 != v254 );
                        v256 = &v253[0x5555555555555558LL * ((unsigned __int64)(v167 - v165 - 24) >> 3) + 8];
                        v257 = v160[5] - (_QWORD)v164;
                        if ( v164 != (char *)v160[5] )
                        {
                          v313 = v251;
                          v323 = v160;
                          v330 = v244;
                          v258 = (char *)memcpy(v256, v164, v160[5] - (_QWORD)v164);
                          v251 = v313;
                          v160 = v323;
                          v244 = v330;
                          v256 = v258;
                        }
                        v259 = &v256[v257];
                        if ( v244 )
                        {
                          v324 = v251;
                          v331 = v160;
                          j_j___libc_free_0(v244, v160[6] - (_QWORD)v244);
                          v251 = v324;
                          v160 = v331;
                        }
                        v160[4] = v251;
                        v160[5] = v259;
                        v160[6] = srca;
                      }
LABEL_245:
                      v157 = (unsigned int)v366;
                    }
                    LODWORD(v366) = v157 + 1;
LABEL_247:
                    v149 += 16;
                    if ( v353 == v149 )
                    {
                      v90 = v316;
                      v82 = v314;
                      break;
                    }
                    continue;
                  }
                }
              }
              if ( !*(_DWORD *)(a5 + 24) )
              {
LABEL_250:
                v169 = (unsigned int)v366;
                goto LABEL_251;
              }
              v198 = *(_DWORD *)(a5 + 32);
              v199 = *(_QWORD *)(a5 + 16);
              if ( v198 )
              {
                v200 = (v198 - 1) & (((unsigned int)v360 >> 9) ^ ((unsigned int)v360 >> 4));
                v201 = (__int64 *)(v199 + 16LL * v200);
                v202 = *v201;
                if ( v360 == *v201 )
                  goto LABEL_287;
                v310 = 1;
                while ( v202 != -8 )
                {
                  v311 = v310 + 1;
                  v200 = (v198 - 1) & (v310 + v200);
                  v201 = (__int64 *)(v199 + 16LL * v200);
                  v202 = *v201;
                  if ( v360 == *v201 )
                    goto LABEL_287;
                  v310 = v311;
                }
              }
              v201 = (__int64 *)(v199 + 16LL * v198);
LABEL_287:
              v203 = v201[1];
              v204 = (unsigned __int64 **)(v203 & 0xFFFFFFFFFFFFFFF8LL);
              v205 = v203 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v203 & 4) != 0 || !v204 )
                v205 = **v204;
              v206 = sub_157ED20(v205);
              v207 = (unsigned int)*(unsigned __int8 *)(v206 + 16) - 34;
              if ( (unsigned int)v207 > 0x36 )
                goto LABEL_250;
              v208 = 0x40018000000001LL;
              if ( !_bittest64(&v208, v207) )
                goto LABEL_250;
              if ( (unsigned int)v366 >= HIDWORD(v366) )
                sub_1740340((__int64)&v365, 0);
              v362 = &v364;
              sub_195E050((__int64 *)&v362, "funclet", (__int64)"");
              v209 = (__m128 *)&v365[7 * (unsigned int)v366];
              if ( v209 )
              {
                v209->m128_u64[0] = (unsigned __int64)&v209[1];
                if ( v362 == &v364 )
                {
                  a9 = _mm_load_si128(&v364);
                  v209[1] = (__m128)a9;
                }
                else
                {
                  v209->m128_u64[0] = (unsigned __int64)v362;
                  v209[1].m128_u64[0] = v364.m128i_i64[0];
                }
                v209->m128_u64[1] = n;
                v362 = &v364;
                n = 0;
                v364.m128i_i8[0] = 0;
                v209[2].m128_u64[0] = 0;
                v209[2].m128_u64[1] = 0;
                v209[3].m128_u64[0] = 0;
                v210 = (__int64 *)sub_22077B0(8);
                v209[2].m128_u64[0] = (unsigned __int64)v210;
                v209[3].m128_u64[0] = (unsigned __int64)(v210 + 1);
                *v210 = v206;
                v209[2].m128_u64[1] = (unsigned __int64)(v210 + 1);
              }
              if ( v362 != &v364 )
                j_j___libc_free_0(v362, v364.m128i_i64[0] + 1);
              v169 = (_DWORD)v366 + 1;
              LODWORD(v366) = (_DWORD)v366 + 1;
LABEL_251:
              v170 = sub_15F60C0(v361, v365, v169, 0);
              v171 = v365;
              v89 = v170;
              v172 = &v365[7 * (unsigned int)v366];
              if ( v365 != v172 )
              {
                do
                {
                  v173 = *(v172 - 3);
                  v172 -= 7;
                  if ( v173 )
                    j_j___libc_free_0(v173, v172[6] - v173);
                  if ( (__int64 *)*v172 != v172 + 2 )
                    j_j___libc_free_0(*v172, v172[2] + 1);
                }
                while ( v171 != v172 );
                v172 = v365;
              }
              if ( v172 != (__int64 *)v367 )
                _libc_free((unsigned __int64)v172);
              goto LABEL_260;
            }
            v239 = 1;
            while ( v88 != -8 )
            {
              v304 = v239 + 1;
              v86 = v85 & (v239 + v86);
              v87 = &v84[2 * v86];
              v88 = *v87;
              if ( v360 == *v87 )
                goto LABEL_119;
              v239 = v304;
            }
            if ( v93 )
            {
              v132 = 64;
              goto LABEL_199;
            }
            v94 = v392;
          }
          break;
        }
        v132 = 2LL * v94;
LABEL_199:
        v87 = &v84[v132];
        goto LABEL_119;
      }
LABEL_35:
      v24 = v370;
      v23 = v369;
      v25 = v28;
    }
    sub_16CCBA0((__int64)&v368, v29);
    v38 = *(_BYTE *)(v29 + 23) & 0x40;
    goto LABEL_58;
  }
  return v352;
}
