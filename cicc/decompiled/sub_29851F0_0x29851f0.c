// Function: sub_29851F0
// Address: 0x29851f0
//
__int64 __fastcall sub_29851F0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // rsi
  const __m128i *v21; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rcx
  __m128i *v25; // rdx
  const __m128i *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  const __m128i *v29; // rcx
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  __m128i *v33; // rdx
  const __m128i *v34; // rax
  __int64 v35; // rcx
  unsigned __int64 v36; // rax
  __int64 *v37; // rbx
  _BYTE *v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // r15
  int v42; // edx
  __int64 *i; // rbx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 *v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned __int64 v51; // rcx
  __int64 v52; // r14
  __int64 v53; // r15
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r15
  __int64 v58; // r13
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rdx
  unsigned __int64 v62; // rax
  unsigned int v63; // r13d
  __int64 v64; // rax
  unsigned __int64 v65; // r12
  __int64 v66; // r13
  unsigned __int64 v67; // rbx
  unsigned __int64 v68; // rbx
  __int64 v69; // rax
  _BYTE *v70; // r13
  _BYTE *v71; // r12
  __int64 v72; // r12
  __int64 v73; // r14
  __int64 *v74; // rax
  __int64 v75; // rcx
  __int64 *v76; // rdx
  __int64 v77; // rbx
  __int64 *v78; // rax
  __int64 *v79; // rax
  _BYTE *v80; // r13
  _BYTE *v81; // r12
  bool v82; // zf
  __int64 v83; // rax
  __int64 v84; // r15
  unsigned int v85; // r13d
  __int64 v86; // rax
  char v87; // dl
  unsigned __int64 v88; // rdx
  char v89; // si
  _QWORD *v90; // rbx
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rax
  __int64 v94; // rax
  _BYTE *v95; // rsi
  _BYTE *v96; // rsi
  _BYTE *v97; // rsi
  __int64 v98; // rdi
  __int64 v99; // r8
  __int64 v100; // rsi
  __int64 *v101; // rax
  __int64 v102; // rdi
  __int64 v103; // rdx
  __int64 v104; // rsi
  unsigned int v105; // ecx
  __int64 *v106; // rax
  __int64 v107; // r9
  __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rbx
  __int64 v111; // r12
  __int64 v112; // r14
  _BYTE *v113; // rsi
  __int64 v114; // r9
  _BYTE *v115; // r14
  __int64 v116; // rax
  __int32 v117; // esi
  unsigned __int64 *v118; // rdx
  __int64 v119; // rax
  __int64 v120; // rax
  unsigned int v121; // r14d
  _BYTE *v122; // rdx
  unsigned __int64 v123; // r15
  int v124; // eax
  bool v125; // al
  __int64 v126; // r14
  _BYTE *v127; // rax
  __int64 v128; // rax
  unsigned int v129; // r14d
  unsigned __int8 *v130; // r12
  __int64 (__fastcall *v131)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  unsigned __int8 *v132; // r14
  _BYTE *v133; // rsi
  char **v134; // r12
  char **v135; // rbx
  char *v136; // rdi
  __int64 v137; // rax
  int v138; // edx
  __int64 v139; // rax
  _QWORD *v140; // rdx
  _QWORD *k; // rax
  __int64 v142; // r12
  __int64 v143; // rbx
  unsigned __int64 v144; // rdi
  int v145; // r12d
  _QWORD *v146; // rbx
  unsigned int v147; // eax
  __int64 v148; // r14
  _QWORD *v149; // r13
  unsigned __int64 v150; // rdi
  __int64 v151; // rax
  int v152; // ecx
  __int64 v153; // r14
  unsigned int v154; // eax
  __int64 v155; // r12
  __int64 v156; // rax
  __int64 v157; // rax
  _QWORD *v158; // rbx
  _QWORD *v159; // r13
  unsigned __int64 v160; // rdi
  _QWORD *v161; // rbx
  unsigned __int64 v162; // rdi
  unsigned __int64 v164; // rdi
  _QWORD *v165; // rdi
  __int64 v166; // rdx
  int v167; // ebx
  unsigned int v168; // eax
  unsigned __int64 v169; // rdx
  unsigned __int64 v170; // rax
  _QWORD *v171; // rax
  __int64 v172; // rdx
  _QWORD *m; // rdx
  bool v174; // al
  __int64 v175; // r12
  __int64 v176; // rax
  int v177; // eax
  bool v178; // al
  _QWORD *v179; // rax
  __int64 **v180; // rax
  __int64 v181; // r15
  unsigned __int64 v182; // rax
  __int32 v183; // r14d
  unsigned __int64 v185; // rcx
  unsigned __int64 v186; // rax
  int v187; // eax
  _BYTE *v188; // rax
  __int64 v189; // rax
  unsigned __int64 v190; // rsi
  unsigned int v191; // r14d
  int v192; // eax
  bool v193; // al
  _BYTE *v194; // rax
  __int64 v195; // rax
  int v196; // eax
  int v197; // r10d
  int v198; // eax
  int v199; // r11d
  unsigned __int8 *v200; // rcx
  __int64 v201; // rdx
  unsigned int v202; // r14d
  bool v203; // al
  _BYTE *v204; // rdx
  _BYTE *v205; // rax
  __int64 v206; // rax
  unsigned int v207; // r14d
  __int64 v208; // rsi
  unsigned __int64 v209; // rax
  _BYTE *v210; // r14
  __int64 v211; // r14
  _BYTE *v212; // rax
  __int64 v213; // rax
  unsigned int v214; // ecx
  unsigned int v215; // edx
  _QWORD *v216; // rdi
  int v217; // ebx
  unsigned __int64 v218; // rdx
  unsigned __int64 v219; // rax
  _QWORD *v220; // rax
  __int64 v221; // rdx
  _QWORD *j; // rdx
  int v223; // esi
  unsigned __int64 v224; // rcx
  _BYTE *v226; // rax
  __int64 v227; // r12
  __int64 v228; // rax
  __int64 v229; // r15
  __int64 v230; // rdx
  unsigned int v231; // esi
  unsigned __int64 v232; // r15
  int v233; // ebx
  __int64 v234; // r13
  __int64 v235; // rax
  __int64 v236; // rax
  _QWORD *v237; // rdx
  unsigned __int64 v238; // rdi
  _QWORD *v239; // rdi
  __int64 v240; // rdx
  int v241; // ecx
  int v242; // ebx
  unsigned int v243; // eax
  unsigned __int64 v244; // rdx
  unsigned __int64 v245; // rax
  _QWORD *v246; // rax
  __int64 v247; // rcx
  _QWORD *n; // rdx
  __int64 v249; // r14
  _BYTE *v250; // rax
  unsigned __int8 *v251; // rcx
  unsigned int v252; // r14d
  __int64 v253; // rsi
  __int64 **v254; // rax
  unsigned int v255; // ecx
  bool v256; // r14
  __int64 v257; // rax
  unsigned int v258; // ecx
  unsigned int v259; // r14d
  int v260; // eax
  _QWORD *v261; // rax
  _QWORD *v262; // rax
  _QWORD *v263; // rax
  __int64 v264; // [rsp+0h] [rbp-3D0h]
  char v265; // [rsp+38h] [rbp-398h]
  __int64 v266; // [rsp+38h] [rbp-398h]
  __int64 *v267; // [rsp+40h] [rbp-390h]
  __int64 v268; // [rsp+48h] [rbp-388h]
  int v269; // [rsp+48h] [rbp-388h]
  unsigned int v270; // [rsp+48h] [rbp-388h]
  int v271; // [rsp+58h] [rbp-378h]
  _BYTE *v272; // [rsp+58h] [rbp-378h]
  _QWORD *v273; // [rsp+58h] [rbp-378h]
  _QWORD *v274; // [rsp+58h] [rbp-378h]
  int v275; // [rsp+58h] [rbp-378h]
  unsigned __int8 *v276; // [rsp+58h] [rbp-378h]
  __int64 v277; // [rsp+60h] [rbp-370h]
  __int64 v278; // [rsp+68h] [rbp-368h]
  __int64 v279; // [rsp+70h] [rbp-360h]
  __int64 v280; // [rsp+70h] [rbp-360h]
  _BYTE *v281; // [rsp+70h] [rbp-360h]
  unsigned __int64 v282; // [rsp+70h] [rbp-360h]
  __int64 v283; // [rsp+70h] [rbp-360h]
  int v284; // [rsp+70h] [rbp-360h]
  unsigned __int8 **v285; // [rsp+78h] [rbp-358h]
  __int64 v286; // [rsp+80h] [rbp-350h]
  __int64 v287; // [rsp+80h] [rbp-350h]
  _QWORD *v288; // [rsp+80h] [rbp-350h]
  __int64 *v289; // [rsp+88h] [rbp-348h]
  _QWORD *v290; // [rsp+88h] [rbp-348h]
  __int64 v291; // [rsp+90h] [rbp-340h]
  __int64 v292; // [rsp+90h] [rbp-340h]
  _QWORD *v293; // [rsp+90h] [rbp-340h]
  unsigned int v294; // [rsp+98h] [rbp-338h]
  _QWORD *v295; // [rsp+98h] [rbp-338h]
  __int64 v296; // [rsp+98h] [rbp-338h]
  unsigned __int8 v297; // [rsp+98h] [rbp-338h]
  _BYTE *v298; // [rsp+A0h] [rbp-330h] BYREF
  unsigned int v299; // [rsp+A8h] [rbp-328h]
  __int64 v300[16]; // [rsp+B0h] [rbp-320h] BYREF
  __m128i v301; // [rsp+130h] [rbp-2A0h] BYREF
  __int64 v302; // [rsp+140h] [rbp-290h] BYREF
  int v303; // [rsp+148h] [rbp-288h]
  char v304; // [rsp+14Ch] [rbp-284h]
  _QWORD v305[8]; // [rsp+150h] [rbp-280h] BYREF
  unsigned __int64 v306; // [rsp+190h] [rbp-240h] BYREF
  __int64 v307; // [rsp+198h] [rbp-238h]
  unsigned __int64 v308; // [rsp+1A0h] [rbp-230h]
  _BYTE *v309; // [rsp+1B0h] [rbp-220h] BYREF
  __int64 *v310; // [rsp+1B8h] [rbp-218h]
  unsigned int v311; // [rsp+1C0h] [rbp-210h]
  unsigned int v312; // [rsp+1C4h] [rbp-20Ch]
  char v313; // [rsp+1CCh] [rbp-204h]
  _WORD v314[32]; // [rsp+1D0h] [rbp-200h] BYREF
  unsigned __int64 v315; // [rsp+210h] [rbp-1C0h] BYREF
  __int64 v316; // [rsp+218h] [rbp-1B8h]
  unsigned __int64 v317; // [rsp+220h] [rbp-1B0h]
  _BYTE *v318; // [rsp+230h] [rbp-1A0h] BYREF
  unsigned __int64 v319; // [rsp+238h] [rbp-198h]
  void (__fastcall *v320)(_QWORD, _QWORD, _QWORD); // [rsp+240h] [rbp-190h]
  char v321; // [rsp+24Ch] [rbp-184h]
  _WORD v322[32]; // [rsp+250h] [rbp-180h] BYREF
  unsigned __int64 v323; // [rsp+290h] [rbp-140h]
  __int64 v324; // [rsp+298h] [rbp-138h]
  __int64 v325; // [rsp+2A0h] [rbp-130h]
  __m128i v326; // [rsp+2B0h] [rbp-120h] BYREF
  _QWORD v327; // [rsp+2C0h] [rbp-110h] BYREF
  char v328; // [rsp+2CCh] [rbp-104h]
  char v329[16]; // [rsp+2D0h] [rbp-100h] BYREF
  __int64 v330; // [rsp+2E0h] [rbp-F0h]
  __int64 v331; // [rsp+2E8h] [rbp-E8h]
  __int64 v332; // [rsp+2F0h] [rbp-E0h]
  _QWORD *v333; // [rsp+2F8h] [rbp-D8h]
  void **v334; // [rsp+300h] [rbp-D0h]
  void **v335; // [rsp+308h] [rbp-C8h]
  const __m128i *v336; // [rsp+310h] [rbp-C0h]
  unsigned __int64 v337; // [rsp+318h] [rbp-B8h]
  unsigned __int64 v338; // [rsp+320h] [rbp-B0h]
  __int64 v339; // [rsp+328h] [rbp-A8h] BYREF
  void *v340; // [rsp+330h] [rbp-A0h] BYREF
  void *v341; // [rsp+338h] [rbp-98h] BYREF
  char v342; // [rsp+344h] [rbp-8Ch]
  char v343[64]; // [rsp+348h] [rbp-88h] BYREF
  const __m128i *v344; // [rsp+388h] [rbp-48h]
  const __m128i *v345; // [rsp+390h] [rbp-40h]
  __int64 v346; // [rsp+398h] [rbp-38h]

  v1 = a1;
  v2 = *(_QWORD *)(a1 + 8);
  memset(v300, 0, 0x78u);
  v300[1] = (__int64)&v300[4];
  LODWORD(v300[2]) = 8;
  BYTE4(v300[3]) = 1;
  v3 = *(_QWORD *)(v2 + 96);
  v302 = 0x100000008LL;
  v301.m128i_i64[1] = (__int64)v305;
  v305[0] = v3;
  v326.m128i_i64[0] = v3;
  v306 = 0;
  v307 = 0;
  v308 = 0;
  v303 = 0;
  v304 = 1;
  v301.m128i_i64[0] = 1;
  LOBYTE(v327) = 0;
  sub_297F010((__int64)&v306, &v326);
  sub_C8CF70((__int64)&v318, v322, 8, (__int64)&v300[4], (__int64)v300);
  v4 = v300[12];
  memset(&v300[12], 0, 24);
  v323 = v4;
  v324 = v300[13];
  v325 = v300[14];
  sub_C8CF70((__int64)&v309, v314, 8, (__int64)v305, (__int64)&v301);
  v5 = v306;
  v306 = 0;
  v315 = v5;
  v6 = v307;
  v307 = 0;
  v316 = v6;
  v7 = v308;
  v308 = 0;
  v317 = v7;
  sub_C8CF70((__int64)&v326, v329, 8, (__int64)v314, (__int64)&v309);
  v8 = v315;
  v315 = 0;
  v336 = (const __m128i *)v8;
  v9 = v316;
  v316 = 0;
  v337 = v9;
  v10 = v317;
  v317 = 0;
  v338 = v10;
  sub_C8CF70((__int64)&v339, v343, 8, (__int64)v322, (__int64)&v318);
  v14 = v323;
  v323 = 0;
  v344 = (const __m128i *)v14;
  v15 = v324;
  v324 = 0;
  v345 = (const __m128i *)v15;
  v16 = v325;
  v325 = 0;
  v346 = v16;
  if ( v315 )
    j_j___libc_free_0(v315);
  if ( !v313 )
    _libc_free((unsigned __int64)v310);
  if ( v323 )
    j_j___libc_free_0(v323);
  if ( !v321 )
    _libc_free(v319);
  if ( v306 )
    j_j___libc_free_0(v306);
  if ( !v304 )
    _libc_free(v301.m128i_u64[1]);
  if ( v300[12] )
    j_j___libc_free_0(v300[12]);
  if ( !BYTE4(v300[3]) )
    _libc_free(v300[1]);
  sub_C8CD80((__int64)&v309, (__int64)v314, (__int64)&v326, v11, v12, v13);
  v20 = v337;
  v21 = v336;
  v315 = 0;
  v316 = 0;
  v317 = 0;
  v22 = v337 - (_QWORD)v336;
  if ( (const __m128i *)v337 == v336 )
  {
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_458;
    v23 = sub_22077B0(v337 - (_QWORD)v336);
    v20 = v337;
    v21 = v336;
    v24 = v23;
  }
  v315 = v24;
  v316 = v24;
  v317 = v24 + v22;
  if ( v21 != (const __m128i *)v20 )
  {
    v25 = (__m128i *)v24;
    v26 = v21;
    do
    {
      if ( v25 )
      {
        *v25 = _mm_loadu_si128(v26);
        v18 = v26[1].m128i_i64[0];
        v25[1].m128i_i64[0] = v18;
      }
      v26 = (const __m128i *)((char *)v26 + 24);
      v25 = (__m128i *)((char *)v25 + 24);
    }
    while ( v26 != (const __m128i *)v20 );
    v24 += 8 * ((unsigned __int64)((char *)&v26[-2].m128i_u64[1] - (char *)v21) >> 3) + 24;
  }
  v21 = (const __m128i *)&v318;
  v316 = v24;
  sub_C8CD80((__int64)&v318, (__int64)v322, (__int64)&v339, v24, v18, v19);
  v29 = v345;
  v20 = (unsigned __int64)v344;
  v323 = 0;
  v324 = 0;
  v325 = 0;
  v30 = (char *)v345 - (char *)v344;
  if ( v345 != v344 )
  {
    if ( v30 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v31 = sub_22077B0((char *)v345 - (char *)v344);
      v29 = v345;
      v20 = (unsigned __int64)v344;
      v32 = v31;
      goto LABEL_29;
    }
LABEL_458:
    sub_4261EA(v21, v20, v17);
  }
  v30 = 0;
  v32 = 0;
LABEL_29:
  v323 = v32;
  v33 = (__m128i *)v32;
  v324 = v32;
  v325 = v32 + v30;
  if ( (const __m128i *)v20 != v29 )
  {
    v34 = (const __m128i *)v20;
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v34);
        v27 = v34[1].m128i_i64[0];
        v33[1].m128i_i64[0] = v27;
      }
      v34 = (const __m128i *)((char *)v34 + 24);
      v33 = (__m128i *)((char *)v33 + 24);
    }
    while ( v34 != v29 );
    v33 = (__m128i *)(v32 + 8 * (((unsigned __int64)&v34[-2].m128i_u64[1] - v20) >> 3) + 24);
  }
  v35 = v316;
  v36 = v315;
  v324 = (__int64)v33;
  if ( (__m128i *)(v316 - v315) != (__m128i *)((char *)v33 - v32) )
    goto LABEL_36;
LABEL_114:
  if ( v35 != v36 )
  {
    v88 = v32;
    while ( *(_QWORD *)v36 == *(_QWORD *)v88 )
    {
      v89 = *(_BYTE *)(v36 + 16);
      if ( v89 != *(_BYTE *)(v88 + 16) || v89 && *(_QWORD *)(v36 + 8) != *(_QWORD *)(v88 + 8) )
        break;
      v36 += 24LL;
      v88 += 24LL;
      if ( v35 == v36 )
        goto LABEL_121;
    }
    while ( 1 )
    {
LABEL_36:
      v37 = *(__int64 **)(v35 - 24);
      v38 = *(_BYTE **)(v1 + 160);
      v39 = *v37;
      v301.m128i_i64[0] = *v37;
      if ( v38 == *(_BYTE **)(v1 + 168) )
      {
        sub_A413F0(v1 + 152, v38, &v301);
      }
      else
      {
        if ( v38 )
        {
          *(_QWORD *)v38 = v39;
          v38 = *(_BYTE **)(v1 + 160);
        }
        *(_QWORD *)(v1 + 160) = v38 + 8;
      }
      v277 = *v37 + 48;
      if ( *(_QWORD *)(*v37 + 56) != v277 )
      {
        v40 = v1;
        v41 = *(_QWORD *)(*v37 + 56);
        while ( 1 )
        {
          if ( !v41 )
LABEL_494:
            BUG();
          v42 = *(unsigned __int8 *)(v41 - 24);
          v291 = v41 - 24;
          if ( v42 == 46 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v41 - 16) + 8LL) != 12 )
              goto LABEL_43;
            v69 = (*(_BYTE *)(v41 - 17) & 0x40) != 0
                ? *(_QWORD *)(v41 - 32)
                : v291 - 32LL * (*(_DWORD *)(v41 - 20) & 0x7FFFFFF);
            v70 = *(_BYTE **)v69;
            v71 = *(_BYTE **)(v69 + 32);
            sub_2980CF0((unsigned __int64 *)v40, *(_BYTE **)v69, v71, v291);
            if ( v70 == v71 )
              goto LABEL_43;
            sub_2980CF0((unsigned __int64 *)v40, v71, v70, v291);
            v41 = *(_QWORD *)(v41 + 8);
            if ( v277 == v41 )
            {
LABEL_82:
              v1 = v40;
              break;
            }
          }
          else
          {
            if ( v42 == 63 )
            {
              if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v41 - 16) + 8LL) - 17 > 1 )
              {
                v301.m128i_i64[0] = (__int64)&v302;
                v301.m128i_i64[1] = 0x400000000LL;
                for ( i = (__int64 *)(v291 + 32 * (1LL - (*(_DWORD *)(v41 - 20) & 0x7FFFFFF)));
                      (__int64 *)v291 != i;
                      ++v301.m128i_i32[2] )
                {
                  v46 = sub_DD8400(*(_QWORD *)(v40 + 16), *i);
                  v47 = v301.m128i_u32[2];
                  v48 = v301.m128i_u32[2] + 1LL;
                  if ( v48 > v301.m128i_u32[3] )
                  {
                    sub_C8D5F0((__int64)&v301, &v302, v48, 8u, v44, v45);
                    v47 = v301.m128i_u32[2];
                  }
                  i += 4;
                  *(_QWORD *)(v301.m128i_i64[0] + 8 * v47) = v46;
                }
                if ( (*(_BYTE *)(v41 - 17) & 0x40) != 0 )
                  v49 = *(_QWORD *)(v41 - 32);
                else
                  v49 = v291 - 32LL * (*(_DWORD *)(v41 - 20) & 0x7FFFFFF);
                v285 = (unsigned __int8 **)(v49 + 32);
                v294 = 1;
                v51 = sub_BB5290(v291) & 0xFFFFFFFFFFFFFFF9LL | 4;
                v271 = *(_DWORD *)(v41 - 20) & 0x7FFFFFF;
                if ( v271 != 1 )
                {
                  v289 = (__int64 *)v40;
                  v52 = v51;
                  v278 = v41;
                  while ( 2 )
                  {
                    v65 = v52 & 0xFFFFFFFFFFFFFFF8LL;
                    v66 = v294;
                    v67 = v52 & 0xFFFFFFFFFFFFFFF8LL;
                    ++v294;
                    v286 = (v52 >> 1) & 3;
                    if ( ((v52 >> 1) & 3) == 0 )
                      goto LABEL_71;
                    v279 = (unsigned int)(v66 - 1);
                    v53 = v289[2];
                    v268 = *(_QWORD *)(v301.m128i_i64[0] + 8 * v279);
                    v54 = sub_D95540(v268);
                    v55 = sub_DA2C50(v53, v54, 0, 0);
                    *(_QWORD *)(v301.m128i_i64[0] + 8 * v279) = v55;
                    v267 = sub_DD8EB0((__int64 *)v289[2], v291, (__int64)&v301);
                    v57 = *(_QWORD *)(v291 + 32 * (v66 - (*(_DWORD *)(v278 - 20) & 0x7FFFFFF)));
                    v58 = *v289;
                    if ( v52 )
                    {
                      if ( v286 == 2 )
                      {
                        v59 = v52 & 0xFFFFFFFFFFFFFFF8LL;
                        if ( v65 )
                          goto LABEL_59;
LABEL_163:
                        v59 = sub_BCBAE0(v52 & 0xFFFFFFFFFFFFFFF8LL, *v285, v56);
LABEL_59:
                        v265 = sub_AE5020(v58, v59);
                        v60 = sub_9208B0(v58, v59);
                        v300[1] = v61;
                        v62 = ((1LL << v265) + ((unsigned __int64)(v60 + 7) >> 3) - 1) >> v265 << v265;
LABEL_60:
                        LOBYTE(v300[1]) = v61;
                        v300[0] = v62;
                        v266 = sub_CA1930(v300);
                        v64 = *(_QWORD *)(*(_QWORD *)(v291 - 32LL * (*(_DWORD *)(v278 - 20) & 0x7FFFFFF)) + 8LL);
                        if ( (unsigned int)*(unsigned __int8 *)(v64 + 8) - 17 <= 1 )
                          v64 = **(_QWORD **)(v64 + 16);
                        v63 = *(_DWORD *)(*(_QWORD *)(v57 + 8) + 8LL) >> 8;
                        if ( v63 <= sub_AE2980(*v289, *(_DWORD *)(v64 + 8) >> 8)[3] )
                        {
                          sub_2981030(v289, (unsigned __int8 *)v57, (__int64)v267, v266, v291);
                          if ( *(_BYTE *)v57 != 69 )
                            goto LABEL_64;
                        }
                        else if ( *(_BYTE *)v57 != 69 )
                        {
LABEL_64:
                          *(_QWORD *)(v301.m128i_i64[0] + 8 * v279) = v268;
                          if ( v52 )
                          {
                            if ( v286 == 2 )
                            {
                              if ( v65 )
                                goto LABEL_67;
                            }
                            else if ( v286 == 1 && v65 )
                            {
                              v67 = *(_QWORD *)(v65 + 24);
LABEL_67:
                              v50 = *(unsigned __int8 *)(v67 + 8);
                              if ( (_BYTE)v50 != 16 )
                                goto LABEL_72;
                              goto LABEL_68;
                            }
                          }
LABEL_71:
                          v67 = sub_BCBAE0(v52 & 0xFFFFFFFFFFFFFFF8LL, *v285, v50);
                          v50 = *(unsigned __int8 *)(v67 + 8);
                          if ( (_BYTE)v50 != 16 )
                          {
LABEL_72:
                            v68 = v67 & 0xFFFFFFFFFFFFFFF9LL;
                            if ( (unsigned int)(unsigned __int8)v50 - 17 > 1 )
                            {
                              v82 = (_BYTE)v50 == 15;
                              v50 = 0;
                              if ( v82 )
                                v50 = v68;
                              v52 = v50;
                            }
                            else
                            {
                              v52 = v68 | 2;
                            }
                            goto LABEL_69;
                          }
LABEL_68:
                          v52 = *(_QWORD *)(v67 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_69:
                          v285 += 4;
                          if ( v271 == v294 )
                          {
                            v41 = v278;
                            v40 = (__int64)v289;
                            goto LABEL_161;
                          }
                          continue;
                        }
                        v84 = *(_QWORD *)(v57 - 32);
                        if ( v84 )
                        {
                          v86 = *(_QWORD *)(*(_QWORD *)(v291 - 32LL * (*(_DWORD *)(v278 - 20) & 0x7FFFFFF)) + 8LL);
                          if ( (unsigned int)*(unsigned __int8 *)(v86 + 8) - 17 <= 1 )
                            v86 = **(_QWORD **)(v86 + 16);
                          v85 = *(_DWORD *)(*(_QWORD *)(v84 + 8) + 8LL) >> 8;
                          if ( v85 <= sub_AE2980(*v289, *(_DWORD *)(v86 + 8) >> 8)[3] )
                            sub_2981030(v289, (unsigned __int8 *)v84, (__int64)v267, v266, v291);
                        }
                        goto LABEL_64;
                      }
                      if ( (_DWORD)v286 != 1 )
                        goto LABEL_163;
                      if ( v65 )
                        v59 = *(_QWORD *)(v65 + 24);
                      else
                        v59 = sub_BCBAE0(0, *v285, v56);
                    }
                    else
                    {
                      v59 = sub_BCBAE0(v65, *v285, v56);
                      if ( v286 != 1 )
                        goto LABEL_59;
                    }
                    break;
                  }
                  v83 = sub_9208B0(v58, v59);
                  v300[1] = v61;
                  v62 = (unsigned __int64)(v83 + 7) >> 3;
                  goto LABEL_60;
                }
LABEL_161:
                if ( (__int64 *)v301.m128i_i64[0] != &v302 )
                  _libc_free(v301.m128i_u64[0]);
              }
            }
            else if ( v42 == 42 && *(_BYTE *)(*(_QWORD *)(v41 - 16) + 8LL) == 12 )
            {
              v79 = (*(_BYTE *)(v41 - 17) & 0x40) != 0
                  ? *(__int64 **)(v41 - 32)
                  : (__int64 *)(v291 - 32LL * (*(_DWORD *)(v41 - 20) & 0x7FFFFFF));
              v80 = (_BYTE *)*v79;
              v81 = (_BYTE *)v79[4];
              sub_2980B00(v40, *v79, v81, v291);
              if ( v80 != v81 )
                sub_2980B00(v40, (__int64)v81, v80, v291);
            }
LABEL_43:
            v41 = *(_QWORD *)(v41 + 8);
            if ( v277 == v41 )
              goto LABEL_82;
          }
        }
      }
      v72 = v316;
      do
      {
        v73 = *(_QWORD *)(v72 - 24);
        if ( !*(_BYTE *)(v72 - 8) )
        {
          v74 = *(__int64 **)(v73 + 24);
          *(_BYTE *)(v72 - 8) = 1;
          *(_QWORD *)(v72 - 16) = v74;
          goto LABEL_86;
        }
        while ( 1 )
        {
          v74 = *(__int64 **)(v72 - 16);
LABEL_86:
          v75 = *(unsigned int *)(v73 + 32);
          if ( v74 == (__int64 *)(*(_QWORD *)(v73 + 24) + 8 * v75) )
            break;
          v76 = v74 + 1;
          *(_QWORD *)(v72 - 16) = v74 + 1;
          v77 = *v74;
          if ( !v313 )
            goto LABEL_111;
          v78 = v310;
          v75 = v312;
          v76 = &v310[v312];
          if ( v310 == v76 )
          {
LABEL_156:
            if ( v312 < v311 )
            {
              ++v312;
              *v76 = v77;
              ++v309;
LABEL_112:
              v301.m128i_i64[0] = v77;
              LOBYTE(v302) = 0;
              sub_297F010((__int64)&v315, &v301);
              v36 = v315;
              v35 = v316;
              goto LABEL_113;
            }
LABEL_111:
            sub_C8CC70((__int64)&v309, v77, (__int64)v76, v75, v27, v28);
            if ( v87 )
              goto LABEL_112;
          }
          else
          {
            while ( v77 != *v78 )
            {
              if ( v76 == ++v78 )
                goto LABEL_156;
            }
          }
        }
        v316 -= 24;
        v36 = v315;
        v72 = v316;
      }
      while ( v316 != v315 );
      v35 = v315;
LABEL_113:
      v32 = v323;
      if ( v35 - v36 == v324 - v323 )
        goto LABEL_114;
    }
  }
LABEL_121:
  if ( v32 )
    j_j___libc_free_0(v32);
  if ( !v321 )
    _libc_free(v319);
  if ( v315 )
    j_j___libc_free_0(v315);
  if ( !v313 )
    _libc_free((unsigned __int64)v310);
  if ( v344 )
    j_j___libc_free_0((unsigned __int64)v344);
  if ( !v342 )
    _libc_free((unsigned __int64)v340);
  if ( v336 )
    j_j___libc_free_0((unsigned __int64)v336);
  if ( !v328 )
    _libc_free(v326.m128i_u64[1]);
  v90 = *(_QWORD **)(v1 + 32);
  v290 = (_QWORD *)(v1 + 32);
  if ( v90 != (_QWORD *)(v1 + 32) )
  {
    while ( 1 )
    {
      v92 = *(unsigned int *)(v1 + 232);
      v98 = v90[6];
      v99 = (__int64)(v90 + 6);
      v100 = *(_QWORD *)(v1 + 216);
      if ( !(_DWORD)v92 )
        goto LABEL_154;
      v35 = ((_DWORD)v92 - 1) & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
      v91 = v100 + 16 * v35;
      v28 = *(_QWORD *)v91;
      if ( *(_QWORD *)v91 != v98 )
        break;
LABEL_140:
      v92 = v100 + 16 * v92;
      if ( v91 == v92 || *(_DWORD *)(v91 + 8) == *(_DWORD *)(v1 + 248) )
        goto LABEL_154;
LABEL_142:
      v93 = v90[7];
      v295 = (_QWORD *)v99;
      if ( v93 )
      {
        v94 = sub_29812B0(v1 + 208, (__int64 *)(v93 + 32), v92, v35, v99, v28);
        v95 = *(_BYTE **)(v94 + 8);
        if ( v95 == *(_BYTE **)(v94 + 16) )
        {
          sub_297E8B0(v94, v95, v295);
        }
        else
        {
          if ( v95 )
          {
            *(_QWORD *)v95 = v90[6];
            v95 = *(_BYTE **)(v94 + 8);
          }
          *(_QWORD *)(v94 + 8) = v95 + 8;
        }
      }
      v326.m128i_i64[0] = v1;
      v326.m128i_i64[1] = (__int64)(v90 + 2);
      v96 = (_BYTE *)v90[10];
      if ( v96 && *v96 > 0x1Cu )
        sub_2981630(v326.m128i_i64, (__int64)v96);
      v97 = (_BYTE *)v90[5];
      if ( *v97 > 0x1Cu )
        sub_2981630(v326.m128i_i64, (__int64)v97);
      v90 = (_QWORD *)*v90;
      if ( v290 == v90 )
        goto LABEL_173;
    }
    v198 = 1;
    while ( v28 != -4096 )
    {
      v199 = v198 + 1;
      v35 = ((_DWORD)v92 - 1) & (unsigned int)(v198 + v35);
      v91 = v100 + 16LL * (unsigned int)v35;
      v28 = *(_QWORD *)v91;
      if ( v98 == *(_QWORD *)v91 )
        goto LABEL_140;
      v198 = v199;
    }
LABEL_154:
    v101 = (__int64 *)sub_29812B0(v1 + 208, v90 + 6, v92, v35, v99, v28);
    v99 = (__int64)(v90 + 6);
    v92 = *v101;
    if ( *v101 != v101[1] )
      v101[1] = v92;
    goto LABEL_142;
  }
LABEL_173:
  sub_2981FA0(v1);
  if ( (_BYTE)qword_5007748 )
    sub_2983070(v1);
  v287 = *(_QWORD *)(v1 + 288);
  v296 = *(_QWORD *)(v1 + 296);
  if ( v287 != v296 )
  {
    v292 = v1;
    while ( 1 )
    {
      v102 = *(_QWORD *)(v296 - 8);
      v103 = *(unsigned int *)(v292 + 280);
      v104 = *(_QWORD *)(v292 + 264);
      if ( (_DWORD)v103 )
      {
        v105 = (v103 - 1) & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
        v106 = (__int64 *)(v104 + 48LL * v105);
        v107 = *v106;
        if ( v102 != *v106 )
        {
          v196 = 1;
          while ( v107 != -4096 )
          {
            v197 = v196 + 1;
            v105 = (v103 - 1) & (v196 + v105);
            v106 = (__int64 *)(v104 + 48LL * v105);
            v107 = *v106;
            if ( v102 == *v106 )
              goto LABEL_179;
            v196 = v197;
          }
          goto LABEL_232;
        }
LABEL_179:
        if ( v106 != (__int64 *)(v104 + 48 * v103) )
        {
          v108 = v106[1];
          v109 = v108 + 8LL * *((unsigned int *)v106 + 4);
          if ( v108 != v109 )
            break;
        }
      }
LABEL_232:
      v296 -= 8;
      if ( v287 == v296 )
      {
        v1 = v292;
        goto LABEL_234;
      }
    }
    while ( 1 )
    {
      v110 = *(_QWORD *)(v109 - 8);
      v111 = *(_QWORD *)(v110 + 40);
      if ( v111 )
        break;
      v109 -= 8;
      if ( v108 == v109 )
        goto LABEL_232;
    }
    v112 = *(_QWORD *)(v110 + 32);
    v333 = (_QWORD *)sub_BD5C60(v112);
    v334 = &v340;
    v335 = &v341;
    WORD2(v337) = 512;
    v326.m128i_i64[0] = (__int64)&v327;
    v326.m128i_i64[1] = 0x200000000LL;
    v340 = &unk_49DA100;
    v330 = 0;
    v331 = 0;
    v336 = 0;
    LODWORD(v337) = 0;
    BYTE6(v337) = 7;
    v338 = 0;
    v339 = 0;
    LOWORD(v332) = 0;
    v341 = &unk_49DA0B0;
    v330 = *(_QWORD *)(v112 + 40);
    v331 = v112 + 24;
    v113 = *(_BYTE **)sub_B46C60(v112);
    v318 = v113;
    if ( v113 && (sub_B96E90((__int64)&v318, (__int64)v113, 1), (v115 = v318) != 0) )
    {
      v116 = v326.m128i_i64[0];
      v117 = v326.m128i_i32[2];
      v118 = (unsigned __int64 *)(v326.m128i_i64[0] + 16LL * v326.m128i_u32[2]);
      if ( (unsigned __int64 *)v326.m128i_i64[0] != v118 )
      {
        while ( *(_DWORD *)v116 )
        {
          v116 += 16;
          if ( v118 == (unsigned __int64 *)v116 )
            goto LABEL_309;
        }
        *(_QWORD *)(v116 + 8) = v318;
        goto LABEL_191;
      }
LABEL_309:
      if ( v326.m128i_u32[2] >= (unsigned __int64)v326.m128i_u32[3] )
      {
        v232 = v264 & 0xFFFFFFFF00000000LL;
        v264 &= 0xFFFFFFFF00000000LL;
        if ( v326.m128i_u32[3] < (unsigned __int64)v326.m128i_u32[2] + 1 )
        {
          sub_C8D5F0((__int64)&v326, &v327, v326.m128i_u32[2] + 1LL, 0x10u, v326.m128i_u32[2] + 1LL, v114);
          v118 = (unsigned __int64 *)(v326.m128i_i64[0] + 16LL * v326.m128i_u32[2]);
        }
        *v118 = v232;
        v118[1] = (unsigned __int64)v115;
        v115 = v318;
        ++v326.m128i_i32[2];
      }
      else
      {
        if ( v118 )
        {
          *(_DWORD *)v118 = 0;
          v118[1] = (unsigned __int64)v115;
          v117 = v326.m128i_i32[2];
          v115 = v318;
        }
        v326.m128i_i32[2] = v117 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v326, 0);
      v115 = v318;
    }
    if ( !v115 )
    {
LABEL_192:
      v280 = *(_QWORD *)v292;
      v119 = *(_QWORD *)(v110 + 16);
      v299 = *(_DWORD *)(v119 + 32);
      if ( v299 > 0x40 )
        sub_C43780((__int64)&v298, (const void **)(v119 + 24));
      else
        v298 = *(_BYTE **)(v119 + 24);
      v120 = *(_QWORD *)(v111 + 16);
      LODWORD(v300[1]) = *(_DWORD *)(v120 + 32);
      if ( LODWORD(v300[1]) > 0x40 )
        sub_C43780((__int64)v300, (const void **)(v120 + 24));
      else
        v300[0] = *(_QWORD *)(v120 + 24);
      sub_297C660(&v298, v300);
      LODWORD(v319) = v299;
      if ( v299 > 0x40 )
        sub_C43780((__int64)&v318, (const void **)&v298);
      else
        v318 = v298;
      sub_C46B40((__int64)&v318, v300);
      v121 = v319;
      v122 = v318;
      v301.m128i_i32[2] = v319;
      v301.m128i_i64[0] = (__int64)v318;
      v123 = *(_QWORD *)(v110 + 64);
      if ( *(_BYTE *)v123 == 17 )
      {
        if ( *(_DWORD *)(v123 + 32) <= 0x40u )
        {
          v125 = *(_QWORD *)(v123 + 24) == 0;
        }
        else
        {
          v269 = *(_DWORD *)(v123 + 32);
          v272 = v318;
          v124 = sub_C444A0(v123 + 24);
          v122 = v272;
          v125 = v269 == v124;
        }
        if ( v125 )
        {
          v123 = 0;
          goto LABEL_206;
        }
      }
      if ( v121 > 0x40 )
      {
        v273 = v122;
        if ( v121 - (unsigned int)sub_C444A0((__int64)&v301) > 0x40 || *v273 != 1 )
        {
          v177 = sub_C445E0((__int64)&v301);
          v122 = v273;
          v178 = v121 == v177;
          goto LABEL_318;
        }
      }
      else if ( v122 != (_BYTE *)1 )
      {
        if ( !v121 )
        {
LABEL_205:
          v322[0] = 257;
          v126 = *(_QWORD *)(v110 + 24);
          v127 = (_BYTE *)sub_AD6530(*(_QWORD *)(v126 + 8), (__int64)v300);
          v128 = sub_929DE0((unsigned int **)&v326, v127, (_BYTE *)v126, (__int64)&v318, 0, 0);
          v121 = v301.m128i_u32[2];
          v123 = v128;
          goto LABEL_206;
        }
        v178 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v121) == (_QWORD)v122;
LABEL_318:
        if ( v178 )
          goto LABEL_205;
        if ( v121 > 0x40 )
        {
          v274 = v122;
          if ( v121 - (unsigned int)sub_C444A0((__int64)&v301) > 0x40 )
            goto LABEL_323;
          v122 = (_BYTE *)*v274;
        }
        if ( !v122 && *(_DWORD *)v110 != 2 )
        {
          if ( *(_DWORD *)(v110 + 48) == 3 )
          {
            if ( *(_DWORD *)v110 == 3 )
            {
              v253 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v110 + 32)
                                           - 32LL * (*(_DWORD *)(*(_QWORD *)(v110 + 32) + 4LL) & 0x7FFFFFF))
                               + 8LL);
              if ( (unsigned int)*(unsigned __int8 *)(v253 + 8) - 17 <= 1 )
                v253 = **(_QWORD **)(v253 + 16);
              v254 = (__int64 **)sub_AE4570(v280, v253);
              v322[0] = 257;
              v123 = sub_2784C30(v326.m128i_i64, v123, v254, (__int64)&v318);
            }
            v190 = *(_QWORD *)(v110 + 16);
            v191 = *(_DWORD *)(v190 + 32);
            if ( v191 <= 0x40 )
            {
              v193 = *(_QWORD *)(v190 + 24) == 1;
            }
            else
            {
              v282 = *(_QWORD *)(v110 + 16);
              v192 = sub_C444A0(v190 + 24);
              v190 = v282;
              v193 = v191 - 1 == v192;
            }
            v121 = v301.m128i_u32[2];
            if ( !v193 )
            {
              v322[0] = 257;
              v194 = (_BYTE *)sub_2784C30(v326.m128i_i64, v190, *(__int64 ***)(v123 + 8), (__int64)&v318);
              v322[0] = 257;
              v195 = sub_A81850((unsigned int **)&v326, (_BYTE *)v123, v194, (__int64)&v318, 0, 0);
              v121 = v301.m128i_u32[2];
              v123 = v195;
            }
          }
          goto LABEL_206;
        }
LABEL_323:
        v179 = (_QWORD *)sub_BD5C60(*(_QWORD *)(v111 + 32));
        v180 = (__int64 **)sub_BCCE00(v179, v121);
        v322[0] = 257;
        v181 = (__int64)v180;
        v182 = sub_2784C30(v326.m128i_i64, *(_QWORD *)(v110 + 24), v180, (__int64)&v318);
        v183 = v301.m128i_i32[2];
        v281 = (_BYTE *)v182;
        if ( v301.m128i_i32[2] > 0x40u )
        {
          if ( (unsigned int)sub_C44630((__int64)&v301) == 1 )
          {
            v187 = sub_C444A0((__int64)&v301);
            LODWORD(v185) = v183 - 1;
            goto LABEL_327;
          }
          if ( (*(_QWORD *)(v301.m128i_i64[0] + 8LL * ((unsigned int)(v183 - 1) >> 6))
              & (1LL << ((unsigned __int8)v183 - 1))) != 0 )
          {
            v275 = sub_C44500((__int64)&v301);
            if ( v183 == v275 + (unsigned int)sub_C44590((__int64)&v301) )
            {
              LODWORD(v310) = v183;
              sub_C43780((__int64)&v309, (const void **)&v301);
              v183 = (int)v310;
              if ( (unsigned int)v310 > 0x40 )
              {
                sub_C43D10((__int64)&v309);
LABEL_373:
                sub_C46250((__int64)&v309);
                v207 = (unsigned int)v310;
                LODWORD(v310) = 0;
                LODWORD(v319) = v207;
                v318 = v309;
                if ( v207 > 0x40 )
                {
                  v208 = v207 - 1 - (unsigned int)sub_C444A0((__int64)&v318);
                }
                else
                {
                  v208 = 0xFFFFFFFFLL;
                  if ( v309 )
                  {
                    _BitScanReverse64(&v209, (unsigned __int64)v309);
                    v208 = 63 - ((unsigned int)v209 ^ 0x3F);
                  }
                }
                v210 = (_BYTE *)sub_ACD640(v181, v208, 0);
                if ( (unsigned int)v319 > 0x40 && v318 )
                  j_j___libc_free_0_0((unsigned __int64)v318);
                if ( (unsigned int)v310 > 0x40 && v309 )
                  j_j___libc_free_0_0((unsigned __int64)v309);
                v322[0] = 257;
                v314[0] = 257;
                v211 = sub_920A70((unsigned int **)&v326, v281, v210, (__int64)&v309, 0, 0);
                v212 = (_BYTE *)sub_AD6530(*(_QWORD *)(v211 + 8), (__int64)v281);
                v213 = sub_929DE0((unsigned int **)&v326, v212, (_BYTE *)v211, (__int64)&v318, 0, 0);
                v121 = v301.m128i_u32[2];
                v123 = v213;
LABEL_206:
                if ( v121 > 0x40 && v301.m128i_i64[0] )
                  j_j___libc_free_0_0(v301.m128i_u64[0]);
                if ( LODWORD(v300[1]) > 0x40 && v300[0] )
                  j_j___libc_free_0_0(v300[0]);
                if ( v299 > 0x40 && v298 )
                  j_j___libc_free_0_0((unsigned __int64)v298);
                if ( !v123 )
                {
                  v132 = *(unsigned __int8 **)(v111 + 32);
                  goto LABEL_226;
                }
                v129 = *(_DWORD *)v110;
                if ( *(_DWORD *)v110 > 2u )
                {
                  if ( v129 != 3 )
                    goto LABEL_494;
                  v174 = sub_B4DE30(*(_QWORD *)(v110 + 32));
                  v322[0] = 257;
                  v175 = *(_QWORD *)(v111 + 32);
                  if ( !v174 )
                    v129 = 0;
                  v309 = (_BYTE *)v123;
                  v176 = sub_BCB2B0(v333);
                  v132 = (unsigned __int8 *)sub_921130(
                                              (unsigned int **)&v326,
                                              v176,
                                              v175,
                                              &v309,
                                              1,
                                              (__int64)&v318,
                                              v129);
                  goto LABEL_225;
                }
                if ( !v129 )
                  goto LABEL_494;
                if ( *(_BYTE *)v123 == 44 )
                {
                  v200 = *(unsigned __int8 **)(v123 - 64);
                  v201 = *v200;
                  if ( (_BYTE)v201 == 17 )
                  {
                    v202 = *((_DWORD *)v200 + 8);
                    if ( v202 <= 0x40 )
                      v203 = *((_QWORD *)v200 + 3) == 0;
                    else
                      v203 = v202 == (unsigned int)sub_C444A0((__int64)(v200 + 24));
                  }
                  else
                  {
                    v249 = *((_QWORD *)v200 + 1);
                    if ( (unsigned int)*(unsigned __int8 *)(v249 + 8) - 17 > 1 || (unsigned __int8)v201 > 0x15u )
                      goto LABEL_219;
                    v283 = *(_QWORD *)(v123 - 64);
                    v250 = sub_AD7630(v283, 0, v201);
                    v251 = (unsigned __int8 *)v283;
                    if ( !v250 || *v250 != 17 )
                    {
                      if ( *(_BYTE *)(v249 + 8) == 17 )
                      {
                        v284 = *(_DWORD *)(v249 + 32);
                        if ( v284 )
                        {
                          v276 = v251;
                          v255 = 0;
                          v256 = 0;
                          while ( 1 )
                          {
                            v270 = v255;
                            v257 = sub_AD69F0(v276, v255);
                            if ( !v257 )
                              break;
                            v258 = v270;
                            if ( *(_BYTE *)v257 != 13 )
                            {
                              if ( *(_BYTE *)v257 != 17 )
                                break;
                              v259 = *(_DWORD *)(v257 + 32);
                              if ( v259 <= 0x40 )
                              {
                                v256 = *(_QWORD *)(v257 + 24) == 0;
                              }
                              else
                              {
                                v260 = sub_C444A0(v257 + 24);
                                v258 = v270;
                                v256 = v259 == v260;
                              }
                              if ( !v256 )
                                break;
                            }
                            v255 = v258 + 1;
                            if ( v284 == v255 )
                            {
                              if ( v256 )
                                goto LABEL_365;
                              goto LABEL_219;
                            }
                          }
                        }
                      }
                      goto LABEL_219;
                    }
                    v252 = *((_DWORD *)v250 + 8);
                    if ( v252 <= 0x40 )
                      v203 = *((_QWORD *)v250 + 3) == 0;
                    else
                      v203 = v252 == (unsigned int)sub_C444A0((__int64)(v250 + 24));
                  }
                  if ( v203 )
                  {
LABEL_365:
                    v204 = *(_BYTE **)(v123 - 32);
                    if ( v204 )
                    {
                      v322[0] = 257;
                      v132 = (unsigned __int8 *)sub_929DE0(
                                                  (unsigned int **)&v326,
                                                  *(_BYTE **)(v111 + 32),
                                                  v204,
                                                  (__int64)&v318,
                                                  0,
                                                  0);
                      v320 = 0;
                      sub_F5CAB0((char *)v123, 0, 0, (__int64)&v318);
                      if ( v320 )
                        v320(&v318, &v318, 3);
                      goto LABEL_225;
                    }
                  }
                }
LABEL_219:
                v314[0] = 257;
                v130 = *(unsigned __int8 **)(v111 + 32);
                v131 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))*((_QWORD *)*v334 + 4);
                if ( v131 == sub_9201A0 )
                {
                  if ( *v130 > 0x15u || *(_BYTE *)v123 > 0x15u )
                    goto LABEL_406;
                  if ( (unsigned __int8)sub_AC47B0(13) )
                    v132 = (unsigned __int8 *)sub_AD5570(13, (__int64)v130, (unsigned __int8 *)v123, 0, 0);
                  else
                    v132 = (unsigned __int8 *)sub_AABE40(0xDu, v130, (unsigned __int8 *)v123);
                }
                else
                {
                  v132 = (unsigned __int8 *)v131((__int64)v334, 13u, v130, (_BYTE *)v123, 0, 0);
                }
                if ( v132 )
                {
LABEL_225:
                  sub_BD6B90(v132, *(unsigned __int8 **)(v110 + 32));
LABEL_226:
                  sub_BD84D0(*(_QWORD *)(v110 + 32), (__int64)v132);
                  v133 = *(_BYTE **)(v292 + 320);
                  if ( v133 == *(_BYTE **)(v292 + 328) )
                  {
                    sub_24454E0(v292 + 312, v133, (_QWORD *)(v110 + 32));
                  }
                  else
                  {
                    if ( v133 )
                    {
                      *(_QWORD *)v133 = *(_QWORD *)(v110 + 32);
                      v133 = *(_BYTE **)(v292 + 320);
                    }
                    *(_QWORD *)(v292 + 320) = v133 + 8;
                  }
                  nullsub_61();
                  v340 = &unk_49DA100;
                  nullsub_63();
                  if ( (void (__fastcall **)(_QWORD, _QWORD, _QWORD))v326.m128i_i64[0] != &v327 )
                    _libc_free(v326.m128i_u64[0]);
                  goto LABEL_232;
                }
LABEL_406:
                v322[0] = 257;
                v132 = (unsigned __int8 *)sub_B504D0(13, (__int64)v130, v123, (__int64)&v318, 0, 0);
                (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE **, __int64, __int64))*v335 + 2))(
                  v335,
                  v132,
                  &v309,
                  v331,
                  v332);
                v227 = v326.m128i_i64[0];
                v228 = 16LL * v326.m128i_u32[2];
                v229 = v326.m128i_i64[0] + v228;
                if ( v326.m128i_i64[0] != v326.m128i_i64[0] + v228 )
                {
                  do
                  {
                    v230 = *(_QWORD *)(v227 + 8);
                    v231 = *(_DWORD *)v227;
                    v227 += 16;
                    sub_B99FD0((__int64)v132, v231, v230);
                  }
                  while ( v229 != v227 );
                }
                goto LABEL_225;
              }
              _RAX = (__int64)v309;
LABEL_403:
              v226 = (_BYTE *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v183) & ~_RAX);
              if ( !v183 )
                v226 = 0;
              v309 = v226;
              goto LABEL_373;
            }
          }
        }
        else
        {
          _RAX = v301.m128i_i64[0];
          if ( v301.m128i_i64[0] )
          {
            v185 = (unsigned int)(v301.m128i_i32[2] - 1);
            if ( (v301.m128i_i64[0] & (v301.m128i_i64[0] - 1)) == 0 )
            {
              _BitScanReverse64(&v186, v301.m128i_u64[0]);
              v187 = v301.m128i_i32[2] + (v186 ^ 0x3F) - 64;
LABEL_327:
              v188 = (_BYTE *)sub_ACD640(v181, (unsigned int)(v185 - v187), 0);
              v322[0] = 257;
              v189 = sub_920A70((unsigned int **)&v326, v281, v188, (__int64)&v318, 0, 0);
              v121 = v301.m128i_u32[2];
              v123 = v189;
              goto LABEL_206;
            }
            if ( _bittest64(&_RAX, v185) )
            {
              if ( !v301.m128i_i32[2] )
                goto LABEL_402;
              v223 = 64;
              if ( v301.m128i_i64[0] << (64 - v301.m128i_i8[8]) != -1 )
              {
                _BitScanReverse64(&v224, ~(v301.m128i_i64[0] << (64 - v301.m128i_i8[8])));
                v223 = v224 ^ 0x3F;
              }
              __asm { tzcnt   rcx, rax }
              if ( (unsigned int)_RCX > v301.m128i_i32[2] )
                LODWORD(_RCX) = v301.m128i_i32[2];
              if ( v301.m128i_i32[2] == v223 + (_DWORD)_RCX )
              {
LABEL_402:
                LODWORD(v310) = v301.m128i_i32[2];
                goto LABEL_403;
              }
            }
          }
        }
        v205 = (_BYTE *)sub_AD8D80(v181, (__int64)&v301);
        v322[0] = 257;
        v206 = sub_A81850((unsigned int **)&v326, v281, v205, (__int64)&v318, 0, 0);
        v121 = v301.m128i_u32[2];
        v123 = v206;
        goto LABEL_206;
      }
      v123 = *(_QWORD *)(v110 + 24);
      goto LABEL_206;
    }
LABEL_191:
    sub_B91220((__int64)&v318, (__int64)v115);
    goto LABEL_192;
  }
LABEL_234:
  v134 = *(char ***)(v1 + 320);
  v135 = *(char ***)(v1 + 312);
  if ( v135 == v134 )
  {
    v297 = 0;
  }
  else
  {
    do
    {
      v136 = *v135;
      if ( *((_QWORD *)*v135 + 5) )
      {
        v327 = 0;
        sub_F5CAB0(v136, 0, 0, (__int64)&v326);
        if ( v327 )
          v327(&v326, &v326, 3);
      }
      ++v135;
    }
    while ( v134 != v135 );
    v297 = 0;
    v137 = *(_QWORD *)(v1 + 312);
    if ( *(_QWORD *)(v1 + 320) != v137 )
    {
      *(_QWORD *)(v1 + 320) = v137;
      v297 = 1;
    }
  }
  v138 = *(_DWORD *)(v1 + 224);
  ++*(_QWORD *)(v1 + 208);
  if ( v138 )
  {
    v214 = 4 * v138;
    v139 = *(unsigned int *)(v1 + 232);
    if ( (unsigned int)(4 * v138) < 0x40 )
      v214 = 64;
    if ( v214 >= (unsigned int)v139 )
      goto LABEL_244;
    v215 = v138 - 1;
    if ( v215 )
    {
      _BitScanReverse(&v215, v215);
      v216 = *(_QWORD **)(v1 + 216);
      v217 = 1 << (33 - (v215 ^ 0x1F));
      if ( v217 < 64 )
        v217 = 64;
      if ( v217 == (_DWORD)v139 )
      {
        *(_QWORD *)(v1 + 224) = 0;
        v263 = &v216[2 * (unsigned int)v217];
        do
        {
          if ( v216 )
            *v216 = -4096;
          v216 += 2;
        }
        while ( v263 != v216 );
        goto LABEL_247;
      }
    }
    else
    {
      v216 = *(_QWORD **)(v1 + 216);
      v217 = 64;
    }
    sub_C7D6A0((__int64)v216, 16 * v139, 8);
    v218 = ((((((((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
              | (4 * v217 / 3u + 1)
              | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 4)
            | (((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
            | (4 * v217 / 3u + 1)
            | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
            | (4 * v217 / 3u + 1)
            | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 4)
          | (((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
          | (4 * v217 / 3u + 1)
          | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 16;
    v219 = (v218
          | (((((((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
              | (4 * v217 / 3u + 1)
              | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 4)
            | (((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
            | (4 * v217 / 3u + 1)
            | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
            | (4 * v217 / 3u + 1)
            | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 4)
          | (((4 * v217 / 3u + 1) | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1)) >> 2)
          | (4 * v217 / 3u + 1)
          | ((unsigned __int64)(4 * v217 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(v1 + 232) = v219;
    v220 = (_QWORD *)sub_C7D670(16 * v219, 8);
    v221 = *(unsigned int *)(v1 + 232);
    *(_QWORD *)(v1 + 224) = 0;
    *(_QWORD *)(v1 + 216) = v220;
    for ( j = &v220[2 * v221]; j != v220; v220 += 2 )
    {
      if ( v220 )
        *v220 = -4096;
    }
  }
  else if ( *(_DWORD *)(v1 + 228) )
  {
    v139 = *(unsigned int *)(v1 + 232);
    if ( (unsigned int)v139 <= 0x40 )
    {
LABEL_244:
      v140 = *(_QWORD **)(v1 + 216);
      for ( k = &v140[2 * v139]; k != v140; v140 += 2 )
        *v140 = -4096;
      *(_QWORD *)(v1 + 224) = 0;
      goto LABEL_247;
    }
    sub_C7D6A0(*(_QWORD *)(v1 + 216), 16 * v139, 8);
    *(_QWORD *)(v1 + 216) = 0;
    *(_QWORD *)(v1 + 224) = 0;
    *(_DWORD *)(v1 + 232) = 0;
  }
LABEL_247:
  v142 = *(_QWORD *)(v1 + 240);
  v143 = v142 + 32LL * *(unsigned int *)(v1 + 248);
  while ( v142 != v143 )
  {
    while ( 1 )
    {
      v144 = *(_QWORD *)(v143 - 24);
      v143 -= 32;
      if ( !v144 )
        break;
      j_j___libc_free_0(v144);
      if ( v142 == v143 )
        goto LABEL_251;
    }
  }
LABEL_251:
  v145 = *(_DWORD *)(v1 + 272);
  ++*(_QWORD *)(v1 + 256);
  *(_DWORD *)(v1 + 248) = 0;
  if ( v145 || *(_DWORD *)(v1 + 276) )
  {
    v146 = *(_QWORD **)(v1 + 264);
    v147 = 4 * v145;
    v148 = 48LL * *(unsigned int *)(v1 + 280);
    if ( (unsigned int)(4 * v145) < 0x40 )
      v147 = 64;
    v149 = &v146[(unsigned __int64)v148 / 8];
    if ( v147 >= *(_DWORD *)(v1 + 280) )
    {
      for ( ; v149 != v146; v146 += 6 )
      {
        if ( *v146 != -4096 )
        {
          if ( *v146 != -8192 )
          {
            v150 = v146[1];
            if ( (_QWORD *)v150 != v146 + 3 )
              _libc_free(v150);
          }
          *v146 = -4096;
        }
      }
      goto LABEL_263;
    }
    do
    {
      if ( *v146 != -8192 && *v146 != -4096 )
      {
        v164 = v146[1];
        if ( (_QWORD *)v164 != v146 + 3 )
          _libc_free(v164);
      }
      v146 += 6;
    }
    while ( v149 != v146 );
    v165 = *(_QWORD **)(v1 + 264);
    v166 = *(unsigned int *)(v1 + 280);
    if ( v145 )
    {
      v167 = 64;
      if ( v145 != 1 )
      {
        _BitScanReverse(&v168, v145 - 1);
        v167 = 1 << (33 - (v168 ^ 0x1F));
        if ( v167 < 64 )
          v167 = 64;
      }
      if ( (_DWORD)v166 == v167 )
      {
        *(_QWORD *)(v1 + 272) = 0;
        v261 = &v165[6 * v166];
        do
        {
          if ( v165 )
            *v165 = -4096;
          v165 += 6;
        }
        while ( v261 != v165 );
      }
      else
      {
        sub_C7D6A0((__int64)v165, v148, 8);
        v169 = ((((((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                  | (4 * v167 / 3u + 1)
                  | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
                | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                | (4 * v167 / 3u + 1)
                | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 8)
              | (((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                | (4 * v167 / 3u + 1)
                | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
              | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
              | (4 * v167 / 3u + 1)
              | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 16;
        v170 = (v169
              | (((((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                  | (4 * v167 / 3u + 1)
                  | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
                | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                | (4 * v167 / 3u + 1)
                | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 8)
              | (((((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
                | (4 * v167 / 3u + 1)
                | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 4)
              | (((4 * v167 / 3u + 1) | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1)) >> 2)
              | (4 * v167 / 3u + 1)
              | ((unsigned __int64)(4 * v167 / 3u + 1) >> 1))
             + 1;
        *(_DWORD *)(v1 + 280) = v170;
        v171 = (_QWORD *)sub_C7D670(48 * v170, 8);
        v172 = *(unsigned int *)(v1 + 280);
        *(_QWORD *)(v1 + 272) = 0;
        *(_QWORD *)(v1 + 264) = v171;
        for ( m = &v171[6 * v172]; m != v171; v171 += 6 )
        {
          if ( v171 )
            *v171 = -4096;
        }
      }
    }
    else
    {
      if ( !(_DWORD)v166 )
      {
LABEL_263:
        *(_QWORD *)(v1 + 272) = 0;
        goto LABEL_264;
      }
      sub_C7D6A0((__int64)v165, v148, 8);
      *(_QWORD *)(v1 + 264) = 0;
      *(_QWORD *)(v1 + 272) = 0;
      *(_DWORD *)(v1 + 280) = 0;
    }
  }
LABEL_264:
  v151 = *(_QWORD *)(v1 + 288);
  if ( v151 != *(_QWORD *)(v1 + 296) )
    *(_QWORD *)(v1 + 296) = v151;
  v152 = *(_DWORD *)(v1 + 72);
  ++*(_QWORD *)(v1 + 56);
  if ( v152 || *(_DWORD *)(v1 + 76) )
  {
    v153 = *(_QWORD *)(v1 + 64);
    v154 = 4 * v152;
    if ( (unsigned int)(4 * v152) < 0x40 )
      v154 = 64;
    v155 = v153 + 56LL * *(unsigned int *)(v1 + 80);
    if ( v154 >= *(_DWORD *)(v1 + 80) )
    {
      if ( v153 == v155 )
        goto LABEL_285;
      while ( 1 )
      {
        v156 = *(_QWORD *)(v153 + 16);
        if ( v156 == -4096 )
          break;
        if ( v156 != -8192 || *(_QWORD *)(v153 + 8) != -8192 || *(_QWORD *)v153 != -8192 )
          goto LABEL_274;
LABEL_283:
        *(_QWORD *)(v153 + 16) = -4096;
        *(_QWORD *)(v153 + 8) = -4096;
        *(_QWORD *)v153 = -4096;
LABEL_284:
        v153 += 56;
        if ( v153 == v155 )
          goto LABEL_285;
      }
      if ( *(_QWORD *)(v153 + 8) == -4096 && *(_QWORD *)v153 == -4096 )
        goto LABEL_284;
LABEL_274:
      v157 = *(unsigned int *)(v153 + 48);
      if ( (_DWORD)v157 )
      {
        v158 = *(_QWORD **)(v153 + 32);
        v159 = &v158[11 * v157];
        do
        {
          if ( *v158 != -8192 && *v158 != -4096 )
          {
            v160 = v158[1];
            if ( (_QWORD *)v160 != v158 + 3 )
              _libc_free(v160);
          }
          v158 += 11;
        }
        while ( v159 != v158 );
        v157 = *(unsigned int *)(v153 + 48);
      }
      sub_C7D6A0(*(_QWORD *)(v153 + 32), 88 * v157, 8);
      goto LABEL_283;
    }
    v233 = v152;
    v234 = 56LL * *(unsigned int *)(v1 + 80);
    while ( 1 )
    {
      v235 = *(_QWORD *)(v153 + 16);
      if ( v235 == -4096 )
      {
        if ( *(_QWORD *)(v153 + 8) != -4096 || *(_QWORD *)v153 != -4096 )
        {
LABEL_417:
          v236 = *(unsigned int *)(v153 + 48);
          if ( (_DWORD)v236 )
          {
            v237 = *(_QWORD **)(v153 + 32);
            v293 = &v237[11 * v236];
            do
            {
              if ( *v237 != -8192 && *v237 != -4096 )
              {
                v238 = v237[1];
                if ( (_QWORD *)v238 != v237 + 3 )
                {
                  v288 = v237;
                  _libc_free(v238);
                  v237 = v288;
                }
              }
              v237 += 11;
            }
            while ( v293 != v237 );
            v236 = *(unsigned int *)(v153 + 48);
          }
          sub_C7D6A0(*(_QWORD *)(v153 + 32), 88 * v236, 8);
        }
      }
      else if ( v235 != -8192 || *(_QWORD *)(v153 + 8) != -8192 || *(_QWORD *)v153 != -8192 )
      {
        goto LABEL_417;
      }
      v153 += 56;
      if ( v153 == v155 )
      {
        v239 = *(_QWORD **)(v1 + 64);
        v240 = *(unsigned int *)(v1 + 80);
        v241 = v233;
        if ( v233 )
        {
          v242 = 64;
          if ( v241 != 1 )
          {
            _BitScanReverse(&v243, v241 - 1);
            v242 = 1 << (33 - (v243 ^ 0x1F));
            if ( v242 < 64 )
              v242 = 64;
          }
          if ( (_DWORD)v240 == v242 )
          {
            *(_QWORD *)(v1 + 72) = 0;
            v262 = &v239[7 * v240];
            do
            {
              if ( v239 )
              {
                *v239 = -4096;
                v239[1] = -4096;
                v239[2] = -4096;
              }
              v239 += 7;
            }
            while ( v262 != v239 );
          }
          else
          {
            sub_C7D6A0((__int64)v239, v234, 8);
            v244 = ((((((((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                      | (4 * v242 / 3u + 1)
                      | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                    | (4 * v242 / 3u + 1)
                    | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 8)
                  | (((((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                    | (4 * v242 / 3u + 1)
                    | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                  | (4 * v242 / 3u + 1)
                  | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 16;
            v245 = (v244
                  | (((((((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                      | (4 * v242 / 3u + 1)
                      | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                    | (4 * v242 / 3u + 1)
                    | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 8)
                  | (((((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                    | (4 * v242 / 3u + 1)
                    | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v242 / 3u + 1) | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1)) >> 2)
                  | (4 * v242 / 3u + 1)
                  | ((unsigned __int64)(4 * v242 / 3u + 1) >> 1))
                 + 1;
            *(_DWORD *)(v1 + 80) = v245;
            v246 = (_QWORD *)sub_C7D670(56 * v245, 8);
            v247 = *(unsigned int *)(v1 + 80);
            *(_QWORD *)(v1 + 72) = 0;
            *(_QWORD *)(v1 + 64) = v246;
            for ( n = &v246[7 * v247]; n != v246; v246 += 7 )
            {
              if ( v246 )
              {
                *v246 = -4096;
                v246[1] = -4096;
                v246[2] = -4096;
              }
            }
          }
          break;
        }
        if ( (_DWORD)v240 )
        {
          sub_C7D6A0((__int64)v239, v234, 8);
          *(_QWORD *)(v1 + 64) = 0;
          *(_QWORD *)(v1 + 72) = 0;
          *(_DWORD *)(v1 + 80) = 0;
          break;
        }
LABEL_285:
        *(_QWORD *)(v1 + 72) = 0;
        break;
      }
    }
  }
  sub_297CA60(v1 + 88);
  sub_297CA60(v1 + 120);
  v161 = *(_QWORD **)(v1 + 32);
  while ( v290 != v161 )
  {
    v162 = (unsigned __int64)v161;
    v161 = (_QWORD *)*v161;
    j_j___libc_free_0(v162);
  }
  *(_QWORD *)(v1 + 48) = 0;
  *(_QWORD *)(v1 + 40) = v290;
  *(_QWORD *)(v1 + 32) = v290;
  return v297;
}
