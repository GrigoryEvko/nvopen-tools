// Function: sub_3487360
// Address: 0x3487360
//
__int64 __fastcall sub_3487360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7,
        int *a8,
        unsigned int a9)
{
  __int64 v9; // r15
  signed int v10; // r13d
  __int64 v11; // r15
  __int64 v13; // r9
  __m128i *v14; // r14
  unsigned __int8 v15; // r11
  char v16; // bl
  unsigned __int16 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 (*v21)(); // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int128 v24; // xmm1
  __int128 v25; // xmm2
  __int128 v26; // rax
  int v27; // ebx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int128 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD *v35; // r12
  unsigned __int64 v36; // r13
  __int64 v37; // rax
  _QWORD *v38; // rbx
  __int64 v39; // rcx
  int v40; // edx
  _QWORD **v41; // rbx
  unsigned __int8 *v42; // rax
  __int64 *v43; // rsi
  unsigned __int64 v44; // r12
  __int64 v45; // rax
  unsigned int v46; // ebx
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int128 v49; // rax
  char v50; // al
  __int64 v51; // rax
  unsigned int v52; // r15d
  __int128 v53; // xmm6
  __int128 v54; // xmm7
  __int128 v55; // rax
  __int64 v56; // r13
  int v57; // ebx
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int128 v61; // rax
  _QWORD *v62; // r15
  unsigned __int64 v63; // r13
  __int64 v64; // rcx
  int v65; // edx
  __int64 v66; // rax
  __int128 v67; // rax
  __int64 v68; // rax
  _QWORD *v69; // r8
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rsi
  __int64 v74; // rdx
  __int128 v75; // rax
  _QWORD *v76; // rbx
  __int64 v77; // r13
  __int128 v78; // kr00_16
  _QWORD *v79; // r11
  int v80; // eax
  __int128 v81; // rax
  unsigned __int8 *v82; // rax
  __int64 v83; // rax
  __int64 v84; // r12
  __int64 v85; // r13
  __int64 v86; // rax
  __int64 v87; // rbx
  __int64 v88; // rbx
  __int64 v89; // rax
  unsigned int v90; // ebx
  __m128i v91; // xmm4
  __m128i v92; // xmm5
  __int64 v93; // rsi
  __int64 v94; // rdx
  __int128 v95; // rax
  __int64 v96; // r15
  __int64 v97; // rsi
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // rax
  __int64 v103; // rax
  __int128 v104; // rax
  _QWORD *v105; // rbx
  unsigned __int64 v106; // r13
  int v107; // eax
  unsigned int v108; // ecx
  __int64 v109; // r8
  int v110; // edx
  __int64 v111; // rax
  bool v112; // r13
  _QWORD *v113; // rax
  __int64 *v114; // rsi
  int v115; // edx
  int v116; // r14d
  __int64 *v117; // r13
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // r8
  __int64 *v121; // rdx
  __int64 v122; // rcx
  __int64 *v123; // rax
  unsigned __int64 v124; // rsi
  unsigned __int64 v125; // rsi
  bool v126; // al
  __int64 v127; // rcx
  __int64 v128; // r9
  __int64 v129; // r8
  unsigned __int8 v130; // r11
  bool v131; // al
  __int64 *v132; // rbx
  unsigned int v133; // r12d
  __int64 v134; // rax
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 v140; // rsi
  __int64 v141; // r14
  _QWORD *v142; // rax
  __int64 v143; // rdx
  __int64 *v144; // rsi
  unsigned __int64 v145; // rsi
  __int64 v146; // rcx
  __int64 v147; // r8
  __int64 v148; // rax
  __int64 v149; // r15
  __int64 *v150; // rsi
  _QWORD *v151; // rbx
  __int64 v152; // rax
  __int64 v153; // r15
  __int64 *v154; // rsi
  _QWORD *v155; // rbx
  __int64 v156; // rax
  __int64 v157; // r15
  __int64 *v158; // rsi
  _QWORD *v159; // rbx
  __int64 v160; // r15
  __int64 *v161; // rsi
  _QWORD *v162; // rbx
  unsigned __int64 v163; // rsi
  _QWORD *m; // rbx
  _QWORD *k; // r15
  _QWORD *j; // rbx
  _QWORD *i; // r15
  _QWORD *kk; // rbx
  _QWORD *jj; // r15
  _QWORD *ii; // rbx
  _QWORD *n; // r15
  __int64 v172; // rcx
  __int64 v173; // r8
  int v174; // edx
  __int128 v175; // rax
  int v176; // eax
  unsigned int v177; // ecx
  int v178; // edx
  __int64 v179; // rbx
  __int64 v180; // rdx
  __int64 v181; // rcx
  __int64 v182; // r8
  signed __int64 v183; // rax
  __int128 v184; // rax
  __int64 v185; // rdi
  bool v186; // bl
  __int64 v187; // rcx
  __int64 v188; // r8
  int v189; // edx
  signed __int64 v190; // rax
  char v191; // al
  int v192; // ecx
  int v193; // edx
  __int64 v194; // rax
  char v195; // al
  char v196; // al
  unsigned int v197; // ecx
  unsigned int v198; // ecx
  unsigned int v199; // ecx
  __int64 v200; // [rsp-10h] [rbp-230h]
  __int128 v201; // [rsp-10h] [rbp-230h]
  __int128 v202; // [rsp-10h] [rbp-230h]
  __int128 v203; // [rsp+0h] [rbp-220h]
  signed int v204; // [rsp+0h] [rbp-220h]
  __int16 v205; // [rsp+2h] [rbp-21Eh]
  unsigned int v206; // [rsp+10h] [rbp-210h]
  int v207; // [rsp+10h] [rbp-210h]
  __int128 v208; // [rsp+10h] [rbp-210h]
  unsigned __int8 v209; // [rsp+10h] [rbp-210h]
  int v210; // [rsp+20h] [rbp-200h]
  __int128 v211; // [rsp+20h] [rbp-200h]
  __int64 v212; // [rsp+20h] [rbp-200h]
  __int64 v213; // [rsp+28h] [rbp-1F8h]
  __int128 v214; // [rsp+30h] [rbp-1F0h]
  _DWORD *v215; // [rsp+30h] [rbp-1F0h]
  int v216; // [rsp+30h] [rbp-1F0h]
  __int64 v217; // [rsp+30h] [rbp-1F0h]
  int v218; // [rsp+30h] [rbp-1F0h]
  __int64 v219; // [rsp+30h] [rbp-1F0h]
  __int64 v220; // [rsp+40h] [rbp-1E0h]
  __int64 v221; // [rsp+40h] [rbp-1E0h]
  __int64 v222; // [rsp+40h] [rbp-1E0h]
  __int64 v223; // [rsp+40h] [rbp-1E0h]
  int v224; // [rsp+40h] [rbp-1E0h]
  unsigned int *v225; // [rsp+40h] [rbp-1E0h]
  unsigned int *v226; // [rsp+40h] [rbp-1E0h]
  unsigned int *v227; // [rsp+40h] [rbp-1E0h]
  unsigned int *v228; // [rsp+40h] [rbp-1E0h]
  __int128 v229; // [rsp+50h] [rbp-1D0h]
  __m128i v230; // [rsp+50h] [rbp-1D0h]
  unsigned __int8 v231; // [rsp+50h] [rbp-1D0h]
  unsigned __int8 v232; // [rsp+50h] [rbp-1D0h]
  unsigned __int8 v233; // [rsp+50h] [rbp-1D0h]
  unsigned __int8 v234; // [rsp+50h] [rbp-1D0h]
  unsigned int v235; // [rsp+60h] [rbp-1C0h]
  __int64 v236; // [rsp+60h] [rbp-1C0h]
  __int64 v237; // [rsp+60h] [rbp-1C0h]
  __int64 v238; // [rsp+60h] [rbp-1C0h]
  __int128 v239; // [rsp+60h] [rbp-1C0h]
  __int128 v240; // [rsp+60h] [rbp-1C0h]
  unsigned __int8 v241; // [rsp+60h] [rbp-1C0h]
  __int64 v242; // [rsp+60h] [rbp-1C0h]
  __int64 (*v243)(); // [rsp+60h] [rbp-1C0h]
  char v244; // [rsp+60h] [rbp-1C0h]
  __int64 (*v245)(); // [rsp+60h] [rbp-1C0h]
  __int64 (*v246)(); // [rsp+60h] [rbp-1C0h]
  __int64 v247; // [rsp+70h] [rbp-1B0h]
  __int64 (*v248)(); // [rsp+70h] [rbp-1B0h]
  unsigned __int8 v249; // [rsp+70h] [rbp-1B0h]
  __int128 v250; // [rsp+70h] [rbp-1B0h]
  __int128 v251; // [rsp+70h] [rbp-1B0h]
  __int128 v252; // [rsp+70h] [rbp-1B0h]
  __int64 v253; // [rsp+70h] [rbp-1B0h]
  __int64 v254; // [rsp+70h] [rbp-1B0h]
  _QWORD *v255; // [rsp+70h] [rbp-1B0h]
  char v256; // [rsp+70h] [rbp-1B0h]
  char v257; // [rsp+70h] [rbp-1B0h]
  __int64 (*v258)(); // [rsp+70h] [rbp-1B0h]
  char v259; // [rsp+70h] [rbp-1B0h]
  unsigned __int8 v260; // [rsp+80h] [rbp-1A0h]
  unsigned __int8 v261; // [rsp+80h] [rbp-1A0h]
  unsigned int v262; // [rsp+80h] [rbp-1A0h]
  signed int v263; // [rsp+80h] [rbp-1A0h]
  __int64 v264; // [rsp+80h] [rbp-1A0h]
  unsigned __int64 v265; // [rsp+80h] [rbp-1A0h]
  __int128 v266; // [rsp+80h] [rbp-1A0h]
  unsigned __int8 v267; // [rsp+80h] [rbp-1A0h]
  __int64 *v268; // [rsp+80h] [rbp-1A0h]
  __int64 *v269; // [rsp+80h] [rbp-1A0h]
  unsigned __int8 v270; // [rsp+80h] [rbp-1A0h]
  unsigned __int8 v271; // [rsp+80h] [rbp-1A0h]
  unsigned __int8 v272; // [rsp+80h] [rbp-1A0h]
  unsigned int v273; // [rsp+90h] [rbp-190h]
  __int64 v274; // [rsp+90h] [rbp-190h]
  _QWORD *v275; // [rsp+90h] [rbp-190h]
  __int64 *v276; // [rsp+90h] [rbp-190h]
  unsigned int v277; // [rsp+90h] [rbp-190h]
  __int64 v278; // [rsp+98h] [rbp-188h]
  __int64 v279; // [rsp+98h] [rbp-188h]
  unsigned __int8 v280; // [rsp+FCh] [rbp-124h] BYREF
  __int64 v281; // [rsp+100h] [rbp-120h] BYREF
  __int64 v282; // [rsp+108h] [rbp-118h]
  int v283; // [rsp+11Ch] [rbp-104h] BYREF
  __int64 v284; // [rsp+120h] [rbp-100h] BYREF
  __int64 v285; // [rsp+128h] [rbp-F8h]
  __int64 v286; // [rsp+130h] [rbp-F0h] BYREF
  int v287; // [rsp+138h] [rbp-E8h]
  _QWORD *v288; // [rsp+140h] [rbp-E0h] BYREF
  _QWORD **v289; // [rsp+148h] [rbp-D8h]
  __int64 v290; // [rsp+150h] [rbp-D0h]
  unsigned int *v291; // [rsp+160h] [rbp-C0h] BYREF
  unsigned __int8 *v292; // [rsp+168h] [rbp-B8h]
  __int64 v293; // [rsp+170h] [rbp-B0h]
  _QWORD *v294; // [rsp+180h] [rbp-A0h] BYREF
  _QWORD *v295; // [rsp+188h] [rbp-98h]
  _BYTE *v296; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v297; // [rsp+1A8h] [rbp-78h]
  _BYTE v298[112]; // [rsp+1B0h] [rbp-70h] BYREF

  v9 = a2;
  v10 = *(_DWORD *)(a2 + 24);
  v282 = a3;
  v281 = a2;
  v280 = a6;
  if ( v10 == 433 || v10 == 244 )
  {
    *a8 = 0;
    return **(_QWORD **)(a2 + 40);
  }
  if ( a9 > 6 )
    return 0;
  v13 = *a4;
  v14 = (__m128i *)a4;
  v15 = a5;
  v16 = a5;
  v273 = *(_DWORD *)(a2 + 28);
  v17 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)v282);
  v18 = *v17;
  v285 = *((_QWORD *)v17 + 1);
  v19 = *(_QWORD *)(a2 + 56);
  LOWORD(v284) = v18;
  if ( !v19 )
    goto LABEL_14;
  v18 = 1;
  do
  {
    while ( (_DWORD)v282 != *(_DWORD *)(v19 + 8) )
    {
      v19 = *(_QWORD *)(v19 + 32);
      if ( !v19 )
        goto LABEL_13;
    }
    if ( !(_DWORD)v18 )
      goto LABEL_14;
    v20 = *(_QWORD *)(v19 + 32);
    if ( !v20 )
      goto LABEL_20;
    if ( (_DWORD)v282 == *(_DWORD *)(v20 + 8) )
      goto LABEL_14;
    v19 = *(_QWORD *)(v20 + 32);
    v18 = 0;
  }
  while ( v19 );
LABEL_13:
  if ( (_DWORD)v18 == 1 )
  {
LABEL_14:
    if ( v10 == 12 )
      goto LABEL_20;
    if ( v10 == 233 )
    {
      v21 = *(__int64 (**)())(*(_QWORD *)a1 + 1560LL);
      if ( v21 != sub_2D566A0 )
      {
        v260 = a5;
        v278 = v13;
        if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD))v21)(
               a1,
               (unsigned int)v284,
               v285,
               *(unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                                   + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL)),
               *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL)
                         + 8)) )
        {
          v13 = v278;
          v15 = v260;
          goto LABEL_20;
        }
      }
    }
    return 0;
  }
LABEL_20:
  v22 = *(_QWORD *)(a2 + 80);
  v286 = v22;
  if ( v22 )
  {
    v261 = v15;
    v279 = v13;
    sub_B96E90((__int64)&v286, v22, 1);
    v15 = v261;
    v13 = v279;
  }
  v290 = 0;
  v262 = a9 + 1;
  v287 = *(_DWORD *)(v9 + 72);
  v289 = &v288;
  v288 = &v288;
  if ( v10 > 151 )
  {
    if ( v10 == 230 )
    {
      *(_QWORD *)&v81 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, int *, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                          a1,
                          **(_QWORD **)(v9 + 40),
                          *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                          v14,
                          v15,
                          v280,
                          a8,
                          v262);
      if ( (_QWORD)v81 )
      {
        v82 = sub_3406EB0(
                v14,
                0xE6u,
                (__int64)&v286,
                (unsigned int)v284,
                v285,
                (__int64)&v286,
                v81,
                *(_OWORD *)(*(_QWORD *)(v9 + 40) + 40LL));
        v41 = (_QWORD **)v288;
        v11 = (__int64)v82;
        goto LABEL_61;
      }
      goto LABEL_59;
    }
    if ( v10 > 230 )
    {
      if ( v10 == 233 || v10 == 248 )
      {
        if ( (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, int *, _QWORD))(*(_QWORD *)a1 + 2264LL))(
               a1,
               **(_QWORD **)(v9 + 40),
               *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
               v14,
               v15,
               v280,
               a8,
               v262) )
        {
          v42 = sub_33FAF80(
                  (__int64)v14,
                  (unsigned int)v10,
                  (__int64)&v286,
                  (unsigned int)v284,
                  v285,
                  (unsigned int)&v286,
                  a7);
          v41 = (_QWORD **)v288;
          v11 = (__int64)v42;
          goto LABEL_61;
        }
        goto LABEL_59;
      }
      goto LABEL_67;
    }
    if ( v10 != 156 )
    {
      if ( (unsigned int)(v10 - 205) <= 1 )
      {
        v45 = *(_QWORD *)(v9 + 40);
        v46 = v15;
        v47 = *(_QWORD *)(v45 + 40);
        v48 = *(_QWORD *)(v45 + 48);
        LODWORD(v294) = 2;
        *(_QWORD *)&v49 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __m128i *, _QWORD, _QWORD, _QWORD **, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                            a1,
                            v47,
                            v48,
                            v14,
                            v15,
                            v280,
                            &v294,
                            v262);
        v274 = v49;
        if ( (_QWORD)v49 )
        {
          if ( (int)v294 <= 1 )
          {
            v250 = v49;
            v238 = sub_22077B0(0x98u);
            v68 = sub_33ECD10(1u);
            v69 = (_QWORD *)v238;
            *(_QWORD *)(v238 + 64) = v68;
            *(_QWORD *)(v238 + 80) = 0x100000000LL;
            *(_QWORD *)(v238 + 104) = 0xFFFFFFFFLL;
            *(_QWORD *)(v238 + 128) = v238 + 16;
            *(_WORD *)(v238 + 50) = -1;
            v70 = v238 + 112;
            *(_WORD *)(v238 + 48) = 0;
            *(_QWORD *)(v238 + 144) = 0;
            *(_QWORD *)(v238 + 16) = 0;
            *(_QWORD *)(v238 + 24) = 0;
            *(_QWORD *)(v238 + 32) = 0;
            *(_QWORD *)(v238 + 40) = 328;
            *(_DWORD *)(v238 + 52) = -1;
            *(_QWORD *)(v238 + 56) = 0;
            *(_QWORD *)(v238 + 72) = 0;
            *(_DWORD *)(v238 + 88) = 0;
            *(_QWORD *)(v238 + 96) = 0;
            *(_QWORD *)(v238 + 136) = 0;
            v71 = *(_QWORD *)(v274 + 56);
            *(_QWORD *)(v238 + 112) = v250;
            *(_DWORD *)(v238 + 120) = DWORD2(v250);
            *(_QWORD *)(v238 + 144) = v71;
            if ( v71 )
              *(_QWORD *)(v71 + 24) = v238 + 144;
            *(_QWORD *)(v238 + 56) = v70;
            *(_QWORD *)(v238 + 136) = v274 + 56;
            *(_QWORD *)(v274 + 56) = v70;
            *(_DWORD *)(v238 + 80) = 1;
            v239 = v250;
            sub_2208C80(v69, (__int64)&v288);
            v72 = *(_QWORD *)(v9 + 40);
            ++v290;
            v73 = *(_QWORD *)(v72 + 80);
            v74 = *(_QWORD *)(v72 + 88);
            LODWORD(v296) = 2;
            *(_QWORD *)&v75 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __m128i *, _QWORD, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                                a1,
                                v73,
                                v74,
                                v14,
                                v46,
                                v280,
                                &v296,
                                v262);
            v76 = v288;
            v251 = v75;
            v77 = v75;
            v78 = v239;
            if ( v288 != &v288 )
            {
              do
              {
                v79 = v76;
                v76 = (_QWORD *)*v76;
                v265 = (unsigned __int64)v79;
                sub_33CF710((__int64)(v79 + 2));
                j_j___libc_free_0(v265);
              }
              while ( v76 != &v288 );
              v78 = v239;
            }
            v290 = 0;
            v289 = &v288;
            v288 = &v288;
            if ( v77 )
            {
              v80 = (int)v296;
              if ( (int)v296 <= 1 && (!(_DWORD)v296 || !(_DWORD)v294) )
              {
                v192 = v285;
                if ( (int)v296 > (int)v294 )
                  v80 = (int)v294;
                v193 = v284;
                *a8 = v80;
                v194 = sub_3288B20(
                         (int)v14,
                         (int)&v286,
                         v193,
                         v192,
                         **(_QWORD **)(v9 + 40),
                         *(_QWORD *)(*(_QWORD *)(v9 + 40) + 8LL),
                         v78,
                         v251,
                         0);
                v41 = (_QWORD **)v288;
                v11 = v194;
                goto LABEL_61;
              }
              if ( !*(_QWORD *)(v274 + 56) )
                sub_33ECEA0(v14, v274);
              if ( !*(_QWORD *)(v77 + 56) )
                sub_33ECEA0(v14, v77);
            }
            else if ( !*(_QWORD *)(v274 + 56) )
            {
              sub_33ECEA0(v14, v274);
            }
          }
          else if ( !*(_QWORD *)(v49 + 56) )
          {
            sub_33ECEA0(v14, v49);
          }
        }
        goto LABEL_59;
      }
      goto LABEL_67;
    }
    v117 = *(__int64 **)(v9 + 40);
    v118 = 5LL * *(unsigned int *)(v9 + 64);
    v268 = &v117[v118];
    v119 = (__int64)(0xCCCCCCCCCCCCCCCDLL * ((v118 * 8) >> 3)) >> 2;
    v120 = v119;
    if ( v119 )
    {
      v121 = *(__int64 **)(v9 + 40);
      v122 = 0x8001000001000LL;
      v123 = &v117[20 * v119];
      do
      {
        v124 = *(unsigned int *)(*v121 + 24);
        if ( (unsigned int)v124 > 0x33 || !_bittest64(&v122, v124) )
        {
          v276 = v121;
          goto LABEL_148;
        }
        v125 = *(unsigned int *)(v121[5] + 24);
        if ( (unsigned int)v125 > 0x33 || !_bittest64(&v122, v125) )
        {
          v276 = v121 + 5;
          goto LABEL_148;
        }
        v145 = *(unsigned int *)(v121[10] + 24);
        if ( (unsigned int)v145 > 0x33 || !_bittest64(&v122, v145) )
        {
          v276 = v121 + 10;
          goto LABEL_148;
        }
        v163 = *(unsigned int *)(v121[15] + 24);
        if ( (unsigned int)v163 > 0x33 || !_bittest64(&v122, v163) )
        {
          v276 = v121 + 15;
          goto LABEL_148;
        }
        v121 += 20;
      }
      while ( v123 != v121 );
      v276 = v121;
    }
    else
    {
      v276 = *(__int64 **)(v9 + 40);
    }
    v183 = (char *)v268 - (char *)v276;
    if ( (char *)v268 - (char *)v276 != 80 )
    {
      if ( v183 != 120 )
      {
        if ( v183 != 40 )
        {
LABEL_298:
          v276 = v268;
          goto LABEL_149;
        }
LABEL_362:
        v199 = *(_DWORD *)(*v276 + 24);
        if ( v199 <= 0x33 && ((0x8001000001000uLL >> v199) & 1) != 0 )
          goto LABEL_298;
LABEL_148:
        v41 = &v288;
        if ( v268 != v276 )
          goto LABEL_60;
LABEL_149:
        v241 = v15;
        v253 = v120;
        v126 = sub_328D6E0(a1, 0xCu, v284);
        v129 = v253;
        v130 = v241;
        if ( v126 )
        {
          v131 = sub_328D6E0(a1, 0x9Cu, v284);
          v129 = v253;
          v130 = v241;
          if ( v131 )
            goto LABEL_151;
        }
        v293 = a1;
        v291 = (unsigned int *)&v284;
        v292 = &v280;
        if ( v129 )
        {
          v219 = v9;
          v209 = v130;
          v269 = &v117[20 * v129];
          do
          {
            if ( *(_DWORD *)(*v117 + 24) != 51 )
            {
              v160 = v293;
              v246 = *(__int64 (**)())(*(_QWORD *)v293 + 616LL);
              v234 = *v292;
              v228 = v291;
              v161 = (__int64 *)(*(_QWORD *)(*v117 + 96) + 24LL);
              v162 = sub_C33340();
              if ( (_QWORD *)*v161 == v162 )
                sub_C3C790(&v294, (_QWORD **)v161);
              else
                sub_C33EB0(&v294, v161);
              if ( v294 == v162 )
                sub_C3CCB0((__int64)&v294);
              else
                sub_C34440((unsigned __int8 *)&v294);
              if ( v294 == v162 )
                sub_C3C840(&v296, &v294);
              else
                sub_C338E0((__int64)&v296, (__int64)&v294);
              v256 = 0;
              if ( v246 != sub_2FE3170 )
                v256 = ((__int64 (__fastcall *)(__int64, _BYTE **, _QWORD, _QWORD, _QWORD))v246)(
                         v160,
                         &v296,
                         *v228,
                         *((_QWORD *)v228 + 1),
                         v234);
              if ( v162 == (_QWORD *)v296 )
              {
                if ( v297 )
                {
                  for ( i = (_QWORD *)(v297 + 24LL * *(_QWORD *)(v297 - 8)); (_QWORD *)v297 != i; sub_91D830(i) )
                    i -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v296);
              }
              if ( v162 == v294 )
              {
                if ( v295 )
                {
                  for ( j = &v295[3 * *(v295 - 1)]; v295 != j; sub_91D830(j) )
                    j -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v294);
              }
              if ( !v256 )
              {
                v9 = v219;
                v130 = v209;
                goto LABEL_307;
              }
            }
            v148 = v117[5];
            if ( *(_DWORD *)(v148 + 24) != 51 )
            {
              v149 = v293;
              v243 = *(__int64 (**)())(*(_QWORD *)v293 + 616LL);
              v231 = *v292;
              v225 = v291;
              v150 = (__int64 *)(*(_QWORD *)(v148 + 96) + 24LL);
              v151 = sub_C33340();
              if ( (_QWORD *)*v150 == v151 )
                sub_C3C790(&v294, (_QWORD **)v150);
              else
                sub_C33EB0(&v294, v150);
              if ( v294 == v151 )
                sub_C3CCB0((__int64)&v294);
              else
                sub_C34440((unsigned __int8 *)&v294);
              if ( v294 == v151 )
                sub_C3C840(&v296, &v294);
              else
                sub_C338E0((__int64)&v296, (__int64)&v294);
              v257 = 0;
              if ( v243 != sub_2FE3170 )
                v257 = ((__int64 (__fastcall *)(__int64, _BYTE **, _QWORD, _QWORD, _QWORD))v243)(
                         v149,
                         &v296,
                         *v225,
                         *((_QWORD *)v225 + 1),
                         v231);
              if ( v151 == (_QWORD *)v296 )
              {
                if ( v297 )
                {
                  for ( k = (_QWORD *)(v297 + 24LL * *(_QWORD *)(v297 - 8)); (_QWORD *)v297 != k; sub_91D830(k) )
                    k -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(k - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v296);
              }
              if ( v151 == v294 )
              {
                if ( v295 )
                {
                  for ( m = &v295[3 * *(v295 - 1)]; v295 != m; sub_91D830(m) )
                    m -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(m - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v294);
              }
              if ( !v257 )
              {
                v9 = v219;
                v117 += 5;
                v130 = v209;
                goto LABEL_307;
              }
            }
            v152 = v117[10];
            if ( *(_DWORD *)(v152 + 24) != 51 )
            {
              v153 = v293;
              v226 = v291;
              v258 = *(__int64 (**)())(*(_QWORD *)v293 + 616LL);
              v232 = *v292;
              v154 = (__int64 *)(*(_QWORD *)(v152 + 96) + 24LL);
              v155 = sub_C33340();
              if ( (_QWORD *)*v154 == v155 )
                sub_C3C790(&v294, (_QWORD **)v154);
              else
                sub_C33EB0(&v294, v154);
              if ( v294 == v155 )
                sub_C3CCB0((__int64)&v294);
              else
                sub_C34440((unsigned __int8 *)&v294);
              if ( v294 == v155 )
                sub_C3C840(&v296, &v294);
              else
                sub_C338E0((__int64)&v296, (__int64)&v294);
              v244 = 0;
              if ( v258 != sub_2FE3170 )
                v244 = ((__int64 (__fastcall *)(__int64, _BYTE **, _QWORD, _QWORD, _QWORD))v258)(
                         v153,
                         &v296,
                         *v226,
                         *((_QWORD *)v226 + 1),
                         v232);
              if ( v155 == (_QWORD *)v296 )
              {
                if ( v297 )
                {
                  for ( n = (_QWORD *)(v297 + 24LL * *(_QWORD *)(v297 - 8)); (_QWORD *)v297 != n; sub_91D830(n) )
                    n -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(n - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v296);
              }
              if ( v155 == v294 )
              {
                if ( v295 )
                {
                  for ( ii = &v295[3 * *(v295 - 1)]; v295 != ii; sub_91D830(ii) )
                    ii -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(ii - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v294);
              }
              if ( !v244 )
              {
                v9 = v219;
                v117 += 10;
                v130 = v209;
                goto LABEL_307;
              }
            }
            v156 = v117[15];
            if ( *(_DWORD *)(v156 + 24) != 51 )
            {
              v157 = v293;
              v245 = *(__int64 (**)())(*(_QWORD *)v293 + 616LL);
              v233 = *v292;
              v227 = v291;
              v158 = (__int64 *)(*(_QWORD *)(v156 + 96) + 24LL);
              v159 = sub_C33340();
              if ( (_QWORD *)*v158 == v159 )
                sub_C3C790(&v294, (_QWORD **)v158);
              else
                sub_C33EB0(&v294, v158);
              if ( v294 == v159 )
                sub_C3CCB0((__int64)&v294);
              else
                sub_C34440((unsigned __int8 *)&v294);
              if ( v294 == v159 )
                sub_C3C840(&v296, &v294);
              else
                sub_C338E0((__int64)&v296, (__int64)&v294);
              v259 = 0;
              if ( v245 != sub_2FE3170 )
                v259 = ((__int64 (__fastcall *)(__int64, _BYTE **, _QWORD, _QWORD, _QWORD))v245)(
                         v157,
                         &v296,
                         *v227,
                         *((_QWORD *)v227 + 1),
                         v233);
              if ( v159 == (_QWORD *)v296 )
              {
                if ( v297 )
                {
                  for ( jj = (_QWORD *)(v297 + 24LL * *(_QWORD *)(v297 - 8)); (_QWORD *)v297 != jj; sub_91D830(jj) )
                    jj -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(jj - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v296);
              }
              if ( v159 == v294 )
              {
                if ( v295 )
                {
                  for ( kk = &v295[3 * *(v295 - 1)]; v295 != kk; sub_91D830(kk) )
                    kk -= 3;
                  j_j_j___libc_free_0_0((unsigned __int64)(kk - 1));
                }
              }
              else
              {
                sub_C338F0((__int64)&v294);
              }
              if ( !v259 )
              {
                v9 = v219;
                v117 += 15;
                v130 = v209;
                goto LABEL_307;
              }
            }
            v117 += 20;
          }
          while ( v269 != v117 );
          v9 = v219;
          v130 = v209;
        }
        v190 = (char *)v276 - (char *)v117;
        if ( (char *)v276 - (char *)v117 != 80 )
        {
          if ( v190 != 120 )
          {
            if ( v190 != 40 )
              goto LABEL_309;
            goto LABEL_334;
          }
          v271 = v130;
          v195 = sub_3445D50(&v291, *v117);
          v130 = v271;
          if ( !v195 )
          {
LABEL_307:
            if ( v276 == v117 )
            {
              v117 = *(__int64 **)(v9 + 40);
              v268 = &v117[5 * *(unsigned int *)(v9 + 64)];
              goto LABEL_151;
            }
            if ( v130 )
              goto LABEL_59;
LABEL_309:
            v117 = *(__int64 **)(v9 + 40);
            v268 = &v117[5 * *(unsigned int *)(v9 + 64)];
LABEL_151:
            v296 = v298;
            v297 = 0x400000000LL;
            if ( v268 != v117 )
            {
              v132 = v117;
              v242 = (__int64)v14;
              HIWORD(v133) = v205;
              do
              {
                v140 = *v132;
                v141 = *v132;
                if ( *(_DWORD *)(*v132 + 24) == 51 )
                {
                  sub_3050D50((__int64)&v296, v140, v132[1], v127, v129, v128);
                }
                else
                {
                  v254 = *(_QWORD *)(v140 + 96);
                  v277 = *((_DWORD *)v132 + 2);
                  v142 = sub_C33340();
                  v143 = v254;
                  v255 = v142;
                  v144 = (__int64 *)(v143 + 24);
                  if ( *(_QWORD **)(v143 + 24) == v142 )
                    sub_C3C790(&v294, (_QWORD **)v144);
                  else
                    sub_C33EB0(&v294, v144);
                  if ( v294 == v255 )
                    sub_C3CCB0((__int64)&v294);
                  else
                    sub_C34440((unsigned __int8 *)&v294);
                  v134 = *(_QWORD *)(v141 + 48) + 16LL * v277;
                  LOWORD(v133) = *(_WORD *)v134;
                  v135 = sub_33FE6E0(v242, (__int64 *)&v294, (__int64)&v286, v133, *(_QWORD *)(v134 + 8), 0, a7);
                  sub_3050D50((__int64)&v296, v135, v136, v137, v138, v139);
                  sub_91D830(&v294);
                }
                v132 += 5;
              }
              while ( v268 != v132 );
              v14 = (__m128i *)v242;
            }
            v180 = (unsigned int)v297;
            v181 = v284;
            v182 = v285;
            *a8 = 1;
            *((_QWORD *)&v202 + 1) = v180;
            *(_QWORD *)&v202 = v296;
            v11 = (__int64)sub_33FC220(v14, 156, (__int64)&v286, v181, v182, v128, v202);
            if ( v296 != v298 )
              _libc_free((unsigned __int64)v296);
            goto LABEL_43;
          }
          v117 += 5;
        }
        v272 = v130;
        v196 = sub_3445D50(&v291, *v117);
        v130 = v272;
        if ( v196 )
        {
          v117 += 5;
LABEL_334:
          v270 = v130;
          v191 = sub_3445D50(&v291, *v117);
          v130 = v270;
          if ( v191 )
            goto LABEL_309;
          goto LABEL_307;
        }
        goto LABEL_307;
      }
      v197 = *(_DWORD *)(*v276 + 24);
      if ( v197 > 0x33 || ((0x8001000001000uLL >> v197) & 1) == 0 )
        goto LABEL_148;
      v276 += 5;
    }
    v198 = *(_DWORD *)(*v276 + 24);
    if ( v198 > 0x33 || ((0x8001000001000uLL >> v198) & 1) == 0 )
      goto LABEL_148;
    v276 += 5;
    goto LABEL_362;
  }
  if ( v10 > 149 )
  {
    if ( (*(_BYTE *)(v13 + 864) & 0x10) == 0 && (v273 & 0x80u) == 0 )
      goto LABEL_67;
    v89 = *(_QWORD *)(v9 + 40);
    v90 = v15;
    v91 = _mm_loadu_si128((const __m128i *)v89);
    v92 = _mm_loadu_si128((const __m128i *)(v89 + 40));
    v93 = *(_QWORD *)(v89 + 80);
    v94 = *(_QWORD *)(v89 + 88);
    LODWORD(v291) = 2;
    v230 = v91;
    v240 = (__int128)v92;
    *(_QWORD *)&v95 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __m128i *, _QWORD, _QWORD, unsigned int **, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                        a1,
                        v93,
                        v94,
                        v14,
                        v15,
                        v280,
                        &v291,
                        v262);
    v252 = v95;
    if ( !(_QWORD)v95 )
      goto LABEL_59;
    v217 = v95;
    v96 = sub_22077B0(0x98u);
    *(_QWORD *)(v96 + 64) = sub_33ECD10(1u);
    *(_QWORD *)(v96 + 128) = v96 + 16;
    *(_QWORD *)(v96 + 80) = 0x100000000LL;
    *(_QWORD *)(v96 + 104) = 0xFFFFFFFFLL;
    v97 = v96 + 112;
    *(_QWORD *)(v96 + 144) = 0;
    *(_QWORD *)(v96 + 16) = 0;
    *(_QWORD *)(v96 + 24) = 0;
    *(_QWORD *)(v96 + 32) = 0;
    *(_QWORD *)(v96 + 40) = 328;
    *(_WORD *)(v96 + 50) = -1;
    *(_DWORD *)(v96 + 52) = -1;
    *(_QWORD *)(v96 + 56) = 0;
    *(_QWORD *)(v96 + 72) = 0;
    *(_DWORD *)(v96 + 88) = 0;
    *(_QWORD *)(v96 + 96) = 0;
    *(_WORD *)(v96 + 48) = 0;
    *(_QWORD *)(v96 + 136) = 0;
    *(_QWORD *)(v96 + 112) = v252;
    v98 = *(_QWORD *)(v217 + 56);
    *(_DWORD *)(v96 + 120) = DWORD2(v252);
    *(_QWORD *)(v96 + 144) = v98;
    if ( v98 )
      *(_QWORD *)(v98 + 24) = v96 + 144;
    *(_QWORD *)(v96 + 136) = v217 + 56;
    *(_QWORD *)(v217 + 56) = v97;
    *(_QWORD *)(v96 + 56) = v97;
    *(_DWORD *)(v96 + 80) = 1;
    sub_2208C80((_QWORD *)v96, (__int64)&v288);
    v99 = *(_QWORD *)a1;
    LODWORD(v294) = 2;
    ++v290;
    v212 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __m128i *, _QWORD, _QWORD, _QWORD **, _QWORD))(v99 + 2264))(
             a1,
             v230.m128i_i64[0],
             v230.m128i_i64[1],
             v14,
             v90,
             v280,
             &v294,
             v262);
    v213 = v100;
    v218 = v100;
    if ( v212 )
    {
      v223 = sub_22077B0(0x98u);
      *(_QWORD *)(v223 + 64) = sub_33ECD10(1u);
      *(_QWORD *)(v223 + 80) = 0x100000000LL;
      *(_QWORD *)(v223 + 128) = v223 + 16;
      *(_QWORD *)(v223 + 104) = 0xFFFFFFFFLL;
      *(_WORD *)(v223 + 50) = -1;
      v101 = v223 + 112;
      *(_WORD *)(v223 + 48) = 0;
      *(_QWORD *)(v223 + 144) = 0;
      *(_QWORD *)(v223 + 16) = 0;
      *(_QWORD *)(v223 + 24) = 0;
      *(_QWORD *)(v223 + 32) = 0;
      *(_QWORD *)(v223 + 40) = 328;
      *(_DWORD *)(v223 + 52) = -1;
      *(_QWORD *)(v223 + 56) = 0;
      *(_QWORD *)(v223 + 72) = 0;
      *(_DWORD *)(v223 + 88) = 0;
      *(_QWORD *)(v223 + 96) = 0;
      *(_QWORD *)(v223 + 136) = 0;
      *(_DWORD *)(v223 + 120) = v218;
      v102 = *(_QWORD *)(v212 + 56);
      *(_QWORD *)(v223 + 112) = v212;
      *(_QWORD *)(v223 + 144) = v102;
      if ( v102 )
        *(_QWORD *)(v102 + 24) = v223 + 144;
      *(_QWORD *)(v223 + 56) = v101;
      *(_QWORD *)(v223 + 136) = v212 + 56;
      *(_QWORD *)(v212 + 56) = v101;
      *(_DWORD *)(v223 + 80) = 1;
      sub_2208C80((_QWORD *)v223, (__int64)&v288);
      v103 = *(_QWORD *)a1;
      LODWORD(v296) = 2;
      ++v290;
      *(_QWORD *)&v104 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, _BYTE **, _QWORD))(v103 + 2264))(
                           a1,
                           v240,
                           *((_QWORD *)&v240 + 1),
                           v14,
                           v90,
                           v280,
                           &v296,
                           v262);
      v105 = v288;
      v208 = v104;
      v264 = v104;
      v224 = DWORD2(v104);
      if ( v288 == &v288 )
      {
        v290 = 0;
        v289 = &v288;
LABEL_126:
        v107 = (int)v294;
        if ( (int)v294 <= (int)v296 )
        {
          if ( (int)v291 <= (int)v294 )
            v107 = (int)v291;
          v108 = v284;
          v109 = v285;
          *a8 = v107;
          v11 = sub_340EC60(v14, v10, (__int64)&v286, v108, v109, v273, v212, v213, v240, v252);
          if ( v264 == v11 && v224 == v110 )
            goto LABEL_43;
          goto LABEL_130;
        }
LABEL_280:
        if ( v264 )
        {
          v176 = (int)v296;
          if ( (int)v291 <= (int)v296 )
            v176 = (int)v291;
          v177 = v284;
          *a8 = v176;
          v179 = sub_340EC60(
                   v14,
                   v10,
                   (__int64)&v286,
                   v177,
                   v285,
                   v273,
                   v230.m128i_i64[0],
                   v230.m128i_i64[1],
                   v208,
                   v252);
          if ( (v212 != v179 || v218 != v178) && v212 && !*(_QWORD *)(v212 + 56) )
            sub_33ECEA0(v14, v212);
          v11 = v179;
          goto LABEL_43;
        }
        goto LABEL_59;
      }
      v204 = v10;
    }
    else
    {
      LODWORD(v296) = 2;
      *(_QWORD *)&v184 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                           a1,
                           v240,
                           *((_QWORD *)&v240 + 1),
                           v14,
                           v90,
                           v280,
                           &v296,
                           v262);
      v105 = v288;
      v208 = v184;
      v264 = v184;
      v224 = DWORD2(v184);
      if ( v288 == &v288 )
      {
        v290 = 0;
        v289 = &v288;
        goto LABEL_280;
      }
      v204 = v10;
    }
    do
    {
      v106 = (unsigned __int64)v105;
      v105 = (_QWORD *)*v105;
      sub_33CF710(v106 + 16);
      j_j___libc_free_0(v106);
    }
    while ( v105 != &v288 );
    v10 = v204;
    v290 = 0;
    v289 = &v288;
    v288 = &v288;
    if ( !v212 )
      goto LABEL_280;
    goto LABEL_126;
  }
  if ( v10 == 97 )
  {
    if ( (*(_BYTE *)(v13 + 864) & 0x10) != 0 || (v273 & 0x80u) != 0 )
    {
      v83 = *(_QWORD *)(v9 + 40);
      v84 = *(_QWORD *)v83;
      v85 = *(_QWORD *)(v83 + 8);
      v266 = (__int128)_mm_loadu_si128((const __m128i *)(v83 + 40));
      v86 = sub_33E1790(*(_QWORD *)v83, v85, 1u, v18, a5, v13);
      if ( v86
        && ((v87 = *(_QWORD *)(v86 + 96), *(void **)(v87 + 24) == sub_C33340())
          ? (v88 = *(_QWORD *)(v87 + 32))
          : (v88 = v87 + 24),
            (*(_BYTE *)(v88 + 20) & 7) == 3) )
      {
        *a8 = 0;
        v11 = v266;
      }
      else
      {
        v146 = (unsigned int)v284;
        v147 = v285;
        *a8 = 1;
        *((_QWORD *)&v201 + 1) = v85;
        *(_QWORD *)&v201 = v84;
        v11 = (__int64)sub_3405C90(v14, 0x61u, (__int64)&v286, v146, v147, v273, a7, v266, v201);
      }
      goto LABEL_43;
    }
    goto LABEL_67;
  }
  if ( v10 <= 97 )
  {
    if ( v10 == 12 )
    {
      v267 = v15;
      v112 = sub_328D6E0(a1, 0xCu, v284);
      v113 = sub_C33340();
      v275 = v113;
      if ( v112 )
        goto LABEL_135;
      v43 = (__int64 *)(*(_QWORD *)(v9 + 96) + 24LL);
      v248 = *(__int64 (**)())(*(_QWORD *)a1 + 616LL);
      if ( v113 == (_QWORD *)*v43 )
        sub_C3C790(&v294, (_QWORD **)v43);
      else
        sub_C33EB0(&v294, v43);
      if ( v275 == v294 )
        sub_C3CCB0((__int64)&v294);
      else
        sub_C34440((unsigned __int8 *)&v294);
      if ( v275 == v294 )
        sub_C3C840(&v296, &v294);
      else
        sub_C338E0((__int64)&v296, (__int64)&v294);
      if ( v248 != sub_2FE3170 )
        v16 = v267
            & (((__int64 (__fastcall *)(__int64, _BYTE **, _QWORD, __int64, _QWORD))v248)(
                 a1,
                 &v296,
                 (unsigned int)v284,
                 v285,
                 v280)
             ^ 1);
      sub_91D830(&v296);
      sub_91D830(&v294);
      if ( !v16 )
      {
LABEL_135:
        v114 = (__int64 *)(*(_QWORD *)(v9 + 96) + 24LL);
        if ( v275 == (_QWORD *)*v114 )
          sub_C3C790(&v296, (_QWORD **)v114);
        else
          sub_C33EB0(&v296, v114);
        if ( v275 == (_QWORD *)v296 )
          sub_C3CCB0((__int64)&v296);
        else
          sub_C34440((unsigned __int8 *)&v296);
        v11 = sub_33FE6E0((__int64)v14, (__int64 *)&v296, (__int64)&v286, v284, v285, 0, a7);
        v116 = v115;
        if ( (unsigned __int8)sub_3286E00(&v281) || (unsigned __int8)sub_33CF8A0(v11, v116) )
        {
          *a8 = 1;
          sub_91D830(&v296);
          v41 = (_QWORD **)v288;
          goto LABEL_61;
        }
        sub_91D830(&v296);
      }
      goto LABEL_59;
    }
    if ( v10 != 96 )
    {
      v41 = &v288;
      goto LABEL_60;
    }
    if ( (*(_BYTE *)(v13 + 864) & 0x10) != 0 || (v273 & 0x80u) != 0 )
    {
      if ( !v15 || (v249 = v15, v50 = sub_328A020(a1, 0x61u, v284, v285, 0), v15 = v249, v50) )
      {
        v51 = *(_QWORD *)(v9 + 40);
        v52 = v15;
        v53 = (__int128)_mm_loadu_si128((const __m128i *)v51);
        v54 = (__int128)_mm_loadu_si128((const __m128i *)(v51 + 40));
        LODWORD(v294) = 2;
        *(_QWORD *)&v55 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, _QWORD **, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                            a1,
                            v53,
                            *((_QWORD *)&v53 + 1),
                            v14,
                            v15,
                            v280,
                            &v294,
                            v262);
        v229 = v55;
        v56 = v55;
        v57 = DWORD2(v55);
        if ( (_QWORD)v55 )
        {
          v221 = sub_22077B0(0x98u);
          *(_QWORD *)(v221 + 64) = sub_33ECD10(1u);
          v58 = v221 + 112;
          *(_QWORD *)(v221 + 80) = 0x100000000LL;
          *(_QWORD *)(v221 + 104) = 0xFFFFFFFFLL;
          *(_QWORD *)(v221 + 128) = v221 + 16;
          *(_QWORD *)(v221 + 144) = 0;
          *(_QWORD *)(v221 + 16) = 0;
          *(_QWORD *)(v221 + 24) = 0;
          *(_QWORD *)(v221 + 32) = 0;
          *(_QWORD *)(v221 + 40) = 328;
          *(_WORD *)(v221 + 50) = -1;
          *(_DWORD *)(v221 + 52) = -1;
          *(_QWORD *)(v221 + 56) = 0;
          *(_QWORD *)(v221 + 72) = 0;
          *(_DWORD *)(v221 + 88) = 0;
          *(_QWORD *)(v221 + 96) = 0;
          *(_WORD *)(v221 + 48) = 0;
          *(_QWORD *)(v221 + 136) = 0;
          *(_DWORD *)(v221 + 120) = v57;
          v59 = *(_QWORD *)(v56 + 56);
          *(_QWORD *)(v221 + 112) = v229;
          *(_QWORD *)(v221 + 144) = v59;
          if ( v59 )
            *(_QWORD *)(v59 + 24) = v221 + 144;
          *(_QWORD *)(v221 + 56) = v58;
          *(_QWORD *)(v221 + 136) = v56 + 56;
          *(_QWORD *)(v56 + 56) = v58;
          *(_DWORD *)(v221 + 80) = 1;
          sub_2208C80((_QWORD *)v221, (__int64)&v288);
          v60 = *(_QWORD *)a1;
          LODWORD(v296) = 2;
          ++v290;
          *(_QWORD *)&v61 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, _BYTE **, _QWORD))(v60 + 2264))(
                              a1,
                              v54,
                              *((_QWORD *)&v54 + 1),
                              v14,
                              v52,
                              v280,
                              &v296,
                              v262);
          v62 = v288;
          v211 = v61;
          v264 = v61;
          v216 = DWORD2(v61);
          if ( v288 == &v288 )
          {
            v290 = 0;
            v289 = &v288;
LABEL_84:
            if ( (int)v294 > (int)v296 )
              goto LABEL_85;
            v187 = (unsigned int)v284;
            v188 = v285;
            *a8 = (int)v294;
            v11 = (__int64)sub_3405C90(v14, 0x61u, (__int64)&v286, v187, v188, v273, a7, v229, v54);
            if ( v264 == v11 && v216 == v189 )
              goto LABEL_43;
LABEL_130:
            v111 = v264;
            if ( !v264 )
              goto LABEL_43;
LABEL_131:
            if ( !*(_QWORD *)(v111 + 56) )
              sub_33ECEA0(v14, v111);
            goto LABEL_43;
          }
          v222 = v56;
        }
        else
        {
          LODWORD(v296) = 2;
          *(_QWORD *)&v175 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                               a1,
                               v54,
                               *((_QWORD *)&v54 + 1),
                               v14,
                               v52,
                               v280,
                               &v296,
                               v262);
          v62 = v288;
          v211 = v175;
          v264 = v175;
          v216 = DWORD2(v175);
          if ( v288 == &v288 )
          {
            v290 = 0;
            v289 = &v288;
            goto LABEL_85;
          }
          v222 = v56;
        }
        do
        {
          v63 = (unsigned __int64)v62;
          v62 = (_QWORD *)*v62;
          sub_33CF710(v63 + 16);
          j_j___libc_free_0(v63);
        }
        while ( v62 != &v288 );
        v56 = v222;
        v290 = 0;
        v289 = &v288;
        v288 = &v288;
        if ( v222 )
          goto LABEL_84;
LABEL_85:
        if ( v264 )
        {
          v64 = (unsigned int)v284;
          *a8 = (int)v296;
          v11 = (__int64)sub_3405C90(v14, 0x61u, (__int64)&v286, v64, v285, v273, a7, v211, v53);
          if ( (v56 != v11 || v57 != v65) && v56 && !*(_QWORD *)(v56 + 56) )
            sub_33ECEA0(v14, v56);
          goto LABEL_43;
        }
        goto LABEL_59;
      }
    }
LABEL_67:
    v41 = &v288;
    goto LABEL_60;
  }
  if ( (unsigned int)(v10 - 98) > 1 )
    goto LABEL_67;
  v23 = *(_QWORD *)(v9 + 40);
  v235 = v15;
  v24 = (__int128)_mm_loadu_si128((const __m128i *)v23);
  v25 = (__int128)_mm_loadu_si128((const __m128i *)(v23 + 40));
  v283 = 2;
  *(_QWORD *)&v26 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, int *, _QWORD))(*(_QWORD *)a1 + 2264LL))(
                      a1,
                      v24,
                      *((_QWORD *)&v24 + 1),
                      v14,
                      v15,
                      v280,
                      &v283,
                      v262);
  v27 = DWORD2(v26);
  v214 = v26;
  v210 = DWORD2(v26);
  v247 = v26;
  if ( (_QWORD)v26 )
  {
    v206 = v235;
    v236 = sub_22077B0(0x98u);
    *(_QWORD *)(v236 + 64) = sub_33ECD10(1u);
    *(_QWORD *)(v236 + 80) = 0x100000000LL;
    *(_QWORD *)(v236 + 128) = v236 + 16;
    *(_QWORD *)(v236 + 104) = 0xFFFFFFFFLL;
    *(_WORD *)(v236 + 50) = -1;
    v28 = v236 + 112;
    *(_WORD *)(v236 + 48) = 0;
    *(_QWORD *)(v236 + 144) = 0;
    *(_QWORD *)(v236 + 16) = 0;
    *(_QWORD *)(v236 + 24) = 0;
    *(_QWORD *)(v236 + 32) = 0;
    *(_QWORD *)(v236 + 40) = 328;
    *(_DWORD *)(v236 + 52) = -1;
    *(_QWORD *)(v236 + 56) = 0;
    *(_QWORD *)(v236 + 72) = 0;
    *(_DWORD *)(v236 + 88) = 0;
    *(_QWORD *)(v236 + 96) = 0;
    *(_QWORD *)(v236 + 136) = 0;
    v29 = *(_QWORD *)(v247 + 56);
    *(_QWORD *)(v236 + 112) = v214;
    *(_DWORD *)(v236 + 120) = v27;
    *(_QWORD *)(v236 + 144) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 24) = v236 + 144;
    *(_QWORD *)(v236 + 56) = v28;
    *(_QWORD *)(v236 + 136) = v247 + 56;
    *(_QWORD *)(v247 + 56) = v28;
    *(_DWORD *)(v236 + 80) = 1;
    sub_2208C80((_QWORD *)v236, (__int64)&v288);
    v30 = *(_QWORD *)a1;
    LODWORD(v291) = 2;
    ++v290;
    *(_QWORD *)&v31 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, unsigned int **, _QWORD))(v30 + 2264))(
                        a1,
                        v25,
                        *((_QWORD *)&v25 + 1),
                        v14,
                        v206,
                        v280,
                        &v291,
                        v262);
    v35 = v288;
    v203 = v31;
    v237 = v31;
    v207 = DWORD2(v31);
    if ( v288 == &v288 )
    {
      v290 = 0;
      v289 = &v288;
      goto LABEL_34;
    }
    v263 = v10;
  }
  else
  {
    v66 = *(_QWORD *)a1;
    LODWORD(v291) = 2;
    *(_QWORD *)&v67 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, unsigned int **))(v66 + 2264))(
                        a1,
                        v25,
                        *((_QWORD *)&v25 + 1),
                        v14,
                        v235,
                        v280,
                        &v291);
    v32 = v200;
    v35 = v288;
    v203 = v67;
    v237 = v67;
    v207 = DWORD2(v67);
    if ( v288 == &v288 )
    {
      v290 = 0;
      v289 = &v288;
      goto LABEL_35;
    }
    v263 = v10;
  }
  do
  {
    v36 = (unsigned __int64)v35;
    v35 = (_QWORD *)*v35;
    sub_33CF710(v36 + 16);
    j_j___libc_free_0(v36);
  }
  while ( v35 != &v288 );
  v290 = 0;
  v10 = v263;
  v289 = &v288;
  v288 = &v288;
  if ( !v247 )
    goto LABEL_35;
LABEL_34:
  if ( v283 > (int)v291 )
  {
LABEL_35:
    v37 = sub_33E1790(
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v9 + 40) + 48LL),
            0,
            v32,
            v33,
            v34);
    if ( !v37 )
      goto LABEL_38;
    v220 = *(_QWORD *)(v37 + 96);
    a7 = (__m128i)0x4000000000000000uLL;
    v215 = sub_C33320();
    sub_C3B1B0((__int64)&v296, 2.0);
    sub_C407B0(&v294, (__int64 *)&v296, v215);
    sub_C338F0((__int64)&v296);
    sub_C41640((__int64 *)&v294, *(_DWORD **)(v220 + 24), 1, (bool *)&v296);
    v38 = *(_QWORD **)(v220 + 24);
    if ( v38 != v294 )
    {
      sub_91D830(&v294);
      goto LABEL_38;
    }
    v185 = v220 + 24;
    if ( v38 == sub_C33340() )
      v186 = sub_C3E590(v185, (__int64)&v294);
    else
      v186 = sub_C33D00(v185, (__int64)&v294);
    sub_91D830(&v294);
    if ( !v186 || *(_DWORD *)(v9 + 24) != 98 )
    {
LABEL_38:
      if ( v237 )
      {
        v39 = (unsigned int)v284;
        *a8 = (int)v291;
        v11 = (__int64)sub_3405C90(v14, v10, (__int64)&v286, v39, v285, v273, a7, v24, v203);
        if ( (v247 != v11 || v210 != v40) && v247 && !*(_QWORD *)(v247 + 56) )
          sub_33ECEA0(v14, v247);
        goto LABEL_43;
      }
    }
LABEL_59:
    v41 = (_QWORD **)v288;
LABEL_60:
    v11 = 0;
    goto LABEL_61;
  }
  v172 = (unsigned int)v284;
  v173 = v285;
  *a8 = v283;
  v11 = (__int64)sub_3405C90(v14, v10, (__int64)&v286, v172, v173, v273, a7, v214, v25);
  if ( v237 != v11 || v207 != v174 )
  {
    v111 = v237;
    if ( v237 )
      goto LABEL_131;
  }
LABEL_43:
  v41 = (_QWORD **)v288;
LABEL_61:
  while ( v41 != &v288 )
  {
    v44 = (unsigned __int64)v41;
    v41 = (_QWORD **)*v41;
    sub_33CF710(v44 + 16);
    j_j___libc_free_0(v44);
  }
  if ( v286 )
    sub_B91220((__int64)&v286, v286);
  return v11;
}
