// Function: sub_1162F40
// Address: 0x1162f40
//
unsigned __int8 *__fastcall sub_1162F40(__m128i *a1, unsigned __int8 *a2)
{
  __int64 v2; // r12
  const __m128i *v3; // rbx
  __int64 v4; // rax
  __m128i v5; // xmm2
  unsigned __int64 v6; // xmm3_8
  __m128i v7; // xmm4
  __int64 v8; // r13
  __int64 v9; // rax
  char v10; // al
  unsigned __int8 *v11; // rax
  unsigned __int8 *v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _BYTE *v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned __int8 *v23; // rax
  _BYTE *v24; // rsi
  __int64 v25; // rax
  bool v26; // zf
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rcx
  __int64 *v30; // r15
  __int64 *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r10
  __int64 v36; // r15
  __int64 v37; // rsi
  __int64 v38; // rdx
  _BYTE *v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // r15
  __int64 v42; // r9
  __int64 v43; // r10
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 *v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // r14
  const __m128i *v56; // r12
  __int64 v57; // rbx
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 *v61; // rax
  __int64 v62; // r10
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // r13
  unsigned int v68; // eax
  unsigned int v69; // edx
  __int64 v70; // rsi
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  int v75; // eax
  __int64 v76; // rax
  __int64 v77; // rax
  unsigned int **v78; // r13
  __int64 v79; // rax
  __int64 v80; // r9
  unsigned int **v81; // r13
  __int64 v82; // r9
  __int64 v83; // rdx
  __int64 v84; // r13
  __int64 v85; // r15
  unsigned __int8 *v86; // rax
  __int64 v87; // rcx
  unsigned int v88; // esi
  __int64 v89; // r8
  __int64 v90; // rdx
  __int64 v91; // rdi
  __int64 v92; // rax
  __int64 v93; // rax
  _BYTE *v94; // rax
  unsigned __int8 *v95; // rax
  __int64 v96; // r9
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rdx
  unsigned __int8 *v103; // rax
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r8
  __int64 v107; // r10
  __int64 v108; // r9
  __int64 v109; // rax
  unsigned __int64 v110; // rdx
  _BYTE *v111; // rax
  _BYTE *v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v116; // rsi
  const __m128i *v117; // r14
  __int64 v118; // rbx
  __int64 v119; // rdx
  unsigned int v120; // esi
  _BYTE *v121; // r13
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // r13
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // r9
  unsigned __int8 *v128; // rax
  unsigned int **v129; // r13
  unsigned __int8 *v130; // rax
  __int64 v131; // rax
  __int64 v132; // r10
  __int64 v133; // r13
  __int64 *v134; // rax
  __int64 *v135; // rdx
  __int64 *v136; // rax
  __int64 *v137; // r15
  __int64 v138; // r15
  __int64 v139; // rdi
  __int64 v140; // r15
  unsigned __int8 *v141; // rax
  __int64 *v142; // rcx
  __int64 *v143; // rax
  __int64 *v144; // rdx
  __int64 v145; // rax
  __int64 *v146; // r15
  __int64 *v147; // rax
  int v148; // eax
  __int64 v149; // rdx
  __int64 v150; // rcx
  __int64 *v151; // rax
  __int64 v152; // r8
  _QWORD *v153; // r15
  __int64 *v154; // rdx
  __int64 *v155; // rax
  __int64 *v156; // rcx
  __int64 v157; // rax
  __int64 *v158; // r9
  __int64 *v159; // rax
  int v160; // eax
  __int64 v161; // rcx
  __int64 v162; // rdx
  __int64 *v163; // r9
  __int64 *v164; // rax
  __int64 v165; // r8
  __int64 v166; // r11
  __int64 v167; // r13
  __int64 *v168; // r12
  __int64 *v169; // r14
  __int64 v170; // rsi
  __int64 *v171; // rax
  __int64 *v172; // r13
  __int64 v173; // r15
  __int64 *v174; // r12
  __int64 v175; // r14
  __int64 v176; // rsi
  __int64 *v177; // rax
  char v178; // al
  __int64 v179; // r9
  int v180; // r15d
  __int64 v181; // r11
  __int64 v182; // rdx
  __int64 v183; // r15
  const __m128i *v184; // r14
  __int64 v185; // rbx
  __int64 v186; // rdx
  unsigned int v187; // esi
  int v188; // eax
  __int64 v189; // rax
  __int64 v190; // rdx
  __int64 v191; // r15
  __int64 v192; // r14
  const __m128i *v193; // r12
  __int64 v194; // rbx
  __int64 v195; // rdx
  unsigned int v196; // esi
  __int64 *v197; // rax
  int v198; // eax
  __int64 v199; // rax
  __int64 v200; // r9
  __int64 v201; // rdx
  __int64 v202; // r14
  const __m128i *v203; // r12
  __int64 v204; // r15
  __int64 v205; // rbx
  __int64 v206; // rdx
  unsigned int v207; // esi
  int v208; // eax
  __int64 v209; // rax
  __int64 v210; // r11
  __int64 v211; // rdx
  __int64 v212; // r15
  __int64 v213; // rdx
  unsigned int v214; // esi
  __int64 v215; // r14
  const __m128i *v216; // r12
  __int64 *v217; // rbx
  _BYTE *v218; // [rsp-8h] [rbp-208h]
  _BYTE *v219; // [rsp+0h] [rbp-200h]
  __int64 v220; // [rsp+10h] [rbp-1F0h]
  __int64 *v221; // [rsp+10h] [rbp-1F0h]
  __int64 v222; // [rsp+10h] [rbp-1F0h]
  __int64 v223; // [rsp+18h] [rbp-1E8h]
  __int64 v224; // [rsp+18h] [rbp-1E8h]
  __int64 v225; // [rsp+18h] [rbp-1E8h]
  __int64 *v226; // [rsp+18h] [rbp-1E8h]
  __int64 v227; // [rsp+18h] [rbp-1E8h]
  __int64 v228; // [rsp+18h] [rbp-1E8h]
  __int64 v229; // [rsp+28h] [rbp-1D8h]
  int v230; // [rsp+28h] [rbp-1D8h]
  __int64 v231; // [rsp+28h] [rbp-1D8h]
  __int64 v232; // [rsp+28h] [rbp-1D8h]
  __int64 *v233; // [rsp+28h] [rbp-1D8h]
  int v234; // [rsp+28h] [rbp-1D8h]
  __int64 v235; // [rsp+28h] [rbp-1D8h]
  __int64 v236; // [rsp+30h] [rbp-1D0h]
  __int64 v237; // [rsp+30h] [rbp-1D0h]
  __int16 v238; // [rsp+30h] [rbp-1D0h]
  __int64 v239; // [rsp+30h] [rbp-1D0h]
  __int64 v240; // [rsp+30h] [rbp-1D0h]
  __int64 v241; // [rsp+30h] [rbp-1D0h]
  __int64 v242; // [rsp+30h] [rbp-1D0h]
  int v243; // [rsp+30h] [rbp-1D0h]
  __int64 v244; // [rsp+38h] [rbp-1C8h]
  unsigned int v245; // [rsp+38h] [rbp-1C8h]
  char v246; // [rsp+38h] [rbp-1C8h]
  __int64 v247; // [rsp+38h] [rbp-1C8h]
  __int64 v248; // [rsp+38h] [rbp-1C8h]
  int v249; // [rsp+38h] [rbp-1C8h]
  __int64 v250; // [rsp+38h] [rbp-1C8h]
  __int64 v251; // [rsp+38h] [rbp-1C8h]
  _BYTE *v252; // [rsp+38h] [rbp-1C8h]
  __int64 v253; // [rsp+38h] [rbp-1C8h]
  __int64 v254; // [rsp+38h] [rbp-1C8h]
  int v255; // [rsp+38h] [rbp-1C8h]
  int v256; // [rsp+38h] [rbp-1C8h]
  int v257; // [rsp+38h] [rbp-1C8h]
  __int64 v258; // [rsp+40h] [rbp-1C0h]
  __int64 *v259; // [rsp+48h] [rbp-1B8h]
  __int64 v260; // [rsp+48h] [rbp-1B8h]
  bool v261; // [rsp+48h] [rbp-1B8h]
  _BYTE *v262; // [rsp+48h] [rbp-1B8h]
  __int64 v263; // [rsp+48h] [rbp-1B8h]
  __int64 v264; // [rsp+48h] [rbp-1B8h]
  __int64 v265; // [rsp+48h] [rbp-1B8h]
  __int64 v266; // [rsp+48h] [rbp-1B8h]
  __int64 v267; // [rsp+48h] [rbp-1B8h]
  __int64 v268; // [rsp+48h] [rbp-1B8h]
  __int64 v269; // [rsp+48h] [rbp-1B8h]
  __int64 v270; // [rsp+48h] [rbp-1B8h]
  __int64 v271; // [rsp+48h] [rbp-1B8h]
  __int64 v272; // [rsp+48h] [rbp-1B8h]
  __int64 v273; // [rsp+50h] [rbp-1B0h]
  __int64 v274; // [rsp+50h] [rbp-1B0h]
  __int64 v275; // [rsp+58h] [rbp-1A8h]
  __int64 *v276; // [rsp+58h] [rbp-1A8h]
  __int64 v277; // [rsp+58h] [rbp-1A8h]
  unsigned int **v278; // [rsp+58h] [rbp-1A8h]
  __int64 v279; // [rsp+58h] [rbp-1A8h]
  __int64 v280; // [rsp+58h] [rbp-1A8h]
  __int64 v281; // [rsp+58h] [rbp-1A8h]
  __int64 v282; // [rsp+58h] [rbp-1A8h]
  __int64 v283; // [rsp+68h] [rbp-198h] BYREF
  _BYTE *v284; // [rsp+70h] [rbp-190h] BYREF
  _BYTE *v285; // [rsp+78h] [rbp-188h] BYREF
  _BYTE *v286[4]; // [rsp+80h] [rbp-180h] BYREF
  __int16 v287; // [rsp+A0h] [rbp-160h]
  __int64 v288; // [rsp+B0h] [rbp-150h] BYREF
  unsigned int v289; // [rsp+B8h] [rbp-148h]
  _BYTE **v290; // [rsp+C0h] [rbp-140h]
  __int64 *v291; // [rsp+C8h] [rbp-138h]
  __int16 v292; // [rsp+D0h] [rbp-130h]
  __int64 v293; // [rsp+E0h] [rbp-120h] BYREF
  __int64 *v294; // [rsp+E8h] [rbp-118h]
  __int64 v295; // [rsp+F0h] [rbp-110h]
  int v296; // [rsp+F8h] [rbp-108h]
  char v297; // [rsp+FCh] [rbp-104h]
  char v298; // [rsp+100h] [rbp-100h] BYREF
  __int64 v299; // [rsp+110h] [rbp-F0h] BYREF
  __int64 *v300; // [rsp+118h] [rbp-E8h]
  __int64 v301; // [rsp+120h] [rbp-E0h]
  int v302; // [rsp+128h] [rbp-D8h]
  char v303; // [rsp+12Ch] [rbp-D4h]
  char v304; // [rsp+130h] [rbp-D0h] BYREF
  __m128i v305; // [rsp+140h] [rbp-C0h] BYREF
  __m128i v306; // [rsp+150h] [rbp-B0h] BYREF
  _BYTE **v307; // [rsp+160h] [rbp-A0h]
  unsigned __int8 *v308; // [rsp+168h] [rbp-98h]
  __m128i v309; // [rsp+170h] [rbp-90h]
  __int64 v310; // [rsp+180h] [rbp-80h]
  __int64 v311; // [rsp+188h] [rbp-78h]
  void **v312; // [rsp+190h] [rbp-70h]
  void **v313; // [rsp+198h] [rbp-68h]
  __int64 v314; // [rsp+1A0h] [rbp-60h]
  int v315; // [rsp+1A8h] [rbp-58h]
  __int16 v316; // [rsp+1ACh] [rbp-54h]
  char v317; // [rsp+1AEh] [rbp-52h]
  __int64 v318; // [rsp+1B0h] [rbp-50h]
  __int64 v319; // [rsp+1B8h] [rbp-48h]
  void *v320; // [rsp+1C0h] [rbp-40h] BYREF
  void *v321; // [rsp+1C8h] [rbp-38h] BYREF

  v2 = (__int64)a2;
  v3 = a1;
  v4 = sub_B43CA0((__int64)a2);
  v5 = _mm_loadu_si128(a1 + 7);
  v6 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v7 = _mm_loadu_si128(a1 + 9);
  v8 = v4;
  v9 = a1[10].m128i_i64[0];
  v305 = _mm_loadu_si128(a1 + 6);
  v307 = (_BYTE **)v6;
  v310 = v9;
  v308 = a2;
  v306 = v5;
  v309 = v7;
  v10 = sub_B45210((__int64)a2);
  v11 = sub_100A740(*((_BYTE **)a2 - 8), *((_BYTE **)a2 - 4), v10, v305.m128i_i64, 0, 1);
  if ( v11 )
    return sub_F162A0((__int64)a1, (__int64)a2, (__int64)v11);
  v13 = sub_F0F270((__int64)a1, a2);
  if ( v13 )
    return v13;
  v13 = (unsigned __int8 *)sub_F11DB0(a1->m128i_i64, a2);
  if ( v13 )
    return v13;
  v13 = sub_115C220((__int64)a1, (__int64)a2);
  if ( v13 )
    return v13;
  v17 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v17 > 0x15u )
    goto LABEL_13;
  v18 = sub_B43CC0((__int64)a2);
  v19 = *((_QWORD *)a2 - 4);
  v275 = v18;
  v305.m128i_i64[0] = (__int64)&v299;
  if ( (unsigned __int8)sub_995E90(&v305, v19, v20, v21, v22) )
  {
    v37 = sub_96E680(12, (__int64)v17);
    if ( v37 )
    {
      v38 = v299;
      LOWORD(v307) = 257;
      goto LABEL_63;
    }
  }
  if ( sub_B451B0(v2) && sub_B451F0(v2) )
  {
    v23 = *(unsigned __int8 **)(v2 - 32);
    v14 = *v23;
    if ( (_BYTE)v14 == 47 )
    {
      v14 = *((_QWORD *)v23 - 8);
      if ( !v14 )
        goto LABEL_13;
      v299 = *((_QWORD *)v23 - 8);
      v39 = (_BYTE *)*((_QWORD *)v23 - 4);
      if ( *v39 <= 0x15u )
      {
        v40 = (__int64)v17;
        v41 = sub_96E6C0(0x15u, (__int64)v17, v39, v275);
LABEL_68:
        if ( v41 && sub_AD7F90(v41, v40, v14, v15, v16) )
        {
          v38 = v299;
          LOWORD(v307) = 257;
          v37 = v41;
LABEL_63:
          v13 = (unsigned __int8 *)sub_B504D0(21, v37, v38, (__int64)&v305, 0, 0);
          sub_B45260(v13, v2, 1);
          if ( v13 )
            return v13;
          goto LABEL_13;
        }
        goto LABEL_13;
      }
      v14 = *v23;
    }
    if ( (_BYTE)v14 != 50 )
      goto LABEL_13;
    v14 = *((_QWORD *)v23 - 8);
    if ( !v14 )
      goto LABEL_13;
    v299 = *((_QWORD *)v23 - 8);
    v14 = *((_QWORD *)v23 - 4);
    if ( *(_BYTE *)v14 > 0x15u )
      goto LABEL_13;
    v40 = (__int64)v17;
    v41 = sub_96E6C0(0x12u, (__int64)v17, (_BYTE *)v14, v275);
    goto LABEL_68;
  }
LABEL_13:
  v24 = (_BYTE *)v2;
  v13 = sub_115A080((__int64)a1, (unsigned __int8 *)v2, v14, v15, v16);
  if ( v13 )
    return v13;
  v25 = *(_QWORD *)(v2 - 64);
  v26 = *(_BYTE *)v2 == 50;
  v293 = 0;
  v295 = 2;
  v258 = v25;
  v27 = *(_QWORD *)(v2 - 32);
  v296 = 0;
  v273 = v27;
  v294 = (__int64 *)&v298;
  v300 = (__int64 *)&v304;
  v297 = 1;
  v288 = 0x3FF0000000000000LL;
  v299 = 0;
  v301 = 2;
  v302 = 0;
  v303 = 1;
  v289 = 335;
  LODWORD(v290) = 0;
  v291 = (__int64 *)v286;
  if ( v26
    && (v24 = (_BYTE *)v258, (unsigned __int8)sub_1009690((double *)&v288, v258))
    && (v43 = *(_QWORD *)(v2 - 32), *(_BYTE *)v43 == 85)
    && (v44 = *(_QWORD *)(v43 - 32)) != 0
    && !*(_BYTE *)v44
    && *(_QWORD *)(v44 + 24) == *(_QWORD *)(v43 + 80)
    && (v45 = v289, *(_DWORD *)(v44 + 36) == v289)
    && (v46 = *(_QWORD *)(v43 + 32 * ((unsigned int)v290 - (unsigned __int64)(*(_DWORD *)(v43 + 4) & 0x7FFFFFF)))) != 0 )
  {
    v47 = v291;
    *v291 = v46;
  }
  else
  {
    v26 = *(_BYTE *)v2 == 50;
    v305.m128i_i32[2] = 335;
    v306.m128i_i32[0] = 0;
    v305.m128i_i64[0] = 0xBFF0000000000000LL;
    v306.m128i_i64[1] = (__int64)v286;
    if ( !v26 )
      goto LABEL_16;
    v24 = *(_BYTE **)(v2 - 64);
    if ( !(unsigned __int8)sub_1009690((double *)v305.m128i_i64, (__int64)v24) )
      goto LABEL_16;
    v62 = *(_QWORD *)(v2 - 32);
    if ( *(_BYTE *)v62 != 85 )
      goto LABEL_16;
    v63 = *(_QWORD *)(v62 - 32);
    if ( !v63 )
      goto LABEL_16;
    if ( *(_BYTE *)v63 )
      goto LABEL_16;
    if ( *(_QWORD *)(v63 + 24) != *(_QWORD *)(v62 + 80) )
      goto LABEL_16;
    v45 = v305.m128i_u32[2];
    if ( *(_DWORD *)(v63 + 36) != v305.m128i_i32[2] )
      goto LABEL_16;
    v64 = *(_QWORD *)(v62 + 32 * (v306.m128i_u32[0] - (unsigned __int64)(*(_DWORD *)(v62 + 4) & 0x7FFFFFF)));
    if ( !v64 )
      goto LABEL_16;
    v47 = (__int64 *)v306.m128i_i64[1];
    *(_QWORD *)v306.m128i_i64[1] = v64;
  }
  v48 = *(_QWORD *)(v2 + 16);
  if ( v48 )
  {
    v49 = *(_QWORD *)(v2 + 16);
    while ( 1 )
    {
      v24 = *(_BYTE **)(v49 + 24);
      if ( *v24 == 47 )
      {
        v50 = *((_QWORD *)v24 - 8);
        if ( v50 )
        {
          if ( v2 == v50 )
          {
            v51 = *((_QWORD *)v24 - 4);
            if ( v51 )
            {
              if ( v2 == v51 )
              {
                if ( !v297 )
                  goto LABEL_99;
                v52 = v294;
                v45 = HIDWORD(v295);
                v47 = &v294[HIDWORD(v295)];
                if ( v294 != v47 )
                {
                  while ( v24 != (_BYTE *)*v52 )
                  {
                    if ( v47 == ++v52 )
                      goto LABEL_97;
                  }
                  goto LABEL_87;
                }
LABEL_97:
                if ( HIDWORD(v295) < (unsigned int)v295 )
                {
                  v45 = (unsigned int)++HIDWORD(v295);
                  *v47 = (__int64)v24;
                  ++v293;
                }
                else
                {
LABEL_99:
                  sub_C8CC70((__int64)&v293, (__int64)v24, (__int64)v47, v45, v48, v42);
                }
              }
            }
          }
        }
      }
LABEL_87:
      v49 = *(_QWORD *)(v49 + 8);
      if ( !v49 )
      {
        v3 = a1;
        break;
      }
    }
  }
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v53 = *(_QWORD *)(v2 - 8);
  else
    v53 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v54 = *(_QWORD *)(*(_QWORD *)(v53 + 32) + 16LL);
  if ( !v54 )
    goto LABEL_16;
  v55 = v2;
  v56 = v3;
  v57 = *(_QWORD *)(*(_QWORD *)(v53 + 32) + 16LL);
  do
  {
    v24 = *(_BYTE **)(v57 + 24);
    if ( *v24 == 50 )
    {
      v58 = (__int64)v286[0];
      if ( v286[0] == *((_BYTE **)v24 - 8) )
      {
        v59 = *((_QWORD *)v24 - 4);
        if ( *(_BYTE *)v59 == 85 )
        {
          v60 = *(_QWORD *)(v59 - 32);
          if ( v60 )
          {
            if ( !*(_BYTE *)v60
              && *(_QWORD *)(v60 + 24) == *(_QWORD *)(v59 + 80)
              && *(_DWORD *)(v60 + 36) == 335
              && v286[0] == *(_BYTE **)(v59 - 32LL * (*(_DWORD *)(v59 + 4) & 0x7FFFFFF)) )
            {
              if ( !v303 )
                goto LABEL_120;
              v61 = v300;
              v58 = HIDWORD(v301);
              v59 = (__int64)&v300[HIDWORD(v301)];
              if ( v300 == (__int64 *)v59 )
              {
LABEL_118:
                if ( HIDWORD(v301) < (unsigned int)v301 )
                {
                  ++HIDWORD(v301);
                  *(_QWORD *)v59 = v24;
                  ++v299;
                  goto LABEL_105;
                }
LABEL_120:
                sub_C8CC70((__int64)&v299, (__int64)v24, v59, v58, v54, v42);
                goto LABEL_105;
              }
              while ( v24 != (_BYTE *)*v61 )
              {
                if ( (__int64 *)v59 == ++v61 )
                  goto LABEL_118;
              }
            }
          }
        }
      }
    }
LABEL_105:
    v57 = *(_QWORD *)(v57 + 8);
  }
  while ( v57 );
  v3 = v56;
  v2 = v55;
LABEL_16:
  v28 = HIDWORD(v295);
  if ( HIDWORD(v295) != v296 && HIDWORD(v301) != v302 )
  {
    v223 = *(_QWORD *)(v2 + 40);
    v29 = v294;
    v276 = v294;
    if ( !v297 )
    {
      v28 = (unsigned int)v295;
      v29 = v294;
    }
    v30 = &v29[v28];
    v31 = v294;
    if ( v294 == v30 )
    {
      v31 = v294;
    }
    else
    {
      do
      {
        if ( (unsigned __int64)*v31 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v31;
      }
      while ( v31 != v30 );
    }
    v236 = *(_QWORD *)(*v31 + 40);
    v259 = v300;
    v244 = (__int64)(v303 ? &v300[HIDWORD(v301)] : &v300[(unsigned int)v301]);
    v32 = v300;
    if ( v300 == (__int64 *)v244 )
    {
      v32 = v300;
    }
    else
    {
      do
      {
        if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v32;
      }
      while ( v32 != (__int64 *)v244 );
    }
    v229 = *(_QWORD *)(*v32 + 40);
    v33 = (*(_BYTE *)(v2 + 7) & 0x40) != 0 ? *(_QWORD *)(v2 - 8) : v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    v220 = *(_QWORD *)(v33 + 32);
    if ( sub_B451B0(v220)
      && sub_B451C0(v220)
      && sub_B451E0(v220)
      && sub_B451D0(v220)
      && sub_B451B0(v2)
      && sub_B451F0(v2)
      && sub_B451D0(v2)
      && (v223 == v236 || v223 == v229) )
    {
      v134 = v276;
      while ( 1 )
      {
        v135 = v134;
        if ( v30 == v134 )
          break;
        ++v134;
        if ( *v135 != -2 && *v135 != -1 )
        {
          v215 = v2;
          v216 = v3;
          v217 = v135;
          while ( v236 == *(_QWORD *)(*v217 + 40) && sub_B451B0(*v217) )
          {
            do
            {
              if ( v30 == ++v217 )
                goto LABEL_363;
            }
            while ( (unsigned __int64)*v217 >= 0xFFFFFFFFFFFFFFFELL );
            if ( v30 == v217 )
            {
LABEL_363:
              v3 = v216;
              v2 = v215;
              goto LABEL_246;
            }
          }
          v3 = v216;
          v2 = v215;
          goto LABEL_39;
        }
      }
LABEL_246:
      v136 = v259;
      while ( 1 )
      {
        v137 = v136;
        if ( (__int64 *)v244 == v136 )
          break;
        ++v136;
        if ( (unsigned __int64)(*v137 + 2) > 1 )
        {
LABEL_332:
          if ( v229 != *(_QWORD *)(*v137 + 40) || !sub_B451B0(*v137) )
            goto LABEL_39;
          v197 = v137 + 1;
          while ( 1 )
          {
            v137 = v197;
            if ( (__int64 *)v244 == v197 )
              goto LABEL_249;
            ++v197;
            if ( (unsigned __int64)(*v137 + 2) > 1 )
              goto LABEL_332;
          }
        }
      }
LABEL_249:
      v138 = *(_QWORD *)(v2 - 32);
      v219 = (_BYTE *)v138;
      v280 = v3[2].m128i_i64[0];
      sub_D5F1F0(v280, v2);
      v139 = *(_QWORD *)(v2 + 8);
      v140 = *(_QWORD *)(v138 - 32LL * (*(_DWORD *)(v138 + 4) & 0x7FFFFFF));
      v292 = 257;
      v141 = sub_AD8DD0(v139, 1.0);
      HIDWORD(v285) = 0;
      v286[0] = (_BYTE *)(unsigned int)v285;
      if ( *(_BYTE *)(v280 + 108) )
      {
        v264 = sub_B35400(v280, 0x69u, (__int64)v141, v140, (unsigned int)v285, (__int64)&v288, 0, 0, 0);
      }
      else
      {
        v242 = (__int64)v141;
        v264 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64, _QWORD))(**(_QWORD **)(v280 + 80)
                                                                                              + 40LL))(
                 *(_QWORD *)(v280 + 80),
                 21,
                 v141,
                 v140,
                 *(unsigned int *)(v280 + 104));
        if ( !v264 )
        {
          v188 = *(_DWORD *)(v280 + 104);
          LOWORD(v307) = 257;
          v255 = v188;
          v189 = sub_B504D0(21, v242, v140, (__int64)&v305, 0, 0);
          v264 = v189;
          v190 = *(_QWORD *)(v280 + 96);
          if ( v190 )
            sub_B99FD0(v189, 3u, v190);
          sub_B45150(v264, v255);
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v280 + 88) + 16LL))(
            *(_QWORD *)(v280 + 88),
            v264,
            &v288,
            *(_QWORD *)(v280 + 56),
            *(_QWORD *)(v280 + 64));
          v191 = *(_QWORD *)v280 + 16LL * *(unsigned int *)(v280 + 8);
          if ( *(_QWORD *)v280 != v191 )
          {
            v192 = v2;
            v193 = v3;
            v194 = *(_QWORD *)v280;
            do
            {
              v195 = *(_QWORD *)(v194 + 8);
              v196 = *(_DWORD *)v194;
              v194 += 16;
              sub_B99FD0(v264, v196, v195);
            }
            while ( v191 != v194 );
            v3 = v193;
            v2 = v192;
          }
        }
      }
      v142 = v294;
      v143 = v294;
      if ( v297 )
        v144 = &v294[HIDWORD(v295)];
      else
        v144 = &v294[(unsigned int)v295];
      if ( v294 != v144 )
      {
        while ( (unsigned __int64)*v143 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( ++v143 == v144 )
            goto LABEL_256;
        }
        v144 = v143;
      }
LABEL_256:
      if ( (*(_BYTE *)(*v144 + 7) & 0x20) != 0 )
      {
        v145 = sub_B91C10(*v144, 3);
        v142 = v294;
        v253 = v145;
      }
      else
      {
        v253 = 0;
      }
      if ( v297 )
        v146 = &v142[HIDWORD(v295)];
      else
        v146 = &v142[(unsigned int)v295];
      if ( v142 == v146 )
      {
        v243 = sub_B45210(*v142);
      }
      else
      {
        v147 = v142;
        do
        {
          if ( (unsigned __int64)*v147 < 0xFFFFFFFFFFFFFFFELL )
            break;
          ++v147;
        }
        while ( v147 != v146 );
        v233 = v142;
        v148 = sub_B45210(*v147);
        v150 = (__int64)v233;
        v243 = v148;
        v151 = v233;
        while ( 1 )
        {
          v152 = *v151;
          if ( (unsigned __int64)*v151 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v146 == ++v151 )
            goto LABEL_267;
        }
        if ( v146 != v151 )
        {
          v235 = v8;
          v172 = v146;
          v173 = v253;
          v228 = v2;
          v174 = v151;
          v175 = *v151;
          do
          {
            v176 = 0;
            if ( (*(_BYTE *)(v175 + 7) & 0x20) != 0 )
              v176 = sub_B91C10(v175, 3);
            v173 = sub_B916B0(v173, v176, v149, v150, v152);
            v243 &= sub_B45210(v175);
            sub_F162A0((__int64)v3, v175, v264);
            sub_F207A0((__int64)v3, (__int64 *)v175);
            v177 = v174 + 1;
            if ( v174 + 1 == v172 )
              break;
            while ( 1 )
            {
              v175 = *v177;
              v174 = v177;
              if ( (unsigned __int64)*v177 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v172 == ++v177 )
                goto LABEL_305;
            }
          }
          while ( v177 != v172 );
LABEL_305:
          v253 = v173;
          v8 = v235;
          v2 = v228;
        }
      }
LABEL_267:
      sub_B99FD0(v264, 3u, v253);
      sub_B45170(v264, v243);
      v153 = (_QWORD *)sub_B47F80(v219);
      sub_B44220(v153, (__int64)(v219 + 24), 0);
      v154 = v300;
      v155 = v300;
      if ( v303 )
        v156 = &v300[HIDWORD(v301)];
      else
        v156 = &v300[(unsigned int)v301];
      if ( v300 != v156 )
      {
        while ( (unsigned __int64)*v155 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( ++v155 == v156 )
            goto LABEL_272;
        }
        v156 = v155;
      }
LABEL_272:
      if ( (*(_BYTE *)(*v156 + 7) & 0x20) != 0 )
      {
        v157 = sub_B91C10(*v156, 3);
        v154 = v300;
        v254 = v157;
      }
      else
      {
        v254 = 0;
      }
      if ( v303 )
        v158 = &v154[HIDWORD(v301)];
      else
        v158 = &v154[(unsigned int)v301];
      if ( v154 == v158 )
      {
        v234 = sub_B45210(*v154);
      }
      else
      {
        v159 = v154;
        do
        {
          if ( (unsigned __int64)*v159 < 0xFFFFFFFFFFFFFFFELL )
            break;
          ++v159;
        }
        while ( v159 != v158 );
        v221 = v154;
        v226 = v158;
        v160 = sub_B45210(*v159);
        v162 = (__int64)v221;
        v163 = v226;
        v234 = v160;
        v164 = v221;
        while ( 1 )
        {
          v165 = *v164;
          if ( (unsigned __int64)*v164 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v226 == ++v164 )
            goto LABEL_283;
        }
        if ( v226 != v164 )
        {
          v227 = v8;
          v167 = *v164;
          v222 = v2;
          v168 = v163;
          v169 = v164;
          do
          {
            v170 = 0;
            if ( (*(_BYTE *)(v167 + 7) & 0x20) != 0 )
              v170 = sub_B91C10(v167, 3);
            v254 = sub_B916B0(v254, v170, v162, v161, v165);
            v234 &= sub_B45210(v167);
            sub_F162A0((__int64)v3, v167, (__int64)v153);
            sub_F207A0((__int64)v3, (__int64 *)v167);
            v171 = v169 + 1;
            if ( v169 + 1 == v168 )
              break;
            while ( 1 )
            {
              v167 = *v171;
              v169 = v171;
              if ( (unsigned __int64)*v171 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v168 == ++v171 )
                goto LABEL_295;
            }
          }
          while ( v168 != v171 );
LABEL_295:
          v8 = v227;
          v2 = v222;
        }
      }
LABEL_283:
      sub_B99FD0((__int64)v153, 3u, v254);
      sub_B45170((__int64)v153, v234);
      v26 = *(_BYTE *)v2 == 50;
      v305.m128i_i64[0] = 0xBFF0000000000000LL;
      v305.m128i_i64[1] = (__int64)v219;
      if ( v26
        && (unsigned __int8)sub_1009690((double *)v305.m128i_i64, *(_QWORD *)(v2 - 64))
        && (v178 = *(_BYTE *)(v280 + 108), *(_QWORD *)(v2 - 32) == v305.m128i_i64[1]) )
      {
        HIDWORD(v285) = 0;
        v292 = 257;
        v286[0] = (_BYTE *)(unsigned int)v285;
        if ( v178 )
        {
          v179 = sub_B35400(v280, 0x6Cu, v264, (__int64)v153, (unsigned int)v285, (__int64)&v288, 0, 0, 0);
        }
        else
        {
          v179 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *, _QWORD))(**(_QWORD **)(v280 + 80) + 40LL))(
                   *(_QWORD *)(v280 + 80),
                   18,
                   v264,
                   v153,
                   *(unsigned int *)(v280 + 104));
          if ( !v179 )
          {
            v198 = *(_DWORD *)(v280 + 104);
            LOWORD(v307) = 257;
            v256 = v198;
            v199 = sub_B504D0(18, v264, (__int64)v153, (__int64)&v305, 0, 0);
            v200 = v199;
            v201 = *(_QWORD *)(v280 + 96);
            if ( v201 )
            {
              v268 = v199;
              sub_B99FD0(v199, 3u, v201);
              v200 = v268;
            }
            v269 = v200;
            sub_B45150(v200, v256);
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v280 + 88) + 16LL))(
              *(_QWORD *)(v280 + 88),
              v269,
              &v288,
              *(_QWORD *)(v280 + 56),
              *(_QWORD *)(v280 + 64));
            v202 = v2;
            v203 = v3;
            v204 = *(_QWORD *)v280;
            v205 = v269;
            v270 = *(_QWORD *)v280 + 16LL * *(unsigned int *)(v280 + 8);
            while ( v204 != v270 )
            {
              v206 = *(_QWORD *)(v204 + 8);
              v207 = *(_DWORD *)v204;
              v204 += 16;
              sub_B99FD0(v205, v207, v206);
            }
            v179 = v205;
            v3 = v203;
            v2 = v202;
          }
        }
        v265 = v179;
        v292 = 257;
        v166 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(v280 + 80) + 48LL))(
                 *(_QWORD *)(v280 + 80),
                 12,
                 v179,
                 *(unsigned int *)(v280 + 104));
        if ( !v166 )
        {
          v180 = *(_DWORD *)(v280 + 104);
          LOWORD(v307) = 257;
          v181 = sub_B50340(12, v265, (__int64)&v305, 0, 0);
          v182 = *(_QWORD *)(v280 + 96);
          if ( v182 )
          {
            v266 = v181;
            sub_B99FD0(v181, 3u, v182);
            v181 = v266;
          }
          v267 = v181;
          sub_B45150(v181, v180);
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v280 + 88) + 16LL))(
            *(_QWORD *)(v280 + 88),
            v267,
            &v288,
            *(_QWORD *)(v280 + 56),
            *(_QWORD *)(v280 + 64));
          v166 = v267;
          v183 = *(_QWORD *)v280 + 16LL * *(unsigned int *)(v280 + 8);
          if ( *(_QWORD *)v280 != v183 )
          {
            v184 = v3;
            v185 = *(_QWORD *)v280;
            do
            {
              v186 = *(_QWORD *)(v185 + 8);
              v187 = *(_DWORD *)v185;
              v185 += 16;
              sub_B99FD0(v267, v187, v186);
            }
            while ( v183 != v185 );
            v166 = v267;
            v3 = v184;
          }
        }
      }
      else
      {
        v292 = 257;
        HIDWORD(v285) = 0;
        v286[0] = (_BYTE *)(unsigned int)v285;
        if ( *(_BYTE *)(v280 + 108) )
        {
          v166 = sub_B35400(v280, 0x6Cu, v264, (__int64)v153, (unsigned int)v285, (__int64)&v288, 0, 0, 0);
        }
        else
        {
          v166 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *, _QWORD))(**(_QWORD **)(v280 + 80) + 40LL))(
                   *(_QWORD *)(v280 + 80),
                   18,
                   v264,
                   v153,
                   *(unsigned int *)(v280 + 104));
          if ( !v166 )
          {
            v208 = *(_DWORD *)(v280 + 104);
            LOWORD(v307) = 257;
            v257 = v208;
            v209 = sub_B504D0(18, v264, (__int64)v153, (__int64)&v305, 0, 0);
            v210 = v209;
            v211 = *(_QWORD *)(v280 + 96);
            if ( v211 )
            {
              v271 = v209;
              sub_B99FD0(v209, 3u, v211);
              v210 = v271;
            }
            v272 = v210;
            sub_B45150(v210, v257);
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v280 + 88) + 16LL))(
              *(_QWORD *)(v280 + 88),
              v272,
              &v288,
              *(_QWORD *)(v280 + 56),
              *(_QWORD *)(v280 + 64));
            v212 = *(_QWORD *)v280;
            v282 = *(_QWORD *)v280 + 16LL * *(unsigned int *)(v280 + 8);
            while ( v282 != v212 )
            {
              v213 = *(_QWORD *)(v212 + 8);
              v214 = *(_DWORD *)v212;
              v212 += 16;
              sub_B99FD0(v272, v214, v213);
            }
            v166 = v272;
          }
        }
      }
      v281 = v166;
      sub_B47C00(v166, v2, 0, 0);
      sub_B45170(
        v281,
        (unsigned __int8)v243 & (unsigned __int8)v234 & 0x71 | ((unsigned __int8)v243 | (unsigned __int8)v234) & 0xE);
      v24 = (_BYTE *)v2;
      v13 = sub_F162A0((__int64)v3, v2, v281);
      if ( v13 )
        goto LABEL_57;
    }
  }
LABEL_39:
  if ( *(_BYTE *)v258 <= 0x15u )
  {
    if ( *(_BYTE *)v273 != 86 )
      goto LABEL_42;
    v24 = (_BYTE *)v2;
    v13 = sub_F26350((__int64)v3, (_BYTE *)v2, v273, 0);
    if ( !v13 )
      goto LABEL_40;
    goto LABEL_57;
  }
LABEL_40:
  if ( *(_BYTE *)v273 <= 0x15u && *(_BYTE *)v258 == 86 )
  {
    v24 = (_BYTE *)v2;
    v13 = sub_F26350((__int64)v3, (_BYTE *)v2, v258, 0);
    if ( v13 )
      goto LABEL_57;
  }
LABEL_42:
  if ( !sub_B451B0(v2) )
    goto LABEL_47;
  if ( sub_B451F0(v2) )
  {
    v305.m128i_i64[0] = (__int64)&v285;
    v305.m128i_i64[1] = (__int64)v286;
    if ( (unsigned __int8)sub_1159650(&v305, v258) && (*v286[0] > 0x15u || *(_BYTE *)v273 > 0x15u) )
    {
      v78 = (unsigned int **)v3[2].m128i_i64[0];
      LOWORD(v307) = 257;
      sub_10A0170((__int64)&v288, v2);
      v79 = sub_A826E0(v78, v286[0], (_BYTE *)v273, v288, (__int64)&v305, 0);
      v24 = v285;
      LOWORD(v307) = 257;
      v13 = sub_109FE60(21, (__int64)v285, v79, v2, (__int64)&v305, v80, 0, 0);
      goto LABEL_57;
    }
    v24 = (_BYTE *)v273;
    v305.m128i_i64[0] = (__int64)&v285;
    v305.m128i_i64[1] = (__int64)v286;
    if ( (unsigned __int8)sub_1159650(&v305, v273) && (*v286[0] > 0x15u || *(_BYTE *)v258 > 0x15u) )
    {
      v81 = (unsigned int **)v3[2].m128i_i64[0];
      LOWORD(v307) = 257;
      sub_10A0170((__int64)&v288, v2);
      v24 = (_BYTE *)sub_A826E0(v81, v286[0], (_BYTE *)v258, v288, (__int64)&v305, 0);
      LOWORD(v307) = 257;
      v13 = sub_109FE60(21, (__int64)v24, (__int64)v285, v2, (__int64)&v305, v82, 0, 0);
      goto LABEL_57;
    }
    v305.m128i_i64[0] = 0x3FF0000000000000LL;
    v305.m128i_i64[1] = (__int64)v286;
    if ( *(_BYTE *)v273 == 50 )
    {
      v24 = *(_BYTE **)(v273 - 64);
      if ( (unsigned __int8)sub_1009690((double *)v305.m128i_i64, (__int64)v24) )
      {
        v97 = *(_QWORD *)(v273 - 32);
        if ( v97 )
        {
          *(_QWORD *)v305.m128i_i64[1] = v97;
          v24 = v286[0];
          LOWORD(v307) = 257;
          v13 = sub_109FE60(18, (__int64)v286[0], v258, v2, (__int64)&v305, v96, 0, 0);
          goto LABEL_57;
        }
      }
    }
    if ( sub_B451B0(v2) )
      goto LABEL_44;
LABEL_47:
    if ( !sub_B451C0(v2) )
      goto LABEL_50;
    if ( !sub_B451D0(v2) )
      goto LABEL_50;
    v24 = *(_BYTE **)(v2 - 64);
    if ( *(_BYTE *)v2 != 50 )
      goto LABEL_50;
    if ( !v24
      || (v83 = *(_QWORD *)(v2 - 32), v283 = *(_QWORD *)(v2 - 64), *(_BYTE *)v83 != 85)
      || (v93 = *(_QWORD *)(v83 - 32)) == 0
      || *(_BYTE *)v93
      || *(_QWORD *)(v93 + 24) != *(_QWORD *)(v83 + 80)
      || *(_DWORD *)(v93 + 36) != 170
      || (v94 = *(_BYTE **)(v83 - 32LL * (*(_DWORD *)(v83 + 4) & 0x7FFFFFF))) == 0
      || v94 != v24 )
    {
      v305.m128i_i32[0] = 170;
      v305.m128i_i32[2] = 0;
      v306.m128i_i64[0] = (__int64)&v283;
      v306.m128i_i64[1] = (__int64)&v283;
      if ( !(unsigned __int8)sub_10E25C0((__int64)&v305, (__int64)v24) )
      {
LABEL_50:
        v35 = *(_QWORD *)(v2 - 32);
LABEL_51:
        if ( *(_BYTE *)v35 == 85 )
        {
          v65 = *(_QWORD *)(v35 - 32);
          if ( v65 )
          {
            if ( !*(_BYTE *)v65 && *(_QWORD *)(v65 + 24) == *(_QWORD *)(v35 + 80) && (*(_BYTE *)(v65 + 33) & 0x20) != 0 )
            {
              v66 = *(_QWORD *)(v35 + 16);
              if ( v66 )
              {
                v13 = *(unsigned __int8 **)(v66 + 8);
                if ( !v13 )
                {
                  v237 = v35;
                  v277 = *(_QWORD *)(v35 - 32);
                  if ( sub_B451B0(v2) && sub_B451F0(v2) )
                  {
                    v67 = v3[2].m128i_i64[0];
                    v260 = *(_QWORD *)(v2 - 64);
                    v68 = *(_DWORD *)(v277 + 36);
                    v305.m128i_i64[0] = (__int64)&v306;
                    v245 = v68;
                    v305.m128i_i64[1] = 0x600000000LL;
                    if ( v68 == 284 )
                    {
                      v113 = *(_QWORD *)(v237 - 32LL * (*(_DWORD *)(v237 + 4) & 0x7FFFFFF));
                      v305.m128i_i32[2] = 1;
                      v292 = 257;
                      v306.m128i_i64[0] = v113;
                      v69 = sub_B45210(v2);
                      v70 = *(_QWORD *)(v237 + 32 * (1LL - (*(_DWORD *)(v237 + 4) & 0x7FFFFFF)));
                      goto LABEL_148;
                    }
                    if ( v68 > 0x11C )
                    {
                      if ( v68 == 285 )
                      {
                        v247 = v237;
                        if ( !sub_B451D0(v2) )
                        {
LABEL_152:
                          if ( (__m128i *)v305.m128i_i64[0] != &v306 )
                            _libc_free(v305.m128i_i64[0], v24);
                          if ( v13 )
                            goto LABEL_57;
                          goto LABEL_52;
                        }
                        v231 = v237;
                        v104 = *(_QWORD *)(v237 - 32LL * (*(_DWORD *)(v237 + 4) & 0x7FFFFFF));
                        v305.m128i_i32[2] = 1;
                        v287 = 257;
                        v306.m128i_i64[0] = v104;
                        v240 = *(_QWORD *)(v237 + 32 * (1LL - (*(_DWORD *)(v237 + 4) & 0x7FFFFFF)));
                        v248 = sub_AD6530(
                                 *(_QWORD *)(*(_QWORD *)(v247 + 32 * (1LL - (*(_DWORD *)(v247 + 4) & 0x7FFFFFF))) + 8LL),
                                 (__int64)v24);
                        v105 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v67 + 80) + 32LL))(
                                 *(_QWORD *)(v67 + 80),
                                 15,
                                 v248,
                                 v240,
                                 0,
                                 0);
                        v107 = v231;
                        v108 = v105;
                        if ( !v105 )
                        {
                          v224 = v231;
                          v292 = 257;
                          v232 = sub_B504D0(15, v248, v240, (__int64)&v288, 0, 0);
                          (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v67 + 88)
                                                                                            + 16LL))(
                            *(_QWORD *)(v67 + 88),
                            v232,
                            v286,
                            *(_QWORD *)(v67 + 56),
                            *(_QWORD *)(v67 + 64));
                          v117 = v3;
                          v107 = v224;
                          v118 = *(_QWORD *)v67;
                          v251 = *(_QWORD *)v67 + 16LL * *(unsigned int *)(v67 + 8);
                          while ( v118 != v251 )
                          {
                            v119 = *(_QWORD *)(v118 + 8);
                            v120 = *(_DWORD *)v118;
                            v225 = v107;
                            v118 += 16;
                            sub_B99FD0(v232, v120, v119);
                            v107 = v225;
                          }
                          v108 = v232;
                          v3 = v117;
                        }
                        v109 = v305.m128i_u32[2];
                        v110 = v305.m128i_u32[2] + 1LL;
                        if ( v110 > v305.m128i_u32[3] )
                        {
                          v241 = v107;
                          v250 = v108;
                          sub_C8D5F0((__int64)&v305, &v306, v110, 8u, v106, v108);
                          v109 = v305.m128i_u32[2];
                          v107 = v241;
                          v108 = v250;
                        }
                        *(_QWORD *)(v305.m128i_i64[0] + 8 * v109) = v108;
                        v111 = *(_BYTE **)(v2 + 8);
                        ++v305.m128i_i32[2];
                        v286[0] = v111;
                        v249 = v305.m128i_i32[2];
                        v112 = *(_BYTE **)(*(_QWORD *)(v107 + 32 * (1LL - (*(_DWORD *)(v107 + 4) & 0x7FFFFFF))) + 8LL);
                        v292 = 257;
                        v286[1] = v112;
                        LODWORD(v285) = sub_B45210(v2);
                        BYTE4(v285) = 1;
                        v76 = sub_B33D10(
                                v67,
                                0x11Du,
                                (__int64)v286,
                                2,
                                v305.m128i_i32[0],
                                v249,
                                (__int64)v285,
                                (__int64)&v288);
LABEL_151:
                        v292 = 257;
                        v24 = (_BYTE *)v2;
                        v13 = (unsigned __int8 *)sub_B504D0(18, v260, v76, (__int64)&v288, 0, 0);
                        sub_B45260(v13, v2, 1);
                        goto LABEL_152;
                      }
                    }
                    else if ( (v68 & 0xFFFFFFFD) == 0x58 )
                    {
                      v292 = 257;
                      v69 = sub_B45210(v2);
                      v70 = *(_QWORD *)(v237 - 32LL * (*(_DWORD *)(v237 + 4) & 0x7FFFFFF));
LABEL_148:
                      v71 = sub_11553A0((__int64 *)v67, v70, v69, 1, (__int64)&v288, 0);
                      v73 = v305.m128i_u32[2];
                      v74 = v305.m128i_u32[2] + 1LL;
                      if ( v74 > v305.m128i_u32[3] )
                      {
                        v239 = v71;
                        sub_C8D5F0((__int64)&v305, &v306, v74, 8u, v71, v72);
                        v73 = v305.m128i_u32[2];
                        v71 = v239;
                      }
                      *(_QWORD *)(v305.m128i_i64[0] + 8 * v73) = v71;
                      v292 = 257;
                      ++v305.m128i_i32[2];
                      v75 = sub_B45210(v2);
                      BYTE4(v286[0]) = 1;
                      LODWORD(v286[0]) = v75;
                      v285 = *(_BYTE **)(v2 + 8);
                      v76 = sub_B33D10(
                              v67,
                              v245,
                              (__int64)&v285,
                              1,
                              v305.m128i_i32[0],
                              v305.m128i_i32[2],
                              (__int64)v286[0],
                              (__int64)&v288);
                      goto LABEL_151;
                    }
                  }
                }
              }
            }
          }
        }
LABEL_52:
        if ( !sub_B451B0(v2) )
          goto LABEL_56;
        if ( sub_B451F0(v2) )
        {
          v36 = *(_QWORD *)(v2 - 32);
          if ( *(_BYTE *)v36 == 85 )
          {
            v114 = *(_QWORD *)(v36 - 32);
            if ( v114 )
            {
              if ( !*(_BYTE *)v114
                && *(_QWORD *)(v114 + 24) == *(_QWORD *)(v36 + 80)
                && (*(_BYTE *)(v114 + 33) & 0x20) != 0
                && *(_DWORD *)(v114 + 36) == 335 )
              {
                v115 = *(_QWORD *)(v36 + 16);
                if ( v115 )
                {
                  if ( !*(_QWORD *)(v115 + 8) && sub_B451B0(*(_QWORD *)(v2 - 32)) && sub_B451F0(v36) )
                  {
                    v116 = *(_QWORD *)(v36 - 32LL * (*(_DWORD *)(v36 + 4) & 0x7FFFFFF));
                    if ( *(_BYTE *)v116 > 0x1Cu )
                    {
                      if ( *(_BYTE *)v116 == 50 )
                      {
                        v252 = *(_BYTE **)(v116 - 64);
                        if ( v252 )
                        {
                          v121 = *(_BYTE **)(v116 - 32);
                          if ( v121 )
                          {
                            v278 = (unsigned int **)v3[2].m128i_i64[0];
                            v263 = *(_QWORD *)(v2 - 64);
                            if ( sub_B451B0(v116) && sub_B451F0(v2) )
                            {
                              v122 = *(_QWORD *)(v116 + 16);
                              if ( v122 )
                              {
                                if ( !*(_QWORD *)(v122 + 8) )
                                {
                                  LOWORD(v307) = 257;
                                  sub_10A0170((__int64)&v288, v116);
                                  v123 = sub_A82920(v278, v121, v252, v288, (__int64)&v305, 0);
                                  LOWORD(v307) = 257;
                                  v124 = v123;
                                  sub_10A0170((__int64)&v288, v36);
                                  v125 = *(_QWORD *)(v36 - 32);
                                  if ( !v125 || *(_BYTE *)v125 || *(_QWORD *)(v125 + 24) != *(_QWORD *)(v36 + 80) )
                                    BUG();
                                  v126 = sub_B33BC0((__int64)v278, *(_DWORD *)(v125 + 36), v124, v288, (__int64)&v305);
                                  LOWORD(v307) = 257;
                                  v128 = sub_109FE60(18, v263, v126, v2, (__int64)&v305, v127, 0, 0);
                                  v24 = v218;
                                  v13 = v128;
                                  if ( v128 )
                                    goto LABEL_57;
                                }
                              }
                            }
                          }
                        }
                      }
                      if ( !sub_B451B0(v2) )
                        goto LABEL_56;
                    }
                  }
                }
              }
            }
          }
        }
        v305.m128i_i32[0] = 284;
        v305.m128i_i32[2] = 0;
        v306.m128i_i64[0] = v273;
        v306.m128i_i32[2] = 1;
        v307 = &v284;
        if ( !(unsigned __int8)sub_10A5370((__int64)&v305, v258) )
        {
LABEL_56:
          v24 = (_BYTE *)v2;
          v13 = sub_115A4C0(v3, (unsigned __int8 *)v2);
          goto LABEL_57;
        }
        v129 = (unsigned int **)v3[2].m128i_i64[0];
        LOWORD(v307) = 257;
        sub_10A0170((__int64)&v288, v2);
        v130 = sub_AD8DD0(*(_QWORD *)(v2 + 8), -1.0);
        v131 = sub_92A220(v129, v284, v130, v288, (__int64)&v305, 0);
        v132 = v3[2].m128i_i64[0];
        LOWORD(v307) = 257;
        v133 = v131;
        v279 = v132;
        sub_10A0170((__int64)&v288, v2);
        v87 = v133;
        v88 = 284;
        v89 = v288;
        v90 = v273;
        v91 = v279;
LABEL_179:
        v92 = sub_B33C40(v91, v88, v90, v87, v89, (__int64)&v305);
        v24 = (_BYTE *)v2;
        v13 = sub_F162A0((__int64)v3, v2, v92);
        goto LABEL_57;
      }
      v35 = *(_QWORD *)(v2 - 32);
      if ( v35 != *(_QWORD *)v306.m128i_i64[1] )
        goto LABEL_51;
    }
    v84 = v3[2].m128i_i64[0];
    LOWORD(v307) = 257;
    sub_10A0170((__int64)&v288, v2);
    v85 = v283;
    v86 = sub_AD8DD0(*(_QWORD *)(v2 + 8), 1.0);
    v87 = v85;
    v88 = 26;
    v89 = v288;
    v90 = (__int64)v86;
    v91 = v84;
    goto LABEL_179;
  }
LABEL_44:
  v34 = *(_QWORD *)(v258 + 16);
  if ( !v34 || *(_QWORD *)(v34 + 8) || (v77 = *(_QWORD *)(v273 + 16)) == 0 || *(_QWORD *)(v77 + 8) )
  {
    if ( !sub_B451C0(v2) )
      goto LABEL_47;
    goto LABEL_46;
  }
  LODWORD(v288) = 325;
  v289 = 0;
  v290 = &v284;
  if ( (unsigned __int8)sub_10E25C0((__int64)&v288, v258) )
  {
    v305.m128i_i32[0] = 63;
    v305.m128i_i32[2] = 0;
    v306.m128i_i64[0] = (__int64)v284;
    if ( sub_11596A0((__int64)&v305, v273) )
    {
      v261 = 0;
      goto LABEL_162;
    }
  }
  v24 = (_BYTE *)v258;
  LODWORD(v288) = 63;
  v289 = 0;
  v290 = &v284;
  if ( (unsigned __int8)sub_10E25C0((__int64)&v288, v258)
    && (v24 = (_BYTE *)v273,
        v305.m128i_i32[0] = 325,
        v305.m128i_i32[2] = 0,
        v306.m128i_i64[0] = (__int64)v284,
        v261 = sub_11596A0((__int64)&v305, v273)) )
  {
LABEL_162:
    v24 = (_BYTE *)v3[4].m128i_i64[1];
    if ( !(unsigned __int8)sub_11C9D70(v8, v24, *(_QWORD *)(v2 + 8), 490, 491, 495) )
      goto LABEL_163;
    v98 = sub_BD5C60(v2);
    v313 = &v321;
    v311 = v98;
    v312 = &v320;
    v305.m128i_i64[0] = (__int64)&v306;
    v320 = &unk_49DA100;
    v305.m128i_i64[1] = 0x200000000LL;
    v314 = 0;
    v321 = &unk_49DA0B0;
    v315 = 0;
    v316 = 512;
    v317 = 7;
    v318 = 0;
    v319 = 0;
    v309 = 0u;
    LOWORD(v310) = 0;
    sub_D5F1F0((__int64)&v305, v2);
    v230 = v315;
    v274 = v314;
    v246 = v317;
    v238 = v316;
    v315 = sub_B45210(v2);
    v99 = *(_QWORD *)(v258 - 32);
    if ( !v99 || *(_BYTE *)v99 || *(_QWORD *)(v99 + 24) != *(_QWORD *)(v258 + 80) )
      BUG();
    v100 = v3[4].m128i_i64[1];
    v285 = *(_BYTE **)(v99 + 120);
    v101 = sub_11CCA60((_DWORD)v284, v100, 490, 491, 495, (unsigned int)&v305, (__int64)&v285);
    v102 = v101;
    if ( v261 )
    {
      v262 = (_BYTE *)v101;
      v292 = 257;
      v103 = sub_AD8DD0(*(_QWORD *)(v2 + 8), 1.0);
      HIDWORD(v286[0]) = 0;
      v102 = sub_A82920((unsigned int **)&v305, v103, v262, LODWORD(v286[0]), (__int64)&v288, 0);
    }
    v24 = (_BYTE *)v2;
    v13 = sub_F162A0((__int64)v3, v2, v102);
    v314 = v274;
    v315 = v230;
    v316 = v238;
    v317 = v246;
    nullsub_61();
    v320 = &unk_49DA100;
    nullsub_63();
    if ( (__m128i *)v305.m128i_i64[0] != &v306 )
      _libc_free(v305.m128i_i64[0], v2);
  }
  else
  {
LABEL_163:
    if ( !sub_B451C0(v2) || !sub_B451B0(v2) )
      goto LABEL_47;
LABEL_46:
    v305.m128i_i64[0] = v258;
    v305.m128i_i64[1] = (__int64)&v284;
    if ( *(_BYTE *)v273 != 47 )
      goto LABEL_47;
    v24 = (_BYTE *)v273;
    if ( !(unsigned __int8)sub_1155120((__int64)&v305, v273) )
      goto LABEL_47;
    v13 = (unsigned __int8 *)v2;
    v95 = sub_AD8DD0(*(_QWORD *)(v2 + 8), 1.0);
    sub_F20660((__int64)v3, v2, 0, (__int64)v95);
    v24 = (_BYTE *)v2;
    sub_F20660((__int64)v3, v2, 1u, (__int64)v284);
  }
LABEL_57:
  if ( !v303 )
    _libc_free(v300, v24);
  if ( !v297 )
    _libc_free(v294, v24);
  return v13;
}
