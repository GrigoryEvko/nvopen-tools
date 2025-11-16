// Function: sub_373DE40
// Address: 0x373de40
//
unsigned __int64 __fastcall sub_373DE40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // r12
  __int64 v13; // rdx
  const __m128i *v14; // rdi
  __m128i *v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rax
  unsigned __int8 v23; // dl
  __int64 v24; // rax
  unsigned __int8 v25; // dl
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int8 v28; // cl
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  _QWORD *v34; // rdx
  int v35; // esi
  __int64 *v36; // rax
  unsigned __int64 *v37; // rax
  __int64 *v38; // rax
  unsigned __int64 v39; // r13
  unsigned __int64 v40; // r14
  int v41; // esi
  __int64 *v42; // rdi
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rbx
  unsigned __int64 v47; // rdx
  unsigned int v48; // eax
  __int64 v49; // rbx
  __int64 v50; // r12
  unsigned int v51; // esi
  unsigned int v52; // eax
  unsigned int v53; // edx
  __int64 *v54; // r13
  unsigned __int64 v55; // rax
  _QWORD *v56; // rdx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rax
  __int64 v60; // rsi
  int v61; // eax
  int v62; // ecx
  unsigned int v63; // eax
  __int64 *v64; // r12
  __int64 v65; // rdx
  unsigned int v66; // r13d
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rax
  _QWORD *v70; // r13
  __int64 v71; // r15
  _QWORD *v72; // r12
  __int64 v73; // rax
  __int64 v74; // rsi
  int v75; // eax
  int v76; // edi
  unsigned int v77; // eax
  __int64 v78; // rcx
  __int64 v79; // rdx
  const __m128i *v80; // rdi
  __m128i *v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rax
  unsigned __int64 v86; // r12
  int v88; // ecx
  _BYTE *v89; // r13
  int v90; // esi
  __int64 *v91; // r8
  unsigned int v92; // edx
  __int64 v93; // r9
  __int64 v94; // rcx
  __int64 v95; // rdi
  unsigned __int64 v96; // rcx
  int v97; // r10d
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // rax
  unsigned __int64 v101; // rbx
  unsigned __int64 v102; // rdx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rax
  unsigned __int64 v106; // rdx
  __int64 v107; // rbx
  _BYTE *v108; // rax
  _BYTE *v109; // rbx
  __int64 v110; // r9
  _QWORD *v111; // rbx
  _BYTE *v112; // r14
  __int64 *v113; // rdi
  int v114; // esi
  unsigned int v115; // ecx
  __int64 *v116; // rdx
  __int64 v117; // r9
  __int64 v118; // rax
  __int64 v119; // rdx
  unsigned __int64 v120; // rax
  unsigned __int64 v121; // r8
  _BYTE *v122; // rax
  int v123; // edx
  int v124; // r8d
  int v125; // r10d
  __int64 *v126; // rsi
  int v127; // ecx
  unsigned int v128; // edx
  __int64 v129; // rdi
  __int64 v130; // rax
  unsigned __int8 v131; // al
  __int64 v132; // r8
  _QWORD *v133; // rdx
  unsigned __int8 *v134; // r14
  __int64 v135; // rax
  unsigned __int64 v136; // rdx
  unsigned __int8 *v137; // r14
  __int64 v138; // rax
  unsigned __int64 v139; // rdx
  unsigned __int8 *v140; // r14
  __int64 v141; // rax
  unsigned __int64 v142; // rdx
  __int64 v143; // rax
  unsigned __int8 v144; // dl
  __int64 *v145; // rbx
  __int64 v146; // rax
  __int64 *v147; // r14
  signed __int64 v148; // rax
  __int64 v149; // r8
  unsigned __int64 v150; // r9
  __int64 v151; // rax
  unsigned __int64 v152; // rdx
  signed __int64 v153; // rax
  __int64 v154; // r8
  unsigned __int64 v155; // r9
  __int64 v156; // rax
  unsigned __int64 v157; // rdx
  signed __int64 v158; // rax
  __int64 v159; // r8
  unsigned __int64 v160; // r9
  __int64 v161; // rax
  unsigned __int64 v162; // rdx
  signed __int64 v163; // rax
  __int64 v164; // r8
  __int64 v165; // r9
  unsigned __int64 v166; // r13
  __int64 v167; // rax
  unsigned __int64 v168; // rdx
  _BYTE *v169; // r13
  char v170; // al
  unsigned __int64 v171; // rax
  __int64 v172; // r8
  unsigned __int64 v173; // r9
  __int64 v174; // rax
  unsigned __int64 v175; // rdx
  unsigned __int64 v176; // rax
  __int64 v177; // r8
  unsigned __int64 v178; // r9
  __int64 v179; // rax
  unsigned __int64 v180; // rdx
  unsigned __int64 v181; // rax
  __int64 v182; // r8
  unsigned __int64 v183; // r9
  __int64 v184; // rax
  unsigned __int64 v185; // rdx
  unsigned __int64 v186; // rax
  __int64 v187; // r9
  unsigned __int64 v188; // r8
  __int64 v189; // rax
  unsigned __int64 v190; // rdx
  __int64 v191; // r11
  __int64 v192; // rcx
  unsigned int v193; // esi
  __int64 v194; // rdx
  __int64 v195; // r9
  unsigned int v196; // edi
  __int64 *v197; // rax
  __int64 v198; // r8
  __int64 *v199; // rsi
  int v200; // ecx
  unsigned int v201; // edx
  __int64 v202; // rdi
  unsigned int v203; // r10d
  __int64 *v204; // r14
  size_t v205; // rdx
  __int64 v206; // r12
  __int64 v207; // r13
  __int64 *v208; // rbx
  unsigned __int64 v209; // r12
  __int64 v210; // rax
  _QWORD *v211; // rdx
  _QWORD *v212; // rax
  __int64 *v213; // rdx
  unsigned __int64 v214; // rcx
  __int64 *v215; // r14
  __int64 *v216; // r15
  __int64 *v217; // r13
  __int64 v218; // rsi
  __int64 *v219; // rax
  __int64 v220; // rax
  int v221; // edi
  unsigned int v222; // edx
  int v223; // eax
  __int64 v224; // r10
  unsigned int v225; // esi
  __int64 v226; // r9
  int v227; // r8d
  int v228; // edi
  int v229; // eax
  int v230; // edx
  unsigned int v231; // esi
  __int64 v232; // r9
  __int64 *v233; // rdi
  unsigned int v234; // r10d
  int v235; // edx
  unsigned __int64 v236; // [rsp+8h] [rbp-338h]
  unsigned __int64 v237; // [rsp+8h] [rbp-338h]
  unsigned __int64 v238; // [rsp+8h] [rbp-338h]
  unsigned __int64 v239; // [rsp+8h] [rbp-338h]
  unsigned __int64 v240; // [rsp+8h] [rbp-338h]
  unsigned __int64 v241; // [rsp+8h] [rbp-338h]
  unsigned __int64 v242; // [rsp+8h] [rbp-338h]
  unsigned __int64 v243; // [rsp+28h] [rbp-318h]
  _BYTE *v244; // [rsp+28h] [rbp-318h]
  __int64 v245; // [rsp+28h] [rbp-318h]
  __int64 v246; // [rsp+28h] [rbp-318h]
  __int64 v247; // [rsp+48h] [rbp-2F8h]
  int v249; // [rsp+50h] [rbp-2F0h]
  int v250; // [rsp+50h] [rbp-2F0h]
  __int64 v252; // [rsp+58h] [rbp-2E8h]
  __int64 *v253; // [rsp+58h] [rbp-2E8h]
  __int64 v254; // [rsp+58h] [rbp-2E8h]
  __int64 v255; // [rsp+58h] [rbp-2E8h]
  __int64 v256; // [rsp+60h] [rbp-2E0h]
  __int64 v257; // [rsp+60h] [rbp-2E0h]
  unsigned int v258; // [rsp+60h] [rbp-2E0h]
  __int64 v259; // [rsp+60h] [rbp-2E0h]
  int v260; // [rsp+60h] [rbp-2E0h]
  __int64 v261; // [rsp+60h] [rbp-2E0h]
  __int64 *v262; // [rsp+68h] [rbp-2D8h]
  __int64 v263; // [rsp+68h] [rbp-2D8h]
  __int64 *v264; // [rsp+68h] [rbp-2D8h]
  __int64 v265; // [rsp+68h] [rbp-2D8h]
  unsigned __int64 v266; // [rsp+70h] [rbp-2D0h] BYREF
  unsigned __int64 v267; // [rsp+78h] [rbp-2C8h] BYREF
  _BYTE *v268; // [rsp+80h] [rbp-2C0h] BYREF
  __int64 v269; // [rsp+88h] [rbp-2B8h]
  _BYTE v270[16]; // [rsp+90h] [rbp-2B0h] BYREF
  _BYTE v271[48]; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 *v272; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v273; // [rsp+D8h] [rbp-268h]
  _BYTE v274[64]; // [rsp+E0h] [rbp-260h] BYREF
  _BYTE *v275; // [rsp+120h] [rbp-220h] BYREF
  __int64 v276; // [rsp+128h] [rbp-218h]
  _BYTE v277[64]; // [rsp+130h] [rbp-210h] BYREF
  __int64 v278; // [rsp+170h] [rbp-1D0h] BYREF
  __int64 v279; // [rsp+178h] [rbp-1C8h]
  __int64 *v280; // [rsp+180h] [rbp-1C0h] BYREF
  unsigned int v281; // [rsp+188h] [rbp-1B8h]
  __int64 v282; // [rsp+1C0h] [rbp-180h] BYREF
  __int64 v283; // [rsp+1C8h] [rbp-178h]
  __int64 *v284; // [rsp+1D0h] [rbp-170h] BYREF
  unsigned int v285; // [rsp+1D8h] [rbp-168h]
  unsigned __int64 v286[16]; // [rsp+210h] [rbp-130h] BYREF
  __int64 v287[22]; // [rsp+290h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a1 + 216);
  v8 = *(_QWORD *)(v7 + 344);
  v9 = *(_DWORD *)(v7 + 360);
  v266 = 0;
  if ( !v9 )
  {
LABEL_35:
    memset(v286, 0, sizeof(v286));
    HIDWORD(v286[7]) = 8;
    v286[3] = (unsigned __int64)&v286[1];
    v286[4] = (unsigned __int64)&v286[1];
    v286[6] = (unsigned __int64)&v286[8];
    goto LABEL_36;
  }
  v10 = (unsigned int)(v9 - 1);
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v8 + 136LL * v11;
  v13 = *(_QWORD *)v12;
  if ( a2 != *(_QWORD *)v12 )
  {
    v35 = 1;
    while ( v13 != -4096 )
    {
      a5 = (unsigned int)(v35 + 1);
      v11 = v10 & (v35 + v11);
      v12 = v8 + 136LL * v11;
      v13 = *(_QWORD *)v12;
      if ( a2 == *(_QWORD *)v12 )
        goto LABEL_3;
      ++v35;
    }
    goto LABEL_35;
  }
LABEL_3:
  v286[5] = 0;
  v286[3] = (unsigned __int64)&v286[1];
  v286[4] = (unsigned __int64)&v286[1];
  v14 = *(const __m128i **)(v12 + 24);
  LODWORD(v286[1]) = 0;
  v286[2] = 0;
  if ( v14 )
  {
    v15 = sub_3735420(v14, (__int64)&v286[1]);
    v10 = (__int64)v15;
    do
    {
      v16 = (unsigned __int64)v15;
      v15 = (__m128i *)v15[1].m128i_i64[0];
    }
    while ( v15 );
    v286[3] = v16;
    v17 = v10;
    do
    {
      v13 = v17;
      v17 = *(_QWORD *)(v17 + 24);
    }
    while ( v17 );
    v18 = *(_QWORD *)(v12 + 48);
    v286[4] = v13;
    v286[2] = v10;
    v286[5] = v18;
  }
  a5 = *(unsigned int *)(v12 + 64);
  v286[6] = (unsigned __int64)&v286[8];
  v286[7] = 0x800000000LL;
  if ( (_DWORD)a5 )
    sub_37351E0((__int64)&v286[6], v12 + 56, v13, v10, a5, a6);
  if ( (unsigned __int64 *)v286[3] != &v286[1] )
  {
    v19 = v286[3];
    do
    {
      v20 = sub_373B0C0((__int64 *)a1, *(_QWORD *)(v19 + 40), a2, &v266);
      *(_QWORD *)(v20 + 40) = a3 & 0xFFFFFFFFFFFFFFFBLL;
      v21 = *(_QWORD **)(a3 + 32);
      if ( v21 )
      {
        *(_QWORD *)v20 = *v21;
        **(_QWORD **)(a3 + 32) = v20 & 0xFFFFFFFFFFFFFFFBLL;
      }
      *(_QWORD *)(a3 + 32) = v20;
      v19 = sub_220EEE0(v19);
    }
    while ( (unsigned __int64 *)v19 != &v286[1] );
  }
  if ( !v286[5] )
    goto LABEL_36;
  v22 = sub_AE7A60(*(_BYTE **)(a2 + 8));
  v23 = *(_BYTE *)(v22 - 16);
  if ( (v23 & 2) != 0 )
  {
    v24 = *(_QWORD *)(*(_QWORD *)(v22 - 32) + 32LL);
    v25 = *(_BYTE *)(v24 - 16);
    if ( (v25 & 2) != 0 )
    {
LABEL_19:
      v26 = *(_QWORD *)(v24 - 32);
      goto LABEL_20;
    }
  }
  else
  {
    v24 = *(_QWORD *)(v22 - 16 - 8LL * ((v23 >> 2) & 0xF) + 32);
    v25 = *(_BYTE *)(v24 - 16);
    if ( (v25 & 2) != 0 )
      goto LABEL_19;
  }
  v26 = v24 - 16 - 8LL * ((v25 >> 2) & 0xF);
LABEL_20:
  v27 = *(_QWORD *)(v26 + 24);
  if ( v27 )
  {
    v28 = *(_BYTE *)(v27 - 16);
    if ( (v28 & 2) != 0 )
    {
      v29 = *(_DWORD *)(v27 - 24);
      if ( v29 > 1 )
      {
        v30 = *(_QWORD *)(v27 - 32);
        v31 = v29 - 1;
LABEL_24:
        if ( !*(_QWORD *)(v30 + 8 * v31) && !sub_3736590((_QWORD *)a1) )
        {
          v32 = *(_QWORD *)(a1 + 88);
          *(_QWORD *)(a1 + 168) += 48LL;
          v33 = (v32 + 15) & 0xFFFFFFFFFFFFFFF0LL;
          if ( *(_QWORD *)(a1 + 96) >= v33 + 48 && v32 )
          {
            *(_QWORD *)(a1 + 88) = v33 + 48;
            if ( !v33 )
            {
              MEMORY[0x28] = 0;
              BUG();
            }
          }
          else
          {
            v33 = sub_9D1E70(a1 + 88, 48, 48, 4);
          }
          *(_BYTE *)(v33 + 30) = 0;
          *(_WORD *)(v33 + 28) = 24;
          *(_QWORD *)v33 = v33 | 4;
          *(_QWORD *)(v33 + 8) = 0;
          *(_QWORD *)(v33 + 16) = 0;
          *(_DWORD *)(v33 + 24) = -1;
          *(_QWORD *)(v33 + 32) = 0;
          *(_QWORD *)(v33 + 40) = a3 & 0xFFFFFFFFFFFFFFFBLL;
          v34 = *(_QWORD **)(a3 + 32);
          if ( v34 )
          {
            *(_QWORD *)v33 = *v34;
            **(_QWORD **)(a3 + 32) = v33 & 0xFFFFFFFFFFFFFFFBLL;
          }
          *(_QWORD *)(a3 + 32) = v33;
        }
      }
    }
    else
    {
      v222 = (*(_WORD *)(v27 - 16) >> 6) & 0xF;
      if ( v222 > 1 )
      {
        v31 = v222 - 1;
        v30 = v27 - 16 - 8LL * ((v28 >> 2) & 0xF);
        goto LABEL_24;
      }
    }
  }
LABEL_36:
  v278 = 0;
  v272 = (__int64 *)v274;
  v273 = 0x800000000LL;
  v276 = 0x800000000LL;
  v36 = (__int64 *)&v280;
  v275 = v277;
  v279 = 1;
  do
  {
    *v36 = -4096;
    v36 += 2;
  }
  while ( v36 != &v282 );
  v37 = (unsigned __int64 *)&v284;
  v282 = 0;
  v283 = 1;
  do
    *v37++ = -4096;
  while ( v37 != v286 );
  v38 = &v287[2];
  v287[0] = 0;
  v287[1] = 1;
  do
    *v38++ = -4096;
  while ( v38 != &v287[10] );
  v39 = v286[6] + 8LL * LODWORD(v286[7]);
  if ( v286[6] == v39 )
    goto LABEL_59;
  v40 = v286[6];
  v256 = a3;
  do
  {
    v49 = *(_QWORD *)(v39 - 8);
    v50 = *(_QWORD *)(v49 + 8);
    if ( (v279 & 1) != 0 )
    {
      v41 = 3;
      v42 = (__int64 *)&v280;
    }
    else
    {
      v51 = v281;
      v42 = v280;
      if ( !v281 )
      {
        v52 = v279;
        ++v278;
        a5 = 0;
        v53 = ((unsigned int)v279 >> 1) + 1;
        goto LABEL_53;
      }
      v41 = v281 - 1;
    }
    v43 = v41 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    v44 = &v42[2 * v43];
    a6 = *v44;
    if ( v50 == *v44 )
      goto LABEL_46;
    v125 = 1;
    a5 = 0;
    while ( a6 != -4096 )
    {
      if ( a5 || a6 != -8192 )
        v44 = (__int64 *)a5;
      a5 = (unsigned int)(v125 + 1);
      v43 = v41 & (v125 + v43);
      a6 = v42[2 * v43];
      if ( v50 == a6 )
        goto LABEL_46;
      ++v125;
      a5 = (__int64)v44;
      v44 = &v42[2 * v43];
    }
    if ( !a5 )
      a5 = (__int64)v44;
    v52 = v279;
    ++v278;
    v53 = ((unsigned int)v279 >> 1) + 1;
    if ( (v279 & 1) == 0 )
    {
      v51 = v281;
LABEL_53:
      if ( 3 * v51 > 4 * v53 )
        goto LABEL_54;
      goto LABEL_151;
    }
    v51 = 4;
    if ( 4 * v53 < 0xC )
    {
LABEL_54:
      if ( v51 - HIDWORD(v279) - v53 > v51 >> 3 )
        goto LABEL_55;
      sub_373BFC0((__int64)&v278, v51);
      if ( (v279 & 1) != 0 )
      {
        v200 = 3;
        v199 = (__int64 *)&v280;
      }
      else
      {
        v199 = v280;
        if ( !v281 )
        {
LABEL_348:
          LODWORD(v279) = (2 * ((unsigned int)v279 >> 1) + 2) | v279 & 1;
          BUG();
        }
        v200 = v281 - 1;
      }
      v52 = v279;
      v201 = v200 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      a5 = (__int64)&v199[2 * v201];
      v202 = *(_QWORD *)a5;
      if ( v50 == *(_QWORD *)a5 )
        goto LABEL_55;
      a6 = 1;
      v130 = 0;
      while ( v202 != -4096 )
      {
        if ( v202 == -8192 && !v130 )
          v130 = a5;
        v203 = a6 + 1;
        a6 = v201 + (unsigned int)a6;
        v201 = v200 & a6;
        a5 = (__int64)&v199[2 * (v200 & (unsigned int)a6)];
        v202 = *(_QWORD *)a5;
        if ( v50 == *(_QWORD *)a5 )
          goto LABEL_159;
        a6 = v203;
      }
      goto LABEL_157;
    }
LABEL_151:
    sub_373BFC0((__int64)&v278, 2 * v51);
    if ( (v279 & 1) != 0 )
    {
      v127 = 3;
      v126 = (__int64 *)&v280;
    }
    else
    {
      v126 = v280;
      if ( !v281 )
        goto LABEL_348;
      v127 = v281 - 1;
    }
    v52 = v279;
    v128 = v127 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    a5 = (__int64)&v126[2 * v128];
    v129 = *(_QWORD *)a5;
    if ( v50 == *(_QWORD *)a5 )
      goto LABEL_55;
    a6 = 1;
    v130 = 0;
    while ( v129 != -4096 )
    {
      if ( v129 == -8192 && !v130 )
        v130 = a5;
      v234 = a6 + 1;
      a6 = v128 + (unsigned int)a6;
      v128 = v127 & a6;
      a5 = (__int64)&v126[2 * (v127 & (unsigned int)a6)];
      v129 = *(_QWORD *)a5;
      if ( v50 == *(_QWORD *)a5 )
        goto LABEL_159;
      a6 = v234;
    }
LABEL_157:
    if ( v130 )
      a5 = v130;
LABEL_159:
    v52 = v279;
LABEL_55:
    LODWORD(v279) = (2 * (v52 >> 1) + 2) | v52 & 1;
    if ( *(_QWORD *)a5 != -4096 )
      --HIDWORD(v279);
    *(_QWORD *)a5 = v50;
    *(_QWORD *)(a5 + 8) = v49;
LABEL_46:
    v45 = (unsigned int)v276;
    v46 = v49 & 0xFFFFFFFFFFFFFFFBLL;
    v47 = (unsigned int)v276 + 1LL;
    if ( v47 > HIDWORD(v276) )
    {
      sub_C8D5F0((__int64)&v275, v277, v47, 8u, a5, a6);
      v45 = (unsigned int)v276;
    }
    v39 -= 8LL;
    *(_QWORD *)&v275[8 * v45] = v46;
    v48 = v276 + 1;
    LODWORD(v276) = v276 + 1;
  }
  while ( v40 != v39 );
  a3 = v256;
  if ( v48 )
  {
    v89 = v271;
    while ( 1 )
    {
      v94 = v48--;
      v95 = *(_QWORD *)&v275[8 * v94 - 8];
      LODWORD(v276) = v48;
      v96 = v95 & 0xFFFFFFFFFFFFFFF8LL;
      v267 = v95 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v283 & 1) != 0 )
      {
        v90 = 7;
        v91 = (__int64 *)&v284;
      }
      else
      {
        v91 = v284;
        if ( !v285 )
          goto LABEL_115;
        v90 = v285 - 1;
      }
      v92 = v90 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
      v93 = v91[v92];
      if ( v96 != v93 )
        break;
LABEL_109:
      if ( !v48 )
        goto LABEL_58;
    }
    v97 = 1;
    while ( v93 != -4096 )
    {
      v92 = v90 & (v97 + v92);
      v93 = v91[v92];
      if ( v96 == v93 )
        goto LABEL_109;
      ++v97;
    }
LABEL_115:
    if ( (v95 & 4) != 0 )
    {
      sub_373C7F0((__int64)v89, (__int64)&v282, (__int64 *)&v267);
      v100 = (unsigned int)v273;
      v101 = v267;
      v102 = (unsigned int)v273 + 1LL;
      if ( v102 > HIDWORD(v273) )
      {
        sub_C8D5F0((__int64)&v272, v274, v102, 8u, v98, v99);
        v100 = (unsigned int)v273;
      }
      v272[v100] = v101;
      v48 = v276;
      LODWORD(v273) = v273 + 1;
      goto LABEL_109;
    }
    sub_373C7F0((__int64)v89, (__int64)v287, (__int64 *)&v267);
    if ( v271[32] )
    {
      v105 = (unsigned int)v276;
      v106 = (unsigned int)v276 + 1LL;
      v107 = v267 | 4;
      if ( v106 > HIDWORD(v276) )
      {
        sub_C8D5F0((__int64)&v275, v277, v106, 8u, v103, v104);
        v105 = (unsigned int)v276;
      }
      *(_QWORD *)&v275[8 * v105] = v107;
      LODWORD(v276) = v276 + 1;
      v268 = v270;
      v269 = 0x200000000LL;
      v108 = (_BYTE *)sub_321DF00(v267);
      v109 = v108;
      if ( *v108 != 14 || (unsigned __int16)sub_AF18C0((__int64)v108) != 1 )
        goto LABEL_124;
      v131 = *(v109 - 16);
      v132 = (__int64)(v109 - 16);
      if ( (v131 & 2) != 0 )
      {
        v133 = (_QWORD *)*((_QWORD *)v109 - 4);
        v134 = (unsigned __int8 *)v133[9];
        if ( v134 )
        {
          if ( (unsigned int)*v134 - 25 <= 1 )
          {
LABEL_166:
            v135 = (unsigned int)v269;
            v136 = (unsigned int)v269 + 1LL;
            if ( v136 > HIDWORD(v269) )
            {
              sub_C8D5F0((__int64)&v268, v270, v136, 8u, v132, v110);
              v135 = (unsigned int)v269;
              v132 = (__int64)(v109 - 16);
            }
            *(_QWORD *)&v268[8 * v135] = v134;
            LODWORD(v269) = v269 + 1;
            v131 = *(v109 - 16);
            if ( (v131 & 2) == 0 )
              goto LABEL_169;
          }
          v133 = (_QWORD *)*((_QWORD *)v109 - 4);
        }
      }
      else
      {
        v133 = (_QWORD *)(v132 - 8LL * ((v131 >> 2) & 0xF));
        v134 = (unsigned __int8 *)v133[9];
        if ( v134 )
        {
          if ( (unsigned int)*v134 - 25 <= 1 )
            goto LABEL_166;
LABEL_169:
          v133 = (_QWORD *)(v132 - 8LL * ((v131 >> 2) & 0xF));
        }
      }
      v137 = (unsigned __int8 *)v133[10];
      if ( v137 )
      {
        if ( (unsigned int)*v137 - 25 <= 1 )
        {
          v138 = (unsigned int)v269;
          v139 = (unsigned int)v269 + 1LL;
          if ( v139 > HIDWORD(v269) )
          {
            v246 = v132;
            sub_C8D5F0((__int64)&v268, v270, v139, 8u, v132, v110);
            v138 = (unsigned int)v269;
            v132 = v246;
          }
          *(_QWORD *)&v268[8 * v138] = v137;
          LODWORD(v269) = v269 + 1;
          v131 = *(v109 - 16);
        }
        if ( (v131 & 2) != 0 )
          v133 = (_QWORD *)*((_QWORD *)v109 - 4);
        else
          v133 = (_QWORD *)(v132 - 8LL * ((v131 >> 2) & 0xF));
      }
      v140 = (unsigned __int8 *)v133[11];
      if ( v140 )
      {
        if ( (unsigned int)*v140 - 25 <= 1 )
        {
          v141 = (unsigned int)v269;
          v142 = (unsigned int)v269 + 1LL;
          if ( v142 > HIDWORD(v269) )
          {
            v245 = v132;
            sub_C8D5F0((__int64)&v268, v270, v142, 8u, v132, v110);
            v141 = (unsigned int)v269;
            v132 = v245;
          }
          *(_QWORD *)&v268[8 * v141] = v140;
          LODWORD(v269) = v269 + 1;
          v131 = *(v109 - 16);
        }
        if ( (v131 & 2) != 0 )
          v133 = (_QWORD *)*((_QWORD *)v109 - 4);
        else
          v133 = (_QWORD *)(v132 - 8LL * ((v131 >> 2) & 0xF));
      }
      v143 = v133[4];
      if ( v143 )
      {
        v144 = *(_BYTE *)(v143 - 16);
        if ( (v144 & 2) != 0 )
        {
          v145 = *(__int64 **)(v143 - 32);
          v146 = *(unsigned int *)(v143 - 24);
        }
        else
        {
          v145 = (__int64 *)(v143 - 16 - 8LL * ((v144 >> 2) & 0xF));
          v146 = (*(_WORD *)(v143 - 16) >> 6) & 0xF;
        }
        v147 = &v145[v146];
        if ( v145 != v147 )
        {
          v244 = v89;
          while ( 1 )
          {
            v169 = (_BYTE *)*v145;
            v170 = *(_BYTE *)*v145;
            if ( v170 == 10 )
              break;
            if ( v170 != 35 )
              goto LABEL_209;
            v171 = sub_AF29C0(*v145);
            v173 = v171 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v171 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v171 & 4) == 0 )
            {
              v174 = (unsigned int)v269;
              v175 = (unsigned int)v269 + 1LL;
              if ( v175 > HIDWORD(v269) )
              {
                v239 = v173;
                sub_C8D5F0((__int64)&v268, v270, v175, 8u, v172, v173);
                v174 = (unsigned int)v269;
                v173 = v239;
              }
              *(_QWORD *)&v268[8 * v174] = v173;
              LODWORD(v269) = v269 + 1;
            }
            v176 = sub_AF2A20((__int64)v169);
            v178 = v176 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v176 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v176 & 4) == 0 )
            {
              v179 = (unsigned int)v269;
              v180 = (unsigned int)v269 + 1LL;
              if ( v180 > HIDWORD(v269) )
              {
                v242 = v178;
                sub_C8D5F0((__int64)&v268, v270, v180, 8u, v177, v178);
                v179 = (unsigned int)v269;
                v178 = v242;
              }
              *(_QWORD *)&v268[8 * v179] = v178;
              LODWORD(v269) = v269 + 1;
            }
            v181 = sub_AF2A80((__int64)v169);
            v183 = v181 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v181 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v181 & 4) == 0 )
            {
              v184 = (unsigned int)v269;
              v185 = (unsigned int)v269 + 1LL;
              if ( v185 > HIDWORD(v269) )
              {
                v241 = v183;
                sub_C8D5F0((__int64)&v268, v270, v185, 8u, v182, v183);
                v184 = (unsigned int)v269;
                v183 = v241;
              }
              *(_QWORD *)&v268[8 * v184] = v183;
              LODWORD(v269) = v269 + 1;
            }
            v186 = sub_AF2AE0((__int64)v169);
            v188 = v186 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v186 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v186 & 4) == 0 )
            {
              v189 = (unsigned int)v269;
              v190 = (unsigned int)v269 + 1LL;
              if ( v190 > HIDWORD(v269) )
              {
                v240 = v188;
                sub_C8D5F0((__int64)&v268, v270, v190, 8u, v188, v187);
                v189 = (unsigned int)v269;
                v188 = v240;
              }
              ++v145;
              *(_QWORD *)&v268[8 * v189] = v188;
              LODWORD(v269) = v269 + 1;
              if ( v147 == v145 )
              {
LABEL_232:
                v89 = v244;
                goto LABEL_124;
              }
            }
            else
            {
LABEL_209:
              if ( v147 == ++v145 )
                goto LABEL_232;
            }
          }
          v148 = sub_AF2780(*v145);
          v150 = v148 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v148 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v148 >> 1) & 3) == 1 )
          {
            v151 = (unsigned int)v269;
            v152 = (unsigned int)v269 + 1LL;
            if ( v152 > HIDWORD(v269) )
            {
              v238 = v150;
              sub_C8D5F0((__int64)&v268, v270, v152, 8u, v149, v150);
              v151 = (unsigned int)v269;
              v150 = v238;
            }
            *(_QWORD *)&v268[8 * v151] = v150;
            LODWORD(v269) = v269 + 1;
          }
          v153 = sub_AF2800((__int64)v169);
          v155 = v153 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v153 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v153 >> 1) & 3) == 1 )
          {
            v156 = (unsigned int)v269;
            v157 = (unsigned int)v269 + 1LL;
            if ( v157 > HIDWORD(v269) )
            {
              v237 = v155;
              sub_C8D5F0((__int64)&v268, v270, v157, 8u, v154, v155);
              v156 = (unsigned int)v269;
              v155 = v237;
            }
            *(_QWORD *)&v268[8 * v156] = v155;
            LODWORD(v269) = v269 + 1;
          }
          v158 = sub_AF2880((__int64)v169);
          v160 = v158 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v158 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v158 >> 1) & 3) == 1 )
          {
            v161 = (unsigned int)v269;
            v162 = (unsigned int)v269 + 1LL;
            if ( v162 > HIDWORD(v269) )
            {
              v236 = v160;
              sub_C8D5F0((__int64)&v268, v270, v162, 8u, v159, v160);
              v161 = (unsigned int)v269;
              v160 = v236;
            }
            *(_QWORD *)&v268[8 * v161] = v160;
            LODWORD(v269) = v269 + 1;
          }
          v163 = sub_AF2900((__int64)v169);
          v166 = v163 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v163 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v163 >> 1) & 3) == 1 )
          {
            v167 = (unsigned int)v269;
            v168 = (unsigned int)v269 + 1LL;
            if ( v168 > HIDWORD(v269) )
            {
              sub_C8D5F0((__int64)&v268, v270, v168, 8u, v164, v165);
              v167 = (unsigned int)v269;
            }
            *(_QWORD *)&v268[8 * v167] = v166;
            LODWORD(v269) = v269 + 1;
          }
          goto LABEL_209;
        }
      }
LABEL_124:
      v111 = v268;
      v112 = &v268[8 * (unsigned int)v269];
      if ( v268 == v112 )
      {
LABEL_138:
        if ( v112 != v270 )
          _libc_free((unsigned __int64)v112);
        v48 = v276;
        goto LABEL_109;
      }
      while ( 2 )
      {
        v122 = (_BYTE *)*v111;
        if ( *(_BYTE *)*v111 == 26 )
        {
          if ( (v279 & 1) != 0 )
          {
            v113 = (__int64 *)&v280;
            v114 = 3;
LABEL_127:
            v115 = v114 & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
            v116 = &v113[2 * v115];
            v117 = *v116;
            if ( v122 == (_BYTE *)*v116 )
            {
LABEL_128:
              v118 = v116[1];
              if ( v118 )
              {
                v119 = (unsigned int)v276;
                v120 = v118 & 0xFFFFFFFFFFFFFFFBLL;
                v121 = (unsigned int)v276 + 1LL;
                if ( v121 > HIDWORD(v276) )
                {
                  v243 = v120;
                  sub_C8D5F0((__int64)&v275, v277, (unsigned int)v276 + 1LL, 8u, v121, v117);
                  v119 = (unsigned int)v276;
                  v120 = v243;
                }
                *(_QWORD *)&v275[8 * v119] = v120;
                LODWORD(v276) = v276 + 1;
              }
            }
            else
            {
              v123 = 1;
              while ( v117 != -4096 )
              {
                v124 = v123 + 1;
                v115 = v114 & (v123 + v115);
                v116 = &v113[2 * v115];
                v117 = *v116;
                if ( v122 == (_BYTE *)*v116 )
                  goto LABEL_128;
                v123 = v124;
              }
            }
          }
          else
          {
            v113 = v280;
            if ( v281 )
            {
              v114 = v281 - 1;
              goto LABEL_127;
            }
          }
        }
        if ( v112 == (_BYTE *)++v111 )
        {
          v112 = v268;
          goto LABEL_138;
        }
        continue;
      }
    }
LABEL_58:
    a3 = v256;
  }
LABEL_59:
  if ( (v287[1] & 1) == 0 )
    sub_C7D6A0(v287[2], 8LL * LODWORD(v287[3]), 8);
  if ( (v283 & 1) == 0 )
    sub_C7D6A0((__int64)v284, 8LL * v285, 8);
  if ( (v279 & 1) == 0 )
    sub_C7D6A0((__int64)v280, 16LL * v281, 8);
  if ( v275 != v277 )
    _libc_free((unsigned __int64)v275);
  if ( &v272[(unsigned int)v273] != v272 )
  {
    v262 = &v272[(unsigned int)v273];
    v54 = v272;
    do
    {
      v55 = sub_373B0C0((__int64 *)a1, *v54, a2, &v266);
      *(_QWORD *)(v55 + 40) = a3 & 0xFFFFFFFFFFFFFFFBLL;
      v56 = *(_QWORD **)(a3 + 32);
      if ( v56 )
      {
        *(_QWORD *)v55 = *v56;
        **(_QWORD **)(a3 + 32) = v55 & 0xFFFFFFFFFFFFFFFBLL;
      }
      *(_QWORD *)(a3 + 32) = v55;
      ++v54;
    }
    while ( v262 != v54 );
  }
  sub_3247E30(a1, a3, *(_QWORD *)(a2 + 8));
  v59 = *(_QWORD *)(a1 + 216);
  v60 = *(_QWORD *)(v59 + 376);
  v61 = *(_DWORD *)(v59 + 392);
  if ( v61 )
  {
    v62 = v61 - 1;
    v63 = (v61 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v64 = (__int64 *)(v60 + 56LL * v63);
    v65 = *v64;
    if ( a2 == *v64 )
    {
LABEL_74:
      v287[0] = (__int64)&v287[2];
      v287[1] = 0x400000000LL;
      v66 = *((_DWORD *)v64 + 4);
      if ( v66 && v287 != v64 + 1 )
      {
        v204 = &v287[2];
        v205 = 8LL * v66;
        if ( v66 <= 4
          || (sub_C8D5F0((__int64)v287, &v287[2], v66, 8u, v57, v58),
              v204 = (__int64 *)v287[0],
              (v205 = 8LL * *((unsigned int *)v64 + 4)) != 0) )
        {
          memcpy(v204, (const void *)v64[1], v205);
          v204 = (__int64 *)v287[0];
        }
        LODWORD(v287[1]) = v66;
        v206 = a3;
        v264 = &v204[v66];
        v207 = a3;
        v208 = v204;
        v209 = v206 & 0xFFFFFFFFFFFFFFFBLL;
        do
        {
          v210 = sub_373B120((__int64 *)a1, *v208, a2);
          *(_QWORD *)(v210 + 40) = v209;
          v211 = *(_QWORD **)(v207 + 32);
          if ( v211 )
          {
            *(_QWORD *)v210 = *v211;
            **(_QWORD **)(v207 + 32) = v210 & 0xFFFFFFFFFFFFFFFBLL;
          }
          *(_QWORD *)(v207 + 32) = v210;
          ++v208;
        }
        while ( v264 != v208 );
        a3 = v207;
        if ( (__int64 *)v287[0] != &v287[2] )
          _libc_free(v287[0]);
      }
    }
    else
    {
      v221 = 1;
      while ( v65 != -4096 )
      {
        v57 = (unsigned int)(v221 + 1);
        v63 = v62 & (v221 + v63);
        v64 = (__int64 *)(v60 + 56LL * v63);
        v65 = *v64;
        if ( a2 == *v64 )
          goto LABEL_74;
        ++v221;
      }
    }
  }
  if ( !sub_3736590((_QWORD *)a1) && !*(_QWORD *)(a2 + 16) )
  {
    v265 = *(_QWORD *)(a1 + 208);
    v287[0] = *(_QWORD *)(a2 + 8);
    v212 = sub_373B590(v265 + 2968, v287);
    v215 = (__int64 *)v212[6];
    v216 = &v215[*((unsigned int *)v212 + 14)];
    if ( v215 != v216 )
    {
      v217 = (__int64 *)v212[6];
      v218 = *v217;
      if ( !*(_BYTE *)(a1 + 556) )
        goto LABEL_279;
LABEL_273:
      v219 = *(__int64 **)(a1 + 536);
      v214 = *(unsigned int *)(a1 + 548);
      v213 = &v219[v214];
      if ( v219 == v213 )
      {
LABEL_284:
        if ( (unsigned int)v214 >= *(_DWORD *)(a1 + 544) )
          goto LABEL_279;
        *(_DWORD *)(a1 + 548) = v214 + 1;
        *v213 = v218;
        ++*(_QWORD *)(a1 + 528);
LABEL_280:
        v220 = *(unsigned int *)(a1 + 600);
        v214 = *(unsigned int *)(a1 + 604);
        v68 = *v217;
        if ( v220 + 1 > v214 )
        {
          v261 = *v217;
          sub_C8D5F0(a1 + 592, (const void *)(a1 + 608), v220 + 1, 8u, v67, v68);
          v220 = *(unsigned int *)(a1 + 600);
          v68 = v261;
        }
        v213 = *(__int64 **)(a1 + 592);
        ++v217;
        v213[v220] = v68;
        ++*(_DWORD *)(a1 + 600);
        if ( v217 != v216 )
          goto LABEL_278;
      }
      else
      {
        while ( v218 != *v219 )
        {
          if ( v213 == ++v219 )
            goto LABEL_284;
        }
        while ( ++v217 != v216 )
        {
LABEL_278:
          v218 = *v217;
          if ( *(_BYTE *)(a1 + 556) )
            goto LABEL_273;
LABEL_279:
          sub_C8CC70(a1 + 528, v218, (__int64)v213, v214, v67, v68);
          if ( (_BYTE)v213 )
            goto LABEL_280;
        }
      }
    }
  }
  v69 = *(_QWORD *)(a2 + 32);
  v263 = v69 + 8LL * *(unsigned int *)(a2 + 40);
  if ( v263 != v69 )
  {
    v70 = (_QWORD *)a1;
    v71 = *(_QWORD *)(a2 + 32);
    while ( 1 )
    {
      v72 = *(_QWORD **)v71;
      if ( **(_BYTE **)(*(_QWORD *)v71 + 8LL) != 18 )
        break;
LABEL_91:
      sub_373DD80((__int64)v70, v72, a3);
LABEL_92:
      v71 += 8;
      if ( v263 == v71 )
        goto LABEL_93;
    }
    v73 = v70[27];
    v74 = *(_QWORD *)(v73 + 344);
    v75 = *(_DWORD *)(v73 + 360);
    if ( v75 )
    {
      v76 = v75 - 1;
      v77 = (v75 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      v78 = v74 + 136LL * v77;
      v79 = *(_QWORD *)v78;
      if ( v72 == *(_QWORD **)v78 )
      {
LABEL_82:
        LODWORD(v287[1]) = 0;
        v287[2] = 0;
        v287[3] = (__int64)&v287[1];
        v287[4] = (__int64)&v287[1];
        v287[5] = 0;
        v80 = *(const __m128i **)(v78 + 24);
        if ( v80 )
        {
          v257 = v78;
          v81 = sub_3735420(v80, (__int64)&v287[1]);
          v78 = v257;
          v82 = (__int64)v81;
          do
          {
            v83 = (__int64)v81;
            v81 = (__m128i *)v81[1].m128i_i64[0];
          }
          while ( v81 );
          v287[3] = v83;
          v84 = v82;
          do
          {
            v79 = v84;
            v84 = *(_QWORD *)(v84 + 24);
          }
          while ( v84 );
          v287[4] = v79;
          v85 = *(_QWORD *)(v257 + 48);
          v287[2] = v82;
          v287[5] = v85;
        }
        v287[6] = (__int64)&v287[8];
        v287[7] = 0x800000000LL;
        if ( *(_DWORD *)(v78 + 64) )
        {
          sub_37351E0((__int64)&v287[6], v78 + 56, v79, v78, v67, v68);
          if ( !v287[5] && !LODWORD(v287[7]) )
            goto LABEL_101;
        }
        else
        {
          if ( v287[5] )
          {
LABEL_90:
            sub_37356F0(v287[2]);
            goto LABEL_91;
          }
LABEL_101:
          if ( sub_3736590(v70) )
            goto LABEL_102;
          v191 = v70[26];
          v192 = v72[1];
          v193 = *(_DWORD *)(v191 + 2992);
          v194 = v191 + 2968;
          if ( !v193 )
          {
            ++*(_QWORD *)(v191 + 2968);
            goto LABEL_297;
          }
          v195 = *(_QWORD *)(v191 + 2976);
          v258 = ((unsigned int)v192 >> 9) ^ ((unsigned int)v192 >> 4);
          v196 = (v193 - 1) & v258;
          v197 = (__int64 *)(v195 + 88LL * v196);
          v198 = *v197;
          if ( v192 != *v197 )
          {
            v249 = 1;
            v253 = 0;
            while ( v198 != -4096 )
            {
              if ( !v253 )
              {
                if ( v198 != -8192 )
                  v197 = 0;
                v253 = v197;
              }
              v196 = (v193 - 1) & (v249 + v196);
              v197 = (__int64 *)(v195 + 88LL * v196);
              v198 = *v197;
              if ( v192 == *v197 )
                goto LABEL_235;
              ++v249;
            }
            v194 = v191 + 2968;
            if ( v253 )
              v197 = v253;
            v228 = *(_DWORD *)(v191 + 2984);
            ++*(_QWORD *)(v191 + 2968);
            v227 = v228 + 1;
            if ( 4 * (v228 + 1) >= 3 * v193 )
            {
LABEL_297:
              v259 = v191;
              v252 = v192;
              sub_3227C70(v194, 2 * v193);
              v191 = v259;
              v223 = *(_DWORD *)(v259 + 2992);
              if ( !v223 )
                goto LABEL_347;
              v192 = v252;
              v224 = *(_QWORD *)(v259 + 2976);
              v260 = v223 - 1;
              v225 = (v223 - 1) & (((unsigned int)v252 >> 9) ^ ((unsigned int)v252 >> 4));
              v197 = (__int64 *)(v224 + 88LL * v225);
              v226 = *v197;
              v227 = *(_DWORD *)(v191 + 2984) + 1;
              if ( *v197 != v252 )
              {
                v235 = 1;
                v233 = 0;
                while ( v226 != -4096 )
                {
                  if ( v226 == -8192 && !v233 )
                    v233 = v197;
                  v225 = v260 & (v235 + v225);
                  v197 = (__int64 *)(v224 + 88LL * v225);
                  v226 = *v197;
                  if ( v252 == *v197 )
                    goto LABEL_299;
                  ++v235;
                }
LABEL_317:
                if ( v233 )
                  v197 = v233;
              }
            }
            else if ( v193 - *(_DWORD *)(v191 + 2988) - v227 <= v193 >> 3 )
            {
              v254 = v191;
              v247 = v192;
              sub_3227C70(v194, v193);
              v191 = v254;
              v229 = *(_DWORD *)(v254 + 2992);
              if ( !v229 )
              {
LABEL_347:
                ++*(_DWORD *)(v191 + 2984);
                BUG();
              }
              v230 = 1;
              v250 = v229 - 1;
              v231 = (v229 - 1) & v258;
              v255 = *(_QWORD *)(v254 + 2976);
              v197 = (__int64 *)(v255 + 88LL * v231);
              v192 = v247;
              v232 = *v197;
              v227 = *(_DWORD *)(v191 + 2984) + 1;
              v233 = 0;
              if ( *v197 != v247 )
              {
                while ( v232 != -4096 )
                {
                  if ( !v233 && v232 == -8192 )
                    v233 = v197;
                  v231 = v250 & (v230 + v231);
                  v197 = (__int64 *)(v255 + 88LL * v231);
                  v232 = *v197;
                  if ( v247 == *v197 )
                    goto LABEL_299;
                  ++v230;
                }
                goto LABEL_317;
              }
            }
LABEL_299:
            *(_DWORD *)(v191 + 2984) = v227;
            if ( *v197 != -4096 )
              --*(_DWORD *)(v191 + 2988);
            *v197 = v192;
            *(_OWORD *)(v197 + 1) = 0;
            v197[2] = (__int64)(v197 + 5);
            *(_OWORD *)(v197 + 3) = 0;
            v197[7] = (__int64)(v197 + 9);
            v197[3] = 2;
            *((_DWORD *)v197 + 8) = 0;
            *((_BYTE *)v197 + 36) = 1;
            v197[8] = 0x200000000LL;
            *(_OWORD *)(v197 + 5) = 0;
            *(_OWORD *)(v197 + 9) = 0;
LABEL_102:
            if ( (__int64 *)v287[6] != &v287[8] )
              _libc_free(v287[6]);
            sub_37356F0(v287[2]);
            sub_373DE40(v70, v72);
            goto LABEL_92;
          }
LABEL_235:
          if ( !*((_DWORD *)v197 + 16) )
            goto LABEL_102;
        }
        if ( (__int64 *)v287[6] != &v287[8] )
        {
          _libc_free(v287[6]);
          sub_37356F0(v287[2]);
          goto LABEL_91;
        }
        goto LABEL_90;
      }
      v88 = 1;
      while ( v79 != -4096 )
      {
        v67 = (unsigned int)(v88 + 1);
        v77 = v76 & (v88 + v77);
        v78 = v74 + 136LL * v77;
        v79 = *(_QWORD *)v78;
        if ( v72 == *(_QWORD **)v78 )
          goto LABEL_82;
        v88 = v67;
      }
    }
    memset(v287, 0, 0x80u);
    HIDWORD(v287[7]) = 8;
    v287[3] = (__int64)&v287[1];
    v287[4] = (__int64)&v287[1];
    v287[6] = (__int64)&v287[8];
    goto LABEL_101;
  }
LABEL_93:
  v86 = v266;
  if ( v272 != (__int64 *)v274 )
    _libc_free((unsigned __int64)v272);
  if ( (unsigned __int64 *)v286[6] != &v286[8] )
    _libc_free(v286[6]);
  sub_37356F0(v286[2]);
  return v86;
}
