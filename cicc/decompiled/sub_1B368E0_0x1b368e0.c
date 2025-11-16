// Function: sub_1B368E0
// Address: 0x1b368e0
//
__int64 *__fastcall sub_1B368E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 **a5,
        __int64 *a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  unsigned int v14; // r12d
  __int64 v15; // rsi
  int v16; // r8d
  __int64 v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  int v21; // r12d
  unsigned __int64 v22; // r13
  unsigned int i; // r14d
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // r9
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r12
  int v31; // ebx
  __int64 v32; // rcx
  __int64 v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rdi
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r15
  int v42; // eax
  __int64 v43; // rax
  int v44; // edx
  __int64 *v45; // r12
  unsigned __int64 v46; // rsi
  __int64 *v47; // rcx
  __int64 v48; // r12
  __int64 *v49; // rbx
  __int64 v50; // rdi
  int v51; // r12d
  int v52; // r12d
  __int64 v53; // r10
  unsigned int v54; // esi
  __int64 v55; // rcx
  __int64 v56; // rdi
  __int64 *result; // rax
  char v58; // dl
  __int64 v59; // r13
  __int16 *v60; // rbx
  __int64 v61; // rax
  __int64 *v62; // r14
  int v63; // edx
  __int16 *v64; // r12
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rsi
  int v68; // r11d
  __int64 v69; // rcx
  __int64 *v70; // r15
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 *v75; // rdi
  __int64 v76; // rsi
  __int64 *v77; // rax
  unsigned __int64 v78; // rcx
  unsigned __int64 *v79; // rcx
  unsigned __int64 v80; // rdx
  double v81; // xmm4_8
  double v82; // xmm5_8
  unsigned int v83; // esi
  unsigned int v84; // r9d
  __int64 v85; // rdi
  unsigned int v86; // r15d
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 *v89; // rcx
  __int64 v90; // r15
  __int64 *v91; // r15
  unsigned __int64 v92; // rdx
  __int64 v93; // rbx
  __int64 *v94; // r12
  __int64 v95; // rdi
  __int64 v96; // rbx
  __int64 v97; // rsi
  __int64 *v98; // rax
  unsigned int v99; // r13d
  char v100; // dl
  __int64 v101; // rsi
  __int64 *v102; // rax
  __int64 *v103; // rdi
  __int64 *v104; // rcx
  __int64 v105; // rsi
  _QWORD *v106; // rcx
  _QWORD *v107; // r15
  __int64 v108; // rdx
  __int64 v109; // rdi
  unsigned int v110; // esi
  __int64 *v111; // rcx
  __int64 v112; // r10
  __int64 v113; // r15
  __int64 v114; // r14
  __int64 v115; // rdx
  __int64 v116; // rax
  bool v117; // cf
  unsigned __int64 v118; // rax
  __int64 v119; // rbx
  __int64 v120; // rbx
  __int64 v121; // r12
  _QWORD *v122; // r12
  _QWORD *v123; // r13
  _QWORD *v124; // r14
  __int64 v125; // rdx
  __int64 v126; // rdi
  __int64 *v127; // rbx
  __int64 *v128; // r15
  __int64 v129; // rsi
  __int64 *v130; // rsi
  unsigned int v131; // edi
  __int64 *v132; // rcx
  __int64 *v133; // rdi
  __int64 *v134; // rcx
  __int16 *v135; // rsi
  __int64 v136; // rdx
  unsigned __int64 v137; // rcx
  __int64 v138; // rdx
  __int64 v139; // rdx
  unsigned int v140; // esi
  __int64 v141; // rdi
  unsigned int v142; // ecx
  __int64 *v143; // rax
  __int64 v144; // r10
  __int64 v145; // rcx
  unsigned __int64 v146; // rsi
  __int64 v147; // rax
  __int64 v148; // rcx
  __int64 v149; // rdx
  __int16 *v150; // rbx
  __int64 *v151; // rdx
  __int64 v152; // rdi
  unsigned int v153; // esi
  __int64 v154; // r8
  unsigned int v155; // ecx
  __int64 *v156; // rax
  __int64 v157; // r11
  __int64 v158; // rsi
  unsigned __int64 v159; // rdi
  __int64 v160; // rax
  __int64 v161; // rsi
  int v162; // r11d
  int v163; // r11d
  int v164; // r11d
  int v165; // edi
  __int64 v166; // rsi
  unsigned int v167; // r15d
  __int64 v168; // rcx
  __int64 *v169; // rbx
  __int64 *v170; // r12
  __int64 v171; // rdi
  __int64 *v172; // rax
  __int64 v173; // rdx
  unsigned __int64 v174; // rdx
  __int64 v175; // rcx
  __int64 v176; // r15
  __int64 *v177; // r13
  __int64 *v178; // r12
  __int64 v179; // rdi
  __int64 v180; // r15
  unsigned __int64 *v181; // rcx
  unsigned __int64 v182; // rdx
  double v183; // xmm4_8
  double v184; // xmm5_8
  unsigned __int64 *v185; // rcx
  unsigned __int64 v186; // rdx
  double v187; // xmm4_8
  double v188; // xmm5_8
  int v189; // r15d
  __int64 v190; // rax
  __int64 v191; // rdx
  __int64 v192; // rax
  __int64 v193; // rdx
  int v194; // eax
  int v195; // r15d
  __int64 j; // r8
  __int64 *v197; // rax
  __int64 v198; // rdx
  unsigned int v199; // edi
  __int64 v200; // r9
  unsigned int v201; // esi
  __int64 *v202; // rcx
  __int64 v203; // r10
  __int64 v204; // rsi
  unsigned __int64 v205; // rdi
  __int64 v206; // rdx
  __int64 v207; // rsi
  int v208; // ecx
  int v209; // r8d
  __int64 v210; // rcx
  __int64 *v211; // r11
  __int64 v212; // r8
  int k; // r10d
  int v214; // ebx
  __int64 v215; // r10
  int v216; // ecx
  int v217; // edx
  __int64 v218; // r10
  int v219; // ecx
  int v220; // edi
  int v221; // edi
  __int64 v222; // r10
  unsigned int v223; // r15d
  __int64 *v224; // r9
  int v225; // esi
  __int64 v226; // rcx
  int v227; // r9d
  int v228; // r9d
  __int64 v229; // r10
  int v230; // esi
  unsigned int v231; // r15d
  __int64 *v232; // rdi
  int v233; // eax
  int v234; // r9d
  int v235; // eax
  int v236; // r10d
  __int64 v237; // [rsp+0h] [rbp-100h]
  __int64 v238; // [rsp+0h] [rbp-100h]
  int v240; // [rsp+10h] [rbp-F0h]
  __int64 v241; // [rsp+10h] [rbp-F0h]
  unsigned int v243; // [rsp+18h] [rbp-E8h]
  int v246; // [rsp+30h] [rbp-D0h]
  __int64 v247; // [rsp+30h] [rbp-D0h]
  __int16 *v248; // [rsp+30h] [rbp-D0h]
  __int16 *v249; // [rsp+30h] [rbp-D0h]
  __int16 *v250; // [rsp+38h] [rbp-C8h]
  _QWORD *v251; // [rsp+38h] [rbp-C8h]
  __int16 *v252; // [rsp+38h] [rbp-C8h]
  int v253; // [rsp+38h] [rbp-C8h]
  int v254; // [rsp+38h] [rbp-C8h]
  __int64 v255; // [rsp+40h] [rbp-C0h]
  __int64 v256; // [rsp+40h] [rbp-C0h]
  int v257; // [rsp+40h] [rbp-C0h]
  __int64 v259; // [rsp+50h] [rbp-B0h]
  _QWORD *v260; // [rsp+50h] [rbp-B0h]
  __int64 v261; // [rsp+50h] [rbp-B0h]
  __int64 v263; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v264; // [rsp+68h] [rbp-98h]
  __int64 *v265; // [rsp+70h] [rbp-90h]
  __int64 v266; // [rsp+78h] [rbp-88h]
  int v267; // [rsp+80h] [rbp-80h]
  _BYTE v268[120]; // [rsp+88h] [rbp-78h] BYREF

  while ( 2 )
  {
    v259 = *(_QWORD *)(a2 + 48);
    if ( !v259 )
LABEL_151:
      BUG();
    if ( *(_BYTE *)(v259 - 8) == 77 )
    {
      v14 = *(_DWORD *)(a1 + 640);
      if ( v14 )
      {
        v15 = *(_QWORD *)(a1 + 624);
        v16 = 1;
        v17 = v259 - 24;
        v18 = (v14 - 1) & (((unsigned int)(v259 - 24) >> 9) ^ ((unsigned int)(v259 - 24) >> 4));
        v19 = *(_QWORD *)(v15 + 16LL * v18);
        if ( v259 - 24 == v19 )
        {
LABEL_5:
          v240 = *(_DWORD *)(v259 - 4) & 0xFFFFFFF;
          v20 = sub_157EBA0(a3);
          v246 = 0;
          if ( v20 )
          {
            v21 = 0;
            v246 = sub_15F4D60(v20);
            v22 = sub_157EBA0(a3);
            if ( v246 )
            {
              for ( i = 0; i != v246; ++i )
              {
                v24 = sub_15F4DF0(v22, i);
                v21 += a2 == v24;
              }
              v246 = v21;
            }
            v259 = *(_QWORD *)(a2 + 48);
            v15 = *(_QWORD *)(a1 + 624);
            v14 = *(_DWORD *)(a1 + 640);
          }
          v25 = v17;
          v237 = a1 + 616;
          if ( v14 )
          {
LABEL_12:
            v26 = v14 - 1;
            LODWORD(v27) = v26 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v28 = v15 + 16LL * (unsigned int)v27;
            v29 = *(_QWORD *)v28;
            if ( *(_QWORD *)v28 == v25 )
            {
              v30 = 8LL * *(unsigned int *)(v28 + 8);
              goto LABEL_14;
            }
            v162 = 1;
            v55 = 0;
            while ( v29 != -8 )
            {
              if ( v29 != -16 || v55 )
                v28 = v55;
              v27 = (unsigned int)v26 & ((_DWORD)v27 + v162);
              v218 = v15 + 16 * v27;
              v29 = *(_QWORD *)v218;
              if ( *(_QWORD *)v218 == v25 )
              {
                v30 = 8LL * *(unsigned int *)(v218 + 8);
                goto LABEL_14;
              }
              v55 = v28;
              ++v162;
              v28 = v15 + 16 * v27;
            }
            if ( !v55 )
              v55 = v28;
            ++*(_QWORD *)(a1 + 616);
            v28 = (unsigned int)(*(_DWORD *)(a1 + 632) + 1);
            if ( 4 * (int)v28 >= 3 * v14 )
              goto LABEL_43;
            if ( v14 - ((_DWORD)v28 + *(_DWORD *)(a1 + 636)) > v14 >> 3 )
              goto LABEL_207;
            sub_1A72540(v237, v14);
            v163 = *(_DWORD *)(a1 + 640);
            if ( v163 )
            {
              v164 = v163 - 1;
              v26 = *(_QWORD *)(a1 + 624);
              v165 = 1;
              v166 = 0;
              v167 = v164 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v28 = (unsigned int)(*(_DWORD *)(a1 + 632) + 1);
              v55 = v26 + 16LL * v167;
              v29 = *(_QWORD *)v55;
              if ( *(_QWORD *)v55 == v25 )
                goto LABEL_207;
              while ( 1 )
              {
                if ( v29 == -8 )
                {
                  if ( v166 )
                    v55 = v166;
                  goto LABEL_207;
                }
                if ( v29 == -16 && !v166 )
                  v166 = v55;
                v167 = v164 & (v165 + v167);
                v55 = v26 + 16LL * v167;
                v29 = *(_QWORD *)v55;
                if ( *(_QWORD *)v55 == v25 )
                  goto LABEL_207;
                ++v165;
              }
            }
LABEL_341:
            ++*(_DWORD *)(a1 + 632);
            BUG();
          }
          while ( 1 )
          {
            ++*(_QWORD *)(a1 + 616);
LABEL_43:
            sub_1A72540(v237, 2 * v14);
            v51 = *(_DWORD *)(a1 + 640);
            if ( !v51 )
              goto LABEL_341;
            v52 = v51 - 1;
            v53 = *(_QWORD *)(a1 + 624);
            v54 = v52 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v28 = (unsigned int)(*(_DWORD *)(a1 + 632) + 1);
            v55 = v53 + 16LL * v54;
            v26 = *(_QWORD *)v55;
            if ( *(_QWORD *)v55 != v25 )
            {
              v29 = 1;
              v56 = 0;
              while ( v26 != -8 )
              {
                if ( !v56 && v26 == -16 )
                  v56 = v55;
                v54 = v52 & (v29 + v54);
                v55 = v53 + 16LL * v54;
                v26 = *(_QWORD *)v55;
                if ( *(_QWORD *)v55 == v25 )
                  goto LABEL_207;
                v29 = (unsigned int)(v29 + 1);
              }
              if ( v56 )
                v55 = v56;
            }
LABEL_207:
            *(_DWORD *)(a1 + 632) = v28;
            if ( *(_QWORD *)v55 != -8 )
              --*(_DWORD *)(a1 + 636);
            *(_QWORD *)v55 = v25;
            v30 = 0;
            *(_DWORD *)(v55 + 8) = 0;
LABEL_14:
            v31 = 0;
            if ( v246 )
            {
              v32 = a3;
              do
              {
                v41 = *(_QWORD *)(*a4 + v30);
                v42 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
                if ( v42 == *(_DWORD *)(v25 + 56) )
                {
                  v255 = v32;
                  sub_15F55D0(v25, (__int64)a4, v28, v32, v29, v26);
                  v32 = v255;
                  v42 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
                }
                v43 = (v42 + 1) & 0xFFFFFFF;
                v44 = v43 | *(_DWORD *)(v25 + 20) & 0xF0000000;
                *(_DWORD *)(v25 + 20) = v44;
                if ( (v44 & 0x40000000) != 0 )
                  v33 = *(_QWORD *)(v25 - 8);
                else
                  v33 = v25 - 24 * v43;
                v34 = (_QWORD *)(v33 + 24LL * (unsigned int)(v43 - 1));
                if ( *v34 )
                {
                  v35 = v34[1];
                  v36 = v34[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v36 = v35;
                  if ( v35 )
                  {
                    v29 = *(_QWORD *)(v35 + 16) & 3LL;
                    *(_QWORD *)(v35 + 16) = v29 | v36;
                  }
                }
                *v34 = v41;
                if ( v41 )
                {
                  v37 = *(_QWORD *)(v41 + 8);
                  v29 = v41 + 8;
                  v34[1] = v37;
                  if ( v37 )
                  {
                    v26 = (__int64)(v34 + 1);
                    *(_QWORD *)(v37 + 16) = (unsigned __int64)(v34 + 1) | *(_QWORD *)(v37 + 16) & 3LL;
                  }
                  v34[2] = v29 | v34[2] & 3LL;
                  *(_QWORD *)(v41 + 8) = v34;
                }
                v38 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
                v39 = (unsigned int)(v38 - 1);
                if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
                  v40 = *(_QWORD *)(v25 - 8);
                else
                  v40 = v25 - 24 * v38;
                ++v31;
                v28 = 3LL * *(unsigned int *)(v25 + 56);
                *(_QWORD *)(v40 + 8 * v39 + 24LL * *(unsigned int *)(v25 + 56) + 8) = v32;
              }
              while ( v31 != v246 );
            }
            *(_QWORD *)(*a4 + v30) = v25;
            v45 = (__int64 *)(*(_QWORD *)(a1 + 648) + v30);
            v46 = *v45 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v45 & 4) != 0 )
            {
              v47 = *(__int64 **)v46;
              v48 = *(_QWORD *)v46 + 8LL * *(unsigned int *)(v46 + 8);
            }
            else
            {
              v47 = v45;
              if ( !v46 )
                goto LABEL_38;
              v48 = (__int64)(v45 + 1);
            }
            if ( (__int64 *)v48 != v47 )
            {
              v49 = v47;
              do
              {
                v50 = *v49++;
                sub_1AEA2D0(v50, (__int64 *)v25, (_QWORD *)(a1 + 32));
              }
              while ( (__int64 *)v48 != v49 );
            }
LABEL_38:
            v259 = *(_QWORD *)(v259 + 8);
            if ( !v259 )
              goto LABEL_151;
            v25 = v259 - 24;
            if ( *(_BYTE *)(v259 - 8) != 77 || v240 != (*(_DWORD *)(v259 - 4) & 0xFFFFFFF) )
              goto LABEL_53;
            v14 = *(_DWORD *)(a1 + 640);
            v15 = *(_QWORD *)(a1 + 624);
            if ( v14 )
              goto LABEL_12;
          }
        }
        while ( v19 != -8 )
        {
          v18 = (v14 - 1) & (v16 + v18);
          v19 = *(_QWORD *)(v15 + 16LL * v18);
          if ( v17 == v19 )
            goto LABEL_5;
          ++v16;
        }
      }
    }
LABEL_53:
    result = *(__int64 **)(a1 + 736);
    if ( *(__int64 **)(a1 + 744) != result )
      goto LABEL_54;
    v130 = &result[*(unsigned int *)(a1 + 756)];
    v131 = *(_DWORD *)(a1 + 756);
    if ( result == v130 )
      goto LABEL_233;
    v132 = 0;
    do
    {
      if ( a2 == *result )
        return result;
      if ( *result == -2 )
        v132 = result;
      ++result;
    }
    while ( v130 != result );
    if ( !v132 )
    {
LABEL_233:
      if ( v131 >= *(_DWORD *)(a1 + 752) )
      {
LABEL_54:
        result = sub_16CCBA0(a1 + 728, a2);
        if ( !v58 )
          return result;
        goto LABEL_55;
      }
      *(_DWORD *)(a1 + 756) = v131 + 1;
      *v130 = a2;
      ++*(_QWORD *)(a1 + 728);
    }
    else
    {
      *v132 = a2;
      --*(_DWORD *)(a1 + 760);
      ++*(_QWORD *)(a1 + 728);
    }
LABEL_55:
    v59 = a1;
    v60 = *(__int16 **)(a2 + 48);
    v260 = (_QWORD *)(a1 + 32);
    v256 = a1 + 616;
    while ( 1 )
    {
      if ( !v60 )
        BUG();
      v62 = (__int64 *)(v60 - 12);
      v63 = *((unsigned __int8 *)v60 - 8);
      if ( (unsigned int)(v63 - 25) <= 9 )
        break;
      v64 = (__int16 *)*((_QWORD *)v60 + 1);
      if ( (_BYTE)v63 == 54 )
      {
        v61 = *((_QWORD *)v60 - 6);
        if ( *(_BYTE *)(v61 + 16) != 53 )
          goto LABEL_57;
        v108 = *(unsigned int *)(v59 + 576);
        if ( !(_DWORD)v108 )
          goto LABEL_57;
        v109 = *(_QWORD *)(v59 + 560);
        v110 = (v108 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
        v111 = (__int64 *)(v109 + 16LL * v110);
        v112 = *v111;
        if ( v61 == *v111 )
        {
LABEL_115:
          if ( v111 == (__int64 *)(v109 + 16 * v108) )
            goto LABEL_57;
          v113 = *(_QWORD *)(*a4 + 8LL * *((unsigned int *)v111 + 2));
          if ( !*(_QWORD *)(v59 + 496) || !*((_QWORD *)v60 + 3) && *(v60 - 3) >= 0 )
          {
            if ( !*(_BYTE *)(v59 + 544) )
              goto LABEL_122;
            v135 = v60 - 24;
            goto LABEL_172;
          }
          if ( sub_1625790((__int64)(v60 - 12), 11)
            && !(unsigned __int8)sub_14BFF20(
                                   v113,
                                   *(_QWORD *)(v59 + 504),
                                   0,
                                   *(_QWORD *)(v59 + 496),
                                   (__int64)(v60 - 12),
                                   *(_QWORD *)(v59 + 24)) )
          {
            sub_1B31B30(*(_QWORD *)(v59 + 496), (__int64 ***)v60 - 3);
          }
          if ( !*(_BYTE *)(v59 + 544) )
          {
LABEL_122:
            sub_164D160((__int64)(v60 - 12), v113, a7, a8, a9, a10, a11, a12, a13, a14);
            goto LABEL_74;
          }
          v135 = v60 - 24;
          if ( *((_QWORD *)v60 - 6) )
          {
LABEL_172:
            v136 = *((_QWORD *)v60 - 5);
            v137 = *((_QWORD *)v60 - 4) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v137 = v136;
            if ( v136 )
              *(_QWORD *)(v136 + 16) = v137 | *(_QWORD *)(v136 + 16) & 3LL;
          }
          *((_QWORD *)v60 - 6) = v113;
          if ( v113 )
          {
            v138 = *(_QWORD *)(v113 + 8);
            *((_QWORD *)v60 - 5) = v138;
            if ( v138 )
              *(_QWORD *)(v138 + 16) = (unsigned __int64)(v60 - 20) | *(_QWORD *)(v138 + 16) & 3LL;
            *((_QWORD *)v60 - 4) = (v113 + 8) | *((_QWORD *)v60 - 4) & 3LL;
            *(_QWORD *)(v113 + 8) = v135;
          }
          goto LABEL_57;
        }
        v208 = 1;
        while ( v112 != -8 )
        {
          v209 = v208 + 1;
          v210 = ((_DWORD)v108 - 1) & (v110 + v208);
          v110 = v210;
          v111 = (__int64 *)(v109 + 16 * v210);
          v112 = *v111;
          if ( v61 == *v111 )
            goto LABEL_115;
          v208 = v209;
        }
      }
      else
      {
        if ( (_BYTE)v63 != 55 )
        {
          if ( *(_BYTE *)(v59 + 544) )
          {
            switch ( (_BYTE)v63 )
            {
              case 'N':
                v189 = *((_DWORD *)v60 - 1) & 0xFFFFFFF;
                if ( *((char *)v60 - 1) < 0 )
                {
                  v190 = sub_1648A40((__int64)(v60 - 12));
                  if ( *((char *)v60 - 1) >= 0 )
                  {
                    if ( (unsigned int)((v190 + v191) >> 4) )
LABEL_338:
                      BUG();
                  }
                  else if ( (unsigned int)((v190 + v191 - sub_1648A40((__int64)(v60 - 12))) >> 4) )
                  {
                    if ( *((char *)v60 - 1) >= 0 )
                      goto LABEL_338;
                    v253 = *(_DWORD *)(sub_1648A40((__int64)(v60 - 12)) + 8);
                    if ( *((char *)v60 - 1) >= 0 )
                      BUG();
                    v192 = sub_1648A40((__int64)(v60 - 12));
                    v194 = *(_DWORD *)(v192 + v193 - 4) - v253;
                    goto LABEL_248;
                  }
                }
                v194 = 0;
LABEL_248:
                v195 = v189 - 1 - v194;
                if ( v195 )
                {
                  for ( j = 0; j != v195; ++j )
                  {
                    v197 = &v62[3 * (j - (*((_DWORD *)v60 - 1) & 0xFFFFFFF))];
                    v198 = *v197;
                    if ( *(_BYTE *)(*v197 + 16) == 53 )
                    {
                      v199 = *(_DWORD *)(v59 + 576);
                      if ( v199 )
                      {
                        v200 = *(_QWORD *)(v59 + 560);
                        v201 = (v199 - 1) & (((unsigned int)v198 >> 9) ^ ((unsigned int)v198 >> 4));
                        v202 = (__int64 *)(v200 + 16LL * v201);
                        v203 = *v202;
                        if ( v198 == *v202 )
                        {
LABEL_254:
                          if ( v202 != (__int64 *)(v200 + 16LL * v199) )
                          {
                            v204 = v197[1];
                            v205 = v197[2] & 0xFFFFFFFFFFFFFFFCLL;
                            v206 = *(_QWORD *)(*a4 + 8LL * *((unsigned int *)v202 + 2));
                            *(_QWORD *)v205 = v204;
                            if ( v204 )
                              *(_QWORD *)(v204 + 16) = v205 | *(_QWORD *)(v204 + 16) & 3LL;
                            *v197 = v206;
                            if ( v206 )
                            {
                              v207 = *(_QWORD *)(v206 + 8);
                              v197[1] = v207;
                              if ( v207 )
                                *(_QWORD *)(v207 + 16) = (unsigned __int64)(v197 + 1) | *(_QWORD *)(v207 + 16) & 3LL;
                              v197[2] = (v206 + 8) | v197[2] & 3;
                              *(_QWORD *)(v206 + 8) = v197;
                            }
                          }
                        }
                        else
                        {
                          v219 = 1;
                          while ( v203 != -8 )
                          {
                            v201 = (v199 - 1) & (v219 + v201);
                            v254 = v219 + 1;
                            v202 = (__int64 *)(v200 + 16LL * v201);
                            v203 = *v202;
                            if ( v198 == *v202 )
                              goto LABEL_254;
                            v219 = v254;
                          }
                        }
                      }
                    }
                  }
                }
                goto LABEL_57;
              case 'G':
                v139 = *((_QWORD *)v60 - 6);
                if ( *(_BYTE *)(v139 + 16) == 53 )
                {
                  v140 = *(_DWORD *)(v59 + 576);
                  if ( v140 )
                  {
                    v141 = *(_QWORD *)(v59 + 560);
                    v142 = (v140 - 1) & (((unsigned int)v139 >> 9) ^ ((unsigned int)v139 >> 4));
                    v143 = (__int64 *)(v141 + 16LL * v142);
                    v144 = *v143;
                    if ( v139 == *v143 )
                    {
LABEL_181:
                      if ( v143 != (__int64 *)(v141 + 16LL * v140) )
                      {
                        v145 = *((_QWORD *)v60 - 5);
                        v146 = *((_QWORD *)v60 - 4) & 0xFFFFFFFFFFFFFFFCLL;
                        v147 = *(_QWORD *)(*a4 + 8LL * *((unsigned int *)v143 + 2));
                        *(_QWORD *)v146 = v145;
                        if ( v145 )
                          *(_QWORD *)(v145 + 16) = v146 | *(_QWORD *)(v145 + 16) & 3LL;
                        *((_QWORD *)v60 - 6) = v147;
                        if ( v147 )
                        {
                          v148 = *(_QWORD *)(v147 + 8);
                          *((_QWORD *)v60 - 5) = v148;
                          if ( v148 )
                            *(_QWORD *)(v148 + 16) = (unsigned __int64)(v60 - 20) | *(_QWORD *)(v148 + 16) & 3LL;
                          v149 = *((_QWORD *)v60 - 4);
                          v150 = v60 - 24;
                          *((_QWORD *)v150 + 2) = (v147 + 8) | v149 & 3;
                          *(_QWORD *)(v147 + 8) = v150;
                        }
                      }
                    }
                    else
                    {
                      v233 = 1;
                      while ( v144 != -8 )
                      {
                        v234 = v233 + 1;
                        v142 = (v140 - 1) & (v233 + v142);
                        v143 = (__int64 *)(v141 + 16LL * v142);
                        v144 = *v143;
                        if ( v139 == *v143 )
                          goto LABEL_181;
                        v233 = v234;
                      }
                    }
                  }
                }
                goto LABEL_57;
              case '8':
                v151 = &v62[-3 * (*((_DWORD *)v60 - 1) & 0xFFFFFFF)];
                v152 = *v151;
                if ( *(_BYTE *)(*v151 + 16) == 53 )
                {
                  v153 = *(_DWORD *)(v59 + 576);
                  if ( v153 )
                  {
                    v154 = *(_QWORD *)(v59 + 560);
                    v155 = (v153 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
                    v156 = (__int64 *)(v154 + 16LL * v155);
                    v157 = *v156;
                    if ( v152 == *v156 )
                    {
LABEL_194:
                      if ( v156 != (__int64 *)(v154 + 16LL * v153) )
                      {
                        v158 = v151[1];
                        v159 = v151[2] & 0xFFFFFFFFFFFFFFFCLL;
                        v160 = *(_QWORD *)(*a4 + 8LL * *((unsigned int *)v156 + 2));
                        *(_QWORD *)v159 = v158;
                        if ( v158 )
                          *(_QWORD *)(v158 + 16) = v159 | *(_QWORD *)(v158 + 16) & 3LL;
                        *v151 = v160;
                        if ( v160 )
                        {
                          v161 = *(_QWORD *)(v160 + 8);
                          v151[1] = v161;
                          if ( v161 )
                            *(_QWORD *)(v161 + 16) = (unsigned __int64)(v151 + 1) | *(_QWORD *)(v161 + 16) & 3LL;
                          v151[2] = (v160 + 8) | v151[2] & 3;
                          *(_QWORD *)(v160 + 8) = v151;
                        }
                      }
                    }
                    else
                    {
                      v235 = 1;
                      while ( v157 != -8 )
                      {
                        v236 = v235 + 1;
                        v155 = (v153 - 1) & (v235 + v155);
                        v156 = (__int64 *)(v154 + 16LL * v155);
                        v157 = *v156;
                        if ( v152 == *v156 )
                          goto LABEL_194;
                        v235 = v236;
                      }
                    }
                  }
                }
                goto LABEL_57;
            }
          }
          if ( (_BYTE)v63 != 77 )
            goto LABEL_57;
          v83 = *(_DWORD *)(v59 + 640);
          if ( !v83 )
            goto LABEL_57;
          v84 = v83 - 1;
          v85 = *(_QWORD *)(v59 + 624);
          v86 = ((unsigned int)v62 >> 4) ^ ((unsigned int)v62 >> 9);
          LODWORD(v87) = (v83 - 1) & v86;
          v88 = v85 + 16LL * (unsigned int)v87;
          v89 = *(__int64 **)v88;
          if ( v62 == *(__int64 **)v88 )
          {
LABEL_82:
            v90 = 8LL * *(unsigned int *)(v88 + 8);
LABEL_83:
            v91 = (__int64 *)(*(_QWORD *)(v59 + 648) + v90);
            v92 = *v91 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v91 & 4) != 0 )
            {
              v91 = *(__int64 **)v92;
              v93 = *(_QWORD *)v92 + 8LL * *(unsigned int *)(v92 + 8);
            }
            else
            {
              if ( !v92 )
                goto LABEL_57;
              v93 = (__int64)(v91 + 1);
            }
            if ( (__int64 *)v93 != v91 )
            {
              v250 = v64;
              v94 = v91;
              do
              {
                v95 = *v94++;
                sub_1AEA2D0(v95, v62, v260);
              }
              while ( (__int64 *)v93 != v94 );
              v64 = v250;
            }
            goto LABEL_57;
          }
          v211 = *(__int64 **)v88;
          LODWORD(v212) = (v83 - 1) & (((unsigned int)v62 >> 4) ^ ((unsigned int)v62 >> 9));
          for ( k = 1; ; ++k )
          {
            if ( v211 == (__int64 *)-8LL )
              goto LABEL_57;
            v212 = v84 & ((_DWORD)v212 + k);
            v211 = *(__int64 **)(v85 + 16 * v212);
            if ( v62 == v211 )
              break;
          }
          v214 = 1;
          v215 = 0;
          while ( v89 != (__int64 *)-8LL )
          {
            if ( v89 == (__int64 *)-16LL && !v215 )
              v215 = v88;
            v87 = v84 & ((_DWORD)v87 + v214);
            v88 = v85 + 16 * v87;
            v89 = *(__int64 **)v88;
            if ( v62 == *(__int64 **)v88 )
              goto LABEL_82;
            ++v214;
          }
          v216 = *(_DWORD *)(v59 + 632);
          if ( v215 )
            v88 = v215;
          ++*(_QWORD *)(v59 + 616);
          v217 = v216 + 1;
          if ( 4 * (v216 + 1) >= 3 * v83 )
          {
            sub_1A72540(v256, 2 * v83);
            v220 = *(_DWORD *)(v59 + 640);
            if ( !v220 )
              goto LABEL_341;
            v221 = v220 - 1;
            v222 = *(_QWORD *)(v59 + 624);
            v223 = v221 & v86;
            v217 = *(_DWORD *)(v59 + 632) + 1;
            v88 = v222 + 16LL * v223;
            v224 = *(__int64 **)v88;
            if ( v62 != *(__int64 **)v88 )
            {
              v225 = 1;
              v226 = 0;
              while ( v224 != (__int64 *)-8LL )
              {
                if ( !v226 && v224 == (__int64 *)-16LL )
                  v226 = v88;
                v223 = v221 & (v225 + v223);
                v88 = v222 + 16LL * v223;
                v224 = *(__int64 **)v88;
                if ( v62 == *(__int64 **)v88 )
                  goto LABEL_274;
                ++v225;
              }
LABEL_308:
              if ( v226 )
                v88 = v226;
            }
          }
          else if ( v83 - *(_DWORD *)(v59 + 636) - v217 <= v83 >> 3 )
          {
            sub_1A72540(v256, v83);
            v227 = *(_DWORD *)(v59 + 640);
            if ( !v227 )
              goto LABEL_341;
            v228 = v227 - 1;
            v229 = *(_QWORD *)(v59 + 624);
            v230 = 1;
            v231 = v228 & v86;
            v217 = *(_DWORD *)(v59 + 632) + 1;
            v226 = 0;
            v88 = v229 + 16LL * v231;
            v232 = *(__int64 **)v88;
            if ( v62 != *(__int64 **)v88 )
            {
              while ( v232 != (__int64 *)-8LL )
              {
                if ( !v226 && v232 == (__int64 *)-16LL )
                  v226 = v88;
                v231 = v228 & (v230 + v231);
                v88 = v229 + 16LL * v231;
                v232 = *(__int64 **)v88;
                if ( v62 == *(__int64 **)v88 )
                  goto LABEL_274;
                ++v230;
              }
              goto LABEL_308;
            }
          }
LABEL_274:
          *(_DWORD *)(v59 + 632) = v217;
          if ( *(_QWORD *)v88 != -8 )
            --*(_DWORD *)(v59 + 636);
          *(_QWORD *)v88 = v62;
          v90 = 0;
          *(_DWORD *)(v88 + 8) = 0;
          goto LABEL_83;
        }
        v65 = *((_QWORD *)v60 - 6);
        if ( *(_BYTE *)(v65 + 16) != 53 )
          goto LABEL_57;
        v66 = *(unsigned int *)(v59 + 576);
        if ( !(_DWORD)v66 )
          goto LABEL_57;
        v67 = *(_QWORD *)(v59 + 560);
        v68 = 1;
        LODWORD(v69) = (v66 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
        v70 = (__int64 *)(v67 + 16LL * (unsigned int)v69);
        v71 = *v70;
        if ( v65 == *v70 )
        {
LABEL_65:
          if ( v70 == (__int64 *)(v67 + 16 * v66) )
            goto LABEL_57;
          v72 = *((unsigned int *)v70 + 2);
          v73 = *((_QWORD *)v60 - 9);
          v74 = *a4;
          if ( !*(_BYTE *)(v59 + 544) )
          {
            *(_QWORD *)(v74 + 8 * v72) = v73;
            v75 = &(*a5)[v72];
            if ( v75 != (__int64 *)(v60 + 12) )
            {
              if ( *v75 )
                sub_161E7C0((__int64)v75, *v75);
              v76 = *((_QWORD *)v60 + 3);
              *v75 = v76;
              if ( v76 )
                sub_1623A60((__int64)v75, v76, 2);
            }
            v77 = (__int64 *)(*(_QWORD *)(v59 + 648) + 8LL * *((unsigned int *)v70 + 2));
            v78 = *v77 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v77 & 4) != 0 )
            {
              v77 = *(__int64 **)v78;
              v168 = *(_QWORD *)v78 + 8LL * *(unsigned int *)(v78 + 8);
            }
            else
            {
              if ( !v78 )
              {
LABEL_74:
                sub_157EA20(a2 + 40, (__int64)v62);
                v79 = (unsigned __int64 *)*((_QWORD *)v60 + 1);
                v80 = *(_QWORD *)v60 & 0xFFFFFFFFFFFFFFF8LL;
                *v79 = v80 | *v79 & 7;
                *(_QWORD *)(v80 + 8) = v79;
                *(_QWORD *)v60 &= 7uLL;
                *((_QWORD *)v60 + 1) = 0;
                sub_164BEC0((__int64)v62, (__int64)v62, v80, (__int64)v79, a7, a8, a9, a10, v81, v82, a13, a14);
                goto LABEL_57;
              }
              v168 = (__int64)(v77 + 1);
            }
            if ( (__int64 *)v168 != v77 )
            {
              v252 = v60;
              v169 = (__int64 *)v168;
              v248 = v64;
              v170 = v77;
              do
              {
                v171 = *v170++;
                sub_1AE9B50(v171, (__int64)v62, v260);
              }
              while ( v169 != v170 );
              v60 = v252;
              v64 = v248;
            }
            goto LABEL_74;
          }
          *(_QWORD *)(v74 + 8 * v72) = *(_QWORD *)(v73 - 24);
          v172 = (__int64 *)(*(_QWORD *)(v59 + 648) + 8LL * *((unsigned int *)v70 + 2));
          v173 = *v172;
          if ( (*v172 & 4) != 0 )
          {
            v174 = v173 & 0xFFFFFFFFFFFFFFF8LL;
            v172 = *(__int64 **)v174;
            v175 = *(_QWORD *)v174 + 8LL * *(unsigned int *)(v174 + 8);
LABEL_226:
            if ( (__int64 *)v175 != v172 )
            {
              v176 = v59;
              v177 = (__int64 *)v175;
              v249 = v64;
              v178 = v172;
              do
              {
                v179 = *v178++;
                sub_1AE9B50(v179, (__int64)v62, v260);
              }
              while ( v177 != v178 );
              v64 = v249;
              v59 = v176;
            }
          }
          else if ( (v173 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v175 = (__int64)(v172 + 1);
            goto LABEL_226;
          }
          v180 = *((_QWORD *)v60 - 9);
          sub_157EA20(a2 + 40, (__int64)v62);
          v181 = (unsigned __int64 *)*((_QWORD *)v60 + 1);
          v182 = *(_QWORD *)v60 & 0xFFFFFFFFFFFFFFF8LL;
          *v181 = v182 | *v181 & 7;
          *(_QWORD *)(v182 + 8) = v181;
          *(_QWORD *)v60 &= 7uLL;
          *((_QWORD *)v60 + 1) = 0;
          sub_164BEC0((__int64)v62, (__int64)v62, v182, (__int64)v181, a7, a8, a9, a10, v183, v184, a13, a14);
          if ( *(_BYTE *)(v180 + 16) == 54 && sub_1648CD0(v180, 0) )
          {
            sub_157EA20(a2 + 40, v180);
            v185 = *(unsigned __int64 **)(v180 + 32);
            v186 = *(_QWORD *)(v180 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            *v185 = v186 | *v185 & 7;
            *(_QWORD *)(v186 + 8) = v185;
            *(_QWORD *)(v180 + 24) &= 7uLL;
            *(_QWORD *)(v180 + 32) = 0;
            sub_164BEC0(v180, v180, v186, (__int64)v185, a7, a8, a9, a10, v187, v188, a13, a14);
          }
          goto LABEL_57;
        }
        while ( v71 != -8 )
        {
          v69 = ((_DWORD)v66 - 1) & (unsigned int)(v69 + v68);
          v70 = (__int64 *)(v67 + 16 * v69);
          v71 = *v70;
          if ( v65 == *v70 )
            goto LABEL_65;
          ++v68;
        }
      }
LABEL_57:
      v60 = v64;
    }
    result = (__int64 *)sub_157EBA0(a2);
    v261 = (__int64)result;
    v96 = (__int64)result;
    if ( result )
    {
      result = (__int64 *)sub_15F4D60((__int64)result);
      v257 = (int)result;
      if ( (_DWORD)result )
      {
        v267 = 0;
        v263 = 0;
        v264 = (__int64 *)v268;
        v265 = (__int64 *)v268;
        v266 = 8;
        v97 = sub_15F4DF0(v96, 0);
        v98 = v264;
        if ( v265 != v264 )
          goto LABEL_94;
        v133 = &v264[HIDWORD(v266)];
        if ( v264 == v133 )
        {
LABEL_235:
          if ( HIDWORD(v266) >= (unsigned int)v266 )
          {
LABEL_94:
            sub_16CCBA0((__int64)&v263, v97);
          }
          else
          {
            ++HIDWORD(v266);
            *v133 = v97;
            ++v263;
          }
        }
        else
        {
          v134 = 0;
          while ( v97 != *v98 )
          {
            if ( *v98 == -2 )
              v134 = v98;
            if ( v133 == ++v98 )
            {
              if ( !v134 )
                goto LABEL_235;
              *v134 = v97;
              --v267;
              ++v263;
              break;
            }
          }
        }
        v241 = sub_15F4DF0(v261, 0);
        v99 = 1;
        if ( v257 == 1 )
        {
LABEL_110:
          if ( v265 != v264 )
            _libc_free((unsigned __int64)v265);
          a3 = a2;
          a2 = v241;
          continue;
        }
        while ( 1 )
        {
          v101 = sub_15F4DF0(v261, v99);
          v102 = v264;
          if ( v265 != v264 )
            goto LABEL_97;
          v103 = &v264[HIDWORD(v266)];
          if ( v264 != v103 )
          {
            v104 = 0;
            while ( v101 != *v102 )
            {
              if ( *v102 == -2 )
                v104 = v102;
              if ( v103 == ++v102 )
              {
                if ( !v104 )
                  goto LABEL_152;
                *v104 = v101;
                --v267;
                ++v263;
                goto LABEL_108;
              }
            }
            goto LABEL_98;
          }
LABEL_152:
          if ( HIDWORD(v266) < (unsigned int)v266 )
          {
            ++HIDWORD(v266);
            *v103 = v101;
            ++v263;
          }
          else
          {
LABEL_97:
            sub_16CCBA0((__int64)&v263, v101);
            if ( !v100 )
              goto LABEL_98;
          }
LABEL_108:
          v105 = sub_15F4DF0(v261, v99);
          v106 = (_QWORD *)a6[1];
          v107 = v106;
          if ( v106 == (_QWORD *)a6[2] )
          {
            v114 = (__int64)v106 - *a6;
            v251 = (_QWORD *)*a6;
            v115 = v114 >> 6;
            if ( v114 >> 6 == 0x1FFFFFFFFFFFFFFLL )
              sub_4262D8((__int64)"vector::_M_realloc_insert");
            v116 = 1;
            if ( v115 )
              v116 = v114 >> 6;
            v117 = __CFADD__(v115, v116);
            v118 = v115 + v116;
            if ( v117 )
            {
              v120 = 0x7FFFFFFFFFFFFFC0LL;
              goto LABEL_131;
            }
            if ( v118 )
            {
              v119 = 0x1FFFFFFFFFFFFFFLL;
              if ( v118 <= 0x1FFFFFFFFFFFFFFLL )
                v119 = v118;
              v120 = v119 << 6;
LABEL_131:
              v247 = sub_22077B0(v120);
              v121 = v247 + 64;
              v238 = v247 + v120;
            }
            else
            {
              v238 = 0;
              v121 = 64;
              v247 = 0;
            }
            sub_1B31FC0((_QWORD *)(v247 + v114), v105, a2, (__int64)a4, a5);
            if ( v107 != v251 )
            {
              v243 = v99;
              v122 = (_QWORD *)v247;
              v123 = v251;
              v124 = v107;
              while ( 1 )
              {
                if ( v122 )
                {
                  *v122 = *v123;
                  v122[1] = v123[1];
                  v122[2] = v123[2];
                  v122[3] = v123[3];
                  v122[4] = v123[4];
                  v125 = v123[5];
                  v123[4] = 0;
                  v123[3] = 0;
                  v123[2] = 0;
                  v122[5] = v125;
                  v122[6] = v123[6];
                  v122[7] = v123[7];
                  v123[7] = 0;
                  v123[6] = 0;
                  v123[5] = 0;
                }
                else
                {
                  v127 = (__int64 *)v123[6];
                  v128 = (__int64 *)v123[5];
                  if ( v127 == v128 )
                  {
                    v129 = v123[7] - (_QWORD)v128;
                  }
                  else
                  {
                    do
                    {
                      if ( *v128 )
                        sub_161E7C0((__int64)v128, *v128);
                      ++v128;
                    }
                    while ( v127 != v128 );
                    v128 = (__int64 *)v123[5];
                    v129 = v123[7] - (_QWORD)v128;
                  }
                  if ( v128 )
                    j_j___libc_free_0(v128, v129);
                }
                v126 = v123[2];
                if ( v126 )
                  j_j___libc_free_0(v126, v123[4] - v126);
                v123 += 8;
                if ( v124 == v123 )
                  break;
                v122 += 8;
              }
              v99 = v243;
              v121 = (__int64)(v122 + 16);
            }
            if ( v251 )
              j_j___libc_free_0(v251, a6[2] - (_QWORD)v251);
            *a6 = v247;
            a6[1] = v121;
            a6[2] = v238;
LABEL_98:
            if ( ++v99 == v257 )
              goto LABEL_110;
            continue;
          }
          ++v99;
          sub_1B31FC0(v106, v105, a2, (__int64)a4, a5);
          a6[1] += 64;
          if ( v99 == v257 )
            goto LABEL_110;
        }
      }
    }
    return result;
  }
}
