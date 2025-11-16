// Function: sub_1866840
// Address: 0x1866840
//
__int64 __fastcall sub_1866840(
        _QWORD *a1,
        __int64 *a2,
        _QWORD *a3,
        __int64 (__fastcall *a4)(__int64, _QWORD *),
        __int64 a5,
        __m128 a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13,
        __int64 a14,
        __int64 (__fastcall *a15)(__int64, _QWORD *),
        __int64 a16,
        __int64 (__fastcall *a17)(__int64, _QWORD *),
        __int64 a18)
{
  __int64 *v18; // rdx
  _QWORD *v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // rax
  unsigned int v22; // eax
  _QWORD *i; // rbx
  __int64 v24; // rsi
  char v25; // al
  __int64 *v26; // rax
  __int64 *v27; // rdi
  __int64 *v28; // rcx
  _QWORD *v29; // r12
  _QWORD *v30; // r14
  __int64 v31; // r13
  __int64 *v32; // rax
  _QWORD *v33; // r12
  _QWORD *j; // r13
  __int64 v35; // r14
  __int64 v36; // rsi
  char v37; // al
  __int64 *v38; // rax
  __int64 *v39; // rdi
  __int64 *v40; // rcx
  _QWORD *k; // rax
  _QWORD *v42; // rbx
  __int64 v43; // rax
  _QWORD *v44; // r15
  _QWORD *v45; // r13
  _QWORD *v46; // r14
  _QWORD *v47; // r15
  _QWORD *v48; // rbx
  _QWORD *v49; // r12
  __int64 *v50; // rsi
  int v51; // r14d
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  __int64 v54; // r13
  double v55; // xmm4_8
  double v56; // xmm5_8
  unsigned __int8 v57; // cl
  unsigned __int8 v58; // dl
  __int64 *v59; // r14
  char v60; // r8
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // r13
  unsigned __int64 v64; // rax
  unsigned __int8 v65; // dl
  unsigned __int64 v66; // r12
  __int64 v67; // r14
  char v68; // r10
  __int64 v69; // rax
  __int64 v70; // r13
  char v71; // al
  __int64 *v72; // rax
  char v73; // al
  int v74; // r12d
  char v75; // al
  _QWORD *v76; // rbx
  _BYTE *v77; // r15
  __int64 v78; // rax
  __int64 v79; // rax
  int v80; // eax
  double v81; // xmm4_8
  double v82; // xmm5_8
  double v83; // xmm4_8
  double v84; // xmm5_8
  __int64 *v85; // rax
  __int64 v86; // r13
  __int64 v87; // r12
  __int64 *v88; // rbx
  _QWORD *v89; // r15
  _QWORD *v90; // r14
  _QWORD *v91; // r12
  char v92; // di
  int v93; // edx
  char v94; // bl
  char v95; // al
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // r8
  _BOOL4 v99; // eax
  __int64 v100; // r8
  int v101; // esi
  __int64 v102; // rax
  double v103; // xmm4_8
  double v104; // xmm5_8
  char v105; // si
  char v106; // al
  char v107; // al
  __int64 *v108; // rax
  __int64 *v109; // rax
  unsigned __int64 *v110; // rsi
  unsigned __int64 v111; // rdx
  double v112; // xmm4_8
  double v113; // xmm5_8
  __int64 *v114; // r12
  __int64 v115; // rdx
  __int64 v116; // rsi
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rbx
  __int64 v120; // r13
  char v121; // bl
  __int64 v122; // rdi
  __int64 ***v123; // rax
  __int64 ***v124; // r12
  __int64 v125; // rdi
  __int64 v126; // rdx
  __int64 v127; // rcx
  char v128; // r15
  __int64 v129; // rax
  double v130; // xmm4_8
  double v131; // xmm5_8
  char v132; // al
  __int64 *v133; // rax
  __int64 *v134; // rax
  __int64 *v135; // rdx
  __int64 v136; // rax
  __int64 *v137; // rsi
  __int64 *v138; // rcx
  char v139; // si
  bool v140; // di
  char v141; // al
  char v142; // al
  int v143; // edx
  __int64 v144; // r14
  __int16 v145; // ax
  __int64 v146; // rbx
  __int64 v147; // r14
  _QWORD *v148; // r13
  int v149; // r12d
  unsigned __int64 v150; // rsi
  __int16 v151; // cx
  unsigned __int64 v152; // rax
  unsigned __int8 v153; // cl
  __int64 v154; // rax
  double v155; // xmm4_8
  double v156; // xmm5_8
  __int64 v157; // rax
  int v158; // ecx
  __int64 v159; // rbx
  unsigned __int64 v160; // rax
  unsigned __int8 v161; // dl
  unsigned __int64 v162; // rcx
  unsigned __int64 v163; // r9
  __int64 *v164; // rax
  __int64 *v165; // rax
  __int64 v166; // rcx
  __int64 v167; // rsi
  __int64 *v168; // rcx
  __int64 v169; // r14
  __int16 v170; // ax
  __int64 v171; // rbx
  __int64 v172; // r14
  _QWORD *v173; // r13
  int v174; // r12d
  unsigned __int64 v175; // rsi
  __int16 v176; // cx
  unsigned __int64 v177; // rax
  unsigned __int8 v178; // cl
  _BOOL4 v179; // eax
  __int64 v180; // rbx
  __int64 *v181; // rdx
  __int64 *v182; // rax
  __int64 *v183; // rdx
  __int64 *v184; // rax
  __int64 v185; // rsi
  __int64 v186; // rdx
  unsigned int v187; // r9d
  int *v188; // rax
  int v189; // ecx
  char v190; // dl
  unsigned __int64 v192; // rax
  char v193; // al
  char v194; // dl
  int v195; // eax
  __int64 *v196; // rdi
  __int64 *v197; // rsi
  __int64 *v198; // rdi
  __int64 *v199; // rsi
  unsigned __int64 v200; // rax
  int v201; // r8d
  unsigned __int8 v202; // [rsp+7h] [rbp-229h]
  __int64 v206; // [rsp+20h] [rbp-210h]
  _QWORD *v207; // [rsp+28h] [rbp-208h]
  __int64 v208; // [rsp+28h] [rbp-208h]
  __int64 v209; // [rsp+40h] [rbp-1F0h]
  _QWORD *v210; // [rsp+40h] [rbp-1F0h]
  _QWORD *v211; // [rsp+48h] [rbp-1E8h]
  unsigned __int8 v212; // [rsp+48h] [rbp-1E8h]
  _QWORD *v213; // [rsp+50h] [rbp-1E0h]
  __int64 ***v214; // [rsp+50h] [rbp-1E0h]
  unsigned __int8 v215; // [rsp+50h] [rbp-1E0h]
  _QWORD *v216; // [rsp+58h] [rbp-1D8h]
  __int64 v217; // [rsp+58h] [rbp-1D8h]
  __int64 *v218; // [rsp+58h] [rbp-1D8h]
  __int64 v219; // [rsp+58h] [rbp-1D8h]
  char v220; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 v221; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 v222; // [rsp+58h] [rbp-1D8h]
  _QWORD *v223; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 v224; // [rsp+58h] [rbp-1D8h]
  _QWORD *v225; // [rsp+58h] [rbp-1D8h]
  _QWORD *v226; // [rsp+58h] [rbp-1D8h]
  _QWORD *v227; // [rsp+70h] [rbp-1C0h]
  _QWORD *v228; // [rsp+78h] [rbp-1B8h]
  __int64 v230; // [rsp+88h] [rbp-1A8h]
  _QWORD *v231; // [rsp+90h] [rbp-1A0h]
  _QWORD *v232; // [rsp+90h] [rbp-1A0h]
  _QWORD *v233; // [rsp+98h] [rbp-198h] BYREF
  int v234; // [rsp+A4h] [rbp-18Ch] BYREF
  _QWORD *v235; // [rsp+A8h] [rbp-188h] BYREF
  __int64 v236; // [rsp+B0h] [rbp-180h] BYREF
  __int64 *v237; // [rsp+B8h] [rbp-178h]
  __int64 *v238; // [rsp+C0h] [rbp-170h]
  _BYTE v239[12]; // [rsp+C8h] [rbp-168h]
  _BYTE s[72]; // [rsp+D8h] [rbp-158h] BYREF
  __int64 *v241; // [rsp+120h] [rbp-110h] BYREF
  __int64 *v242; // [rsp+128h] [rbp-108h]
  __int64 *v243; // [rsp+130h] [rbp-100h]
  __int64 v244; // [rsp+138h] [rbp-F8h]
  int v245; // [rsp+140h] [rbp-F0h]
  _BYTE v246[64]; // [rsp+148h] [rbp-E8h] BYREF
  __int64 v247; // [rsp+188h] [rbp-A8h] BYREF
  __int64 *v248; // [rsp+190h] [rbp-A0h]
  __int64 *v249; // [rsp+198h] [rbp-98h]
  __int64 v250; // [rsp+1A0h] [rbp-90h]
  int v251; // [rsp+1A8h] [rbp-88h]
  _BYTE v252[64]; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v253; // [rsp+1F0h] [rbp-40h]
  __int64 v254; // [rsp+1F8h] [rbp-38h]

  v233 = a3;
  v18 = (__int64 *)s;
  v237 = (__int64 *)s;
  v238 = (__int64 *)s;
  *(_QWORD *)v239 = 8;
  *(_DWORD *)&v239[8] = 0;
  v202 = 0;
  v19 = a1 + 1;
  v20 = (__int64 *)s;
  v228 = v19;
  v21 = 0;
  while ( 2 )
  {
    v236 = v21 + 1;
    if ( v20 == v18 )
      goto LABEL_7;
    v22 = 4 * (*(_DWORD *)&v239[4] - *(_DWORD *)&v239[8]);
    if ( v22 < 0x20 )
      v22 = 32;
    if ( *(_DWORD *)v239 <= v22 )
    {
      memset(v20, -1, 8LL * *(unsigned int *)v239);
LABEL_7:
      *(_QWORD *)&v239[4] = 0;
      goto LABEL_8;
    }
    sub_16CC920((__int64)&v236);
LABEL_8:
    for ( i = (_QWORD *)a1[2]; i != v228; i = (_QWORD *)i[1] )
    {
LABEL_12:
      if ( !i )
        BUG();
      v24 = *(i - 1);
      if ( v24 )
      {
        if ( (v25 = *(_BYTE *)(i - 3) & 0xF, ((v25 + 9) & 0xFu) > 1) && ((v25 + 15) & 0xFu) > 2 || *(i - 6) )
        {
          v26 = v237;
          if ( v238 != v237 )
            goto LABEL_10;
          v27 = &v237[*(unsigned int *)&v239[4]];
          if ( v237 != v27 )
          {
            v28 = 0;
            while ( v24 != *v26 )
            {
              if ( *v26 == -2 )
                v28 = v26;
              if ( v27 == ++v26 )
              {
                if ( !v28 )
                  goto LABEL_238;
                *v28 = v24;
                --*(_DWORD *)&v239[8];
                i = (_QWORD *)i[1];
                ++v236;
                if ( i != v228 )
                  goto LABEL_12;
                goto LABEL_26;
              }
            }
            continue;
          }
LABEL_238:
          if ( *(_DWORD *)&v239[4] < *(_DWORD *)v239 )
          {
            ++*(_DWORD *)&v239[4];
            *v27 = v24;
            ++v236;
          }
          else
          {
LABEL_10:
            sub_16CCBA0((__int64)&v236, v24);
          }
        }
      }
    }
LABEL_26:
    v29 = (_QWORD *)a1[4];
    v231 = a1 + 3;
    v30 = a1 + 3;
    if ( a1 + 3 != v29 )
    {
      while ( 1 )
      {
        if ( !v29 )
          BUG();
        v31 = *(v29 - 1);
        if ( !v31 || (unsigned __int8)sub_15E36F0((__int64)(v29 - 7)) )
          goto LABEL_28;
        v32 = v237;
        if ( v238 == v237 )
        {
          v137 = &v237[*(unsigned int *)&v239[4]];
          if ( v237 == v137 )
          {
LABEL_332:
            if ( *(_DWORD *)&v239[4] >= *(_DWORD *)v239 )
              goto LABEL_33;
            ++*(_DWORD *)&v239[4];
            *v137 = v31;
            ++v236;
          }
          else
          {
            v138 = 0;
            while ( v31 != *v32 )
            {
              if ( *v32 == -2 )
                v138 = v32;
              if ( v137 == ++v32 )
              {
                if ( !v138 )
                  goto LABEL_332;
                *v138 = v31;
                --*(_DWORD *)&v239[8];
                ++v236;
                break;
              }
            }
          }
LABEL_28:
          v29 = (_QWORD *)v29[1];
          if ( v30 == v29 )
            break;
        }
        else
        {
LABEL_33:
          sub_16CCBA0((__int64)&v236, v31);
          v29 = (_QWORD *)v29[1];
          if ( v30 == v29 )
            break;
        }
      }
    }
    v33 = (_QWORD *)a1[6];
    v230 = (__int64)(a1 + 5);
    for ( j = a1 + 5; j != v33; v33 = (_QWORD *)v33[1] )
    {
LABEL_38:
      v35 = (__int64)(v33 - 6);
      if ( !v33 )
        v35 = 0;
      v36 = sub_15E4F10(v35);
      if ( v36 )
      {
        if ( (v37 = *(_BYTE *)(v35 + 32) & 0xF, ((v37 + 15) & 0xFu) > 2) && ((v37 + 9) & 0xFu) > 1
          || *(_QWORD *)(v35 + 8) )
        {
          v38 = v237;
          if ( v238 != v237 )
            goto LABEL_36;
          v39 = &v237[*(unsigned int *)&v239[4]];
          if ( v237 != v39 )
          {
            v40 = 0;
            while ( v36 != *v38 )
            {
              if ( *v38 == -2 )
                v40 = v38;
              if ( v39 == ++v38 )
              {
                if ( !v40 )
                  goto LABEL_236;
                *v40 = v36;
                --*(_DWORD *)&v239[8];
                v33 = (_QWORD *)v33[1];
                ++v236;
                if ( j != v33 )
                  goto LABEL_38;
                goto LABEL_53;
              }
            }
            continue;
          }
LABEL_236:
          if ( *(_DWORD *)&v239[4] < *(_DWORD *)v239 )
          {
            ++*(_DWORD *)&v239[4];
            *v39 = v36;
            ++v236;
          }
          else
          {
LABEL_36:
            sub_16CCBA0((__int64)&v236, v36);
          }
        }
      }
    }
LABEL_53:
    v241 = 0;
    v242 = 0;
    v211 = v233;
    v243 = 0;
    for ( k = (_QWORD *)a1[4]; v231 != k; k = v42 )
    {
      v42 = (_QWORD *)k[1];
      v43 = a4(a5, k - 7);
      if ( (unsigned __int8)sub_14A2E10(v43) )
        goto LABEL_58;
    }
    if ( byte_4FAB3C0 )
    {
LABEL_58:
      if ( v231 == (_QWORD *)a1[4] )
        goto LABEL_117;
      v44 = (_QWORD *)a1[4];
      do
      {
        while ( 1 )
        {
          v45 = v44;
          v44 = (_QWORD *)v44[1];
          v235 = v45 - 7;
          v216 = v45 + 2;
          if ( (_QWORD *)v45[3] != v45 + 2 )
            break;
LABEL_71:
          v50 = v242;
          if ( v242 == v243 )
          {
            sub_14F2380((__int64)&v241, v242, &v235);
            if ( v231 == v44 )
              goto LABEL_76;
          }
          else
          {
            if ( v242 )
            {
              *v242 = (__int64)v235;
              v50 = v242;
            }
            v242 = v50 + 1;
LABEL_75:
            if ( v231 == v44 )
              goto LABEL_76;
          }
        }
        v227 = v45 - 7;
        v46 = v44;
        v47 = (_QWORD *)v45[3];
        while ( 1 )
        {
          if ( !v47 )
            BUG();
          v48 = (_QWORD *)v47[3];
          v49 = v47 + 2;
          if ( v48 != v47 + 2 )
            break;
LABEL_69:
          v47 = (_QWORD *)v47[1];
          if ( v216 == v47 )
          {
            v44 = v46;
            goto LABEL_71;
          }
        }
        v213 = v47;
        v44 = v46;
        while ( 1 )
        {
          if ( !v48 )
            BUG();
          if ( *((_BYTE *)v48 - 8) == 78 )
          {
            v70 = *(v48 - 6);
            v71 = *(_BYTE *)(v70 + 16);
            if ( v71 != 20 )
            {
              if ( v71 || (*(_BYTE *)(v70 + 32) & 0xFu) - 7 > 1 )
                goto LABEL_75;
              if ( !*(_DWORD *)(v70 + 36) )
              {
                if ( !(unsigned __int8)sub_185B0A0(*(v48 - 6))
                  || *(_DWORD *)(*(_QWORD *)(v70 + 24) + 8LL) >> 8
                  || (unsigned __int8)sub_15E3650(v70, 0) )
                {
                  goto LABEL_75;
                }
                v72 = (__int64 *)a15(a16, v227);
                if ( !sub_185AD30((unsigned __int64)(v48 - 3) | 4, v72) )
                  break;
              }
            }
          }
          v48 = (_QWORD *)v48[1];
          if ( v49 == v48 )
          {
            v47 = v213;
            goto LABEL_69;
          }
        }
      }
      while ( v231 != v46 );
    }
LABEL_76:
    v51 = 0;
    v52 = (_QWORD *)a1[4];
    if ( v231 != v52 )
    {
LABEL_80:
      v53 = v52;
      v52 = (_QWORD *)v52[1];
      v54 = (__int64)(v53 - 7);
      if ( (unsigned __int8)sub_1560180((__int64)(v53 + 7), 18) )
        goto LABEL_79;
      if ( (*((_BYTE *)v53 - 33) & 0x20) == 0 && !sub_15E4F60((__int64)(v53 - 7)) )
      {
        v73 = *((_BYTE *)v53 - 24);
        if ( (v73 & 0xFu) - 7 > 1 )
        {
          *((_BYTE *)v53 - 23) |= 0x40u;
          *((_BYTE *)v53 - 24) = v73 & 0xC0 | 7;
        }
      }
      if ( (unsigned __int8)sub_185C3B0((__int64)(v53 - 7), (__int64)&v236) )
        goto LABEL_78;
      if ( !sub_15E4F60((__int64)(v53 - 7)) )
      {
        v221 = sub_1AF0CE0(v53 - 7, 0, 0);
        if ( v221 )
        {
          v136 = a17(a18, v53 - 7);
          *(_QWORD *)(v136 + 64) = v54;
          sub_15D3930(v136);
          v51 = v221;
        }
      }
      v51 |= sub_1864060(
               (__int64)(v53 - 7),
               v211,
               (__int64 (__fastcall *)(_QWORD, _QWORD))a17,
               a18,
               a6,
               a7,
               a8,
               a9,
               v55,
               v56,
               a12,
               a13);
      if ( (*(_BYTE *)(v53 - 3) & 0xFu) - 7 > 1 )
        goto LABEL_79;
      v57 = sub_185B0A0((__int64)(v53 - 7));
      if ( !v57 )
        goto LABEL_87;
      if ( *(_DWORD *)(*(v53 - 4) + 8LL) >> 8 )
        goto LABEL_87;
      v224 = v57;
      if ( (unsigned __int8)sub_15E3650((__int64)(v53 - 7), 0) )
        goto LABEL_87;
      v157 = a4(a5, v53 - 7);
      v158 = v224;
      if ( !byte_4FAB3C0 )
      {
        if ( !(unsigned __int8)sub_14A2E10(v157) || !*(v53 - 6) )
          goto LABEL_87;
        v215 = v224;
        v225 = v52;
        v159 = *(v53 - 6);
        while ( 1 )
        {
          v160 = (unsigned __int64)sub_1648700(v159);
          v161 = *(_BYTE *)(v160 + 16);
          if ( v161 != 4 )
            break;
LABEL_283:
          v159 = *(_QWORD *)(v159 + 8);
          if ( !v159 )
          {
            v52 = v225;
            v158 = v215;
            goto LABEL_285;
          }
        }
        if ( v161 <= 0x17u )
        {
          v162 = 0;
          v163 = 0;
        }
        else if ( v161 == 78 )
        {
          v163 = v160 | 4;
          v162 = v160 & 0xFFFFFFFFFFFFFFF8LL;
        }
        else
        {
          v162 = 0;
          v163 = 0;
          if ( v161 == 29 )
          {
            v163 = v160 & 0xFFFFFFFFFFFFFFFBLL;
            v162 = v160 & 0xFFFFFFFFFFFFFFF8LL;
          }
        }
        v208 = v163;
        v210 = *(_QWORD **)(*(_QWORD *)(v162 + 40) + 56LL);
        v164 = (__int64 *)a15(a16, v210);
        if ( !sub_185AD30(v208, v164) )
          goto LABEL_371;
        v165 = v241;
        v166 = ((char *)v242 - (char *)v241) >> 5;
        v167 = v242 - v241;
        if ( v166 <= 0 )
          goto LABEL_367;
        v168 = &v241[4 * v166];
        do
        {
          if ( v210 == (_QWORD *)*v165 )
            goto LABEL_282;
          if ( v210 == (_QWORD *)v165[1] )
          {
            ++v165;
            goto LABEL_282;
          }
          if ( v210 == (_QWORD *)v165[2] )
          {
            v165 += 2;
            goto LABEL_282;
          }
          if ( v210 == (_QWORD *)v165[3] )
          {
            v165 += 3;
            goto LABEL_282;
          }
          v165 += 4;
        }
        while ( v165 != v168 );
        v167 = v242 - v165;
LABEL_367:
        switch ( v167 )
        {
          case 2LL:
LABEL_379:
            if ( v210 != (_QWORD *)*v165 )
            {
              ++v165;
LABEL_370:
              if ( v210 != (_QWORD *)*v165 )
              {
LABEL_371:
                v52 = v225;
                goto LABEL_87;
              }
            }
            break;
          case 3LL:
            if ( v210 != (_QWORD *)*v165 )
            {
              ++v165;
              goto LABEL_379;
            }
            break;
          case 1LL:
            goto LABEL_370;
          default:
            goto LABEL_371;
        }
LABEL_282:
        if ( v242 == v165 )
          goto LABEL_371;
        goto LABEL_283;
      }
LABEL_285:
      v169 = *(v53 - 6);
      v170 = *((_WORD *)v53 - 19) & 0xC00F;
      LOBYTE(v170) = v170 | 0x90;
      *((_WORD *)v53 - 19) = v170;
      if ( !v169 )
        goto LABEL_362;
      v226 = v52;
      v171 = v169;
      v172 = (__int64)(v53 - 7);
      v173 = v53;
      v174 = v158;
      while ( 1 )
      {
        v177 = (unsigned __int64)sub_1648700(v171);
        v178 = *(_BYTE *)(v177 + 16);
        if ( v178 == 4 )
          goto LABEL_291;
        if ( v178 > 0x17u )
        {
          if ( v178 == 78 )
          {
            v200 = v177 | 4;
            goto LABEL_359;
          }
          v175 = 0;
          if ( v178 == 29 )
          {
            v200 = v177 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_359:
            v175 = v200 & 0xFFFFFFFFFFFFFFF8LL;
            v176 = *(_WORD *)((v200 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x8003 | 0x24;
            if ( (v200 & 4) != 0 )
              goto LABEL_290;
          }
        }
        else
        {
          v175 = 0;
        }
        v176 = *(_WORD *)(v175 + 18) & 0x8003 | 0x24;
LABEL_290:
        *(_WORD *)(v175 + 18) = v176;
LABEL_291:
        v171 = *(_QWORD *)(v171 + 8);
        if ( !v171 )
        {
          v52 = v226;
          v158 = v174;
          v53 = v173;
          v54 = v172;
LABEL_362:
          v51 = v158;
LABEL_87:
          v58 = sub_185B0A0(v54);
          if ( !v58 )
            goto LABEL_89;
          if ( *(_DWORD *)(*(v53 - 4) + 8LL) >> 8 )
            goto LABEL_89;
          v222 = v58;
          v142 = sub_15E3650(v54, 0);
          v143 = v222;
          if ( v142 )
            goto LABEL_89;
          v144 = *(v53 - 6);
          v145 = *((_WORD *)v53 - 19) & 0xC00F;
          LOBYTE(v145) = v145 | 0x80;
          *((_WORD *)v53 - 19) = v145;
          if ( !v144 )
            goto LABEL_340;
          v223 = v52;
          v146 = v144;
          v147 = v54;
          v148 = v53;
          v149 = v143;
          while ( 2 )
          {
            v152 = (unsigned __int64)sub_1648700(v146);
            v153 = *(_BYTE *)(v152 + 16);
            if ( v153 != 4 )
            {
              if ( v153 <= 0x17u )
              {
                v150 = 0;
                goto LABEL_255;
              }
              if ( v153 == 78 )
              {
                v192 = v152 | 4;
              }
              else
              {
                v150 = 0;
                if ( v153 != 29 )
                  goto LABEL_255;
                v192 = v152 & 0xFFFFFFFFFFFFFFFBLL;
              }
              v150 = v192 & 0xFFFFFFFFFFFFFFF8LL;
              v151 = *(_WORD *)((v192 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x8003 | 0x20;
              if ( (v192 & 4) == 0 )
LABEL_255:
                v151 = *(_WORD *)(v150 + 18) & 0x8003 | 0x20;
              *(_WORD *)(v150 + 18) = v151;
            }
            v146 = *(_QWORD *)(v146 + 8);
            if ( v146 )
              continue;
            break;
          }
          v52 = v223;
          v143 = v149;
          v53 = v148;
          v54 = v147;
LABEL_340:
          v51 = v143;
LABEL_89:
          v235 = (_QWORD *)v53[7];
          if ( (unsigned __int8)sub_1560490(&v235, 19, 0) && !(unsigned __int8)sub_15E3650(v54, 0) )
          {
            v217 = v53[7];
            v59 = (__int64 *)sub_15E0530(v54);
            v235 = (_QWORD *)v217;
            v60 = sub_1560490(&v235, 19, &v234);
            v61 = (__int64)v235;
            if ( v60 )
              v61 = sub_1563C10((__int64 *)&v235, v59, v234, 19);
            v62 = *(v53 - 6);
            v53[7] = v61;
            if ( v62 )
            {
              v209 = v54;
              v63 = v62;
              v207 = v52;
              while ( 1 )
              {
                v64 = (unsigned __int64)sub_1648700(v63);
                v65 = *(_BYTE *)(v64 + 16);
                if ( v65 != 4 )
                  break;
LABEL_96:
                v63 = *(_QWORD *)(v63 + 8);
                if ( !v63 )
                {
                  v52 = v207;
                  goto LABEL_78;
                }
              }
              if ( v65 > 0x17u )
              {
                if ( v65 == 78 )
                {
                  v180 = v64 | 4;
                }
                else
                {
                  v66 = 0;
                  if ( v65 != 29 )
                    goto LABEL_101;
                  v180 = v64 & 0xFFFFFFFFFFFFFFFBLL;
                }
                v66 = v180 & 0xFFFFFFFFFFFFFFF8LL;
                v67 = *(_QWORD *)((v180 & 0xFFFFFFFFFFFFFFF8LL) + 56);
                if ( ((v180 >> 2) & 1) == 0 )
LABEL_101:
                  v67 = *(_QWORD *)(v66 + 56);
                v218 = (__int64 *)sub_15E0530(v209);
                v235 = (_QWORD *)v67;
                v68 = sub_1560490(&v235, 19, &v234);
                v69 = (__int64)v235;
                if ( v68 )
                  v69 = sub_1563C10((__int64 *)&v235, v218, v234, 19);
                *(_QWORD *)(v66 + 56) = v69;
                goto LABEL_96;
              }
              v66 = 0;
              goto LABEL_101;
            }
LABEL_78:
            v51 = 1;
          }
LABEL_79:
          if ( v231 == v52 )
            goto LABEL_118;
          goto LABEL_80;
        }
      }
    }
LABEL_117:
    LOBYTE(v51) = 0;
LABEL_118:
    if ( v241 )
      j_j___libc_free_0(v241, (char *)v243 - (char *)v241);
    v74 = 0;
    v241 = a2;
    v242 = (__int64 *)&v233;
    v75 = sub_1AC3260(a1, sub_185E830, &v241);
    v76 = (_QWORD *)a1[2];
    v212 = v51 | v75;
    v232 = v233;
    if ( v76 != v228 )
    {
      do
      {
        while ( 1 )
        {
          v77 = v76;
          v76 = (_QWORD *)v76[1];
          if ( (*(v77 - 33) & 0x20) == 0 && !sub_15E4F60((__int64)(v77 - 56)) )
          {
            v95 = *(v77 - 24);
            if ( (v95 & 0xFu) - 7 > 1 )
            {
              *(v77 - 23) |= 0x40u;
              *(v77 - 24) = v95 & 0xC0 | 7;
            }
          }
          if ( !sub_15E4F60((__int64)(v77 - 56)) )
          {
            v219 = *((_QWORD *)v77 - 10);
            if ( v219 )
            {
              v78 = sub_1632FA0((__int64)a1);
              v79 = sub_14DBA30(v219, v78, (__int64)v232);
              if ( v79 )
              {
                if ( v219 != v79 )
                  sub_15E5440((__int64)(v77 - 56), v79);
              }
            }
          }
          v80 = sub_185C3B0((__int64)(v77 - 56), (__int64)&v236);
          if ( (_BYTE)v80 )
            break;
          v74 |= sub_1864060(
                   (__int64)(v77 - 56),
                   v232,
                   (__int64 (__fastcall *)(_QWORD, _QWORD))a17,
                   a18,
                   a6,
                   a7,
                   a8,
                   a9,
                   v81,
                   v82,
                   a12,
                   a13);
          if ( v76 == v228 )
            goto LABEL_131;
        }
        v74 = v80;
      }
      while ( v76 != v228 );
LABEL_131:
      v212 |= v74;
    }
    v242 = (__int64 *)v246;
    v243 = (__int64 *)v246;
    v241 = 0;
    v244 = 8;
    v245 = 0;
    v247 = 0;
    v248 = (__int64 *)v252;
    v249 = (__int64 *)v252;
    v250 = 8;
    v251 = 0;
    v253 = sub_1633E30((__int64)a1, (__int64)&v241, 0);
    v254 = sub_1633E30((__int64)a1, (__int64)&v247, 1);
    v85 = v243;
    if ( v243 == v242 )
      v86 = (__int64)&v243[HIDWORD(v244)];
    else
      v86 = (__int64)&v243[(unsigned int)v244];
    if ( v243 != (__int64 *)v86 )
    {
      while ( 1 )
      {
        v87 = *v85;
        v88 = v85;
        if ( (unsigned __int64)*v85 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( (__int64 *)v86 == ++v85 )
          goto LABEL_137;
      }
      if ( v85 != (__int64 *)v86 )
      {
        v133 = v248;
        if ( v249 == v248 )
          goto LABEL_213;
        while ( 1 )
        {
          v133 = sub_16CC9F0((__int64)&v247, v87);
          if ( *v133 == v87 )
          {
            if ( v249 == v248 )
              v135 = &v249[HIDWORD(v250)];
            else
              v135 = &v249[(unsigned int)v250];
            goto LABEL_217;
          }
          if ( v249 == v248 )
            break;
          while ( 1 )
          {
            v134 = v88 + 1;
            if ( v88 + 1 == (__int64 *)v86 )
              goto LABEL_137;
            v87 = *v134;
            for ( ++v88; (unsigned __int64)*v134 >= 0xFFFFFFFFFFFFFFFELL; v88 = v134 )
            {
              if ( (__int64 *)v86 == ++v134 )
                goto LABEL_137;
              v87 = *v134;
            }
            if ( v88 == (__int64 *)v86 )
              goto LABEL_137;
            v133 = v248;
            if ( v249 != v248 )
              break;
LABEL_213:
            v135 = &v133[HIDWORD(v250)];
            if ( v133 == v135 )
            {
LABEL_221:
              v133 = v135;
            }
            else
            {
              while ( *v133 != v87 )
              {
                if ( v135 == ++v133 )
                  goto LABEL_221;
              }
            }
LABEL_217:
            if ( v135 != v133 )
            {
              *v133 = -2;
              ++v251;
            }
          }
        }
        v133 = &v249[HIDWORD(v250)];
        v135 = v133;
        goto LABEL_217;
      }
    }
LABEL_137:
    v220 = 0;
    v89 = (_QWORD *)a1[6];
    if ( (_QWORD *)v230 == v89 )
      goto LABEL_168;
    do
    {
      v90 = v89;
      v89 = (_QWORD *)v89[1];
      v91 = v90 - 6;
      if ( (*((_BYTE *)v90 - 25) & 0x20) == 0 && !sub_15E4F60((__int64)(v90 - 6)) )
      {
        v132 = *((_BYTE *)v90 - 16);
        if ( (v132 & 0xFu) - 7 > 1 )
        {
          *((_BYTE *)v90 - 15) |= 0x40u;
          *((_BYTE *)v90 - 16) = v132 & 0xC0 | 7;
        }
      }
      v92 = (_BYTE)v90 - 48;
      v94 = sub_185C3B0((__int64)(v90 - 6), (__int64)&v236);
      if ( !v94 )
      {
        switch ( *(_BYTE *)(v90 - 2) & 0xF )
        {
          case 0:
          case 1:
            v214 = (__int64 ***)*(v90 - 9);
            v96 = sub_1649C60((__int64)v214);
            v86 = v96;
            if ( *(_BYTE *)(v96 + 16) <= 3u )
            {
              sub_159D9E0(v96);
              v97 = *(v90 - 5);
              if ( v97 )
              {
                if ( *(_QWORD *)(v97 + 8) )
                {
                  v94 = 1;
                }
                else if ( !sub_16898F0((__int64)&v241, (__int64)(v90 - 6)) )
                {
                  v94 = !sub_16898F0((__int64)&v247, (__int64)(v90 - 6));
                }
              }
              if ( (*(_BYTE *)(v90 - 2) & 0xFu) - 7 > 1
                || sub_16898F0((__int64)&v241, (__int64)(v90 - 6))
                || sub_16898F0((__int64)&v247, (__int64)(v90 - 6)) )
              {
                v98 = sub_1649C60(*(v90 - 9));
                if ( (*(_BYTE *)(v98 + 32) & 0xFu) - 7 <= 1 )
                {
                  v206 = v98;
                  v99 = sub_16898F0((__int64)&v241, v98);
                  v100 = v206;
                  if ( v99 || (v179 = sub_16898F0((__int64)&v247, v206), v100 = v206, v179) )
                    v101 = 3;
                  else
                    v101 = 2;
                  if ( !(unsigned __int8)sub_1648D00(v100, v101) )
                  {
                    v102 = sub_15A4510(v214, (__int64 **)*(v90 - 6), 0);
                    sub_164D160(
                      (__int64)(v90 - 6),
                      v102,
                      a6,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v103,
                      v104,
                      a12,
                      a13);
                    sub_164B7C0(v86, (__int64)(v90 - 6));
                    v93 = *(_BYTE *)(v90 - 2) & 0xF;
                    v92 = *(_BYTE *)(v90 - 2) & 0xF;
                    if ( (unsigned int)(v93 - 7) > 1 )
                      goto LABEL_241;
                    v105 = v92 | *(_BYTE *)(v86 + 32) & 0xC0;
                    *(_BYTE *)(v86 + 32) = v105;
                    if ( v93 == 7 )
                    {
                      v106 = *(_BYTE *)(v86 + 33) | 0x40;
                      *(_BYTE *)(v86 + 33) = v106;
                      v107 = *((_BYTE *)v90 - 15) & 0x40 | v106 & 0xBF;
                      *(_BYTE *)(v86 + 33) = v107;
                      *(_BYTE *)(v86 + 32) = *(_BYTE *)(v90 - 2) & 0x30 | v105;
                      goto LABEL_158;
                    }
                    goto LABEL_242;
                  }
                }
              }
              if ( v94 )
              {
                v129 = sub_15A4510(v214, (__int64 **)*(v90 - 6), 0);
                sub_164D160(
                  (__int64)(v90 - 6),
                  v129,
                  a6,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  v130,
                  v131,
                  a12,
                  a13);
                if ( (*(_BYTE *)(v90 - 2) & 0xFu) - 7 > 1 || sub_16898F0((__int64)&v241, (__int64)(v90 - 6)) )
                  break;
                if ( !sub_16898F0((__int64)&v247, (__int64)(v90 - 6)) )
                  goto LABEL_165;
                v220 = 1;
              }
            }
            continue;
          default:
LABEL_241:
            *(_BYTE *)(v86 + 32) = v92 | *(_BYTE *)(v86 + 32) & 0xF0;
LABEL_242:
            if ( v93 == 8 )
            {
              v193 = *(_BYTE *)(v86 + 33) | 0x40;
              *(_BYTE *)(v86 + 33) = v193;
              v107 = *((_BYTE *)v90 - 15) & 0x40 | v193 & 0xBF;
              v194 = *(_BYTE *)(v86 + 32);
              *(_BYTE *)(v86 + 33) = v107;
              *(_BYTE *)(v86 + 32) = *(_BYTE *)(v90 - 2) & 0x30 | v194 & 0xCF;
              goto LABEL_158;
            }
            v139 = *(_BYTE *)(v86 + 32);
            v140 = v92 != 9;
            if ( (v139 & 0x30) != 0 && v140 )
            {
              v141 = *(_BYTE *)(v86 + 33) | 0x40;
              *(_BYTE *)(v86 + 33) = v141;
              v107 = *((_BYTE *)v90 - 15) & 0x40 | v141 & 0xBF;
              *(_BYTE *)(v86 + 33) = v107;
              *(_BYTE *)(v86 + 32) = *(_BYTE *)(v90 - 2) & 0x30 | v139 & 0xCF;
              if ( v93 == 7 )
                goto LABEL_158;
            }
            else
            {
              v107 = *((_BYTE *)v90 - 15) & 0x40 | *(_BYTE *)(v86 + 33) & 0xBF;
              v190 = *(_BYTE *)(v86 + 32);
              *(_BYTE *)(v86 + 33) = v107;
              *(_BYTE *)(v86 + 32) = *(_BYTE *)(v90 - 2) & 0x30 | v190 & 0xCF;
            }
            if ( (*(_BYTE *)(v86 + 32) & 0x30) == 0 || !v140 )
            {
LABEL_159:
              *(_BYTE *)(v86 + 33) = *((_BYTE *)v90 - 15) & 3 | v107 & 0xFC;
              v108 = v242;
              if ( v243 == v242 )
              {
                v183 = &v242[HIDWORD(v244)];
                if ( v242 == v183 )
                {
LABEL_327:
                  v108 = &v242[HIDWORD(v244)];
                }
                else
                {
                  while ( v91 != (_QWORD *)*v108 )
                  {
                    if ( v183 == ++v108 )
                      goto LABEL_327;
                  }
                }
              }
              else
              {
                v108 = sub_16CC9F0((__int64)&v241, (__int64)(v90 - 6));
                if ( v91 == (_QWORD *)*v108 )
                {
                  if ( v243 == v242 )
                    v183 = &v243[HIDWORD(v244)];
                  else
                    v183 = &v243[(unsigned int)v244];
                }
                else
                {
                  if ( v243 != v242 )
                    goto LABEL_162;
                  v108 = &v243[HIDWORD(v244)];
                  v183 = v108;
                }
              }
              if ( v183 != v108 )
              {
                *v108 = -2;
                v184 = v242;
                ++v245;
                if ( v243 != v242 )
                {
LABEL_313:
                  sub_16CCBA0((__int64)&v241, v86);
                  goto LABEL_162;
                }
                v198 = &v242[HIDWORD(v244)];
                if ( v242 == v198 )
                {
LABEL_383:
                  if ( HIDWORD(v244) >= (unsigned int)v244 )
                    goto LABEL_313;
                  ++HIDWORD(v244);
                  *v198 = v86;
                  v241 = (__int64 *)((char *)v241 + 1);
                }
                else
                {
                  v199 = 0;
                  while ( v86 != *v184 )
                  {
                    if ( *v184 == -2 )
                      v199 = v184;
                    if ( v198 == ++v184 )
                    {
                      if ( !v199 )
                        goto LABEL_383;
                      *v199 = v86;
                      --v245;
                      v241 = (__int64 *)((char *)v241 + 1);
                      break;
                    }
                  }
                }
              }
LABEL_162:
              v109 = v248;
              if ( v249 == v248 )
              {
                v181 = &v248[HIDWORD(v250)];
                if ( v248 == v181 )
                {
LABEL_328:
                  v109 = &v248[HIDWORD(v250)];
                }
                else
                {
                  while ( v91 != (_QWORD *)*v109 )
                  {
                    if ( v181 == ++v109 )
                      goto LABEL_328;
                  }
                }
              }
              else
              {
                v109 = sub_16CC9F0((__int64)&v247, (__int64)(v90 - 6));
                if ( v91 == (_QWORD *)*v109 )
                {
                  if ( v249 == v248 )
                    v181 = &v249[HIDWORD(v250)];
                  else
                    v181 = &v249[(unsigned int)v250];
                }
                else
                {
                  if ( v249 != v248 )
                    goto LABEL_165;
                  v109 = &v249[HIDWORD(v250)];
                  v181 = v109;
                }
              }
              if ( v181 != v109 )
              {
                *v109 = -2;
                v182 = v248;
                ++v251;
                if ( v249 != v248 )
                  goto LABEL_306;
                v196 = &v248[HIDWORD(v250)];
                if ( v248 != v196 )
                {
                  v197 = 0;
                  while ( v86 != *v182 )
                  {
                    if ( *v182 == -2 )
                      v197 = v182;
                    if ( v196 == ++v182 )
                    {
                      if ( !v197 )
                        goto LABEL_381;
                      *v197 = v86;
                      --v251;
                      ++v247;
                      goto LABEL_165;
                    }
                  }
                  goto LABEL_165;
                }
LABEL_381:
                if ( HIDWORD(v250) < (unsigned int)v250 )
                {
                  ++HIDWORD(v250);
                  *v196 = v86;
                  ++v247;
                }
                else
                {
LABEL_306:
                  sub_16CCBA0((__int64)&v247, v86);
                }
              }
LABEL_165:
              sub_1631C90(v230, (__int64)(v90 - 6));
              v110 = (unsigned __int64 *)v90[1];
              v111 = *v90 & 0xFFFFFFFFFFFFFFF8LL;
              *v110 = v111 | *v110 & 7;
              *(_QWORD *)(v111 + 8) = v110;
              *v90 &= 7uLL;
              v90[1] = 0;
              sub_159D9E0((__int64)(v90 - 6));
              sub_164BE60(
                (__int64)(v90 - 6),
                a6,
                *(double *)a7.m128i_i64,
                *(double *)a8.m128i_i64,
                a9,
                v112,
                v113,
                a12,
                a13);
              sub_1648B90((__int64)(v90 - 6));
              v220 = 1;
              continue;
            }
LABEL_158:
            v107 |= 0x40u;
            *(_BYTE *)(v86 + 33) = v107;
            goto LABEL_159;
        }
      }
      v220 = 1;
    }
    while ( (_QWORD *)v230 != v89 );
    v212 |= v220;
LABEL_168:
    if ( v253 )
      sub_185C140(v253, (__int64)&v241, a6, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v83, v84, a12, a13);
    if ( v254 )
      sub_185C140(v254, (__int64)&v247, a6, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v83, v84, a12, a13);
    if ( v249 != v248 )
      _libc_free((unsigned __int64)v249);
    if ( v243 != v242 )
      _libc_free((unsigned __int64)v243);
    v114 = v233;
    LODWORD(v241) = 70;
    v115 = *v233;
    if ( (((int)*(unsigned __int8 *)(*v233 + 17LL) >> 4) & 3) == 0 )
      goto LABEL_191;
    if ( (((int)*(unsigned __int8 *)(*v233 + 17LL) >> 4) & 3) != 3 )
    {
      v185 = *(_QWORD *)(v115 + 120);
      v186 = *(unsigned int *)(v115 + 136);
      if ( (_DWORD)v186 )
      {
        v187 = ((_WORD)v186 - 1) & 0xA1E;
        v188 = (int *)(v185 + 40LL * (((_WORD)v186 - 1) & 0xA1E));
        v189 = *v188;
        if ( *v188 == 70 )
        {
LABEL_320:
          v116 = *((_QWORD *)v188 + 1);
          v117 = *((_QWORD *)v188 + 2);
          goto LABEL_179;
        }
        v195 = 1;
        while ( v189 != -1 )
        {
          v201 = v195 + 1;
          v187 = (v186 - 1) & (v195 + v187);
          v188 = (int *)(v185 + 40LL * v187);
          v189 = *v188;
          if ( *v188 == 70 )
            goto LABEL_320;
          v195 = v201;
        }
      }
      v188 = (int *)(v185 + 40 * v186);
      goto LABEL_320;
    }
    v116 = qword_4F9B700[140];
    v117 = qword_4F9B700[141];
LABEL_179:
    v118 = sub_16321A0((__int64)a1, v116, v117);
    v119 = v118;
    if ( v118 )
    {
      if ( sub_149CB50(*v114, v118, (unsigned int *)&v241) )
      {
        if ( (_DWORD)v241 == 70 )
        {
          v120 = *(_QWORD *)(v119 + 8);
          v121 = 0;
          if ( v120 )
          {
            while ( 1 )
            {
              v122 = v120;
              v120 = *(_QWORD *)(v120 + 8);
              v123 = (__int64 ***)sub_1648700(v122);
              v124 = v123;
              if ( *((_BYTE *)v123 + 16) != 78 )
                goto LABEL_184;
              v125 = sub_1649C60((__int64)v123[-3 * (*((_DWORD *)v123 + 5) & 0xFFFFFFF)]);
              if ( *(_BYTE *)(v125 + 16) )
                goto LABEL_184;
              v242 = (__int64 *)v246;
              v241 = 0;
              v243 = (__int64 *)v246;
              v244 = 8;
              v245 = 0;
              v128 = sub_185C700(v125, (__int64)&v241);
              if ( v128 )
              {
                v154 = sub_15A06D0(*v124, (__int64)&v241, v126, v127);
                sub_164D160(
                  (__int64)v124,
                  v154,
                  a6,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  v155,
                  v156,
                  a12,
                  a13);
                sub_15F20C0(v124);
                if ( v243 != v242 )
                  _libc_free((unsigned __int64)v243);
                v121 = v128;
                goto LABEL_184;
              }
              if ( v243 == v242 )
              {
LABEL_184:
                if ( !v120 )
                  goto LABEL_190;
              }
              else
              {
                _libc_free((unsigned __int64)v243);
                if ( !v120 )
                {
LABEL_190:
                  v212 |= v121;
                  break;
                }
              }
            }
          }
        }
      }
    }
LABEL_191:
    v20 = v238;
    v18 = v237;
    if ( v212 )
    {
      v202 = v212;
      v21 = v236;
      continue;
    }
    break;
  }
  if ( v237 != v238 )
    _libc_free((unsigned __int64)v238);
  return v202;
}
