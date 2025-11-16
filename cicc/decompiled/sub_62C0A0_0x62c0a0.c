// Function: sub_62C0A0
// Address: 0x62c0a0
//
__int64 *__fastcall sub_62C0A0(
        unsigned __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _QWORD **a5,
        __m128i *a6,
        __int64 *a7,
        __int64 *a8,
        int *a9,
        int *a10,
        int *a11,
        int *a12,
        __int64 **a13,
        _QWORD *a14,
        __int64 a15,
        __int64 *a16)
{
  __int64 v17; // r13
  __m128i *v18; // r12
  unsigned __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // rax
  __m128i v22; // xmm3
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // r10d
  unsigned __int16 v29; // ax
  __int64 *v30; // rdx
  _QWORD *v31; // rcx
  unsigned __int16 v32; // ax
  __int64 j; // rax
  __int64 *v34; // rsi
  __int8 v35; // al
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // r8
  __int64 v39; // rcx
  __m128i *v40; // rsi
  unsigned __int16 v41; // ax
  char v42; // al
  __int64 v43; // rdi
  __int64 *v44; // rcx
  __int64 v45; // rsi
  char v46; // al
  __int64 v47; // rsi
  __int64 v48; // rdi
  unsigned __int64 i; // rcx
  __int64 v50; // rdx
  char v51; // al
  __int64 v52; // r14
  unsigned __int64 v53; // rax
  int v54; // r14d
  int v55; // eax
  int v56; // r14d
  __int64 v57; // rdx
  unsigned __int64 v58; // rdx
  __int64 v59; // rcx
  int v60; // eax
  char *v61; // r14
  unsigned __int64 v62; // r14
  __int64 v63; // rdx
  __int64 v64; // rcx
  int v65; // edx
  unsigned int v66; // eax
  __int64 v67; // rdi
  int v68; // r8d
  unsigned __int16 v69; // ax
  __m128i v70; // xmm7
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rcx
  char v74; // al
  _QWORD *v75; // rax
  _DWORD *v76; // rdx
  int v77; // ecx
  _QWORD *v78; // r12
  unsigned __int64 v79; // r13
  _QWORD *v80; // rbx
  int v81; // r8d
  char v82; // al
  int v83; // edx
  __int64 v84; // rcx
  __int64 v85; // rdx
  char v86; // dl
  __int64 v87; // rax
  __int64 v88; // r8
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // rdx
  char v93; // dl
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // r14
  char k; // dl
  __int64 v99; // rcx
  __int64 v100; // rax
  __int64 v101; // rcx
  __int64 m; // rax
  __int64 v103; // rax
  __int64 v104; // rdi
  __int64 v105; // rax
  __m128i v106; // xmm7
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int8 v110; // al
  __int8 v111; // al
  __m128i *v112; // rsi
  __int64 v113; // rcx
  __m128i *v114; // rdi
  __int64 v115; // rax
  __int64 v116; // r14
  __int64 v117; // rdi
  char v118; // si
  char v119; // al
  char v120; // al
  __int64 v121; // r14
  __m128i v122; // xmm7
  __int64 v123; // rax
  __int64 v124; // rdx
  __int64 v125; // rdx
  __int64 v126; // r14
  __int64 v127; // rax
  char v128; // cl
  bool v129; // zf
  __int64 v130; // rax
  __int64 v131; // rdi
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rcx
  __m128i v135; // xmm3
  __int64 v136; // rax
  unsigned int v137; // eax
  __int64 v138; // rdx
  char v139; // dl
  __int64 v140; // rax
  __int64 v141; // rdx
  int v142; // ecx
  __int8 v143; // al
  char *v144; // rdx
  __m128i v145; // xmm7
  __int64 v146; // rax
  __m128i v147; // xmm5
  __m128i v148; // xmm6
  __m128i v149; // xmm7
  __int64 v150; // r14
  __int64 v151; // rsi
  __int64 v152; // r9
  unsigned __int64 v153; // rax
  __int64 v154; // rax
  __int64 v155; // rdx
  __int64 *v156; // rsi
  __m128i v157; // xmm7
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rcx
  unsigned int v162; // esi
  char v163; // al
  unsigned __int8 v164; // al
  int v165; // r8d
  __int64 v166; // rcx
  __int64 v167; // rsi
  __int64 v168; // rbx
  bool v169; // r14
  __int64 v170; // rax
  int v171; // edx
  int v172; // edx
  _BYTE *v173; // rdx
  __int64 v174; // rdx
  __int64 v175; // rcx
  __int64 v176; // rdx
  __int64 v177; // rcx
  __int64 v178; // rax
  __int64 v179; // rdx
  __int64 v180; // rdx
  __int64 v181; // rdi
  int v182; // eax
  __int64 v183; // rax
  unsigned int v184; // eax
  __int64 v185; // rdi
  __int64 v186; // rdx
  __int64 v187; // rax
  char v188; // si
  int v189; // eax
  __int64 v190; // rax
  __int64 v191; // rax
  unsigned __int8 v192; // al
  __int64 v193; // rdx
  __int64 v194; // rcx
  __int64 v195; // [rsp+8h] [rbp-138h]
  __int64 v196; // [rsp+10h] [rbp-130h]
  __int64 v197; // [rsp+10h] [rbp-130h]
  __int64 v198; // [rsp+10h] [rbp-130h]
  int v199; // [rsp+10h] [rbp-130h]
  int v200; // [rsp+18h] [rbp-128h]
  int v201; // [rsp+18h] [rbp-128h]
  int v202; // [rsp+18h] [rbp-128h]
  int v203; // [rsp+18h] [rbp-128h]
  int v204; // [rsp+18h] [rbp-128h]
  __int64 v205; // [rsp+18h] [rbp-128h]
  unsigned int v206; // [rsp+2Ch] [rbp-114h]
  int v207; // [rsp+30h] [rbp-110h]
  int v208; // [rsp+30h] [rbp-110h]
  int v209; // [rsp+38h] [rbp-108h]
  _BOOL4 v210; // [rsp+38h] [rbp-108h]
  int v211; // [rsp+38h] [rbp-108h]
  _DWORD *v212; // [rsp+38h] [rbp-108h]
  int v213; // [rsp+40h] [rbp-100h]
  bool v214; // [rsp+40h] [rbp-100h]
  unsigned int v215; // [rsp+40h] [rbp-100h]
  int v216; // [rsp+40h] [rbp-100h]
  int v217; // [rsp+40h] [rbp-100h]
  unsigned __int64 v218; // [rsp+48h] [rbp-F8h]
  unsigned int v219; // [rsp+48h] [rbp-F8h]
  _BOOL4 v220; // [rsp+48h] [rbp-F8h]
  int v221; // [rsp+48h] [rbp-F8h]
  __int16 v222; // [rsp+48h] [rbp-F8h]
  int v223; // [rsp+48h] [rbp-F8h]
  unsigned int v224; // [rsp+50h] [rbp-F0h]
  _BOOL4 v225; // [rsp+54h] [rbp-ECh]
  unsigned int v227; // [rsp+60h] [rbp-E0h]
  __m128i *v228; // [rsp+60h] [rbp-E0h]
  char v229; // [rsp+60h] [rbp-E0h]
  int v230; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v232; // [rsp+78h] [rbp-C8h]
  __int64 v233; // [rsp+78h] [rbp-C8h]
  __int64 v234; // [rsp+78h] [rbp-C8h]
  char v235[4]; // [rsp+84h] [rbp-BCh] BYREF
  char v236; // [rsp+88h] [rbp-B8h] BYREF
  int v237; // [rsp+8Ch] [rbp-B4h] BYREF
  __int64 v238; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v239; // [rsp+98h] [rbp-A8h] BYREF
  __int64 v240; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v241; // [rsp+A8h] [rbp-98h] BYREF
  __int64 *v242; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v243; // [rsp+B8h] [rbp-88h] BYREF
  _QWORD *v244; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v245; // [rsp+C8h] [rbp-78h] BYREF
  __int64 v246; // [rsp+D0h] [rbp-70h] BYREF
  char v247[12]; // [rsp+D8h] [rbp-68h] BYREF
  char v248[12]; // [rsp+E4h] [rbp-5Ch] BYREF
  _QWORD v249[10]; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v250; // [rsp+190h] [rbp+50h]

  v17 = a3;
  v18 = a6;
  v19 = a1;
  v242 = 0;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  v245 = *(_QWORD *)&dword_4F063F8;
  *a2 = 0;
  v232 = a1 & 1;
  v218 = a1 & 2;
  v209 = (a1 >> 3) & 1;
  v230 = v209;
  if ( (a1 & 0x40) != 0 )
  {
    v227 = 1;
    v225 = a4 != 0;
  }
  else
  {
    v225 = 0;
    v227 = 0;
  }
  v20 = (a1 >> 14) & 1;
  v224 = (a1 >> 14) & 1;
  if ( (a1 & 1) != 0 )
  {
    v21 = 0;
    if ( (a1 & 0x100) == 0 )
      v21 = a15;
    v250 = v21;
    if ( a6 )
    {
      *a6 = _mm_loadu_si128(xmmword_4F06660);
      a6[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a6[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v22 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v23 = *(_QWORD *)dword_4F07508;
      a6[1].m128i_i8[1] |= 0x20u;
      a6[3] = v22;
      a6->m128i_i64[1] = v23;
    }
  }
  else
  {
    v250 = 0;
    v18 = 0;
  }
  v24 = a3;
  v25 = a4;
  v26 = sub_626600(
          a4,
          a3,
          dword_4F077C4 == 2,
          (__int64)v247,
          (__int64)v248,
          (__int64)v235,
          (int)&v236,
          &v237,
          (__int64)a16);
  v238 = v26;
  if ( a4 != v26 && v26 )
  {
    v25 = (__int64)a2;
    v28 = v237;
    v27 = *a2;
    *a2 |= 0x40uLL;
    if ( v28 )
    {
      LOBYTE(v27) = v27 | 0xC0;
      *a2 = v27;
    }
    if ( a4 )
    {
      *(_QWORD *)(v17 + 280) = v26;
      *(_QWORD *)(v17 + 288) = v26;
    }
  }
  v239 = 0;
  v240 = 0;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  v29 = word_4F06418[0];
  if ( word_4F06418[0] != 27 )
  {
    v30 = (__int64 *)dword_4D04408;
    if ( dword_4D04408 && word_4F06418[0] == 76 )
    {
      v24 = 0;
      v25 = 0;
      if ( (unsigned __int16)sub_7BE840(0, 0) != 28 || (unsigned int)sub_867060() || *(_QWORD *)(v17 + 368) )
      {
        v74 = *(_BYTE *)(v17 + 123);
        if ( (v74 & 0x20) != 0 )
        {
          *(_BYTE *)(v17 + 123) = v74 | 0x40;
          v246 = *(_QWORD *)&dword_4F063F8;
          if ( (v19 & 0x220000) == 0x20000 || *(_QWORD *)(v17 + 368) )
          {
            sub_7B8B50(0, 0, v30, v73);
            v29 = word_4F06418[0];
            goto LABEL_17;
          }
          sub_867610();
        }
      }
      v29 = word_4F06418[0];
    }
LABEL_17:
    v31 = &dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      if ( v29 == 1 && (unk_4D04A11 & 2) != 0 )
      {
        if ( v232 )
          goto LABEL_54;
        goto LABEL_23;
      }
      v24 = 0;
      v25 = 0;
      if ( !(unsigned int)sub_7C0F00(0, 0) )
      {
        v29 = word_4F06418[0];
        goto LABEL_19;
      }
    }
    else if ( v29 != 1 )
    {
LABEL_19:
      if ( v29 != 37 && v29 != 156 )
      {
        if ( !v232 || v218 )
          goto LABEL_23;
LABEL_51:
        if ( word_4F06418[0] != 25 )
          goto LABEL_54;
        if ( !dword_4D0485C )
        {
          if ( dword_4F077C4 != 2 )
            goto LABEL_54;
          if ( unk_4F07778 > 201102 )
          {
            if ( dword_4F077BC )
            {
              if ( !(_DWORD)qword_4F077B4 )
              {
                if ( qword_4F077A8 <= 0x1116Fu )
                  goto LABEL_54;
                goto LABEL_334;
              }
              goto LABEL_333;
            }
LABEL_332:
            if ( !(_DWORD)qword_4F077B4 )
              goto LABEL_54;
LABEL_333:
            if ( unk_4F077A0 <= 0x9C3Fu )
              goto LABEL_54;
            goto LABEL_334;
          }
          if ( !dword_4F07774 )
            goto LABEL_54;
          v25 = dword_4F077BC;
          if ( !dword_4F077BC )
            goto LABEL_332;
          if ( (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1116Fu )
          {
            if ( !dword_4F07774 )
              goto LABEL_54;
            goto LABEL_332;
          }
        }
LABEL_334:
        if ( a4 && !v218 && (*(_BYTE *)(v17 + 121) & 0x40) == 0 )
        {
          v245 = *(_QWORD *)&dword_4F063F8;
          sub_87AA70(v18, &v245);
          v126 = sub_7ADF90(v18, &v245, v125);
          *a16 = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)(v17 + 48) = *(_QWORD *)&dword_4F063F8;
          if ( (*(_BYTE *)(v17 + 125) & 1) != 0 )
          {
            if ( !dword_4D0485C )
              sub_684B30(3356, &dword_4F063F8);
          }
          else if ( *(char *)(v17 + 121) >= 0 )
          {
            sub_6851C0(2825, v17 + 32);
          }
          v127 = *(_QWORD *)(v17 + 8);
          if ( (v127 & 2) != 0 )
          {
            sub_6851C0(2836, v17 + 88);
            v127 = *(_QWORD *)(v17 + 8);
          }
          else
          {
            if ( (v127 & 0x180000) == 0 )
            {
              v128 = *(_BYTE *)(v17 + 268);
              if ( v128 )
              {
                if ( dword_4F077C4 == 2 )
                {
                  if ( unk_4F07778 > 202001 && v128 == 2 )
                    goto LABEL_346;
                  if ( unk_4F07778 > 202001 )
                  {
                    sub_6851C0(2985, v17 + 260);
                    goto LABEL_345;
                  }
                }
              }
              else if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 || (v127 & 0x400000) == 0 )
              {
                goto LABEL_346;
              }
              sub_6851C0(2838, v17 + 260);
LABEL_345:
              *(_BYTE *)(v17 + 269) = 0;
              v127 = *(_QWORD *)(v17 + 8);
              goto LABEL_346;
            }
            sub_6851C0(2837, v17 + 112);
            v127 = *(_QWORD *)(v17 + 8);
          }
LABEL_346:
          v129 = dword_4F077C4 == 2;
          *(_QWORD *)(v17 + 8) = v127 & 0xFFFFFFFFFFE7FFFDLL;
          if ( !v129 || unk_4F07778 <= 202001 )
            *(_QWORD *)(v17 + 8) = v127 & 0xFFFFFFFFFFA7FFFDLL;
          v130 = *(_QWORD *)(v17 + 288);
          if ( *(_BYTE *)(v130 + 140) == 6 && (*(_BYTE *)(v130 + 168) & 1) == 0 )
            sub_6851C0(2826, v17 + 40);
          if ( (*(_BYTE *)(v17 + 120) & 2) != 0 )
          {
            v131 = 4;
            if ( dword_4F077C4 == 2 )
              v131 = (unsigned int)(unk_4F07778 >= 202002) + 4;
            sub_684AA0(v131, 3015, v17 + 72);
          }
          sub_7ADF70(v126, 0);
          sub_7C6040(v126, 0);
          v132 = qword_4F063F0;
          a16[7] = qword_4F063F0;
          unk_4F061D8 = v132;
          sub_7AE360(v126);
          sub_7B8B50(v126, a16, v133, v134);
          sub_7AE210(v126);
          *(_BYTE *)(v17 + 131) |= 0x10u;
          *(_QWORD *)(v17 + 368) = v126;
          if ( (*(_BYTE *)(v17 + 130) & 0x20) == 0 )
          {
            if ( (unsigned __int16)(word_4F06418[0] - 27) > 0x2Eu
              || (v186 = 0x400020000001LL, !_bittest64(&v186, (unsigned int)word_4F06418[0] - 27)) )
            {
              sub_6851C0(2827, &dword_4F063F8);
            }
          }
          goto LABEL_32;
        }
LABEL_54:
        *a14 = sub_869D30();
        v37 = *(_QWORD **)(v17 + 416);
        if ( v37 )
        {
          while ( 1 )
          {
            v37 = (_QWORD *)*v37;
            if ( !v37 )
              break;
            *(_QWORD *)(v17 + 416) = v37;
          }
        }
        *a2 |= 2uLL;
        v249[0] = *(_QWORD *)&dword_4F063F8;
        if ( a16 )
          a16[2] = *(_QWORD *)&dword_4F063F8;
        v206 = 0;
        v38 = (v19 & 4) == 0 ? 14 : 2;
        if ( (v19 & 0x80u) != 0LL )
        {
          if ( (v19 & 0x1000) != 0 )
          {
            v206 = 1;
            v38 = (unsigned int)v38 | 0x2080;
          }
          else
          {
            LOBYTE(v38) = ((v19 & 4) == 0 ? 14 : 2) | 0x80;
          }
        }
        if ( (v19 & 0x400) != 0 )
          v38 = (unsigned int)v38 | 0x100000;
        v39 = (unsigned int)dword_4F077C4;
        if ( unk_4D043C8 )
          goto LABEL_244;
        if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
        {
          v25 = dword_4F077BC;
          if ( dword_4F077BC )
          {
            if ( !(_DWORD)qword_4F077B4 )
            {
              if ( qword_4F077A8 <= 0xC34Fu )
                goto LABEL_67;
LABEL_244:
              v220 = 1;
              if ( (v38 & 0x80u) != 0LL )
                goto LABEL_68;
              goto LABEL_67;
            }
          }
          else if ( !(_DWORD)qword_4F077B4 )
          {
            goto LABEL_67;
          }
          if ( unk_4F077A0 > 0x76BFu )
            goto LABEL_244;
        }
LABEL_67:
        v220 = 1;
        if ( (v19 & 0x83000) == 0 )
        {
          v220 = 0;
          if ( (v19 & 0x400) != 0 )
            v220 = (v38 & 0x80u) == 0LL;
        }
LABEL_68:
        v40 = 0;
        if ( a5 )
          v40 = (__m128i *)((v19 >> 12) & 1);
        v207 = (int)v40;
        v41 = word_4F06418[0];
        if ( dword_4F077C4 == 2 )
        {
          if ( word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0 )
          {
            v40 = 0;
            v25 = 2;
            v215 = v38;
            if ( !(unsigned int)sub_7C0F00(2, 0) )
              goto LABEL_313;
            v38 = v215;
          }
        }
        else if ( word_4F06418[0] != 1 )
        {
          if ( (v19 & 0x400) != 0 )
            goto LABEL_277;
LABEL_381:
          sub_64E7D0(v25, v40, v36, v39, v38);
LABEL_314:
          v41 = word_4F06418[0];
          if ( word_4F06418[0] == 1 )
          {
            if ( (unk_4D04A10 & 0x20) != 0 )
            {
              *a11 = 1;
              if ( (unk_4D04A11 & 0x20) == 0 )
              {
                if ( (v19 & 0x400) != 0 && (unk_4D04A10 & 1) == 0 )
                {
                  v40 = (__m128i *)dword_4F07508;
                  v25 = 1010;
                }
                else
                {
                  if ( (v19 & 0x100) == 0 )
                  {
                    v121 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
                    if ( *(_BYTE *)(v121 + 4) == 6 )
                    {
                      v25 = **(_QWORD **)(v121 + 208);
                      if ( (unsigned int)sub_87A630(v25, v40, v36, v39, v38) )
                      {
                        *v18 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
                        v184 = dword_4F06650[0];
                        v18[1] = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
                        v18[2] = _mm_loadu_si128(&xmmword_4D04A20);
                        v18[3] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
                        *(_DWORD *)(v17 + 68) = v184;
                        a5 = *(_QWORD ***)(*(_QWORD *)(v121 + 184) + 32LL);
                      }
                      else
                      {
                        sub_6851C0(279, dword_4F07508);
                        v40 = xmmword_4F06660;
                        v39 = 16;
                        v25 = (__int64)v18;
                        while ( v39 )
                        {
                          *(_DWORD *)v25 = v40->m128i_i32[0];
                          v40 = (__m128i *)((char *)v40 + 4);
                          v25 += 4;
                          --v39;
                        }
                        v191 = *(_QWORD *)dword_4F07508;
                        v18[1].m128i_i8[1] |= 0x20u;
                        v18->m128i_i64[1] = v191;
                      }
                      goto LABEL_323;
                    }
                    v40 = (__m128i *)dword_4F07508;
                    v25 = 279;
                    sub_6851C0(279, dword_4F07508);
                    goto LABEL_322;
                  }
                  v40 = (__m128i *)dword_4F07508;
                  v25 = 279;
                }
                sub_6851C0(v25, dword_4F07508);
                *v18 = _mm_loadu_si128(xmmword_4F06660);
                v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
                v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
                v183 = *(_QWORD *)dword_4F07508;
                v18[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
                v18[1].m128i_i8[1] |= 0x20u;
                v18->m128i_i64[1] = v183;
                goto LABEL_323;
              }
LABEL_322:
              *v18 = _mm_loadu_si128(xmmword_4F06660);
              v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
              v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
              v122 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v123 = *(_QWORD *)dword_4F07508;
              v18[1].m128i_i8[1] |= 0x20u;
              v18[3] = v122;
              v18->m128i_i64[1] = v123;
LABEL_323:
              if ( a16 )
              {
                v124 = qword_4F063F0;
                v39 = (__int64)a16;
                a16[3] = qword_4F063F0;
                a16[7] = v124;
              }
              unk_4F061D8 = qword_4F063F0;
              sub_7B8B50(v25, v40, qword_4F063F0, v39);
              v213 = 0;
              v230 = 0;
LABEL_280:
              v109 = v18[1].m128i_i64[1];
              if ( v109 && *(_BYTE *)(v109 + 80) == 23 )
              {
                sub_6851C0(728, v249);
                *v18 = _mm_loadu_si128(xmmword_4F06660);
                v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
                v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
                v159 = *(_QWORD *)dword_4F07508;
                v18[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
                v18[1].m128i_i8[1] |= 0x20u;
                v18->m128i_i64[1] = v159;
              }
              if ( !v220 && (v18[1].m128i_i32[0] & 0x12000) == 0x10000 )
              {
                sub_6851C0(891, v249);
                *v18 = _mm_loadu_si128(xmmword_4F06660);
                v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
                v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
                v160 = *(_QWORD *)dword_4F07508;
                v18[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
                v18[1].m128i_i8[1] |= 0x20u;
                v18->m128i_i64[1] = v160;
              }
              v110 = v18[1].m128i_i8[0];
              if ( (v110 & 8) == 0 )
              {
                if ( (v110 & 0x10) != 0 )
                {
                  if ( !a5 || !v18[1].m128i_i64[1] && (v19 & 0x10) == 0 && !v207 )
                  {
                    sub_6851C0(436, &v18->m128i_u64[1]);
                    *v18 = _mm_loadu_si128(xmmword_4F06660);
                    v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
                    v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
                    v135 = _mm_loadu_si128(&xmmword_4F06660[3]);
                    v18[1].m128i_i16[0] |= 0x2010u;
                    v136 = *(_QWORD *)dword_4F07508;
                    v18[3] = v135;
                    v18->m128i_i64[1] = v136;
                  }
                }
                else if ( (v110 & 0x40) != 0 && (v19 & 0x100) != 0 )
                {
                  sub_6851C0(502, &v18->m128i_u64[1]);
                  *v18 = _mm_loadu_si128(xmmword_4F06660);
                  v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
                  v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
                  v178 = *(_QWORD *)dword_4F07508;
                  v18[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
                  v18[1].m128i_i8[1] |= 0x20u;
                  v18->m128i_i64[1] = v178;
                }
                goto LABEL_301;
              }
              if ( (v19 & 0x100) != 0 )
              {
                sub_6851C0(502, &v18->m128i_u64[1]);
              }
              else
              {
                if ( a5 )
                {
                  v230 = 0;
                  if ( v207 )
                    goto LABEL_301;
                  if ( v18[1].m128i_i64[1] | v19 & 0x10 )
                    goto LABEL_301;
                  v111 = v18[3].m128i_i8[8];
                  if ( (unsigned __int8)(v111 - 1) <= 3u )
                    goto LABEL_301;
                  if ( v111 != 42 )
                  {
                    sub_6851C0(342, &v18->m128i_u64[1]);
                    v112 = xmmword_4F06660;
                    v113 = 16;
                    v114 = v18;
                    while ( v113 )
                    {
                      v114->m128i_i32[0] = v112->m128i_i32[0];
                      v112 = (__m128i *)((char *)v112 + 4);
                      v114 = (__m128i *)((char *)v114 + 4);
                      --v113;
                    }
                    v115 = *(_QWORD *)dword_4F07508;
                    v18[1].m128i_i8[1] |= 0x20u;
                    v18->m128i_i64[1] = v115;
                    goto LABEL_301;
                  }
                  if ( dword_4F077C4 == 2 && unk_4F07778 > 202301 )
                    goto LABEL_301;
                  if ( dword_4F077BC )
                  {
                    if ( !(_DWORD)qword_4F077B4 )
                    {
                      v192 = qword_4F077A8 < 0x1FBD0u ? 7 : 5;
LABEL_677:
                      sub_684AA0(v192, 342, &v18->m128i_u64[1]);
                      v230 = 0;
                      goto LABEL_301;
                    }
                  }
                  else
                  {
                    v192 = 7;
                    if ( !(_DWORD)qword_4F077B4 )
                      goto LABEL_677;
                  }
                  v192 = unk_4F077A0 < 0x27100u ? 7 : 5;
                  goto LABEL_677;
                }
                v143 = v18[3].m128i_i8[8];
                if ( v143 == 42 )
                {
                  v144 = "()";
                }
                else if ( (unsigned __int8)v143 > 0x2Au )
                {
                  v144 = "[]";
                  if ( v143 != 43 )
                    goto LABEL_399;
                }
                else if ( v143 == 15 )
                {
                  v144 = "=";
                  if ( HIDWORD(qword_4D0495C) )
                  {
                    sub_684B10(341, &v18->m128i_u64[1], "=");
                    v230 = 0;
                    goto LABEL_301;
                  }
                }
                else
                {
                  if ( v143 != 41 )
                  {
LABEL_399:
                    v230 = 0;
LABEL_301:
                    v25 = 6;
                    v243 = sub_5CC190(6);
                    v116 = v243;
                    if ( v243 )
                    {
                      if ( HIDWORD(qword_4F077B4) && *(_BYTE *)(v17 + 268) != 4 )
                      {
                        v117 = *(_QWORD *)(v17 + 280);
                        v244 = 0;
                        v118 = 7;
                        v119 = *(_BYTE *)(sub_8D21F0(v117) + 140);
                        if ( v119 != 6 )
                          v118 = 2 * (v119 == 13) + 5;
                        v31 = &v244;
                        v30 = &v243;
                        do
                        {
                          v120 = *(_BYTE *)(v116 + 11);
                          if ( (*(_BYTE *)(v116 + 9) == 2 || (v120 & 0x10) != 0) && (v120 & 2) != 0 )
                          {
                            *v30 = *(_QWORD *)v116;
                            *(_BYTE *)(v116 + 10) = v118;
                            *v31 = v116;
                            v31 = (_QWORD *)v116;
                          }
                          else
                          {
                            v30 = (__int64 *)v116;
                          }
                          v116 = *v30;
                        }
                        while ( *v30 );
                        if ( v244 )
                          sub_5CF030((__int64 *)(v17 + 280), v244, v17);
                        v116 = v243;
                      }
                      v25 = v17 + 200;
                      if ( *(_QWORD *)(v17 + 200) )
                        v25 = (__int64)sub_5CB9F0((_QWORD **)v25);
                      *(_QWORD *)v25 = v116;
                    }
                    if ( (*(_BYTE *)(v17 + 122) & 2) == 0 )
                      v238 = *(_QWORD *)(v17 + 280);
                    v24 = (__int64)a16;
                    v245 = v18->m128i_i64[1];
                    *a16 = v245;
                    goto LABEL_27;
                  }
                  v144 = "->";
                }
                sub_6851A0(341, &v18->m128i_u64[1], v144);
              }
              *v18 = _mm_loadu_si128(xmmword_4F06660);
              v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
              v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
              v145 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v146 = *(_QWORD *)dword_4F07508;
              v18[1].m128i_i8[1] |= 0x20u;
              v18[3] = v145;
              v18->m128i_i64[1] = v146;
              goto LABEL_399;
            }
            goto LABEL_278;
          }
LABEL_277:
          if ( v41 == 179 && (*(_BYTE *)(v17 + 130) & 4) != 0 )
          {
            v104 = 3195;
            *(_QWORD *)dword_4F07508 = *(_QWORD *)(v17 + 24);
            goto LABEL_279;
          }
LABEL_278:
          v104 = 40;
LABEL_279:
          v105 = qword_4F061C8;
          ++*(_BYTE *)(qword_4F061C8 + 35LL);
          ++*(_BYTE *)(v105 + 33);
          *v18 = _mm_loadu_si128(xmmword_4F06660);
          v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
          v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
          v106 = _mm_loadu_si128(&xmmword_4F06660[3]);
          v18[1].m128i_i8[1] |= 0x20u;
          v107 = *(_QWORD *)&dword_4F063F8;
          v18[3] = v106;
          v18->m128i_i64[1] = v107;
          sub_6851D0(v104);
          v108 = qword_4F061C8;
          v213 = 0;
          v230 = 0;
          --*(_BYTE *)(qword_4F061C8 + 35LL);
          --*(_BYTE *)(v108 + 33);
          goto LABEL_280;
        }
        v42 = unk_4D04A10;
        v39 = unk_4D04A10 & 0x21;
        if ( (_BYTE)v39 != 32 )
        {
          v43 = unk_4D04A10 & 1;
          v44 = &qword_4D0495C;
          v45 = (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
          if ( qword_4D0495C )
          {
            if ( !(_BYTE)v43 )
              goto LABEL_80;
            v46 = unk_4D04A12;
            if ( (unk_4D04A12 & 2) == 0 )
              goto LABEL_469;
            v44 = (__int64 *)xmmword_4D04A20.m128i_i64[0];
            if ( !xmmword_4D04A20.m128i_i64[0] || (v19 & 0x100) == 0 )
            {
LABEL_76:
              if ( (v19 & 0x400) == 0 )
              {
                v217 = v38;
                sub_64E7D0(v43, v45, v36, v44, v38);
                LODWORD(v38) = v217;
                v46 = unk_4D04A12;
              }
              if ( (v46 & 0x20) != 0 )
              {
                v200 = v38;
                sub_684AA0(((unsigned int)qword_4F077B4 | dword_4D04964) == 0 ? 5 : 8, 2613, &qword_4D04A08);
                LODWORD(v38) = v200;
              }
              goto LABEL_80;
            }
            a5 = (_QWORD **)xmmword_4D04A20.m128i_i64[0];
            *a2 |= 4uLL;
            v42 = unk_4D04A10 & 0xFE;
            unk_4D04A10 &= ~1u;
          }
          if ( (v42 & 1) == 0 )
            goto LABEL_80;
          v46 = unk_4D04A12;
          if ( (unk_4D04A12 & 2) != 0 )
            goto LABEL_76;
LABEL_469:
          v161 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( *(_BYTE *)(v161 + 4) == 8 )
          {
            v161 -= 776;
            v162 = 1;
          }
          else
          {
            v162 = dword_4F077BC;
            if ( dword_4F077BC )
            {
              v162 = qword_4F077B4;
              if ( (_DWORD)qword_4F077B4 )
              {
                v162 = 0;
              }
              else if ( ((v19 & 0x400) != 0 || qword_4F077A8 <= 0x9CA4u && dword_4F04C58 != -1) && (v46 & 1) == 0 )
              {
                v161 = qword_4F04C68[0] + 776LL * unk_4F04C34;
              }
            }
          }
          if ( (unk_4D04A11 & 0x20) == 0 )
          {
            v163 = *(_BYTE *)(v161 + 4);
            if ( (unsigned __int8)(v163 - 3) <= 1u )
            {
              if ( *(_QWORD *)(*(_QWORD *)(v161 + 184) + 32LL) != xmmword_4D04A20.m128i_i64[0] )
                goto LABEL_80;
LABEL_481:
              if ( v162 && !(dword_4F077BC | unk_4D047C8) && xmmword_4D04A20.m128i_i64[0] )
              {
                v196 = v161;
                v201 = v38;
                sub_878790(&qword_4D04A00);
                v164 = 8;
                unk_4D04A18 = 0;
                unk_4D04A11 |= 0x20u;
                v165 = v201;
                v166 = v196;
                goto LABEL_485;
              }
              if ( (v19 & 0x3000) != 0 )
              {
                if ( !dword_4D04964 )
                {
                  v197 = v161;
                  v202 = v38;
                  sub_878790(&qword_4D04A00);
                  v164 = 4;
                  v165 = v202;
                  v166 = v197;
LABEL_601:
                  v167 = 2411;
                  if ( dword_4F04C58 == -1 )
                  {
                    if ( !dword_4F077BC || (v167 = 970, (v19 & 0x400) == 0) )
                      v167 = *(_BYTE *)(v166 + 4) == 0 ? 1372 : 1285;
                  }
                  goto LABEL_487;
                }
              }
              else if ( !dword_4D04964 )
              {
                v195 = v161;
                v199 = v38;
                v205 = unk_4D04A18;
                v187 = sub_7CFB70(&qword_4D04A00, 0);
                v165 = v199;
                v166 = v195;
                unk_4D04A18 = v205;
                if ( v187 )
                {
                  v188 = *(_BYTE *)(v187 + 80);
                  if ( (unsigned __int8)(v188 - 4) > 2u
                    && (v188 != 3 || !*(_BYTE *)(v187 + 104))
                    && v195 == qword_4F04C68[0] + 776LL * dword_4F04C64 )
                  {
                    if ( !dword_4D04964 )
                    {
                      v164 = 4;
                      goto LABEL_601;
                    }
LABEL_641:
                    v164 = 7;
LABEL_485:
                    v167 = 1372;
                    if ( *(_BYTE *)(v166 + 4) )
                      v167 = 756;
LABEL_487:
                    v216 = v165;
                    sub_684AA0(v164, v167, &dword_4F063F8);
                    LODWORD(v38) = v216;
                    goto LABEL_80;
                  }
                }
                v198 = v195;
                v204 = v165;
                if ( dword_4F077BC )
                {
                  sub_878790(&qword_4D04A00);
                  v164 = 5;
                  v165 = v204;
                  v166 = v195;
                  goto LABEL_601;
                }
LABEL_640:
                sub_878790(&qword_4D04A00);
                v165 = v204;
                v166 = v198;
                goto LABEL_641;
              }
              v198 = v161;
              v204 = v38;
              goto LABEL_640;
            }
            if ( !v163 )
            {
              if ( xmmword_4D04A20.m128i_i64[0] )
                goto LABEL_80;
              goto LABEL_481;
            }
            if ( v163 == 9 )
            {
              v203 = v38;
              sub_878790(&qword_4D04A00);
              LODWORD(v38) = v203;
            }
          }
LABEL_80:
          v47 = 12;
          v48 = (unsigned int)v38;
          v213 = sub_7C8410((unsigned int)v38, 12, &v241);
          if ( v213 )
          {
            v213 = v241;
            if ( (_DWORD)v241 )
            {
              v213 = 0;
              goto LABEL_420;
            }
            if ( (unk_4D04A10 & 1) == 0 )
              goto LABEL_369;
            if ( (unk_4D04A12 & 2) == 0 )
            {
LABEL_566:
              a5 = (_QWORD **)xmmword_4D04A20.m128i_i64[0];
              if ( xmmword_4D04A20.m128i_i64[0] )
              {
                v48 = xmmword_4D04A20.m128i_i64[0];
                sub_864360(xmmword_4D04A20.m128i_i64[0], 0);
                a5 = 0;
                v47 = (unsigned int)v241;
                *a2 |= 8uLL;
                goto LABEL_368;
              }
LABEL_510:
              v47 = (unsigned int)v241;
              goto LABEL_368;
            }
            v50 = xmmword_4D04A20.m128i_i64[0];
            a5 = (_QWORD **)xmmword_4D04A20.m128i_i64[0];
            if ( !xmmword_4D04A20.m128i_i64[0] )
            {
LABEL_369:
              *v18 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
              v137 = dword_4F06650[0];
              v18[1] = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
              v18[2] = _mm_loadu_si128(&xmmword_4D04A20);
              v18[3] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
              *(_DWORD *)(v17 + 68) = v137;
              if ( a16 )
              {
                v138 = qword_4F063F0;
                i = (unsigned __int64)a16;
                a16[3] = qword_4F063F0;
                a16[7] = v138;
              }
              unk_4F061D8 = qword_4F063F0;
              sub_7B8B50(v48, v47, qword_4F063F0, i);
              goto LABEL_280;
            }
            v51 = *(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 140);
            if ( v51 == 12 )
            {
              do
              {
                v50 = *(_QWORD *)(v50 + 160);
                v51 = *(_BYTE *)(v50 + 140);
              }
              while ( v51 == 12 );
              a5 = (_QWORD **)v50;
            }
            if ( v51 == 14 )
            {
              v48 = (__int64)a5;
              a5 = (_QWORD **)sub_7D0530(a5);
              if ( !a5 )
              {
                if ( (unk_4D04A12 & 2) != 0 )
                  goto LABEL_510;
                goto LABEL_566;
              }
            }
            v52 = unk_4D04A18;
            v53 = *(unsigned __int8 *)(unk_4D04A18 + 80LL);
            if ( (unsigned __int8)v53 <= 0x14u )
            {
              v180 = 1180672;
              if ( _bittest64(&v180, v53) )
              {
                if ( (unsigned __int8)sub_877F80(unk_4D04A18) == 1 )
                {
                  v54 = 0;
                  *a9 = 1;
                }
                else
                {
                  v185 = v52;
                  v54 = 0;
                  if ( (unsigned __int8)sub_877F80(v185) == 2 )
                    *a11 = 1;
                }
                goto LABEL_95;
              }
            }
            if ( (((_BYTE)v53 - 7) & 0xFD) == 0 && (*(_BYTE *)(unk_4D04A18 + 81LL) & 0x10) != 0 )
            {
              if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
              {
                v54 = 1;
                sub_8602F0();
                v213 = v209;
              }
              else
              {
                v54 = 1;
                v213 = v209;
              }
LABEL_95:
              v48 = (__int64)a5;
              sub_865D70(a5, 0, v206, 0, 0, 0);
              *a2 |= 8uLL;
              if ( dword_4F04C40 == -1
                || (i = (unsigned __int64)qword_4F04C68,
                    !*(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456)) )
              {
                v182 = v213;
                v47 = (unsigned int)v241;
                v213 = v54;
                v230 = v182;
              }
              else
              {
                sub_87DD80();
                v48 = dword_4F04C40;
                sub_8845B0(dword_4F04C40);
                v55 = v213;
                v47 = (unsigned int)v241;
                v213 = v54;
                v230 = v55;
              }
              goto LABEL_368;
            }
            if ( (*(_BYTE *)(v17 + 8) & 8) == 0 )
              goto LABEL_510;
            if ( (*(_BYTE *)(v17 + 122) & 2) == 0 )
              goto LABEL_510;
            if ( (unk_4D04A12 & 3) != 2 )
              goto LABEL_510;
            if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
              goto LABEL_510;
            if ( (*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 177) & 0x20) == 0 )
              goto LABEL_510;
            if ( qword_4D04A00 != **a5 )
              goto LABEL_510;
            v48 = 0;
            if ( (unsigned __int16)sub_7BE840(0, 0) != 28 )
              goto LABEL_510;
            v47 = (unsigned int)v241;
            *a9 = 1;
          }
          else
          {
            v47 = (unsigned int)v241;
            if ( dword_4F077C4 == 2 )
            {
              v170 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              v171 = *(unsigned __int8 *)(v170 + 4);
              for ( i = (unsigned int)(v171 - 8); (unsigned __int8)(v171 - 8) <= 1u; i = (unsigned int)(v171 - 8) )
              {
                v171 = *(unsigned __int8 *)(v170 - 772);
                v170 -= 776;
              }
              if ( (unsigned __int8)(v171 - 6) <= 1u )
              {
                if ( (_DWORD)v241 )
                  goto LABEL_420;
                v213 = 0;
                if ( (v19 & 0x10000) == 0 )
                  goto LABEL_369;
                v48 = *(_QWORD *)(v170 + 208);
                v213 = sub_672080(v48, v17);
                if ( v213 )
                {
                  v213 = 0;
                  *a9 = 1;
                }
                goto LABEL_510;
              }
            }
          }
LABEL_368:
          if ( !(_DWORD)v47 )
            goto LABEL_369;
LABEL_420:
          if ( (v19 & 4) == 0 && ((unk_4D04A12 & 2) != 0 || xmmword_4D04A20.m128i_i64[0]) )
          {
            v147 = _mm_loadu_si128(&xmmword_4F06660[1]);
            v148 = _mm_loadu_si128(&xmmword_4F06660[2]);
            v149 = _mm_loadu_si128(&xmmword_4F06660[3]);
            qword_4D04A00 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
            unk_4D04A10 = v147;
            xmmword_4D04A20 = v148;
            unk_4D04A11 = v147.m128i_i8[1] | 0x20;
            qword_4D04A08 = *(_QWORD *)dword_4F07508;
            unk_4D04A30 = v149;
          }
          else
          {
            unk_4D04A11 |= 0x20u;
            unk_4D04A18 = 0;
          }
          goto LABEL_369;
        }
LABEL_313:
        if ( (v19 & 0x400) != 0 )
          goto LABEL_314;
        goto LABEL_381;
      }
    }
    if ( v232 )
      goto LABEL_51;
LABEL_23:
    if ( (*(_BYTE *)(v17 + 123) & 0x40) != 0 && !a4 )
    {
      v24 = (__int64)&v246;
      v25 = 2101;
      sub_6851C0(2101, &v246);
    }
    v213 = 0;
    v230 = 0;
LABEL_27:
    v219 = 0;
    v32 = word_4F06418[0];
    goto LABEL_28;
  }
  v214 = (*(_BYTE *)(v17 + 122) & 2) != 0;
  sub_7B8B50(v25, v24, v27, dword_4F07508);
  v67 = sub_5CC190(1);
  if ( v67 )
    sub_5CC9F0(v67);
  if ( v218 )
  {
    v246 = 0;
    if ( word_4F06418[0] == 28 )
    {
LABEL_251:
      v213 = 0;
      v210 = 0;
      v219 = 0;
      goto LABEL_111;
    }
    v68 = sub_868D90(&v246, v249, 1, 0, 0);
    v69 = word_4F06418[0];
    if ( !v68 || word_4F06418[0] == 28 )
    {
LABEL_248:
      if ( v69 == 76 && v246 )
        sub_867030(v246);
      goto LABEL_251;
    }
    if ( (unsigned int)sub_651B00(6) || word_4F06418[0] == 76 && (unsigned __int16)sub_7BE840(0, 0) == 28 )
    {
      v69 = word_4F06418[0];
      goto LABEL_248;
    }
    if ( v246 )
      sub_867030(v246);
  }
  *(_BYTE *)(v17 + 122) |= 2u;
  v75 = (_QWORD *)sub_5CC190(11);
  v76 = (_DWORD *)&qword_4F077B4 + 1;
  if ( v75 )
  {
    if ( HIDWORD(qword_4F077B4) )
    {
      v228 = v18;
      v77 = 0;
      v78 = v75;
      v221 = v20;
      v20 = v17;
      v79 = v19;
      v80 = v75;
      do
      {
        if ( *((_BYTE *)v80 + 9) != 2 && (*((_BYTE *)v80 + 11) & 0x10) == 0 && !v77 )
        {
          v212 = v76;
          sub_6851C0(2408, v80 + 7);
          v76 = v212;
          v77 = 1;
        }
        v80 = (_QWORD *)*v80;
      }
      while ( v80 );
      v75 = v78;
      v19 = v79;
      v18 = v228;
      v17 = v20;
      LODWORD(v20) = v221;
    }
    else
    {
      sub_6851C0(1098, v75 + 7);
      v76 = (_DWORD *)&qword_4F077B4 + 1;
      v75 = 0;
    }
  }
  v81 = *v76;
  v242 = v75;
  if ( v81 )
  {
    if ( a4 )
      goto LABEL_178;
    if ( !v238 && a13 )
    {
      *a13 = v75;
LABEL_178:
      a13 = &v242;
      if ( v75 )
        a13 = sub_5CB9F0(&v242);
      goto LABEL_180;
    }
    if ( v75 )
      sub_6851C0(1098, v75 + 7);
    v242 = 0;
    a13 = 0;
  }
LABEL_180:
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  v229 = *(_BYTE *)(v17 + 120) & 0x7F;
  v211 = *(_DWORD *)(v17 + 72);
  v222 = *(_WORD *)(v17 + 76);
  sub_62C0A0(
    v19 & 0xFFFFFFF7,
    (unsigned int)v249,
    v17,
    0,
    (_DWORD)a5,
    (_DWORD)v18,
    (__int64)&v239,
    (__int64)&v240,
    (__int64)a9,
    (__int64)a10,
    (__int64)a11,
    (__int64)a12,
    (__int64)a13,
    (__int64)a14,
    v250,
    (__int64)a16);
  *(_BYTE *)(v17 + 122) = *(_BYTE *)(v17 + 122) & 0xFD | (2 * v214);
  *(_BYTE *)(v17 + 120) = v229 | *(_BYTE *)(v17 + 120) & 0x80;
  v82 = v249[0];
  *(_DWORD *)(v17 + 72) = v211;
  *(_WORD *)(v17 + 76) = v222;
  if ( (v82 & 0x40) != 0 )
  {
    *(_BYTE *)(v17 + 124) |= 8u;
    v83 = 1;
    if ( (v19 & 0x40000) == 0 )
      v83 = v20;
    v224 = v83;
    v84 = *a2;
    v85 = *a2 | 0x40;
    if ( v82 < 0 )
    {
      LOBYTE(v84) = *a2 | 0xC0;
      v85 = v84;
    }
    *a2 = v85;
  }
  if ( (v82 & 2) != 0 )
  {
    *a2 |= 2uLL;
    v245 = *(_QWORD *)dword_4F07508;
  }
  else
  {
    v230 = 0;
  }
  if ( (v82 & 4) != 0 )
  {
    *a2 |= 4uLL;
  }
  else
  {
    if ( a5 )
      goto LABEL_192;
    if ( !v18 )
    {
LABEL_463:
      a5 = 0;
      goto LABEL_192;
    }
    if ( (v18[1].m128i_i8[1] & 0x20) != 0 && (*a9 || *a10 || *a11 || *a12) )
    {
      *a10 = 0;
      *a9 = 0;
      *a12 = 0;
      *a11 = 0;
      goto LABEL_192;
    }
  }
  if ( (v18[1].m128i_i8[2] & 2) == 0 )
    goto LABEL_463;
  a5 = (_QWORD **)v18[2].m128i_i64[0];
LABEL_192:
  if ( (v82 & 8) != 0 )
    *a2 |= 8uLL;
  v24 = 18;
  v25 = 28;
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  v32 = word_4F06418[0];
  if ( word_4F06418[0] == 142 )
  {
    if ( (*(_BYTE *)(v17 + 122) & 2) != 0 )
      goto LABEL_31;
    v25 = v17;
    sub_650E40(v17);
    v32 = word_4F06418[0];
  }
  v213 = 0;
  v227 = 0;
  v219 = 0;
  while ( 1 )
  {
LABEL_28:
    switch ( v32 )
    {
      case 0x1Bu:
        v210 = 0;
        break;
      case 0x19u:
        v210 = 0;
        v62 = v19 & 0x800;
        if ( (v19 & 0x10) == 0 )
          goto LABEL_129;
        goto LABEL_155;
      case 0x1Cu:
        if ( !v225 )
          goto LABEL_31;
        if ( !dword_4F077BC )
          goto LABEL_31;
        if ( qword_4F077A8 > 0x76BFu )
          goto LABEL_31;
        v24 = 0;
        v25 = 0;
        if ( (unsigned __int16)sub_7BE840(0, 0) != 25 )
          goto LABEL_31;
        sub_7B8B50(0, 0, v63, v64);
        --*(_BYTE *)(qword_4F061C8 + 36LL);
        *a2 |= 0x100uLL;
        v210 = v225;
        if ( word_4F06418[0] != 27 )
        {
          v62 = v19 & 0x800;
          if ( (v19 & 0x10) == 0 )
          {
LABEL_129:
            if ( v62 && !v239 )
            {
              v66 = 1;
              v65 = 0;
              goto LABEL_133;
            }
            goto LABEL_131;
          }
LABEL_155:
          if ( !v239 )
          {
            v65 = 1;
            if ( v62 )
            {
              v66 = 1;
              v65 = 1;
              goto LABEL_133;
            }
LABEL_132:
            v66 = 0;
LABEL_133:
            sub_625AB0(v17, &v241, v227, v224, (v19 >> 15) & 1, 0, v65, v66, (__int64)a16);
            v227 = 0;
            goto LABEL_134;
          }
LABEL_131:
          v65 = 0;
          goto LABEL_132;
        }
        break;
      default:
        goto LABEL_31;
    }
    v246 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(v25, v24, v30, v31);
    if ( v230 )
    {
      if ( word_4F06418[0] != 28 && word_4F06418[0] != 76 )
      {
        v56 = sub_868D90(&v243, &v244, 1, 0, 0);
        if ( dword_4F04C44 != -1 )
          break;
        v179 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v179 + 6) & 6) != 0 || *(_BYTE *)(v179 + 4) == 12 || !v244 || !*((_BYTE *)v244 + 61) )
          break;
      }
    }
LABEL_111:
    v60 = dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      if ( v239 )
      {
        v150 = v240;
        if ( v240 && (unsigned int)sub_8D3D10(v240) )
        {
          a5 = (_QWORD **)sub_8D4890(v150);
          *a10 = 0;
          *a9 = 0;
          *a12 = 0;
          *a11 = 0;
          v172 = dword_4F077C4;
        }
        else
        {
          *a10 = 0;
          *a9 = 0;
          *a12 = 0;
          *a11 = 0;
          if ( !v219 )
          {
            if ( dword_4F077C4 == 2 )
            {
              v61 = 0;
              a5 = 0;
              v141 = (v19 >> 17) & 1;
              goto LABEL_431;
            }
LABEL_686:
            a5 = 0;
            v219 = 0;
            goto LABEL_113;
          }
          a5 = 0;
          v172 = dword_4F077C4;
        }
        *(_BYTE *)(v17 + 130) |= 1u;
        if ( v172 == 2 )
        {
          v219 = 1;
          v61 = 0;
          v141 = (v19 >> 17) & 1;
          goto LABEL_431;
        }
        v61 = 0;
LABEL_383:
        v219 = 1;
        v60 = v172;
        goto LABEL_384;
      }
      if ( (*a2 & 4) != 0 )
      {
        *a10 = 0;
        *a9 = 0;
        *a12 = 0;
        *a11 = 0;
        v172 = dword_4F077C4;
        *(_BYTE *)(v17 + 130) |= 1u;
        if ( v172 == 2 )
        {
          v141 = (v19 >> 17) & 1;
          if ( !v250 )
          {
            v219 = 1;
            v61 = 0;
            goto LABEL_431;
          }
          goto LABEL_519;
        }
      }
      else
      {
        if ( !v250 )
        {
          *a10 = 0;
          *a9 = 0;
          *a12 = 0;
          *a11 = 0;
          if ( dword_4F077C4 != 2 )
            goto LABEL_686;
          v219 = 0;
          v61 = 0;
          a5 = 0;
          v141 = (v19 >> 17) & 1;
LABEL_431:
          v142 = dword_4F06978;
          if ( dword_4F06978 )
          {
            v142 = 0;
            goto LABEL_390;
          }
          if ( (v19 & 0x100) == 0 && (v232 || !dword_4D04964) )
          {
            if ( v239 )
            {
              v151 = v239;
              v152 = 6338;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v151 + 140) )
                {
                  case 6:
                  case 8:
                    goto LABEL_441;
                  case 7:
                    goto LABEL_390;
                  case 0xC:
                    if ( *(_QWORD *)(v151 + 8) )
                      goto LABEL_443;
                    v153 = *(unsigned __int8 *)(v151 + 184);
                    if ( (unsigned __int8)v153 <= 0xCu )
                    {
                      if ( _bittest64(&v152, v153) )
                        goto LABEL_443;
                    }
LABEL_441:
                    v151 = *(_QWORD *)(v151 + 160);
LABEL_442:
                    if ( v151 )
                      continue;
LABEL_443:
                    v208 = v141;
                    if ( (unsigned int)sub_8D3320(v239) )
                    {
                      v154 = sub_8D46C0(v239);
                      LODWORD(v141) = v208;
                      v142 = v154 != 0;
                    }
                    else
                    {
                      v189 = sub_8D3D10(v239);
                      LODWORD(v141) = v208;
                      v142 = 1;
                      if ( v189 )
                      {
                        v190 = sub_8D4870(v239);
                        LODWORD(v141) = v208;
                        v142 = v190 != 0;
                      }
                    }
                    break;
                  case 0xD:
                    v151 = *(_QWORD *)(v151 + 168);
                    goto LABEL_442;
                  default:
                    goto LABEL_443;
                }
                break;
              }
            }
            goto LABEL_390;
          }
LABEL_424:
          v142 = 1;
          goto LABEL_390;
        }
        if ( !*a9 && !*a11 && !*a12 )
        {
          if ( ((v19 & 0x10) == 0 || (v18[1].m128i_i8[0] & 8) != 0 && (unsigned __int8)(v18[3].m128i_i8[8] - 1) <= 3u)
            && !v219 )
          {
            v141 = (v19 >> 17) & 1;
            goto LABEL_520;
          }
LABEL_595:
          *(_BYTE *)(v17 + 130) |= 1u;
LABEL_587:
          v141 = (v19 >> 17) & 1;
LABEL_519:
          v219 = 1;
LABEL_520:
          v61 = (char *)v250;
          if ( (v19 & 0x1000) == 0 )
          {
LABEL_386:
            if ( (v19 & 0x2000) != 0 && !dword_4F077BC )
              LODWORD(v141) = 1;
            v142 = 1;
            if ( v60 != 2 )
            {
LABEL_390:
              sub_627530(v17, v19, &v241, v61, v18, (__int64)a5, v219, *a9, *a10, *a11, *a12, v141, v142, (__int64)a16);
              if ( (*(_BYTE *)(v17 + 123) & 0x10) != 0 )
LABEL_114:
                v238 = *(_QWORD *)(v17 + 272);
              if ( v61 && (v19 & 0x2000) != 0 && (v61[65] & 1) != 0 )
                sub_684B30(949, &v18->m128i_u64[1]);
              goto LABEL_119;
            }
            goto LABEL_431;
          }
          LODWORD(v141) = 1;
          goto LABEL_431;
        }
        if ( (*a2 & 0x40) == 0 || (*(_BYTE *)(v17 + 8) & 1) != 0 )
          goto LABEL_595;
        v181 = 1595;
        if ( !*a9 )
          v181 = *a11 == 0 ? 2033 : 279;
        v223 = dword_4F077C4;
        sub_6851C0(v181, v17 + 40);
        v172 = dword_4F077C4;
        *(_BYTE *)(v17 + 130) |= 1u;
        v60 = v223;
        if ( v172 == 2 )
          goto LABEL_587;
      }
      v61 = (char *)v250;
      goto LABEL_383;
    }
    v61 = (char *)v250;
    if ( v239 )
      goto LABEL_113;
LABEL_384:
    if ( v61 )
    {
      LODWORD(v141) = 1;
      if ( (v19 & 0x1000) != 0 )
        goto LABEL_424;
      goto LABEL_386;
    }
LABEL_113:
    v61 = 0;
    sub_627530(v17, v19, &v241, 0, v18, (__int64)a5, v219, *a9, *a10, *a11, *a12, 1, 1, (__int64)a16);
    if ( (*(_BYTE *)(v17 + 123) & 0x10) != 0 )
      goto LABEL_114;
LABEL_119:
    if ( *(_BYTE *)(v241 + 140) == 7 )
    {
      if ( v239 )
      {
        if ( (unsigned int)sub_8D32B0(v239) )
        {
          v155 = *(_QWORD *)(v241 + 168);
          if ( !*(_QWORD *)(v155 + 40)
            && ((*(_BYTE *)(v155 + 18) | (unsigned __int8)(*(_WORD *)(v155 + 18) >> 7)) & 0x7F) != 0 )
          {
            v156 = &v246;
            if ( v18 )
              v156 = &v18->m128i_i64[1];
            sub_6851C0(990, v156);
          }
        }
      }
    }
    if ( v250 )
      *(_QWORD *)(v250 + 72) = *a14;
    v62 = v19 & 0x800;
LABEL_134:
    if ( a4 && v242 )
    {
      sub_5CF030(&v241, v242, v17);
      v242 = 0;
    }
    if ( (*(_BYTE *)(v17 + 122) & 2) == 0 )
      v238 = *(_QWORD *)(v17 + 280);
    v25 = v241;
    v24 = (__int64)&v239;
    sub_624710(v241, &v239, &v240, v17, v62 != 0);
    if ( v210 )
      goto LABEL_31;
    v32 = word_4F06418[0];
  }
  if ( (v18[1].m128i_i8[0] & 1) != 0 )
  {
    v57 = v18[1].m128i_i64[1];
    if ( v57 )
    {
      v58 = *(unsigned __int8 *)(v57 + 80);
      if ( (unsigned __int8)v58 <= 0x14u )
      {
        v59 = 1182720;
        if ( _bittest64(&v59, v58) )
        {
LABEL_109:
          if ( v244 )
            *((_BYTE *)v244 + 61) = 1;
          goto LABEL_111;
        }
      }
    }
  }
  if ( !v56 && v244 && !*((_BYTE *)v244 + 61) )
    goto LABEL_514;
  if ( !v213
    || dword_4F04C64 == -1
    || (v173 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64), (v173[12] & 2) == 0)
    || dword_4F04C44 == -1 && (v173[6] & 6) == 0 && v173[4] != 12 )
  {
    if ( (v19 & 0x80u) == 0LL )
    {
      if ( unk_4F04C48 != -1
        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0
        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 13) & 4) != 0 )
      {
        goto LABEL_109;
      }
      goto LABEL_542;
    }
    if ( !*(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C44 + 512) && !unk_4D043C8 )
    {
      if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
        goto LABEL_109;
      if ( dword_4F077BC )
      {
        if ( !(_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A8 <= 0xC34Fu )
            goto LABEL_109;
          goto LABEL_542;
        }
      }
      else if ( !(_DWORD)qword_4F077B4 )
      {
        goto LABEL_109;
      }
      if ( unk_4F077A0 <= 0x76BFu )
        goto LABEL_109;
    }
LABEL_542:
    if ( (unsigned int)sub_679C10(3) )
      goto LABEL_109;
  }
  if ( !v56 && v244 && !*((_BYTE *)v244 + 61) || word_4F06418[0] != 1 || dword_4F077BC )
    goto LABEL_514;
  sub_7ADF70(v249, 0);
  do
  {
    sub_7AE360(v249);
    if ( (unsigned __int16)sub_7B8B50(v249, 0, v176, v177) != 67 )
      break;
    sub_7AE360(v249);
  }
  while ( (unsigned __int16)sub_7B8B50(v249, 0, v174, v175) == 1 );
  if ( word_4F06418[0] == 28 )
  {
    sub_7AE360(v249);
    if ( (unsigned __int16)sub_7B8B50(v249, 0, v193, v194) == 73 || (unsigned int)sub_651B00(2) )
    {
      sub_7BC000(v249);
      goto LABEL_109;
    }
  }
  sub_7BC000(v249);
LABEL_514:
  *a2 |= 1uLL;
  if ( a16 )
    a16[8] = v246;
LABEL_31:
  if ( !a4 )
  {
    v34 = (__int64 *)dword_4F07508;
    *(_QWORD *)dword_4F07508 = v245;
    goto LABEL_234;
  }
LABEL_32:
  for ( j = a4; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  *(_BYTE *)(j + 88) |= 4u;
  *(_QWORD *)dword_4F07508 = v245;
  v34 = v242;
  if ( v242 )
  {
    sub_5CF030(&v238, v242, v17);
    v242 = 0;
  }
  if ( (v19 & 0x200) == 0 )
  {
    if ( !v18 )
      goto LABEL_40;
    v35 = v18[1].m128i_i8[0];
    if ( (v35 & 0x58) != 0 )
    {
      v34 = &v18->m128i_i64[1];
      sub_6851C0(502, &v18->m128i_u64[1]);
      *v18 = _mm_loadu_si128(xmmword_4F06660);
      v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v70 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v71 = *(_QWORD *)dword_4F07508;
      v18[1].m128i_i8[1] |= 0x20u;
      v18[3] = v70;
      v18->m128i_i64[1] = v71;
      v238 = sub_72C930();
      v72 = v238;
    }
    else
    {
LABEL_39:
      if ( (v35 & 0x10) == 0 )
        goto LABEL_40;
      if ( (v18[1].m128i_i8[1] & 0x20) != 0 )
        goto LABEL_379;
      v96 = *(_QWORD *)(v17 + 272);
      v97 = v18[1].m128i_i64[1];
      for ( k = *(_BYTE *)(v96 + 140); k == 12; k = *(_BYTE *)(v96 + 140) )
        v96 = *(_QWORD *)(v96 + 160);
      v99 = v239;
      if ( k != 21 && (v19 & 0x10000) == 0 )
      {
        v34 = (__int64 *)(v17 + 32);
        v233 = v239;
        sub_6851C0(452, v17 + 32);
        v99 = v233;
      }
      if ( v97 )
      {
        if ( (*(_DWORD *)(v97 + 80) & 0x400FF) == 0x4000A )
        {
          v100 = *(_QWORD *)(v97 + 96);
          if ( v100 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v100 + 32) + 80LL) == 20 )
            {
              if ( v99 )
              {
                v234 = v99;
                if ( (unsigned int)sub_8D2310(v99) )
                {
                  v101 = v234;
                  for ( m = *(_QWORD *)(v18[1].m128i_i64[1] + 64); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                    ;
                  v103 = *(_QWORD *)(*(_QWORD *)m + 96LL);
                  if ( *(_BYTE *)(v234 + 140) == 12 )
                  {
                    do
                      v101 = *(_QWORD *)(v101 + 160);
                    while ( *(_BYTE *)(v101 + 140) == 12 );
                  }
                  v34 = *(__int64 **)(v103 + 48);
                  v18[1].m128i_i64[1] = sub_7D24E0(v18, v34, 1, *(_BYTE *)(*(_QWORD *)(v101 + 168) + 18LL) & 0x7F);
                }
              }
            }
          }
        }
      }
      v72 = v18[3].m128i_i64[1];
      v238 = v72;
    }
LABEL_203:
    v88 = v239;
    if ( !v239 )
      goto LABEL_235;
LABEL_204:
    if ( !v72 || !v240 )
    {
      v238 = v88;
      goto LABEL_222;
    }
    if ( *(_BYTE *)(v240 + 140) == 6 && (*(_BYTE *)(v240 + 168) & 1) != 0 )
    {
      if ( (unsigned int)sub_8D32E0(v72) )
      {
        v168 = v17 + 72;
        if ( (*(_BYTE *)(v17 + 120) & 0x7F) == 4 )
          v168 = v17 + 80;
        if ( a4 && (v169 = (*(_BYTE *)(v240 + 168) & 2) != 0, (unsigned int)sub_8D32E0(a4)) )
        {
          v34 = (__int64 *)v169;
          v239 = sub_72D790(v238, v169, 0, *(_BYTE *)(v17 + 120) & 0x7F, v168, 0);
          v72 = v239;
        }
        else
        {
          v34 = (__int64 *)dword_4F07508;
          sub_6851C0(249, dword_4F07508);
          v239 = sub_72C930();
          v72 = v239;
        }
        *(_BYTE *)(v17 + 124) &= ~0x20u;
        v240 = v72;
        goto LABEL_209;
      }
      v72 = v238;
    }
    v34 = &v239;
    sub_624710(v72, &v239, &v240, v17, (v19 & 0x800) != 0);
    v72 = v239;
LABEL_209:
    v238 = v72;
    goto LABEL_210;
  }
  if ( v18 )
  {
    v35 = v18[1].m128i_i8[0];
    goto LABEL_39;
  }
LABEL_40:
  if ( *a9 || *a10 )
  {
    v86 = *(_BYTE *)(a4 + 140);
    if ( v86 == 12 )
    {
      v87 = a4;
      do
      {
        v87 = *(_QWORD *)(v87 + 160);
        v86 = *(_BYTE *)(v87 + 140);
      }
      while ( v86 == 12 );
    }
    if ( v86 != 21 && (v19 & 0x10000) == 0 )
    {
      v34 = (__int64 *)(v17 + 32);
      sub_6851C0(963, v17 + 32);
    }
LABEL_202:
    v238 = sub_72CBE0();
    v72 = v238;
    goto LABEL_203;
  }
  if ( *a11 )
  {
    if ( (v18[1].m128i_i8[1] & 0x20) == 0 )
    {
      if ( (v19 & 0x10000) != 0 )
      {
        if ( !v239 || !(unsigned int)sub_8D2310(v239) )
        {
          v34 = (__int64 *)(v17 + 24);
          sub_6851C0(279, v17 + 24);
        }
      }
      else
      {
        v34 = (__int64 *)(v17 + 32);
        sub_6851C0(964, v17 + 32);
      }
      goto LABEL_202;
    }
LABEL_379:
    v238 = sub_72C930();
    v72 = v238;
    goto LABEL_203;
  }
  v139 = *(_BYTE *)(a4 + 140);
  if ( v139 == 12 )
  {
    v140 = a4;
    do
    {
      v140 = *(_QWORD *)(v140 + 160);
      v139 = *(_BYTE *)(v140 + 140);
    }
    while ( v139 == 12 );
  }
  if ( v139 == 21 && (!dword_4F077C0 || *(char *)(v17 + 124) >= 0) )
  {
    v34 = (__int64 *)(v17 + 40);
    sub_6851C0(77, v17 + 40);
    goto LABEL_379;
  }
LABEL_234:
  v88 = v239;
  v72 = v238;
  if ( v239 )
    goto LABEL_204;
LABEL_235:
  if ( v72 )
  {
    v88 = v72;
LABEL_222:
    v93 = *(_BYTE *)(v88 + 140);
    if ( v93 == 12 )
    {
      v94 = v88;
      do
      {
        v94 = *(_QWORD *)(v94 + 160);
        v93 = *(_BYTE *)(v94 + 140);
      }
      while ( v93 == 12 );
    }
    if ( !v93 )
    {
      v72 = v238;
      goto LABEL_227;
    }
    v95 = sub_8D4940(v88);
    v72 = v238;
    v240 = v95;
  }
  else
  {
LABEL_227:
    v240 = 0;
  }
LABEL_210:
  if ( a4 )
  {
    if ( (unsigned int)sub_8D2310(v72) )
    {
      if ( v250 )
      {
        v72 = v238;
        if ( (*(_BYTE *)(v17 + 132) & 1) != 0 && *(_BYTE *)(v238 + 140) == 7 )
          *(_BYTE *)(*(_QWORD *)(v238 + 168) + 20LL) |= 0x80u;
        v34 = (__int64 *)v250;
        *(_QWORD *)(v250 + 80) = sub_624310(v72, v250);
      }
    }
    else if ( v18 && (v18[1].m128i_i8[0] & 0x58) != 0 )
    {
      v34 = &v18->m128i_i64[1];
      v72 = 501;
      sub_6851C0(501, &v18->m128i_u64[1]);
      *v18 = _mm_loadu_si128(xmmword_4F06660);
      v18[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      v18[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v157 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v158 = *(_QWORD *)dword_4F07508;
      v18[1].m128i_i8[1] |= 0x20u;
      v18[3] = v157;
      v18->m128i_i64[1] = v158;
      v240 = sub_72C930();
      v238 = v240;
    }
    if ( (*(_BYTE *)a2 & 8) != 0 )
    {
      if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 7 )
        sub_866010(v72, v34, qword_4F04C68, v89, v90);
      else
        sub_8645D0();
      *a2 &= ~8uLL;
    }
    v72 = v238;
  }
  v91 = v240;
  *a7 = v72;
  *a8 = v91;
  return a8;
}
