// Function: sub_627530
// Address: 0x627530
//
_QWORD *__fastcall sub_627530(
        __int64 a1,
        unsigned __int64 a2,
        __int64 *a3,
        char *a4,
        _BYTE *a5,
        __int64 a6,
        unsigned int a7,
        int a8,
        int a9,
        int a10,
        int a11,
        int a12,
        int a13,
        __int64 a14)
{
  __int64 v14; // rax
  _BYTE *v15; // rax
  int v16; // eax
  __int64 v17; // rcx
  unsigned int v18; // esi
  char v19; // al
  __int64 v20; // rsi
  __int64 v21; // rdx
  char v22; // al
  _BYTE *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int32 v26; // r13d
  __int64 *v27; // rdi
  __m128i v28; // xmm7
  __m128i v29; // xmm6
  __m128i v30; // xmm7
  __int64 *v31; // rsi
  __int64 v32; // rcx
  __m128i *v33; // rax
  __int64 v34; // rdx
  int v35; // eax
  int v36; // r14d
  __int64 v37; // rax
  __int64 v38; // rax
  _BOOL8 v39; // rbx
  unsigned int v40; // edx
  unsigned int v41; // eax
  char v42; // al
  char v43; // bl
  unsigned int v44; // eax
  unsigned __int8 v45; // r12
  __int64 v46; // rsi
  __m128i v47; // xmm7
  __m128i v48; // xmm6
  __m128i v49; // xmm7
  __int64 v50; // rdi
  char v51; // bl
  __int64 v52; // r12
  unsigned int v53; // ebx
  __int64 v54; // rax
  __m128i *v55; // rax
  int v56; // r11d
  __int64 v57; // rdi
  __m128i *v58; // rax
  __m128i v59; // xmm5
  unsigned __int64 v60; // rcx
  __int64 v61; // rsi
  char v62; // dl
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rsi
  __int64 v67; // rax
  __m128i *v68; // rdx
  __int8 v69; // cl
  char v70; // al
  __int8 v71; // cl
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // rcx
  __int64 v74; // rdi
  unsigned __int16 *v75; // rax
  __int64 v76; // r8
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  bool v80; // zf
  unsigned __int64 v81; // r12
  char v82; // al
  char v83; // al
  unsigned __int64 v84; // rdi
  char v85; // al
  unsigned int v86; // r9d
  __int64 v87; // rsi
  _BOOL4 v88; // edi
  __int64 v89; // rdx
  __int64 v90; // rax
  char v91; // cl
  unsigned int v92; // eax
  __int64 v93; // rax
  _QWORD *v94; // rdx
  _QWORD *v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rdi
  __int64 v99; // rax
  char v100; // r12
  char v101; // bl
  __int64 v102; // rdx
  char *v103; // rcx
  __int64 v104; // rax
  __int64 v105; // rax
  _QWORD *result; // rax
  _QWORD *v107; // rcx
  _QWORD *v108; // rdx
  __int64 v109; // rdx
  unsigned __int32 v110; // ebx
  __int64 v111; // rdi
  __int64 v112; // rax
  __int64 i; // rsi
  int v114; // eax
  __int64 v115; // rax
  int v116; // eax
  __int64 v117; // rax
  int v118; // eax
  unsigned __int8 v119; // dl
  __int64 v120; // rdi
  _BYTE *v121; // rax
  char v122; // dl
  _QWORD *v123; // rcx
  __int64 *v124; // rax
  unsigned int v125; // edx
  __int64 j; // rsi
  int v127; // eax
  char v128; // al
  __int64 v129; // rsi
  _BOOL4 v130; // r14d
  char v131; // al
  __int64 v132; // r12
  int v133; // ebx
  int v134; // eax
  __int64 v135; // rdx
  unsigned __int64 v136; // rdx
  __int64 v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // rdi
  unsigned __int16 v141; // ax
  char v142; // r14
  __int64 v143; // rdx
  __int64 v144; // rcx
  bool v145; // al
  char v146; // dl
  int v147; // ebx
  __int64 v148; // rcx
  __int64 v149; // r12
  __int16 v150; // r14
  __int64 v151; // rax
  unsigned __int16 v152; // r12
  unsigned int v153; // eax
  __int64 v154; // rdx
  unsigned __int64 v155; // rbx
  __int64 v156; // rdx
  __int64 v157; // rcx
  __int64 v158; // rax
  _QWORD *v159; // r12
  __int64 v160; // r13
  __int64 v161; // rax
  __int64 v162; // rax
  __int64 v163; // rcx
  __int64 v164; // r8
  char v165; // dl
  __int64 m; // rdi
  __int64 v167; // rax
  const __m128i *v168; // rax
  __int64 v169; // rdi
  __int64 v170; // rdx
  __int64 v171; // rcx
  _QWORD *v172; // r14
  int v173; // eax
  __int64 v174; // rax
  __int64 v175; // rcx
  __int64 k; // rsi
  int v177; // eax
  unsigned int v178; // eax
  _QWORD *v179; // rax
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // rdx
  __int64 v183; // rcx
  __int64 v184; // rax
  char v185; // cl
  char v186; // dl
  __int64 v187; // rax
  __int64 v188; // rdx
  unsigned int v189; // ecx
  __int64 v190; // rax
  __int64 v191; // rcx
  char v192; // al
  __int64 v193; // rax
  char n; // dl
  char v195; // dl
  __int64 v196; // rax
  __int64 v197; // rax
  __int64 v198; // r14
  __int64 *v199; // r14
  __int64 v200; // rdi
  __int64 v201; // rsi
  unsigned int v202; // eax
  __int64 v203; // rdx
  __int64 v204; // rax
  __int64 v205; // rax
  __int64 v206; // rax
  __int64 v207; // rsi
  int v208; // r12d
  char v209; // al
  __int64 v210; // rdx
  __int64 v211; // rax
  __int64 v212; // rdx
  __int64 v213; // rcx
  int v214; // eax
  int v215; // eax
  __int16 v216; // dx
  unsigned __int16 *v217; // rax
  __int64 v218; // rdi
  __int64 v219; // rax
  __int64 v220; // rax
  __int64 v221; // rax
  char v222; // al
  __int64 v223; // rax
  __int64 v224; // rdi
  __int64 v225; // rsi
  __int64 v226; // rax
  _QWORD *v227; // [rsp-10h] [rbp-460h]
  __int64 v228; // [rsp-8h] [rbp-458h]
  unsigned int v229; // [rsp+Ch] [rbp-444h]
  unsigned int v230; // [rsp+10h] [rbp-440h]
  char v231; // [rsp+14h] [rbp-43Ch]
  __int64 v232; // [rsp+18h] [rbp-438h]
  unsigned __int64 v233; // [rsp+28h] [rbp-428h]
  _BOOL4 v236; // [rsp+40h] [rbp-410h]
  _BOOL4 v237; // [rsp+44h] [rbp-40Ch]
  unsigned __int64 v238; // [rsp+48h] [rbp-408h]
  unsigned int v239; // [rsp+50h] [rbp-400h]
  unsigned int v240; // [rsp+54h] [rbp-3FCh]
  __int64 v241; // [rsp+58h] [rbp-3F8h]
  int v244; // [rsp+70h] [rbp-3E0h]
  char v245; // [rsp+77h] [rbp-3D9h]
  unsigned int v246; // [rsp+78h] [rbp-3D8h]
  int v247; // [rsp+7Ch] [rbp-3D4h]
  _BYTE *v248; // [rsp+80h] [rbp-3D0h]
  unsigned int v249; // [rsp+90h] [rbp-3C0h]
  int v250; // [rsp+90h] [rbp-3C0h]
  char v251; // [rsp+98h] [rbp-3B8h]
  unsigned int v252; // [rsp+98h] [rbp-3B8h]
  unsigned int v253; // [rsp+98h] [rbp-3B8h]
  unsigned int v254; // [rsp+9Ch] [rbp-3B4h]
  char v255; // [rsp+A0h] [rbp-3B0h]
  const __m128i *v256; // [rsp+A0h] [rbp-3B0h]
  __int64 v257; // [rsp+A8h] [rbp-3A8h]
  int v258; // [rsp+A8h] [rbp-3A8h]
  unsigned int v259; // [rsp+B0h] [rbp-3A0h]
  __int64 v260; // [rsp+B0h] [rbp-3A0h]
  char *v261; // [rsp+B8h] [rbp-398h]
  unsigned __int32 v262; // [rsp+C8h] [rbp-388h]
  unsigned int v263; // [rsp+C8h] [rbp-388h]
  unsigned int v264; // [rsp+C8h] [rbp-388h]
  unsigned int v265; // [rsp+C8h] [rbp-388h]
  _QWORD *v266; // [rsp+D0h] [rbp-380h]
  unsigned int v267; // [rsp+D0h] [rbp-380h]
  char v268; // [rsp+D0h] [rbp-380h]
  unsigned __int8 v269; // [rsp+D0h] [rbp-380h]
  char v270; // [rsp+D0h] [rbp-380h]
  char v271; // [rsp+D0h] [rbp-380h]
  unsigned __int8 v272; // [rsp+D0h] [rbp-380h]
  bool v273; // [rsp+D0h] [rbp-380h]
  int v274; // [rsp+D0h] [rbp-380h]
  unsigned __int8 v275; // [rsp+D0h] [rbp-380h]
  char v276; // [rsp+D0h] [rbp-380h]
  unsigned int v277; // [rsp+D0h] [rbp-380h]
  unsigned int v278; // [rsp+D0h] [rbp-380h]
  unsigned int v279; // [rsp+D0h] [rbp-380h]
  unsigned int v280; // [rsp+D0h] [rbp-380h]
  _QWORD *v281; // [rsp+D8h] [rbp-378h]
  char v282; // [rsp+D8h] [rbp-378h]
  __int64 v283; // [rsp+D8h] [rbp-378h]
  __m128i *v285; // [rsp+F0h] [rbp-360h] BYREF
  __int64 v286; // [rsp+F8h] [rbp-358h] BYREF
  __int64 v287; // [rsp+100h] [rbp-350h] BYREF
  __int64 v288; // [rsp+108h] [rbp-348h] BYREF
  __int64 v289; // [rsp+110h] [rbp-340h] BYREF
  __int64 v290; // [rsp+118h] [rbp-338h] BYREF
  __int64 v291; // [rsp+120h] [rbp-330h] BYREF
  __int64 v292; // [rsp+128h] [rbp-328h] BYREF
  __m128i v293; // [rsp+130h] [rbp-320h] BYREF
  __m128i v294; // [rsp+140h] [rbp-310h]
  __m128i v295; // [rsp+150h] [rbp-300h]
  __m128i v296; // [rsp+160h] [rbp-2F0h]
  __m128i v297[6]; // [rsp+170h] [rbp-2E0h] BYREF
  char v298[8]; // [rsp+1D0h] [rbp-280h] BYREF
  _BYTE v299[104]; // [rsp+1D8h] [rbp-278h] BYREF
  _OWORD v300[33]; // [rsp+240h] [rbp-210h] BYREF

  v232 = a2 & 0x100;
  v288 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  v261 = a4;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  if ( a4 )
  {
    if ( dword_4F077C4 == 2 )
    {
      v184 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v185 = *(_BYTE *)(v184 + 4);
      v186 = v185;
      if ( (unsigned __int8)(v185 - 8) <= 1u )
        v186 = *(_BYTE *)(v184 - 772);
      if ( v186 == 6 || v185 == 9 && v186 == 7 )
      {
        *(_BYTE *)(a1 + 129) = *(_BYTE *)(a1 + 129) & 0xBF
                             | ((((unsigned __int8)(*(_QWORD *)(a1 + 8) >> 3) ^ 1) & 1) << 6);
        v254 = 1;
      }
      else
      {
        v254 = 1;
        if ( v186 == 7 )
          *(_BYTE *)(a1 + 129) |= 0x80u;
      }
    }
    else
    {
      v254 = 1;
    }
  }
  else
  {
    sub_87E3B0(v298);
    v254 = 0;
    v261 = v298;
  }
  v285 = 0;
  v14 = sub_7259C0(7);
  *a3 = v14;
  v15 = *(_BYTE **)(v14 + 168);
  v248 = v15;
  if ( a8 )
  {
    v16 = (unsigned __int8)v15[17] | 1;
    v248[17] |= 1u;
  }
  else
  {
    v16 = (unsigned __int8)v15[17];
  }
  if ( a10 )
  {
    v16 |= 2u;
    v248[17] = v16;
  }
  *(_QWORD *)v248 = 0;
  v17 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v18 = v16 & 0xFFFFFF8F;
  v19 = v16 & 0x8F | 0x20;
  v20 = (8 * *(_BYTE *)(v17 + 9)) & 0x70 | v18;
  if ( ((8 * *(_BYTE *)(v17 + 9)) & 0x60) == 0x20 )
    v19 = v20;
  v248[17] = v19;
  if ( (*(_BYTE *)(v17 + 9) & 0x10) != 0 )
    v248[17] = v19 | 0x80;
  v21 = (unsigned int)dword_4F077C4;
  if ( word_4F06418[0] == 28 )
  {
    v23 = v248;
    if ( dword_4F077C4 != 2 )
    {
      v248[16] &= ~2u;
      if ( (v248[16] & 2) == 0 )
        goto LABEL_34;
      goto LABEL_357;
    }
LABEL_33:
    v23[16] |= 2u;
    if ( (v248[16] & 2) == 0 )
    {
LABEL_34:
      if ( v254 )
        sub_643E40(sub_622E00, a1, 1);
      else
        sub_684B00(3117, &v288);
      v246 = 0;
      v236 = 0;
      goto LABEL_229;
    }
LABEL_357:
    v286 = 0;
    v287 = 0;
    v246 = 0;
    v244 = 0;
    goto LABEL_39;
  }
  if ( word_4F06418[0] == 76 )
  {
    if ( dword_4F077C4 != 2 )
    {
      v21 = (unsigned int)qword_4F077B4 | unk_4D04364;
      if ( !((unsigned int)qword_4F077B4 | unk_4D04364) )
        goto LABEL_381;
    }
    v290 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(7, v20, v21, v17);
    if ( a10 )
    {
      sub_6851C0(407, &v290);
    }
    else
    {
      v248[16] |= 1u;
      if ( dword_4F077C4 != 2 )
      {
        if ( !(_DWORD)qword_4F077B4 || unk_4D04364 )
        {
          if ( dword_4D04964 && unk_4F07778 <= 202310 )
            sub_684AA0(byte_4F07472[0], 668, &v290);
        }
        else
        {
          sub_643E40(sub_622E70, a1, 1);
        }
      }
    }
    v23 = v248;
    goto LABEL_33;
  }
  if ( dword_4F077C4 == 2 && (!unk_4D04950 || a6) )
  {
    v248[16] |= 2u;
    goto LABEL_18;
  }
  if ( word_4F06418[0] == 1 )
  {
    if ( sub_6512E0(0, 0, 1, 0, 0, 0) )
    {
      if ( dword_4F077C4 != 1 )
        goto LABEL_507;
    }
    else
    {
      v145 = 0;
      if ( dword_4F077C4 == 1 )
        goto LABEL_383;
    }
    v216 = sub_7BE840(0, 0);
    v145 = v216 != 28 && v216 != 67;
    goto LABEL_383;
  }
LABEL_381:
  if ( (unsigned int)sub_651B00(2) )
  {
LABEL_507:
    v145 = 1;
    goto LABEL_383;
  }
  v145 = dword_4F077C4 != 1;
LABEL_383:
  v248[16] = (2 * v145) | v248[16] & 0xFD;
LABEL_18:
  if ( (v248[16] & 2) != 0 )
  {
    v286 = 0;
    v287 = 0;
    if ( a12 )
    {
      v246 = 1;
      v244 = 0;
    }
    else if ( !a5
           || (v22 = a5[16], (v22 & 0x10) != 0)
           || (a5[18] & 1) != 0 && (a2 & 0x2000) == 0
           || (v22 & 8) != 0 && a5[56] != 42 )
    {
      v246 = 1;
      v244 = (a2 >> 20) & 1;
    }
    else
    {
      v246 = 1;
      v244 = 1;
    }
LABEL_39:
    sub_8600D0(1, 0xFFFFFFFFLL, *a3, 0);
    v24 = 776LL * dword_4F04C64;
    v25 = v24 + qword_4F04C68[0];
    *(_QWORD *)(v24 + qword_4F04C68[0] + 624) = a1;
    v236 = (*(_BYTE *)(a1 + 133) & 1) == 0;
    if ( a8 && (dword_4F04C44 != -1 || (*(_BYTE *)(v25 + 6) & 6) != 0 || *(_BYTE *)(v25 + 4) == 12) )
    {
      *(_BYTE *)(v25 + 6) |= 0x80u;
      v25 = v24 + qword_4F04C68[0];
    }
    *((_DWORD *)v261 + 10) = *(_DWORD *)v25;
    if ( !v246 )
    {
LABEL_224:
      *(_BYTE *)(a1 + 131) &= ~0x40u;
      if ( v254 )
      {
        v93 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        v94 = *(_QWORD **)(v93 + 24);
        v95 = (_QWORD *)(v93 + 32);
        if ( !v94 )
          v94 = v95;
        *(_QWORD *)v261 = *v94;
        v96 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        *((_QWORD *)v261 + 11) = *(_QWORD *)(v96 + 328);
        *(_QWORD *)(v96 + 328) = 0;
      }
      sub_854430();
      v97 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      *((_QWORD *)v261 + 6) = *(_QWORD *)(v97 + 232);
      *(_QWORD *)(v97 + 232) = 0;
      goto LABEL_229;
    }
    v26 = 0;
    while ( 1 )
    {
      v31 = &v287;
      v27 = &v286;
      v35 = sub_868D90(&v286, &v287, 0, 0, 0);
      if ( v35 )
        break;
      if ( v287 && *(_QWORD *)(v287 + 40) )
      {
        v27 = (__int64 *)v300;
        v28 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v300[0] = _mm_loadu_si128(xmmword_4F06660);
        v29 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v300[1] = v28;
        v30 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v300[2] = v29;
        *((_QWORD *)&v300[0] + 1) = *(_QWORD *)&dword_4F063F8;
        v300[3] = v30;
        *(_QWORD *)&v300[0] = *(_QWORD *)(v287 + 40);
        v31 = *(__int64 **)&dword_4D03B80;
        sub_885C70(
          (unsigned int)v300,
          dword_4D03B80,
          (unsigned int)&dword_4F063F8,
          3,
          (_DWORD)v261,
          0,
          (__int64)&v285,
          0);
        v33 = v285;
        v285[2].m128i_i8[10] |= 8u;
        v34 = *(_QWORD *)&dword_4D03B80;
        v33[7].m128i_i32[2] = v26;
        v33[1].m128i_i64[1] = v34;
      }
      ++v26;
      if ( word_4F06418[0] == 76 )
      {
        sub_7B8B50(v27, v31, v34, v32);
        v246 = 0;
        v248[16] |= 1u;
        goto LABEL_224;
      }
      if ( word_4F06418[0] == 28 )
      {
        v246 = 0;
        goto LABEL_224;
      }
    }
    v36 = v35;
    v262 = v26;
    v237 = 0;
    v230 = 0;
    v37 = a2 & 0x400;
    v239 = 0;
    v241 = v37 | a6;
    LOBYTE(v240) = 0;
    v229 = v37 != 0;
    v38 = -(__int64)(v232 == 0);
    LOBYTE(v39) = 0;
    v231 = 0;
    LOBYTE(v38) = 0;
    v38 += 2307;
    v266 = 0;
    v238 = v38;
    BYTE1(v38) |= 0xC0u;
    v233 = v38;
LABEL_152:
    while ( 2 )
    {
      v79 = v286;
      if ( v286 && *(_QWORD *)(v286 + 16) )
      {
        v80 = (unsigned int)sub_866580() == 0;
        v79 = v286;
        v259 = v246;
        if ( !v80 )
        {
          v251 = 1;
          v255 = 1;
LABEL_156:
          *(_BYTE *)(a1 + 131) |= 0x40u;
          v245 = v36 == 0 && v79 != 0;
          if ( !v245 )
          {
            v81 = (-(__int64)(dword_4D043F8 == 0) & 0xFFFFFFFFF8000000LL) + 134219787;
            if ( dword_4D043E0 )
              v81 = ((-(__int64)(dword_4D043F8 == 0) & 0xFFFFFFFFF8000000LL) + 134219787) | 0x400000;
            ++*(_BYTE *)(qword_4F061C8 + 75LL);
            memset(v300, 0, 0x1D8u);
            *((_QWORD *)&v300[9] + 1) = v300;
            *((_QWORD *)&v300[1] + 1) = *(_QWORD *)&dword_4F063F8;
            if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
              BYTE2(v300[11]) |= 1u;
            *((_QWORD *)&v300[26] + 1) = a1;
            v82 = BYTE11(v300[7]) & 0x79 | (4 * v254 + 2) & 0x86 | (v255 << 7);
            BYTE11(v300[7]) = v82;
            if ( !v254 )
            {
              v83 = (8 * (word_4D04430 & 1)) | v82 & 0xF7;
              BYTE11(v300[7]) = v83;
              if ( dword_4F04C64 != -1 )
                BYTE11(v300[7]) = (32 * (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1)) | v83 & 0xDF;
              if ( *(char *)(a1 + 132) < 0 )
                goto LABEL_61;
              goto LABEL_57;
            }
            if ( (*(_QWORD *)(a1 + 128) & 0x100000C0000LL) == 0
              && (dword_4D04490 || unk_4D04484 && (*(_BYTE *)(a1 + 131) & 8) != 0) )
            {
              v40 = DWORD2(v300[7]) & 0xF7FFFFFF | ((word_4D04430 & 1) << 27);
              *((_QWORD *)&v300[7] + 1) = *((_QWORD *)&v300[7] + 1) & 0xFFFFFFBFF7FFFFFFLL
                                        | ((unsigned __int64)(word_4D04430 & 1) << 27)
                                        | 0x4000000000LL;
              v41 = HIBYTE(v40);
              if ( dword_4F04C64 == -1 )
              {
LABEL_57:
                if ( !v241 && (*(_BYTE *)(a1 + 133) & 1) == 0 )
                {
                  v42 = 0;
                  if ( !a5 || (a5[16] & 1) == 0 )
                  {
LABEL_62:
                    BYTE4(v300[8]) = (v42 << 7) | BYTE4(v300[8]) & 0x7F;
                    v289 = *(_QWORD *)&dword_4F063F8;
                    memset(v297, 0, 0x58u);
                    *((_QWORD *)&v300[11] + 1) = sub_5CC190(1);
                    sub_672A20(v81, v300, v297);
                    v43 = BYTE8(v300[0]);
                    v44 = 0;
                    if ( (BYTE5(v300[8]) & 0x40) == 0 )
                      v44 = a7;
                    v45 = BYTE12(v300[16]);
                    a7 = v44;
                    v257 = BYTE8(v300[0]) & 0x80;
                    if ( *(_QWORD *)&v300[23] && v286 )
                      *(_BYTE *)(v286 + 43) = 1;
                    if ( v266 )
                    {
LABEL_68:
                      if ( (v43 & 0x20) != 0 && dword_4F077C4 == 2 )
                      {
                        sub_6851C0(255, &v289);
                        *(_QWORD *)&v300[17] = sub_72C930();
                        *((_QWORD *)&v300[17] + 1) = *(_QWORD *)&v300[17];
                        *(_QWORD *)&v300[18] = *(_QWORD *)&v300[17];
                      }
                      else
                      {
                        v46 = *(_QWORD *)&v300[18];
                        if ( (v43 & 1) != 0 )
                        {
                          while ( *(_BYTE *)(v46 + 140) == 12 )
                            v46 = *(_QWORD *)(v46 + 160);
                          *(_BYTE *)(v46 + 88) |= 4u;
                        }
                        else
                        {
                          sub_64E990(&dword_4F063F8, *(_QWORD *)&v300[18], 0, 0, 0, 1);
                        }
                      }
                      if ( v257 )
                        goto LABEL_78;
                      if ( word_4F06418[0] != 1 )
                      {
                        if ( word_4F06418[0] == 27 || word_4F06418[0] == 34 )
                          goto LABEL_179;
                        if ( dword_4F077C4 != 2 )
                        {
                          if ( word_4F06418[0] == 25 )
                            goto LABEL_179;
                          goto LABEL_78;
                        }
                        if ( unk_4D04474 && word_4F06418[0] == 52
                          || dword_4D0485C && word_4F06418[0] == 25
                          || word_4F06418[0] == 156
                          || ((word_4F06418[0] - 25) & 0xFFF7) == 0 )
                        {
                          goto LABEL_218;
                        }
                        goto LABEL_393;
                      }
                      if ( dword_4F077C4 != 2 )
                        goto LABEL_179;
                      if ( (unk_4D04A11 & 2) != 0 )
                      {
                        if ( (unk_4D04A12 & 1) == 0 )
                          goto LABEL_218;
                      }
                      else
                      {
                        if ( !(unsigned int)sub_7C0F00(0, 0) || (unk_4D04A12 & 1) == 0 || word_4F06418[0] == 25 )
                          goto LABEL_179;
                        if ( dword_4F077C4 != 2 )
                        {
LABEL_78:
                          if ( !unk_4D04408 )
                            goto LABEL_81;
                          if ( word_4F06418[0] == 76
                            && ((unsigned __int16)sub_7BE840(0, 0) != 28
                             || (unsigned int)sub_867060()
                             || *(_QWORD *)&v300[23]) )
                          {
                            goto LABEL_179;
                          }
                          goto LABEL_80;
                        }
                        if ( word_4F06418[0] != 1 )
                        {
LABEL_393:
                          if ( !(unsigned int)sub_7C0F00(0, 0) && word_4F06418[0] == 15 )
                            goto LABEL_179;
                          goto LABEL_78;
                        }
                      }
                      if ( (unk_4D04A11 & 2) != 0 )
                      {
                        if ( !unk_4D04408 )
                        {
LABEL_81:
                          *(_QWORD *)&v300[3] = *(_QWORD *)&dword_4F063F8;
                          v47 = _mm_loadu_si128(&xmmword_4F06660[1]);
                          v293 = _mm_loadu_si128(xmmword_4F06660);
                          v48 = _mm_loadu_si128(&xmmword_4F06660[2]);
                          v294 = v47;
                          v49 = _mm_loadu_si128(&xmmword_4F06660[3]);
                          v295 = v48;
                          v296 = v49;
                          v294.m128i_i8[1] |= 0x20u;
                          v293.m128i_i64[1] = *(_QWORD *)dword_4F07508;
                          if ( (BYTE12(v300[7]) & 0x20) != 0 )
                            sub_6451E0(v300);
                          if ( dword_4F077C4 != 2 )
                          {
LABEL_84:
                            sub_6455C0(v300, &v289);
                            if ( !v45 )
                              v45 = 3;
                            v247 = v45;
                            if ( !v254
                              || dword_4F04C44 != -1
                              || (v112 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v112 + 6) & 6) != 0)
                              || *(_BYTE *)(v112 + 4) == 12 )
                            {
                              if ( *(_QWORD *)&v300[22] )
                              {
                                sub_869FD0(*(_QWORD *)&v300[22], (unsigned int)dword_4F04C64);
                                *(_QWORD *)&v300[22] = 0;
                              }
                            }
                            else if ( !*(_QWORD *)&v300[22] )
                            {
                              *(_QWORD *)&v300[22] = sub_869D30();
                            }
                            if ( (v294.m128i_i8[1] & 0x20) != 0 )
                              v261[64] |= 1u;
                            v50 = *(_QWORD *)&v300[18];
                            if ( unk_4F0690C )
                            {
                              if ( *(_BYTE *)(*(_QWORD *)&v300[18] + 140LL) == 12 )
                              {
                                v249 = sub_8D4C10(*(_QWORD *)&v300[18], 1);
                                *(_QWORD *)&v300[18] = sub_73D4C0(*(_QWORD *)&v300[18], dword_4F077C4 == 2);
                                v50 = *(_QWORD *)&v300[18];
                                if ( (v249 & 8) != 0 )
                                {
                                  *(_QWORD *)&v300[18] = sub_73C570(*(_QWORD *)&v300[18], 8, -1);
                                  v50 = *(_QWORD *)&v300[18];
                                }
                                v51 = v249 & 0x7F;
                                if ( (v249 & 4) != 0 && unk_4F06908 )
                                {
                                  *(_QWORD *)&v300[18] = sub_73C570(v50, 4, -1);
                                  v50 = *(_QWORD *)&v300[18];
                                }
                              }
                              else
                              {
                                v51 = 0;
                                v249 = 0;
                                *(_QWORD *)&v300[18] = sub_73D4C0(*(_QWORD *)&v300[18], dword_4F077C4 == 2);
                                v50 = *(_QWORD *)&v300[18];
                              }
                            }
                            else
                            {
                              v249 = 0;
                              v51 = 0;
                            }
                            v52 = sub_72B0C0(v50, &v289);
                            *(_QWORD *)(v52 + 16) = *((_QWORD *)&v300[17] + 1);
                            *(_DWORD *)(v52 + 32) = ((v51 & 0x7F) << 11) | *(_DWORD *)(v52 + 32) & 0xFFFC07FF;
                            if ( (BYTE5(v300[8]) & 0x40) != 0 )
                              *(_BYTE *)(v52 + 35) |= 1u;
                            if ( (BYTE11(v300[7]) & 0x40) != 0 )
                            {
                              if ( dword_4F04C44 != -1
                                || (v109 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v109 + 6) & 6) != 0)
                                || *(_BYTE *)(v109 + 4) == 12 )
                              {
                                if ( !v255 )
                                {
                                  v53 = 0;
                                  if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 - 772) != 8
                                    && (*(_BYTE *)(a1 + 131) & 8) == 0 )
                                  {
                                    v53 = v244;
                                  }
                                  v54 = *(_QWORD *)&v300[23];
                                  if ( *(_QWORD *)&v300[23] )
                                    goto LABEL_104;
                                  goto LABEL_105;
                                }
                              }
                              if ( !v251 )
                              {
                                if ( (a2 & 2) != 0 )
                                {
                                  v121 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
                                  v122 = v121[6];
                                }
                                else if ( dword_4F04C44 != -1
                                       || (v121 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64),
                                           v122 = v121[6],
                                           (v122 & 6) != 0)
                                       || v121[4] == 12 )
                                {
                                  v85 = ((*(_BYTE *)(a1 + 131) >> 6) ^ 1) & 1 | *(_BYTE *)(v52 + 33) & 0xFE;
                                  *(_BYTE *)(v52 + 33) = v85;
                                  goto LABEL_188;
                                }
                                if ( (v122 & 4) != 0 )
                                {
                                  v80 = (v121[12] & 0x10) == 0;
                                  v85 = *(_BYTE *)(v52 + 33);
                                  if ( v80 )
                                  {
                                    v53 = v244;
                                    v259 = 0;
                                    *(_BYTE *)(v52 + 33) = v85 & 0xFC | 1;
                                    goto LABEL_191;
                                  }
LABEL_188:
                                  *(_BYTE *)(v52 + 33) = (2 * (v259 & 1)) | v85 & 0xFD;
                                  v53 = v244;
                                  if ( v259 )
                                  {
                                    v53 = 0;
                                    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 - 772) != 9 )
                                      v53 = v244;
                                  }
LABEL_191:
                                  v54 = *(_QWORD *)&v300[23];
                                  if ( *(_QWORD *)&v300[23] && (BYTE11(v300[7]) & 0x40) != 0 )
LABEL_104:
                                    *(_BYTE *)(v54 + 32) |= 1u;
LABEL_105:
                                  if ( SBYTE12(v300[7]) < 0 )
                                    *(_BYTE *)(v52 + 33) |= 4u;
                                  if ( (v294.m128i_i8[1] & 0x20) != 0 )
                                  {
                                    if ( (*(_BYTE *)(a1 + 133) & 1) != 0 )
                                      sub_684B30(3056, &v300[3]);
                                  }
                                  else
                                  {
                                    *(_QWORD *)(v52 + 24) = *(_QWORD *)(v293.m128i_i64[0] + 8);
                                  }
                                  sub_645120(v300, v52);
                                  v55 = (__m128i *)sub_7274B0(*(_BYTE *)(v52 - 8) & 1);
                                  *v55 = _mm_loadu_si128(&v297[1]);
                                  v55[1] = _mm_loadu_si128(&v297[2]);
                                  v55[2] = _mm_loadu_si128(&v297[3]);
                                  *(_QWORD *)(v52 + 72) = v55;
                                  if ( v266 )
                                    *v266 = v52;
                                  else
                                    *(_QWORD *)v248 = v52;
                                  v56 = v300[18];
                                  if ( v249 )
                                    v56 = sub_73C570(*(_QWORD *)&v300[18], v249, -1);
                                  v57 = (__int64)&v293;
                                  sub_885C70(
                                    (unsigned int)&v293,
                                    v56,
                                    (unsigned int)&v289,
                                    v247,
                                    (_DWORD)v261,
                                    v300[22],
                                    (__int64)&v285,
                                    v259);
                                  v58 = v285;
                                  v285[1].m128i_i64[1] = *((_QWORD *)&v300[17] + 1);
                                  v58[4] = _mm_loadu_si128(&v297[2]);
                                  v58[5] = _mm_loadu_si128(&v297[3]);
                                  v59 = _mm_loadu_si128(&v297[1]);
                                  v58[7].m128i_i32[2] = v262;
                                  v58[6] = v59;
                                  *(_DWORD *)(v52 + 36) = v262;
                                  v60 = (unsigned __int64)v227;
                                  v61 = v228;
                                  if ( (WORD4(v300[7]) & 0x3F80) != 0 )
                                  {
                                    v61 = a1;
                                    v58[2].m128i_i8[9] = v58[2].m128i_i8[9] & 0x80 | (WORD4(v300[7]) >> 7) & 0x7F;
                                    v60 = (*(_BYTE *)(a1 + 122) & 8) != 0;
                                    v237 = (*(_BYTE *)(a1 + 122) & 8) != 0;
                                  }
                                  v62 = *(_BYTE *)(v52 + 33);
                                  if ( (v62 & 2) != 0 )
                                  {
                                    v63 = v58[2].m128i_u8[10];
                                    v60 = v58[2].m128i_u8[10] | 2u;
                                    v58[2].m128i_i8[10] |= 2u;
                                    if ( (*(_BYTE *)(v52 + 33) & 1) == 0 )
                                      goto LABEL_121;
                                    v58[2].m128i_i8[10] = v63 | 3;
                                    v62 = *(_BYTE *)(v52 + 33);
                                  }
                                  v63 = v62 & 1;
                                  if ( (_DWORD)v63 )
                                  {
                                    v63 = v286;
                                    if ( v286 )
                                    {
                                      if ( !*(_QWORD *)(v286 + 16) )
                                      {
                                        v63 = v287;
                                        if ( v287 )
                                        {
                                          v60 = v58->m128i_u64[1];
                                          if ( !v60 )
                                          {
LABEL_123:
                                            v267 = 0;
                                            if ( dword_4F077C4 != 2 )
                                              goto LABEL_124;
                                            v267 = v246;
                                            if ( word_4F06418[0] != 56 )
                                              goto LABEL_124;
                                            if ( !v53 )
                                            {
                                              sub_6851C0(347, &dword_4F063F8);
                                              sub_7B8B50(347, &dword_4F063F8, v138, v139);
                                              v140 = v286;
                                              v291 = *(_QWORD *)&dword_4F063F8;
                                              if ( v286 )
                                              {
                                                v86 = *(_QWORD *)(v286 + 16) != 0;
                                                goto LABEL_360;
                                              }
                                              v240 = 0;
                                              v141 = word_4F06418[0];
                                              v142 = word_4F06418[0] == 28 || (word_4F06418[0] & 0xFFF7) == 67;
                                              if ( !v142 )
                                              {
                                                v86 = dword_4D04428;
                                                if ( !dword_4D04428 )
                                                  goto LABEL_595;
                                                v245 = 0;
                                              }
                                              goto LABEL_363;
                                            }
                                            if ( !v254 )
                                            {
                                              if ( !dword_4F077BC || (v57 = 4, qword_4F077A8 > 0x76BFu) )
                                                v57 = qword_4D0495C == 0 ? 7 : 4;
                                              v61 = 949;
                                              sub_684AA0(v57, 949, &dword_4F063F8);
                                            }
                                            sub_7B8B50(v57, v61, v63, v60);
                                            v86 = 0;
                                            v291 = *(_QWORD *)&dword_4F063F8;
                                            if ( v286 )
                                              v86 = *(_QWORD *)(v286 + 16) != 0;
                                            v87 = *(_QWORD *)(a1 + 368);
                                            v88 = v87 != 0;
                                            v89 = 776LL * dword_4F04C64;
                                            v90 = qword_4F04C68[0] + v89 - 776;
                                            v91 = *(_BYTE *)(v90 + 4);
                                            switch ( v91 )
                                            {
                                              case 6:
                                                v187 = *(_QWORD *)(v90 + 208);
                                                if ( *(_BYTE *)(v187 + 140) == 9
                                                  && (*(_BYTE *)(*(_QWORD *)(v187 + 168) + 109LL) & 0x20) != 0 )
                                                {
LABEL_209:
                                                  if ( !v254 || (a2 & 0x102) != 0 || (*(_BYTE *)(a1 + 131) & 8) != 0 )
                                                  {
                                                    v240 = 0;
                                                    v92 = 0;
                                                  }
                                                  else
                                                  {
                                                    v240 = 0;
                                                    v92 = 0;
                                                    if ( (*(_BYTE *)(a1 + 124) & 8) == 0 )
                                                    {
                                                      v92 = v254;
                                                      v86 = v254;
                                                    }
                                                  }
                                                }
                                                else if ( v254 )
                                                {
                                                  v240 = v254;
                                                  v86 = v254;
                                                  v92 = 0;
                                                }
                                                else
                                                {
                                                  v92 = 0;
                                                  v240 = 0;
                                                  if ( dword_4F04C44 == -1 )
                                                  {
                                                    v188 = qword_4F04C68[0] + v89;
                                                    if ( (*(_BYTE *)(v188 + 6) & 6) == 0 )
                                                    {
                                                      v189 = 0;
                                                      if ( *(_BYTE *)(v188 + 4) != 12 )
                                                      {
                                                        v86 = v53;
                                                        v189 = v53;
                                                      }
                                                      v240 = v189;
                                                    }
                                                  }
                                                }
LABEL_509:
                                                v245 = (v231 ^ 1) & 1;
                                                if ( (word_4F06418[0] & 0xFFF7) != 0x43 && word_4F06418[0] != 28 )
                                                {
                                                  if ( dword_4D04428 )
                                                  {
                                                    v142 = (v231 ^ 1) & 1;
                                                    if ( !v87 )
                                                    {
                                                      v140 = v52;
                                                      if ( !v86 )
                                                        goto LABEL_363;
                                                      goto LABEL_603;
                                                    }
LABEL_630:
                                                    if ( !v86 )
                                                    {
                                                      v245 = v142;
LABEL_362:
                                                      v140 = 0;
                                                      goto LABEL_363;
                                                    }
                                                    v140 = 0;
                                                    goto LABEL_603;
                                                  }
                                                  if ( (unsigned __int16)(word_4F06418[0] - 73) > 1u )
                                                  {
                                                    v142 = (v231 ^ 1) & 1;
                                                    goto LABEL_597;
                                                  }
                                                }
                                                if ( v87 )
                                                  goto LABEL_361;
                                                v140 = v52;
                                                if ( v86 )
                                                  goto LABEL_513;
                                                goto LABEL_363;
                                              case 8:
                                                v240 = 0;
                                                v86 = v53;
                                                v92 = 0;
                                                goto LABEL_509;
                                              case 9:
                                                v141 = word_4F06418[0];
                                                v245 = (v231 ^ 1) & 1;
                                                if ( (word_4F06418[0] & 0xFFF7) == 0x43 || word_4F06418[0] == 28 )
                                                {
                                                  v240 = 0;
                                                  goto LABEL_513;
                                                }
                                                v142 = (v231 ^ 1) & 1;
                                                v86 = v53;
                                                if ( dword_4D04428 )
                                                {
                                                  v140 = 0;
                                                  goto LABEL_612;
                                                }
LABEL_595:
                                                if ( (unsigned __int16)(v141 - 73) <= 1u )
                                                {
                                                  v245 = v142;
                                                  v240 = 0;
                                                  goto LABEL_361;
                                                }
                                                v240 = 0;
                                                v88 = v246;
                                                v92 = 0;
LABEL_597:
                                                if ( !v88 )
                                                {
                                                  v140 = v52;
                                                  if ( !v86 )
                                                  {
                                                    v245 = v142;
                                                    goto LABEL_363;
                                                  }
LABEL_603:
                                                  if ( v240 )
                                                  {
                                                    sub_5EA8C0(v140, v229, v262);
                                                    if ( !v142 )
                                                      goto LABEL_569;
                                                    *(_BYTE *)(v52 + 32) |= 4u;
LABEL_366:
                                                    *(_BYTE *)(v52 + 32) |= 8u;
LABEL_367:
                                                    v261[65] |= 1u;
                                                    v267 = 0;
                                                    LOBYTE(v240) = v246;
                                                    v231 = 0;
                                                    goto LABEL_127;
                                                  }
                                                  if ( v92 )
                                                    sub_5EA680(a1, v140);
                                                  else
LABEL_612:
                                                    sub_8977C0(v140, v262);
                                                  if ( v142 )
                                                  {
                                                    *(_BYTE *)(v52 + 32) |= 4u;
                                                    goto LABEL_367;
                                                  }
                                                  goto LABEL_568;
                                                }
                                                goto LABEL_630;
                                            }
                                            if ( (v91 & 0xFD) != 5 )
                                              goto LABEL_209;
                                            while ( 1 )
                                            {
                                              v146 = *(_BYTE *)(v90 + 4);
                                              if ( v146 == 7 )
                                              {
                                                if ( *(char *)(v90 + 8) < 0 )
                                                  v90 -= 776;
                                              }
                                              else if ( v146 != 5 )
                                              {
                                                v240 = 0;
                                                v92 = 0;
                                                if ( v146 != 8 )
                                                  goto LABEL_509;
                                                if ( unk_4D0423C )
                                                {
                                                  v86 = v53;
                                                  goto LABEL_509;
                                                }
                                                v280 = v86;
                                                sub_684AA0(8, 347, &dword_4F063F8);
                                                v231 = v53;
                                                v86 = v280;
LABEL_360:
                                                v240 = 0;
                                                v141 = word_4F06418[0];
                                                v142 = word_4F06418[0] == 28 || (word_4F06418[0] & 0xFFF7) == 67;
                                                if ( !v142 )
                                                {
                                                  if ( !dword_4D04428 )
                                                    goto LABEL_595;
                                                  if ( v86 )
                                                  {
                                                    v140 = 0;
                                                    goto LABEL_612;
                                                  }
                                                  v245 = 0;
                                                  v140 = 0;
LABEL_363:
                                                  sub_6D05D0(v140, v240, (*(_QWORD *)(a1 + 8) >> 20) & 1LL);
                                                  goto LABEL_364;
                                                }
LABEL_361:
                                                if ( !v86 )
                                                  goto LABEL_362;
LABEL_513:
                                                sub_6851C0(29, &dword_4F063F8);
                                                if ( word_4F06418[0] == 73 )
                                                  sub_6793E0(0, 0, 0, 0, 0);
LABEL_364:
                                                if ( v245 )
                                                {
                                                  *(_BYTE *)(v52 + 32) |= 4u;
                                                  if ( v240 )
                                                    goto LABEL_366;
                                                  goto LABEL_367;
                                                }
LABEL_568:
                                                LOBYTE(v240) = v246;
LABEL_569:
                                                v267 = 0;
LABEL_124:
                                                if ( !(a12 | v53) && *(_QWORD *)v248 == v52 && a5 && (a5[16] & 8) != 0 )
                                                {
                                                  v116 = v244;
                                                  if ( (unsigned __int8)(a5[56] - 1) <= 3u )
                                                    v116 = v246;
                                                  v244 = v116;
                                                }
LABEL_127:
                                                if ( !v257 && dword_4F077C4 == 2 )
                                                {
                                                  if ( (v294.m128i_i8[1] & 0x20) == 0
                                                    || (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0)
                                                    && !(unsigned int)sub_7C0F00(0, 0)
                                                    || (unk_4D04A12 & 1) == 0 )
                                                  {
                                                    goto LABEL_177;
                                                  }
LABEL_130:
                                                  LODWORD(v39) = 0;
                                                  sub_6851C0(253, &dword_4F063F8);
                                                }
                                                else
                                                {
                                                  if ( word_4F06418[0] == 76 )
                                                    goto LABEL_130;
LABEL_177:
                                                  v39 = (unsigned int)sub_7BE800(67) == 0;
                                                }
                                                v65 = v286;
                                                v66 = 1;
                                                --*(_BYTE *)(qword_4F061C8 + 75LL);
                                                v67 = sub_867630(v65, 1);
                                                *(_QWORD *)(v52 + 80) = v67;
                                                if ( v67 )
                                                {
                                                  *(_BYTE *)(v52 + 33) |= 1u;
                                                  v68 = v285;
                                                  v69 = v285[2].m128i_i8[10];
                                                  v70 = v69 | 5;
                                                  v71 = v69 | 1;
                                                  v285[2].m128i_i8[10] = v71;
                                                  v66 = a1;
                                                  if ( (*(_BYTE *)(a1 + 131) & 0x40) == 0 )
                                                    v70 = v71;
                                                  v68[2].m128i_i8[10] = v70;
                                                }
                                                v36 = sub_866C00(v286);
                                                if ( v36 || (v286 = 0, v39) )
                                                {
LABEL_136:
                                                  v74 = v267;
                                                  LOBYTE(v73) = a6 != 0;
                                                  if ( v267 )
                                                    goto LABEL_137;
                                                }
                                                else
                                                {
                                                  v110 = v262;
                                                  do
                                                  {
                                                    v74 = (__int64)&v286;
                                                    LODWORD(v75) = sub_869470(&v286);
                                                    if ( (_DWORD)v75 )
                                                    {
                                                      v262 = v110;
                                                      v36 = (int)v75;
                                                      LODWORD(v39) = 0;
                                                      goto LABEL_136;
                                                    }
                                                    v73 = (unsigned __int64)word_4F06418;
                                                    ++v110;
                                                  }
                                                  while ( word_4F06418[0] != 28 );
                                                  v262 = v110;
                                                  v36 = 0;
                                                  LOBYTE(v73) = a6 != 0;
                                                  if ( !v267 )
                                                  {
                                                    LODWORD(v39) = v246;
LABEL_142:
                                                    LOBYTE(v75) = v36 == 0;
                                                    v76 = (unsigned int)v39 & (unsigned int)v75;
                                                    goto LABEL_143;
                                                  }
                                                  LODWORD(v39) = v267;
LABEL_137:
                                                  if ( (*(_BYTE *)(v52 + 33) & 1) == 0 )
                                                  {
                                                    v72 = (unsigned __int8)v240;
                                                    LOBYTE(v72) = v73 & v240;
                                                    if ( ((unsigned __int8)v73 & (unsigned __int8)v240) != 0 )
                                                    {
                                                      v66 = v246;
                                                      v73 = (unsigned int)v72;
                                                      v72 = (unsigned __int64)&dword_4F04C44;
                                                      LOBYTE(v240) = v246;
                                                      if ( dword_4F04C64 - 1 == dword_4F04C44 )
                                                      {
                                                        v66 = (__int64)&v291;
                                                        v74 = 306;
                                                        v275 = v73;
                                                        sub_6851C0(306, &v291);
                                                        v73 = v275;
                                                        LOBYTE(v240) = 0;
                                                      }
                                                    }
                                                  }
                                                }
                                                v75 = word_4F06418;
                                                if ( word_4F06418[0] != 76 )
                                                  goto LABEL_142;
                                                v269 = v73;
                                                v290 = *(_QWORD *)&dword_4F063F8;
                                                sub_7B8B50(v74, v66, v72, v73);
                                                v111 = v286;
                                                v73 = v269;
                                                v248[16] |= 1u;
                                                if ( v111 )
                                                {
                                                  sub_867030(v111);
                                                  v36 = 0;
                                                  v286 = 0;
                                                  LODWORD(v39) = v246;
                                                  v73 = v269;
                                                  v76 = 1;
                                                }
                                                else
                                                {
                                                  LODWORD(v39) = v246;
                                                  v76 = 1;
                                                  v36 = 0;
                                                }
LABEL_143:
                                                if ( a8 && (_BYTE)v73 )
                                                {
                                                  v77 = **(_QWORD **)v248;
                                                  if ( v77 )
                                                  {
                                                    if ( !v239 )
                                                      goto LABEL_151;
                                                    if ( v77 != v52 )
                                                      goto LABEL_277;
                                                    if ( (*(_BYTE *)(v52 + 32) & 4) != 0 )
                                                    {
                                                      if ( v230 )
                                                      {
                                                        v268 = v76;
                                                        sub_685360(408, &v292);
                                                        v281 = *(_QWORD **)v248;
                                                        v78 = sub_72C930();
                                                        v239 = 0;
                                                        LOBYTE(v76) = v268;
                                                        v281[1] = v78;
                                                        *(_BYTE *)(*(_QWORD *)v248 + 32LL) &= ~1u;
                                                        goto LABEL_151;
                                                      }
LABEL_277:
                                                      for ( i = *(_QWORD *)(v52 + 8);
                                                            *(_BYTE *)(i + 140) == 12;
                                                            i = *(_QWORD *)(i + 160) )
                                                      {
                                                        ;
                                                      }
                                                      if ( !unk_4D04460 )
                                                      {
                                                        if ( a6 == i
                                                          || (v270 = v76,
                                                              v114 = sub_8D97D0(a6, i, 0, v73, v76),
                                                              LOBYTE(v76) = v270,
                                                              v114) )
                                                        {
                                                          v271 = v76;
                                                          sub_685360(408, &v289);
                                                          v115 = sub_72C930();
                                                          LOBYTE(v76) = v271;
                                                          v239 = 0;
                                                          *(_QWORD *)(v52 + 8) = v115;
                                                        }
                                                      }
                                                    }
                                                    else
                                                    {
                                                      v239 = 0;
                                                    }
                                                  }
                                                  else
                                                  {
                                                    for ( j = *(_QWORD *)&v300[18];
                                                          *(_BYTE *)(j + 140) == 12;
                                                          j = *(_QWORD *)(j + 160) )
                                                    {
                                                      ;
                                                    }
                                                    if ( j == a6
                                                      || (v272 = v76,
                                                          v127 = sub_8D97D0(a6, j, 0, v73, v76),
                                                          LOBYTE(v76) = v272,
                                                          v127) )
                                                    {
                                                      if ( (_BYTE)v76 )
                                                      {
                                                        v276 = v76;
                                                        sub_685360(408, &v289);
                                                        v190 = sub_72C930();
                                                        *(_BYTE *)(v52 + 32) &= ~1u;
                                                        LOBYTE(v76) = v276;
                                                        *(_QWORD *)(v52 + 8) = v190;
                                                      }
                                                      else if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 - 772) != 9 )
                                                      {
                                                        v292 = v289;
                                                        v230 = v246;
                                                        v239 = v246;
                                                      }
                                                    }
                                                    else if ( !v39 )
                                                    {
                                                      v173 = sub_8D30C0(*(_QWORD *)&v300[18]);
                                                      LOBYTE(v76) = v272;
                                                      if ( v173 )
                                                      {
                                                        v174 = sub_8D46C0(*(_QWORD *)&v300[18]);
                                                        LOBYTE(v76) = v272;
                                                        for ( k = v174; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                                                          ;
                                                        if ( a6 == k )
                                                        {
                                                          v239 = v246;
                                                        }
                                                        else
                                                        {
                                                          v177 = sub_8D97D0(a6, k, 0, v175, v272);
                                                          LOBYTE(v76) = v272;
                                                          v80 = v177 == 0;
                                                          v178 = v239;
                                                          if ( !v80 )
                                                            v178 = v246;
                                                          v239 = v178;
                                                        }
                                                      }
                                                    }
                                                  }
                                                }
LABEL_151:
                                                v282 = v76;
                                                sub_643EB0(v300, 0);
                                                v266 = (_QWORD *)v52;
                                                if ( v282 )
                                                  goto LABEL_223;
                                                goto LABEL_152;
                                              }
                                              v90 -= 776;
                                            }
                                          }
                                          v60 = *(_QWORD *)v60;
                                          *(_QWORD *)(v287 + 40) = v60;
                                        }
                                      }
                                    }
                                  }
LABEL_121:
                                  v64 = v58->m128i_i64[1];
                                  if ( v64 )
                                  {
                                    v63 = (4 * *(_BYTE *)(v64 + 82)) & 0x10;
                                    *(_BYTE *)(v52 + 34) = v63 | *(_BYTE *)(v52 + 34) & 0xEF;
                                  }
                                  goto LABEL_123;
                                }
LABEL_262:
                                v85 = *(_BYTE *)(v52 + 33);
                                goto LABEL_188;
                              }
                            }
                            else if ( !v251 )
                            {
                              goto LABEL_262;
                            }
                            *(_BYTE *)(v52 + 34) |= 0x10u;
                            v85 = *(_BYTE *)(v52 + 33);
                            goto LABEL_188;
                          }
LABEL_183:
                          sub_65C470(v300);
                          goto LABEL_84;
                        }
LABEL_80:
                        if ( !v255 )
                          goto LABEL_81;
LABEL_179:
                        if ( unk_4D047EC )
                        {
                          v84 = v233;
                          if ( dword_4F077C4 == 2 )
                            v84 = v238;
LABEL_182:
                          sub_626F50(v84, (__int64)v300, 0, (__int64)&v293, 0, v297);
                          if ( dword_4F077C4 != 2 )
                            goto LABEL_84;
                          goto LABEL_183;
                        }
LABEL_218:
                        v84 = v238;
                        goto LABEL_182;
                      }
                      goto LABEL_393;
                    }
                    if ( word_4F06418[0] == 28 )
                    {
                      if ( (v43 & 0x40) != 0 )
                      {
                        if ( (BYTE1(v300[8]) & 0x20) != 0 )
                          sub_684B30(1727, &v289);
                        v218 = v286;
                        --*(_BYTE *)(qword_4F061C8 + 75LL);
                        sub_867030(v218);
                        v286 = 0;
                        if ( *((_QWORD *)&v300[11] + 1) )
                        {
                          v219 = sub_72C930();
                          v220 = sub_72B0C0(v219, &v289);
                          sub_645120(v300, v220);
                        }
                        goto LABEL_223;
                      }
                      if ( (unsigned int)sub_8D2600(*(_QWORD *)&v300[18]) )
                      {
                        if ( dword_4F077C4 == 2 )
                        {
                          if ( unk_4F07778 > 201102 || dword_4F07774 )
                          {
LABEL_296:
                            if ( !v45 )
                            {
                              if ( dword_4F04C44 == -1 )
                              {
                                v117 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                                if ( (*(_BYTE *)(v117 + 6) & 6) == 0
                                  && *(_BYTE *)(v117 + 4) != 12
                                  && (unk_4F04C48 == -1
                                   || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0) )
                                {
                                  v118 = 0;
                                  if ( (*(_BYTE *)(*(_QWORD *)&v300[18] + 140LL) & 0xFB) == 8 )
                                    v118 = sub_8D4C10(*(_QWORD *)&v300[18], dword_4F077C4 != 2);
                                  if ( (BYTE14(v300[7]) & 0x10) == 0 )
                                  {
                                    if ( dword_4F077C4 == 2 )
                                    {
                                      if ( unk_4F07778 > 201102 || dword_4F07774 )
                                        goto LABEL_827;
LABEL_307:
                                      v119 = 8;
                                      if ( !v118 )
                                      {
                                        v119 = 5;
                                        if ( dword_4D04964 )
                                          v119 = unk_4F07471;
                                      }
                                      sub_684AA0(v119, 494, &v289);
                                    }
                                    else
                                    {
                                      if ( unk_4F07778 <= 199900 )
                                        goto LABEL_307;
LABEL_827:
                                      if ( v118 )
                                        sub_6851C0(2410, &v289);
                                    }
                                  }
                                  v120 = v286;
                                  --*(_BYTE *)(qword_4F061C8 + 75LL);
                                  sub_867030(v120);
                                  v286 = 0;
                                  goto LABEL_223;
                                }
                              }
                              sub_6851C0(526, &v289);
                              *(_QWORD *)&v300[17] = sub_72C930();
                              *((_QWORD *)&v300[17] + 1) = *(_QWORD *)&v300[17];
                              *(_QWORD *)&v300[18] = *(_QWORD *)&v300[17];
                            }
                            goto LABEL_220;
                          }
                        }
                        else if ( unk_4F07778 > 199900 )
                        {
                          goto LABEL_296;
                        }
                        if ( (*(_BYTE *)(*(_QWORD *)&v300[18] + 140LL) & 0xFB) != 8
                          || !(unsigned int)sub_8D4C10(*(_QWORD *)&v300[18], dword_4F077C4 != 2) )
                        {
                          goto LABEL_296;
                        }
                      }
                    }
LABEL_220:
                    if ( a10 )
                    {
                      sub_6851C0(407, dword_4F07508);
                    }
                    else if ( a11 )
                    {
                      sub_6851C0(2034, dword_4F07508);
                    }
                    goto LABEL_68;
                  }
                }
LABEL_61:
                v42 = 1;
                goto LABEL_62;
              }
            }
            else
            {
              LOBYTE(v41) = (8 * (word_4D04430 & 1)) | BYTE11(v300[7]) & 0xF7;
              BYTE11(v300[7]) = v41;
              if ( dword_4F04C64 == -1 )
                goto LABEL_57;
            }
            BYTE11(v300[7]) = (32 * (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1)) | v41 & 0xDF;
            goto LABEL_57;
          }
          v36 = 0;
          if ( !v39 )
            continue;
LABEL_223:
          v246 = v237;
          goto LABEL_224;
        }
        v255 = 1;
      }
      else
      {
        v259 = 0;
        v255 = 0;
      }
      break;
    }
    ++v262;
    v251 = 0;
    goto LABEL_156;
  }
  if ( v254 )
  {
    sub_643E40(sub_622E00, a1, 1);
    if ( dword_4F077C4 == 2 )
      sub_684AC0(unk_4F07470, 267);
  }
  else
  {
    sub_684B00(3117, &v288);
    sub_6851C0(92, dword_4F07508);
  }
  do
  {
    ++*(_BYTE *)(qword_4F061C8 + 75LL);
    if ( word_4F06418[0] == 1 )
    {
      if ( dword_4F077C4 != 1 && sub_6512E0(0, 0, 1, 0, 0, 0) )
        sub_6851C0((unsigned int)(dword_4F077C4 != 2) + 150, dword_4F07508);
      sub_885C70((unsigned int)&qword_4D04A00, 0, 0, 0, (_DWORD)v261, 0, (__int64)&v285, 0);
      v143 = qword_4D04A08;
      v285[7].m128i_i64[0] = qword_4D04A08;
      sub_7B8B50(&qword_4D04A00, 0, v143, v144);
    }
    else if ( word_4F06418[0] == 76 && (unsigned __int16)sub_7BE840(0, 0) == 28 )
    {
      sub_684AC0(8, 819);
      sub_7B8B50(8, 819, v182, v183);
    }
    else
    {
      sub_7BE280(1, 40, 0, 0);
    }
    --*(_BYTE *)(qword_4F061C8 + 75LL);
  }
  while ( (unsigned int)sub_7BE800(67) );
  v236 = 0;
  v246 = 0;
LABEL_229:
  if ( a14 )
    *(_QWORD *)(a14 + 56) = *(_QWORD *)&dword_4F063F8;
  v98 = 28;
  unk_4F061D8 = qword_4F063F0;
  v99 = qword_4F061C8;
  v100 = *(_BYTE *)(qword_4F061C8 + 75LL);
  v101 = *(_BYTE *)(qword_4F061C8 + 64LL);
  *(_BYTE *)(qword_4F061C8 + 75LL) = 0;
  *(_BYTE *)(v99 + 64) = 0;
  sub_7BE280(28, 18, 0, 0);
  v103 = (char *)qword_4F04C68;
  v104 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  *(_BYTE *)(v104 + 75) = v100;
  *(_BYTE *)(v104 + 64) = v101;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) |= 0x40u;
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] == 281 )
    {
      sub_7B8B50(28, 18, v102, qword_4F04C68);
      if ( word_4F06418[0] == 27 )
      {
        sub_7BDC20(0);
        if ( word_4F06418[0] == 28 )
          sub_7B8B50(0, 18, v212, v213);
      }
      else
      {
        sub_6851C0(125, &dword_4F063F8);
      }
    }
    goto LABEL_233;
  }
  v128 = *(_BYTE *)(a1 + 133);
  if ( (v128 & 0x20) == 0 || !*(_QWORD *)(a1 + 368) )
  {
    if ( (v128 & 1) != 0 && (v248[16] & 1) != 0 )
    {
      v98 = 3055;
      sub_6851C0(3055, &v290);
    }
    v129 = a1;
    v130 = 1;
    v283 = *a3;
    v260 = *(_QWORD *)(*a3 + 168);
    v131 = *(_BYTE *)(a1 + 131);
    if ( v232 )
    {
      if ( (v131 & 8) != 0 )
        goto LABEL_344;
    }
    else
    {
      v103 = (char *)(v131 & 0x20);
      v102 = *(_BYTE *)(a1 + 123) & 1;
      LOBYTE(v102) = (unsigned __int8)v103 | v102;
      v130 = (_BYTE)v102 != 0;
      if ( (v131 & 8) != 0 )
      {
LABEL_344:
        v132 = 0x6004000001LL;
        v133 = 0;
        while ( 2 )
        {
          while ( 1 )
          {
            v134 = word_4F06418[0];
            if ( word_4F06418[0] != 174 )
              break;
LABEL_351:
            v137 = **(_QWORD **)(v283 + 168);
            if ( v137 && (*(_BYTE *)(v137 + 35) & 1) != 0 )
            {
              v129 = (__int64)dword_4F07508;
              v98 = 3209;
              sub_6851C0(3209, dword_4F07508);
            }
            else if ( *(_BYTE *)(a1 + 269) == 2 )
            {
              v129 = (__int64)&dword_4F063F8;
              v98 = 3394;
              sub_6851C0(3394, &dword_4F063F8);
            }
            else if ( v133 )
            {
              v129 = (__int64)&dword_4F063F8;
              v98 = 240;
              sub_6851C0(240, &dword_4F063F8);
            }
            else
            {
              v180 = *((_QWORD *)v261 + 7);
              if ( v180 )
              {
                *(_BYTE *)(v180 + 24) |= 2u;
                v102 = *(_QWORD *)&dword_4F063F8;
                *(_QWORD *)(*((_QWORD *)v261 + 7) + 60LL) = *(_QWORD *)&dword_4F063F8;
              }
            }
            sub_7B8B50(v98, v129, v102, v103);
            v133 = 1;
          }
          while ( 1 )
          {
            v135 = (unsigned int)(v134 - 244);
            if ( (unsigned __int16)(v134 - 244) <= 1u )
            {
              if ( dword_4D041E0 )
              {
                v103 = v261;
                v135 = *((_QWORD *)v261 + 7);
                if ( v135 )
                {
                  v103 = (char *)*(unsigned __int8 *)(v135 + 24);
                  if ( ((unsigned __int8)v103 & 0xC) != 0 )
                  {
                    v98 = 240;
                    v129 = (__int64)&dword_4F063F8;
                    if ( (((unsigned __int8)v103 & 8) != 0) != ((_WORD)v134 == 245) )
                      v98 = 2923;
                    sub_6851C0(v98, &dword_4F063F8);
                    v103 = v261;
                    LOWORD(v134) = word_4F06418[0];
                    v135 = *((_QWORD *)v261 + 7);
                  }
                  if ( (_WORD)v134 == 244 )
                    *(_BYTE *)(v135 + 24) |= 4u;
                  else
                    *(_BYTE *)(v135 + 24) |= 8u;
                  v135 = *(_QWORD *)&dword_4F063F8;
                  *(_QWORD *)(*((_QWORD *)v261 + 7) + 68LL) = *(_QWORD *)&dword_4F063F8;
                }
              }
              else
              {
                v129 = (__int64)&dword_4F063F8;
                v98 = 2853;
                sub_6851C0(2853, &dword_4F063F8);
              }
              goto LABEL_541;
            }
            if ( (_WORD)v134 == 100 )
              break;
            v136 = (unsigned int)(v134 - 81);
            if ( (unsigned __int16)(v134 - 81) <= 0x26u )
            {
              if ( !_bittest64(&v132, v136) )
              {
                if ( *(_BYTE *)(a1 + 269) != 2 )
                  goto LABEL_554;
LABEL_643:
                if ( (_WORD)v134 != 52 )
                {
                  if ( !a5 )
                  {
                    v150 = 0;
                    goto LABEL_646;
                  }
                  v149 = 0;
                  v148 = 0;
                  v150 = 0;
                  v147 = 0;
LABEL_423:
                  if ( (*((_DWORD *)a5 + 4) & 0x22000) == 0x22000 )
                  {
                    v148 = 0;
                  }
                  else
                  {
LABEL_424:
                    if ( v149 )
                      goto LABEL_425;
                  }
LABEL_438:
                  if ( v147 )
                  {
LABEL_439:
                    v256 = 0;
                    if ( (*(_BYTE *)(a1 + 131) & 8) != 0 && dword_4F077BC )
                    {
                      v98 = 9;
                      if ( (_DWORD)qword_4F077B4 )
                        v256 = (const __m128i *)sub_5CC970(9);
                      else
                        v256 = (const __m128i *)sub_5CC190(9);
                    }
                    v152 = word_4F06418[0];
                    v153 = dword_4D048B8;
                    LOBYTE(v129) = word_4F06418[0] == 243;
                    if ( word_4F06418[0] == 162
                      || word_4F06418[0] == 281
                      || (v154 = dword_4D048B8 | (unsigned __int8)v129) != 0 )
                    {
                      v154 = *(_QWORD *)&dword_4F063F8;
                      *((_QWORD *)v261 + 3) = *(_QWORD *)&dword_4F063F8;
                    }
                    LOBYTE(v148) = v152 != 162;
                    LOBYTE(v154) = v152 != 281;
                    if ( v152 != 281 && v152 != 162 )
                    {
                      v155 = 0;
                      if ( v152 != 243 )
                        goto LABEL_482;
                    }
                    if ( a13 )
                    {
                      if ( v254 && (_BYTE)v129 )
                      {
                        v155 = 0;
                        if ( (unsigned __int16)sub_7BE840(0, 0) == 27 )
                          goto LABEL_690;
                        goto LABEL_782;
                      }
                      v155 = 0;
LABEL_754:
                      v129 = 552;
                      v98 = v153 == 0 ? 5 : 7;
                      sub_684AA0(v98, 552, &dword_4F063F8);
                      v258 = 1;
                    }
                    else
                    {
                      if ( !v153 )
                      {
                        v250 = dword_4F06978;
                        if ( !dword_4F06978 )
                        {
                          if ( !v254 || !(_BYTE)v129 )
                          {
                            v258 = 1;
                            v155 = 0;
                            goto LABEL_454;
                          }
                          if ( (unsigned __int16)sub_7BE840(0, 0) != 27 )
                          {
                            v129 = 0;
                            v98 = 0;
                            v155 = 0;
                            if ( (unsigned __int16)sub_7BE840(0, 0) == 27 )
                            {
                              v258 = 1;
LABEL_456:
                              sub_7B8B50(v98, v129, v154, v148);
                              sub_7B80F0();
                              v158 = qword_4F061C8;
                              ++*(_BYTE *)(qword_4F061C8 + 83LL);
                              ++*(_BYTE *)(v158 + 81);
                              ++*(_BYTE *)(v158 + 36);
                              if ( word_4F06418[0] != 27 )
                              {
                                sub_6851C0(125, dword_4F07508);
                                goto LABEL_458;
                              }
                              sub_7B8B50(v98, v129, v156, v157);
                              if ( v152 == 243 )
                              {
                                v225 = v254
                                    && ((*(_BYTE *)(a1 + 8) & 8) == 0 || (unsigned int)sub_899810())
                                    && (*(_QWORD *)(a1 + 128) & 0x80008000000LL) != 0x8000000;
                                sub_623870(v155, v225, a1, (__int64 *)v261);
                                goto LABEL_733;
                              }
                              if ( word_4F06418[0] == 28 )
                              {
                                if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 && !v250 && dword_4F06978 )
                                  sub_684AA0(4, 3182, v261 + 24);
LABEL_733:
                                if ( !v155 )
                                {
                                  v155 = 0;
LABEL_481:
                                  --*(_BYTE *)(qword_4F061C8 + 36LL);
                                  sub_7BE280(28, 18, 0, 0);
                                  v167 = qword_4F061C8;
                                  --*(_BYTE *)(qword_4F061C8 + 81LL);
                                  --*(_BYTE *)(v167 + 83);
                                  sub_7B8160();
LABEL_482:
                                  *(_QWORD *)(v260 + 56) = v155;
                                  v168 = (const __m128i *)sub_5CC190(9);
                                  v169 = (__int64)v256;
                                  v172 = sub_5CF720(v256, v168);
                                  if ( word_4F06418[0] == 30 )
                                  {
                                    v207 = a1;
                                    v208 = *(_DWORD *)&word_4D04430;
                                    v209 = *(_BYTE *)(a1 + 131) & 8;
                                    if ( *(_DWORD *)&word_4D04430 )
                                    {
                                      v208 = 0;
                                      if ( !v209 )
                                      {
                                        v222 = *(_BYTE *)(a1 + 125);
                                        if ( (v222 & 0x10) == 0 )
                                        {
                                          if ( (v222 & 5) == 1 )
                                          {
                                            if ( (*(_BYTE *)(a1 + 122) & 2) != 0 )
                                            {
                                              v207 = (__int64)dword_4F07508;
                                              v169 = 1824;
                                              v208 = 1;
                                              sub_6851C0(1824, dword_4F07508);
                                            }
                                            else
                                            {
                                              v207 = *(_QWORD *)(a1 + 304);
                                              if ( *(_QWORD *)(a1 + 288) != v207 )
                                              {
                                                v207 = a1 + 40;
                                                v169 = 1825;
                                                v208 = 1;
                                                sub_6851C0(1825, a1 + 40);
                                              }
                                            }
                                          }
                                          else
                                          {
                                            v207 = (__int64)dword_4F07508;
                                            v169 = 1823;
                                            v208 = 1;
                                            sub_6851C0(1823, dword_4F07508);
                                          }
                                        }
                                      }
                                    }
                                    else if ( !v209 && (*(_BYTE *)(a1 + 125) & 0x10) == 0 )
                                    {
                                      goto LABEL_483;
                                    }
                                    if ( *(char *)(a1 + 121) < 0 && *(_QWORD *)(a1 + 312) )
                                    {
                                      v169 = 5;
                                      if ( dword_4D04964 )
                                        v169 = unk_4F07471;
                                      v207 = 2409;
                                      sub_684AA0(v169, 2409, a1 + 104);
                                    }
                                    *(_BYTE *)(a1 + 124) &= ~0x80u;
                                    sub_7B8B50(v169, v207, v170, v171);
                                    v210 = *(_QWORD *)&dword_4F063F8;
                                    *(_QWORD *)(a1 + 56) = *(_QWORD *)&dword_4F063F8;
                                    memset(v300, 0, 0x1D8u);
                                    *((_QWORD *)&v300[1] + 1) = v210;
                                    *((_QWORD *)&v300[9] + 1) = v300;
                                    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
                                      BYTE2(v300[11]) |= 1u;
                                    WORD5(v300[7]) = WORD5(v300[7]) & 0xF7EF | ((word_4D04430 & 1) << 11) | 0x10;
                                    if ( unk_4F0774C && (*(_BYTE *)(a1 + 124) & 0x40) != 0 )
                                      BYTE12(v300[7]) |= 0x40u;
                                    if ( unk_4D047B8 )
                                      sub_6256B0(0);
                                    *(_BYTE *)(a1 + 123) |= 0x10u;
                                    sub_65C7C0(v300);
                                    if ( unk_4D047B8 )
                                      sub_6256B0(1);
                                    if ( v208 )
                                    {
                                      v223 = sub_72C930();
                                      *(_QWORD *)(a1 + 288) = v223;
                                      *(_QWORD *)(a1 + 280) = v223;
                                      *(_QWORD *)(a1 + 272) = v223;
                                      *(_QWORD *)(a1 + 120) &= 0xFFFFFC7FEFFFFFFFLL;
                                    }
                                    else
                                    {
                                      v211 = *(_QWORD *)&v300[18];
                                      *(_QWORD *)(a1 + 288) = *(_QWORD *)&v300[18];
                                      *(_QWORD *)(a1 + 280) = v211;
                                      *(_QWORD *)(a1 + 272) = v211;
                                      *(_BYTE *)(*(_QWORD *)(v283 + 168) + 16LL) |= 8u;
                                      if ( SBYTE12(v300[7]) < 0
                                        && (*((_QWORD *)&v300[7] + 1) & 0x80010000000LL) != 0x10000000 )
                                      {
                                        *(_BYTE *)(a1 + 125) |= 8u;
                                      }
                                    }
                                  }
                                  else
                                  {
LABEL_483:
                                    *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 32);
                                  }
                                  if ( v172 )
                                    sub_5CC9F0((__int64)v172);
LABEL_233:
                                  sub_622ED0(a1, a3);
                                  goto LABEL_234;
                                }
LABEL_480:
                                *(_QWORD *)(v155 + 24) = *(_QWORD *)&dword_4F063F8;
                                goto LABEL_481;
                              }
LABEL_458:
                              v159 = 0;
                              while ( 1 )
                              {
                                ++*(_BYTE *)(qword_4F061C8 + 75LL);
                                if ( (unsigned int)sub_869470(v297) )
                                  break;
LABEL_472:
                                --*(_BYTE *)(qword_4F061C8 + 75LL);
                                if ( word_4F06418[0] == 75
                                  || word_4F06418[0] == 28
                                  || (word_4F06418[0] & 0xFFBF) == 9
                                  || !(unsigned int)sub_7BE800(67) )
                                {
                                  if ( v155 )
                                  {
                                    if ( !*(_QWORD *)(v155 + 8) && (*(_BYTE *)v155 & 5) != 4 || v250 )
                                      goto LABEL_480;
                                    if ( !dword_4F06978 )
                                    {
                                      if ( dword_4F077C4 == 2
                                        && (unk_4F07778 > 201102 || dword_4F07774)
                                        && !v258
                                        && !sub_6440B0(99, a1) )
                                      {
                                        v224 = 4;
                                        if ( dword_4F077C4 == 2 )
                                          v224 = (unsigned int)(unk_4F07778 >= 201402) + 4;
                                        sub_684AA0(v224, 2381, v261 + 24);
                                      }
                                      goto LABEL_480;
                                    }
                                    sub_684AA0(7, 2863, v261 + 24);
                                  }
                                  v155 = 0;
                                  goto LABEL_481;
                                }
                              }
                              while ( 2 )
                              {
                                v160 = sub_725E90();
                                v161 = *(_QWORD *)&dword_4F063F8;
                                *(_QWORD *)(v160 + 20) = *(_QWORD *)&dword_4F063F8;
                                *(_QWORD *)&v300[0] = v161;
                                sub_65CD60(v160 + 8);
                                sub_645520(v160 + 8);
                                v162 = sub_73D4C0(*(_QWORD *)(v160 + 8), dword_4F077C4 == 2);
                                *(_QWORD *)(v160 + 8) = v162;
                                v165 = *(_BYTE *)(v162 + 140);
                                for ( m = v162; v165 == 12; v165 = *(_BYTE *)(v162 + 140) )
                                  v162 = *(_QWORD *)(v162 + 160);
                                if ( v165 )
                                {
                                  if ( dword_4D04964 )
                                  {
                                    if ( (unsigned int)sub_8D3110(m) )
                                    {
                                      v197 = sub_8D46C0(*(_QWORD *)(v160 + 8));
                                      if ( !(unsigned int)sub_8D3D40(v197) )
                                      {
                                        sub_684AA0(unk_4F07471, 2467, v300);
                                        goto LABEL_466;
                                      }
                                    }
                                  }
                                  v164 = dword_4D048B8;
                                  if ( dword_4D048B8 )
                                  {
                                    if ( (((unsigned __int8)v258 ^ 1) & 1) != 0 )
                                    {
                                      v198 = *(_QWORD *)(v160 + 8);
                                      if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(*(_QWORD *)(v160 + 8)) )
                                        sub_8AE000(v198);
                                      if ( (unsigned int)sub_8D23B0(v198) && !(unsigned int)sub_8E34C0(v198) )
                                      {
                                        v202 = sub_67F240(v198);
                                        v164 = v202;
                                        if ( !v202 )
                                          goto LABEL_466;
                                      }
                                      else
                                      {
                                        if ( !(unsigned int)sub_8D3320(v198) )
                                          goto LABEL_466;
                                        v198 = sub_8D46C0(v198);
                                        if ( (unsigned int)sub_8D2600(v198) )
                                          goto LABEL_466;
                                        if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v198) )
                                          sub_8AE000(v198);
                                        if ( !(unsigned int)sub_8D23B0(v198) || (unsigned int)sub_8E34C0(v198) )
                                          goto LABEL_466;
                                        LODWORD(v164) = 833;
                                      }
                                      if ( !v254 || (v253 = v164, v215 = sub_8D2600(v198), LODWORD(v164) = v253, v215) )
                                      {
                                        if ( dword_4D04964 )
                                        {
                                          sub_685A50((unsigned int)v164, v300, v198, unk_4F07471);
                                        }
                                        else
                                        {
                                          v252 = v164;
                                          v214 = sub_8D2600(v198);
                                          v164 = v252;
                                          if ( v214 )
                                            sub_685A50(v252, v300, v198, 8);
                                        }
                                      }
                                      else
                                      {
                                        sub_87E2E0(v261, v253, v300, v198);
                                      }
                                    }
                                  }
                                }
LABEL_466:
                                if ( v155 )
                                {
                                  if ( v159 )
                                  {
                                    v193 = *(_QWORD *)(v160 + 8);
                                    for ( n = *(_BYTE *)(v193 + 140); n == 12; n = *(_BYTE *)(v193 + 140) )
                                      v193 = *(_QWORD *)(v193 + 160);
                                    if ( n )
                                    {
                                      v199 = *(__int64 **)(v155 + 8);
                                      if ( v199 )
                                      {
                                        while ( 1 )
                                        {
                                          if ( !*((_BYTE *)v199 + 16) )
                                          {
                                            v200 = *(_QWORD *)(v160 + 8);
                                            v201 = v199[1];
                                            if ( v200 == v201 || (unsigned int)sub_8D97D0(v200, v201, 0, v163, v164) )
                                              break;
                                          }
                                          v199 = (__int64 *)*v199;
                                          if ( !v199 )
                                            goto LABEL_651;
                                        }
                                        sub_684B00(535, v300);
                                        *(_BYTE *)(v160 + 16) = 1;
                                      }
                                    }
LABEL_651:
                                    *v159 = v160;
                                    v159 = (_QWORD *)v160;
                                    if ( !*(_BYTE *)(v160 + 16) )
                                      goto LABEL_652;
                                  }
                                  else
                                  {
                                    *(_QWORD *)(v155 + 8) = v160;
                                    v159 = (_QWORD *)v160;
                                    if ( *(_BYTE *)(v160 + 16) )
                                      goto LABEL_469;
LABEL_652:
                                    v195 = *(_BYTE *)(*(_QWORD *)(v160 + 8) + 140LL);
                                    if ( v195 == 12 )
                                    {
                                      v196 = *(_QWORD *)(v160 + 8);
                                      do
                                      {
                                        v196 = *(_QWORD *)(v196 + 160);
                                        v195 = *(_BYTE *)(v196 + 140);
                                      }
                                      while ( v195 == 12 );
                                    }
                                    v159 = (_QWORD *)v160;
                                    if ( v195 )
                                      sub_8DCE90();
                                  }
                                }
LABEL_469:
                                if ( sub_867630(v297[0].m128i_i64[0], 0) )
                                  *(_BYTE *)(v160 + 17) = 1;
                                if ( !(unsigned int)sub_866C00(v297[0].m128i_i64[0]) )
                                  goto LABEL_472;
                                continue;
                              }
                            }
LABEL_697:
                            sub_7B8B50(0, 0, v154, v148);
                            goto LABEL_482;
                          }
LABEL_690:
                          if ( dword_4F04C44 != -1
                            || (v203 = qword_4F04C68[0],
                                v226 = qword_4F04C68[0] + 776LL * dword_4F04C64,
                                (*(_BYTE *)(v226 + 6) & 6) != 0)
                            || *(_BYTE *)(v226 + 4) == 12
                            || (v155 = 0, unk_4F04C48 != -1)
                            && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
                          {
                            v204 = sub_725E60(0, 0, v203);
                            *(_BYTE *)v204 |= 1u;
                            v155 = v204;
                            *(_QWORD *)(v204 + 16) = *(_QWORD *)&dword_4F063F8;
                          }
                          if ( !a13 )
                          {
                            v258 = 1;
LABEL_694:
                            v129 = 0;
                            v98 = 0;
                            if ( (unsigned __int16)sub_7BE840(0, 0) == 27 )
                            {
                              v250 = 0;
                              goto LABEL_456;
                            }
                            if ( v155 )
                            {
                              v205 = qword_4F063F0;
                              *(_QWORD *)(v155 + 8) = 0;
                              *(_QWORD *)(v155 + 24) = v205;
                            }
                            goto LABEL_697;
                          }
LABEL_782:
                          v153 = dword_4D048B8;
                          goto LABEL_754;
                        }
                      }
                      v206 = sub_725E60(v98, v129, v154);
                      v129 = (unsigned __int8)v129;
                      v155 = v206;
                      *(_BYTE *)v206 = v129 | *(_BYTE *)v206 & 0xFE;
                      *(_QWORD *)(v206 + 16) = *(_QWORD *)&dword_4F063F8;
                      v258 = unk_4D04324;
                      if ( unk_4D04324 )
                      {
                        v129 = 876;
                        v98 = (__int64)&dword_4F063F8;
                        sub_684AB0(&dword_4F063F8, 876);
                        v258 = 0;
                      }
                    }
LABEL_454:
                    if ( v152 != 243 )
                    {
                      v250 = word_4F06418[0] == 281;
                      goto LABEL_456;
                    }
                    goto LABEL_694;
                  }
LABEL_647:
                  v192 = v148;
                  v148 &= 0x7Bu;
                  v129 = v260;
                  *(_WORD *)(v260 + 18) = (v150 << 14) | v148 | ((v192 & 4) << 7);
                  goto LABEL_439;
                }
LABEL_722:
                v147 = 0;
                v149 = 0;
                v148 = 0;
LABEL_723:
                v136 = (*(_BYTE *)(a1 + 129) >> 6) ^ 1u;
                LOBYTE(v136) = (a6 != 0) & ((*(_BYTE *)(a1 + 129) >> 6) ^ 1);
                v273 = v136;
                if ( (_BYTE)v136 )
                {
                  a7 = 0;
                  goto LABEL_415;
                }
                if ( v130 )
                {
                  v273 = 1;
                  a7 = 0;
                  goto LABEL_415;
                }
                a7 = 0;
                v273 = 1;
LABEL_419:
                v129 = (__int64)&dword_4F063F8;
                v98 = 2448;
                v147 = 1;
                v150 = 0;
                v263 = v148;
                sub_6851C0(2448, &dword_4F063F8);
                v148 = v263;
LABEL_420:
                v264 = v148;
                sub_7B8B50(v98, v129, v136, v148);
                v148 = v264;
LABEL_421:
                if ( a5 && v273 )
                  goto LABEL_423;
                if ( v147 | (unsigned int)v148 )
                  goto LABEL_424;
                v98 = a7;
                if ( !a7 )
                  goto LABEL_424;
                v149 = a6;
                if ( !a6 )
                {
LABEL_646:
                  LOBYTE(v148) = 0;
                  goto LABEL_647;
                }
LABEL_425:
                if ( *(_BYTE *)(v149 + 140) == 14 )
                {
                  v98 = v149;
                  v279 = v148;
                  v221 = sub_7CFE40(v149);
                  v148 = v279;
                  v149 = v221;
                  if ( (*(_BYTE *)(a1 + 10) & 8) != 0 )
                  {
                    v129 = v254;
                    if ( v254 )
                    {
                      if ( !(a8 | a11 | a10) )
                        goto LABEL_429;
                    }
                  }
LABEL_435:
                  if ( !v149 )
                    goto LABEL_438;
                }
                else if ( (*(_BYTE *)(a1 + 10) & 8) != 0 )
                {
                  v98 = v254;
                  if ( v254 )
                  {
                    if ( !(a8 | a11 | a10) )
                    {
LABEL_429:
                      if ( unk_4D0487C )
                      {
                        if ( (v148 & 1) == 0 && (dword_4F077C4 != 2 || unk_4F07778 <= 201401) )
                        {
                          v98 = 2659;
                          v274 = v148;
                          v129 = a1 + 112;
                          sub_684B30(2659, a1 + 112);
                          LODWORD(v148) = v274;
                        }
                        v148 = (unsigned int)v148 | 1;
                      }
                      else if ( (v148 & 1) == 0 )
                      {
                        *(_BYTE *)(v260 + 20) |= 0x40u;
                      }
                      goto LABEL_435;
                    }
                  }
                }
                v151 = **(_QWORD **)(v283 + 168);
                if ( !v151 || (*(_BYTE *)(v151 + 35) & 1) == 0 )
                {
                  while ( *(_BYTE *)(v149 + 140) == 12 )
                    v149 = *(_QWORD *)(v149 + 160);
                  *(_BYTE *)(v260 + 21) |= 1u;
                  *(_QWORD *)(v260 + 40) = v149;
                }
                goto LABEL_438;
              }
            }
            else
            {
              v136 = (unsigned int)(v134 - 263);
              if ( (unsigned __int16)(v134 - 263) > 3u )
              {
                v129 = a1;
                if ( *(_BYTE *)(a1 + 269) == 2 )
                {
                  if ( (_WORD)v134 != 33 )
                    goto LABEL_643;
                  goto LABEL_722;
                }
LABEL_554:
                v149 = a6;
                v148 = (unsigned int)(1 - v133);
                v147 = 0;
LABEL_412:
                if ( (_WORD)v134 == 33 )
                {
LABEL_414:
                  v273 = 0;
                  if ( !a7 )
                    goto LABEL_723;
LABEL_415:
                  if ( a8 | a11 | a10 || a5 && (a5[16] & 8) != 0 && (unsigned __int8)(a5[56] - 1) <= 3u )
                    goto LABEL_419;
                  if ( !unk_4D048A8 )
                  {
                    v129 = 3420;
                    v265 = v148;
                    v98 = (_DWORD)qword_4F077B4 == 0 ? 8 : 5;
                    sub_684AA0(v98, 3420, &dword_4F063F8);
                    v148 = v265;
                  }
                  v217 = word_4F06418;
                  LOBYTE(v217) = word_4F06418[0] != 33;
                  v150 = (_WORD)v217 + 1;
                  goto LABEL_420;
                }
LABEL_413:
                if ( (_WORD)v134 == 52 )
                  goto LABEL_414;
LABEL_625:
                v273 = a7 == 0;
                v150 = 0;
                goto LABEL_421;
              }
            }
            sub_6851C0(1760, dword_4F07508);
            v98 = a14;
            v129 = 0;
            sub_624060(a14);
            v134 = word_4F06418[0];
            if ( word_4F06418[0] == 174 )
              goto LABEL_351;
          }
          if ( *(_BYTE *)(a1 + 269) == 2 )
          {
            v129 = (__int64)&dword_4F063F8;
            v98 = 240;
            sub_6851C0(240, &dword_4F063F8);
            goto LABEL_541;
          }
          if ( v133 )
          {
            v129 = (__int64)&dword_4F063F8;
            v98 = 3394;
            sub_6851C0(3394, &dword_4F063F8);
            goto LABEL_541;
          }
          v181 = *((_QWORD *)v261 + 7);
          if ( v181 && ((*(_BYTE *)(v181 + 24) & 0x10) != 0 || *(_QWORD *)v181) )
          {
            v129 = (__int64)&dword_4F063F8;
            v98 = 3396;
            sub_6851C0(3396, &dword_4F063F8);
LABEL_541:
            sub_7B8B50(v98, v129, v135, v103);
            continue;
          }
          break;
        }
        *(_BYTE *)(a1 + 269) = 2;
        if ( dword_4F077C4 == 2 && unk_4F07778 > 202301 )
          goto LABEL_541;
        if ( dword_4F077BC )
        {
          if ( !(_DWORD)qword_4F077B4 )
          {
            v98 = qword_4F077A8 < 0x1FBD0u ? 7 : 5;
LABEL_566:
            v129 = 3395;
            sub_684AA0(v98, 3395, &dword_4F063F8);
            goto LABEL_541;
          }
        }
        else
        {
          v98 = 7;
          if ( !(_DWORD)qword_4F077B4 )
            goto LABEL_566;
        }
        v98 = unk_4F077A0 < 0x27100u ? 7 : 5;
        goto LABEL_566;
      }
    }
    LOWORD(v134) = word_4F06418[0];
    v136 = (unsigned int)word_4F06418[0] - 81;
    if ( (unsigned __int16)(word_4F06418[0] - 81) <= 0x26u )
    {
      v191 = 0x6004000001LL;
      if ( !_bittest64(&v191, v136) )
      {
        v147 = 0;
        v149 = 0;
        v148 = 0;
        goto LABEL_413;
      }
      if ( (*(_BYTE *)(v260 + 16) & 2) == 0 )
      {
        v147 = 0;
        v149 = 0;
        v148 = 0;
        goto LABEL_625;
      }
    }
    else
    {
      v136 = (unsigned int)word_4F06418[0] - 263;
      if ( (unsigned __int16)(word_4F06418[0] - 263) > 3u || (*(_BYTE *)(v260 + 16) & 2) == 0 )
      {
        v147 = 0;
        v148 = 0;
        v149 = 0;
        goto LABEL_412;
      }
    }
    v98 = a14;
    v129 = 0;
    *(_QWORD *)&v300[0] = *(_QWORD *)&dword_4F063F8;
    v148 = (unsigned int)sub_624060(a14);
    if ( !a5 || (a5[16] & 8) == 0 || (v149 = 0, v98 = 1669, (unsigned __int8)(a5[56] - 1) > 3u) )
    {
      if ( a6 || (v149 = 0, v98 = 1670, v130) )
      {
        v147 = a7 | v130;
        if ( !(a7 | v130) && ((*(_BYTE *)(a1 + 129) & 0x40) != 0 || a9) )
        {
          v149 = 0;
          v98 = 1667;
          v136 = (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
          if ( qword_4D0495C && a6 )
          {
            v129 = (__int64)v300;
            v278 = v148;
            sub_684B30(1667, v300);
            v148 = v278;
LABEL_639:
            LOWORD(v134) = word_4F06418[0];
            goto LABEL_412;
          }
        }
        else
        {
          v149 = a6;
          v147 = a8 | a11 | a10;
          if ( !v147 )
            goto LABEL_639;
          v148 = 0;
          v98 = a11 == 0 ? 1668 : 2035;
          if ( HIDWORD(qword_4D0495C) )
          {
            v129 = (__int64)v300;
            v147 = 0;
            sub_684B30(v98, v300);
            v148 = 0;
            goto LABEL_639;
          }
        }
      }
    }
    v147 = 1;
    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 9 )
    {
      v129 = (__int64)v300;
      v277 = v148;
      sub_6851C0(v98, v300);
      v148 = v277;
    }
    goto LABEL_639;
  }
LABEL_234:
  if ( v236 )
    sub_863FC0();
  if ( v254 )
  {
    if ( v246 )
    {
      v123 = *(_QWORD **)v248;
      v124 = (__int64 *)*((_QWORD *)v261 + 1);
      if ( *(_QWORD *)v248 )
      {
        while ( v124 )
        {
          v125 = *((_DWORD *)v123 + 9);
          while ( *((_DWORD *)v124 + 30) < v125 )
          {
            v124 = (__int64 *)*v124;
            if ( !v124 )
              goto LABEL_327;
          }
          if ( *((_DWORD *)v124 + 30) == v125 )
            *((_DWORD *)v123 + 8) = v123[4] & 0xFFFC07FF
                                  | (((unsigned __int8)~(*((_BYTE *)v124 + 41) & 0x7F)
                                    & (unsigned __int8)(*((_DWORD *)v123 + 8) >> 11)
                                    & 0x7F) << 11);
LABEL_327:
          v123 = (_QWORD *)*v123;
          if ( !v123 )
            goto LABEL_241;
        }
        do
        {
          v179 = (_QWORD *)*v123;
          if ( !*v123 )
            break;
          v123 = (_QWORD *)*v179;
        }
        while ( *v179 );
      }
    }
  }
  else if ( dword_4F04C64 == -1
         || (v105 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v105 + 7) & 1) == 0)
         || dword_4F04C44 == -1 && (*(_BYTE *)(v105 + 6) & 2) == 0 )
  {
    if ( (v299[57] & 8) == 0 )
      sub_87E280(v299);
  }
LABEL_241:
  *(_QWORD *)dword_4F07508 = v288;
  *(_BYTE *)(a1 + 124) |= 0x10u;
  result = (_QWORD *)(*(_QWORD *)(a1 + 120) & 0x800002000000LL);
  if ( !result )
  {
    result = *(_QWORD **)(a1 + 368);
    v107 = 0;
    if ( result )
    {
      while ( 1 )
      {
        v108 = (_QWORD *)*result;
        *result = v107;
        v107 = result;
        if ( !v108 )
          break;
        result = v108;
      }
      *(_QWORD *)(a1 + 368) = result;
    }
  }
  return result;
}
