// Function: sub_64BAA0
// Address: 0x64baa0
//
__int64 __fastcall sub_64BAA0(__m128i *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  int v8; // ecx
  char v9; // dl
  int v10; // ecx
  __int64 v11; // rdi
  _BOOL4 v12; // r15d
  char v13; // al
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rsi
  char v17; // al
  char v18; // al
  char v19; // cl
  __int64 v20; // r8
  unsigned __int64 v21; // rdx
  _QWORD *v22; // r9
  __m128i v23; // xmm6
  __int64 v24; // rax
  __int64 v25; // r11
  __int64 v26; // r10
  __int64 i; // rax
  int v28; // eax
  __m128i v29; // xmm7
  __int64 v30; // rax
  int v31; // r13d
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r9
  __m128i v39; // xmm5
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rcx
  char v45; // dl
  unsigned int v46; // r15d
  char v47; // al
  __int64 v48; // rax
  char v49; // r8
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // rsi
  __m128i v53; // xmm5
  __int64 v54; // rax
  char v55; // r8
  char v56; // al
  __int64 v57; // rsi
  _DWORD *v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 m; // rbx
  _QWORD *v63; // rbx
  __int64 v64; // rsi
  __int64 v65; // r14
  char v66; // al
  __int64 jj; // rbx
  __int64 *v68; // rax
  __int64 v69; // r14
  __int64 kk; // rsi
  int v71; // eax
  __int64 v72; // rax
  _BYTE *v73; // rsi
  _QWORD *mm; // rdx
  __int64 v75; // rcx
  char v76; // al
  char v77; // al
  __int64 v79; // rdx
  __int64 v80; // rax
  char v81; // al
  char v82; // al
  bool v83; // sf
  __int8 v84; // al
  __int64 *v85; // r15
  __int64 *v86; // rdx
  __int64 *v87; // r8
  unsigned __int8 v88; // al
  __int64 *v89; // r15
  __int64 v90; // rdi
  __int64 v91; // r15
  __int64 v92; // rax
  __int64 v93; // r15
  __int64 ii; // rax
  _QWORD *v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  char v101; // di
  __int64 **v102; // rax
  __int64 j; // rax
  __int64 **v104; // r13
  __int64 **v105; // rbx
  const char *v106; // r8
  __int64 v107; // r8
  __int64 k; // rax
  __int64 *v109; // r15
  __int64 v110; // rcx
  __int64 *v111; // r13
  unsigned int v112; // ebx
  int v113; // eax
  unsigned __int64 v114; // rdx
  unsigned __int64 v115; // rsi
  int v116; // edi
  __int64 v117; // rsi
  _BYTE *v118; // rsi
  unsigned int v119; // edx
  __int64 v120; // rcx
  unsigned int v121; // eax
  unsigned int v122; // r8d
  __int64 v123; // rax
  __int64 v124; // r10
  __int64 v125; // r9
  const char *v126; // r14
  void *v127; // r11
  size_t v128; // r12
  _QWORD *v129; // rax
  size_t v130; // rdx
  __int64 v131; // rcx
  __m128i *v132; // rax
  __int64 v133; // rcx
  __m128i *v134; // rax
  __m128i *v135; // rdi
  __int64 v136; // r8
  char v137; // al
  __int64 v138; // rsi
  __int64 v139; // rdi
  __int64 v140; // r15
  __int64 v141; // rax
  bool v142; // cf
  bool v143; // zf
  char *v144; // rdi
  __m128i v145; // xmm2
  __int64 v146; // rax
  __int64 v147; // r10
  __int64 v148; // rax
  __int64 v149; // rax
  int v150; // eax
  __int64 v151; // rdx
  unsigned __int64 v152; // rax
  __int64 v153; // rdx
  __m128i v154; // xmm2
  __int64 v155; // rax
  __int64 v156; // r10
  __int64 v157; // rdx
  __int64 v158; // rax
  _QWORD *v159; // rdi
  size_t v160; // rdx
  _BOOL8 v161; // rcx
  __int64 v162; // rax
  __int64 v163; // rdx
  __int64 v164; // rdi
  __int64 v165; // rdx
  int v166; // r10d
  __int64 v167; // r8
  __int64 v168; // rdi
  __int8 v169; // al
  char v170; // al
  __int64 v171; // rax
  __int64 v172; // [rsp-8h] [rbp-1B8h]
  __int64 v173; // [rsp-8h] [rbp-1B8h]
  void *src; // [rsp+0h] [rbp-1B0h]
  __m128i *v175; // [rsp+8h] [rbp-1A8h]
  __int64 v176; // [rsp+10h] [rbp-1A0h]
  __int64 v177; // [rsp+18h] [rbp-198h]
  __int64 v178; // [rsp+20h] [rbp-190h]
  __int64 v179; // [rsp+30h] [rbp-180h]
  __int64 v180; // [rsp+30h] [rbp-180h]
  __int64 v181; // [rsp+30h] [rbp-180h]
  __int64 v182; // [rsp+30h] [rbp-180h]
  __int64 v183; // [rsp+30h] [rbp-180h]
  __int64 v184; // [rsp+30h] [rbp-180h]
  __int64 v185; // [rsp+30h] [rbp-180h]
  __int64 v186; // [rsp+30h] [rbp-180h]
  int v187; // [rsp+38h] [rbp-178h]
  __int64 v188; // [rsp+38h] [rbp-178h]
  int v189; // [rsp+38h] [rbp-178h]
  int v190; // [rsp+38h] [rbp-178h]
  __int64 v191; // [rsp+40h] [rbp-170h]
  int v193; // [rsp+50h] [rbp-160h]
  int v194; // [rsp+54h] [rbp-15Ch]
  _BOOL4 v195; // [rsp+58h] [rbp-158h]
  unsigned int v196; // [rsp+58h] [rbp-158h]
  __int64 v197; // [rsp+58h] [rbp-158h]
  __int64 *v198; // [rsp+60h] [rbp-150h]
  char v199; // [rsp+68h] [rbp-148h]
  int v200; // [rsp+68h] [rbp-148h]
  __int64 *v201; // [rsp+68h] [rbp-148h]
  __int64 *v202; // [rsp+68h] [rbp-148h]
  bool v203; // [rsp+70h] [rbp-140h]
  __int64 v204; // [rsp+70h] [rbp-140h]
  __int64 v205; // [rsp+78h] [rbp-138h]
  __int64 v206; // [rsp+78h] [rbp-138h]
  __int64 v207; // [rsp+78h] [rbp-138h]
  __int64 v208; // [rsp+78h] [rbp-138h]
  __int8 *v209; // [rsp+78h] [rbp-138h]
  __int64 v210; // [rsp+78h] [rbp-138h]
  __int64 v211; // [rsp+78h] [rbp-138h]
  __int64 v212; // [rsp+78h] [rbp-138h]
  __int64 v213; // [rsp+78h] [rbp-138h]
  __int64 v214; // [rsp+78h] [rbp-138h]
  __int64 v215; // [rsp+78h] [rbp-138h]
  __int64 v218; // [rsp+90h] [rbp-120h] BYREF
  __int64 v219; // [rsp+98h] [rbp-118h] BYREF
  void *dest; // [rsp+A0h] [rbp-110h] BYREF
  size_t n; // [rsp+A8h] [rbp-108h]
  _QWORD v222[2]; // [rsp+B0h] [rbp-100h] BYREF
  _QWORD *v223; // [rsp+C0h] [rbp-F0h] BYREF
  size_t v224; // [rsp+C8h] [rbp-E8h]
  _QWORD v225[2]; // [rsp+D0h] [rbp-E0h] BYREF
  __m128i *v226; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v227; // [rsp+E8h] [rbp-C8h]
  __m128i v228; // [rsp+F0h] [rbp-C0h] BYREF
  size_t v229; // [rsp+100h] [rbp-B0h] BYREF
  size_t v230; // [rsp+108h] [rbp-A8h]
  __m128i v231; // [rsp+110h] [rbp-A0h] BYREF
  __m128i *v232; // [rsp+120h] [rbp-90h] BYREF
  __int128 v233; // [rsp+128h] [rbp-88h]
  __int128 v234; // [rsp+138h] [rbp-78h]
  __int64 v235; // [rsp+148h] [rbp-68h]
  __int64 v236; // [rsp+150h] [rbp-60h]
  __int64 v237; // [rsp+158h] [rbp-58h]
  __int64 v238; // [rsp+160h] [rbp-50h]
  __int128 v239; // [rsp+168h] [rbp-48h]

  v6 = *(_QWORD *)a4;
  v7 = *(_QWORD *)(*(_QWORD *)a4 + 288LL);
  v198 = *(__int64 **)(a4 + 192);
  v8 = *(_DWORD *)(a4 + 200);
  v218 = v7;
  v9 = *(_BYTE *)(v6 + 269);
  v194 = v8;
  v10 = *(_DWORD *)(a4 + 24);
  v11 = *(_QWORD *)(a4 + 336);
  v219 = 0;
  v193 = v10;
  v191 = v11;
  if ( (*(_BYTE *)(a2 + 64) & 2) == 0 || (v199 = 2, dword_4D04824) )
  {
    v19 = 1;
    if ( v9 )
      v19 = v9;
    v199 = v19;
  }
  v239 = (unsigned __int64)v198;
  v235 = 0;
  BYTE4(v235) = v199;
  v236 = a2;
  v238 = 2;
  v237 = v7;
  v232 = a1;
  v233 = 0;
  v234 = 0;
  sub_6413B0((__int64)&v232, v194);
  v203 = 1;
  v195 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v235 + 6) & 4) != 0;
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v235 + 6) & 4) == 0 )
    v203 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)v235 + 6) & 0xA) != 0;
  v12 = 0;
  if ( (v238 & 1) != 0 && !unk_4D047CC )
  {
    v12 = 1;
    if ( dword_4F077BC )
    {
      if ( qword_4F077A8 <= 0xC34Fu && !(_DWORD)qword_4F077B4 )
        v12 = (a1[1].m128i_i8[0] & 8) != 0;
    }
  }
  if ( (a1[1].m128i_i32[0] & 0x20001) != 0x20001 || (v20 = a1[1].m128i_i64[1]) == 0 )
  {
    if ( (a1[1].m128i_i8[1] & 0x20) != 0 || !sub_6461D0((__int64)a1, v218, 0) )
    {
      v187 = 0;
      goto LABEL_10;
    }
    v41 = 556;
    if ( (a1[3].m128i_i8[8] & 0xFD) != 1 )
      v41 = 507;
    sub_6851C0(v41, &a1->m128i_u64[1]);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    goto LABEL_78;
  }
  v21 = *(unsigned __int8 *)(v20 + 80);
  if ( (_BYTE)v21 == 16 )
  {
    sub_6851C0(298, &a1->m128i_u64[1]);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    goto LABEL_78;
  }
  v22 = *(_QWORD **)(v20 + 64);
  if ( !v203 )
  {
    if ( (*(_BYTE *)(v20 + 81) & 0x10) != 0 && (unsigned __int8)v21 <= 0x14u )
    {
      v35 = 1180672;
      if ( _bittest64(&v35, v21) )
      {
        v188 = *v22;
        v208 = *(_QWORD *)(v20 + 64);
        v36 = sub_5EDAE0(a1[1].m128i_i64[1], v6, *v198, &v229);
        v15 = v36;
        if ( !v36 )
          goto LABEL_69;
        v38 = v208;
        if ( v229 )
        {
          sub_6854C0(266, &a1->m128i_u64[1], v36);
          goto LABEL_69;
        }
        if ( *(_BYTE *)(v36 + 80) == 20 )
        {
          v165 = 16 * (*(_DWORD *)(a4 + 24) & 1u);
          *(_BYTE *)(v6 + 127) = (16 * (*(_DWORD *)(a4 + 24) & 1)) | *(_BYTE *)(v6 + 127) & 0xEF;
        }
        else
        {
          if ( (unsigned __int8)(*(_BYTE *)(v188 + 80) - 4) > 1u || *(char *)(*(_QWORD *)(v188 + 88) + 177LL) >= 0 )
          {
LABEL_445:
            v163 = a1[1].m128i_i64[1];
            v164 = 493;
            if ( *(_BYTE *)(v163 + 80) != 17 )
              v164 = 147;
            sub_6854C0(v164, &a1->m128i_u64[1], v163);
LABEL_70:
            *a1 = _mm_loadu_si128(xmmword_4F06660);
            a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
            a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
            v39 = _mm_loadu_si128(&xmmword_4F06660[3]);
            v40 = *(_QWORD *)dword_4F07508;
            a1[1].m128i_i8[1] |= 0x20u;
            a1[3] = v39;
            a1->m128i_i64[1] = v40;
            v187 = 0;
            v13 = *(_BYTE *)(a2 + 64);
            if ( (v13 & 4) == 0 )
              goto LABEL_71;
            goto LABEL_11;
          }
          v171 = sub_880E20(v36);
          v38 = v208;
          v15 = v171;
          if ( !v171 )
          {
LABEL_69:
            if ( v229 )
              goto LABEL_70;
            goto LABEL_445;
          }
        }
        if ( dword_4D04964 )
          goto LABEL_454;
        v166 = qword_4F077B4;
        v37 = dword_4F077BC;
        if ( dword_4F077BC )
        {
          if ( !(_DWORD)qword_4F077B4 )
          {
            if ( qword_4F077A8 <= 0xEA5Fu )
              goto LABEL_455;
            goto LABEL_454;
          }
        }
        else if ( !(_DWORD)qword_4F077B4 )
        {
          goto LABEL_455;
        }
        v166 = 0;
        if ( qword_4F077A0 )
LABEL_454:
          v166 = (*(_DWORD *)(v38 + 176) & 0x11000) != 4096;
LABEL_455:
        *(_BYTE *)(v6 + 129) |= 0x80u;
        switch ( *(_BYTE *)(v15 + 80) )
        {
          case 4:
          case 5:
            v167 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 80LL);
            break;
          case 6:
            v167 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v167 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v167 = *(_QWORD *)(v15 + 88);
            break;
          default:
LABEL_486:
            BUG();
        }
        v168 = *(_QWORD *)(v167 + 176);
        v215 = *(_QWORD *)(v168 + 152);
        v169 = a1[1].m128i_i8[0];
        if ( (v169 & 0x20) != 0 || (v169 & 8) != 0 && ((a1[3].m128i_i8[8] - 2) & 0xFD) == 0 )
        {
          v185 = v167;
          v189 = v166;
          sub_5F93D0(v168, &v218);
          v167 = v185;
          v166 = v189;
        }
        v190 = v166;
        v186 = v167;
        sub_8958E0(a4, v15, v165, v37, v167, v38);
        sub_6464A0(v218, v15, (unsigned int *)(a2 + 24), 1u);
        sub_71CE00(v218, v215);
        v16 = v218;
        sub_649200(*(_QWORD *)(v186 + 176), v218, 1, 0, v6);
        if ( *(_BYTE *)(v15 + 80) == 20 )
        {
          v16 = a4;
          sub_89BD20(**(_QWORD **)(a4 + 192), a4, v15, (_DWORD)a1 + 8, v190, 0, 1, 7);
        }
        v170 = *(_BYTE *)(a2 + 64);
        if ( (v170 & 4) != 0 )
        {
          LOBYTE(v238) = v238 | 8;
          if ( v170 < 0 )
          {
            sub_6851C0(93, dword_4F07508);
            v14 = v218;
            if ( *(_BYTE *)(v218 + 140) == 12 )
            {
              v187 = 0;
              goto LABEL_14;
            }
            v16 = 1;
            v218 = sub_73EDA0(v218, 1);
          }
        }
        v187 = 0;
        goto LABEL_16;
      }
    }
    sub_6854C0(147, &a1->m128i_u64[1], a1[1].m128i_i64[1]);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
LABEL_78:
    v42 = *(_QWORD *)dword_4F07508;
    a1[1].m128i_i8[1] |= 0x20u;
    v187 = 0;
    a1->m128i_i64[1] = v42;
    goto LABEL_10;
  }
  if ( (v238 & 1) == 0 )
  {
    sub_686A10(135, &a1->m128i_u64[1], *(_QWORD *)(*(_QWORD *)v20 + 8LL), *v22);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    goto LABEL_385;
  }
  if ( word_4F06418[0] != 75 )
  {
    sub_6854C0(551, &a1->m128i_u64[1], a1[1].m128i_i64[1]);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
LABEL_385:
    v148 = *(_QWORD *)dword_4F07508;
    a1[1].m128i_i8[1] |= 0x20u;
    v187 = 1;
    a1->m128i_i64[1] = v148;
LABEL_10:
    v13 = *(_BYTE *)(a2 + 64);
    if ( (v13 & 4) == 0 )
      goto LABEL_71;
    goto LABEL_11;
  }
  v187 = 1;
  v13 = *(_BYTE *)(a2 + 64);
  if ( (v13 & 4) == 0 )
  {
    if ( (a1[1].m128i_i8[2] & 2) != 0 )
      goto LABEL_35;
LABEL_159:
    if ( a1[2].m128i_i64[0] )
      goto LABEL_36;
    goto LABEL_35;
  }
LABEL_11:
  LOBYTE(v238) = v238 | 8;
  if ( v13 >= 0 )
    goto LABEL_71;
  sub_6851C0(93, dword_4F07508);
  v14 = v218;
  if ( *(_BYTE *)(v218 + 140) != 12 )
  {
    v218 = sub_73EDA0(v218, 1);
    goto LABEL_71;
  }
  v15 = 0;
  do
LABEL_14:
    v14 = *(_QWORD *)(v14 + 160);
  while ( *(_BYTE *)(v14 + 140) == 12 );
  v16 = 1;
  v218 = sub_73EDA0(v14, 1);
  if ( v15 )
  {
LABEL_16:
    v17 = v238 & 1;
    if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
    {
      if ( v17 )
      {
        if ( (*(_BYTE *)(a2 + 64) & 4) != 0 )
        {
          v16 = (__int64)&a1->m128i_i64[1];
          sub_6854C0(551, &a1->m128i_u64[1], a1[1].m128i_i64[1]);
          v18 = *(_BYTE *)(v15 + 80);
          goto LABEL_20;
        }
LABEL_19:
        v18 = *(_BYTE *)(v15 + 80);
LABEL_20:
        switch ( v18 )
        {
          case 4:
          case 5:
            v204 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 80LL);
            goto LABEL_80;
          case 6:
            v204 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 32LL);
            goto LABEL_80;
          case 9:
          case 10:
            v204 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 56LL);
            goto LABEL_80;
          case 19:
          case 20:
          case 21:
          case 22:
            goto LABEL_79;
          default:
            goto LABEL_486;
        }
      }
    }
    else if ( v17 && ((*(_BYTE *)(a2 + 64) & 4) == 0 || v203) )
    {
      goto LABEL_19;
    }
    v16 = qword_4F04C68[0] + 776LL * (int)v235;
    if ( !(unsigned int)sub_85ED80(v15, v16) )
    {
      if ( v193 )
      {
        v16 = v15;
        sub_6854E0(503, v15);
        v18 = *(_BYTE *)(v15 + 80);
      }
      else
      {
        v16 = v15;
        if ( (*(_BYTE *)(a2 + 64) & 4) != 0 )
          sub_6854E0(551, v15);
        else
          sub_6854E0(755, v15);
        v18 = *(_BYTE *)(v15 + 80);
      }
      goto LABEL_20;
    }
    goto LABEL_19;
  }
LABEL_71:
  v25 = a1[1].m128i_i64[1];
  if ( !v25 )
    goto LABEL_44;
  if ( (a1[1].m128i_i8[2] & 2) == 0 )
    goto LABEL_159;
LABEL_35:
  if ( (a1[1].m128i_i8[0] & 4) == 0 )
  {
LABEL_271:
    v25 = 0;
    goto LABEL_44;
  }
LABEL_36:
  if ( v203 )
    goto LABEL_271;
  if ( dword_4F077BC && (!(_DWORD)qword_4F077B4 || qword_4F077A0 <= 0x15F8Fu) )
    BYTE14(v239) = 1;
  v16 = v6;
  sub_648CF0(&v232, v6);
  v15 = v233;
  if ( !(_QWORD)v233 )
  {
    v25 = *((_QWORD *)&v233 + 1);
    goto LABEL_44;
  }
  if ( *(_BYTE *)(v233 + 80) != 20 )
  {
    sub_6854C0(147, &a1->m128i_u64[1], v233);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v23 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v24 = *(_QWORD *)dword_4F07508;
    a1[1].m128i_i8[1] |= 0x20u;
    v25 = *((_QWORD *)&v233 + 1);
    a1[3] = v23;
    a1->m128i_i64[1] = v24;
LABEL_44:
    v26 = qword_4F04C68[0] + 776LL * (int)v235;
    if ( (*(_BYTE *)(v26 + 6) & 2) != 0 )
    {
LABEL_45:
      if ( (a1[1].m128i_i8[1] & 0x20) == 0 )
      {
        if ( !v187 )
        {
          v184 = v25;
          v214 = v26;
          sub_646070(v218, 0, a1);
          v25 = v184;
          v26 = v214;
        }
        v179 = v25;
        v205 = v26;
        sub_646240((__int64)a1, v6);
        v25 = v179;
        v26 = v205;
      }
      for ( i = *(_QWORD *)(v6 + 288); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( !*(_QWORD *)(v6 + 456) )
      {
        v183 = v25;
        v212 = v26;
        sub_648BD0(**(__int64 ****)(i + 168), (__int64)dword_4F07508);
        v25 = v183;
        v26 = v212;
      }
      if ( v25 )
      {
        v180 = v26;
        v206 = v25;
        v28 = sub_8E0850(v25, v218, v6, &v229);
        v25 = v206;
        v26 = v180;
        if ( !v28 )
        {
          sub_6851C0((unsigned int)v229, &a1->m128i_u64[1]);
          v25 = 0;
          *a1 = _mm_loadu_si128(xmmword_4F06660);
          a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
          a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
          v29 = _mm_loadu_si128(&xmmword_4F06660[3]);
          v30 = *(_QWORD *)dword_4F07508;
          a1[1].m128i_i8[1] |= 0x20u;
          v26 = v180;
          a1[3] = v29;
          a1->m128i_i64[1] = v30;
        }
      }
      v31 = v187 | v195;
      if ( v187 | v195 || v203 && (a1[1].m128i_i8[0] & 1) != 0 )
      {
        v16 = a1->m128i_i64[0];
        v207 = v26;
        v32 = sub_87EBB0(20, a1->m128i_i64[0]);
        v33 = v207;
        v34 = a1[2].m128i_i64[0];
        v15 = v32;
        if ( (a1[1].m128i_i8[2] & 2) != 0 )
        {
          v16 = 0;
          sub_877E20(v32, 0, v34);
          v33 = v207;
        }
        else if ( v34 )
        {
          v16 = 0;
          sub_877E90(v32, 0);
          v33 = v207;
        }
        v31 = 1;
        *(_BYTE *)(v15 + 81) = a1[1].m128i_i8[1] & 0x20 | *(_BYTE *)(v15 + 81) & 0xDF;
      }
      else if ( v25 )
      {
        v16 = 1;
        v210 = v26;
        v149 = sub_641B60((__int64)a1, 1, v25, (__int64)&v219, v12, v238 & 1);
        v33 = v210;
        v15 = v149;
      }
      else
      {
        v161 = 0;
        if ( (v238 & 1) != 0 )
          v161 = BYTE14(v239) != 0;
        v16 = (__int64)a1;
        v211 = v26;
        v162 = sub_647630(0x14u, (__int64)a1, (unsigned int)v235, v161);
        v33 = v211;
        v15 = v162;
        if ( v12 )
          *(_BYTE *)(v162 + 83) |= 0x40u;
        else
          v31 = 0;
      }
      switch ( *(_BYTE *)(v15 + 80) )
      {
        case 4:
        case 5:
          v204 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 80LL);
          break;
        case 6:
          v204 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v204 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v204 = *(_QWORD *)(v15 + 88);
          break;
        default:
          MEMORY[0xA0] &= ~8u;
          BUG();
      }
      v150 = (8 * (*(_DWORD *)(a4 + 84) & 1)) | *(_BYTE *)(v204 + 160) & 0xF7;
      *(_BYTE *)(v204 + 160) = v150;
      v143 = *(_QWORD *)(v204 + 328) == 0;
      *(_BYTE *)(v204 + 160) = (16 * (*(_BYTE *)(a4 + 88) & 1)) | v150 & 0xEF;
      if ( v143 )
      {
        v16 = 0;
        v213 = v33;
        sub_879080(v204 + 296, 0, v198);
        v33 = v213;
      }
      if ( !v31 )
      {
        if ( (v238 & 1) != 0 && (*(_BYTE *)(v33 + 6) & 2) != 0 )
          v33 = qword_4F04C68[0] + 776LL * dword_4F04C34;
        if ( (unsigned __int8)(*(_BYTE *)(v33 + 4) - 3) <= 1u )
        {
          v16 = 0;
          sub_877E90(v15, 0);
        }
      }
      *(_QWORD *)v6 = v15;
      *(_BYTE *)(v15 + 84) = (16 * ((*(_BYTE *)(v15 + 84) & 0x10) != 0 || unk_4D03A10 != 0))
                           | *(_BYTE *)(v15 + 84) & 0xEF;
      v209 = &a1->m128i_i8[8];
      goto LABEL_186;
    }
    sub_644100((__int64)&v232, v6);
    v101 = v199;
    v25 = *((_QWORD *)&v233 + 1);
    v15 = v233;
    if ( BYTE4(v235) )
      v101 = BYTE4(v235);
    v199 = v101;
    v219 = *((_QWORD *)&v234 + 1);
    if ( !(_QWORD)v233 )
    {
      if ( !*((_QWORD *)&v233 + 1) )
        goto LABEL_423;
      v152 = *(unsigned __int8 *)(*((_QWORD *)&v233 + 1) + 80LL);
      v153 = *((_QWORD *)&v233 + 1);
      if ( (_BYTE)v152 == 16 )
      {
        v153 = **(_QWORD **)(*((_QWORD *)&v233 + 1) + 88LL);
        v152 = *(unsigned __int8 *)(v153 + 80);
      }
      if ( (_BYTE)v152 == 24 )
        v152 = *(unsigned __int8 *)(*(_QWORD *)(v153 + 88) + 80LL);
      if ( (unsigned __int8)v152 > 0x14u || (v157 = 1182720, !_bittest64(&v157, v152)) )
      {
        sub_6854C0(147, &a1->m128i_u64[1], *((_QWORD *)&v233 + 1));
        v25 = 0;
        *a1 = _mm_loadu_si128(xmmword_4F06660);
        a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
        a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
        v154 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v155 = *(_QWORD *)dword_4F07508;
        a1[1].m128i_i8[1] |= 0x20u;
        v156 = (int)v235;
        a1[3] = v154;
        a1->m128i_i64[1] = v155;
        v26 = qword_4F04C68[0] + 776 * v156;
      }
      else
      {
LABEL_423:
        v26 = qword_4F04C68[0] + 776LL * (int)v235;
      }
      goto LABEL_45;
    }
    v181 = *((_QWORD *)&v233 + 1);
    v209 = &a1->m128i_i8[8];
    if ( *(_BYTE *)(v233 + 80) != 20 )
    {
      sub_6854C0(147, &a1->m128i_u64[1], v233);
      *a1 = _mm_loadu_si128(xmmword_4F06660);
      a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v145 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v146 = *(_QWORD *)dword_4F07508;
      a1[1].m128i_i8[1] |= 0x20u;
      v147 = (int)v235;
      v25 = v181;
      a1[3] = v145;
      a1->m128i_i64[1] = v146;
      v26 = qword_4F04C68[0] + 776 * v147;
      goto LABEL_45;
    }
    v204 = *(_QWORD *)(v233 + 88);
    v43 = *(_QWORD *)(v204 + 176);
    *(_QWORD *)(v204 + 376) = 0;
    v102 = *(__int64 ***)(v218 + 168);
    while ( 1 )
    {
      v102 = (__int64 **)*v102;
      if ( !v102 )
        break;
      if ( ((_BYTE)v102[4] & 4) != 0 )
      {
        if ( *(_QWORD *)(v204 + 168) )
        {
          if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
          {
            v139 = 8;
            v138 = 803;
          }
          else
          {
            v138 = 803;
            v139 = qword_4F077A8 < 0xC350u ? 5 : 8;
          }
        }
        else if ( dword_4D04964 )
        {
          v138 = 802;
          v139 = byte_4F07472[0];
        }
        else
        {
          v139 = 5;
          v138 = 802;
        }
        sub_684AA0(v139, v138, v209);
        break;
      }
    }
    if ( v43 && (*(_BYTE *)(v43 + 198) & 0x20) != 0 )
    {
      for ( j = *(_QWORD *)(v43 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v100 = **(_QWORD **)(v218 + 168);
      if ( v100 )
      {
        if ( **(_QWORD **)(j + 168) )
        {
          v197 = v43;
          v104 = **(__int64 ****)(j + 168);
          v182 = v6;
          v105 = **(__int64 ****)(v218 + 168);
          while ( 1 )
          {
            v97 = (_BYTE)v105[4] & 2;
            if ( ((_BYTE)v104[4] & 2) != 0 )
            {
              if ( !(_BYTE)v97 )
              {
LABEL_291:
                v106 = (const char *)v105[3];
                if ( !v106 )
                  v106 = byte_3F871B3;
                sub_6861C0(7, 3699, v209, v15 + 48, v106);
              }
            }
            else if ( (_BYTE)v97 )
            {
              goto LABEL_291;
            }
            v104 = (__int64 **)*v104;
            v105 = (__int64 **)*v105;
            if ( !v104 || !v105 )
            {
              v43 = v197;
              v6 = v182;
              break;
            }
          }
        }
      }
    }
    sub_8958E0(a4, v15, v97, v98, v99, v100);
    sub_6464A0(v218, v15, (unsigned int *)(a2 + 24), 1u);
    sub_649200(v43, v218, 1, 0, v6);
    v16 = a4;
    sub_89BD20(**(_QWORD **)(v16 + 192), v16, v15, (_DWORD)v209, 1, 0, 1, 7);
    v107 = v173;
    if ( (*(_BYTE *)(v6 + 122) & 1) != 0 )
    {
      for ( k = *(_QWORD *)(v43 + 152); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      v109 = **(__int64 ***)(k + 168);
      if ( v109 )
      {
        v110 = 0;
        v178 = v43;
        v111 = **(__int64 ***)(v218 + 168);
        v177 = v6;
        v112 = 0;
        v176 = v15;
        v175 = a1;
        while ( 1 )
        {
          ++v112;
          v16 = dword_4D046F8;
          v113 = (*((_DWORD *)v111 + 8) >> 11) & 0x7F;
          if ( dword_4D046F8 )
          {
            if ( (((unsigned __int8)v113 ^ (unsigned __int8)(*((_DWORD *)v109 + 8) >> 11)) & 3) != 0 )
              break;
          }
LABEL_305:
          *((_DWORD *)v109 + 8) = v109[4] & 0xFFFC07FF | ((v113 & 0x7F) << 11);
          v109 = (__int64 *)*v109;
          v111 = (__int64 *)*v111;
          if ( !v109 )
          {
            v43 = v178;
            v6 = v177;
            v15 = v176;
            a1 = v175;
            goto LABEL_350;
          }
        }
        if ( v112 > 9 )
        {
          if ( v112 <= 0x63 )
          {
            dest = v222;
            sub_2240A50(&dest, 2, 0, v110, v107);
            v118 = dest;
            v119 = v112;
          }
          else
          {
            if ( v112 <= 0x3E7 )
            {
              v117 = 3;
            }
            else
            {
              v114 = v112;
              if ( v112 <= 0x270F )
              {
                v117 = 4;
              }
              else
              {
                LODWORD(v110) = 1;
                do
                {
                  v115 = v114;
                  v116 = v110;
                  v110 = (unsigned int)(v110 + 4);
                  v114 /= 0x2710u;
                  if ( v115 <= 0x1869F )
                  {
                    v117 = (unsigned int)v110;
                    goto LABEL_318;
                  }
                  if ( (unsigned int)v114 <= 0x63 )
                  {
                    v117 = (unsigned int)(v116 + 5);
                    dest = v222;
                    goto LABEL_319;
                  }
                  if ( (unsigned int)v114 <= 0x3E7 )
                  {
                    v117 = (unsigned int)(v116 + 6);
                    goto LABEL_318;
                  }
                }
                while ( (unsigned int)v114 > 0x270F );
                v117 = (unsigned int)(v116 + 7);
              }
            }
LABEL_318:
            dest = v222;
LABEL_319:
            sub_2240A50(&dest, v117, 0, v110, v107);
            v118 = dest;
            v119 = v112;
            LODWORD(v120) = n - 1;
            do
            {
              v121 = v119 % 0x64;
              v122 = v119;
              v119 /= 0x64u;
              v123 = 2 * v121;
              v124 = (unsigned int)(v123 + 1);
              LOBYTE(v123) = a00010203040506[v123];
              v118[(unsigned int)v120] = a00010203040506[v124];
              v125 = (unsigned int)(v120 - 1);
              v120 = (unsigned int)(v120 - 2);
              v118[v125] = v123;
            }
            while ( v122 > 0x270F );
            if ( v122 <= 0x3E7 )
              goto LABEL_322;
          }
          v151 = 2 * v119;
          v118[1] = a00010203040506[(unsigned int)(v151 + 1)];
          *v118 = a00010203040506[v151];
LABEL_323:
          v126 = (const char *)v111[3];
          if ( !v126 )
          {
LABEL_346:
            v16 = 3744;
            sub_6861C0(5, 3744, v111[9] + 16, v109[9] + 16, dest);
            if ( dest != v222 )
            {
              v16 = v222[0] + 1LL;
              j_j___libc_free_0(dest, v222[0] + 1LL);
            }
            v113 = (*((_DWORD *)v111 + 8) >> 11) & 0x7F;
            goto LABEL_305;
          }
          v127 = dest;
          v128 = n;
          v223 = v225;
          if ( (char *)dest + n && !dest )
            sub_426248((__int64)"basic_string::_M_construct null not valid");
          v229 = n;
          if ( n > 0xF )
          {
            src = dest;
            v158 = sub_22409D0(&v223, &v229, 0);
            v127 = src;
            v223 = (_QWORD *)v158;
            v159 = (_QWORD *)v158;
            v225[0] = v229;
          }
          else
          {
            if ( n == 1 )
            {
              LOBYTE(v225[0]) = *(_BYTE *)dest;
              v129 = v225;
              goto LABEL_329;
            }
            if ( !n )
            {
              v129 = v225;
              goto LABEL_329;
            }
            v159 = v225;
          }
          memcpy(v159, v127, v128);
          v128 = v229;
          v129 = v223;
LABEL_329:
          v224 = v128;
          *((_BYTE *)v129 + v128) = 0;
          if ( 0x3FFFFFFFFFFFFFFFLL - v224 <= 2 )
            goto LABEL_485;
          sub_2241490(&v223, " (\"", 3, v120);
          v130 = strlen(v126);
          if ( v130 > 0x3FFFFFFFFFFFFFFFLL - v224 )
            goto LABEL_485;
          v132 = (__m128i *)sub_2241490(&v223, v126, v130, v131);
          v226 = &v228;
          if ( (__m128i *)v132->m128i_i64[0] == &v132[1] )
          {
            v228 = _mm_loadu_si128(v132 + 1);
          }
          else
          {
            v226 = (__m128i *)v132->m128i_i64[0];
            v228.m128i_i64[0] = v132[1].m128i_i64[0];
          }
          v133 = v132->m128i_i64[1];
          v227 = v133;
          v132->m128i_i64[0] = (__int64)v132[1].m128i_i64;
          v132->m128i_i64[1] = 0;
          v132[1].m128i_i8[0] = 0;
          if ( v227 == 0x3FFFFFFFFFFFFFFFLL || v227 == 4611686018427387902LL )
LABEL_485:
            sub_4262D8((__int64)"basic_string::append");
          v134 = (__m128i *)sub_2241490(&v226, "\")", 2, v133);
          v229 = (size_t)&v231;
          if ( (__m128i *)v134->m128i_i64[0] == &v134[1] )
          {
            v231 = _mm_loadu_si128(v134 + 1);
          }
          else
          {
            v229 = v134->m128i_i64[0];
            v231.m128i_i64[0] = v134[1].m128i_i64[0];
          }
          v230 = v134->m128i_u64[1];
          v134->m128i_i64[0] = (__int64)v134[1].m128i_i64;
          v135 = (__m128i *)dest;
          v134->m128i_i64[1] = 0;
          v134[1].m128i_i8[0] = 0;
          if ( (__m128i *)v229 == &v231 )
          {
            v160 = v230;
            if ( v230 )
            {
              if ( v230 == 1 )
                v135->m128i_i8[0] = v231.m128i_i8[0];
              else
                memcpy(v135, &v231, v230);
              v160 = v230;
              v135 = (__m128i *)dest;
            }
            n = v160;
            v135->m128i_i8[v160] = 0;
            v135 = (__m128i *)v229;
            goto LABEL_340;
          }
          if ( v135 == (__m128i *)v222 )
          {
            dest = (void *)v229;
            n = v230;
            v222[0] = v231.m128i_i64[0];
          }
          else
          {
            v136 = v222[0];
            dest = (void *)v229;
            n = v230;
            v222[0] = v231.m128i_i64[0];
            if ( v135 )
            {
              v229 = (size_t)v135;
              v231.m128i_i64[0] = v136;
LABEL_340:
              v230 = 0;
              v135->m128i_i8[0] = 0;
              if ( (__m128i *)v229 != &v231 )
                j_j___libc_free_0(v229, v231.m128i_i64[0] + 1);
              if ( v226 != &v228 )
                j_j___libc_free_0(v226, v228.m128i_i64[0] + 1);
              if ( v223 != v225 )
                j_j___libc_free_0(v223, v225[0] + 1LL);
              goto LABEL_346;
            }
          }
          v229 = (size_t)&v231;
          v135 = &v231;
          goto LABEL_340;
        }
        dest = v222;
        sub_2240A50(&dest, 1, 0, v110, v107);
        v118 = dest;
        LOBYTE(v119) = v112;
LABEL_322:
        *v118 = v119 + 48;
        goto LABEL_323;
      }
    }
LABEL_350:
    v137 = *(_BYTE *)(v15 + 83);
    if ( (v137 & 0x40) != 0 && (v238 & 1) == 0 )
    {
      *(_BYTE *)(v15 + 83) = v137 & 0xBF;
      if ( (v137 & 0x20) != 0 )
        *(_BYTE *)(v219 + 83) &= ~0x40u;
    }
    goto LABEL_185;
  }
LABEL_79:
  v204 = *(_QWORD *)(v15 + 88);
LABEL_80:
  v43 = *(_QWORD *)(v204 + 176);
  v209 = &a1->m128i_i8[8];
  if ( (*(_BYTE *)(a2 + 64) & 4) != 0 )
  {
    if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
    {
      if ( (a1[1].m128i_i8[2] & 2) == 0 )
        goto LABEL_185;
    }
    else if ( !*(_QWORD *)(v15 + 64) || (a1[1].m128i_i8[2] & 2) != 0 )
    {
      goto LABEL_185;
    }
    if ( a1[2].m128i_i64[0] )
    {
      *(_BYTE *)(v43 + 203) |= 1u;
      *(_QWORD *)v6 = v15;
      *(_BYTE *)(v15 + 84) = (16 * ((*(_BYTE *)(v15 + 84) & 0x10) != 0 || unk_4D03A10 != 0))
                           | *(_BYTE *)(v15 + 84) & 0xEF;
      goto LABEL_85;
    }
  }
LABEL_185:
  *(_QWORD *)v6 = v15;
  *(_BYTE *)(v15 + 84) = (16 * ((*(_BYTE *)(v15 + 84) & 0x10) != 0 || unk_4D03A10 != 0)) | *(_BYTE *)(v15 + 84) & 0xEF;
  if ( !v43 )
  {
LABEL_186:
    *(_BYTE *)(v6 + 127) |= 0x10u;
    sub_7296C0(&v223);
    v43 = sub_725FD0(&v223, v16, v79);
    *(_QWORD *)(v204 + 176) = v43;
    sub_729730((unsigned int)v223);
    *(_QWORD *)(v43 + 152) = v218;
    *(_BYTE *)(v43 + 172) = v199;
    if ( (*(_BYTE *)(a2 + 64) & 4) != 0 )
      *(_BYTE *)(v43 + 173) = *(_BYTE *)(v6 + 268);
    *(_BYTE *)(v43 + 207) = (2 * *(_BYTE *)(v6 + 125)) & 0x10 | *(_BYTE *)(v43 + 207) & 0xEF;
    if ( (*(_BYTE *)(a2 + 64) & 2) != 0 )
      sub_736C90(v43, 1);
    v80 = *(_QWORD *)(v6 + 8);
    if ( (v80 & 0x180000) != 0 )
    {
      v143 = (v80 & 0x100000) == 0;
      v81 = *(_BYTE *)(v43 + 193);
      v82 = v143 ? v81 | 1 : v81 | 4;
      *(_BYTE *)(v43 + 193) = v82;
      v83 = *(char *)(v43 + 192) < 0;
      *(_BYTE *)(v43 + 193) = v82 | 2;
      if ( !v83 )
        sub_736C90(v43, 1);
    }
    v84 = a1[1].m128i_i8[0];
    if ( (v84 & 8) != 0 )
    {
      sub_725ED0(v43, 5);
      *(_BYTE *)(v43 + 176) = a1[3].m128i_i8[8];
    }
    else if ( (v84 & 0x40) != 0 )
    {
      sub_725ED0(v43, 4);
    }
    else if ( (v84 & 0x20) != 0 )
    {
      if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
        sub_725ED0(v43, 2);
    }
    else if ( ((*(_BYTE *)(v6 + 9) & 4) != 0 || (*(_BYTE *)(v6 + 16) & 0x10) != 0) && (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
    {
      sub_725ED0(v43, 1);
    }
    v140 = sub_880CF0(v15, v43, *v198);
    sub_877D80(v43, v140);
    sub_877F10(v43, v140);
    sub_897580(a4, v15, v204);
    *(_BYTE *)(v43 + 88) = (16 * ((v199 == 1) + 1)) | *(_BYTE *)(v43 + 88) & 0x8F;
    v141 = *(_QWORD *)(v6 + 400);
    if ( v141 )
    {
      *(_QWORD *)(v43 + 216) = v141;
      *(_QWORD *)(v6 + 400) = 0;
    }
    sub_8CCE20(v140, v204);
    v196 = dword_4F07590;
    if ( dword_4F07590 || (v200 = 0, *(char *)(v204 + 160) < 0) )
    {
      v200 = 0;
      v196 = 0;
      if ( (a1[1].m128i_i8[1] & 0x20) == 0 )
        sub_7362F0(v43, (unsigned int)(v187 - 1));
    }
    goto LABEL_103;
  }
LABEL_85:
  *(_QWORD *)(a4 + 148) = *(_QWORD *)(v15 + 48);
  v44 = *(_QWORD *)(v6 + 8);
  v45 = *(_BYTE *)(v43 + 193);
  if ( (v45 & 1) != ((v44 & 0x80000) != 0) || ((v45 & 4) != 0) != ((v44 & 0x100000) != 0) )
  {
    v46 = 2930;
    if ( (v45 & 4) == 0 )
    {
      v46 = 2383;
      if ( (*(_BYTE *)(v43 + 193) & 2) == 0 )
        v46 = (v44 & 0x100000) == 0 ? 2384 : 2931;
    }
    if ( !sub_736C60(99, *(_QWORD *)(v6 + 184))
      && !sub_736C60(99, *(_QWORD *)(v6 + 208))
      && !sub_736C60(99, *(_QWORD *)(v6 + 200)) )
    {
      sub_6854C0(v46, v6 + 32, v15);
    }
    v47 = *(_BYTE *)(v43 + 193);
    if ( (v47 & 4) == 0 && (*(_BYTE *)(v6 + 10) & 0x10) == 0 )
      *(_BYTE *)(v43 + 193) = v47 | 1;
    if ( (*(_BYTE *)(v15 + 81) & 2) == 0 )
      *(_BYTE *)(v43 + 193) |= 2u;
  }
  v196 = 0;
  if ( *(char *)(v43 + 192) >= 0 && ((*(_BYTE *)(a2 + 64) & 2) != 0 || (*(_BYTE *)(v43 + 193) & 1) != 0) )
  {
    sub_736C90(v43, 1);
    v196 = 1;
  }
  if ( *(_DWORD *)(a4 + 24) )
  {
    v48 = *(_QWORD *)(v6 + 400);
    if ( v48 )
    {
      *(_QWORD *)(v43 + 216) = v48;
      *(_QWORD *)(v6 + 400) = 0;
    }
  }
  v200 = 1;
  if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
    goto LABEL_103;
  v85 = *(__int64 **)(v6 + 184);
  v86 = *(__int64 **)(v43 + 104);
  if ( v85 )
  {
    v87 = 0;
    do
    {
      while ( 1 )
      {
        v88 = *((_BYTE *)v85 + 8);
        if ( ((unsigned __int8)(v88 - 86) > 0x1Cu || ((1LL << (v88 - 86)) & 0x107BFFFF) == 0)
          && (*((_BYTE *)v85 + 9) == 2 || (*((_BYTE *)v85 + 11) & 0x10) != 0) )
        {
          break;
        }
        v85 = (__int64 *)*v85;
        if ( !v85 )
          goto LABEL_232;
      }
      if ( !v87 && v88 > 1u )
      {
        v201 = v86;
        v92 = sub_736C60(v88, v86);
        v87 = 0;
        v86 = v201;
        if ( !v92 )
          v87 = v85;
      }
      *((_BYTE *)v85 + 8) = 0;
      v85 = (__int64 *)*v85;
    }
    while ( v85 );
LABEL_232:
    v89 = *(__int64 **)(v6 + 200);
    if ( !v89 )
      goto LABEL_239;
  }
  else
  {
    v89 = *(__int64 **)(v6 + 200);
    if ( !v89 )
      goto LABEL_103;
    v87 = 0;
  }
  do
  {
    if ( *((_BYTE *)v89 + 9) == 2 || (*((_BYTE *)v89 + 11) & 0x10) != 0 )
    {
      v90 = *((unsigned __int8 *)v89 + 8);
      if ( (unsigned __int8)v90 > 1u )
      {
        if ( !v87 )
        {
          v202 = v86;
          v96 = sub_736C60(v90, v86);
          v87 = 0;
          v86 = v202;
          if ( !v96 )
            v87 = v89;
        }
        *((_BYTE *)v89 + 8) = 0;
      }
    }
    v89 = (__int64 *)*v89;
  }
  while ( v89 );
LABEL_239:
  v200 = 1;
  if ( v87 )
  {
    while ( 1 )
    {
      if ( !v86 )
      {
        sub_684B30(2189, v87 + 7);
        v200 = 1;
        goto LABEL_103;
      }
      if ( *((_DWORD *)v86 + 14) )
        break;
      v86 = (__int64 *)*v86;
    }
    sub_6854F0(5, 2190, v87 + 7, v86 + 7);
    v200 = 1;
  }
LABEL_103:
  v49 = *(_BYTE *)(a2 + 64);
  if ( *(_DWORD *)(a4 + 16) && (v49 & 4) != 0 )
  {
    *(_BYTE *)(v43 + 202) |= 0x80u;
    v49 = *(_BYTE *)(a2 + 64);
  }
  sub_644920((_QWORD *)v6, (v49 & 4) != 0);
  if ( *(_BYTE *)(v43 + 172) != 2 )
  {
    LOBYTE(v229) = *(_BYTE *)(v43 + 200) & 7;
    sub_5D0D60(&v229, 0);
    *(_BYTE *)(v43 + 200) = v229 & 7 | *(_BYTE *)(v43 + 200) & 0xF8;
  }
  sub_5F06F0((_BYTE *)v6, a2, v209, v50, v51);
  v52 = a1[2].m128i_i64[1];
  if ( !v52 || (a1[1].m128i_i8[2] & 1) != 0 )
    goto LABEL_114;
  if ( !v200 && !*(_DWORD *)(a4 + 16) )
  {
    a1[2].m128i_i64[1] = 0;
    if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
      goto LABEL_120;
    goto LABEL_113;
  }
  v229 = 0;
  v91 = sub_8B1C20(v15, v52, &v229, **(_QWORD **)(v204 + 328), 0);
  sub_725130(v229);
  a1[2].m128i_i64[1] = 0;
  if ( (a1[1].m128i_i8[1] & 0x20) == 0 )
  {
    if ( !v91 )
    {
LABEL_113:
      sub_6851C0(891, v209);
      *a1 = _mm_loadu_si128(xmmword_4F06660);
      a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v53 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v54 = *(_QWORD *)dword_4F07508;
      a1[1].m128i_i8[1] |= 0x20u;
      a1[3] = v53;
      a1->m128i_i64[1] = v54;
      goto LABEL_114;
    }
    sub_684B30(1762, v209);
LABEL_114:
    if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
      goto LABEL_120;
    v55 = *(_BYTE *)(a2 + 64);
    v56 = *(_BYTE *)(v15 + 81);
    if ( (v55 & 4) != 0 )
    {
      if ( (v56 & 2) == 0 )
      {
LABEL_119:
        sub_64A300(v43, *(_QWORD *)(a2 + 80));
        goto LABEL_120;
      }
      if ( (*(_BYTE *)(v15 + 84) & 0x10) != 0 )
      {
        ++unk_4F07488;
        goto LABEL_119;
      }
      sub_685920(v209, v15, 8);
    }
    else
    {
      if ( (v56 & 0x10) == 0 || v193 || v55 & 0x18 | v238 & 1 )
        goto LABEL_207;
      sub_6854C0(392, v209, v15);
    }
    if ( (*(_BYTE *)(a2 + 64) & 4) != 0 )
      goto LABEL_119;
LABEL_207:
    if ( (*(_BYTE *)(v6 + 127) & 0x10) == 0 )
      goto LABEL_122;
    goto LABEL_121;
  }
LABEL_120:
  if ( (*(_BYTE *)(a2 + 64) & 4) == 0 )
    goto LABEL_207;
LABEL_121:
  sub_729470(v43, a4 + 344);
LABEL_122:
  v57 = v6 + 224;
  sub_648B00(v43, (_BYTE *)(v6 + 224), (__int64)v209);
  if ( v219 )
  {
    v58 = &dword_4D04340;
    if ( dword_4D04340 )
    {
      for ( m = *(_QWORD *)(v219 + 88); m; m = *(_QWORD *)(m + 8) )
      {
        while ( *(_BYTE *)(m + 80) != 11 )
        {
          m = *(_QWORD *)(m + 8);
          if ( !m )
            goto LABEL_129;
        }
        v57 = m;
        sub_8BFCA0(v15, m, *v198, v191);
      }
    }
  }
LABEL_129:
  if ( unk_4D04534
    || !*(_QWORD *)(v43 + 152)
    || (v58 = &qword_4F60228, *(_QWORD *)v15 != qword_4F60228)
    && (v58 = &qword_4F60220, *(_QWORD *)v15 != qword_4F60220) )
  {
LABEL_133:
    if ( !v196 )
      goto LABEL_145;
    v63 = *(_QWORD **)(v204 + 168);
    if ( !v63 )
      goto LABEL_145;
    goto LABEL_135;
  }
  if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
  {
    v58 = (_DWORD *)v196;
    if ( !v196 )
      goto LABEL_165;
    v63 = *(_QWORD **)(v204 + 168);
    if ( !v63 )
      goto LABEL_165;
    goto LABEL_135;
  }
  v93 = *(_QWORD *)(v15 + 64);
  if ( v93 && unk_4D049B8 )
  {
    if ( (unsigned int)sub_880920(v15, v57, v58) )
      v93 = *(_QWORD *)(*(_QWORD *)v93 + 64LL);
    for ( ii = *(_QWORD *)(v43 + 152); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
      ;
    v95 = **(_QWORD ***)(ii + 168);
    if ( v95 && !*v95 && *(_QWORD *)(unk_4D049B8 + 88LL) == v93 )
      *(_DWORD *)(v43 + 196) |= 0x181200u;
    goto LABEL_133;
  }
  if ( v196 )
  {
    v63 = *(_QWORD **)(v204 + 168);
    if ( v63 )
    {
      do
      {
LABEL_135:
        v64 = v63[3];
        v65 = *(_QWORD *)(v64 + 88);
        if ( (*(_BYTE *)(v65 + 195) & 2) == 0 )
        {
          if ( !dword_4D04824 && *(_BYTE *)(v65 + 172) != 2 )
          {
            if ( (*(_BYTE *)(v15 + 81) & 0x10) == 0 )
              sub_6854B0(553, v64);
            v66 = *(_BYTE *)(v65 + 88);
            *(_BYTE *)(v65 + 172) = 2;
            *(_BYTE *)(v65 + 88) = v66 & 0x8F | 0x10;
          }
          if ( (*(_WORD *)(v65 + 192) & 0x4080) == 0x4000 )
            sub_685480(479, v63[3]);
          sub_736C90(v65, 1);
        }
        v63 = (_QWORD *)*v63;
      }
      while ( v63 );
LABEL_145:
      if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 )
        goto LABEL_165;
    }
  }
  v61 = (unsigned int)dword_4D04340;
  if ( dword_4D04340 )
  {
    for ( jj = *(_QWORD *)(*(_QWORD *)v15 + 40LL); jj; jj = *(_QWORD *)(jj + 8) )
    {
      if ( *(_BYTE *)(jj + 80) == 15
        && *(_QWORD *)(jj + 64) == *(_QWORD *)(v15 + 64)
        && *(_DWORD *)(jj + 40) == unk_4F066A8 )
      {
        v68 = *(__int64 **)(*(_QWORD *)(jj + 88) + 8LL);
        if ( (v68[11] & 4) != 0 )
        {
          v69 = *v68;
          if ( !*(_QWORD *)(*v68 + 96) )
          {
            for ( kk = v68[19]; *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
              ;
            v71 = sub_8B7B10(v15, kk, (unsigned int)&v226, (unsigned int)&v229, *v198, 0, 1, 0);
            v60 = v172;
            if ( v71 )
              sub_6854E0(750, v69);
          }
        }
      }
    }
  }
LABEL_165:
  if ( !v200 )
  {
    v72 = *(_QWORD *)v15;
    v73 = *(_BYTE **)(*(_QWORD *)v15 + 8LL);
    if ( v73 )
    {
      if ( (*(_BYTE *)(v15 + 81) & 0x10) == 0 && !*(_QWORD *)(v15 + 64) )
      {
        v142 = 0;
        v143 = (*(_BYTE *)(v72 + 73) & 2) == 0;
        if ( (*(_BYTE *)(v72 + 73) & 2) != 0 )
        {
          v59 = 5;
          v144 = "main";
          do
          {
            if ( !v59 )
              break;
            v142 = *v73 < (unsigned __int8)*v144;
            v143 = *v73++ == (unsigned __int8)*v144++;
            --v59;
          }
          while ( v143 );
          if ( (!v142 && !v143) == v142 )
            sub_6851C0(466, v209);
        }
      }
    }
  }
  if ( !unk_4D047D8 )
  {
LABEL_210:
    if ( (v238 & 0x200) == 0 )
      goto LABEL_174;
    if ( (v238 & 1) != 0 )
    {
LABEL_173:
      sub_8645D0();
      goto LABEL_174;
    }
    goto LABEL_212;
  }
  if ( (v238 & 1) != 0 )
  {
    if ( (*(_BYTE *)(v15 + 83) & 0x40) == 0 )
    {
      if ( (v238 & 0x200) == 0 )
        goto LABEL_174;
      goto LABEL_173;
    }
    sub_87FDB0(v15, *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * v194 + 184) + 32LL), v58, v59, v60, v61);
    goto LABEL_210;
  }
  if ( (v238 & 0x200) != 0 )
LABEL_212:
    sub_8642D0();
LABEL_174:
  for ( mm = *(_QWORD **)(v204 + 168); mm; mm = (_QWORD *)*mm )
  {
    v75 = *(_QWORD *)(mm[3] + 88LL);
    if ( (*(_BYTE *)(v75 + 195) & 2) == 0 )
    {
      v76 = *(_BYTE *)(v43 + 198) & 8 | *(_BYTE *)(v75 + 198) & 0xF7;
      *(_BYTE *)(v75 + 198) = v76;
      v77 = *(_BYTE *)(v43 + 198) & 0x10 | v76 & 0xEF;
      *(_BYTE *)(v75 + 198) = v77;
      *(_BYTE *)(v75 + 198) = *(_BYTE *)(v43 + 198) & 0x20 | v77 & 0xDF;
    }
  }
  if ( *(char *)(v43 + 90) >= 0 )
    sub_8D9350(v218, v209);
  if ( *(_BYTE *)(v43 + 174) == 4 )
    sub_642470(*(_QWORD *)(v15 + 88), (__int64)v209);
  *a3 = v15;
  return sub_826060(v43, v209);
}
