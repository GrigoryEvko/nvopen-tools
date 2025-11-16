// Function: sub_8A6CC0
// Address: 0x8a6cc0
//
unsigned __int64 *__fastcall sub_8A6CC0(__m128i *a1, unsigned __int64 *a2, _DWORD *a3, int a4)
{
  __m128i *v4; // r14
  _QWORD **v5; // r12
  __int64 *v6; // rbx
  __int64 v7; // rax
  unsigned int *v8; // rsi
  unsigned __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int16 v12; // ax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // r13d
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  char v23; // al
  __int64 v24; // rsi
  unsigned __int64 v25; // r15
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __int64 v33; // rax
  __m128i v34; // xmm3
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __m128i v43; // xmm5
  __m128i v44; // xmm6
  __m128i v45; // xmm7
  unsigned __int16 v46; // ax
  int v47; // ecx
  unsigned int v48; // r12d
  __int64 v49; // rax
  int v50; // ecx
  char v51; // dl
  __int32 v52; // r9d
  char v53; // al
  unsigned __int8 v54; // di
  char v55; // dl
  __int64 v56; // rax
  __int8 v57; // dl
  __int64 v58; // rax
  __m128i *v59; // r12
  int v60; // esi
  __int32 v61; // esi
  bool v62; // dl
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  int v67; // ebx
  __int8 v68; // dl
  __int8 v69; // al
  bool v70; // zf
  char v71; // al
  __int64 v72; // rsi
  unsigned int v73; // ebx
  __int64 v74; // rax
  __int64 v75; // r13
  char v77; // al
  __int64 v78; // rax
  char v79; // dl
  int v80; // eax
  char v81; // cl
  __m128i v82; // xmm5
  __m128i v83; // xmm6
  __m128i v84; // xmm7
  _QWORD **v85; // r13
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  char v94; // al
  char v95; // al
  __int64 v96; // rsi
  __int64 v97; // rax
  _BOOL4 v98; // ebx
  __int64 v99; // rcx
  __int64 v100; // rax
  __int64 v101; // rax
  int v102; // ebx
  int v103; // eax
  int v104; // eax
  int v105; // eax
  __int64 v106; // r12
  __int64 v107; // rbx
  __int64 v108; // rax
  __int64 v109; // rdx
  char *v110; // rdx
  __m128i *v111; // rsi
  __int8 v112; // dl
  unsigned __int16 v113; // ax
  __int64 v114; // rax
  _QWORD *v115; // rdi
  __int64 v116; // rax
  __int64 *v117; // rsi
  __int64 j; // rdi
  __int64 v119; // rax
  __int64 v120; // rsi
  char v121; // dl
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rcx
  __int64 v125; // rsi
  _QWORD *v126; // r12
  __int64 v127; // rbx
  __int64 v128; // r13
  FILE *v129; // rdx
  __int64 *v130; // rax
  __m128i *v131; // rbx
  __int64 v132; // r12
  __m128i *v133; // r13
  __int64 v134; // rdi
  __int64 v135; // rax
  char v136; // dl
  __int64 v137; // rax
  __m128i v138; // xmm1
  __m128i v139; // xmm2
  __m128i v140; // xmm3
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 v144; // r9
  __int64 v145; // rdx
  __int64 v146; // rcx
  __int64 v147; // r8
  __int64 v148; // r9
  __int64 v149; // rdx
  __int64 v150; // rcx
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // r8
  __int64 v156; // r9
  __int64 v157; // rdx
  __int64 v158; // rcx
  __int64 v159; // r8
  __int64 v160; // r9
  __int64 v161; // rdx
  __int64 v162; // rcx
  __int64 v163; // r8
  __int64 v164; // r9
  __int64 v165; // rdx
  __int64 v166; // r8
  __int64 v167; // r9
  char v168; // al
  __int64 *i; // rax
  __int64 v170; // rax
  __int64 v171; // rax
  __int64 v172; // rax
  __int32 v173; // eax
  int v174; // edx
  int v175; // r8d
  __int64 v176; // rdx
  __int64 v177; // rax
  __int64 v178; // rsi
  __int64 v179; // rcx
  _UNKNOWN *__ptr32 *v180; // r8
  const __m128i *v181; // r10
  __int64 v182; // rbx
  __int64 v183; // r13
  char v184; // al
  __int16 v185; // dx
  __int64 v186; // rax
  __int64 v187; // rdx
  unsigned __int64 *v188; // r14
  __int128 *v189; // r12
  __int64 v190; // rdx
  __int64 v191; // rdx
  __int64 v192; // rax
  __int64 v193; // rdx
  __int64 v194; // rax
  __int64 v195; // rdx
  __int64 v196; // rcx
  __int64 v197; // r8
  __int64 *v198; // r9
  __int64 v199; // rcx
  __int64 v200; // r8
  __int64 v201; // r9
  __int64 v202; // r10
  __int64 **v203; // rax
  __int64 v204; // rdx
  __int64 v205; // rcx
  __int64 v206; // r8
  __int64 v207; // r9
  __int64 v208; // rax
  __int64 v209; // rax
  __int64 v210; // rdx
  __int64 v211; // rcx
  __int64 v212; // rdx
  _QWORD *v213; // rax
  __m128i *v214; // rsi
  __int64 v215; // rdx
  __int64 v216; // [rsp-8h] [rbp-2A8h]
  __m128i *v217; // [rsp+8h] [rbp-298h]
  char v218; // [rsp+10h] [rbp-290h]
  int v219; // [rsp+10h] [rbp-290h]
  unsigned int v220; // [rsp+14h] [rbp-28Ch]
  int v222; // [rsp+28h] [rbp-278h]
  __int64 *v223; // [rsp+28h] [rbp-278h]
  __int64 v224; // [rsp+28h] [rbp-278h]
  __int32 v225; // [rsp+30h] [rbp-270h]
  __int64 v227; // [rsp+40h] [rbp-260h]
  __int64 v228; // [rsp+40h] [rbp-260h]
  int v229; // [rsp+40h] [rbp-260h]
  char v231; // [rsp+48h] [rbp-258h]
  char v232; // [rsp+48h] [rbp-258h]
  __int64 v233; // [rsp+48h] [rbp-258h]
  unsigned __int8 v234; // [rsp+50h] [rbp-250h]
  __m128i *v235; // [rsp+50h] [rbp-250h]
  __int64 v236; // [rsp+50h] [rbp-250h]
  __int64 v237; // [rsp+58h] [rbp-248h]
  int v238; // [rsp+58h] [rbp-248h]
  __int64 v239; // [rsp+58h] [rbp-248h]
  __int64 v240; // [rsp+58h] [rbp-248h]
  int v241; // [rsp+60h] [rbp-240h]
  __int64 v242; // [rsp+60h] [rbp-240h]
  __m128i *v243; // [rsp+60h] [rbp-240h]
  int v244; // [rsp+60h] [rbp-240h]
  __int128 *v245; // [rsp+60h] [rbp-240h]
  char v246; // [rsp+68h] [rbp-238h]
  unsigned int v247; // [rsp+68h] [rbp-238h]
  int v248; // [rsp+68h] [rbp-238h]
  __int64 v249; // [rsp+68h] [rbp-238h]
  unsigned __int16 v250; // [rsp+7Ah] [rbp-226h] BYREF
  int v251; // [rsp+7Ch] [rbp-224h] BYREF
  int v252; // [rsp+80h] [rbp-220h] BYREF
  int v253; // [rsp+84h] [rbp-21Ch] BYREF
  _QWORD *v254; // [rsp+88h] [rbp-218h] BYREF
  _QWORD *v255; // [rsp+90h] [rbp-210h] BYREF
  __int64 v256; // [rsp+98h] [rbp-208h] BYREF
  __m128i v257[2]; // [rsp+A0h] [rbp-200h] BYREF
  __m128i v258; // [rsp+C0h] [rbp-1E0h] BYREF
  __m128i v259; // [rsp+D0h] [rbp-1D0h]
  __m128i v260; // [rsp+E0h] [rbp-1C0h]
  __m128i v261; // [rsp+F0h] [rbp-1B0h]
  __m128i v262[22]; // [rsp+100h] [rbp-1A0h] BYREF
  int v263; // [rsp+260h] [rbp-40h]
  __int16 v264; // [rsp+264h] [rbp-3Ch]

  v4 = a1;
  v5 = &v254;
  v6 = (__int64 *)a1[12].m128i_i64[0];
  v7 = *v6;
  v251 = 0;
  v237 = v7;
  v252 = 0;
  v253 = 0;
  v254 = 0;
  v220 = dword_4F06650[0];
  v8 = 0;
  v9 = (unsigned __int64)&v254;
  v254 = (_QWORD *)sub_5CC190(1);
  sub_6446A0((__int64 *)&v254, 0);
  v12 = word_4F06418[0];
  if ( word_4F06418[0] == 103 )
  {
    v9 = 935;
    goto LABEL_26;
  }
  if ( word_4F06418[0] == 77 || word_4F06418[0] == 95 )
  {
    v9 = 481;
LABEL_26:
    v8 = dword_4F07508;
    sub_6851C0(v9, dword_4F07508);
    sub_7B8B50(v9, dword_4F07508, v39, v40, v41, v42);
    v12 = word_4F06418[0];
  }
  v13 = *(_QWORD *)&dword_4F063F8;
  v4[23].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
  if ( v12 == 153 )
  {
    v4[1].m128i_i32[0] = 1;
    v85 = &v254;
    v256 = v13;
    sub_7B8B50(v9, v8, v13, (__int64)&dword_4F063F8, v10, v11);
    if ( v254 )
      v85 = sub_5CB9F0(&v254);
    *v85 = (_QWORD *)sub_5CC190(1);
    sub_6446A0((__int64 *)&v254, 0);
    sub_88D9B0(1u, 0, v86, v87, v88, v89);
    switch ( word_4F06418[0] )
    {
      case 0x68u:
        v4[24].m128i_i64[0] = qword_4F063F0;
        sub_7B8B50(1u, 0, v90, v91, v92, v93);
        v234 = 11;
        v255 = (_QWORD *)sub_5CC190(2);
        break;
      case 0x97u:
        v4[24].m128i_i64[0] = qword_4F063F0;
        sub_7B8B50(1u, 0, v90, v91, v92, v93);
        v234 = 9;
        v255 = (_QWORD *)sub_5CC190(2);
        break;
      case 0x65u:
        v4[24].m128i_i64[0] = qword_4F063F0;
        sub_7B8B50(1u, 0, v90, v91, v92, v93);
        v234 = 10;
        v255 = (_QWORD *)sub_5CC190(2);
        break;
      default:
        goto LABEL_28;
    }
    v18 = 1;
    sub_6446A0((__int64 *)&v255, 0);
  }
  else
  {
    sub_88D9B0(1u, 0, v13, (__int64)&dword_4F063F8, v10, v11);
    switch ( word_4F06418[0] )
    {
      case 0x68u:
        v18 = 0;
        v4[24].m128i_i64[0] = qword_4F063F0;
        sub_7B8B50(1u, 0, v14, v15, v16, v17);
        v234 = 11;
        v255 = (_QWORD *)sub_5CC190(2);
        break;
      case 0x97u:
        v18 = 0;
        v4[24].m128i_i64[0] = qword_4F063F0;
        sub_7B8B50(1u, 0, v14, v15, v16, v17);
        v234 = 9;
        v255 = (_QWORD *)sub_5CC190(2);
        break;
      case 0x65u:
        v18 = 0;
        v4[24].m128i_i64[0] = qword_4F063F0;
        sub_7B8B50(1u, 0, v14, v15, v16, v17);
        v234 = 10;
        v255 = (_QWORD *)sub_5CC190(2);
        break;
      default:
LABEL_28:
        sub_721090();
    }
  }
  if ( v254 )
    v5 = sub_5CB9F0(&v254);
  *v5 = v255;
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] == 1 )
      goto LABEL_12;
LABEL_32:
    v30 = 40;
    v29 = (__int64)dword_4F07508;
    sub_6851C0(0x28u, dword_4F07508);
    v225 = 0;
    v43 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v44 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v45 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v258.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v25 = 0;
    v259 = v43;
    v258.m128i_i64[1] = *(_QWORD *)dword_4F07508;
    v259.m128i_i8[1] = v43.m128i_i8[1] | 0x20;
    v260 = v44;
    v261 = v45;
    v250 = word_4F06418[0];
    goto LABEL_33;
  }
  if ( word_4F06418[0] == 1 )
  {
    v23 = HIBYTE(word_4D04A10);
    if ( (word_4D04A10 & 0x200) != 0 )
      goto LABEL_13;
  }
  if ( !(unsigned int)sub_7C0F00(0x8401u, 0, v19, v20, v21, v22) )
    goto LABEL_32;
LABEL_12:
  v23 = HIBYTE(word_4D04A10);
LABEL_13:
  if ( (v23 & 0x20) != 0 )
  {
    v82 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v83 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v84 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v258 = _mm_loadu_si128(xmmword_4F06660);
    v259 = v82;
    v260 = v83;
    v261 = v84;
  }
  else
  {
    if ( (word_4D04A10 & 0x58) == 0 )
    {
      if ( (*(_BYTE *)(qword_4D04A00 + 73) & 2) != 0 && word_4F06418[0] == 1 && sub_887690(qword_4D04A00) )
      {
        v192 = qword_4D04A00;
        *(_BYTE *)(qword_4D04A00 + 73) &= ~2u;
        sub_684AE0(0xCF4u, &dword_4F063F8, *(_QWORD *)(v192 + 8));
      }
      v4[22].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
      v225 = v4[1].m128i_i32[0];
      if ( v225 )
      {
        v262[0].m128i_i32[0] = 0;
        v24 = 13;
      }
      else
      {
        if ( (*(_DWORD *)&word_4D04A10 & 0x10001) == 0 )
        {
          v173 = v4[12].m128i_i32[2];
          v174 = dword_4F04C5C;
          v175 = dword_4D03F98[0];
          dword_4D03F98[0] = 0;
          dword_4F04C5C = v173;
          v244 = v174;
          v248 = v175;
          v25 = sub_7CFB70(&qword_4D04A00, 0);
          dword_4D03F98[0] = v248;
          dword_4F04C5C = v244;
          goto LABEL_24;
        }
        v262[0].m128i_i32[0] = 0;
        v24 = 8;
      }
      v25 = sub_7BF130(0x1000u, v24, v262);
      v225 = v262[0].m128i_i32[0];
      if ( v262[0].m128i_i32[0] )
      {
        v25 = 0;
        v26 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v27 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v28 = _mm_loadu_si128(&xmmword_4F06660[3]);
        qword_4D04A00 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        *(__m128i *)&word_4D04A10 = v26;
        HIBYTE(word_4D04A10) = v26.m128i_i8[1] | 0x20;
        v225 = 0;
        qword_4D04A08 = *(_QWORD *)dword_4F07508;
        xmmword_4D04A20 = v27;
        unk_4D04A30 = v28;
LABEL_24:
        v29 = 0;
        v30 = 0;
        v31 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
        v32 = _mm_loadu_si128(&xmmword_4D04A20);
        v33 = qword_4F063F0;
        v34 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
        v258 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
        v259 = v31;
        v4[23].m128i_i64[0] = qword_4F063F0;
        v4[24].m128i_i64[0] = v33;
        v260 = v32;
        v261 = v34;
        v250 = sub_7BE840(0, 0);
        goto LABEL_33;
      }
      if ( v4[1].m128i_i32[0] )
      {
        if ( !v25 )
        {
          if ( !dword_4D047D0 && dword_4F04C64 > 0 )
          {
            v225 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 - 770) & 2) != 0;
            goto LABEL_24;
          }
          goto LABEL_439;
        }
        if ( sub_89D460(v25, (__int64)v4) )
        {
          v225 = 1;
          v25 = 0;
          goto LABEL_24;
        }
      }
      else if ( !v25 )
      {
        goto LABEL_439;
      }
      v168 = *(_BYTE *)(v25 + 80);
      if ( (unsigned __int8)(v168 - 4) > 1u )
      {
LABEL_423:
        if ( !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
        {
LABEL_433:
          if ( v168 != 3 )
            goto LABEL_24;
          goto LABEL_434;
        }
        if ( v168 != 3 )
          goto LABEL_24;
        for ( i = *(__int64 **)(v25 + 88); *((_BYTE *)i + 140) == 12; i = (__int64 *)i[20] )
          ;
        v170 = *i;
        if ( !v170
          || (unsigned __int8)(*(_BYTE *)(v170 + 80) - 4) > 1u
          || *(char *)(*(_QWORD *)(v170 + 88) + 177LL) >= 0 )
        {
LABEL_434:
          if ( *(_BYTE *)(v25 + 104) )
          {
            v172 = *(_QWORD *)(v25 + 88);
            if ( (*(_BYTE *)(v172 + 177) & 0x10) != 0 )
            {
              if ( *(_QWORD *)(*(_QWORD *)(v172 + 168) + 168LL) )
                v25 = sub_880FE0(v25);
            }
          }
          goto LABEL_24;
        }
        v171 = sub_892920(*(_QWORD *)(*(_QWORD *)(v170 + 96) + 72LL));
        v25 = v171;
        if ( v171 )
        {
          v168 = *(_BYTE *)(v171 + 80);
          goto LABEL_433;
        }
LABEL_439:
        v25 = 0;
        goto LABEL_24;
      }
      if ( !*(_QWORD *)(*(_QWORD *)(v25 + 96) + 72LL) || (unk_4D04A12 & 1) == 0 )
        goto LABEL_24;
      v176 = sub_892950(v25);
      v177 = *(_QWORD *)(v176 + 88);
      if ( (*(_BYTE *)(v177 + 266) & 0x20) != 0 )
      {
        sub_6854C0(0x317u, (FILE *)&qword_4D04A08, v176);
        v4[1].m128i_i32[3] = 0;
        goto LABEL_439;
      }
      v178 = *(_QWORD *)(v25 + 88);
      v245 = *(__int128 **)(v4[30].m128i_i64[1] + 16);
      v179 = *(_QWORD *)(v177 + 152);
      v249 = v179;
      if ( v179 )
      {
        v229 = 0;
        v177 = *(_QWORD *)(v179 + 88);
        if ( v245 )
          goto LABEL_459;
      }
      else
      {
        if ( *(char *)(v178 + 177) < 0 )
        {
          if ( v245 )
          {
            v249 = v176;
            v229 = 1;
LABEL_459:
            if ( (*(_BYTE *)(v178 + 177) & 0x20) != 0 )
            {
              v180 = *(_UNKNOWN *__ptr32 **)(v177 + 144);
              v181 = *(const __m128i **)(*(_QWORD *)(v178 + 168) + 168LL);
              if ( !v180 )
              {
LABEL_524:
                if ( v229 )
                  v214 = (__m128i *)sub_896D70(v249, *(_QWORD *)v4[12].m128i_i64[0], 1);
                else
                  v214 = sub_72F240(v181);
                v25 = (unsigned __int64)sub_893FE0(v249, v214->m128i_i64, 1u);
                *(_BYTE *)(v4->m128i_i64[0] + 127) |= 0x10u;
                v4[1].m128i_i32[3] = 1;
                if ( !v25 )
                  goto LABEL_439;
                v168 = *(_BYTE *)(v25 + 80);
                goto LABEL_423;
              }
              v223 = v6;
              v182 = *(_QWORD *)(*(_QWORD *)(v178 + 168) + 168LL);
              v219 = v18;
              v183 = *(_QWORD *)(v177 + 144);
              v217 = v4;
              while ( 1 )
              {
                v186 = *(_QWORD *)(v183 + 88);
                v187 = *(_QWORD *)(v186 + 104);
                v188 = *(unsigned __int64 **)(v187 + 192);
                v189 = *(__int128 **)(*(_QWORD *)(v187 + 176) + 16LL);
                if ( *(_BYTE *)(v183 + 80) == 19 )
                {
                  v190 = *(_QWORD *)(v186 + 200);
                  if ( v190 )
                    v186 = *(_QWORD *)(v190 + 88);
                }
                v184 = *(_BYTE *)(v186 + 160);
                v185 = 2 * ((v184 & 6) != 0);
                if ( (v184 & 0x10) != 0 )
                  v185 = (2 * ((v184 & 6) != 0)) | 0x20;
                if ( sub_89AB40(*(_QWORD *)(v188[21] + 168), v182, v185 | 0x48u, v179, v180)
                  && (unsigned int)sub_739400(v189, v245) )
                {
                  break;
                }
                v183 = *(_QWORD *)(v183 + 8);
                if ( !v183 )
                {
                  v181 = (const __m128i *)v182;
                  v18 = v219;
                  v6 = v223;
                  v4 = v217;
                  goto LABEL_524;
                }
              }
              v181 = (const __m128i *)v182;
              v18 = v219;
              v25 = *v188;
              v6 = v223;
              v4 = v217;
              if ( !v25 )
                goto LABEL_524;
            }
            goto LABEL_503;
          }
          goto LABEL_503;
        }
        v249 = v176;
        if ( v245 )
        {
LABEL_522:
          v229 = 0;
          goto LABEL_459;
        }
      }
      if ( (*(_BYTE *)(v177 + 160) & 0x40) != 0 )
        goto LABEL_522;
LABEL_503:
      v4[1].m128i_i32[3] = 1;
      v168 = *(_BYTE *)(v25 + 80);
      goto LABEL_423;
    }
    sub_6851C0(0x1F6u, &qword_4D04A08);
    v138 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v139 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v140 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v258 = _mm_loadu_si128(xmmword_4F06660);
    v259 = v138;
    v260 = v139;
    v261 = v140;
  }
  v29 = 0;
  v30 = 0;
  v25 = 0;
  v259.m128i_i8[1] |= 0x20u;
  v258.m128i_i64[1] = *(_QWORD *)dword_4F07508;
  v225 = 0;
  v250 = sub_7BE840(0, 0);
LABEL_33:
  if ( ((word_4D04A10 & 1) != 0 || (v36 = dword_4D04870) != 0) && dword_4F04C40 != -1 )
  {
    v35 = (__int64)qword_4F04C68;
    if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
      sub_87DD80();
  }
  if ( v250 == 1
    && dword_4F077C4 == 2
    && (unk_4F07778 > 201102 || (v35 = dword_4F07774) != 0 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9EFBu) )
  {
    v30 = (unsigned __int64)&v250;
    v29 = 73;
    sub_668C50(&v250, 73, 1);
  }
  v46 = word_4F06418[0];
  if ( word_4F06418[0] == 1 )
  {
    sub_7B8B50(v30, (unsigned int *)v29, v35, v36, v37, v38);
    if ( dword_4F077C4 == 2
      && (unk_4F07778 > 201102 || dword_4F07774 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9EFBu) )
    {
      sub_66A730(v234, &v252, &v251, &v253);
    }
    v46 = word_4F06418[0];
  }
  v47 = v4[1].m128i_i32[0];
  v48 = dword_4F0664C;
  v241 = v47;
  if ( v46 == 55 )
  {
    v4[2].m128i_i32[1] = 1;
    if ( v47 )
      goto LABEL_84;
LABEL_62:
    v246 = 1;
    v241 = 1;
    goto LABEL_63;
  }
  if ( v46 != 73 )
  {
    v4[2].m128i_i32[1] = 0;
    if ( !v47 )
    {
      v246 = 0;
      goto LABEL_63;
    }
    v246 = 0;
    v241 = 0;
    if ( v4[15].m128i_i64[0] )
    {
LABEL_44:
      if ( v25 )
      {
        if ( (*(_WORD *)(v25 + 80) & 0x40FF) == 0x4013 )
        {
          v25 = 0;
        }
        else if ( (unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0)
               && *(_BYTE *)(v25 + 80) == 19
               && (*(_BYTE *)(*(_QWORD *)(v25 + 88) + 160LL) & 2) != 0 )
        {
          sub_6851C0(0x3F5u, &v258.m128i_i32[2]);
        }
      }
      v49 = v237;
      if ( v237 )
      {
        v50 = 0;
        do
        {
          v51 = *(_BYTE *)(v49 + 56);
          if ( (v51 & 1) != 0 )
          {
            v50 = 1;
            *(_BYTE *)(v49 + 56) = v51 & 0xFE;
          }
          v49 = *(_QWORD *)v49;
        }
        while ( v49 );
        if ( v50 && !v4[2].m128i_i32[3] )
          sub_684AA0(unk_4F07471, 0x58Du, &v258.m128i_i32[2]);
      }
LABEL_57:
      if ( !v4[1].m128i_i32[0] )
        goto LABEL_63;
      goto LABEL_58;
    }
    goto LABEL_281;
  }
  v4[2].m128i_i32[1] = 1;
  if ( !v47 )
    goto LABEL_62;
LABEL_84:
  if ( v4[15].m128i_i64[0] )
  {
    if ( dword_4F077BC && qword_4F077A8 <= 0x76BFu && (v259.m128i_i32[0] & 0x10001) == 0 )
    {
      v25 = 0;
      sub_684B30(0x561u, &v258.m128i_i32[2]);
      v4[1].m128i_i32[0] = 0;
      v241 = 1;
      v246 = 1;
      v4[12].m128i_i32[3] = dword_4F04C64 - 1;
      goto LABEL_63;
    }
    sub_6851C0(0x298u, &v258.m128i_i32[2]);
    v4[3].m128i_i32[1] = 1;
    v241 = 1;
    v246 = 1;
    goto LABEL_44;
  }
  v241 = 1;
  v246 = 1;
LABEL_281:
  if ( v18 )
  {
    sub_6851C0(0xEFu, &v256);
    v4[3].m128i_i32[1] = 1;
    goto LABEL_57;
  }
LABEL_58:
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 8) == 0
    || (v4[3].m128i_i32[1] = 1, sub_6851C0(0x3D2u, &v258.m128i_i32[2]), v4[1].m128i_i32[0]) )
  {
    *(_BYTE *)(v4[21].m128i_i64[0] + 121) |= 8u;
  }
LABEL_63:
  v227 = 0;
  if ( !v4[1].m128i_i32[3] )
    goto LABEL_64;
  if ( (unsigned __int8)(*(_BYTE *)(v25 + 80) - 4) > 1u )
    goto LABEL_90;
  v94 = *(_BYTE *)(*(_QWORD *)(v25 + 88) + 177LL);
  if ( v94 < 0 )
  {
    v25 = *(_QWORD *)(*(_QWORD *)(v25 + 96) + 72LL);
    if ( !*(_QWORD *)(*(_QWORD *)(v25 + 88) + 152LL) )
    {
      sub_684AA0(8u, 0x348u, &v258.m128i_i32[2]);
      goto LABEL_91;
    }
  }
  else
  {
    if ( (v94 & 0x30) != 0x30 )
    {
LABEL_90:
      sub_6854C0(0x342u, (FILE *)&v258.m128i_u64[1], v25);
LABEL_91:
      v227 = 0;
LABEL_92:
      v4[1].m128i_i32[3] = 0;
      v4[3].m128i_i32[1] = 1;
      if ( a4 )
        goto LABEL_93;
      sub_8975E0(v4, v48, 0);
      v52 = v4[3].m128i_i32[1];
      goto LABEL_174;
    }
    v95 = *(_BYTE *)(v25 + 81) & 0x10;
    if ( v4[1].m128i_i32[0] || (v191 = v4[15].m128i_i64[0]) == 0 )
    {
      if ( !v95 )
      {
        v96 = qword_4F04C68[0] + 776LL * v4[12].m128i_i32[3];
        if ( *(_QWORD *)(v96 + 224) != *(_QWORD *)(v25 + 64) && !dword_4F077BC && !(unsigned int)sub_85ED80(v25, v96) )
        {
          sub_6851C0(0x359u, &v258.m128i_i32[2]);
          goto LABEL_91;
        }
      }
    }
    else if ( !v95 || v191 != *(_QWORD *)(v25 + 64) )
    {
      sub_6854C0(0x40Bu, (FILE *)&v258.m128i_u64[1], v25);
      goto LABEL_91;
    }
    v97 = sub_892950(v25);
    if ( (*(_BYTE *)(v97 + 81) & 0x10) != 0 && !v4[15].m128i_i64[0] )
    {
      sub_890490((__int64)v4, *(__int64 **)(v97 + 64));
      v227 = v25;
      v25 = 0;
    }
    else
    {
      v227 = v25;
      v25 = 0;
    }
  }
  if ( v4[1].m128i_i32[0] && !v4[3].m128i_i32[1] )
  {
    sub_6851C0(0x361u, &v258.m128i_i32[2]);
    goto LABEL_92;
  }
LABEL_64:
  if ( a4 )
  {
    v52 = v4[3].m128i_i32[1];
    if ( v25 )
      goto LABEL_66;
LABEL_174:
    if ( (v259.m128i_i8[0] & 1) != 0 )
    {
      if ( !v52 )
      {
        v25 = 0;
        v59 = 0;
        goto LABEL_177;
      }
      goto LABEL_93;
    }
    if ( v52 )
      goto LABEL_93;
    goto LABEL_71;
  }
  sub_8975E0(v4, v48, 0);
  v52 = v4[3].m128i_i32[1];
  if ( !v25 )
    goto LABEL_174;
LABEL_66:
  if ( v52 )
    goto LABEL_93;
  if ( (*(_BYTE *)(v25 + 81) & 0x20) == 0 )
  {
    v53 = *(_BYTE *)(v25 + 80);
    if ( v53 == 19 )
    {
      v56 = *(_QWORD *)(v25 + 88);
      if ( v56 && (*(_BYTE *)(v56 + 265) & 1) != 0 )
      {
LABEL_70:
        if ( (v259.m128i_i8[0] & 1) != 0 )
        {
          sub_6854C0(0x2DAu, (FILE *)&v258.m128i_u64[1], v25);
          v4[3].m128i_i32[1] = 1;
          goto LABEL_93;
        }
LABEL_71:
        v25 = 0;
        if ( v4[14].m128i_i64[0] <= 1uLL )
          goto LABEL_442;
LABEL_72:
        if ( !dword_4F077BC || !v4[1].m128i_i32[3] || (v54 = 5, *(_QWORD *)(v4[12].m128i_i64[0] + 24)) )
        {
          v4[3].m128i_i32[1] = 1;
          v54 = 8;
        }
        sub_684AA0(v54, 0x309u, &v258.m128i_i32[2]);
        v52 = v4[3].m128i_i32[1];
        if ( !v52 )
        {
          if ( v25 )
            goto LABEL_77;
LABEL_442:
          v59 = 0;
          goto LABEL_177;
        }
LABEL_93:
        v259.m128i_i64[1] = 0;
        v57 = v259.m128i_i8[1] | 0x20;
        v259.m128i_i8[1] |= 0x20u;
        v58 = *v6;
        if ( *v6 )
        {
          v25 = 0;
          LOBYTE(v52) = 0;
          v59 = 0;
          goto LABEL_97;
        }
        if ( v237 )
        {
          v222 = v225;
          goto LABEL_187;
        }
LABEL_508:
        v60 = v4[1].m128i_i32[3];
LABEL_104:
        v259.m128i_i64[1] = 0;
        v259.m128i_i8[1] = v57 | 0x20;
        v222 = 1;
LABEL_105:
        sub_88F9D0((__int64 *)v237, v60);
        v61 = v4[1].m128i_i32[3];
        v62 = v61 != 0;
        if ( !v227 )
        {
          if ( v61 )
          {
            v4[1].m128i_i32[3] = 0;
            v227 = 0;
          }
          goto LABEL_108;
        }
LABEL_259:
        if ( v62 )
        {
          v106 = sub_892920(*(_QWORD *)(*(_QWORD *)(v227 + 96) + 72LL));
          v107 = *(_QWORD *)(v106 + 88);
          v25 = (unsigned __int64)sub_87EBB0(0x13u, *(_QWORD *)v106, &v258.m128i_i64[1]);
          v108 = *(_QWORD *)(v25 + 88);
          *(_DWORD *)(v25 + 40) = *(_DWORD *)(v106 + 40);
          v109 = *(_QWORD *)(v106 + 64);
          if ( (*(_BYTE *)(v106 + 81) & 0x10) != 0 )
          {
            *(_BYTE *)(v25 + 81) |= 0x10u;
            *(_QWORD *)(v25 + 64) = v109;
            *(_BYTE *)(v108 + 265) = (v4[10].m128i_i8[4] << 6) | *(_BYTE *)(v108 + 265) & 0x3F;
          }
          else if ( v109 )
          {
            *(_QWORD *)(v25 + 64) = v109;
          }
          *(_QWORD *)(v108 + 152) = v106;
          if ( !v4[3].m128i_i32[1] && (v259.m128i_i8[1] & 0x20) == 0 )
          {
            *(_QWORD *)(v25 + 8) = *(_QWORD *)(v107 + 144);
            *(_QWORD *)(v107 + 144) = v25;
            if ( *(_QWORD *)(v4[30].m128i_i64[1] + 16) )
              *(_BYTE *)(v107 + 160) |= 0x40u;
            if ( (v234 == 11) != (*(_BYTE *)(v107 + 264) == 11) )
            {
              v110 = "struct";
              if ( v234 != 10 )
              {
                v110 = "class";
                if ( v234 == 11 )
                  v110 = "union";
              }
              sub_686A10(0x1D5u, &v258.m128i_i32[2], (__int64)v110, v106);
            }
          }
          v59 = *(__m128i **)(v25 + 88);
          v67 = 0;
          goto LABEL_111;
        }
LABEL_108:
        v25 = (unsigned __int64)sub_87EF90(0x13u, (__int64)&v258);
        v67 = v225 ^ 1;
        if ( v225 )
        {
          *(_BYTE *)(v25 + 83) |= 0x40u;
        }
        else
        {
          v66 = dword_4D047D0;
          if ( !dword_4D047D0 )
          {
            v63 = (unsigned __int8)(v4[1].m128i_i8[0] & 1) << 6;
            *(_BYTE *)(v25 + 83) = ((v4[1].m128i_i8[0] & 1) << 6) | *(_BYTE *)(v25 + 83) & 0xBF;
          }
        }
        v59 = *(__m128i **)(v25 + 88);
        sub_88DD80((__int64)v4, v25, v63, v64, v65, v66);
LABEL_111:
        v59[16].m128i_i8[8] = v234;
        v68 = (8 * (v4[5].m128i_i8[4] & 1)) | v59[10].m128i_i8[0] & 0xF7;
        v59[10].m128i_i8[0] = v68;
        v69 = v68 & 0xEF | (16 * (v4[5].m128i_i8[8] & 1));
        v59[10].m128i_i8[0] = v69;
        v70 = dword_4F077C4 == 2;
        v59[10].m128i_i8[0] = (32 * (v4[8].m128i_i8[0] & 1)) | v69 & 0xDF;
        v71 = 2;
        if ( v70 && (unk_4F07778 > 201102 || dword_4F07774) )
          v71 = ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 8) & 2) == 0) + 1;
        v59[16].m128i_i8[9] = (4 * v71) | v59[16].m128i_i8[9] & 0xE3;
        sub_897580((__int64)v4, (__int64 *)v25, (__int64)v59);
        v231 = 1;
        if ( v67 )
        {
          v67 = 0;
          sub_885A00(v25, v4[12].m128i_i32[3], v222);
          if ( *(_QWORD *)v25 == qword_4F60230
            && (*(_BYTE *)(v25 + 81) & 0x10) == 0
            && qword_4D049B8
            && *(_QWORD *)(v25 + 64) == qword_4D049B8[11]
            && !*(_QWORD *)&dword_4D04988 )
          {
            if ( v237
              && !*(_QWORD *)v237
              && (v215 = *(_QWORD *)(v237 + 8)) != 0
              && *(_BYTE *)(v215 + 80) == 3
              && (*(_BYTE *)(v237 + 56) & 0x11) == 0 )
            {
              *(_QWORD *)&dword_4D04988 = v25;
              v67 = 0;
              v59[16].m128i_i8[10] |= 0x20u;
              v231 = 1;
            }
            else
            {
              v67 = 0;
              sub_6851C0(0x92Eu, &v258.m128i_i32[2]);
              v231 = 1;
            }
          }
        }
        goto LABEL_113;
      }
    }
    else if ( (unsigned __int8)(v53 - 4) > 1u )
    {
      goto LABEL_70;
    }
  }
  if ( (v259.m128i_i8[0] & 1) == 0 && v4[14].m128i_i64[0] > 1uLL )
    goto LABEL_72;
LABEL_77:
  v55 = *(_BYTE *)(v25 + 80);
  switch ( v55 )
  {
    case 4:
    case 5:
      v59 = *(__m128i **)(*(_QWORD *)(v25 + 96) + 80LL);
      goto LABEL_276;
    case 6:
      v59 = *(__m128i **)(*(_QWORD *)(v25 + 96) + 32LL);
      goto LABEL_291;
    case 9:
    case 10:
      v59 = *(__m128i **)(*(_QWORD *)(v25 + 96) + 56LL);
LABEL_291:
      LOBYTE(v52) = 0;
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v59 = *(__m128i **)(v25 + 88);
      goto LABEL_276;
    default:
      v59 = 0;
LABEL_276:
      LOBYTE(v52) = 0;
      if ( (unsigned __int8)(v55 - 4) <= 1u
        && (*(_BYTE *)(v25 + 81) & 0x10) != 0
        && (!v4[1].m128i_i32[1] || (v52 = v4[1].m128i_i32[0]) != 0) )
      {
        LOBYTE(v52) = ~v259.m128i_i8[2] & (v59 != 0);
      }
      break;
  }
LABEL_177:
  v58 = *v6;
  if ( !*v6 )
  {
LABEL_178:
    if ( (v259.m128i_i8[0] & 1) == 0 )
    {
      v232 = v52;
      if ( v25 )
      {
        v80 = sub_88F4E0(v4, v25, (__int64)&v258, v241);
        LOBYTE(v52) = v232;
        v222 = v80;
        goto LABEL_181;
      }
      v222 = v225;
LABEL_253:
      v60 = v4[1].m128i_i32[3];
      if ( v60 && v227 )
      {
        v104 = sub_88F4E0(v4, v227, (__int64)&v258, v241);
        v60 = v4[1].m128i_i32[3];
        v222 = v104;
LABEL_257:
        if ( !v237 )
          goto LABEL_103;
        sub_88F9D0((__int64 *)v237, v60);
        v62 = v4[1].m128i_i32[3] != 0;
        goto LABEL_259;
      }
      goto LABEL_102;
    }
    v218 = v52;
    v222 = v225;
    if ( v25 )
    {
      v103 = sub_890C40((__int64)v4, v25, (__int64)&v258, v241, 0);
      LOBYTE(v52) = v218;
      v222 = v103;
LABEL_181:
      if ( v237 )
        goto LABEL_148;
      goto LABEL_166;
    }
LABEL_100:
    v60 = v4[1].m128i_i32[3];
    if ( v227 && v60 )
    {
      v105 = sub_890C40((__int64)v4, v227, (__int64)&v258, v241, a4);
      v60 = v4[1].m128i_i32[3];
      v222 = v105;
      goto LABEL_257;
    }
LABEL_102:
    if ( !v237 )
      goto LABEL_103;
    goto LABEL_105;
  }
  do
  {
LABEL_97:
    if ( **(_QWORD **)(v58 + 8) == v258.m128i_i64[0] )
    {
      sub_6851C0(0x1FCu, &v258.m128i_i32[2]);
      if ( !v4[3].m128i_i32[1] )
      {
        v222 = 1;
        if ( (v259.m128i_i8[0] & 1) == 0 )
          goto LABEL_253;
        goto LABEL_100;
      }
      if ( v237 )
      {
        v222 = 1;
        goto LABEL_187;
      }
LABEL_507:
      v57 = v259.m128i_i8[1];
      goto LABEL_508;
    }
    v58 = *(_QWORD *)v58;
  }
  while ( v58 );
  if ( !v4[3].m128i_i32[1] )
    goto LABEL_178;
  if ( !v237 )
    goto LABEL_507;
  v222 = v225;
  if ( !v25 )
    goto LABEL_187;
LABEL_148:
  v77 = *(_BYTE *)(v25 + 80);
  if ( v77 != 19 && (v52 & 1) == 0 )
  {
    if ( (v259.m128i_i8[0] & 1) != 0 )
    {
      if ( (unsigned __int8)(v77 - 4) <= 1u )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v25 + 96) + 72LL) )
        {
          if ( (v259.m128i_i8[2] & 1) != 0 )
          {
            v78 = *(_QWORD *)(v25 + 88);
            if ( !v78 || (*(_BYTE *)(v78 + 177) & 0x20) != 0 )
            {
              sub_6854C0(0x311u, (FILE *)&v258.m128i_u64[1], v25);
              v60 = v4[1].m128i_i32[3];
              v57 = v259.m128i_i8[1];
              goto LABEL_104;
            }
          }
        }
      }
      sub_6854C0(0x2DAu, (FILE *)&v258.m128i_u64[1], v25);
      v60 = v4[1].m128i_i32[3];
LABEL_103:
      v57 = v259.m128i_i8[1];
      goto LABEL_104;
    }
LABEL_187:
    v60 = v4[1].m128i_i32[3];
    goto LABEL_105;
  }
  if ( (v234 == 11) != (v59[16].m128i_i8[8] == 11) )
  {
    sub_6854C0(0x93u, (FILE *)&v258.m128i_u64[1], v25);
    if ( v4[3].m128i_i32[1] )
      goto LABEL_159;
LABEL_241:
    if ( (*(_BYTE *)(v25 + 81) & 0x10) != 0 && !v4[2].m128i_i32[3] && !v4[3].m128i_i32[0] )
    {
      v102 = 1;
LABEL_245:
      if ( (v4[15].m128i_i64[0] && !v4[1].m128i_i32[0]
         || (unsigned int)sub_89BFC0((__int64)v4, v25, v4[2].m128i_i32[1] == 0, (FILE *)dword_4F07508))
        && !v102 )
      {
        v81 = *(_BYTE *)(v25 + 81);
        v77 = *(_BYTE *)(v25 + 80);
        v79 = v81;
        goto LABEL_191;
      }
    }
LABEL_166:
    v60 = v4[1].m128i_i32[3];
    goto LABEL_103;
  }
  v79 = *(_BYTE *)(v25 + 81);
  v81 = v79;
  if ( (v79 & 2) == 0 )
  {
    if ( !v4[3].m128i_i32[1] )
    {
      if ( (v79 & 0x10) == 0 )
        goto LABEL_191;
      goto LABEL_293;
    }
    goto LABEL_160;
  }
  if ( v4[2].m128i_i32[1] )
  {
    sub_685920(&v258.m128i_i32[2], (FILE *)v25, 8u);
    if ( !v4[3].m128i_i32[1] )
      goto LABEL_241;
LABEL_159:
    v79 = *(_BYTE *)(v25 + 81);
    goto LABEL_160;
  }
  if ( v4[3].m128i_i32[1] )
  {
LABEL_160:
    if ( (v79 & 0x10) != 0 && !v4[2].m128i_i32[3] && !v4[3].m128i_i32[0] && (!v4[15].m128i_i64[0] || v4[1].m128i_i32[0]) )
      sub_89BFC0((__int64)v4, v25, v4[2].m128i_i32[1] == 0, (FILE *)dword_4F07508);
    goto LABEL_166;
  }
  if ( (v79 & 0x10) != 0 )
  {
LABEL_293:
    if ( !v4[2].m128i_i32[3] )
    {
      v102 = v4[3].m128i_i32[0];
      if ( !v102 )
        goto LABEL_245;
      v79 = *(_BYTE *)(v25 + 81);
    }
LABEL_191:
    if ( v77 == 19 && (v59[10].m128i_i8[0] & 2) == 0 )
    {
      v98 = 1;
      if ( (v81 & 0x10) != 0 && !v4[15].m128i_i64[0] )
        v98 = (*(_BYTE *)(*(_QWORD *)(v25 + 64) + 177LL) & 0x90) != 0x90;
      goto LABEL_227;
    }
    goto LABEL_192;
  }
  if ( v77 == 19 )
  {
    if ( (v59[10].m128i_i8[0] & 2) != 0 )
      goto LABEL_192;
    v98 = 1;
LABEL_227:
    if ( !dword_4F077BC )
      goto LABEL_492;
    if ( !v4[1].m128i_i32[0] )
      goto LABEL_492;
    v99 = 776LL * dword_4F04C64;
    v100 = *(int *)(qword_4F04C68[0] + v99 + 552);
    if ( (int)v100 <= 0 )
      goto LABEL_492;
    while ( 1 )
    {
      v101 = qword_4F04C68[0] + 776 * v100;
      if ( *(_BYTE *)(v101 + 4) == 9 && v25 == *(_QWORD *)(v101 + 368) )
        break;
      v100 = *(int *)(v101 + 552);
      if ( (int)v100 <= 0 )
        goto LABEL_492;
    }
    if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x1FBCFu
      || dword_4F04C64 > 0 && (*(_BYTE *)(qword_4F04C68[0] + v99 - 770) & 2) != 0 )
    {
LABEL_238:
      if ( (v259.m128i_i8[0] & 1) != 0 )
      {
        sub_89BD20(v237, (__int64)v4, v25, &v258.m128i_i32[2], v98, 0, 1, 5u);
        v79 = *(_BYTE *)(v25 + 81);
        goto LABEL_192;
      }
    }
    else
    {
LABEL_492:
      sub_88FB80(*(_QWORD *)(v59[2].m128i_i64[0] + 32), *(_QWORD *)(v4[12].m128i_i64[0] + 32), (__int64)&v258, v25);
      if ( dword_4F077BC && v4[1].m128i_i32[0] )
        goto LABEL_238;
    }
    if ( !sub_89BD20(v237, (__int64)v4, v25, &v258.m128i_i32[2], v98, 0, 1, 8u) )
      goto LABEL_166;
    v79 = *(_BYTE *)(v25 + 81);
LABEL_192:
    if ( (v79 & 2) == 0 && v246 )
      *a3 = 1;
  }
  v67 = 1;
  sub_88F9D0((__int64 *)v237, v4[1].m128i_i32[3]);
  v231 = 0;
LABEL_113:
  sub_88F380((__int64)v4, (__int64)v59);
  if ( v246 )
    v59[16].m128i_i8[8] = v234;
  if ( !v4[1].m128i_i32[0] )
  {
    if ( (*(_WORD *)(v25 + 80) & 0x10FF) == 0x1013 )
      goto LABEL_299;
    goto LABEL_120;
  }
  v72 = v4[15].m128i_i64[0];
  if ( !v72 || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0 )
    goto LABEL_120;
  sub_8967D0(v59, v72);
  if ( (*(_WORD *)(v25 + 80) & 0x10FF) != 0x1013 || v4[1].m128i_i32[0] )
    goto LABEL_120;
LABEL_299:
  if ( v4[2].m128i_i32[3] || v4[3].m128i_i32[0] || v4[4].m128i_i32[1] )
  {
    if ( v59[4].m128i_i32[0] )
    {
LABEL_396:
      if ( v4[1].m128i_i32[2] )
        goto LABEL_307;
LABEL_122:
      if ( !v67 )
        goto LABEL_123;
LABEL_318:
      sub_5D2670(v254, (__int64)v59, (__int64)&v258.m128i_i64[1], v241);
      v70 = v59[11].m128i_i64[0] == 0;
      v254 = 0;
      if ( !v70 )
        goto LABEL_124;
      goto LABEL_319;
    }
    v59[4].m128i_i32[0] = v220;
LABEL_120:
    if ( v4[1].m128i_i32[2] && !v4[1].m128i_i32[0] )
      goto LABEL_307;
    goto LABEL_122;
  }
  if ( !v4[15].m128i_i64[0] )
    goto LABEL_396;
  if ( v67 )
  {
    if ( v4[1].m128i_i32[2] )
      goto LABEL_307;
    goto LABEL_318;
  }
  if ( !v222 )
  {
    v208 = sub_8788F0(**(_QWORD **)(v25 + 64));
    if ( v208 )
    {
      v209 = sub_883800(*(_QWORD *)(v208 + 96) + 192LL, *(_QWORD *)v25);
      if ( v209 )
      {
        while ( *(_BYTE *)(v209 + 80) != 19 )
        {
          v209 = *(_QWORD *)(v209 + 32);
          if ( !v209 )
            goto LABEL_305;
        }
        v210 = *(_QWORD *)(v209 + 88);
        if ( v220 != *(_DWORD *)(v210 + 64) )
        {
          v209 = *(_QWORD *)(v210 + 144);
          if ( !v209 )
            goto LABEL_305;
          while ( v220 != *(_DWORD *)(*(_QWORD *)(v209 + 88) + 64LL) )
          {
            v209 = *(_QWORD *)(v209 + 8);
            if ( !v209 )
              goto LABEL_305;
          }
        }
        v211 = *(_QWORD *)(v209 + 88);
        v212 = *(_QWORD *)(v25 + 88);
        *(_BYTE *)(v212 + 265) |= 2u;
        *(_QWORD *)(v212 + 88) = v209;
        v240 = v211;
        *(_BYTE *)(v212 + 264) = *(_BYTE *)(v211 + 264);
        v213 = sub_878440();
        v213[1] = v25;
        *v213 = *(_QWORD *)(v240 + 96);
        *(_QWORD *)(v240 + 96) = v213;
      }
    }
  }
LABEL_305:
  if ( v4[1].m128i_i32[2] && !v4[1].m128i_i32[0] )
  {
LABEL_307:
    if ( (v59[10].m128i_i8[0] & 1) == 0 )
      sub_899910(v25, (__int64)v59, (FILE *)&v258.m128i_u64[1]);
    goto LABEL_122;
  }
LABEL_123:
  v70 = v59[11].m128i_i64[0] == 0;
  v59[8].m128i_i64[0] = (__int64)v254;
  if ( v70 )
LABEL_319:
    sub_896F00((__int64)v4, v25, (__int64)v59, v227, v4[1].m128i_i32[3]);
LABEL_124:
  if ( (!v59[5].m128i_i64[1] || (v59[10].m128i_i8[0] & 1) != 0)
    && dword_4F077C4 == 2
    && (unk_4F07778 > 201102 || dword_4F07774 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9EFBu) )
  {
    sub_66A7B0(*(_QWORD *)(v59[11].m128i_i64[0] + 88), v252);
  }
  if ( !v4[1].m128i_i32[3] || !v231 )
    goto LABEL_129;
  v123 = *(_QWORD *)(v25 + 88);
  v124 = *(_QWORD *)(v123 + 176);
  v242 = *(_QWORD *)(v124 + 88);
  if ( *(_QWORD *)v4[12].m128i_i64[0] )
  {
    v125 = *(_QWORD *)(v124 + 88);
    v235 = v59;
    v126 = *(_QWORD **)v4[12].m128i_i64[0];
    v127 = *(_QWORD *)(v123 + 176);
    while ( 1 )
    {
      v128 = v126[1];
      if ( (unsigned int)sub_88FAD0(*(_BYTE *)(v128 + 80), *(_QWORD *)(v128 + 88), v125, 1u) )
        goto LABEL_350;
      v129 = (FILE *)(v128 + 48);
      if ( dword_4F077BC )
      {
        sub_686B60(5u, 0x34Au, v129, v128, v127);
        v126 = (_QWORD *)*v126;
        if ( !v126 )
        {
LABEL_354:
          v59 = v235;
          v123 = *(_QWORD *)(v25 + 88);
          break;
        }
      }
      else
      {
        sub_686B60(8u, 0x34Au, v129, v128, v127);
        v4[3].m128i_i32[1] = 1;
LABEL_350:
        v126 = (_QWORD *)*v126;
        if ( !v126 )
          goto LABEL_354;
      }
    }
  }
  v130 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v123 + 152) + 88LL) + 32LL);
  if ( !v130 )
    goto LABEL_375;
  v228 = *v130;
  v131 = *(__m128i **)(*(_QWORD *)(v242 + 168) + 168LL);
  if ( !v131 )
    goto LABEL_375;
  v243 = v59;
  v132 = *v130;
  v133 = v131;
  do
  {
    if ( (v133[1].m128i_i8[8] & 0x10) != 0 && v133->m128i_i64[0] )
    {
      sub_6851C0(0x995u, (_DWORD *)(v25 + 48));
      v4[3].m128i_i32[1] = 1;
    }
    if ( v133->m128i_i8[8] != 1 )
      goto LABEL_358;
    v134 = v133[2].m128i_i64[0];
    if ( !*(_BYTE *)(v134 + 173)
      || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0 )
    {
      goto LABEL_358;
    }
    v135 = sub_730E00(v134);
    v136 = *(_BYTE *)(v135 + 173);
    if ( v136 == 12 && !*(_BYTE *)(v135 + 176) )
      goto LABEL_358;
    if ( HIDWORD(qword_4F077B4) )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 <= 0x1D4BFu )
          goto LABEL_446;
        goto LABEL_371;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_371;
    }
    if ( qword_4F077A0 <= 0x9C3Fu )
    {
LABEL_446:
      if ( v136 == 12 )
        goto LABEL_373;
      goto LABEL_358;
    }
LABEL_371:
    if ( (*(_BYTE *)(v132 + 57) & 8) != 0 )
    {
      if ( !(unsigned int)sub_8DBE70(*(_QWORD *)(v135 + 128)) )
        goto LABEL_358;
LABEL_373:
      sub_6851C0(0x34Cu, (_DWORD *)(v25 + 48));
      v133[2].m128i_i64[0] = (__int64)sub_72C9A0();
      goto LABEL_358;
    }
    v202 = *(_QWORD *)(*(_QWORD *)(v132 + 64) + 128LL);
    v257[0].m128i_i32[0] = 0;
    v224 = v202;
    sub_892150(v262);
    v203 = sub_8A2270(v224, v131, v228, 0, 0, v257[0].m128i_i32, v262);
    if ( !v257[0].m128i_i32[0] && (unsigned int)sub_8DBE70(v203) )
      goto LABEL_373;
LABEL_358:
    if ( (*(_BYTE *)(v132 + 56) & 0x10) == 0 )
      v132 = *(_QWORD *)v132;
    v133 = (__m128i *)v133->m128i_i64[0];
  }
  while ( v133 );
  v59 = v243;
LABEL_375:
  if ( !v4[3].m128i_i32[1] )
  {
    v137 = v4[28].m128i_i64[1];
    if ( v137 )
    {
      sub_6851A0(0x59Fu, &v258.m128i_i32[2], *(_QWORD *)(*(_QWORD *)v137 + 8LL));
      v4[3].m128i_i32[1] = 1;
    }
  }
LABEL_129:
  v73 = dword_4F07590;
  if ( dword_4F07590 )
  {
    v73 = dword_4F04C3C;
    dword_4F04C3C = 1;
  }
  if ( v246 )
  {
    sub_8756F0(3, v25, &v258.m128i_i64[1], 0);
    if ( dword_4F077BC )
    {
      v74 = *(_QWORD *)(v59[11].m128i_i64[0] + 88);
      v75 = *(_QWORD *)(v74 + 168);
      if ( (*(_BYTE *)(v75 + 109) & 7) == 0 )
      {
        v262[0].m128i_i8[0] = 0;
        sub_5D0D60(v262, (*(_BYTE *)(v74 + 89) & 4) != 0);
        *(_BYTE *)(v75 + 109) = v262[0].m128i_i8[0] & 7 | *(_BYTE *)(v75 + 109) & 0xF8;
      }
    }
    if ( *(_BYTE *)(v25 + 80) == 19 )
    {
      if ( v59[2].m128i_i64[0] )
      {
        v115 = *(_QWORD **)v4[12].m128i_i64[0];
        v116 = *(_QWORD *)(*(_QWORD *)(v59[11].m128i_i64[0] + 88) + 168LL);
        v117 = *(__int64 **)(v116 + 176);
        v257[0].m128i_i64[0] = (__int64)v117;
        if ( !v117 )
        {
          v117 = *(__int64 **)(v116 + 168);
          v257[0].m128i_i64[0] = (__int64)v117;
        }
        sub_89A1A0(v115, v117, v262, (__int64 **)v257);
        for ( j = v262[0].m128i_i64[0]; v262[0].m128i_i64[0]; j = v262[0].m128i_i64[0] )
        {
          v119 = *(_QWORD *)(j + 8);
          v120 = v257[0].m128i_i64[0];
          v121 = *(_BYTE *)(v119 + 80);
          v122 = *(_QWORD *)(v119 + 88);
          if ( v121 != 3 && v121 != 2 )
            v122 = *(_QWORD *)(v122 + 104);
          *(_QWORD *)(v257[0].m128i_i64[0] + 32) = v122;
          if ( (*(_BYTE *)(v120 + 24) & 0x10) != 0 )
            sub_866610(j, v120);
          sub_89A1C0(v262[0].m128i_i64, (__int64 **)v257);
        }
        goto LABEL_136;
      }
    }
    else
    {
LABEL_136:
      if ( v59[2].m128i_i64[0] )
        goto LABEL_137;
    }
    sub_879080(v59 + 14, 0, v4[12].m128i_i64[0]);
LABEL_137:
    sub_879080(v59, 0, v4[12].m128i_i64[0]);
  }
  else
  {
    sub_8756F0(1, v25, &v258.m128i_i64[1], 0);
    if ( !v59[2].m128i_i64[0] )
    {
      sub_879080(v59 + 14, 0, v4[12].m128i_i64[0]);
      if ( !v59[2].m128i_i64[0] )
        goto LABEL_137;
    }
  }
  if ( v4[1].m128i_i32[3] && v231 && !v4[3].m128i_i32[1] )
    sub_8A6B70(v25, 0);
  if ( *(_BYTE *)(v25 + 80) == 19 && !dword_4D047D0 && !v4[1].m128i_i32[0] )
    *(_BYTE *)(v25 + 83) &= ~0x40u;
  if ( v246 )
  {
    v247 = dword_4F06650[0];
    sub_7ADF70((__int64)v257, 1);
    v111 = 0;
    v112 = 0;
    memset(v262, 0, sizeof(v262));
    v113 = word_4F06418[0];
    v263 = 0;
    v264 = 0;
    v262[4].m128i_i8[11] = 1;
    if ( word_4F06418[0] == 55 )
    {
      sub_7BDB60(1);
      if ( dword_4D04428 )
      {
        switch ( *(_BYTE *)(v25 + 80) )
        {
          case 4:
          case 5:
            v193 = *(_QWORD *)(*(_QWORD *)(v25 + 96) + 80LL);
            break;
          case 6:
            v193 = *(_QWORD *)(*(_QWORD *)(v25 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v193 = *(_QWORD *)(*(_QWORD *)(v25 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v193 = *(_QWORD *)(v25 + 88);
            break;
          default:
            BUG();
        }
        v233 = v193;
        v236 = *(_QWORD *)(v193 + 176);
        v239 = *(_QWORD *)(v236 + 88);
        v194 = sub_892330(v239);
        v111 = (__m128i *)v239;
        ++*(_BYTE *)(qword_4F061C8 + 81LL);
        sub_864700(*(_QWORD *)(v233 + 32), v239, 0, v236, v25, v194, 0, 2u);
        sub_5ED6F0(v25);
        sub_863FE0(v25, v239, v195, v196, v197, v198);
        if ( word_4F06418[0] != 73 )
          sub_7BE180(v25, v239, v216, v199, v200, v201);
        --*(_BYTE *)(qword_4F061C8 + 81LL);
      }
      else
      {
        v111 = v262;
        ++v262[4].m128i_i8[9];
        sub_7C6880(0, (__int64)v262, v165, dword_4D04428, v166, v167);
        --v262[4].m128i_i8[9];
      }
      sub_7BDC00();
      if ( v247 != dword_4F06650[0] )
      {
        v111 = (__m128i *)v247;
        sub_7AE700((__int64)(qword_4F061C0 + 3), v247, dword_4F06650[0], 0, (__int64)v257);
        sub_7AE340((__int64)v257);
      }
      v112 = v262[4].m128i_i8[11] - 1;
      v113 = word_4F06418[0];
    }
    v262[4].m128i_i8[11] = v112;
    v238 = 0;
    if ( v113 == 73 )
    {
      v4[29].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
      sub_7AE360((__int64)v257);
      sub_7BBF40((unsigned __int64)v257, (unsigned int *)v111, v141, v142, v143, v144);
      ++v262[4].m128i_i8[10];
      sub_7C6880((unsigned __int64)v257, (__int64)v262, v145, v146, v147, v148);
      v4[30].m128i_i64[0] = qword_4F063F0;
      if ( word_4F06418[0] == 74 )
      {
        sub_7AE360((__int64)v257);
        v238 = dword_4F06650[0];
        sub_7B8B50((unsigned __int64)v257, (unsigned int *)v262, v204, v205, v206, v207);
      }
      if ( dword_4D043E0 )
      {
        if ( word_4F06418[0] == 142 )
        {
          --v262[4].m128i_i8[10];
          sub_7AE360((__int64)v257);
          sub_7BBF40((unsigned __int64)v257, (unsigned int *)v262, v149, v150, v151, v152);
          sub_7AE360((__int64)v257);
          sub_7BBF40((unsigned __int64)v257, (unsigned int *)v262, v153, v154, v155, v156);
          ++v262[1].m128i_i8[12];
          sub_7C6880((unsigned __int64)v257, (__int64)v262, v157, v158, v159, v160);
          v4[30].m128i_i64[0] = qword_4F063F0;
          if ( word_4F06418[0] == 28 )
          {
            sub_7AE360((__int64)v257);
            v238 = dword_4F06650[0];
            sub_7B8B50((unsigned __int64)v257, (unsigned int *)v262, v161, v162, v163, v164);
          }
        }
      }
    }
    sub_7AE210((__int64)v257);
    if ( v4[2].m128i_i32[3] || v4[3].m128i_i32[0] )
    {
      v114 = v4[15].m128i_i64[0];
      if ( v114 )
      {
        if ( (*(_BYTE *)(v114 + 178) & 4) == 0 && *(_BYTE *)(v25 + 80) == 19 )
          v59[5].m128i_i64[0] = sub_888280(v25, (__int64)v59, v247, v238);
      }
    }
    sub_879080(v59 + 14, v257, 0);
    sub_879080(v59, v257, 0);
  }
  if ( dword_4F07590 )
    dword_4F04C3C = v73;
  *a2 = v25;
  return a2;
}
