// Function: sub_6B4800
// Address: 0x6b4800
//
__int64 __fastcall sub_6B4800(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rbx
  char v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int *v10; // r8
  _BYTE *v11; // rax
  unsigned int v12; // esi
  char v13; // dl
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rdx
  unsigned int v17; // r14d
  char v18; // cl
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // r13d
  int v22; // edx
  __int64 v23; // rax
  char j; // dl
  int v25; // edx
  __int64 v26; // rax
  int v28; // r13d
  unsigned int v29; // r9d
  _BYTE *v30; // rax
  bool v31; // r13
  char v32; // cl
  unsigned int v33; // r9d
  __int64 v34; // rdx
  char v35; // cl
  __int64 v36; // rsi
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // ecx
  __int64 v41; // rdi
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  int v48; // eax
  __m128i *v49; // r10
  char v50; // al
  __int64 v51; // rcx
  __int64 v52; // r8
  bool v53; // al
  char v54; // dl
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 v57; // rdi
  char v58; // cl
  __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rdx
  int v62; // eax
  int v63; // eax
  int v64; // r13d
  int v65; // eax
  int v66; // r11d
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // rdi
  __int64 v72; // rsi
  _BOOL4 v73; // r14d
  int v74; // eax
  __int8 v75; // dl
  int v76; // r8d
  bool v77; // al
  __int64 i; // rax
  __int64 v79; // r14
  int v80; // eax
  __int64 v81; // r8
  int v82; // r13d
  int v83; // eax
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // r13
  __int64 i1; // r8
  __int64 v88; // rax
  __int64 v89; // r8
  __int64 v90; // r14
  __int64 i2; // rsi
  int v92; // eax
  __int64 i3; // r8
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r13
  __int64 v100; // rcx
  __int64 v101; // r8
  int v102; // eax
  __m128i v103; // xmm1
  __m128i v104; // xmm2
  __m128i v105; // xmm3
  __m128i v106; // xmm4
  __m128i v107; // xmm5
  __m128i v108; // xmm6
  __m128i v109; // xmm7
  __m128i v110; // xmm0
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 mm; // r14
  __int64 nn; // rcx
  __int64 n; // rax
  __int64 v117; // rdx
  __int64 ii; // rax
  __int64 v119; // rcx
  __int64 jj; // rax
  __int64 v121; // rsi
  __int64 kk; // rax
  int v123; // eax
  __int64 v124; // r8
  int v125; // r15d
  int v126; // eax
  __int64 v127; // rdi
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rcx
  __int64 v133; // r8
  _OWORD *v134; // rdx
  __m128i *v135; // rdx
  __int64 m; // rax
  __int64 v137; // rax
  __int64 v138; // rax
  bool v139; // cf
  bool v140; // zf
  __int64 v141; // rcx
  char *v142; // rdi
  __int64 v143; // rax
  _QWORD **v144; // rax
  __m128i v145; // xmm2
  __m128i v146; // xmm3
  __m128i v147; // xmm4
  __m128i v148; // xmm5
  __m128i v149; // xmm6
  __m128i v150; // xmm7
  __m128i v151; // xmm1
  __m128i v152; // xmm2
  __m128i v153; // xmm3
  __m128i v154; // xmm4
  __m128i v155; // xmm5
  __m128i v156; // xmm6
  char v157; // dl
  __int64 v158; // rax
  char v159; // dl
  __int64 v160; // rax
  int v161; // eax
  int v162; // eax
  __int64 v163; // rcx
  char v164; // al
  __int64 v165; // rdx
  __int64 v166; // rax
  char v167; // cl
  __int64 v168; // rax
  __int64 v169; // rax
  __int64 v170; // rdx
  __int64 v171; // rcx
  __int64 v172; // r8
  __m128i *v173; // rdx
  __int64 k; // rax
  __int64 i4; // rax
  __int64 v176; // rdx
  __int64 i5; // rax
  unsigned int v178; // [rsp+10h] [rbp-540h]
  int v179; // [rsp+20h] [rbp-530h]
  unsigned __int8 v180; // [rsp+27h] [rbp-529h]
  bool v181; // [rsp+28h] [rbp-528h]
  int v182; // [rsp+28h] [rbp-528h]
  int v183; // [rsp+30h] [rbp-520h]
  _BOOL4 v184; // [rsp+38h] [rbp-518h]
  bool v185; // [rsp+40h] [rbp-510h]
  char v186; // [rsp+48h] [rbp-508h]
  unsigned int v187; // [rsp+48h] [rbp-508h]
  int v188; // [rsp+50h] [rbp-500h]
  char v189; // [rsp+54h] [rbp-4FCh]
  int v190; // [rsp+54h] [rbp-4FCh]
  bool v191; // [rsp+60h] [rbp-4F0h]
  int v192; // [rsp+60h] [rbp-4F0h]
  int v193; // [rsp+60h] [rbp-4F0h]
  char v194; // [rsp+68h] [rbp-4E8h]
  unsigned int v195; // [rsp+68h] [rbp-4E8h]
  __int64 v196; // [rsp+68h] [rbp-4E8h]
  __int64 v197; // [rsp+68h] [rbp-4E8h]
  __int64 v198; // [rsp+68h] [rbp-4E8h]
  int v199; // [rsp+68h] [rbp-4E8h]
  __int64 v200; // [rsp+68h] [rbp-4E8h]
  int v201; // [rsp+78h] [rbp-4D8h]
  unsigned int v202; // [rsp+78h] [rbp-4D8h]
  unsigned int v203; // [rsp+78h] [rbp-4D8h]
  unsigned int v204; // [rsp+80h] [rbp-4D0h] BYREF
  int v205; // [rsp+84h] [rbp-4CCh] BYREF
  int v206; // [rsp+88h] [rbp-4C8h] BYREF
  int v207; // [rsp+8Ch] [rbp-4C4h] BYREF
  __int64 v208; // [rsp+90h] [rbp-4C0h] BYREF
  __int64 v209; // [rsp+98h] [rbp-4B8h] BYREF
  _QWORD v210[6]; // [rsp+A0h] [rbp-4B0h] BYREF
  _QWORD v211[6]; // [rsp+D0h] [rbp-480h] BYREF
  _BYTE v212[352]; // [rsp+100h] [rbp-450h] BYREF
  __m128i v213; // [rsp+260h] [rbp-2F0h] BYREF
  __m128i v214; // [rsp+270h] [rbp-2E0h] BYREF
  __m128i v215; // [rsp+280h] [rbp-2D0h] BYREF
  __m128i v216; // [rsp+290h] [rbp-2C0h] BYREF
  __m128i v217; // [rsp+2A0h] [rbp-2B0h] BYREF
  __m128i v218; // [rsp+2B0h] [rbp-2A0h] BYREF
  __m128i v219; // [rsp+2C0h] [rbp-290h] BYREF
  __m128i v220; // [rsp+2D0h] [rbp-280h] BYREF
  __m128i v221; // [rsp+2E0h] [rbp-270h] BYREF
  __m128i v222; // [rsp+2F0h] [rbp-260h] BYREF
  __m128i v223; // [rsp+300h] [rbp-250h] BYREF
  __m128i v224; // [rsp+310h] [rbp-240h] BYREF
  __m128i v225; // [rsp+320h] [rbp-230h] BYREF
  __m128i v226; // [rsp+330h] [rbp-220h] BYREF
  __m128i v227; // [rsp+340h] [rbp-210h] BYREF
  __m128i v228; // [rsp+350h] [rbp-200h] BYREF
  __m128i v229; // [rsp+360h] [rbp-1F0h] BYREF
  __m128i v230; // [rsp+370h] [rbp-1E0h] BYREF
  __m128i v231; // [rsp+380h] [rbp-1D0h] BYREF
  __m128i v232; // [rsp+390h] [rbp-1C0h] BYREF
  __m128i v233; // [rsp+3A0h] [rbp-1B0h] BYREF
  __m128i v234; // [rsp+3B0h] [rbp-1A0h] BYREF
  __m128i v235; // [rsp+3C0h] [rbp-190h] BYREF
  __m128i v236; // [rsp+3D0h] [rbp-180h]
  __m128i v237; // [rsp+3E0h] [rbp-170h]
  __m128i v238; // [rsp+3F0h] [rbp-160h]
  __m256i v239; // [rsp+400h] [rbp-150h]
  __m128i v240; // [rsp+420h] [rbp-130h]
  __m128i v241; // [rsp+430h] [rbp-120h]
  __m128i v242; // [rsp+440h] [rbp-110h]
  _OWORD v243[16]; // [rsp+450h] [rbp-100h] BYREF

  v205 = 0;
  v4 = *(unsigned __int8 *)(qword_4D03C50 + 17LL);
  v5 = *(_BYTE *)(qword_4D03C50 + 17LL) & 1;
  v189 = v4 & 1;
  v194 = *(_BYTE *)(qword_4D03C50 + 17LL) & 1;
  v181 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) != 0;
  if ( a2 )
  {
    v6 = v212;
    v188 = (*(_BYTE *)(*(_QWORD *)a2 + 59LL) & 0x10) != 0;
    sub_6F8AB0(
      a2,
      (unsigned int)v212,
      (unsigned int)&v213,
      (unsigned int)&v235,
      (unsigned int)&v209,
      (unsigned int)&v204,
      (__int64)&v208);
  }
  else
  {
    v6 = a1;
    v209 = *(_QWORD *)&dword_4F063F8;
    v204 = dword_4F06650[0];
    sub_7B8B50(a1, v4 & 1, v4, v5);
    v188 = HIDWORD(qword_4F077B4);
    if ( !HIDWORD(qword_4F077B4) )
    {
      v7 = 1;
      goto LABEL_5;
    }
    if ( word_4F06418[0] != 55 )
    {
      v7 = 1;
      v188 = 0;
      goto LABEL_4;
    }
    v73 = dword_4F077C4 == 2 && sub_8D3A70(*a1) != 0;
    if ( (unsigned int)sub_8D3410(*a1) )
      sub_6FB570(a1);
    sub_6FF9F0(a1, &v213, v73, &v235, 1);
    v188 = 1;
  }
  v7 = 0;
  if ( !HIDWORD(qword_4F077B4) )
  {
LABEL_5:
    sub_688FA0(v6);
    v180 = 103;
    v201 = 0;
    goto LABEL_6;
  }
LABEL_4:
  if ( dword_4F077C4 != 2 || !(unsigned int)sub_8D2B80(*v6) )
    goto LABEL_5;
  for ( i = *v6; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v79 = *(_QWORD *)(i + 160);
  if ( (unsigned int)sub_8D2930(v79) || (unsigned int)sub_8D3D40(v79) )
  {
    sub_6F69D0(v6, 0);
    v180 = 104;
    v201 = 0;
  }
  else
  {
    sub_6E6890(1695, v6);
    v180 = 104;
    v201 = 1;
  }
LABEL_6:
  sub_6E16F0(v6[11], 0);
  sub_6E17F0(v6);
  v186 = v194;
  if ( *((_BYTE *)v6 + 16) == 2 )
  {
    if ( (unsigned int)sub_70FCE0(v6 + 18) )
    {
      v186 = 0;
      if ( !(unsigned int)sub_6E9820(v6) )
      {
        v50 = v194;
        v194 = 0;
        v186 = v50;
      }
    }
  }
  if ( v7 )
  {
    v11 = (_BYTE *)qword_4D03C50;
    v12 = word_4D04898;
    v13 = *(_BYTE *)(qword_4D03C50 + 17LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    v191 = v12;
    v11[17] = v186 & 1 | v13 & 0xFE;
    if ( v12 )
    {
      v191 = (v11[19] & 0x20) != 0;
      v11[19] &= ~0x20u;
    }
    v11[18] |= 0x10u;
    sub_69ED20((__int64)&v213, 0, 0, 0);
    v14 = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v181) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
    v15 = word_4D04898;
    *(_BYTE *)(v14 + 17) = v189 | *(_BYTE *)(v14 + 17) & 0xFE;
    v185 = v15;
    if ( v15 )
    {
      v185 = (*(_BYTE *)(v14 + 19) & 0x20) != 0;
      *(_BYTE *)(v14 + 19) = (32 * v191) | *(_BYTE *)(v14 + 19) & 0xDF;
    }
    --*(_QWORD *)(v14 + 40);
  }
  else
  {
    if ( a2 )
    {
      v185 = 0;
      v191 = 0;
      goto LABEL_24;
    }
    v191 = 0;
    v185 = 0;
  }
  v208 = *(_QWORD *)&dword_4F063F8;
  if ( (unsigned int)sub_7BE280(55, 53, 0, 0) )
  {
    v16 = qword_4D03C50;
    v17 = word_4D04898;
    *(_BYTE *)(qword_4D03C50 + 17LL) = v194 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFE;
    if ( v17 )
    {
      v18 = *(_BYTE *)(v16 + 19);
      v191 = (v18 & 0x20) != 0;
      *(_BYTE *)(v16 + 19) = v18 & 0xDF;
    }
    *(_BYTE *)(v16 + 18) |= 0x10u;
    v19 = 3;
    if ( dword_4F077C4 == 2 )
      v19 = 2 - ((unsigned int)(qword_4D0495C == 0) - 1);
    sub_69ED20((__int64)&v235, 0, v19, 0);
    v20 = qword_4D03C50;
    v9 = 16 * (unsigned int)v181;
    *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v181) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
    v10 = &word_4D04898;
    v21 = word_4D04898;
    v8 = *(_BYTE *)(v20 + 17) & 0xFE;
    LOBYTE(v8) = v189 | *(_BYTE *)(v20 + 17) & 0xFE;
    *(_BYTE *)(v20 + 17) = v8;
    if ( v21 )
    {
      v22 = *(unsigned __int8 *)(v20 + 19);
      if ( (v22 & 0x20) != 0 && v185 )
      {
        v8 = v22 | 0x20u;
        *(_BYTE *)(v20 + 19) = v8;
        v185 = 1;
      }
      else
      {
        v9 = 32 * (unsigned int)v191;
        v8 = (unsigned int)v9 | *(_BYTE *)(v20 + 19) & 0xDF;
        *(_BYTE *)(v20 + 19) = (32 * v191) | *(_BYTE *)(v20 + 19) & 0xDF;
      }
    }
LABEL_24:
    if ( dword_4F077C4 != 2 )
    {
      v184 = 0;
      v183 = v205 | v201;
      if ( v205 | v201 )
      {
        v183 = 0;
LABEL_27:
        if ( v201 )
          goto LABEL_33;
        v182 = 0;
LABEL_29:
        if ( *((_BYTE *)v6 + 16) )
        {
          v23 = *v6;
          for ( j = *(_BYTE *)(*v6 + 140LL); j == 12; j = *(_BYTE *)(v23 + 140) )
            v23 = *(_QWORD *)(v23 + 160);
          if ( j )
          {
            if ( !v205 )
            {
              sub_6FFD30(
                (_DWORD)v6,
                (unsigned int)&v213,
                (unsigned int)&v235,
                v210[0],
                v201,
                v183,
                0,
                0,
                v188,
                (__int64)&v209,
                (__int64)&v208,
                a3);
              if ( v201 )
              {
                v59 = v218.m128i_i64[1];
                if ( v218.m128i_i64[1] )
                {
                  if ( v239.m256i_i64[3] )
                  {
                    v60 = v218.m128i_i64[1];
                    do
                    {
                      v61 = v60;
                      v60 = *(_QWORD *)(v60 + 48);
                    }
                    while ( v60 );
                    *(_QWORD *)(v61 + 48) = v239.m256i_i64[3];
                  }
                }
                else
                {
                  v59 = v239.m256i_i64[3];
                }
                *(_QWORD *)(a3 + 88) = v59;
                if ( v182 )
                  sub_6ED1A0(a3);
              }
            }
            goto LABEL_34;
          }
        }
LABEL_33:
        sub_6E6260(a3);
LABEL_34:
        if ( unk_4D047E0 && !(dword_4F077BC | dword_4D04964) )
          *(_BYTE *)(a3 + 19) = (v236.m128i_i8[3] | v214.m128i_i8[3]) & 0x10 | *(_BYTE *)(a3 + 19) & 0xEF;
        if ( v184 )
          *(_BYTE *)(a3 + 19) |= 0x20u;
        goto LABEL_39;
      }
      v28 = 0;
LABEL_41:
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
      {
LABEL_43:
        if ( v214.m128i_i8[0] == 1 && *(_BYTE *)(v222.m128i_i64[0] + 24) == 8 )
        {
          if ( v236.m128i_i8[0] == 1 && *(_BYTE *)(*(_QWORD *)&v243[0] + 24LL) == 8 )
          {
            v29 = 0;
            v30 = (_BYTE *)qword_4D03C50;
            v31 = (*(_BYTE *)(qword_4D03C50 + 17LL) & 4) != 0;
            goto LABEL_49;
          }
          goto LABEL_158;
        }
        if ( v236.m128i_i8[0] == 1 && *(_BYTE *)(*(_QWORD *)&v243[0] + 24LL) == 8 )
        {
          if ( v214.m128i_i8[0] != 1 || *(_BYTE *)(v222.m128i_i64[0] + 24) != 8 )
          {
            v36 = 6;
            sub_6F69D0(&v213, 6);
            v80 = sub_6ED0A0(&v213);
            v75 = v214.m128i_i8[1];
            v182 = v80;
            if ( v214.m128i_i8[1] == 1 )
            {
              v201 = 1;
              if ( !(unsigned int)sub_6ED0A0(&v213) )
                goto LABEL_175;
              v75 = v214.m128i_i8[1];
            }
            goto LABEL_173;
          }
LABEL_158:
          v36 = 6;
          sub_6F69D0(&v235, 6);
          v74 = sub_6ED0A0(&v235);
          v75 = v236.m128i_i8[1];
          v182 = v74;
          if ( v236.m128i_i8[1] == 1 )
          {
            v76 = sub_6ED0A0(&v235);
            v77 = 1;
            if ( !v76 )
            {
LABEL_174:
              v201 = v77;
LABEL_175:
              v41 = v213.m128i_i64[0];
              v210[0] = v213.m128i_i64[0];
              if ( dword_4F077C4 != 2 || !v28 )
              {
LABEL_64:
                if ( v214.m128i_i8[0] == 1 && *(_BYTE *)(v222.m128i_i64[0] + 24) == 8 )
                {
                  v210[0] = v235.m128i_i64[0];
                  goto LABEL_29;
                }
                if ( v236.m128i_i8[0] == 1 && *(_BYTE *)(*(_QWORD *)&v243[0] + 24LL) == 8 )
                  goto LABEL_29;
                if ( dword_4F077C0 )
                {
                  if ( (unsigned int)sub_8D2600(v41)
                    || (v41 = v235.m128i_i64[0], (unsigned int)sub_8D2600(v235.m128i_i64[0])) )
                  {
                    v210[0] = sub_72CBE0(v41, v36, v43, v44, v45, v46);
                    if ( !(unsigned int)sub_8D2600(v213.m128i_i64[0]) )
                      sub_6F7220(&v213, v210[0]);
                    if ( !(unsigned int)sub_8D2600(v235.m128i_i64[0]) )
                      sub_6F7220(&v235, v210[0]);
                    goto LABEL_29;
                  }
                  v41 = v213.m128i_i64[0];
                }
                v64 = sub_8D2E30(v41);
                v65 = sub_8D2E30(v235.m128i_i64[0]);
                v66 = v65;
                if ( dword_4F077C4 == 2 )
                {
                  v193 = v65;
                  v199 = sub_8D3D10(v213.m128i_i64[0]);
                  v125 = sub_8D3D10(v235.m128i_i64[0]);
                  v190 = sub_8D2660(v213.m128i_i64[0]);
                  v126 = sub_8D2660(v235.m128i_i64[0]);
                  v66 = v193;
                  if ( !(v193 | v64) )
                  {
                    if ( v125 | v199 )
                    {
                      if ( !(unsigned int)sub_6FC4F0(&v213, &v235, &v208, v211) )
                        goto LABEL_33;
                      v99 = v211[0];
                      v210[0] = v211[0];
                      goto LABEL_234;
                    }
                    if ( v126 | v190 )
                    {
                      if ( !(unsigned int)sub_6E8FF0(&v213, &v235, &v208, v210) )
                        goto LABEL_33;
                      goto LABEL_233;
                    }
LABEL_141:
                    if ( !HIDWORD(qword_4F077B4) )
                      goto LABEL_142;
                    if ( (unsigned int)sub_8D2B80(v213.m128i_i64[0]) )
                    {
                      v99 = v235.m128i_i64[0];
                      if ( v213.m128i_i64[0] == v235.m128i_i64[0] )
                      {
LABEL_291:
                        while ( *(_BYTE *)(v99 + 140) == 12 )
                          v99 = *(_QWORD *)(v99 + 160);
                        v210[0] = v99;
                        goto LABEL_234;
                      }
                      if ( (unsigned int)sub_8D97D0(v213.m128i_i64[0], v235.m128i_i64[0], 0, v132, v133) )
                      {
                        v99 = v213.m128i_i64[0];
                        goto LABEL_291;
                      }
                    }
                    if ( HIDWORD(qword_4F077B4) && dword_4F077C4 == 2 && v180 == 104 )
                    {
                      if ( (unsigned int)sub_8D2B80(v213.m128i_i64[0]) )
                      {
                        v134 = v236.m128i_i8[0] == 2 ? v243 : 0LL;
                        if ( (unsigned int)sub_6E8F10(v213.m128i_i64[0], v235.m128i_i64[0], v134, 1) )
                        {
                          sub_6FD210(&v235, v213.m128i_i64[0]);
                          for ( k = v213.m128i_i64[0]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                            ;
                          v210[0] = k;
                          goto LABEL_309;
                        }
                      }
                      if ( HIDWORD(qword_4F077B4) && dword_4F077C4 == 2 && (unsigned int)sub_8D2B80(v235.m128i_i64[0]) )
                      {
                        v135 = &v222;
                        if ( v214.m128i_i8[0] != 2 )
                          v135 = 0;
                        if ( (unsigned int)sub_6E8F10(v235.m128i_i64[0], v213.m128i_i64[0], v135, 1) )
                        {
                          sub_6FD210(&v213, v235.m128i_i64[0]);
                          for ( m = v235.m128i_i64[0]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                            ;
                          v210[0] = m;
LABEL_309:
                          sub_6FC7D0(v210[0], &v213, &v235, 104);
LABEL_235:
                          if ( !(unsigned int)sub_8D2B80(v210[0]) )
                          {
                            v169 = sub_8D4620(*v6);
                            v210[0] = sub_72B5A0(v210[0], v169, 0);
                            sub_6FD210(&v213, v210[0]);
                            sub_6FD210(&v235, v210[0]);
                          }
                          for ( n = *v6; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
                            ;
                          v117 = *(_QWORD *)(n + 160);
                          for ( ii = v210[0]; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
                            ;
                          v119 = *(_QWORD *)(ii + 160);
                          for ( jj = v117; *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
                            ;
                          v121 = *(_QWORD *)(jj + 128);
                          for ( kk = v119; *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
                            ;
                          if ( v121 != *(_QWORD *)(kk + 128) )
                            sub_6861A0(0xCBCu, &v209, v117, v119);
                          goto LABEL_29;
                        }
                      }
                    }
LABEL_142:
                    if ( !(unsigned int)sub_8D2D80(v213.m128i_i64[0]) )
                    {
                      if ( !(unsigned int)sub_8D3A70(v213.m128i_i64[0]) && !(unsigned int)sub_8D2600(v213.m128i_i64[0]) )
                      {
                        v157 = *(_BYTE *)(v213.m128i_i64[0] + 140);
                        if ( v157 == 12 )
                        {
                          v158 = v213.m128i_i64[0];
                          do
                          {
                            v158 = *(_QWORD *)(v158 + 160);
                            v157 = *(_BYTE *)(v158 + 140);
                          }
                          while ( v157 == 12 );
                        }
                        if ( !v157 )
                          goto LABEL_33;
                        v159 = *(_BYTE *)(v235.m128i_i64[0] + 140);
                        if ( v159 == 12 )
                        {
                          v160 = v235.m128i_i64[0];
                          do
                          {
                            v160 = *(_QWORD *)(v160 + 160);
                            v159 = *(_BYTE *)(v160 + 140);
                          }
                          while ( v159 == 12 );
                        }
                        if ( !v159 )
                          goto LABEL_33;
                        goto LABEL_338;
                      }
                      v71 = v213.m128i_i64[0];
                      v72 = v235.m128i_i64[0];
                      if ( v213.m128i_i64[0] != v235.m128i_i64[0]
                        && !(unsigned int)sub_8DED30(v213.m128i_i64[0], v235.m128i_i64[0], 1) )
                      {
LABEL_338:
                        sub_6E5ED0(42, &v208, v213.m128i_i64[0], v235.m128i_i64[0]);
                        goto LABEL_33;
                      }
                      if ( dword_4F077C4 == 2 )
                      {
                        sub_6E6000(v71, v72, v67, v68, v69, v70);
                        goto LABEL_33;
                      }
                      v99 = v210[0];
                      goto LABEL_234;
                    }
                    sub_6E9580(&v235);
                    v210[0] = sub_6E8B10(&v213, &v235, v111, v112, v113);
                    v99 = v210[0];
                    if ( dword_4F077C4 == 2 )
                      goto LABEL_234;
                    for ( mm = v213.m128i_i64[0]; *(_BYTE *)(mm + 140) == 12; mm = *(_QWORD *)(mm + 160) )
                      ;
                    for ( nn = v235.m128i_i64[0]; *(_BYTE *)(nn + 140) == 12; nn = *(_QWORD *)(nn + 160) )
                      ;
                    v198 = nn;
                    if ( !(unsigned int)sub_8D2930(mm) )
                      goto LABEL_233;
                    v162 = sub_8D2930(v198);
                    v163 = v198;
                    if ( !v162 )
                      goto LABEL_233;
                    v164 = *(_BYTE *)(v198 + 161);
                    if ( (*(_BYTE *)(mm + 161) & 8) != 0 )
                    {
                      if ( (v164 & 8) != 0 )
                      {
                        v99 = v210[0];
                        v165 = mm;
                        if ( v198 != mm )
                          goto LABEL_352;
                        goto LABEL_357;
                      }
                      v165 = mm;
                    }
                    else
                    {
                      v165 = *(_QWORD *)(mm + 168);
                      if ( (v164 & 8) != 0 )
                        goto LABEL_349;
                    }
                    v163 = *(_QWORD *)(v198 + 168);
LABEL_349:
                    if ( v165 )
                    {
                      v99 = v210[0];
                      if ( v165 != v163 )
                      {
                        if ( !v163 )
                          goto LABEL_234;
LABEL_352:
                        if ( !dword_4F07588 )
                          goto LABEL_234;
                        v166 = *(_QWORD *)(v165 + 32);
                        if ( *(_QWORD *)(v163 + 32) != v166 || !v166 )
                          goto LABEL_234;
                      }
LABEL_357:
                      while ( *(_BYTE *)(v99 + 140) == 12 )
                        v99 = *(_QWORD *)(v99 + 160);
                      v167 = *(_BYTE *)(v99 + 160);
                      if ( v167 == *(_BYTE *)(mm + 160) )
                      {
                        v210[0] = mm;
                        v99 = mm;
                      }
                      else
                      {
                        if ( (**(_BYTE **)(v165 + 176) & 1) != 0 )
                        {
                          v168 = *(_QWORD *)(v165 + 168);
                          if ( (*(_BYTE *)(v165 + 161) & 0x10) != 0 )
                            v168 = *(_QWORD *)(v168 + 96);
                          if ( v168 )
                          {
                            v99 = *(_QWORD *)(v168 + 128);
                            if ( *(_BYTE *)(v99 + 140) == 2 && v167 == *(_BYTE *)(v99 + 160) )
                              goto LABEL_215;
                          }
                        }
                        v200 = v165;
                        v99 = sub_7259C0(2);
                        sub_73C230(v210[0], v99);
                        *(_BYTE *)(v99 + 161) &= ~8u;
                        *(_QWORD *)(v99 + 168) = v200;
                        v210[0] = v99;
                      }
LABEL_234:
                      sub_6FC7D0(v99, &v213, &v235, v180);
                      if ( v180 != 104 )
                        goto LABEL_29;
                      goto LABEL_235;
                    }
LABEL_233:
                    v99 = v210[0];
                    goto LABEL_234;
                  }
                }
                else if ( !(v65 | v64) )
                {
                  goto LABEL_141;
                }
                v192 = v66;
                sub_6E6B60(&v213, 0);
                sub_6E6B60(&v235, 0);
                if ( !(unsigned int)sub_6EB6C0(
                                      (unsigned int)&v213,
                                      (unsigned int)&v235,
                                      (unsigned int)&v208,
                                      44,
                                      1,
                                      1,
                                      1,
                                      1,
                                      (__int64)v211) )
                  goto LABEL_33;
                if ( !v64 || !v192 )
                {
                  v99 = v211[0];
                  v210[0] = v211[0];
                  goto LABEL_234;
                }
                v86 = sub_8D46C0(v213.m128i_i64[0]);
                for ( i1 = v86; *(_BYTE *)(i1 + 140) == 12; i1 = *(_QWORD *)(i1 + 160) )
                  ;
                v196 = i1;
                v88 = sub_8D46C0(v235.m128i_i64[0]);
                v89 = v196;
                v90 = v88;
                for ( i2 = v88; *(_BYTE *)(i2 + 140) == 12; i2 = *(_QWORD *)(i2 + 160) )
                  ;
                if ( v196 == i2 || (v92 = sub_8DED30(v196, i2, 1), v89 = v196, v92) )
                {
                  i3 = sub_8D79B0(v89, i2);
                }
                else
                {
                  for ( i3 = sub_8D46C0(v211[0]); *(_BYTE *)(i3 + 140) == 12; i3 = *(_QWORD *)(i3 + 160) )
                    ;
                }
                v197 = i3;
                sub_8D46C0(v211[0]);
                v94 = sub_73CA70(v197, v86);
                v95 = sub_73CA70(v94, v90);
                v96 = sub_72D2E0(v95, 0);
                v99 = v213.m128i_i64[0];
                v210[0] = v96;
                if ( v96 == v213.m128i_i64[0] )
                  goto LABEL_215;
                if ( (unsigned int)sub_8D97D0(v96, v213.m128i_i64[0], 0, v97, v98) )
                {
                  v99 = v213.m128i_i64[0];
                }
                else
                {
                  v99 = v210[0];
                  if ( v235.m128i_i64[0] != v210[0] )
                  {
                    if ( !(unsigned int)sub_8D97D0(v210[0], v235.m128i_i64[0], 0, v100, v101) )
                      goto LABEL_233;
                    v99 = v235.m128i_i64[0];
                  }
                }
LABEL_215:
                v210[0] = v99;
                goto LABEL_234;
              }
              if ( !v201 )
                goto LABEL_103;
LABEL_178:
              v201 = 1;
              if ( v214.m128i_i8[1] == 3 || v236.m128i_i8[1] == 3 )
                goto LABEL_103;
              if ( (*(_BYTE *)(v41 + 140) & 0xFB) == 8 )
              {
                v161 = sub_8D4C10(v41, 0);
                v81 = v235.m128i_i64[0];
                v82 = v161;
                if ( (*(_BYTE *)(v235.m128i_i64[0] + 140) & 0xFB) != 8 )
                {
                  v83 = 0;
                  goto LABEL_184;
                }
              }
              else
              {
                v81 = v235.m128i_i64[0];
                if ( (*(_BYTE *)(v235.m128i_i64[0] + 140) & 0xFB) != 8 )
                  goto LABEL_341;
                v82 = 0;
              }
              v83 = sub_8D4C10(v81, dword_4F077C4 != 2);
LABEL_184:
              if ( v83 != v82 )
              {
                if ( (v82 & ~v83) != 0 )
                {
                  v84 = v213.m128i_i64[0];
                  if ( (v83 & ~v82) != 0 )
                  {
                    while ( *(_BYTE *)(v84 + 140) == 12 )
                      v84 = *(_QWORD *)(v84 + 160);
                    v210[0] = v84;
                    v201 = 0;
                    goto LABEL_103;
                  }
                }
                else
                {
                  v84 = v235.m128i_i64[0];
                }
                v210[0] = v84;
LABEL_188:
                sub_6F7690(&v213);
                sub_6F7690(&v235);
                v201 = 1;
                goto LABEL_103;
              }
              v41 = v213.m128i_i64[0];
LABEL_341:
              v210[0] = v41;
              goto LABEL_188;
            }
            v75 = v236.m128i_i8[1];
          }
LABEL_173:
          v77 = v75 == 3 || v182 != 0;
          goto LABEL_174;
        }
LABEL_45:
        v29 = 0;
        v30 = (_BYTE *)qword_4D03C50;
        v31 = (*(_BYTE *)(qword_4D03C50 + 17LL) & 4) != 0;
        if ( !v183 && (v214.m128i_i8[0] != 1 || *(_BYTE *)(v222.m128i_i64[0] + 24) != 8) )
        {
          v29 = 128;
          if ( v236.m128i_i8[0] == 1 )
            v29 = (*(_BYTE *)(*(_QWORD *)&v243[0] + 24LL) != 8) << 7;
        }
LABEL_49:
        v32 = v186 & 1;
        if ( word_4D04898 )
        {
          if ( v30[16] <= 3u && *((_BYTE *)v6 + 16) == 2 && *((_BYTE *)v6 + 317) == 12 )
            v30[17] |= 4u;
          v30[17] = v32 | v30[17] & 0xFE;
          v58 = v30[19];
          v191 = (v58 & 0x20) != 0;
          v30[19] = v58 & 0xDF;
        }
        else
        {
          v30[17] = v32 | v30[17] & 0xFE;
        }
        if ( (*(_BYTE *)(v213.m128i_i64[0] + 140) & 0xFB) == 8 )
        {
          v203 = v29;
          v178 = sub_8D4C10(v213.m128i_i64[0], dword_4F077C4 != 2);
          sub_6F69D0(&v213, v203);
          v33 = v203;
          if ( dword_4F077C4 == 2 && v178 && (unsigned __int8)(*(_BYTE *)(v213.m128i_i64[0] + 140) - 9) <= 2u )
          {
            v85 = sub_73C570(v213.m128i_i64[0], v178, -1);
            sub_6F7980(&v213, v85);
            v33 = v203;
          }
        }
        else
        {
          v187 = v29;
          sub_6F69D0(&v213, v29);
          v33 = v187;
        }
        v34 = qword_4D03C50;
        if ( word_4D04898 )
        {
          v35 = *(_BYTE *)(qword_4D03C50 + 19LL);
          v185 = (v35 & 0x20) != 0;
          *(_BYTE *)(qword_4D03C50 + 19LL) = v35 & 0xDF;
        }
        *(_BYTE *)(v34 + 17) = v194 | *(_BYTE *)(v34 + 17) & 0xFE;
        if ( (*(_BYTE *)(v235.m128i_i64[0] + 140) & 0xFB) == 8 )
        {
          v195 = v33;
          v202 = sub_8D4C10(v235.m128i_i64[0], dword_4F077C4 != 2);
          v36 = v195;
          sub_6F69D0(&v235, v195);
          v37 = dword_4F077C4;
          if ( v202 && dword_4F077C4 == 2 )
          {
            if ( (unsigned __int8)(*(_BYTE *)(v235.m128i_i64[0] + 140) - 9) > 2u )
            {
              v38 = qword_4D03C50;
              v39 = *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFA;
              *(_BYTE *)(qword_4D03C50 + 17LL) = v189 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFA | (4 * v31);
              if ( !word_4D04898 )
              {
                v42 = v213.m128i_i64[0];
LABEL_98:
                while ( *(_BYTE *)(v42 + 140) == 12 )
                  v42 = *(_QWORD *)(v42 + 160);
                v36 = (__int64)&v235;
                v201 = sub_688610(v213.m128i_i64, v235.m128i_i64, v38, v39, (__int64)&word_4D04898);
                if ( v201 )
                {
                  v41 = v213.m128i_i64[0];
                  v53 = 1;
                }
                else
                {
                  v41 = v213.m128i_i64[0];
                  if ( (unsigned __int8)(*(_BYTE *)(v42 + 140) - 9) > 2u )
                  {
                    v210[0] = v213.m128i_i64[0];
                    v182 = 0;
                    goto LABEL_64;
                  }
                  v36 = v235.m128i_i64[0];
                  v53 = 1;
                  if ( v235.m128i_i64[0] != v213.m128i_i64[0] )
                  {
                    v102 = sub_8D97D0(v213.m128i_i64[0], v235.m128i_i64[0], 32, v51, v52);
                    v41 = v213.m128i_i64[0];
                    v53 = v102 != 0;
                  }
                }
                v210[0] = v41;
                if ( dword_4F077C4 != 2 )
                  goto LABEL_63;
                v182 = 0;
                v201 = 0;
                if ( !v53 )
                  goto LABEL_63;
LABEL_103:
                v54 = *(_BYTE *)(v235.m128i_i64[0] + 140);
                if ( v54 == 12 )
                {
                  v55 = v235.m128i_i64[0];
                  do
                  {
                    v55 = *(_QWORD *)(v55 + 160);
                    v54 = *(_BYTE *)(v55 + 140);
                  }
                  while ( v54 == 12 );
                }
                if ( !v54 )
                {
                  v210[0] = v235.m128i_i64[0];
                  goto LABEL_29;
                }
                if ( !HIDWORD(qword_4F077B4) || dword_4F077C4 != 2 )
                  goto LABEL_29;
                if ( (unsigned int)sub_8D2B80(*v6) && !(unsigned int)sub_8D2B80(v213.m128i_i64[0]) )
                {
                  v173 = v214.m128i_i8[0] == 2 ? &v222 : 0LL;
                  if ( (unsigned int)sub_6E8F10(*v6, v213.m128i_i64[0], v173, 0) )
                  {
                    sub_6FD210(&v213, *v6);
                    sub_6FD210(&v235, *v6);
                    v210[0] = *v6;
                    goto LABEL_29;
                  }
                }
                if ( !HIDWORD(qword_4F077B4)
                  || dword_4F077C4 != 2
                  || !(unsigned int)sub_8D2B80(*v6)
                  || !(unsigned int)sub_8D2B80(v213.m128i_i64[0]) )
                {
                  goto LABEL_29;
                }
                v56 = sub_8D4620(*v6);
                if ( v56 == sub_8D4620(v213.m128i_i64[0]) )
                {
                  for ( i4 = *v6; *(_BYTE *)(i4 + 140) == 12; i4 = *(_QWORD *)(i4 + 160) )
                    ;
                  do
                    i4 = *(_QWORD *)(i4 + 160);
                  while ( *(_BYTE *)(i4 + 140) == 12 );
                  v57 = v213.m128i_i64[0];
                  v176 = *(_QWORD *)(i4 + 128);
                  for ( i5 = v213.m128i_i64[0]; *(_BYTE *)(i5 + 140) == 12; i5 = *(_QWORD *)(i5 + 160) )
                    ;
                  do
                    i5 = *(_QWORD *)(i5 + 160);
                  while ( *(_BYTE *)(i5 + 140) == 12 );
                  if ( v176 == *(_QWORD *)(i5 + 128) )
                    goto LABEL_29;
                }
                else
                {
                  v57 = v213.m128i_i64[0];
                }
                sub_6E5ED0(42, &v209, *v6, v57);
                goto LABEL_33;
              }
              v37 = 2;
LABEL_58:
              v40 = *(unsigned __int8 *)(v38 + 19);
              if ( (v40 & 0x20) != 0 && v185 )
              {
                v39 = v40 | 0x20u;
                *(_BYTE *)(v38 + 19) = v39;
              }
              else
              {
                v36 = 32 * (unsigned int)v191;
                v39 = (unsigned int)v36 | *(_BYTE *)(v38 + 19) & 0xDF;
                *(_BYTE *)(v38 + 19) = (32 * v191) | *(_BYTE *)(v38 + 19) & 0xDF;
              }
LABEL_61:
              v41 = v213.m128i_i64[0];
              v42 = v213.m128i_i64[0];
              if ( v37 != 2 )
              {
                v210[0] = v213.m128i_i64[0];
LABEL_63:
                v182 = 0;
                v201 = 0;
                goto LABEL_64;
              }
              goto LABEL_98;
            }
            v36 = sub_73C570(v235.m128i_i64[0], v202, -1);
            sub_6F7980(&v235, v36);
            v37 = dword_4F077C4;
          }
        }
        else
        {
          v36 = v33;
          sub_6F69D0(&v235, v33);
          v37 = dword_4F077C4;
        }
        v38 = qword_4D03C50;
        v39 = word_4D04898;
        *(_BYTE *)(qword_4D03C50 + 17LL) = v189 | (4 * v31) | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFA;
        if ( !(_DWORD)v39 )
          goto LABEL_61;
        goto LABEL_58;
      }
LABEL_132:
      if ( qword_4F077A8 <= 0x1869Fu
        && (v214.m128i_i8[1] == 1 && !(unsigned int)sub_6ED0A0(&v213)
         || v236.m128i_i8[1] == 1 && !(unsigned int)sub_6ED0A0(&v235)) )
      {
        goto LABEL_45;
      }
      goto LABEL_43;
    }
    v36 = (__int64)&v235;
    v184 = ((v236.m128i_i8[3] | v214.m128i_i8[3]) & 0x30) != 0;
    v28 = sub_688610(v213.m128i_i64, v235.m128i_i64, v8, v9, (__int64)v10);
    if ( dword_4F04C44 != -1
      || (v47 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v47 + 6) & 6) != 0)
      || *(_BYTE *)(v47 + 4) == 12 )
    {
      v48 = sub_8DBE70(*v6);
      LODWORD(v49) = (unsigned int)&v213;
      if ( v48
        || (v62 = sub_8DBE70(v213.m128i_i64[0]), LODWORD(v49) = (unsigned int)&v213, v62)
        || (v63 = sub_8DBE70(v235.m128i_i64[0]), v49 = &v213, v63) )
      {
        sub_700CE0((_DWORD)v6, (_DWORD)v49, (unsigned int)&v235, v188, (unsigned int)&v209, (unsigned int)&v208, a3);
        v205 = 1;
        v183 = 0;
        goto LABEL_27;
      }
    }
    if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000002 )
    {
      v127 = *v6;
      if ( (unsigned int)sub_7306C0(*v6)
        || (v127 = v213.m128i_i64[0], (unsigned int)sub_7306C0(v213.m128i_i64[0]))
        || (v127 = v235.m128i_i64[0], (unsigned int)sub_7306C0(v235.m128i_i64[0])) )
      {
        if ( (unsigned int)sub_6E5430(v127, &v235, v128, v129, v130, v131) )
          sub_6851C0(0x369u, &v208);
        sub_6E6260(a3);
        sub_6E6450(v6);
        sub_6E6450(&v213);
        sub_6E6450(&v235);
        goto LABEL_33;
      }
    }
    v183 = sub_8D2600(v213.m128i_i64[0]);
    if ( v183 )
    {
      v183 = 0;
LABEL_79:
      v182 = v205 | v201;
      if ( v205 | v201 )
        goto LABEL_27;
      if ( dword_4F077C4 != 2 )
        goto LABEL_265;
      if ( !v28 )
        goto LABEL_41;
      if ( v180 == 104 )
      {
LABEL_265:
        if ( !v28 )
          goto LABEL_41;
      }
      else if ( (unsigned int)sub_6ED1E0(&v213) && (unsigned int)sub_6ED1E0(&v235) )
      {
        goto LABEL_90;
      }
      if ( !(unsigned int)sub_6ED0A0(&v213) || !(unsigned int)sub_6ED0A0(&v235) )
        goto LABEL_41;
      if ( dword_4F077BC )
      {
        if ( !(_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A8 <= 0x9FC3u )
            goto LABEL_132;
          v182 = 1;
LABEL_90:
          v41 = v213.m128i_i64[0];
          v210[0] = v213.m128i_i64[0];
          if ( dword_4F077C4 != 2 )
          {
            v201 = 1;
            goto LABEL_64;
          }
          goto LABEL_178;
        }
      }
      else
      {
        v182 = 1;
        if ( !(_DWORD)qword_4F077B4 )
          goto LABEL_90;
      }
      v182 = 1;
      v137 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( *(_BYTE *)(v137 + 4) == 6 )
      {
        v138 = *(_QWORD *)(v137 + 208);
        v36 = *(_QWORD *)(v138 + 8);
        if ( v36 )
        {
          v139 = 0;
          v140 = (*(_BYTE *)(v138 + 89) & 4) == 0;
          if ( (*(_BYTE *)(v138 + 89) & 4) == 0 )
          {
            v141 = 12;
            v142 = "common_type";
            do
            {
              if ( !v141 )
                break;
              v139 = *(_BYTE *)v36 < (unsigned __int8)*v142;
              v140 = *(_BYTE *)v36++ == (unsigned __int8)*v142++;
              --v141;
            }
            while ( v140 );
            if ( (!v139 && !v140) == v139 )
            {
              v143 = *(_QWORD *)(v138 + 40);
              if ( v143 )
              {
                if ( *(_BYTE *)(v143 + 28) == 3 )
                {
                  v144 = *(_QWORD ***)(v143 + 32);
                  if ( v144 )
                  {
                    if ( *v144 == qword_4D049B8 )
                      goto LABEL_43;
                  }
                }
              }
            }
          }
        }
      }
      goto LABEL_90;
    }
    if ( v28 | (unsigned int)sub_8D2600(v235.m128i_i64[0]) )
      goto LABEL_79;
    if ( !(unsigned int)sub_8D3A70(v213.m128i_i64[0]) )
    {
      v28 = sub_8D3A70(v235.m128i_i64[0]);
      if ( !v28 )
        goto LABEL_79;
    }
    if ( dword_4F077BC
      && qword_4F077A8 <= 0x9C3Fu
      && (unsigned int)sub_8D3A70(v213.m128i_i64[0])
      && (unsigned int)sub_8D3A70(v235.m128i_i64[0]) )
    {
      if ( v214.m128i_i8[1] == 1
        && !(unsigned int)sub_6ED0A0(&v213)
        && (v236.m128i_i8[1] == 2 || (unsigned int)sub_6ED0A0(&v235))
        && sub_8D5CE0(v235.m128i_i64[0], v213.m128i_i64[0]) )
      {
        sub_6FA330(&v235, 0);
      }
      else if ( v236.m128i_i8[1] == 1
             && !(unsigned int)sub_6ED0A0(&v235)
             && (v214.m128i_i8[1] == 2 || (unsigned int)sub_6ED0A0(&v213))
             && sub_8D5CE0(v213.m128i_i64[0], v235.m128i_i64[0]) )
      {
        sub_6FA330(&v213, 0);
      }
    }
    v179 = sub_841550(&v213, &v235, v210, &v206);
    v123 = sub_841550(&v235, &v213, v211, &v207);
    v124 = 1;
    v183 = qword_4D0495C | HIDWORD(qword_4D0495C);
    if ( qword_4D0495C )
    {
      v183 = 1;
      v124 = 0;
    }
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x10u;
    if ( v179 && v123 )
    {
      v28 = 0;
      v36 = (__int64)&v208;
      sub_6E5ED0(979, &v208, v213.m128i_i64[0], v235.m128i_i64[0]);
      v201 = 1;
    }
    else
    {
      v28 = v179 | v123;
      if ( v179 | v123 )
      {
        if ( v179 )
        {
          if ( v206 )
            sub_841550(&v213, &v235, v210, 0);
          else
            sub_8449E0(&v213, v235.m128i_i64[0], v210, 0, v124);
        }
        else if ( v207 )
        {
          sub_841550(&v235, &v213, v211, 0);
        }
        else
        {
          sub_8449E0(&v235, v213.m128i_i64[0], v211, 0, v124);
        }
      }
      else
      {
        v36 = 0;
        sub_84EC30(
          44,
          0,
          0,
          1,
          1,
          (unsigned int)&v213,
          (__int64)&v235,
          (__int64)&v209,
          0,
          0,
          (__int64)&v208,
          a3,
          0,
          0,
          (__int64)&v205);
        if ( v205 )
        {
          v201 = 1;
          goto LABEL_262;
        }
      }
      v36 = (__int64)&v235;
      v28 = sub_688610(v213.m128i_i64, v235.m128i_i64, v170, v171, v172);
    }
LABEL_262:
    *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v181) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
    goto LABEL_79;
  }
  sub_6E6260(a3);
  v103 = _mm_loadu_si128(&v214);
  v104 = _mm_loadu_si128(&v215);
  v105 = _mm_loadu_si128(&v216);
  v106 = _mm_loadu_si128(&v217);
  v107 = _mm_loadu_si128(&v218);
  v235 = _mm_loadu_si128(&v213);
  v108 = _mm_loadu_si128(&v219);
  v109 = _mm_loadu_si128(&v220);
  v236 = v103;
  v110 = _mm_loadu_si128(&v221);
  v237 = v104;
  v238 = v105;
  *(__m128i *)v239.m256i_i8 = v106;
  *(__m128i *)&v239.m256i_u64[2] = v107;
  v240 = v108;
  v241 = v109;
  v242 = v110;
  if ( v214.m128i_i8[0] == 2 )
  {
    v145 = _mm_loadu_si128(&v223);
    v146 = _mm_loadu_si128(&v224);
    v147 = _mm_loadu_si128(&v225);
    v148 = _mm_loadu_si128(&v226);
    v149 = _mm_loadu_si128(&v227);
    v243[0] = _mm_loadu_si128(&v222);
    v150 = _mm_loadu_si128(&v228);
    v151 = _mm_loadu_si128(&v229);
    v243[1] = v145;
    v243[2] = v146;
    v152 = _mm_loadu_si128(&v230);
    v153 = _mm_loadu_si128(&v231);
    v243[3] = v147;
    v154 = _mm_loadu_si128(&v232);
    v243[4] = v148;
    v155 = _mm_loadu_si128(&v233);
    v243[5] = v149;
    v156 = _mm_loadu_si128(&v234);
    v243[6] = v150;
    v243[7] = v151;
    v243[8] = v152;
    v243[9] = v153;
    v243[10] = v154;
    v243[11] = v155;
    v243[12] = v156;
  }
  else if ( v214.m128i_i8[0] == 5 || v214.m128i_i8[0] == 1 )
  {
    *(_QWORD *)&v243[0] = v222.m128i_i64[0];
  }
  sub_6E6840(&v235);
LABEL_39:
  v25 = *((_DWORD *)v6 + 17);
  *(_WORD *)(a3 + 72) = *((_WORD *)v6 + 36);
  *(_DWORD *)(a3 + 68) = v25;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v26 = *(__int64 *)((char *)&v239.m256i_i64[1] + 4);
  *(_QWORD *)(a3 + 76) = *(__int64 *)((char *)&v239.m256i_i64[1] + 4);
  unk_4F061D8 = v26;
  return sub_6E3280(a3, &v209);
}
