// Function: sub_6CA0E0
// Address: 0x6ca0e0
//
__int64 __fastcall sub_6CA0E0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        _BYTE *a7,
        __int16 a8)
{
  char *v8; // r15
  _QWORD *v10; // r12
  unsigned __int8 v11; // bl
  _QWORD *v12; // rdx
  char v13; // al
  _BOOL8 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 *v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdi
  __int64 v25; // rdi
  int v26; // ebx
  __int64 j; // rdx
  int v28; // eax
  __int64 v29; // rdx
  __m128i *v30; // rsi
  __int64 v31; // r13
  __int64 v32; // rcx
  char v33; // al
  int v34; // eax
  bool v35; // zf
  bool v36; // al
  _BOOL4 v37; // edx
  __int64 v38; // rsi
  char v39; // al
  __int64 v40; // rdi
  __int64 v41; // rax
  bool v42; // al
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  char v48; // al
  __int16 v49; // ax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 result; // rax
  __int64 v53; // rdi
  char v54; // dl
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // r8
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdi
  __int64 v64; // rax
  _QWORD *v65; // rdi
  __int64 v66; // rdi
  _DWORD *v67; // r13
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // rax
  char v75; // al
  __int64 v76; // rax
  int v77; // eax
  __int64 v78; // rax
  int v79; // r10d
  __int64 v80; // r14
  __int64 v81; // rax
  int v82; // eax
  __int64 v83; // rax
  _QWORD *v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // rdi
  __int64 v87; // rdx
  __int64 v88; // rax
  __m128i v89; // xmm2
  __m128i v90; // xmm3
  __m128i v91; // xmm4
  __m128i v92; // xmm5
  __m128i v93; // xmm6
  __m128i v94; // xmm7
  __m128i v95; // xmm0
  __m128i v96; // xmm1
  __int8 v97; // al
  __m128i v98; // xmm3
  __m128i v99; // xmm4
  __m128i v100; // xmm5
  __m128i v101; // xmm6
  __m128i v102; // xmm7
  __m128i v103; // xmm0
  __m128i v104; // xmm1
  __m128i v105; // xmm2
  __m128i v106; // xmm4
  __m128i v107; // xmm5
  __m128i v108; // xmm6
  __m128i v109; // xmm7
  __m128i v110; // xmm0
  __m128i v111; // xmm3
  __m128i v112; // xmm4
  __m128i v113; // xmm5
  __m128i v114; // xmm6
  __m128i v115; // xmm7
  __m128i v116; // xmm0
  __m128i v117; // xmm3
  __m128i v118; // xmm5
  __m128i v119; // xmm6
  __m128i v120; // xmm7
  __m128i v121; // xmm0
  __m128i v122; // xmm3
  __m128i v123; // xmm1
  __m128i v124; // xmm4
  __m128i v125; // xmm2
  __m128i v126; // xmm5
  __m128i v127; // xmm6
  __m128i v128; // xmm7
  __m128i v129; // xmm0
  __int64 v130; // rdi
  __int64 v131; // rdi
  __int64 v132; // rax
  __int64 v133; // rax
  char i; // dl
  __int64 v135; // rdi
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 v140; // r13
  __int64 v141; // rax
  char v142; // dl
  bool v143; // al
  __int64 v144; // rax
  __int64 v145; // r13
  __int64 v146; // rax
  __int64 v147; // rdx
  __m128i *v148; // rcx
  __int64 v149; // rsi
  int v150; // [rsp+8h] [rbp-6F8h]
  __int64 v151; // [rsp+8h] [rbp-6F8h]
  char v152; // [rsp+18h] [rbp-6E8h]
  int v153; // [rsp+18h] [rbp-6E8h]
  __int64 v154; // [rsp+18h] [rbp-6E8h]
  int v155; // [rsp+20h] [rbp-6E0h]
  _BOOL4 v157; // [rsp+28h] [rbp-6D8h]
  __int64 v158; // [rsp+28h] [rbp-6D8h]
  bool v159; // [rsp+3Ah] [rbp-6C6h]
  bool v160; // [rsp+3Bh] [rbp-6C5h]
  unsigned int v161; // [rsp+3Ch] [rbp-6C4h]
  __int64 v162; // [rsp+40h] [rbp-6C0h]
  __int64 v163; // [rsp+48h] [rbp-6B8h]
  __int64 v164; // [rsp+48h] [rbp-6B8h]
  __int64 v165; // [rsp+58h] [rbp-6A8h] BYREF
  __int64 v166; // [rsp+60h] [rbp-6A0h] BYREF
  __int64 v167; // [rsp+68h] [rbp-698h] BYREF
  unsigned __int8 v168; // [rsp+7Fh] [rbp-681h] BYREF
  int v169; // [rsp+80h] [rbp-680h] BYREF
  int v170; // [rsp+84h] [rbp-67Ch] BYREF
  __int64 v171; // [rsp+88h] [rbp-678h] BYREF
  __int64 v172; // [rsp+90h] [rbp-670h] BYREF
  __int64 *v173; // [rsp+98h] [rbp-668h] BYREF
  __int64 v174; // [rsp+A0h] [rbp-660h] BYREF
  __int64 v175; // [rsp+A8h] [rbp-658h] BYREF
  char v176[8]; // [rsp+B0h] [rbp-650h] BYREF
  __int64 v177; // [rsp+B8h] [rbp-648h] BYREF
  __m128i v178[9]; // [rsp+C0h] [rbp-640h] BYREF
  __m128i v179; // [rsp+150h] [rbp-5B0h]
  __m128i v180; // [rsp+160h] [rbp-5A0h]
  __m128i v181; // [rsp+170h] [rbp-590h]
  __m128i v182; // [rsp+180h] [rbp-580h]
  __m128i v183; // [rsp+190h] [rbp-570h]
  __m128i v184; // [rsp+1A0h] [rbp-560h]
  __m128i v185; // [rsp+1B0h] [rbp-550h]
  __m128i v186; // [rsp+1C0h] [rbp-540h]
  __m128i v187; // [rsp+1D0h] [rbp-530h]
  __m128i v188; // [rsp+1E0h] [rbp-520h]
  __m128i v189; // [rsp+1F0h] [rbp-510h]
  __m128i v190; // [rsp+200h] [rbp-500h]
  __m128i v191; // [rsp+210h] [rbp-4F0h]
  __m128i v192; // [rsp+220h] [rbp-4E0h] BYREF
  __m128i v193; // [rsp+230h] [rbp-4D0h] BYREF
  __m128i v194; // [rsp+240h] [rbp-4C0h] BYREF
  __m128i v195; // [rsp+250h] [rbp-4B0h] BYREF
  __m128i v196; // [rsp+260h] [rbp-4A0h] BYREF
  __m128i v197; // [rsp+270h] [rbp-490h] BYREF
  __m128i v198; // [rsp+280h] [rbp-480h] BYREF
  __m128i v199; // [rsp+290h] [rbp-470h] BYREF
  __m128i v200; // [rsp+2A0h] [rbp-460h] BYREF
  __m128i v201; // [rsp+2B0h] [rbp-450h] BYREF
  __m128i v202; // [rsp+2C0h] [rbp-440h] BYREF
  __m128i v203; // [rsp+2D0h] [rbp-430h] BYREF
  __m128i v204; // [rsp+2E0h] [rbp-420h] BYREF
  __m128i v205; // [rsp+2F0h] [rbp-410h] BYREF
  __m128i v206; // [rsp+300h] [rbp-400h] BYREF
  __m128i v207; // [rsp+310h] [rbp-3F0h] BYREF
  __m128i v208; // [rsp+320h] [rbp-3E0h] BYREF
  __m128i v209; // [rsp+330h] [rbp-3D0h] BYREF
  __m128i v210; // [rsp+340h] [rbp-3C0h] BYREF
  __m128i v211; // [rsp+350h] [rbp-3B0h] BYREF
  __m128i v212; // [rsp+360h] [rbp-3A0h] BYREF
  __m128i v213; // [rsp+370h] [rbp-390h] BYREF
  __m128i v214[22]; // [rsp+380h] [rbp-380h] BYREF
  int v215; // [rsp+4E0h] [rbp-220h]
  __int16 v216; // [rsp+4E4h] [rbp-21Ch]
  _QWORD v217[66]; // [rsp+4F0h] [rbp-210h] BYREF

  v8 = a6;
  v10 = a7;
  v167 = a2;
  v11 = (a3 ^ 1) & (a1 == 0);
  v166 = a4;
  v165 = a5;
  v161 = v11;
  v168 = 0;
  v171 = 0;
  v12 = (_QWORD *)qword_4D03C50;
  v172 = 0;
  v169 = 0;
  v13 = *(_BYTE *)(qword_4D03C50 + 20LL);
  v173 = 0;
  *(_BYTE *)(qword_4D03C50 + 20LL) = v13 & 0xF7;
  v14 = (v13 & 8) != 0;
  v160 = (v13 & 8) != 0;
  if ( (v13 & 8) == 0 )
  {
    v159 = 0;
    v152 = 0;
    goto LABEL_23;
  }
  if ( unk_4F04C48 == -1 || (v12 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0) )
  {
    a4 = v160;
    v42 = 0;
    if ( dword_4F04C44 != -1 )
      v42 = v160;
    v159 = v42;
    v152 = dword_4F04C44 != -1;
LABEL_23:
    if ( !a1 )
      goto LABEL_5;
LABEL_24:
    v8 = v176;
    sub_6FE5D0(
      a1,
      v167,
      (unsigned int)v176,
      (unsigned int)&v165,
      (unsigned int)&v177,
      (unsigned int)&v173,
      (__int64)&v167,
      (__int64)a7,
      (__int64)v178);
    if ( *(_BYTE *)(a1 + 56) )
    {
      sub_6E6260(a7);
      v162 = 0;
      goto LABEL_85;
    }
    if ( v167 && (*(_BYTE *)(v167 + 50) & 0x20) != 0 )
    {
      sub_68D9C0((__int64)&v165, (__int64)v176, &v177, a1, v173, (__int64)a7, a8);
      v162 = 0;
      goto LABEL_85;
    }
    v157 = v173 == 0;
    v25 = v165;
    if ( *(_QWORD *)a1 )
      v175 = *(_QWORD *)(*(_QWORD *)a1 + 44LL);
    else
      v175 = v177;
    goto LABEL_29;
  }
  v152 = 1;
  v159 = (v13 & 8) != 0;
  if ( a1 )
    goto LABEL_24;
LABEL_5:
  v167 = 0;
  v177 = *(_QWORD *)a6;
  if ( !v11 )
    goto LABEL_6;
  if ( word_4F06418[0] != 27 )
  {
    if ( !dword_4D04428 || word_4F06418[0] != 73 )
    {
      sub_7BE280(27, 125, 0, 0);
      v165 = sub_72C930(27);
      v15 = v165;
LABEL_7:
      v157 = 0;
      if ( (unsigned int)sub_8D3F60(v15) )
        goto LABEL_8;
      goto LABEL_30;
    }
LABEL_6:
    v15 = v165;
    goto LABEL_7;
  }
  sub_7B8B50(a1, v14, v12, a4);
  v157 = 1;
  v25 = v165;
LABEL_29:
  if ( (unsigned int)sub_8D3F60(v25) )
  {
LABEL_8:
    memset(v217, 0, 0x1D8u);
    v217[19] = v217;
    v217[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v217[22]) |= 1u;
    v16 = sub_8D3FA0(v165, &v177);
    LOBYTE(v217[22]) |= 1u;
    v217[36] = v16;
    v217[35] = v16;
    v217[38] = v16;
    v165 = v16;
    HIDWORD(v217[15]) |= 0x8000480u;
    v217[13] = v177;
    v217[6] = *(_QWORD *)&dword_4F063F8;
    v162 = *(_QWORD *)(qword_4D03C50 + 136LL);
    if ( a1 )
    {
      v18 = v173;
      if ( v157 )
        v18 = (__int64 *)sub_690A60(*(_QWORD *)(a1 + 16), a1, v17);
      BYTE4(v217[16]) |= 8u;
      if ( v18 )
        sub_6E1C20(v18, 1, &v217[41]);
      sub_6BDE10((__int64)v217, v157);
      *(_QWORD *)(qword_4D03C50 + 136LL) = 0;
    }
    else
    {
      sub_6BDE10((__int64)v217, v157);
    }
    v165 = v217[36];
    if ( v157 )
    {
      sub_6E2220(v217);
      *(_QWORD *)dword_4F07508 = *(_QWORD *)v8;
      v155 = sub_68E4D0(&v165, (__int64)&v177, 0, dword_4D04428 != 0, dword_4D04428, v152);
      if ( dword_4D04428 )
        goto LABEL_19;
LABEL_21:
      v24 = v165;
      goto LABEL_32;
    }
    v173 = (__int64 *)sub_6E1C80(&v217[41]);
    *(_QWORD *)dword_4F07508 = *(_QWORD *)v8;
    v155 = sub_68E4D0(&v165, (__int64)&v177, 0, dword_4D04428 != 0, dword_4D04428, v152);
    v161 = dword_4D04428;
    if ( !dword_4D04428 )
      goto LABEL_21;
    goto LABEL_172;
  }
LABEL_30:
  *(_QWORD *)dword_4F07508 = *(_QWORD *)v8;
  v155 = sub_68E4D0(&v165, (__int64)&v177, 0, dword_4D04428 != 0, dword_4D04428, v152);
  if ( !dword_4D04428 )
  {
    v24 = v165;
    v162 = 0;
    goto LABEL_32;
  }
  if ( v157 )
  {
    v162 = 0;
    goto LABEL_19;
  }
  if ( v11 )
  {
    if ( word_4F06418[0] != 73 )
    {
      v162 = 0;
      v161 = 1;
      goto LABEL_19;
    }
    sub_6BD7E0(v165, 2, (__int64)v173, (__int64)a7);
    v162 = 0;
LABEL_76:
    v164 = 0;
    v175 = *(_QWORD *)&dword_4F061D8;
    goto LABEL_77;
  }
  v162 = 0;
LABEL_172:
  if ( v173 )
  {
    sub_6BD7E0(v165, 2, (__int64)v173, (__int64)a7);
    if ( a1 )
    {
      if ( !a7[16] )
        goto LABEL_265;
      v133 = *(_QWORD *)a7;
      for ( i = *(_BYTE *)(*(_QWORD *)a7 + 140LL); i == 12; i = *(_BYTE *)(v133 + 140) )
        v133 = *(_QWORD *)(v133 + 160);
      if ( !i )
LABEL_265:
        *(_BYTE *)(a1 + 56) = 1;
    }
    goto LABEL_76;
  }
  v161 = 0;
LABEL_19:
  v19 = v165;
  if ( !(unsigned int)sub_8D3410(v165) || dword_4D0478C )
    goto LABEL_21;
  if ( (unsigned int)sub_6E5430(v19, &v177, v20, v21, v22, v23) )
  {
    v140 = v165;
    v19 = (unsigned int)sub_8D23E0(v165) == 0 ? 119 : 2363;
    sub_685360(v19, &v177, v140);
  }
  v155 = 1;
  v165 = sub_72C930(v19);
  v24 = v165;
LABEL_32:
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  v26 = sub_8D3A70(v24);
  if ( !v26 )
  {
    v164 = 0;
    v31 = 0;
    goto LABEL_41;
  }
  for ( j = v165; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v163 = j;
  v28 = sub_8D23B0(j);
  v29 = v163;
  if ( v28 )
  {
    v77 = sub_8D3A70(v163);
    v29 = v163;
    if ( v77 )
    {
      sub_8AD220(v163, 0);
      v29 = v163;
    }
  }
  v26 = 0;
  v30 = *(__m128i **)(*(_QWORD *)v29 + 96LL);
  v31 = v30->m128i_i64[1];
  v164 = (__int64)v30;
  if ( v31 )
  {
    v32 = (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
    if ( qword_4D0495C && v161 && (v30[11].m128i_i8[2] & 0x10) != 0 )
    {
      v30 = 0;
      v151 = v29;
      sub_7ADF70(&v192, 0);
      v87 = v151;
      if ( word_4F06418[0] != 28 )
      {
        v30 = v214;
        memset(v214, 0, sizeof(v214));
        v215 = 0;
        v216 = 0;
        v214[4].m128i_i8[3] = 1;
        v214[1].m128i_i8[12] = 1;
        sub_7C6880(&v192, v214);
        v87 = v151;
        if ( word_4F06418[0] == 28 )
        {
          sub_7BC000(&v192);
          goto LABEL_41;
        }
      }
      v154 = v87;
      sub_7BC000(&v192);
      v29 = v154;
    }
    v26 = 1;
    v33 = *(_BYTE *)(v164 + 176);
    if ( (v33 & 2) == 0 )
    {
      if ( (v33 & 8) == 0 || (v29 = *(_QWORD *)(v29 + 168), (*(_BYTE *)(v29 + 109) & 0x20) != 0) )
      {
        v26 = 1;
        if ( (v33 & 0x40) == 0 )
        {
          v26 = 0;
          if ( *(_QWORD *)(v164 + 8) )
            v26 = sub_879360(v164, v30, v29, v32) != 0;
        }
      }
    }
  }
LABEL_41:
  v150 = sub_8D3EA0(v165);
  if ( v150 )
  {
    if ( (dword_4F077C4 != 2 || unk_4F07778 <= 202301) && (unsigned int)sub_6E53E0(5, 3347, v8) )
      sub_684B30(0xD13u, v8);
    v37 = 0;
    v36 = 0;
  }
  else
  {
    v34 = sub_8DD3B0(v165);
    v35 = v34 == 0;
    v36 = v34 != 0;
    v37 = !v35;
  }
  v38 = (__int64)&dword_4F077BC;
  if ( dword_4F077BC && !((unsigned int)qword_4F077B4 | v37) )
  {
    if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0
      || ((*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0
       || (v39 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4), ((v39 - 15) & 0xFD) != 0)
       && v39 != 2
       && (v39 != 1 || (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) == 0))
      && (qword_4F077A8 > 0x76BFu
       || unk_4F07734
       || *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 1
       || *(_BYTE *)(qword_4D03C50 + 16LL) != 2)
      || (unsigned int)sub_8D2FB0(v165) )
    {
      if ( v26 )
        goto LABEL_51;
      goto LABEL_97;
    }
    v26 = 1;
    goto LABEL_68;
  }
  if ( !v26 )
  {
    if ( !v36 )
    {
LABEL_97:
      v54 = *(_BYTE *)(v165 + 140);
      if ( v54 == 12 )
      {
        v55 = v165;
        do
        {
          v55 = *(_QWORD *)(v55 + 160);
          v54 = *(_BYTE *)(v55 + 140);
        }
        while ( v54 == 12 );
      }
      if ( v54 )
      {
        if ( dword_4D0478C && v157 && (unsigned int)sub_8D3BB0(v165) )
        {
          v38 = v161;
          if ( v161 )
          {
            v83 = qword_4D03C50;
            if ( (!qword_4D03C50 || (v84 = *(_QWORD **)(qword_4D03C50 + 136LL)) == 0 || !*v84) && word_4F06418[0] == 28 )
            {
              ++*(_BYTE *)(qword_4F061C8 + 36LL);
              ++*(_QWORD *)(v83 + 40);
              goto LABEL_119;
            }
            sub_6BE930(v165, a1, a3, &v166, (unsigned int *)v214);
            sub_832E80(v166);
            v175 = qword_4F063F0;
LABEL_216:
            if ( v214[0].m128i_i32[0] )
            {
              v85 = 2;
              sub_6BD7E0(v165, 2, v166, (__int64)a7);
              v86 = v166;
              if ( a3 )
                *(_QWORD *)(v166 + 24) = 0;
            }
            else
            {
              if ( *(_BYTE *)(v166 + 8) == 1 )
                sub_6E9FE0(v166, a7);
              else
                sub_6E6610(v166, a7, 0);
              v85 = (__int64)a7;
              v86 = v165;
              sub_6BF2D0(v165, (__m128i *)a7, 0, 2u, a8, v155, &v177, v8, &v175);
              if ( a3 )
              {
LABEL_220:
                if ( !a1 )
                  sub_690B80(v86, v85);
                goto LABEL_59;
              }
              v86 = v166;
            }
            sub_6E1990(v86);
            v166 = 0;
            goto LABEL_220;
          }
          if ( a3 )
          {
            if ( v166 )
            {
LABEL_282:
              sub_6BE930(v165, a1, a3, &v166, (unsigned int *)v214);
              sub_832E80(v166);
              goto LABEL_216;
            }
            goto LABEL_118;
          }
          if ( !a1 || !*(_QWORD *)(a1 + 16) )
            goto LABEL_282;
LABEL_192:
          if ( !*(_QWORD *)(a1 + 16) )
            goto LABEL_119;
          goto LABEL_193;
        }
        if ( v161 )
        {
          v56 = qword_4D03C50;
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          ++*(_QWORD *)(v56 + 40);
          if ( word_4F06418[0] == 28 )
            goto LABEL_119;
          if ( a3 )
          {
            v57 = (__int64 *)v166;
LABEL_140:
            v66 = *v57;
            if ( *v57 )
            {
              if ( *(_BYTE *)(v66 + 8) == 3 )
                v66 = sub_6BBB10(v57);
              v67 = (_DWORD *)sub_6E1A20(v66);
              if ( (unsigned int)sub_6E5430(v66, v38, v68, v69, v70, v71) )
              {
                v38 = (__int64)v67;
                sub_6851C0(0x8AAu, v67);
              }
              sub_6E6260(a7);
            }
            else
            {
              v141 = v57[3];
              *(__m128i *)a7 = _mm_loadu_si128((const __m128i *)(v141 + 8));
              *((__m128i *)a7 + 1) = _mm_loadu_si128((const __m128i *)(v141 + 24));
              *((__m128i *)a7 + 2) = _mm_loadu_si128((const __m128i *)(v141 + 40));
              *((__m128i *)a7 + 3) = _mm_loadu_si128((const __m128i *)(v141 + 56));
              *((__m128i *)a7 + 4) = _mm_loadu_si128((const __m128i *)(v141 + 72));
              *((__m128i *)a7 + 5) = _mm_loadu_si128((const __m128i *)(v141 + 88));
              *((__m128i *)a7 + 6) = _mm_loadu_si128((const __m128i *)(v141 + 104));
              *((__m128i *)a7 + 7) = _mm_loadu_si128((const __m128i *)(v141 + 120));
              *((__m128i *)a7 + 8) = _mm_loadu_si128((const __m128i *)(v141 + 136));
              v142 = *(_BYTE *)(v141 + 24);
              if ( v142 == 2 )
              {
                *((__m128i *)a7 + 9) = _mm_loadu_si128((const __m128i *)(v141 + 152));
                *((__m128i *)a7 + 10) = _mm_loadu_si128((const __m128i *)(v141 + 168));
                *((__m128i *)a7 + 11) = _mm_loadu_si128((const __m128i *)(v141 + 184));
                *((__m128i *)a7 + 12) = _mm_loadu_si128((const __m128i *)(v141 + 200));
                *((__m128i *)a7 + 13) = _mm_loadu_si128((const __m128i *)(v141 + 216));
                *((__m128i *)a7 + 14) = _mm_loadu_si128((const __m128i *)(v141 + 232));
                *((__m128i *)a7 + 15) = _mm_loadu_si128((const __m128i *)(v141 + 248));
                *((__m128i *)a7 + 16) = _mm_loadu_si128((const __m128i *)(v141 + 264));
                *((__m128i *)a7 + 17) = _mm_loadu_si128((const __m128i *)(v141 + 280));
                *((__m128i *)a7 + 18) = _mm_loadu_si128((const __m128i *)(v141 + 296));
                *((__m128i *)a7 + 19) = _mm_loadu_si128((const __m128i *)(v141 + 312));
                *((__m128i *)a7 + 20) = _mm_loadu_si128((const __m128i *)(v141 + 328));
                *((__m128i *)a7 + 21) = _mm_loadu_si128((const __m128i *)(v141 + 344));
              }
              else if ( v142 == 5 || v142 == 1 )
              {
                *((_QWORD *)a7 + 18) = *(_QWORD *)(v141 + 152);
              }
            }
LABEL_146:
            if ( !v169 )
              goto LABEL_147;
LABEL_119:
            if ( !v155 )
            {
              v58 = v165;
              if ( (unsigned int)sub_8D32E0(v165) | v150 )
              {
                if ( (unsigned int)sub_6E5430(v58, v38, v59, v60, v61, v62) )
                {
                  v38 = (__int64)v8;
                  sub_6851C0(0xABu, v8);
                }
              }
              else
              {
                v135 = v165;
                if ( (unsigned int)sub_8D3410(v165) )
                {
                  if ( (unsigned int)sub_6E5430(v135, v38, v136, v137, v138, v139) )
                  {
                    v38 = (__int64)v8;
                    sub_685360(0x77u, v8, v165);
                  }
                }
                else
                {
                  sub_6E7080(v10, 0);
                  v38 = v165;
                  if ( (unsigned int)sub_69A8F0(v10, v165, a8, &v177, &v168) )
                  {
                    if ( (unsigned int)sub_8D2600(v165) )
                    {
                      v38 = v165;
                      v63 = (__int64)v10;
                      sub_6F7220(v10, v165);
                      v164 = 0;
                      goto LABEL_124;
                    }
                    if ( v164 )
                    {
                      if ( (unsigned int)sub_6E6160(v165, v8) )
                      {
                        v143 = dword_4D04230 != 0;
                      }
                      else
                      {
                        v143 = 1;
                        if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
                        {
                          if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (v143 = 1, qword_4F077A8 > 0x9FC3u) )
                          {
                            v143 = 1;
                            if ( dword_4F04C44 == -1 )
                            {
                              v147 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                              if ( (*(_BYTE *)(v147 + 6) & 6) == 0 && *(_BYTE *)(v147 + 4) != 12 )
                              {
                                v148 = v214;
                                v214[0].m128i_i32[0] = 0;
                                v149 = v165;
                                if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
                                  v148 = 0;
                                while ( *(_BYTE *)(v149 + 140) == 12 )
                                  v149 = *(_QWORD *)(v149 + 160);
                                sub_640330(0, v149, 0, v148);
                                if ( v214[0].m128i_i32[0] && *(char *)(qword_4D03C50 + 18LL) < 0 )
                                  sub_6E50A0(0, v149);
                                v143 = 1;
                              }
                            }
                          }
                        }
                      }
                      v172 = sub_6ECBB0(v165, v143, v8);
                      sub_6E70E0(v172, v10);
                      v38 = (__int64)v10;
                      v63 = 2;
                      sub_6E26D0(2, v10);
                      v164 = 0;
                      goto LABEL_124;
                    }
                    if ( (unsigned int)sub_8DBE70(v165) )
                    {
                      v38 = (__int64)v10;
                      v172 = sub_6ECBB0(v165, 1, v8);
                      sub_6E70E0(v172, v10);
                      v63 = (__int64)v10;
                      sub_6F4B70(v10);
                      goto LABEL_124;
                    }
                    if ( (unsigned int)sub_8D2B80(v165) || (unsigned int)sub_8D2BF0(v165) )
                    {
                      v172 = sub_6ECBB0(v165, 1, v8);
                      sub_6E70E0(v172, v10);
                      v38 = 0;
                      v63 = (__int64)v10;
                      sub_6E6B60(v10, 0);
                      goto LABEL_124;
                    }
                    v63 = v165;
                    v38 = (__int64)v10;
                    sub_6FC3F0(v165, v10, 0);
                    if ( *(_BYTE *)(qword_4D03C50 + 16LL) && *((_BYTE *)v10 + 16) == 2 )
                    {
                      v63 = v165;
                      v38 = 1;
                      v172 = sub_6ECBB0(v165, 1, v8);
                      v10[36] = v172;
                      goto LABEL_124;
                    }
LABEL_258:
                    v164 = 0;
                    goto LABEL_124;
                  }
                }
              }
            }
            v63 = (__int64)v10;
            sub_6E6260(v10);
            v164 = 0;
            goto LABEL_124;
          }
          if ( a1 )
          {
LABEL_193:
            v169 = 1;
            if ( *(_QWORD *)(a1 + 16) )
            {
              v158 = a1;
              v79 = 1;
              v80 = *(_QWORD *)(a1 + 16);
              while ( 1 )
              {
                if ( (*(_BYTE *)(v80 + 26) & 4) != 0 )
                {
                  v153 = v79;
                  v81 = sub_6E3DA0(v80, 0);
                  v38 = *(_QWORD *)(v158 + 32);
                  v82 = sub_869530(
                          *(_QWORD *)(v81 + 128),
                          v38,
                          *(_QWORD *)(v158 + 24),
                          (unsigned int)&v174,
                          *(_DWORD *)(v158 + 40),
                          *(_QWORD *)(v158 + 48),
                          (__int64)&v170);
                  v79 = v153;
                  if ( v170 )
                    *(_BYTE *)(v158 + 56) = 1;
                  if ( v82 )
                  {
                    while ( 1 )
                    {
                      sub_6E2E50(0, &v192);
                      sub_6E2E50(0, v214);
                      sub_6F85E0(v80, v158, 2, &v192, v214);
                      if ( v153 )
                      {
                        v89 = _mm_loadu_si128(&v193);
                        v90 = _mm_loadu_si128(&v194);
                        v91 = _mm_loadu_si128(&v195);
                        v92 = _mm_loadu_si128(&v196);
                        v93 = _mm_loadu_si128(&v197);
                        *(__m128i *)a7 = _mm_loadu_si128(&v192);
                        v94 = _mm_loadu_si128(&v198);
                        *((__m128i *)a7 + 1) = v89;
                        v95 = _mm_loadu_si128(&v199);
                        v96 = _mm_loadu_si128(&v200);
                        v97 = v193.m128i_i8[0];
                        *((__m128i *)a7 + 2) = v90;
                        *((__m128i *)a7 + 3) = v91;
                        *((__m128i *)a7 + 4) = v92;
                        *((__m128i *)a7 + 5) = v93;
                        *((__m128i *)a7 + 6) = v94;
                        *((__m128i *)a7 + 7) = v95;
                        *((__m128i *)a7 + 8) = v96;
                        if ( v97 == 2 )
                        {
                          v106 = _mm_loadu_si128(&v202);
                          v107 = _mm_loadu_si128(&v203);
                          v108 = _mm_loadu_si128(&v204);
                          v109 = _mm_loadu_si128(&v205);
                          v110 = _mm_loadu_si128(&v206);
                          *((__m128i *)a7 + 9) = _mm_loadu_si128(&v201);
                          v111 = _mm_loadu_si128(&v207);
                          *((__m128i *)a7 + 10) = v106;
                          v112 = _mm_loadu_si128(&v208);
                          *((__m128i *)a7 + 11) = v107;
                          v113 = _mm_loadu_si128(&v209);
                          *((__m128i *)a7 + 12) = v108;
                          v114 = _mm_loadu_si128(&v210);
                          *((__m128i *)a7 + 13) = v109;
                          v115 = _mm_loadu_si128(&v211);
                          *((__m128i *)a7 + 14) = v110;
                          v116 = _mm_loadu_si128(&v212);
                          *((__m128i *)a7 + 15) = v111;
                          v117 = _mm_loadu_si128(&v213);
                          *((__m128i *)a7 + 16) = v112;
                          *((__m128i *)a7 + 17) = v113;
                          *((__m128i *)a7 + 18) = v114;
                          *((__m128i *)a7 + 19) = v115;
                          *((__m128i *)a7 + 20) = v116;
                          *((__m128i *)a7 + 21) = v117;
                        }
                        else if ( v97 == 5 || v97 == 1 )
                        {
                          *((_QWORD *)a7 + 18) = v201.m128i_i64[0];
                        }
                        v98 = _mm_loadu_si128(&v214[1]);
                        v99 = _mm_loadu_si128(&v214[2]);
                        v100 = _mm_loadu_si128(&v214[3]);
                        v101 = _mm_loadu_si128(&v214[4]);
                        v102 = _mm_loadu_si128(&v214[5]);
                        v178[0] = _mm_loadu_si128(v214);
                        v103 = _mm_loadu_si128(&v214[6]);
                        v104 = _mm_loadu_si128(&v214[7]);
                        v178[1] = v98;
                        v105 = _mm_loadu_si128(&v214[8]);
                        v178[2] = v99;
                        v178[3] = v100;
                        v178[4] = v101;
                        v178[5] = v102;
                        v178[6] = v103;
                        v178[7] = v104;
                        v178[8] = v105;
                        if ( v214[1].m128i_i8[0] == 2 )
                        {
                          v118 = _mm_loadu_si128(&v214[10]);
                          v119 = _mm_loadu_si128(&v214[11]);
                          v120 = _mm_loadu_si128(&v214[12]);
                          v121 = _mm_loadu_si128(&v214[13]);
                          v122 = _mm_loadu_si128(&v214[14]);
                          v179 = _mm_loadu_si128(&v214[9]);
                          v123 = _mm_loadu_si128(&v214[20]);
                          v124 = _mm_loadu_si128(&v214[15]);
                          v180 = v118;
                          v125 = _mm_loadu_si128(&v214[21]);
                          v126 = _mm_loadu_si128(&v214[16]);
                          v181 = v119;
                          v182 = v120;
                          v127 = _mm_loadu_si128(&v214[17]);
                          v128 = _mm_loadu_si128(&v214[18]);
                          v183 = v121;
                          v129 = _mm_loadu_si128(&v214[19]);
                          v184 = v122;
                          v185 = v124;
                          v186 = v126;
                          v187 = v127;
                          v188 = v128;
                          v189 = v129;
                          v190 = v123;
                          v191 = v125;
                        }
                        else if ( v214[1].m128i_i8[0] == 5 || v214[1].m128i_i8[0] == 1 )
                        {
                          v179.m128i_i64[0] = v214[9].m128i_i64[0];
                        }
                        v169 = 0;
                      }
                      else
                      {
                        *(_BYTE *)(v158 + 56) = 1;
                      }
                      v38 = 0;
                      sub_867630(v174, 0);
                      if ( !(unsigned int)sub_866C00(v174) )
                        break;
                      v88 = *(_QWORD *)(*(_QWORD *)(v174 + 16) + 8LL);
                      if ( *(_DWORD *)(v88 + 32) == 2 && (*(_BYTE *)(*(_QWORD *)(v88 + 80) + 33LL) & 1) != 0 )
                        break;
                      v153 = 0;
                    }
                    v79 = 0;
                  }
                }
                else
                {
                  if ( !v79 )
                  {
                    v10 = a7;
                    *(_BYTE *)(v158 + 56) = 1;
                    goto LABEL_146;
                  }
                  v38 = v158;
                  sub_6F85E0(v80, v158, 2, a7, v178);
                  v79 = 0;
                  v169 = 0;
                }
                v80 = *(_QWORD *)(v80 + 16);
                if ( !v80 )
                {
                  v10 = a7;
                  goto LABEL_146;
                }
              }
            }
            goto LABEL_119;
          }
        }
        else
        {
          if ( a3 )
          {
LABEL_118:
            v57 = (__int64 *)v166;
            if ( !v166 )
              goto LABEL_119;
            goto LABEL_140;
          }
          if ( a1 )
            goto LABEL_192;
        }
        v38 = 1;
        sub_6C9F90(v165, 1u, 0, (__m128i *)a7, v178, &v169);
        goto LABEL_146;
      }
      sub_6C0D90(a1, a3, v166);
      sub_6E6260(a7);
      goto LABEL_75;
    }
LABEL_68:
    v43 = a1;
    sub_6C55E0(a1, a3, v166, 0, (__m128i *)a7, &v171);
    if ( v171 )
    {
      if ( *(_BYTE *)(v171 + 48) == 5 && !*(_QWORD *)(v171 + 64) && (v43 = v165, v26 | (unsigned int)sub_8D3D40(v165)) )
      {
        v172 = sub_6ECBB0(v165, 1, v8);
        sub_6E70E0(v172, a7);
        sub_6F4B70(a7);
      }
      else
      {
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
        {
          if ( (unsigned int)sub_6E5430(v43, a3, v44, v45, v46, v47) )
            sub_6851C0(0x1Cu, v8);
          sub_6E6260(a7);
        }
        else
        {
          v172 = sub_6EC670(v165, v171, 0, 1);
          sub_6E70E0(v172, a7);
        }
        sub_6E26D0(2, a7);
      }
    }
    else
    {
      sub_6F4200(a7, v165, 2, 0);
    }
LABEL_75:
    if ( !v161 )
    {
LABEL_59:
      v164 = 0;
      goto LABEL_60;
    }
    goto LABEL_76;
  }
  v26 = 0;
  if ( v36 )
    goto LABEL_68;
LABEL_51:
  v214[0].m128i_i32[0] = 0;
  sub_6BE930(v165, a1, a3, &v166, (unsigned int *)v214);
  sub_832E80(v166);
  if ( v214[0].m128i_i32[0] )
  {
    v130 = v165;
    sub_6BD7E0(v165, 2, v166, (__int64)a7);
    if ( v161 )
    {
      v175 = qword_4F063F0;
      sub_690B80(v130, 2);
      v131 = v166;
    }
    else
    {
      v131 = v166;
      if ( a3 )
        *(_QWORD *)(v166 + 24) = 0;
    }
    sub_6E1990(v131);
    v166 = 0;
    v132 = sub_68B740((__int64)a7);
    if ( v132 )
      *(_BYTE *)(v132 + 169) = *(_BYTE *)(v132 + 169) & 0x9F | 0x40;
    goto LABEL_59;
  }
  sub_6C5750(v31, (__int64)v8, 0, v165, 1, 1, 0, 1024, a1, 1u, v166, 0, 0, 0, 0, &v192, 0, (int)a7, &v171, &v172, &v175);
  if ( !a3 )
  {
    sub_6E1990(v166);
    if ( !a1 )
      sub_7BE280(28, 18, 0, 0);
  }
  *(_QWORD *)dword_4F07508 = *(_QWORD *)v8;
  if ( !v192.m128i_i32[0] )
  {
    if ( v155 || !v171 )
    {
      sub_6E6260(a7);
    }
    else if ( word_4D04898
           && !*(_QWORD *)(v171 + 16)
           && (v144 = sub_730250(v171), (v145 = v144) != 0)
           && (*(_BYTE *)(v144 + 171) & 2) != 0 )
    {
      v146 = sub_730290(v171);
      *(_BYTE *)(v146 + 50) |= 0x10u;
      sub_6E6A50(v145, a7);
      *(_QWORD *)(a7 + 68) = *(_QWORD *)v8;
    }
    else
    {
      nullsub_4();
      v40 = v171;
      *(_BYTE *)(v171 + 50) |= 0x10u;
      v41 = sub_730290(v40);
      *(_BYTE *)(v41 + 50) |= 0x10u;
      sub_6E70E0(v172, a7);
      *(_QWORD *)(a7 + 68) = *(_QWORD *)v8;
      sub_6E26D0(2, a7);
    }
    goto LABEL_59;
  }
LABEL_147:
  if ( (unsigned int)sub_8D3410(v165) )
  {
    v63 = v165;
    v38 = (__int64)v10;
    if ( !(unsigned int)sub_68BAB0(v165, v10, &v177, v72, v73) )
      v165 = sub_72C930(v63);
    goto LABEL_258;
  }
  if ( dword_4F04C44 == -1 )
  {
    v74 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v74 + 6) & 6) == 0 && *(_BYTE *)(v74 + 4) != 12 )
    {
LABEL_151:
      if ( !v150 )
      {
LABEL_157:
        if ( v161 )
          v175 = qword_4F063F0;
        v75 = *((_BYTE *)v10 + 16);
        if ( v75 == 1 )
        {
          v164 = v10[18];
        }
        else
        {
          v164 = 0;
          if ( v75 == 2 )
          {
            v164 = v10[36];
            if ( !v164 && *((_BYTE *)v10 + 317) == 12 && *((_BYTE *)v10 + 320) == 1 )
              v164 = sub_72E9A0(v10 + 18);
          }
        }
        v63 = v165;
        v38 = (__int64)v10;
        sub_6BF2D0(v165, (__m128i *)v10, v178, 2u, a8, v155, &v177, v8, &v175);
        goto LABEL_124;
      }
LABEL_156:
      v165 = sub_6EEB30(*v10, 0);
      goto LABEL_157;
    }
  }
  if ( (unsigned int)sub_8DBE70(v165) )
  {
    if ( v150 && !(unsigned int)sub_696840((__int64)v10) )
      goto LABEL_156;
  }
  else if ( !(unsigned int)sub_696840((__int64)v10) )
  {
    goto LABEL_151;
  }
  if ( (*((_BYTE *)v10 + 18) & 1) != 0 )
    sub_68FA30(v165, v8, (const __m128i *)v10, (__int64)v178);
  v38 = v165;
  v63 = (__int64)v10;
  sub_6F4200(v10, v165, 2, 0);
  v164 = 0;
LABEL_124:
  if ( v161 )
  {
    v175 = qword_4F063F0;
    sub_690B80(v63, v38);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    goto LABEL_77;
  }
  if ( a3 )
  {
    v64 = v166;
    if ( v166 )
    {
      while ( 1 )
      {
        v65 = (_QWORD *)v64;
        v64 = *(_QWORD *)v64;
        if ( !v64 )
          break;
        if ( *(_BYTE *)(v64 + 8) == 3 )
          v64 = sub_6BBB10(v65);
      }
      v175 = *(_QWORD *)sub_6E1A20(v65);
    }
    else
    {
      v175 = *(_QWORD *)v8;
    }
LABEL_77:
    if ( !v159 )
      goto LABEL_78;
    goto LABEL_61;
  }
LABEL_60:
  if ( !v159 )
    goto LABEL_78;
LABEL_61:
  if ( (unsigned int)sub_8D23B0(v165) )
    sub_697260(v165, (__int64)v10, (__int64)&v177);
LABEL_78:
  v48 = *((_BYTE *)v10 + 16);
  if ( v48 == 1 )
  {
    v76 = v10[18];
  }
  else
  {
    if ( v48 != 2 )
      goto LABEL_80;
    v76 = v10[36];
    if ( v76 )
    {
LABEL_164:
      *(_BYTE *)(v76 + 25) |= 0x80u;
      if ( *(_BYTE *)(v76 + 24) == 5 )
        v171 = *(_QWORD *)(v76 + 56);
      goto LABEL_80;
    }
    if ( *((_BYTE *)v10 + 317) != 12 || *((_BYTE *)v10 + 320) != 1 )
      goto LABEL_80;
    v76 = sub_72E9A0(v10 + 18);
  }
  if ( v76 )
    goto LABEL_164;
LABEL_80:
  v49 = *((_WORD *)v8 + 2);
  *((_DWORD *)v10 + 17) = *(_DWORD *)v8;
  *((_WORD *)v10 + 36) = v49;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)((char *)v10 + 68);
  v50 = v175;
  *(_QWORD *)((char *)v10 + 76) = v175;
  *(_QWORD *)&dword_4F061D8 = v50;
  sub_6E3280(v10, v8);
  sub_6E41D0(v10, v164, 2, v8, &v177, v165);
  if ( v172 )
  {
    v51 = *(_QWORD *)(v172 + 80);
    if ( v51 )
    {
      if ( v171 )
      {
        if ( *(_BYTE *)(v171 + 48) == 2 )
        {
          v78 = *(_QWORD *)(*(_QWORD *)(v171 + 56) + 144LL);
          if ( v78 )
          {
            if ( (unsigned __int8)(*(_BYTE *)(v78 + 24) - 5) <= 1u )
              *(_QWORD *)(v78 + 80) = v51;
          }
        }
      }
    }
  }
  sub_6E26D0(v168, v10);
LABEL_85:
  *(_BYTE *)(qword_4D03C50 + 20LL) = (8 * v160) | *(_BYTE *)(qword_4D03C50 + 20LL) & 0xF7;
  sub_6E1990(v173);
  result = qword_4D03C50;
  v53 = *(_QWORD *)(qword_4D03C50 + 136LL);
  if ( v53 != v162 )
  {
    if ( v53 )
    {
      sub_6E1BF0(v53);
      result = qword_4D03C50;
    }
    *(_QWORD *)(result + 136) = v162;
  }
  return result;
}
