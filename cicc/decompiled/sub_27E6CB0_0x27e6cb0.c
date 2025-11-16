// Function: sub_27E6CB0
// Address: 0x27e6cb0
//
__int64 __fastcall sub_27E6CB0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned int v4; // r13d
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  void *v16; // rsi
  __int64 v17; // rbx
  void **v18; // r13
  __int64 v19; // rcx
  unsigned __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  __int64 v23; // rdx
  char v24; // al
  __int64 v25; // rbx
  char *v26; // rax
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r9
  __int64 i; // r13
  char *v33; // rdi
  char v34; // dl
  __int64 v35; // r13
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // r14
  const char *v39; // rax
  __m128i v40; // xmm0
  __m128i v41; // xmm1
  int v42; // r14d
  unsigned __int8 v43; // al
  __int64 v44; // rax
  __int64 v45; // r15
  __int64 v46; // rax
  unsigned __int8 v47; // al
  __int64 v48; // rdx
  __int64 *v49; // rdx
  __int64 v50; // rbx
  __int64 v51; // rdx
  void *v52; // rdi
  __int64 v53; // rdx
  char v54; // cl
  __int64 v55; // rax
  int v56; // ecx
  __int64 v57; // rax
  __int64 v58; // r15
  unsigned __int8 *v59; // r13
  const char *v60; // rsi
  __int64 v61; // r13
  __int64 v62; // rbx
  __int64 v63; // rdx
  unsigned __int64 v64; // rdi
  __int64 *v65; // r13
  __int64 j; // rax
  __int64 v67; // rcx
  unsigned __int64 *v68; // rdx
  __int64 v69; // rdx
  unsigned __int64 v70; // rax
  __int64 v71; // r13
  int v72; // eax
  int v73; // eax
  unsigned int v74; // ecx
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rcx
  _BYTE **v78; // rbx
  _BYTE **v79; // r14
  _BYTE *v80; // r13
  __int64 v81; // rsi
  unsigned __int8 *v82; // rsi
  __int64 v83; // r13
  __int64 v84; // rdx
  int v85; // r15d
  __int64 v86; // rcx
  __int64 v87; // r8
  char *v88; // r13
  __int64 *v89; // rbx
  __int64 v90; // rsi
  _QWORD *v91; // rax
  __int64 v92; // r15
  __int64 v93; // rdx
  __int64 *v94; // rbx
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // r9
  char *v99; // rax
  unsigned __int64 v100; // rdx
  char v101; // r15
  _QWORD *v102; // rax
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // r13
  const char *v106; // rsi
  const char **v107; // r15
  int v108; // eax
  __int64 *v109; // rdx
  __int64 v110; // rax
  unsigned __int64 v111; // rdx
  __int64 *v112; // rdx
  __int64 v113; // rsi
  unsigned __int8 *v114; // rsi
  unsigned __int64 v115; // rax
  int v116; // eax
  unsigned __int64 v117; // rdx
  __int64 *v118; // rax
  __int64 v119; // [rsp+28h] [rbp-578h]
  __int64 v121; // [rsp+40h] [rbp-560h]
  __int64 v122; // [rsp+40h] [rbp-560h]
  __int64 v123; // [rsp+48h] [rbp-558h]
  __int64 v124; // [rsp+48h] [rbp-558h]
  __int64 v125; // [rsp+60h] [rbp-540h]
  char v126; // [rsp+70h] [rbp-530h]
  int v127; // [rsp+78h] [rbp-528h]
  __int64 v128; // [rsp+78h] [rbp-528h]
  __int16 v129; // [rsp+78h] [rbp-528h]
  __int64 v130; // [rsp+78h] [rbp-528h]
  __int64 v131; // [rsp+78h] [rbp-528h]
  unsigned int v132; // [rsp+80h] [rbp-520h]
  __int64 v133; // [rsp+88h] [rbp-518h]
  int v134; // [rsp+88h] [rbp-518h]
  __int64 v135; // [rsp+88h] [rbp-518h]
  __int64 v136; // [rsp+88h] [rbp-518h]
  __int64 v137; // [rsp+88h] [rbp-518h]
  char v138; // [rsp+9Fh] [rbp-501h] BYREF
  __int64 v139; // [rsp+A0h] [rbp-500h] BYREF
  __int16 v140; // [rsp+A8h] [rbp-4F8h]
  __m128i v141; // [rsp+B0h] [rbp-4F0h] BYREF
  __m128i v142; // [rsp+C0h] [rbp-4E0h] BYREF
  _BYTE *v143; // [rsp+D0h] [rbp-4D0h] BYREF
  __int64 v144; // [rsp+D8h] [rbp-4C8h]
  _BYTE v145[64]; // [rsp+E0h] [rbp-4C0h] BYREF
  __int64 **v146; // [rsp+120h] [rbp-480h] BYREF
  __int64 v147; // [rsp+128h] [rbp-478h]
  _BYTE v148[64]; // [rsp+130h] [rbp-470h] BYREF
  __int64 v149; // [rsp+170h] [rbp-430h] BYREF
  char *v150; // [rsp+178h] [rbp-428h]
  __int64 v151; // [rsp+180h] [rbp-420h]
  int v152; // [rsp+188h] [rbp-418h]
  char v153; // [rsp+18Ch] [rbp-414h]
  char v154; // [rsp+190h] [rbp-410h] BYREF
  char *v155; // [rsp+1D0h] [rbp-3D0h] BYREF
  _OWORD *v156; // [rsp+1D8h] [rbp-3C8h]
  __m128i v157; // [rsp+1E0h] [rbp-3C0h]
  __m128i v158; // [rsp+1F0h] [rbp-3B0h] BYREF
  void *base; // [rsp+230h] [rbp-370h] BYREF
  __int64 v160; // [rsp+238h] [rbp-368h]
  _BYTE v161[16]; // [rsp+240h] [rbp-360h] BYREF
  __int16 v162; // [rsp+250h] [rbp-350h]
  _QWORD *v163[3]; // [rsp+2C0h] [rbp-2E0h] BYREF
  __int64 v164; // [rsp+2D8h] [rbp-2C8h]
  __int64 v165; // [rsp+2E0h] [rbp-2C0h] BYREF
  unsigned int v166; // [rsp+2E8h] [rbp-2B8h]
  _QWORD v167[2]; // [rsp+420h] [rbp-180h] BYREF
  char v168; // [rsp+430h] [rbp-170h]
  _BYTE *v169; // [rsp+438h] [rbp-168h]
  __int64 v170; // [rsp+440h] [rbp-160h]
  _BYTE v171[136]; // [rsp+448h] [rbp-158h] BYREF
  _QWORD v172[2]; // [rsp+4D0h] [rbp-D0h] BYREF
  __int64 v173; // [rsp+4E0h] [rbp-C0h]
  __int64 v174; // [rsp+4E8h] [rbp-B8h] BYREF
  unsigned int v175; // [rsp+4F0h] [rbp-B0h]
  char v176; // [rsp+568h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  if ( sub_AA54C0(v3) )
    return 0;
  v6 = sub_AA4FF0(v3);
  if ( !v6 )
    BUG();
  v7 = (unsigned int)*(unsigned __int8 *)(v6 - 24) - 39;
  if ( (unsigned int)v7 <= 0x38 )
  {
    v8 = 0x100060000000001LL;
    if ( _bittest64(&v8, v7) )
      return 0;
  }
  v9 = *(_QWORD *)(a2 - 32);
  v119 = v9;
  if ( *(_BYTE *)v9 > 0x1Cu && *(_QWORD *)(v9 + 40) == v3 && *(_BYTE *)v9 != 84 )
    return 0;
  v163[2] = 0;
  v140 = 0;
  v10 = (_QWORD *)a1[5];
  v139 = a2 + 24;
  v164 = 1;
  v163[0] = v10;
  v163[1] = v10;
  v11 = &v165;
  do
  {
    *v11 = -4;
    v11 += 5;
    *(v11 - 4) = -3;
    *(v11 - 3) = -4;
    *(v11 - 2) = -3;
  }
  while ( v11 != v167 );
  v167[1] = 0;
  v167[0] = v172;
  v169 = v171;
  v170 = 0x400000000LL;
  v168 = 0;
  v171[128] = 0;
  v172[1] = 0;
  v173 = 1;
  v172[0] = &unk_49DDBE8;
  v12 = &v174;
  do
  {
    *v12 = -4096;
    v12 += 2;
  }
  while ( v12 != (__int64 *)&v176 );
  v171[129] = 0;
  v13 = (_BYTE *)sub_D319E0(a2, v3, &v139, qword_4F86CA8[8], v163, &v138, 0);
  v133 = (__int64)v13;
  if ( v13 )
  {
    if ( v138 )
    {
      v50 = (__int64)v13;
      sub_F57030(v13, a2, 0);
      sub_22C1820(a1[4], v50);
    }
    if ( a2 == v133 )
      v133 = sub_ACADE0(*(__int64 ***)(a2 + 8));
    v14 = *(_QWORD *)(a2 + 8);
    if ( *(_QWORD *)(v133 + 8) == v14 )
      goto LABEL_24;
    v162 = 257;
    v15 = sub_B52260(v133, v14, (__int64)&base, a2 + 24, 0);
    v16 = *(void **)(a2 + 48);
    v133 = v15;
    v17 = v15;
    base = v16;
    if ( v16 )
    {
      v18 = (void **)(v15 + 48);
      sub_B96E90((__int64)&base, (__int64)v16, 1);
      if ( v18 == &base )
      {
        if ( base )
          sub_B91220(v17 + 48, (__int64)base);
        goto LABEL_24;
      }
      v27 = *(_QWORD *)(v17 + 48);
      if ( !v27 )
      {
LABEL_48:
        v28 = (unsigned __int8 *)base;
        *(_QWORD *)(v133 + 48) = base;
        if ( v28 )
          sub_B976B0((__int64)&base, v28, (__int64)v18);
        goto LABEL_24;
      }
    }
    else
    {
      v18 = (void **)(v15 + 48);
      if ( (void **)(v15 + 48) == &base || (v27 = *(_QWORD *)(v15 + 48)) == 0 )
      {
LABEL_24:
        v4 = 1;
        sub_BD84D0(a2, v133);
        sub_B43D60((_QWORD *)a2);
        goto LABEL_25;
      }
    }
    sub_B91220((__int64)v18, v27);
    goto LABEL_48;
  }
  v4 = 0;
  if ( *(_QWORD *)(v3 + 56) != v139 )
    goto LABEL_25;
  sub_B91FC0(v141.m128i_i64, a2);
  v151 = 8;
  v150 = &v154;
  base = v161;
  v152 = 0;
  v153 = 1;
  v160 = 0x800000000LL;
  v22 = *(_QWORD *)(v3 + 16);
  v149 = 0;
  v143 = v145;
  v144 = 0x800000000LL;
  if ( !v22 )
    goto LABEL_54;
  while ( 1 )
  {
    v23 = *(_QWORD *)(v22 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
      break;
    v22 = *(_QWORD *)(v22 + 8);
    if ( !v22 )
      goto LABEL_51;
  }
  v123 = v3;
  v24 = 1;
LABEL_35:
  v25 = *(_QWORD *)(v23 + 40);
  if ( !v24 )
    goto LABEL_67;
  v26 = v150;
  v19 = HIDWORD(v151);
  v23 = (__int64)&v150[8 * HIDWORD(v151)];
  if ( v150 != (char *)v23 )
  {
    while ( v25 != *(_QWORD *)v26 )
    {
      v26 += 8;
      if ( (char *)v23 == v26 )
        goto LABEL_84;
    }
    goto LABEL_40;
  }
LABEL_84:
  if ( HIDWORD(v151) >= (unsigned int)v151 )
  {
LABEL_67:
    sub_C8CC70((__int64)&v149, v25, v23, v19, v20, v21);
    if ( v34 )
      goto LABEL_68;
LABEL_40:
    v25 = v133;
    goto LABEL_41;
  }
  ++HIDWORD(v151);
  *(_QWORD *)v23 = v25;
  ++v149;
LABEL_68:
  v35 = *(_QWORD *)(a2 + 8);
  v140 = 0;
  v139 = v25 + 48;
  LODWORD(v146) = 0;
  v36 = sub_B43CC0(a2);
  v155 = (char *)sub_9208B0(v36, v35);
  v156 = (_OWORD *)v37;
  v38 = (unsigned __int64)(v155 + 7) >> 3;
  if ( (_BYTE)v37 )
    v38 |= 0x4000000000000000uLL;
  v39 = (const char *)sub_BD5BF0(v119, v123, v25);
  v40 = _mm_load_si128(&v141);
  v41 = _mm_load_si128(&v142);
  v155 = (char *)v39;
  v156 = (_OWORD *)v38;
  v157 = v40;
  v42 = qword_4F86CA8[8];
  v158 = v41;
  v43 = sub_B46500((unsigned __int8 *)a2);
  v44 = sub_D31270((__int64)&v155, v35, v43, v25, &v139, v42, v163, &v138, &v146);
  if ( v44 )
    goto LABEL_78;
  if ( !v25 )
  {
LABEL_41:
    v133 = v25;
    goto LABEL_42;
  }
  v121 = v22;
  v45 = v25;
  do
  {
    if ( *(_QWORD *)(v45 + 56) != v139
      || LODWORD(qword_4F86CA8[8]) <= (unsigned int)v146
      || (v46 = sub_AA54C0(v45), (v45 = v46) == 0) )
    {
      v22 = v121;
      goto LABEL_41;
    }
    v139 = v46 + 48;
    v140 = 0;
    v127 = LODWORD(qword_4F86CA8[8]) - (_DWORD)v146;
    v47 = sub_B46500((unsigned __int8 *)a2);
    v44 = sub_D31270((__int64)&v155, v35, v47, v45, &v139, v127, v163, &v138, &v146);
  }
  while ( !v44 );
  v22 = v121;
LABEL_78:
  if ( v138 )
  {
    v51 = (unsigned int)v144;
    v20 = (unsigned int)v144 + 1LL;
    if ( v20 > HIDWORD(v144) )
    {
      v130 = v44;
      sub_C8D5F0((__int64)&v143, v145, (unsigned int)v144 + 1LL, 8u, v20, v21);
      v51 = (unsigned int)v144;
      v44 = v130;
    }
    *(_QWORD *)&v143[8 * v51] = v44;
    LODWORD(v144) = v144 + 1;
  }
  v48 = (unsigned int)v160;
  v19 = (unsigned int)v160;
  if ( (unsigned int)v160 >= (unsigned __int64)HIDWORD(v160) )
  {
    v20 = (unsigned int)v160 + 1LL;
    if ( HIDWORD(v160) < v20 )
    {
      v131 = v44;
      sub_C8D5F0((__int64)&base, v161, (unsigned int)v160 + 1LL, 0x10u, v20, v21);
      v48 = (unsigned int)v160;
      v44 = v131;
    }
    v112 = (__int64 *)((char *)base + 16 * v48);
    *v112 = v25;
    v112[1] = v44;
    LODWORD(v160) = v160 + 1;
  }
  else
  {
    v49 = (__int64 *)((char *)base + 16 * (unsigned int)v160);
    if ( v49 )
    {
      *v49 = v25;
      v49[1] = v44;
      LODWORD(v19) = v160;
    }
    v19 = (unsigned int)(v19 + 1);
    LODWORD(v160) = v19;
  }
LABEL_42:
  while ( 1 )
  {
    v22 = *(_QWORD *)(v22 + 8);
    if ( !v22 )
      break;
    v23 = *(_QWORD *)(v22 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
    {
      v24 = v153;
      goto LABEL_35;
    }
  }
  v29 = (unsigned int)v160;
  v30 = v123;
  if ( !(_DWORD)v160 )
    goto LABEL_51;
  if ( HIDWORD(v151) - v152 == (_DWORD)v160 )
  {
    v52 = base;
    goto LABEL_92;
  }
  if ( !sub_991A70((unsigned __int8 *)a2, 0, 0, 0, 0, 1u, 0) )
  {
    for ( i = *(_QWORD *)(v123 + 56); ; i = *(_QWORD *)(i + 8) )
    {
      if ( i )
      {
        v33 = (char *)(i - 24);
        if ( a2 == i - 24 )
          break;
      }
      else
      {
        v33 = 0;
      }
      if ( !(unsigned __int8)sub_98CD80(v33) )
        goto LABEL_51;
    }
  }
  v83 = (unsigned int)v160;
  v84 = (unsigned int)(HIDWORD(v151) - v152);
  v85 = v160;
  v86 = v84;
  if ( v84 == (unsigned int)v160 + 1LL )
  {
    v132 = HIDWORD(v151) - v152;
    v115 = sub_986580(v133);
    v116 = sub_B46E30(v115);
    v86 = v132;
    if ( v116 == 1 )
    {
LABEL_164:
      if ( !v133 )
      {
        v52 = base;
        v29 = (unsigned int)v160;
        goto LABEL_92;
      }
      v122 = *(_QWORD *)(a2 + 8);
      v124 = sub_BD5BF0(v119, v30, v133);
      v99 = (char *)sub_BD5D20(a2);
      v156 = (_OWORD *)v100;
      LOWORD(v100) = *(_WORD *)(a2 + 2);
      v158.m128i_i16[0] = 773;
      v155 = v99;
      v157.m128i_i64[0] = (__int64)".pr";
      _BitScanReverse64((unsigned __int64 *)&v99, 1LL << ((unsigned __int16)v100 >> 1));
      v101 = 63 - ((unsigned __int8)v99 ^ 0x3F);
      v126 = *(_BYTE *)(a2 + 72);
      v129 = ((unsigned __int16)v100 >> 7) & 7;
      v125 = sub_986580(v133) + 24;
      v102 = sub_BD2C40(80, unk_3F10A14);
      v105 = (__int64)v102;
      if ( v102 )
        sub_B4D0A0((__int64)v102, v122, v124, (__int64)&v155, 0, v101, v129, v126, v125, 0);
      v106 = *(const char **)(a2 + 48);
      v107 = (const char **)(v105 + 48);
      v155 = (char *)v106;
      if ( v106 )
      {
        sub_B96E90((__int64)&v155, (__int64)v106, 1);
        if ( v107 == (const char **)&v155 )
        {
          if ( v155 )
            sub_B91220((__int64)&v155, (__int64)v155);
          goto LABEL_171;
        }
        v113 = *(_QWORD *)(v105 + 48);
        if ( !v113 )
        {
LABEL_195:
          v114 = (unsigned __int8 *)v155;
          *(_QWORD *)(v105 + 48) = v155;
          if ( v114 )
            sub_B976B0((__int64)&v155, v114, v105 + 48);
LABEL_171:
          if ( v141.m128i_i64[0] || __PAIR128__(v141.m128i_u64[1], 0) != v142.m128i_u64[0] || v142.m128i_i64[1] )
            sub_B9A100(v105, v141.m128i_i64);
          v108 = v160;
          if ( (unsigned int)v160 >= (unsigned __int64)HIDWORD(v160) )
          {
            v117 = (unsigned int)v160 + 1LL;
            if ( HIDWORD(v160) < v117 )
              sub_C8D5F0((__int64)&base, v161, v117, 0x10u, v103, v104);
            v118 = (__int64 *)((char *)base + 16 * (unsigned int)v160);
            *v118 = v133;
            v118[1] = v105;
            v52 = base;
            v29 = (unsigned int)(v160 + 1);
            LODWORD(v160) = v160 + 1;
          }
          else
          {
            v52 = base;
            v109 = (__int64 *)((char *)base + 16 * (unsigned int)v160);
            if ( v109 )
            {
              v109[1] = v105;
              *v109 = v133;
              v108 = v160;
              v52 = base;
            }
            v29 = (unsigned int)(v108 + 1);
            LODWORD(v160) = v29;
          }
LABEL_92:
          if ( v29 > 1 )
            qsort(v52, (__int64)(16 * v29) >> 4, 0x10u, (__compar_fn_t)sub_27DB820);
          v158.m128i_i16[0] = 257;
          v53 = *(_QWORD *)(v30 + 16);
          do
          {
            if ( !v53 )
            {
              v134 = 0;
              goto LABEL_102;
            }
            v54 = **(_BYTE **)(v53 + 24);
            v55 = v53;
            v53 = *(_QWORD *)(v53 + 8);
          }
          while ( (unsigned __int8)(v54 - 30) > 0xAu );
          v56 = 0;
          while ( 1 )
          {
            v55 = *(_QWORD *)(v55 + 8);
            if ( !v55 )
              break;
            while ( (unsigned __int8)(**(_BYTE **)(v55 + 24) - 30) <= 0xAu )
            {
              v55 = *(_QWORD *)(v55 + 8);
              ++v56;
              if ( !v55 )
                goto LABEL_101;
            }
          }
LABEL_101:
          v134 = v56 + 1;
LABEL_102:
          v128 = *(_QWORD *)(a2 + 8);
          v57 = sub_BD2DA0(80);
          v58 = v57;
          if ( v57 )
          {
            v59 = (unsigned __int8 *)v57;
            sub_B44260(v57, v128, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v58 + 72) = v134;
            sub_BD6B50((unsigned __int8 *)v58, (const char **)&v155);
            sub_BD2A10(v58, *(_DWORD *)(v58 + 72), 1);
          }
          else
          {
            v59 = 0;
          }
          sub_B44220(v59, *(_QWORD *)(v30 + 56), 1);
          sub_BD6B90(v59, (unsigned __int8 *)a2);
          v60 = *(const char **)(a2 + 48);
          v155 = (char *)v60;
          if ( v60 )
          {
            v61 = v58 + 48;
            sub_B96E90((__int64)&v155, (__int64)v60, 1);
            if ( (char **)(v58 + 48) == &v155 )
            {
              if ( v155 )
                sub_B91220((__int64)&v155, (__int64)v155);
              goto LABEL_108;
            }
          }
          else
          {
            v61 = v58 + 48;
            if ( (char **)(v58 + 48) == &v155 )
            {
LABEL_108:
              v62 = *(_QWORD *)(v30 + 16);
              if ( v62 )
              {
                while ( 1 )
                {
                  v63 = *(_QWORD *)(v62 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v63 - 30) <= 0xAu )
                    break;
                  v62 = *(_QWORD *)(v62 + 8);
                  if ( !v62 )
                    goto LABEL_131;
                }
LABEL_110:
                v64 = *(_QWORD *)(v63 + 40);
                v65 = (__int64 *)base;
                for ( j = (unsigned int)v160; j > 0; j = j - v67 - 1 )
                {
                  while ( 1 )
                  {
                    v67 = j >> 1;
                    v68 = (unsigned __int64 *)&v65[2 * (j >> 1)];
                    if ( v64 > *v68 )
                      break;
                    j >>= 1;
                    if ( v67 <= 0 )
                      goto LABEL_114;
                  }
                  v65 = (__int64 *)(v68 + 2);
                }
LABEL_114:
                v69 = v65[1];
                v135 = *(_QWORD *)(a2 + 8);
                if ( *(_QWORD *)(v69 + 8) != v135 )
                {
                  v70 = sub_986580(v64);
                  v158.m128i_i16[0] = 257;
                  v69 = sub_B52260(v65[1], v135, (__int64)&v155, v70 + 24, 0);
                  v65[1] = v69;
                }
                v71 = *v65;
                v72 = *(_DWORD *)(v58 + 4) & 0x7FFFFFF;
                if ( v72 == *(_DWORD *)(v58 + 72) )
                {
                  v136 = v69;
                  sub_B48D90(v58);
                  v69 = v136;
                  v72 = *(_DWORD *)(v58 + 4) & 0x7FFFFFF;
                }
                v73 = (v72 + 1) & 0x7FFFFFF;
                v74 = v73 | *(_DWORD *)(v58 + 4) & 0xF8000000;
                v75 = *(_QWORD *)(v58 - 8) + 32LL * (unsigned int)(v73 - 1);
                *(_DWORD *)(v58 + 4) = v74;
                if ( *(_QWORD *)v75 )
                {
                  v76 = *(_QWORD *)(v75 + 8);
                  **(_QWORD **)(v75 + 16) = v76;
                  if ( v76 )
                    *(_QWORD *)(v76 + 16) = *(_QWORD *)(v75 + 16);
                }
                *(_QWORD *)v75 = v69;
                if ( v69 )
                {
                  v77 = *(_QWORD *)(v69 + 16);
                  *(_QWORD *)(v75 + 8) = v77;
                  if ( v77 )
                    *(_QWORD *)(v77 + 16) = v75 + 8;
                  *(_QWORD *)(v75 + 16) = v69 + 16;
                  *(_QWORD *)(v69 + 16) = v75;
                }
                *(_QWORD *)(*(_QWORD *)(v58 - 8)
                          + 32LL * *(unsigned int *)(v58 + 72)
                          + 8LL * ((*(_DWORD *)(v58 + 4) & 0x7FFFFFFu) - 1)) = v71;
                while ( 1 )
                {
                  v62 = *(_QWORD *)(v62 + 8);
                  if ( !v62 )
                    break;
                  v63 = *(_QWORD *)(v62 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v63 - 30) <= 0xAu )
                    goto LABEL_110;
                }
              }
LABEL_131:
              v78 = (_BYTE **)v143;
              if ( v143 != &v143[8 * (unsigned int)v144] )
              {
                v79 = (_BYTE **)&v143[8 * (unsigned int)v144];
                do
                {
                  v80 = *v78++;
                  sub_F57030(v80, a2, 1);
                  sub_22C1820(a1[4], (__int64)v80);
                }
                while ( v79 != v78 );
              }
              v4 = 1;
              sub_BD84D0(a2, v58);
              sub_B43D60((_QWORD *)a2);
              goto LABEL_52;
            }
          }
          v81 = *(_QWORD *)(v58 + 48);
          if ( v81 )
            sub_B91220(v61, v81);
          v82 = (unsigned __int8 *)v155;
          *(_QWORD *)(v58 + 48) = v155;
          if ( v82 )
            sub_B976B0((__int64)&v155, v82, v61);
          goto LABEL_108;
        }
      }
      else
      {
        if ( v107 == (const char **)&v155 )
          goto LABEL_171;
        v113 = *(_QWORD *)(v105 + 48);
        if ( !v113 )
          goto LABEL_171;
      }
      sub_B91220(v105 + 48, v113);
      goto LABEL_195;
    }
  }
  v52 = base;
  if ( (_DWORD)v86 == v85 )
  {
    v29 = (unsigned int)v160;
    goto LABEL_92;
  }
  v155 = 0;
  v87 = 1;
  v88 = (char *)base + 16 * v83;
  v146 = (__int64 **)v148;
  v147 = 0x800000000LL;
  v156 = &v158;
  v157.m128i_i64[0] = 8;
  v157.m128i_i32[2] = 0;
  v157.m128i_i8[12] = 1;
  if ( v88 == base )
    goto LABEL_152;
  v89 = (__int64 *)base;
  while ( 2 )
  {
    v90 = *v89;
    if ( !(_BYTE)v87 )
      goto LABEL_179;
    v91 = v156;
    v84 = (__int64)v156 + 8 * v157.m128i_u32[1];
    if ( v156 == (_OWORD *)v84 )
    {
LABEL_180:
      if ( v157.m128i_i32[1] < (unsigned __int32)v157.m128i_i32[0] )
      {
        ++v157.m128i_i32[1];
        *(_QWORD *)v84 = v90;
        v87 = v157.m128i_u8[12];
        ++v155;
        goto LABEL_150;
      }
LABEL_179:
      sub_C8CC70((__int64)&v155, v90, v84, v86, v87, v31);
      v87 = v157.m128i_u8[12];
      goto LABEL_150;
    }
    while ( v90 != *v91 )
    {
      if ( (_QWORD *)v84 == ++v91 )
        goto LABEL_180;
    }
LABEL_150:
    v89 += 2;
    if ( v88 != (char *)v89 )
      continue;
    break;
  }
  v30 = v123;
LABEL_152:
  v92 = *(_QWORD *)(v30 + 16);
  if ( !v92 )
  {
LABEL_160:
    v133 = sub_27E3B20(a1, v30, v146, (unsigned int)v147, "thread-pre-split");
    if ( !v157.m128i_i8[12] )
      _libc_free((unsigned __int64)v156);
    if ( v146 != (__int64 **)v148 )
      _libc_free((unsigned __int64)v146);
    goto LABEL_164;
  }
  while ( 1 )
  {
    v93 = *(_QWORD *)(v92 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v93 - 30) <= 0xAu )
      break;
    v92 = *(_QWORD *)(v92 + 8);
    if ( !v92 )
      goto LABEL_160;
  }
  v137 = v30;
LABEL_156:
  v94 = *(__int64 **)(v93 + 40);
  if ( *(_BYTE *)sub_986580((__int64)v94) != 33 )
  {
    if ( !(unsigned __int8)sub_B19060((__int64)&v155, (__int64)v94, v95, v96) )
    {
      v110 = (unsigned int)v147;
      v111 = (unsigned int)v147 + 1LL;
      if ( v111 > HIDWORD(v147) )
      {
        sub_C8D5F0((__int64)&v146, v148, v111, 8u, v97, v98);
        v110 = (unsigned int)v147;
      }
      v146[v110] = v94;
      LODWORD(v147) = v147 + 1;
    }
    while ( 1 )
    {
      v92 = *(_QWORD *)(v92 + 8);
      if ( !v92 )
        break;
      v93 = *(_QWORD *)(v92 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v93 - 30) <= 0xAu )
        goto LABEL_156;
    }
    v30 = v137;
    goto LABEL_160;
  }
  if ( !v157.m128i_i8[12] )
    _libc_free((unsigned __int64)v156);
  if ( v146 != (__int64 **)v148 )
    _libc_free((unsigned __int64)v146);
LABEL_51:
  v4 = 0;
LABEL_52:
  if ( v143 != v145 )
    _libc_free((unsigned __int64)v143);
LABEL_54:
  if ( base != v161 )
    _libc_free((unsigned __int64)base);
  if ( !v153 )
    _libc_free((unsigned __int64)v150);
LABEL_25:
  v172[0] = &unk_49DDBE8;
  if ( (v173 & 1) == 0 )
    sub_C7D6A0(v174, 16LL * v175, 8);
  nullsub_184();
  if ( v169 != v171 )
    _libc_free((unsigned __int64)v169);
  if ( (v164 & 1) == 0 )
    sub_C7D6A0(v165, 40LL * v166, 8);
  return v4;
}
