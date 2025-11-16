// Function: sub_184B020
// Address: 0x184b020
//
__int64 __fastcall sub_184B020(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 (__fastcall *v3)(__int64 *, __int64 *, int); // rax
  __int64 m128i_i64; // rsi
  int v5; // eax
  __int8 *v6; // rbx
  void (__fastcall *v7)(_BYTE *, _BYTE *, __int64); // rcx
  __m128i *v8; // rdi
  unsigned int v9; // r12d
  __m128i *v10; // r14
  unsigned int v11; // r13d
  __m128i *v12; // rbx
  unsigned __int64 v13; // r12
  void (__fastcall *v14)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v15)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v16)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 (__fastcall *v20)(__int64 *, __int64 *, int); // rax
  int v21; // eax
  __int8 *v22; // r12
  void (__fastcall *v23)(_BYTE *, _BYTE *, __int64); // rcx
  __m128i *v24; // r13
  __m128i *v25; // r15
  void (__fastcall *v26)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v27)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v28)(__m128i *, __m128i *, __int64); // rax
  __int64 *v29; // rax
  __m128i *v30; // r15
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __m128i *v36; // r13
  __m128i *v37; // r12
  __m128i *v38; // r15
  __m128i v39; // xmm0
  __int64 v40; // rdx
  __int64 v41; // rax
  __m128i v42; // xmm0
  __m128i *v43; // rsi
  __m128i v44; // xmm0
  __int64 v45; // rsi
  __int64 v46; // rdx
  __m128i v47; // xmm0
  __m128i *v48; // rax
  __m128i v49; // xmm0
  __int64 v50; // rsi
  __m128i v51; // xmm0
  __m128i *v52; // rax
  __int64 v53; // r8
  __int64 v54; // r13
  __m128i *v55; // r12
  unsigned __int32 v56; // eax
  __int64 v57; // r14
  void (__fastcall *v58)(__int64, __int64, __int64); // rax
  void (__fastcall *v59)(__int64, __int64, __int64); // rax
  void (__fastcall *v60)(__int64, __int64, __int64); // rax
  __int64 v61; // rdx
  __m128i **v62; // r15
  __int64 v63; // rax
  __m128i *v64; // rbx
  __m128i *v65; // r12
  __m128i *v66; // rbx
  void (__fastcall *v67)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v68)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v69)(__m128i *, __m128i *, __int64); // rax
  char v70; // al
  __m128i *v71; // r12
  char v72; // al
  char v73; // al
  char v74; // al
  char v75; // al
  __m128i *v76; // rax
  __int64 v77; // r14
  __int64 v78; // r13
  __int64 v79; // rcx
  __m128i *v80; // r12
  __m128i *v81; // rbx
  __int64 v82; // rax
  __int64 v83; // r14
  __m128i *v84; // r14
  __m128i v85; // xmm0
  __int64 v86; // rdx
  __int64 v87; // rax
  __m128i v88; // xmm0
  __int64 (__fastcall *v89)(_BYTE *, __int64, int); // rsi
  __m128i v90; // xmm0
  __int64 v91; // rsi
  __int64 v92; // rdx
  __m128i v93; // xmm0
  __int64 (__fastcall *v94)(_BYTE *, __int64, int); // rax
  __m128i v95; // xmm0
  __int64 v96; // rsi
  __m128i v97; // xmm0
  __int64 (__fastcall *v98)(_BYTE *, __int64, int); // rax
  __int64 v99; // r14
  __int64 v100; // r12
  __m128i *v101; // r14
  __m128i v102; // xmm0
  __int64 v103; // rdx
  __int64 v104; // rax
  __m128i v105; // xmm0
  __int64 (__fastcall *v106)(_BYTE *, __int64, int); // rcx
  __m128i v107; // xmm0
  __int64 v108; // rcx
  __int64 v109; // rdx
  __m128i v110; // xmm0
  __int64 (__fastcall *v111)(_BYTE *, __int64, int); // rax
  __m128i v112; // xmm0
  __int64 v113; // rcx
  __int64 v114; // rdx
  __m128i v115; // xmm0
  __int64 (__fastcall *v116)(_BYTE *, __int64, int); // rax
  __int32 v117; // eax
  __int64 v118; // rax
  __int64 v119; // r12
  void (__fastcall *v120)(__int64, __int64, __int64); // rax
  void (__fastcall *v121)(__int64, __int64, __int64); // rax
  void (__fastcall *v122)(__int64, __int64, __int64); // rax
  __m128i *i; // rax
  __int64 *v124; // rcx
  __m128i *v125; // rbx
  void (__fastcall *v126)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v127)(__m128i *, __m128i *, __int64); // rax
  void (__fastcall *v128)(__m128i *, __m128i *, __int64); // rax
  __m128i *v129; // r14
  char v130; // al
  char v131; // al
  char v132; // al
  __int64 *v134; // [rsp+18h] [rbp-5E8h]
  __int64 v135; // [rsp+30h] [rbp-5D0h]
  __m128i *v136; // [rsp+38h] [rbp-5C8h]
  __m128i *v137; // [rsp+40h] [rbp-5C0h]
  __m128i *v138; // [rsp+40h] [rbp-5C0h]
  __int64 *v139; // [rsp+48h] [rbp-5B8h]
  __m128i *v140; // [rsp+50h] [rbp-5B0h]
  __int64 v141; // [rsp+50h] [rbp-5B0h]
  __m128i **v142; // [rsp+58h] [rbp-5A8h]
  _BYTE v143[16]; // [rsp+60h] [rbp-5A0h] BYREF
  __int64 (__fastcall *v144)(__m128i **, __int64, int); // [rsp+70h] [rbp-590h]
  __int64 (__fastcall *v145)(__int64, __int64); // [rsp+78h] [rbp-588h]
  _QWORD v146[2]; // [rsp+80h] [rbp-580h] BYREF
  __int64 (__fastcall *v147)(__int64 *, __int64 *, int); // [rsp+90h] [rbp-570h]
  void *v148; // [rsp+98h] [rbp-568h]
  __m128i v149; // [rsp+A0h] [rbp-560h] BYREF
  __int64 (__fastcall *v150)(_BYTE *, __int64, int); // [rsp+B0h] [rbp-550h]
  __int64 (__fastcall *v151)(__int64, __int64); // [rsp+B8h] [rbp-548h]
  __m128i *v152; // [rsp+C0h] [rbp-540h] BYREF
  __int64 v153; // [rsp+C8h] [rbp-538h]
  _BYTE v154[416]; // [rsp+D0h] [rbp-530h] BYREF
  __m128i *v155; // [rsp+270h] [rbp-390h] BYREF
  __int64 v156; // [rsp+278h] [rbp-388h]
  void (__fastcall *v157)(__int8 *, __m128i **, __int64); // [rsp+280h] [rbp-380h] BYREF
  __int64 (__fastcall *v158)(__int64, __int64); // [rsp+288h] [rbp-378h]
  _BYTE v159[16]; // [rsp+290h] [rbp-370h] BYREF
  __int64 (__fastcall *v160)(__int64 *, __int64 *, int); // [rsp+2A0h] [rbp-360h]
  void *v161; // [rsp+2A8h] [rbp-358h]
  _BYTE v162[16]; // [rsp+2B0h] [rbp-350h] BYREF
  void (__fastcall *v163)(_BYTE *, _BYTE *, __int64); // [rsp+2C0h] [rbp-340h]
  __int64 (__fastcall *v164)(__int64, __int64); // [rsp+2C8h] [rbp-338h]
  int v165; // [rsp+2D0h] [rbp-330h]
  char v166; // [rsp+2D4h] [rbp-32Ch]
  __m128i v167; // [rsp+420h] [rbp-1E0h] BYREF
  __m128i *v168; // [rsp+430h] [rbp-1D0h] BYREF
  __int64 v169; // [rsp+438h] [rbp-1C8h]
  __m128i *v170; // [rsp+470h] [rbp-190h]
  _BYTE v171[384]; // [rsp+480h] [rbp-180h] BYREF

  v152 = (__m128i *)v154;
  v153 = 0x400000000LL;
  v145 = sub_1848150;
  v144 = (__int64 (__fastcall *)(__m128i **, __int64, int))sub_1848110;
  sub_184ACC0((__int64)&v167, a1);
  v147 = 0;
  v1 = sub_22077B0(160);
  v2 = v1;
  if ( v1 )
    sub_184ACC0(v1, (__int64)&v167);
  v146[0] = v2;
  v151 = sub_1848180;
  v150 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_1848120;
  v148 = sub_18498D0;
  v3 = sub_184AF70;
  v147 = sub_184AF70;
  v157 = 0;
  if ( v144 )
  {
    m128i_i64 = (__int64)v143;
    v144(&v155, (__int64)v143, 2);
    v160 = 0;
    v158 = v145;
    v157 = (void (__fastcall *)(__int8 *, __m128i **, __int64))v144;
    v3 = v147;
    if ( !v147 )
      goto LABEL_6;
  }
  else
  {
    v160 = 0;
  }
  m128i_i64 = (__int64)v146;
  v3((__int64 *)v159, v146, 2);
  v161 = v148;
  v160 = v147;
LABEL_6:
  v163 = 0;
  if ( v150 )
  {
    m128i_i64 = (__int64)&v149;
    v150(v162, (__int64)&v149, 2);
    v164 = v151;
    v163 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v150;
  }
  v165 = 8;
  v5 = v153;
  v166 = 0;
  if ( (unsigned int)v153 >= HIDWORD(v153) )
  {
    m128i_i64 = 0;
    sub_1849670((__int64)&v152, 0);
    v5 = v153;
  }
  v6 = &v152->m128i_i8[104 * v5];
  if ( v6 )
  {
    *((_QWORD *)v6 + 2) = 0;
    if ( v157 )
    {
      m128i_i64 = (__int64)&v155;
      v157(v6, &v155, 2);
      *((_QWORD *)v6 + 3) = v158;
      *((_QWORD *)v6 + 2) = v157;
    }
    *((_QWORD *)v6 + 6) = 0;
    if ( v160 )
    {
      m128i_i64 = (__int64)v159;
      v160((__int64 *)v6 + 4, (__int64 *)v159, 2);
      *((_QWORD *)v6 + 7) = v161;
      *((_QWORD *)v6 + 6) = v160;
    }
    *((_QWORD *)v6 + 10) = 0;
    v7 = v163;
    if ( v163 )
    {
      m128i_i64 = (__int64)v162;
      v163(v6 + 64, v162, 2);
      *((_QWORD *)v6 + 11) = v164;
      *((_QWORD *)v6 + 10) = v163;
      v7 = v163;
    }
    *((_DWORD *)v6 + 24) = v165;
    v6[100] = v166;
    v5 = v153;
  }
  else
  {
    v7 = v163;
  }
  LODWORD(v153) = v5 + 1;
  if ( v7 )
  {
    m128i_i64 = (__int64)v162;
    v7(v162, v162, 3);
  }
  if ( v160 )
  {
    m128i_i64 = (__int64)v159;
    v160((__int64 *)v159, (__int64 *)v159, 3);
  }
  if ( v157 )
  {
    m128i_i64 = (__int64)&v155;
    v157((__int8 *)&v155, &v155, 3);
  }
  if ( v150 )
  {
    m128i_i64 = (__int64)&v149;
    v150(&v149, (__int64)&v149, 3);
  }
  if ( v147 )
  {
    m128i_i64 = (__int64)v146;
    v147(v146, v146, 3);
  }
  v8 = v170;
  if ( v170 != (__m128i *)v171 )
    _libc_free((unsigned __int64)v170);
  if ( (v167.m128i_i8[8] & 1) == 0 )
  {
    v8 = v168;
    j___libc_free_0(v168);
  }
  if ( v144 )
  {
    v8 = (__m128i *)v143;
    m128i_i64 = (__int64)v143;
    v144((__m128i **)v143, (__int64)v143, 3);
  }
  if ( byte_4FAA660 )
    goto LABEL_35;
  v145 = sub_1848170;
  v144 = (__int64 (__fastcall *)(__m128i **, __int64, int))sub_1848130;
  sub_184ACC0((__int64)&v167, a1);
  v147 = 0;
  v18 = sub_22077B0(160);
  v19 = v18;
  if ( v18 )
    sub_184ACC0(v18, (__int64)&v167);
  v146[0] = v19;
  v151 = sub_18481A0;
  v150 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_1848140;
  v148 = sub_1848CC0;
  v20 = sub_184AEC0;
  v147 = sub_184AEC0;
  v157 = 0;
  if ( !v144 )
  {
    v160 = 0;
    goto LABEL_54;
  }
  m128i_i64 = (__int64)v143;
  v144(&v155, (__int64)v143, 2);
  v160 = 0;
  v158 = v145;
  v157 = (void (__fastcall *)(__int8 *, __m128i **, __int64))v144;
  v20 = v147;
  if ( v147 )
  {
LABEL_54:
    m128i_i64 = (__int64)v146;
    v20((__int64 *)v159, v146, 2);
    v161 = v148;
    v160 = v147;
  }
  v163 = 0;
  if ( v150 )
  {
    m128i_i64 = (__int64)&v149;
    v150(v162, (__int64)&v149, 2);
    v164 = v151;
    v163 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v150;
  }
  v165 = 30;
  v21 = v153;
  v166 = 1;
  if ( (unsigned int)v153 >= HIDWORD(v153) )
  {
    m128i_i64 = 0;
    sub_1849670((__int64)&v152, 0);
    v21 = v153;
  }
  v22 = &v152->m128i_i8[104 * v21];
  if ( v22 )
  {
    *((_QWORD *)v22 + 2) = 0;
    if ( v157 )
    {
      m128i_i64 = (__int64)&v155;
      v157(v22, &v155, 2);
      *((_QWORD *)v22 + 3) = v158;
      *((_QWORD *)v22 + 2) = v157;
    }
    *((_QWORD *)v22 + 6) = 0;
    if ( v160 )
    {
      m128i_i64 = (__int64)v159;
      v160((__int64 *)v22 + 4, (__int64 *)v159, 2);
      *((_QWORD *)v22 + 7) = v161;
      *((_QWORD *)v22 + 6) = v160;
    }
    *((_QWORD *)v22 + 10) = 0;
    v23 = v163;
    if ( v163 )
    {
      m128i_i64 = (__int64)v162;
      v163(v22 + 64, v162, 2);
      *((_QWORD *)v22 + 11) = v164;
      *((_QWORD *)v22 + 10) = v163;
      v23 = v163;
    }
    *((_DWORD *)v22 + 24) = v165;
    v22[100] = v166;
    v21 = v153;
  }
  else
  {
    v23 = v163;
  }
  LODWORD(v153) = v21 + 1;
  if ( v23 )
  {
    m128i_i64 = (__int64)v162;
    v23(v162, v162, 3);
  }
  if ( v160 )
  {
    m128i_i64 = (__int64)v159;
    v160((__int64 *)v159, (__int64 *)v159, 3);
  }
  if ( v157 )
  {
    m128i_i64 = (__int64)&v155;
    v157((__int8 *)&v155, &v155, 3);
  }
  if ( v150 )
  {
    m128i_i64 = (__int64)&v149;
    v150(&v149, (__int64)&v149, 3);
  }
  if ( v147 )
  {
    m128i_i64 = (__int64)v146;
    v147(v146, v146, 3);
  }
  v8 = v170;
  if ( v170 != (__m128i *)v171 )
    _libc_free((unsigned __int64)v170);
  if ( (v167.m128i_i8[8] & 1) == 0 )
  {
    v8 = v168;
    j___libc_free_0(v168);
  }
  if ( v144 )
  {
    v8 = (__m128i *)v143;
    m128i_i64 = (__int64)v143;
    v144((__m128i **)v143, (__int64)v143, 3);
  }
LABEL_35:
  v9 = v153;
  v155 = (__m128i *)&v157;
  v156 = 0x400000000LL;
  if ( !(_DWORD)v153 )
  {
    v10 = (__m128i *)&v157;
    v11 = 0;
    if ( !*(_DWORD *)(a1 + 88) )
      goto LABEL_37;
    goto LABEL_316;
  }
  if ( (unsigned int)v153 > 4 )
  {
    v8 = (__m128i *)&v155;
    m128i_i64 = (unsigned int)v153;
    sub_1849670((__int64)&v155, (unsigned int)v153);
    v10 = v155;
  }
  else
  {
    v10 = (__m128i *)&v157;
  }
  v24 = v152;
  v25 = (__m128i *)((char *)v152 + 104 * (unsigned int)v153);
  if ( v152 != v25 )
  {
    do
    {
      if ( v10 )
      {
        v10[1].m128i_i64[0] = 0;
        v26 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v24[1].m128i_i64[0];
        if ( v26 )
        {
          m128i_i64 = (__int64)v24;
          v8 = v10;
          v26(v10, v24, 2);
          v10[1].m128i_i64[1] = v24[1].m128i_i64[1];
          v10[1].m128i_i64[0] = v24[1].m128i_i64[0];
        }
        v10[3].m128i_i64[0] = 0;
        v27 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v24[3].m128i_i64[0];
        if ( v27 )
        {
          m128i_i64 = (__int64)v24[2].m128i_i64;
          v8 = v10 + 2;
          v27(v10 + 2, v24 + 2, 2);
          v10[3].m128i_i64[1] = v24[3].m128i_i64[1];
          v10[3].m128i_i64[0] = v24[3].m128i_i64[0];
        }
        v10[5].m128i_i64[0] = 0;
        v28 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v24[5].m128i_i64[0];
        if ( v28 )
        {
          m128i_i64 = (__int64)v24[4].m128i_i64;
          v8 = v10 + 4;
          v28(v10 + 4, v24 + 4, 2);
          v10[5].m128i_i64[1] = v24[5].m128i_i64[1];
          v10[5].m128i_i64[0] = v24[5].m128i_i64[0];
        }
        v10[6].m128i_i32[0] = v24[6].m128i_i32[0];
        v10[6].m128i_i8[4] = v24[6].m128i_i8[4];
      }
      v24 = (__m128i *)((char *)v24 + 104);
      v10 = (__m128i *)((char *)v10 + 104);
    }
    while ( v25 != v24 );
    v10 = v155;
  }
  LODWORD(v156) = v9;
  v29 = *(__int64 **)(a1 + 80);
  v134 = &v29[*(unsigned int *)(a1 + 88)];
  if ( v29 == v134 )
  {
    v11 = 0;
    v66 = (__m128i *)((char *)v10 + 104 * v9);
LABEL_154:
    if ( v66 != v10 )
    {
      do
      {
        v67 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v66[-2].m128i_i64[1];
        v66 = (__m128i *)((char *)v66 - 104);
        if ( v67 )
          v67(v66 + 4, v66 + 4, 3);
        v68 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v66[3].m128i_i64[0];
        if ( v68 )
          v68(v66 + 2, v66 + 2, 3);
        v69 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v66[1].m128i_i64[0];
        if ( v69 )
          v69(v66, v66, 3);
      }
      while ( v66 != v10 );
      v10 = v155;
    }
    goto LABEL_163;
  }
  v139 = *(__int64 **)(a1 + 80);
  v30 = &v149;
  v31 = v9;
  do
  {
    v32 = *v139;
    if ( !(_DWORD)v31 )
      goto LABEL_316;
    v33 = 104 * v31;
    v140 = (__m128i *)((char *)v10 + v33);
    v34 = 0x4EC4EC4EC4EC4EC5LL * (v33 >> 3);
    v35 = v34 >> 2;
    if ( v34 >> 2 )
    {
      v36 = &v10[26 * v35];
      while ( 1 )
      {
        if ( !v10[1].m128i_i64[0] )
          goto LABEL_313;
        m128i_i64 = v32;
        v8 = v10;
        if ( !((unsigned __int8 (__fastcall *)(__m128i *, __int64))v10[1].m128i_i64[1])(v10, v32) )
        {
          v8 = (__m128i *)v32;
          if ( sub_15E4F60(v32) )
            goto LABEL_106;
          if ( v10[6].m128i_i8[4] )
          {
            v8 = (__m128i *)v32;
            if ( sub_15E4F60(v32) )
              goto LABEL_106;
            v8 = (__m128i *)v32;
            sub_15E4B50(v32);
            if ( v70 )
              goto LABEL_106;
          }
        }
        v71 = (__m128i *)((char *)v10 + 104);
        if ( !v10[7].m128i_i64[1] )
          goto LABEL_313;
        m128i_i64 = v32;
        v8 = (__m128i *)((char *)v10 + 104);
        if ( !((unsigned __int8 (__fastcall *)(__int64 *, __int64))v10[8].m128i_i64[0])(&v10[6].m128i_i64[1], v32) )
        {
          v8 = (__m128i *)v32;
          if ( sub_15E4F60(v32)
            || v10[12].m128i_i8[12]
            && ((v8 = (__m128i *)v32, sub_15E4F60(v32)) || (v8 = (__m128i *)v32, sub_15E4B50(v32), v72)) )
          {
LABEL_188:
            v10 = v71;
            goto LABEL_106;
          }
        }
        v71 = v10 + 13;
        if ( !v10[14].m128i_i64[0] )
          goto LABEL_313;
        m128i_i64 = v32;
        v8 = v10 + 13;
        if ( !((unsigned __int8 (__fastcall *)(__int64 *, __int64))v10[14].m128i_i64[1])(v10[13].m128i_i64, v32) )
        {
          v8 = (__m128i *)v32;
          if ( sub_15E4F60(v32) )
            goto LABEL_188;
          if ( v10[19].m128i_i8[4] )
          {
            v8 = (__m128i *)v32;
            if ( sub_15E4F60(v32) )
              goto LABEL_188;
            v8 = (__m128i *)v32;
            sub_15E4B50(v32);
            if ( v73 )
            {
              v10 += 13;
              goto LABEL_106;
            }
          }
        }
        v71 = (__m128i *)((char *)v10 + 312);
        if ( !v10[20].m128i_i64[1] )
          goto LABEL_313;
        m128i_i64 = v32;
        v8 = (__m128i *)((char *)v10 + 312);
        if ( !((unsigned __int8 (__fastcall *)(__int64 *, __int64))v10[21].m128i_i64[0])(&v10[19].m128i_i64[1], v32) )
        {
          v8 = (__m128i *)v32;
          if ( sub_15E4F60(v32) )
            goto LABEL_188;
          if ( v10[25].m128i_i8[12] )
          {
            v8 = (__m128i *)v32;
            if ( sub_15E4F60(v32) )
              goto LABEL_188;
            v8 = (__m128i *)v32;
            sub_15E4B50(v32);
            if ( v74 )
            {
              v10 = (__m128i *)((char *)v10 + 312);
              goto LABEL_106;
            }
          }
        }
        v10 += 26;
        if ( v36 == v10 )
        {
          v34 = 0x4EC4EC4EC4EC4EC5LL * (((char *)v140 - (char *)v10) >> 3);
          break;
        }
      }
    }
    if ( v34 == 2 )
      goto LABEL_300;
    if ( v34 == 3 )
    {
      if ( !v10[1].m128i_i64[0] )
        goto LABEL_313;
      m128i_i64 = v32;
      v8 = v10;
      if ( !((unsigned __int8 (__fastcall *)(__m128i *, __int64, __int64))v10[1].m128i_i64[1])(v10, v32, v35) )
      {
        v8 = (__m128i *)v32;
        if ( sub_15E4F60(v32) )
          goto LABEL_106;
        if ( v10[6].m128i_i8[4] )
        {
          v8 = (__m128i *)v32;
          if ( sub_15E4F60(v32) )
            goto LABEL_106;
          v8 = (__m128i *)v32;
          sub_15E4B50(v32);
          if ( v131 )
            goto LABEL_106;
        }
      }
      v10 = (__m128i *)((char *)v10 + 104);
LABEL_300:
      if ( !v10[1].m128i_i64[0] )
        goto LABEL_313;
      m128i_i64 = v32;
      v8 = v10;
      if ( !((unsigned __int8 (__fastcall *)(__m128i *, __int64, __int64))v10[1].m128i_i64[1])(v10, v32, v35) )
      {
        v8 = (__m128i *)v32;
        if ( sub_15E4F60(v32) )
          goto LABEL_106;
        if ( v10[6].m128i_i8[4] )
        {
          v8 = (__m128i *)v32;
          if ( sub_15E4F60(v32) )
            goto LABEL_106;
          v8 = (__m128i *)v32;
          sub_15E4B50(v32);
          if ( v132 )
            goto LABEL_106;
        }
      }
      v10 = (__m128i *)((char *)v10 + 104);
      goto LABEL_305;
    }
    if ( v34 != 1 )
      goto LABEL_185;
LABEL_305:
    if ( !v10[1].m128i_i64[0] )
      goto LABEL_313;
    m128i_i64 = v32;
    if ( ((unsigned __int8 (__fastcall *)(__m128i *, __int64, __int64))v10[1].m128i_i64[1])(v10, v32, v35)
      || (v8 = (__m128i *)v32, !sub_15E4F60(v32))
      && (!v10[6].m128i_i8[4]
       || (v8 = (__m128i *)v32, !sub_15E4F60(v32)) && (v8 = (__m128i *)v32, sub_15E4B50(v32), !v130)) )
    {
LABEL_185:
      v10 = v140;
      goto LABEL_122;
    }
LABEL_106:
    if ( v140 != v10 )
    {
      v37 = (__m128i *)((char *)v10 + 104);
      if ( v140 != (__m128i *)&v10[6].m128i_u64[1] )
      {
        v137 = v30;
        v38 = v10;
        do
        {
          if ( !v37[1].m128i_i64[0] )
            goto LABEL_313;
          m128i_i64 = v32;
          v8 = v37;
          if ( ((unsigned __int8 (__fastcall *)(__m128i *, __int64))v37[1].m128i_i64[1])(v37, v32)
            || (v8 = (__m128i *)v32, !sub_15E4F60(v32))
            && (!v37[6].m128i_i8[4]
             || (v8 = (__m128i *)v32, !sub_15E4F60(v32)) && (v8 = (__m128i *)v32, sub_15E4B50(v32), !v75)) )
          {
            v39 = _mm_loadu_si128(v37);
            *v37 = _mm_loadu_si128(&v167);
            v167 = v39;
            v40 = v37[1].m128i_i64[0];
            v41 = v37[1].m128i_i64[1];
            v37[1].m128i_i64[0] = 0;
            v37[1].m128i_i64[1] = v169;
            v42 = _mm_loadu_si128(&v167);
            v167 = _mm_loadu_si128(v38);
            v43 = (__m128i *)v38[1].m128i_i64[0];
            *v38 = v42;
            v168 = v43;
            v38[1].m128i_i64[0] = v40;
            v169 = v38[1].m128i_i64[1];
            v38[1].m128i_i64[1] = v41;
            if ( v168 )
            {
              v8 = &v167;
              ((void (__fastcall *)(__m128i *, __m128i *, __int64))v168)(&v167, &v167, 3);
            }
            v44 = _mm_loadu_si128(v37 + 2);
            v37[2] = _mm_loadu_si128(&v167);
            v167 = v44;
            v45 = v37[3].m128i_i64[0];
            v46 = v37[3].m128i_i64[1];
            v37[3].m128i_i64[0] = 0;
            v37[3].m128i_i64[1] = v169;
            v47 = _mm_loadu_si128(&v167);
            v167 = _mm_loadu_si128(v38 + 2);
            v48 = (__m128i *)v38[3].m128i_i64[0];
            v38[2] = v47;
            v168 = v48;
            v38[3].m128i_i64[0] = v45;
            v169 = v38[3].m128i_i64[1];
            v38[3].m128i_i64[1] = v46;
            if ( v48 )
            {
              v8 = &v167;
              ((void (__fastcall *)(__m128i *, __m128i *, __int64))v48)(&v167, &v167, 3);
            }
            v49 = _mm_loadu_si128(v37 + 4);
            v37[4] = _mm_loadu_si128(&v167);
            v167 = v49;
            v50 = v37[5].m128i_i64[0];
            v35 = v37[5].m128i_i64[1];
            v37[5].m128i_i64[0] = 0;
            v37[5].m128i_i64[1] = v169;
            v51 = _mm_loadu_si128(&v167);
            v167 = _mm_loadu_si128(v38 + 4);
            v52 = (__m128i *)v38[5].m128i_i64[0];
            v38[4] = v51;
            v168 = v52;
            v38[5].m128i_i64[0] = v50;
            m128i_i64 = v38[5].m128i_i64[1];
            v169 = m128i_i64;
            v38[5].m128i_i64[1] = v35;
            if ( v52 )
            {
              m128i_i64 = (__int64)&v167;
              v8 = &v167;
              ((void (__fastcall *)(__m128i *, __m128i *, __int64))v52)(&v167, &v167, 3);
            }
            v38 = (__m128i *)((char *)v38 + 104);
            v38[-1].m128i_i32[2] = v37[6].m128i_i32[0];
            v38[-1].m128i_i8[12] = v37[6].m128i_i8[4];
          }
          v37 = (__m128i *)((char *)v37 + 104);
        }
        while ( v140 != v37 );
        v10 = v38;
        v30 = v137;
      }
    }
LABEL_122:
    v8 = (__m128i *)&v155;
    m128i_i64 = (__int64)v10;
    sub_1848A70((__int64 *)&v155, v10, v140);
    v53 = (__int64)v155;
    v167.m128i_i64[0] = (__int64)&v168;
    v54 = (__int64)v155;
    v167.m128i_i64[1] = 0x400000000LL;
    v35 = 3LL * (unsigned int)v156;
    v55 = (__m128i *)((char *)v155 + 104 * (unsigned int)v156);
    if ( v155 == v55 )
      goto LABEL_142;
    do
    {
      while ( 1 )
      {
        if ( !*(_QWORD *)(v54 + 16) )
          goto LABEL_313;
        m128i_i64 = v32;
        v8 = (__m128i *)v54;
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(v54 + 24))(v54, v32) )
          break;
        v54 += 104;
        if ( v55 == (__m128i *)v54 )
          goto LABEL_138;
      }
      v56 = v167.m128i_u32[2];
      if ( v167.m128i_i32[2] >= (unsigned __int32)v167.m128i_i32[3] )
      {
        v8 = &v167;
        m128i_i64 = 0;
        sub_1849670((__int64)&v167, 0);
        v56 = v167.m128i_u32[2];
      }
      v35 = v167.m128i_i64[0];
      v57 = v167.m128i_i64[0] + 104LL * v56;
      if ( v57 )
      {
        *(_QWORD *)(v57 + 16) = 0;
        v58 = *(void (__fastcall **)(__int64, __int64, __int64))(v54 + 16);
        if ( v58 )
        {
          m128i_i64 = v54;
          v8 = (__m128i *)v57;
          v58(v57, v54, 2);
          *(_QWORD *)(v57 + 24) = *(_QWORD *)(v54 + 24);
          *(_QWORD *)(v57 + 16) = *(_QWORD *)(v54 + 16);
        }
        *(_QWORD *)(v57 + 48) = 0;
        v59 = *(void (__fastcall **)(__int64, __int64, __int64))(v54 + 48);
        if ( v59 )
        {
          m128i_i64 = v54 + 32;
          v8 = (__m128i *)(v57 + 32);
          v59(v57 + 32, v54 + 32, 2);
          *(_QWORD *)(v57 + 56) = *(_QWORD *)(v54 + 56);
          *(_QWORD *)(v57 + 48) = *(_QWORD *)(v54 + 48);
        }
        *(_QWORD *)(v57 + 80) = 0;
        v60 = *(void (__fastcall **)(__int64, __int64, __int64))(v54 + 80);
        if ( v60 )
        {
          m128i_i64 = v54 + 64;
          v8 = (__m128i *)(v57 + 64);
          v60(v57 + 64, v54 + 64, 2);
          *(_QWORD *)(v57 + 88) = *(_QWORD *)(v54 + 88);
          *(_QWORD *)(v57 + 80) = *(_QWORD *)(v54 + 80);
        }
        *(_DWORD *)(v57 + 96) = *(_DWORD *)(v54 + 96);
        *(_BYTE *)(v57 + 100) = *(_BYTE *)(v54 + 100);
        v56 = v167.m128i_u32[2];
      }
      v54 += 104;
      v167.m128i_i32[2] = v56 + 1;
    }
    while ( v55 != (__m128i *)v54 );
LABEL_138:
    v61 = v167.m128i_u32[2];
    if ( !v167.m128i_i32[2] )
    {
      v8 = (__m128i *)v167.m128i_i64[0];
      if ( (__m128i **)v167.m128i_i64[0] != &v168 )
        _libc_free(v167.m128i_u64[0]);
LABEL_141:
      v53 = (__int64)v155;
      goto LABEL_142;
    }
    v76 = *(__m128i **)(v32 + 80);
    v8 = (__m128i *)(v32 + 72);
    v136 = (__m128i *)(v32 + 72);
    v138 = v76;
    if ( (__m128i *)(v32 + 72) == v76 )
    {
      v141 = 0;
    }
    else
    {
      do
      {
        if ( !v76 )
LABEL_325:
          BUG();
        m128i_i64 = v76[1].m128i_i64[1];
        if ( (__m128i *)m128i_i64 != &v76[1] )
          break;
        v76 = (__m128i *)v76->m128i_i64[1];
      }
      while ( v8 != v76 );
      v138 = v76;
      v141 = m128i_i64;
    }
    v77 = v167.m128i_i64[0];
    if ( v136 != v138 )
    {
      v78 = v167.m128i_i64[0];
      while ( 1 )
      {
        v79 = 0x4EC4EC4EC4EC4EC5LL;
        v80 = (__m128i *)(v141 - 24);
        if ( !v141 )
          v80 = 0;
        v81 = (__m128i *)(v78 + 104 * v61);
        v82 = 0x4EC4EC4EC4EC4EC5LL * ((104 * v61) >> 3);
        v35 = v82 >> 2;
        if ( !(v82 >> 2) )
          break;
        v83 = v78 + 416 * v35;
        while ( 1 )
        {
          if ( !*(_QWORD *)(v78 + 48) )
            goto LABEL_313;
          v8 = (__m128i *)(v78 + 32);
          m128i_i64 = (__int64)v80;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64, __int64))(v78 + 56))(
                 v78 + 32,
                 v80,
                 v35,
                 v79) )
          {
            goto LABEL_211;
          }
          if ( !*(_QWORD *)(v78 + 152) )
            goto LABEL_313;
          v8 = (__m128i *)(v78 + 136);
          m128i_i64 = (__int64)v80;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *))(v78 + 160))(v78 + 136, v80) )
          {
            v78 += 104;
            v8 = (__m128i *)&v155;
            m128i_i64 = v78;
            sub_1848D60((__int64)&v155, v78);
            goto LABEL_212;
          }
          if ( !*(_QWORD *)(v78 + 256) )
            goto LABEL_313;
          v8 = (__m128i *)(v78 + 240);
          m128i_i64 = (__int64)v80;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *))(v78 + 264))(v78 + 240, v80) )
            break;
          if ( !*(_QWORD *)(v78 + 360) )
            goto LABEL_313;
          v8 = (__m128i *)(v78 + 344);
          m128i_i64 = (__int64)v80;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *))(v78 + 368))(v78 + 344, v80) )
          {
            v78 += 312;
            v8 = (__m128i *)&v155;
            m128i_i64 = v78;
            sub_1848D60((__int64)&v155, v78);
            goto LABEL_212;
          }
          v78 += 416;
          if ( v83 == v78 )
          {
            v79 = 0x4EC4EC4EC4EC4EC5LL;
            v82 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v81->m128i_i64 - v78) >> 3);
            goto LABEL_237;
          }
        }
        v78 += 208;
        v8 = (__m128i *)&v155;
        m128i_i64 = v78;
        sub_1848D60((__int64)&v155, v78);
LABEL_212:
        if ( v81 != (__m128i *)v78 )
        {
          v84 = (__m128i *)(v78 + 104);
          if ( v81 != (__m128i *)(v78 + 104) )
          {
            while ( v84[3].m128i_i64[0] )
            {
              v8 = v84 + 2;
              if ( ((unsigned __int8 (__fastcall *)(__int64 *, __m128i *))v84[3].m128i_i64[1])(v84[2].m128i_i64, v80) )
              {
                v8 = (__m128i *)&v155;
                m128i_i64 = (__int64)v84;
                sub_1848D60((__int64)&v155, (__int64)v84);
              }
              else
              {
                v85 = _mm_loadu_si128(v84);
                *v84 = _mm_loadu_si128(&v149);
                v149 = v85;
                v86 = v84[1].m128i_i64[0];
                v87 = v84[1].m128i_i64[1];
                v84[1].m128i_i64[0] = 0;
                v84[1].m128i_i64[1] = (__int64)v151;
                v88 = _mm_loadu_si128(&v149);
                v149 = _mm_loadu_si128((const __m128i *)v78);
                v89 = *(__int64 (__fastcall **)(_BYTE *, __int64, int))(v78 + 16);
                *(__m128i *)v78 = v88;
                v150 = v89;
                *(_QWORD *)(v78 + 16) = v86;
                v151 = *(__int64 (__fastcall **)(__int64, __int64))(v78 + 24);
                *(_QWORD *)(v78 + 24) = v87;
                if ( v150 )
                {
                  v8 = v30;
                  v150(v30, (__int64)v30, 3);
                }
                v90 = _mm_loadu_si128(v84 + 2);
                v84[2] = _mm_loadu_si128(&v149);
                v149 = v90;
                v91 = v84[3].m128i_i64[0];
                v92 = v84[3].m128i_i64[1];
                v84[3].m128i_i64[0] = 0;
                v84[3].m128i_i64[1] = (__int64)v151;
                v93 = _mm_loadu_si128(&v149);
                v149 = _mm_loadu_si128((const __m128i *)(v78 + 32));
                v94 = *(__int64 (__fastcall **)(_BYTE *, __int64, int))(v78 + 48);
                *(__m128i *)(v78 + 32) = v93;
                v150 = v94;
                *(_QWORD *)(v78 + 48) = v91;
                v151 = *(__int64 (__fastcall **)(__int64, __int64))(v78 + 56);
                *(_QWORD *)(v78 + 56) = v92;
                if ( v94 )
                {
                  v8 = v30;
                  v94(v30, (__int64)v30, 3);
                }
                v95 = _mm_loadu_si128(v84 + 4);
                v84[4] = _mm_loadu_si128(&v149);
                v149 = v95;
                v96 = v84[5].m128i_i64[0];
                v35 = v84[5].m128i_i64[1];
                v84[5].m128i_i64[0] = 0;
                v84[5].m128i_i64[1] = (__int64)v151;
                v97 = _mm_loadu_si128(&v149);
                v149 = _mm_loadu_si128((const __m128i *)(v78 + 64));
                v98 = *(__int64 (__fastcall **)(_BYTE *, __int64, int))(v78 + 80);
                *(__m128i *)(v78 + 64) = v97;
                v150 = v98;
                *(_QWORD *)(v78 + 80) = v96;
                m128i_i64 = *(_QWORD *)(v78 + 88);
                v151 = (__int64 (__fastcall *)(__int64, __int64))m128i_i64;
                *(_QWORD *)(v78 + 88) = v35;
                if ( v98 )
                {
                  m128i_i64 = (__int64)v30;
                  v8 = v30;
                  v98(v30, (__int64)v30, 3);
                }
                v78 += 104;
                *(_DWORD *)(v78 - 8) = v84[6].m128i_i32[0];
                *(_BYTE *)(v78 - 4) = v84[6].m128i_i8[4];
              }
              v84 = (__m128i *)((char *)v84 + 104);
              if ( v81 == v84 )
                goto LABEL_241;
            }
LABEL_313:
            sub_4263D6(v8, m128i_i64, v35);
          }
        }
LABEL_241:
        m128i_i64 = 3LL * v167.m128i_u32[2];
        v99 = v167.m128i_i64[0] + 104LL * v167.m128i_u32[2];
        v135 = v99 - (_QWORD)v81;
        v100 = 0x4EC4EC4EC4EC4EC5LL * ((v99 - (__int64)v81) >> 3);
        if ( v99 - (__int64)v81 <= 0 )
        {
          v119 = v78;
          v78 = v167.m128i_i64[0];
        }
        else
        {
          v101 = (__m128i *)v78;
          do
          {
            v102 = _mm_loadu_si128(v81);
            *v81 = _mm_loadu_si128(&v149);
            v149 = v102;
            v103 = v81[1].m128i_i64[0];
            v104 = v81[1].m128i_i64[1];
            v81[1].m128i_i64[0] = 0;
            v81[1].m128i_i64[1] = (__int64)v151;
            v105 = _mm_loadu_si128(&v149);
            v149 = _mm_loadu_si128(v101);
            v106 = (__int64 (__fastcall *)(_BYTE *, __int64, int))v101[1].m128i_i64[0];
            *v101 = v105;
            v150 = v106;
            v101[1].m128i_i64[0] = v103;
            v151 = (__int64 (__fastcall *)(__int64, __int64))v101[1].m128i_i64[1];
            v101[1].m128i_i64[1] = v104;
            if ( v150 )
            {
              m128i_i64 = (__int64)v30;
              v8 = v30;
              v150(v30, (__int64)v30, 3);
            }
            v107 = _mm_loadu_si128(v81 + 2);
            v81[2] = _mm_loadu_si128(&v149);
            v149 = v107;
            v108 = v81[3].m128i_i64[0];
            v109 = v81[3].m128i_i64[1];
            v81[3].m128i_i64[0] = 0;
            v81[3].m128i_i64[1] = (__int64)v151;
            v110 = _mm_loadu_si128(&v149);
            v149 = _mm_loadu_si128(v101 + 2);
            v111 = (__int64 (__fastcall *)(_BYTE *, __int64, int))v101[3].m128i_i64[0];
            v101[2] = v110;
            v150 = v111;
            v101[3].m128i_i64[0] = v108;
            v151 = (__int64 (__fastcall *)(__int64, __int64))v101[3].m128i_i64[1];
            v101[3].m128i_i64[1] = v109;
            if ( v111 )
            {
              m128i_i64 = (__int64)v30;
              v8 = v30;
              v111(v30, (__int64)v30, 3);
            }
            v112 = _mm_loadu_si128(v81 + 4);
            v81[4] = _mm_loadu_si128(&v149);
            v149 = v112;
            v113 = v81[5].m128i_i64[0];
            v114 = v81[5].m128i_i64[1];
            v81[5].m128i_i64[0] = 0;
            v81[5].m128i_i64[1] = (__int64)v151;
            v115 = _mm_loadu_si128(&v149);
            v149 = _mm_loadu_si128(v101 + 4);
            v116 = (__int64 (__fastcall *)(_BYTE *, __int64, int))v101[5].m128i_i64[0];
            v101[4] = v115;
            v150 = v116;
            v101[5].m128i_i64[0] = v113;
            v151 = (__int64 (__fastcall *)(__int64, __int64))v101[5].m128i_i64[1];
            v101[5].m128i_i64[1] = v114;
            if ( v116 )
            {
              m128i_i64 = (__int64)v30;
              v8 = v30;
              v116(v30, (__int64)v30, 3);
            }
            v117 = v81[6].m128i_i32[0];
            v101 = (__m128i *)((char *)v101 + 104);
            v81 = (__m128i *)((char *)v81 + 104);
            v101[-1].m128i_i32[2] = v117;
            v101[-1].m128i_i8[12] = v81[-1].m128i_i8[12];
            --v100;
          }
          while ( v100 );
          v118 = v135;
          if ( v135 <= 0 )
            v118 = 104;
          v119 = v78 + v118;
          v78 = v167.m128i_i64[0];
          v99 = v167.m128i_i64[0] + 104LL * v167.m128i_u32[2];
        }
        if ( v99 != v119 )
        {
          do
          {
            v120 = *(void (__fastcall **)(__int64, __int64, __int64))(v99 - 24);
            v99 -= 104;
            if ( v120 )
            {
              v8 = (__m128i *)(v99 + 64);
              m128i_i64 = v99 + 64;
              v120(v99 + 64, v99 + 64, 3);
            }
            v121 = *(void (__fastcall **)(__int64, __int64, __int64))(v99 + 48);
            if ( v121 )
            {
              v8 = (__m128i *)(v99 + 32);
              m128i_i64 = v99 + 32;
              v121(v99 + 32, v99 + 32, 3);
            }
            v122 = *(void (__fastcall **)(__int64, __int64, __int64))(v99 + 16);
            if ( v122 )
            {
              m128i_i64 = v99;
              v8 = (__m128i *)v99;
              v122(v99, v99, 3);
            }
          }
          while ( v99 != v119 );
          v78 = v167.m128i_i64[0];
        }
        v167.m128i_i32[2] = -991146299 * ((v119 - v78) >> 3);
        v61 = v167.m128i_u32[2];
        if ( !v167.m128i_i32[2] )
        {
          v129 = (__m128i *)v78;
          goto LABEL_281;
        }
        v8 = v136;
        m128i_i64 = *(_QWORD *)(v141 + 8);
        for ( i = v138; ; m128i_i64 = i[1].m128i_i64[1] )
        {
          v124 = &i[-2].m128i_i64[1];
          if ( !i )
            v124 = 0;
          if ( (__int64 *)m128i_i64 != v124 + 5 )
            break;
          i = (__m128i *)i->m128i_i64[1];
          if ( v136 == i )
            goto LABEL_271;
          if ( !i )
            goto LABEL_325;
        }
        v138 = i;
        v141 = m128i_i64;
        if ( i == v136 )
        {
LABEL_271:
          v77 = v78;
          goto LABEL_272;
        }
      }
LABEL_237:
      if ( v82 == 2 )
        goto LABEL_287;
      if ( v82 != 3 )
      {
        if ( v82 != 1 )
          goto LABEL_240;
        goto LABEL_290;
      }
      if ( !*(_QWORD *)(v78 + 48) )
        goto LABEL_313;
      v8 = (__m128i *)(v78 + 32);
      m128i_i64 = (__int64)v80;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64, __int64))(v78 + 56))(
              v78 + 32,
              v80,
              v35,
              0x4EC4EC4EC4EC4EC5LL) )
      {
        v78 += 104;
LABEL_287:
        if ( !*(_QWORD *)(v78 + 48) )
          goto LABEL_313;
        v8 = (__m128i *)(v78 + 32);
        m128i_i64 = (__int64)v80;
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64, __int64))(v78 + 56))(
                v78 + 32,
                v80,
                v35,
                v79) )
        {
          v78 += 104;
LABEL_290:
          if ( !*(_QWORD *)(v78 + 48) )
            goto LABEL_313;
          v8 = (__m128i *)(v78 + 32);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64, __int64))(v78 + 56))(
                  v78 + 32,
                  v80,
                  v35,
                  v79) )
          {
LABEL_240:
            v78 = (__int64)v81;
            goto LABEL_241;
          }
        }
      }
LABEL_211:
      v8 = (__m128i *)&v155;
      m128i_i64 = v78;
      sub_1848D60((__int64)&v155, v78);
      goto LABEL_212;
    }
LABEL_272:
    v125 = (__m128i *)(v77 + 104LL * (unsigned int)v61);
    do
    {
      v126 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v125[-2].m128i_i64[1];
      v125 = (__m128i *)((char *)v125 - 104);
      if ( v126 )
      {
        v8 = v125 + 4;
        m128i_i64 = (__int64)v125[4].m128i_i64;
        v126(v125 + 4, v125 + 4, 3);
      }
      v127 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v125[3].m128i_i64[0];
      if ( v127 )
      {
        v8 = v125 + 2;
        m128i_i64 = (__int64)v125[2].m128i_i64;
        v127(v125 + 2, v125 + 2, 3);
      }
      v128 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v125[1].m128i_i64[0];
      if ( v128 )
      {
        m128i_i64 = (__int64)v125;
        v8 = v125;
        v128(v125, v125, 3);
      }
    }
    while ( v125 != (__m128i *)v77 );
    v129 = (__m128i *)v167.m128i_i64[0];
LABEL_281:
    if ( v129 == (__m128i *)&v168 )
      goto LABEL_141;
    v8 = v129;
    _libc_free((unsigned __int64)v129);
    v53 = (__int64)v155;
LABEL_142:
    ++v139;
    v31 = (unsigned int)v156;
    v10 = (__m128i *)v53;
  }
  while ( v134 != v139 );
  if ( (_DWORD)v156 )
  {
    v35 = *(_QWORD *)(a1 + 80);
    v142 = (__m128i **)(v35 + 8LL * *(unsigned int *)(a1 + 88));
    if ( (__m128i **)v35 == v142 )
    {
      v11 = 0;
      v66 = (__m128i *)(v53 + 104LL * (unsigned int)v156);
    }
    else
    {
      v62 = *(__m128i ***)(a1 + 80);
      v11 = 0;
      v63 = (unsigned int)v156;
      do
      {
        v64 = *v62;
        v65 = (__m128i *)((char *)v10 + 104 * v63);
        if ( v65 == v10 )
        {
          v66 = v10;
        }
        else
        {
          do
          {
            if ( !v10[1].m128i_i64[0] )
              goto LABEL_313;
            m128i_i64 = (__int64)v64;
            v8 = v10;
            if ( !((unsigned __int8 (__fastcall *)(__m128i *, __m128i *))v10[1].m128i_i64[1])(v10, v64) )
            {
              if ( !v10[5].m128i_i64[0] )
                goto LABEL_313;
              v8 = v10 + 4;
              m128i_i64 = (__int64)v64;
              v11 = 1;
              ((void (__fastcall *)(__int64 *, __m128i *))v10[5].m128i_i64[1])(v10[4].m128i_i64, v64);
            }
            v10 = (__m128i *)((char *)v10 + 104);
          }
          while ( v65 != v10 );
          v63 = (unsigned int)v156;
          v10 = v155;
          v66 = (__m128i *)((char *)v155 + 104 * (unsigned int)v156);
        }
        ++v62;
      }
      while ( v142 != v62 );
    }
    goto LABEL_154;
  }
LABEL_316:
  v11 = 0;
LABEL_163:
  if ( v10 != (__m128i *)&v157 )
    _libc_free((unsigned __int64)v10);
LABEL_37:
  v12 = v152;
  v13 = (unsigned __int64)v152 + 104 * (unsigned int)v153;
  if ( v152 != (__m128i *)v13 )
  {
    do
    {
      v14 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v13 - 24);
      v13 -= 104LL;
      if ( v14 )
        v14(v13 + 64, v13 + 64, 3);
      v15 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v13 + 48);
      if ( v15 )
        v15(v13 + 32, v13 + 32, 3);
      v16 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v13 + 16);
      if ( v16 )
        v16(v13, v13, 3);
    }
    while ( v12 != (__m128i *)v13 );
    v13 = (unsigned __int64)v152;
  }
  if ( (_BYTE *)v13 != v154 )
    _libc_free(v13);
  return v11;
}
