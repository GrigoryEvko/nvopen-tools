// Function: sub_263CA00
// Address: 0x263ca00
//
_QWORD *__fastcall sub_263CA00(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 *v7; // r8
  __int64 *v8; // rcx
  __int64 v9; // rbx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  unsigned __int64 v16; // rdi
  __int64 *v17; // r15
  __int64 *v18; // rbx
  unsigned __int64 i; // rax
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 *v23; // rbx
  __int64 *v24; // r12
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // rbx
  _QWORD *v29; // r14
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // r15
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  _QWORD *v34; // rbx
  _QWORD *v35; // r14
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // r15
  unsigned __int64 v38; // rdi
  __int64 *v39; // r15
  __int64 *v40; // rbx
  unsigned __int64 j; // rax
  __int64 v42; // rdi
  unsigned int v43; // ecx
  __int64 v44; // rsi
  __int64 *v45; // rbx
  __int64 *v46; // r12
  __int64 v47; // rsi
  __int64 v48; // rdi
  unsigned __int64 v49; // r8
  __int64 v50; // r12
  __int64 v51; // rbx
  _QWORD *v52; // rdi
  __int64 v53; // r15
  unsigned __int64 v54; // r12
  unsigned __int64 v55; // rbx
  unsigned __int64 v56; // rdi
  __int64 v57; // rax
  _QWORD *v58; // rbx
  _QWORD *v59; // r12
  unsigned __int64 v60; // rdi
  _QWORD *v61; // rsi
  _QWORD *v62; // rdx
  __m128i *v64; // rax
  __int64 v65; // rax
  __m128i *v66; // rsi
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  char v70; // al
  bool v71; // zf
  __m128i *v72; // rax
  __int64 v73; // rax
  __int64 v74; // r10
  __int64 v75; // rsi
  unsigned int v76; // eax
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rax
  _BYTE *v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  char v85; // [rsp+28h] [rbp-5F8h]
  __int64 v86; // [rsp+28h] [rbp-5F8h]
  __int64 v87; // [rsp+28h] [rbp-5F8h]
  __int64 v88; // [rsp+38h] [rbp-5E8h] BYREF
  __int64 v89; // [rsp+40h] [rbp-5E0h] BYREF
  __int64 (__fastcall **v90)(); // [rsp+48h] [rbp-5D8h]
  __m128i *v91; // [rsp+50h] [rbp-5D0h] BYREF
  __int64 v92; // [rsp+58h] [rbp-5C8h]
  __m128i v93; // [rsp+60h] [rbp-5C0h] BYREF
  _DWORD v94[4]; // [rsp+70h] [rbp-5B0h] BYREF
  __int64 (__fastcall *v95)(_QWORD *, _DWORD *, int); // [rsp+80h] [rbp-5A0h]
  __int64 (__fastcall *v96)(unsigned int *); // [rsp+88h] [rbp-598h]
  __m128i *v97; // [rsp+90h] [rbp-590h] BYREF
  __int64 v98; // [rsp+98h] [rbp-588h]
  __m128i v99; // [rsp+A0h] [rbp-580h] BYREF
  _DWORD v100[4]; // [rsp+B0h] [rbp-570h] BYREF
  __int64 (__fastcall *v101)(_QWORD *, _DWORD *, int); // [rsp+C0h] [rbp-560h]
  __int64 (__fastcall *v102)(unsigned int *); // [rsp+C8h] [rbp-558h]
  _BYTE v103[8]; // [rsp+F0h] [rbp-530h] BYREF
  int v104; // [rsp+F8h] [rbp-528h] BYREF
  _QWORD *v105; // [rsp+100h] [rbp-520h]
  int *v106; // [rsp+108h] [rbp-518h]
  int *v107; // [rsp+110h] [rbp-510h]
  __int64 v108; // [rsp+118h] [rbp-508h]
  unsigned __int64 v109; // [rsp+120h] [rbp-500h]
  __int64 v110; // [rsp+128h] [rbp-4F8h]
  __int64 v111; // [rsp+130h] [rbp-4F0h]
  _QWORD v112[2]; // [rsp+138h] [rbp-4E8h] BYREF
  __int64 *v113; // [rsp+148h] [rbp-4D8h]
  __int64 v114; // [rsp+150h] [rbp-4D0h]
  _BYTE v115[32]; // [rsp+158h] [rbp-4C8h] BYREF
  __int64 *v116; // [rsp+178h] [rbp-4A8h]
  __int64 v117; // [rsp+180h] [rbp-4A0h]
  _QWORD v118[4]; // [rsp+188h] [rbp-498h] BYREF
  __int64 v119; // [rsp+1A8h] [rbp-478h]
  __int64 v120; // [rsp+1B0h] [rbp-470h]
  __int64 v121; // [rsp+1B8h] [rbp-468h]
  __int64 v122; // [rsp+1C8h] [rbp-458h] BYREF
  unsigned __int64 v123; // [rsp+1D0h] [rbp-450h]
  __int64 *v124; // [rsp+1D8h] [rbp-448h]
  __int64 *v125; // [rsp+1E0h] [rbp-440h]
  __int64 v126; // [rsp+1E8h] [rbp-438h]
  int v127; // [rsp+1F8h] [rbp-428h] BYREF
  _QWORD *v128; // [rsp+200h] [rbp-420h]
  int *v129; // [rsp+208h] [rbp-418h]
  int *v130; // [rsp+210h] [rbp-410h]
  __int64 v131; // [rsp+218h] [rbp-408h]
  __int64 v132; // [rsp+220h] [rbp-400h]
  __int64 v133; // [rsp+228h] [rbp-3F8h]
  __int64 v134; // [rsp+230h] [rbp-3F0h]
  unsigned int v135; // [rsp+238h] [rbp-3E8h]
  __int64 v136; // [rsp+240h] [rbp-3E0h]
  int v137; // [rsp+248h] [rbp-3D8h]
  __int64 v138; // [rsp+250h] [rbp-3D0h]
  _QWORD *v139; // [rsp+258h] [rbp-3C8h]
  __int64 v140; // [rsp+260h] [rbp-3C0h]
  unsigned int v141; // [rsp+268h] [rbp-3B8h]
  __int64 v142; // [rsp+270h] [rbp-3B0h]
  _QWORD *v143; // [rsp+278h] [rbp-3A8h]
  __int64 v144; // [rsp+280h] [rbp-3A0h]
  unsigned int v145; // [rsp+288h] [rbp-398h]
  _QWORD v146[2]; // [rsp+290h] [rbp-390h] BYREF
  __int64 *v147; // [rsp+2A0h] [rbp-380h]
  __int64 v148; // [rsp+2A8h] [rbp-378h]
  _BYTE v149[32]; // [rsp+2B0h] [rbp-370h] BYREF
  __int64 *v150; // [rsp+2D0h] [rbp-350h]
  __int64 v151; // [rsp+2D8h] [rbp-348h]
  _QWORD v152[4]; // [rsp+2E0h] [rbp-340h] BYREF
  unsigned __int64 v153; // [rsp+300h] [rbp-320h]
  __int64 v154; // [rsp+308h] [rbp-318h]
  __int64 v155; // [rsp+310h] [rbp-310h]
  __int64 v156; // [rsp+318h] [rbp-308h]
  __int64 v157; // [rsp+320h] [rbp-300h]
  __int64 v158; // [rsp+328h] [rbp-2F8h]
  unsigned int v159; // [rsp+330h] [rbp-2F0h]
  __m128i v160; // [rsp+340h] [rbp-2E0h] BYREF
  __m128i v161; // [rsp+350h] [rbp-2D0h] BYREF
  __int16 v162; // [rsp+360h] [rbp-2C0h]
  _QWORD *v163; // [rsp+3C8h] [rbp-258h]
  unsigned int v164; // [rsp+3D8h] [rbp-248h]
  unsigned __int64 v165; // [rsp+3E0h] [rbp-240h]
  __int64 v166; // [rsp+3E8h] [rbp-238h]
  __int64 v167; // [rsp+410h] [rbp-210h]
  unsigned int v168; // [rsp+420h] [rbp-200h]

  if ( !*(_BYTE *)a2 )
  {
    sub_262DD10((__int64 **)&v160, a3, a4, *(__int64 **)(a2 + 8), *(__int64 **)(a2 + 16), *(_DWORD *)(a2 + 24));
    v85 = sub_2638ED0((__int64)&v160);
    sub_C7D6A0(v167, 8LL * v168, 8);
    v53 = v166;
    v54 = v165;
    if ( v166 != v165 )
    {
      do
      {
        v55 = *(_QWORD *)(v54 + 16);
        while ( v55 )
        {
          sub_261DCB0(*(_QWORD *)(v55 + 24));
          v56 = v55;
          v55 = *(_QWORD *)(v55 + 16);
          j_j___libc_free_0(v56);
        }
        v54 += 80LL;
      }
      while ( v53 != v54 );
      v54 = v165;
    }
    if ( v54 )
      j_j___libc_free_0(v54);
    v57 = v164;
    if ( v164 )
    {
      v58 = v163;
      v59 = &v163[5 * v164];
      do
      {
        if ( *v58 != -4096 && *v58 != -8192 )
        {
          v60 = v58[1];
          if ( v60 )
            j_j___libc_free_0(v60);
        }
        v58 += 5;
      }
      while ( v59 != v58 );
      v57 = v164;
    }
    sub_C7D6A0((__int64)v163, 40 * v57, 8);
    goto LABEL_93;
  }
  v104 = 0;
  v106 = &v104;
  v107 = &v104;
  v116 = v118;
  v111 = 0x2000000000LL;
  v118[2] = v112;
  v113 = (__int64 *)v115;
  v124 = &v122;
  v125 = &v122;
  v114 = 0x400000000LL;
  v129 = &v127;
  v130 = &v127;
  v105 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v112[0] = 0;
  v112[1] = 0;
  v117 = 0;
  v118[0] = 0;
  v118[1] = 1;
  v118[3] = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v131 = 0;
  v148 = 0x400000000LL;
  v150 = v152;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146[0] = 0;
  v146[1] = 0;
  v147 = (__int64 *)v149;
  v151 = 0;
  v152[0] = 0;
  v152[1] = 1;
  v152[2] = v146;
  v152[3] = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  if ( qword_4FF2E50 )
  {
    sub_8FD6D0((__int64)&v91, "-lowertypetests-read-summary: ", &qword_4FF2E48);
    if ( v92 == 0x3FFFFFFFFFFFFFFFLL || v92 == 4611686018427387902LL )
      goto LABEL_134;
    v64 = (__m128i *)sub_2241490((unsigned __int64 *)&v91, ": ", 2u);
    v160.m128i_i64[0] = (__int64)&v161;
    if ( (__m128i *)v64->m128i_i64[0] == &v64[1] )
    {
      v161 = _mm_loadu_si128(v64 + 1);
    }
    else
    {
      v160.m128i_i64[0] = v64->m128i_i64[0];
      v161.m128i_i64[0] = v64[1].m128i_i64[0];
    }
    v160.m128i_i64[1] = v64->m128i_i64[1];
    v64->m128i_i64[0] = (__int64)v64[1].m128i_i64;
    v64->m128i_i64[1] = 0;
    v64[1].m128i_i8[0] = 0;
    v97 = &v99;
    if ( (__m128i *)v160.m128i_i64[0] == &v161 )
    {
      v99 = _mm_load_si128(&v161);
    }
    else
    {
      v97 = (__m128i *)v160.m128i_i64[0];
      v99.m128i_i64[0] = v161.m128i_i64[0];
    }
    v65 = v160.m128i_i64[1];
    v160.m128i_i64[0] = (__int64)&v161;
    v160.m128i_i64[1] = 0;
    v98 = v65;
    v161.m128i_i8[0] = 0;
    v102 = sub_226E290;
    v100[0] = 1;
    v101 = sub_226EF00;
    sub_2240A30((unsigned __int64 *)&v160);
    sub_2240A30((unsigned __int64 *)&v91);
    v162 = 260;
    v66 = &v160;
    v160.m128i_i64[0] = (__int64)&qword_4FF2E48;
    sub_C7EA90((__int64)&v91, v160.m128i_i64, 1, 1u, 0, 0);
    v70 = v93.m128i_i8[0] & 1;
    if ( (v93.m128i_i8[0] & 1) != 0 && (v66 = (__m128i *)(unsigned int)v91, v69 = v92, (_DWORD)v91) )
    {
      sub_C63CA0(&v89, (int)v91, v92);
      v67 = v89 | 1;
      v71 = (v89 & 0xFFFFFFFFFFFFFFFELL) == 0;
      v89 |= 1uLL;
      if ( !v71 )
        sub_261BDF0((__int64)&v97, &v89, v67);
      v74 = 0;
      v70 = v93.m128i_i8[0] & 1;
    }
    else
    {
      v74 = (__int64)v91;
      v91 = 0;
    }
    if ( !v70 && v91 )
    {
      v87 = v74;
      (*(void (__fastcall **)(__m128i *, __m128i *, __int64, __int64, __int64))(v91->m128i_i64[0] + 8))(
        v91,
        v66,
        v67,
        v68,
        v69);
      v74 = v87;
    }
    v75 = *(_QWORD *)(v74 + 8);
    v86 = v74;
    sub_CB0A90((__int64)&v160, v75, *(_QWORD *)(v74 + 16) - v75, 0, 0, 0);
    sub_CB4D10((__int64)&v160, v75);
    sub_CB0300((__int64)&v160);
    sub_2633C40((__int64)&v160, (__int64)v103);
    sub_CB1A30((__int64)&v160);
    v76 = sub_CB0000((__int64)&v160);
    v77 = v76;
    sub_C63CA0(&v89, v76, v78);
    v79 = v89;
    v89 = 0;
    v91 = (__m128i *)(v79 | 1);
    if ( (v79 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_261BDF0((__int64)&v97, (__int64 *)&v91, v79 | 1);
    sub_CB34B0((__int64)&v160, v77);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v86 + 8LL))(v86);
    if ( v101 )
      v101(v100, v100, 3);
    sub_2240A30((unsigned __int64 *)&v97);
  }
  if ( dword_4FF2F48 == 1 )
  {
    v7 = (__int64 *)v103;
  }
  else
  {
    v7 = 0;
    v8 = (__int64 *)v103;
    if ( dword_4FF2F48 == 2 )
      goto LABEL_5;
  }
  v8 = 0;
LABEL_5:
  sub_262DD10((__int64 **)&v160, a3, a4, v8, v7, 0);
  v85 = sub_2638ED0((__int64)&v160);
  sub_C7D6A0(v167, 8LL * v168, 8);
  v9 = v166;
  v10 = v165;
  if ( v166 != v165 )
  {
    do
    {
      v11 = *(_QWORD *)(v10 + 16);
      while ( v11 )
      {
        sub_261DCB0(*(_QWORD *)(v11 + 24));
        v12 = v11;
        v11 = *(_QWORD *)(v11 + 16);
        j_j___libc_free_0(v12);
      }
      v10 += 80LL;
    }
    while ( v9 != v10 );
    v10 = v165;
  }
  if ( v10 )
    j_j___libc_free_0(v10);
  v13 = v164;
  if ( v164 )
  {
    v14 = v163;
    v15 = &v163[5 * v164];
    do
    {
      if ( *v14 != -8192 && *v14 != -4096 )
      {
        v16 = v14[1];
        if ( v16 )
          j_j___libc_free_0(v16);
      }
      v14 += 5;
    }
    while ( v15 != v14 );
    v13 = v164;
  }
  sub_C7D6A0((__int64)v163, 40 * v13, 8);
  if ( !qword_4FF2D50 )
    goto LABEL_21;
  sub_8FD6D0((__int64)&v97, "-lowertypetests-write-summary: ", &qword_4FF2D48);
  if ( v98 == 0x3FFFFFFFFFFFFFFFLL || v98 == 4611686018427387902LL )
LABEL_134:
    sub_4262D8((__int64)"basic_string::append");
  v72 = (__m128i *)sub_2241490((unsigned __int64 *)&v97, ": ", 2u);
  v160.m128i_i64[0] = (__int64)&v161;
  if ( (__m128i *)v72->m128i_i64[0] == &v72[1] )
  {
    v161 = _mm_loadu_si128(v72 + 1);
  }
  else
  {
    v160.m128i_i64[0] = v72->m128i_i64[0];
    v161.m128i_i64[0] = v72[1].m128i_i64[0];
  }
  v160.m128i_i64[1] = v72->m128i_i64[1];
  v72->m128i_i64[0] = (__int64)v72[1].m128i_i64;
  v72->m128i_i64[1] = 0;
  v72[1].m128i_i8[0] = 0;
  v91 = &v93;
  if ( (__m128i *)v160.m128i_i64[0] == &v161 )
  {
    v93 = _mm_load_si128(&v161);
  }
  else
  {
    v91 = (__m128i *)v160.m128i_i64[0];
    v93.m128i_i64[0] = v161.m128i_i64[0];
  }
  v94[0] = 1;
  v92 = v160.m128i_i64[1];
  v96 = sub_226E290;
  v95 = sub_226EF00;
  sub_2240A30((unsigned __int64 *)&v97);
  LODWORD(v89) = 0;
  v90 = sub_2241E40();
  sub_CB7060((__int64)&v97, (_BYTE *)qword_4FF2D48, qword_4FF2D50, (__int64)&v89, 3u);
  sub_C63CA0(&v88, v89, (__int64)v90);
  v73 = v88;
  v88 = 0;
  v160.m128i_i64[0] = v73 | 1;
  if ( (v73 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_261BDF0((__int64)&v91, v160.m128i_i64, v73 | 1);
  sub_CB1A80((__int64)&v160, (__int64)&v97, 0, 70);
  sub_CB2850((__int64)&v160);
  v80 = 0;
  if ( (unsigned __int8)sub_CB2870((__int64)&v160, 0) )
  {
    sub_CB05C0(&v160, 0, v81, v82, v83, v84);
    v80 = v103;
    sub_2633C40((__int64)&v160, (__int64)v103);
    sub_CB2220(&v160);
    nullsub_173();
  }
  sub_CB1B70((__int64)&v160);
  sub_CB0A00(&v160, (__int64)v80);
  sub_CB5B00((int *)&v97, (__int64)v80);
  if ( v95 )
    v95(v94, v94, 3);
  sub_2240A30((unsigned __int64 *)&v91);
LABEL_21:
  sub_C7D6A0(v157, 16LL * v159, 8);
  if ( v153 )
    j_j___libc_free_0(v153);
  v17 = v147;
  v18 = &v147[(unsigned int)v148];
  if ( v147 != v18 )
  {
    for ( i = (unsigned __int64)v147; ; i = (unsigned __int64)v147 )
    {
      v20 = *v17;
      v21 = (unsigned int)((__int64)((__int64)v17 - i) >> 3) >> 7;
      v22 = 4096LL << v21;
      if ( v21 >= 0x1E )
        v22 = 0x40000000000LL;
      ++v17;
      sub_C7D6A0(v20, v22, 16);
      if ( v18 == v17 )
        break;
    }
  }
  v23 = v150;
  v24 = &v150[2 * (unsigned int)v151];
  if ( v150 != v24 )
  {
    do
    {
      v25 = v23[1];
      v26 = *v23;
      v23 += 2;
      sub_C7D6A0(v26, v25, 16);
    }
    while ( v24 != v23 );
    v24 = v150;
  }
  if ( v24 != v152 )
    _libc_free((unsigned __int64)v24);
  if ( v147 != (__int64 *)v149 )
    _libc_free((unsigned __int64)v147);
  v27 = v145;
  if ( v145 )
  {
    v28 = v143;
    v29 = &v143[7 * v145];
    do
    {
      while ( 1 )
      {
        if ( *v28 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v30 = v28[3];
          if ( v30 )
            break;
        }
        v28 += 7;
        if ( v29 == v28 )
          goto LABEL_45;
      }
      do
      {
        v31 = v30;
        sub_261C7C0(*(_QWORD **)(v30 + 24));
        v32 = *(_QWORD *)(v30 + 32);
        v30 = *(_QWORD *)(v30 + 16);
        if ( v32 != v31 + 48 )
          j_j___libc_free_0(v32);
        j_j___libc_free_0(v31);
      }
      while ( v30 );
      v28 += 7;
    }
    while ( v29 != v28 );
LABEL_45:
    v27 = v145;
  }
  sub_C7D6A0((__int64)v143, 56 * v27, 8);
  v33 = v141;
  if ( v141 )
  {
    v34 = v139;
    v35 = &v139[7 * v141];
    do
    {
      while ( 1 )
      {
        if ( *v34 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v36 = v34[3];
          if ( v36 )
            break;
        }
        v34 += 7;
        if ( v35 == v34 )
          goto LABEL_55;
      }
      do
      {
        v37 = v36;
        sub_261C7C0(*(_QWORD **)(v36 + 24));
        v38 = *(_QWORD *)(v36 + 32);
        v36 = *(_QWORD *)(v36 + 16);
        if ( v38 != v37 + 48 )
          j_j___libc_free_0(v38);
        j_j___libc_free_0(v37);
      }
      while ( v36 );
      v34 += 7;
    }
    while ( v35 != v34 );
LABEL_55:
    v33 = v141;
  }
  sub_C7D6A0((__int64)v139, 56 * v33, 8);
  sub_C7D6A0(v133, 16LL * v135, 8);
  sub_261BE90(v128);
  sub_261C4E0(v123);
  sub_C7D6A0(v119, 16LL * (unsigned int)v121, 8);
  v39 = v113;
  v40 = &v113[(unsigned int)v114];
  if ( v113 != v40 )
  {
    for ( j = (unsigned __int64)v113; ; j = (unsigned __int64)v113 )
    {
      v42 = *v39;
      v43 = (unsigned int)((__int64)((__int64)v39 - j) >> 3) >> 7;
      v44 = 4096LL << v43;
      if ( v43 >= 0x1E )
        v44 = 0x40000000000LL;
      ++v39;
      sub_C7D6A0(v42, v44, 16);
      if ( v40 == v39 )
        break;
    }
  }
  v45 = v116;
  v46 = &v116[2 * (unsigned int)v117];
  if ( v116 != v46 )
  {
    do
    {
      v47 = v45[1];
      v48 = *v45;
      v45 += 2;
      sub_C7D6A0(v48, v47, 16);
    }
    while ( v46 != v45 );
    v46 = v116;
  }
  if ( v46 != v118 )
    _libc_free((unsigned __int64)v46);
  if ( v113 != (__int64 *)v115 )
    _libc_free((unsigned __int64)v113);
  v49 = v109;
  if ( HIDWORD(v110) && (_DWORD)v110 )
  {
    v50 = 8LL * (unsigned int)v110;
    v51 = 0;
    do
    {
      v52 = *(_QWORD **)(v49 + v51);
      if ( v52 && v52 != (_QWORD *)-8LL )
      {
        sub_C7D6A0((__int64)v52, *v52 + 33LL, 8);
        v49 = v109;
      }
      v51 += 8;
    }
    while ( v51 != v50 );
  }
  _libc_free(v49);
  sub_261AA10(v105);
LABEL_93:
  v61 = a1 + 4;
  v62 = a1 + 10;
  if ( v85 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v61;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v62;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v61;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v62;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}
