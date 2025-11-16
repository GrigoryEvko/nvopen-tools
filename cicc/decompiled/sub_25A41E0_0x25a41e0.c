// Function: sub_25A41E0
// Address: 0x25a41e0
//
_QWORD *__fastcall sub_25A41E0(_QWORD *a1, char **a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  char *v6; // r12
  char *v7; // r15
  __int64 v8; // rdx
  unsigned __int64 v9; // r13
  unsigned __int64 *v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 *v16; // r14
  __int64 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 *v20; // r12
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r12
  __int64 v26; // r15
  _BYTE *v27; // rbx
  __int64 v28; // r13
  __int64 v29; // r12
  size_t v30; // r14
  const char *v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rdi
  _BYTE *v34; // rsi
  __int64 *v35; // rbx
  __int64 *v36; // r15
  __int64 v37; // r12
  __int64 **v38; // r13
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdi
  __int64 *v42; // rbx
  __int64 v43; // r13
  __int64 v44; // r15
  __int64 v45; // r12
  _QWORD *v46; // r14
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // r14
  __int64 v50; // r14
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r14
  __int64 v55; // rbx
  __int64 *v56; // rbx
  __int64 *v57; // r13
  __int64 v58; // r12
  char v59; // al
  __int64 i; // rax
  bool v61; // zf
  _QWORD *v62; // rsi
  _QWORD *v63; // rdx
  __int64 v64; // r12
  unsigned __int64 *v65; // r13
  __int64 v66; // rbx
  unsigned __int64 v67; // r14
  unsigned __int64 *v68; // rbx
  unsigned __int64 *v69; // rbx
  unsigned __int64 *v70; // r12
  __int64 v72; // rax
  unsigned __int64 *v73; // r13
  unsigned __int64 *v74; // rbx
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 *v78; // r14
  __int64 *v79; // r14
  char *v80; // rsi
  char *v81; // rdx
  int v82; // esi
  unsigned __int64 v83; // r15
  __int64 v84; // rdx
  __int64 *v85; // rbx
  __int64 v86; // rdx
  _BYTE *v87; // rsi
  unsigned int v88; // ebx
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rcx
  __m128i *v92; // rbx
  unsigned __int64 v93; // rsi
  __int64 v94; // rdx
  __m128i *v95; // rax
  __int64 v96; // rcx
  unsigned __int64 *v97; // r14
  unsigned __int64 *v98; // rbx
  unsigned __int64 v99; // r14
  unsigned __int64 *v100; // rbx
  char *v101; // rbx
  __int64 v102; // rsi
  __int64 *v103; // r14
  __int64 v104; // rsi
  unsigned __int64 v105; // rbx
  _BYTE *v106; // rdi
  size_t v107; // r13
  __int64 v108; // rax
  __int64 v109; // rax
  unsigned __int64 *v110; // rbx
  _QWORD *v111; // r14
  char ***v112; // rbx
  char *v113; // rcx
  signed __int64 v114; // r15
  unsigned __int64 v115; // r15
  unsigned __int64 *v116; // r12
  _BYTE *src; // [rsp+0h] [rbp-780h]
  unsigned int v118; // [rsp+8h] [rbp-778h]
  char **v119; // [rsp+20h] [rbp-760h]
  _BYTE *v120; // [rsp+38h] [rbp-748h]
  __int64 v121; // [rsp+48h] [rbp-738h]
  unsigned int v122; // [rsp+50h] [rbp-730h]
  _QWORD *v124; // [rsp+60h] [rbp-720h]
  __int64 v125; // [rsp+68h] [rbp-718h]
  unsigned __int64 *v126; // [rsp+68h] [rbp-718h]
  __int64 v127; // [rsp+70h] [rbp-710h]
  __int64 v128; // [rsp+70h] [rbp-710h]
  __int64 v129; // [rsp+78h] [rbp-708h]
  __int64 v130; // [rsp+78h] [rbp-708h]
  __int64 *v131; // [rsp+78h] [rbp-708h]
  __int64 *v132; // [rsp+78h] [rbp-708h]
  __int64 v133; // [rsp+80h] [rbp-700h]
  _BYTE *v134; // [rsp+80h] [rbp-700h]
  __int64 *v135; // [rsp+80h] [rbp-700h]
  void *s2; // [rsp+88h] [rbp-6F8h]
  char s2a; // [rsp+88h] [rbp-6F8h]
  char **s2b; // [rsp+88h] [rbp-6F8h]
  unsigned __int64 *s2c; // [rsp+88h] [rbp-6F8h]
  size_t v140; // [rsp+98h] [rbp-6E8h] BYREF
  _QWORD *v141; // [rsp+A0h] [rbp-6E0h] BYREF
  char v142; // [rsp+B0h] [rbp-6D0h]
  __m128i *v143; // [rsp+C0h] [rbp-6C0h] BYREF
  __int64 v144; // [rsp+C8h] [rbp-6B8h]
  __m128i v145; // [rsp+D0h] [rbp-6B0h] BYREF
  char **v146; // [rsp+E0h] [rbp-6A0h] BYREF
  __int64 v147; // [rsp+E8h] [rbp-698h]
  _QWORD v148[8]; // [rsp+F0h] [rbp-690h] BYREF
  __int64 *v149; // [rsp+130h] [rbp-650h] BYREF
  __int64 v150; // [rsp+138h] [rbp-648h]
  _BYTE v151[64]; // [rsp+140h] [rbp-640h] BYREF
  unsigned __int64 *v152; // [rsp+180h] [rbp-600h] BYREF
  __int64 v153; // [rsp+188h] [rbp-5F8h]
  _BYTE v154[136]; // [rsp+190h] [rbp-5F0h] BYREF
  __int64 v155; // [rsp+218h] [rbp-568h]
  unsigned int v156; // [rsp+228h] [rbp-558h]
  __int64 v157; // [rsp+238h] [rbp-548h]
  unsigned int v158; // [rsp+248h] [rbp-538h]
  __m128i *v159; // [rsp+250h] [rbp-530h] BYREF
  __int64 v160; // [rsp+258h] [rbp-528h]
  __m128i v161; // [rsp+260h] [rbp-520h] BYREF
  unsigned __int64 *v162; // [rsp+270h] [rbp-510h] BYREF
  __int64 v163; // [rsp+278h] [rbp-508h]
  _BYTE v164[16]; // [rsp+280h] [rbp-500h] BYREF
  __int64 v165; // [rsp+290h] [rbp-4F0h]
  unsigned int v166; // [rsp+2A0h] [rbp-4E0h]
  unsigned __int64 *v167; // [rsp+2A8h] [rbp-4D8h]
  char *v168; // [rsp+2B8h] [rbp-4C8h] BYREF
  char v169; // [rsp+2C8h] [rbp-4B8h] BYREF
  _QWORD *v170; // [rsp+2F8h] [rbp-488h]
  _QWORD v171[6]; // [rsp+308h] [rbp-478h] BYREF
  unsigned int v172; // [rsp+338h] [rbp-448h]
  __int64 **v173; // [rsp+340h] [rbp-440h]
  __int64 *v174; // [rsp+350h] [rbp-430h] BYREF
  __int64 v175; // [rsp+358h] [rbp-428h]
  _BYTE v176[16]; // [rsp+360h] [rbp-420h] BYREF
  __int16 v177; // [rsp+370h] [rbp-410h]
  unsigned __int64 *v178; // [rsp+460h] [rbp-320h] BYREF
  unsigned __int64 *v179; // [rsp+468h] [rbp-318h]
  unsigned __int64 *v180; // [rsp+470h] [rbp-310h]
  char v181; // [rsp+478h] [rbp-308h]
  __int64 v182; // [rsp+480h] [rbp-300h] BYREF
  __int64 v183; // [rsp+488h] [rbp-2F8h]
  _BYTE v184[752]; // [rsp+490h] [rbp-2F0h] BYREF

  v6 = a2[1];
  v7 = *a2;
  v8 = *((unsigned __int8 *)a2 + 24);
  v182 = (__int64)v184;
  v124 = a1;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = v8;
  v183 = 0x400000000LL;
  v9 = v6 - v7;
  if ( v6 == v7 )
  {
    v10 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_241:
      sub_4261EA(a1, a2, v8);
    a1 = (_QWORD *)(v6 - v7);
    v109 = sub_22077B0(v6 - v7);
    v110 = (unsigned __int64 *)v109;
    v111 = (_QWORD *)v109;
    if ( v6 != v7 )
    {
      s2c = (unsigned __int64 *)v109;
      v112 = (char ***)v7;
      do
      {
        if ( v111 )
        {
          v115 = (char *)v112[1] - (char *)*v112;
          *v111 = 0;
          v111[1] = 0;
          v111[2] = 0;
          if ( v115 )
          {
            if ( v115 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_241;
            a1 = (_QWORD *)v115;
            v113 = (char *)sub_22077B0(v115);
          }
          else
          {
            v113 = 0;
          }
          *v111 = v113;
          v111[1] = v113;
          v111[2] = &v113[v115];
          a2 = *v112;
          v114 = (char *)v112[1] - (char *)*v112;
          if ( v112[1] != *v112 )
          {
            a1 = v113;
            v113 = (char *)memmove(v113, a2, (char *)v112[1] - (char *)*v112);
          }
          v111[1] = &v113[v114];
        }
        v112 += 3;
        v111 += 3;
      }
      while ( v6 != (char *)v112 );
      v110 = s2c;
    }
    v116 = v178;
    if ( v179 != v178 )
    {
      do
      {
        if ( *v116 )
          j_j___libc_free_0(*v116);
        v116 += 3;
      }
      while ( v179 != v116 );
      v116 = v178;
    }
    if ( v116 )
      j_j___libc_free_0((unsigned __int64)v116);
    v178 = v110;
    v10 = (unsigned __int64 *)((char *)v110 + v9);
    v180 = v10;
  }
  v179 = v10;
  if ( qword_4FEFBB0 )
  {
    v177 = 260;
    v174 = &qword_4FEFBA8;
    sub_C7EA90((__int64)&v141, (__int64 *)&v174, 0, 1u, 0, 0);
    if ( (v142 & 1) != 0 && (_DWORD)v141 )
      sub_C64ED0("BlockExtractor couldn't load the file.", 1u);
    v175 = 0x1000000000LL;
    v174 = (__int64 *)v176;
    v76 = v141[2] - v141[1];
    v159 = (__m128i *)v141[1];
    v160 = v76;
    sub_C93960((char **)&v159, (__int64)&v174, 10, -1, 0, v75);
    v77 = 2LL * (unsigned int)v175;
    v78 = &v174[v77];
    if ( v174 == &v174[v77] )
    {
LABEL_186:
      if ( v78 != (__int64 *)v176 )
        _libc_free((unsigned __int64)v78);
      if ( (v142 & 1) == 0 && v141 )
        (*(void (__fastcall **)(_QWORD *))(*v141 + 8LL))(v141);
      goto LABEL_4;
    }
    s2b = (char **)v174;
    v119 = (char **)&v174[v77];
    while ( 1 )
    {
      v147 = 0x400000000LL;
      v146 = (char **)v148;
      sub_C93960(s2b, (__int64)&v146, 32, -1, 0, a6);
      if ( (_DWORD)v147 )
        break;
LABEL_182:
      if ( v146 != v148 )
        _libc_free((unsigned __int64)v146);
      s2b += 2;
      if ( v119 == s2b )
      {
        v78 = v174;
        goto LABEL_186;
      }
    }
    if ( (_DWORD)v147 != 2 )
      sub_C64ED0("Invalid line format, expecting lines like: 'funcname bb1[;bb2..]'", 0);
    v149 = (__int64 *)v151;
    v150 = 0x400000000LL;
    sub_C93960(v146 + 2, (__int64)&v149, 59, -1, 0, a6);
    if ( !(_DWORD)v150 )
      sub_C64ED0("Missing bbs name", 1u);
    v79 = (__int64 *)v154;
    v80 = *v146;
    v81 = v146[1];
    v143 = &v145;
    sub_25A3700((__int64 *)&v143, v80, (__int64)&v81[(_QWORD)v80]);
    v82 = 0;
    v83 = (unsigned __int64)v149;
    v84 = 2LL * (unsigned int)v150;
    v152 = (unsigned __int64 *)v154;
    v153 = 0x400000000LL;
    v85 = &v149[v84];
    v128 = (v84 * 8) >> 4;
    if ( (unsigned __int64)v84 > 8 )
    {
      sub_95D880((__int64)&v152, (v84 * 8) >> 4);
      v82 = v153;
      v79 = (__int64 *)&v152[4 * (unsigned int)v153];
    }
    if ( (__int64 *)v83 != v85 )
    {
      do
      {
        if ( v79 )
        {
          v86 = *(_QWORD *)(v83 + 8);
          v87 = *(_BYTE **)v83;
          *v79 = (__int64)(v79 + 2);
          sub_25A3700(v79, v87, (__int64)&v87[v86]);
        }
        v83 += 16LL;
        v79 += 4;
      }
      while ( v85 != (__int64 *)v83 );
      v82 = v153;
    }
    v159 = &v161;
    v88 = v82 + v128;
    LODWORD(v153) = v82 + v128;
    if ( v143 == &v145 )
    {
      v161 = _mm_load_si128(&v145);
    }
    else
    {
      v159 = v143;
      v161.m128i_i64[0] = v145.m128i_i64[0];
    }
    v89 = v144;
    v143 = &v145;
    v144 = 0;
    v160 = v89;
    v145.m128i_i8[0] = 0;
    v162 = (unsigned __int64 *)v164;
    v163 = 0x400000000LL;
    if ( !v88 )
    {
LABEL_155:
      v90 = (unsigned int)v183;
      v91 = v182;
      v92 = (__m128i *)&v159;
      v93 = (unsigned int)v183 + 1LL;
      v94 = (unsigned int)v183;
      if ( v93 > HIDWORD(v183) )
      {
        if ( v182 > (unsigned __int64)&v159
          || (v94 = 5LL * (unsigned int)v183,
              (unsigned __int64)&v159 >= v182 + 176 * (unsigned __int64)(unsigned int)v183) )
        {
          sub_25A4000((__int64)&v182, v93, v94, v182, (__int64)a5, a6);
          v90 = (unsigned int)v183;
          v91 = v182;
          LODWORD(v94) = v183;
        }
        else
        {
          v101 = (char *)&v159 - v182;
          sub_25A4000((__int64)&v182, v93, v94, v182, (__int64)a5, a6);
          v91 = v182;
          v90 = (unsigned int)v183;
          v92 = (__m128i *)&v101[v182];
          LODWORD(v94) = v183;
        }
      }
      v95 = (__m128i *)(v91 + 176 * v90);
      if ( v95 )
      {
        v95->m128i_i64[0] = (__int64)v95[1].m128i_i64;
        if ( (__m128i *)v92->m128i_i64[0] == &v92[1] )
        {
          v95[1] = _mm_loadu_si128(v92 + 1);
        }
        else
        {
          v95->m128i_i64[0] = v92->m128i_i64[0];
          v95[1].m128i_i64[0] = v92[1].m128i_i64[0];
        }
        v96 = v92->m128i_i64[1];
        v92->m128i_i64[0] = (__int64)v92[1].m128i_i64;
        v92->m128i_i64[1] = 0;
        v95->m128i_i64[1] = v96;
        v92[1].m128i_i8[0] = 0;
        v95[2].m128i_i64[0] = (__int64)v95[3].m128i_i64;
        v95[2].m128i_i64[1] = 0x400000000LL;
        if ( v92[2].m128i_i32[2] )
          sub_25A3B40((__int64)v95[2].m128i_i64, (__int64)v92[2].m128i_i64);
        LODWORD(v94) = v183;
      }
      v97 = v162;
      LODWORD(v183) = v94 + 1;
      v98 = &v162[4 * (unsigned int)v163];
      if ( v162 != v98 )
      {
        do
        {
          v98 -= 4;
          if ( (unsigned __int64 *)*v98 != v98 + 2 )
            j_j___libc_free_0(*v98);
        }
        while ( v97 != v98 );
        v98 = v162;
      }
      if ( v98 != (unsigned __int64 *)v164 )
        _libc_free((unsigned __int64)v98);
      if ( v159 != &v161 )
        j_j___libc_free_0((unsigned __int64)v159);
      v99 = (unsigned __int64)v152;
      v100 = &v152[4 * (unsigned int)v153];
      if ( v152 != v100 )
      {
        do
        {
          v100 -= 4;
          if ( (unsigned __int64 *)*v100 != v100 + 2 )
            j_j___libc_free_0(*v100);
        }
        while ( (unsigned __int64 *)v99 != v100 );
        v100 = v152;
      }
      if ( v100 != (unsigned __int64 *)v154 )
        _libc_free((unsigned __int64)v100);
      if ( v143 != &v145 )
        j_j___libc_free_0((unsigned __int64)v143);
      if ( v149 != (__int64 *)v151 )
        _libc_free((unsigned __int64)v149);
      goto LABEL_182;
    }
    v102 = v88;
    if ( v88 > 4 )
    {
      sub_95D880((__int64)&v162, v88);
      v103 = (__int64 *)v162;
      v102 = (unsigned int)v153;
    }
    else
    {
      v103 = (__int64 *)v164;
    }
    v104 = 4 * v102;
    v126 = &v152[v104];
    if ( v152 == &v152[v104] )
    {
LABEL_211:
      LODWORD(v163) = v88;
      goto LABEL_155;
    }
    v118 = v88;
    v105 = (unsigned __int64)v152;
    while ( 1 )
    {
      if ( !v103 )
        goto LABEL_202;
      *v103 = (__int64)(v103 + 2);
      a5 = *(_BYTE **)v105;
      v107 = *(_QWORD *)(v105 + 8);
      if ( v107 + *(_QWORD *)v105 && !a5 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v140 = *(_QWORD *)(v105 + 8);
      if ( v107 > 0xF )
        break;
      v106 = (_BYTE *)*v103;
      if ( v107 != 1 )
      {
        if ( !v107 )
          goto LABEL_201;
        goto LABEL_208;
      }
      *v106 = *a5;
      v107 = v140;
      v106 = (_BYTE *)*v103;
LABEL_201:
      v103[1] = v107;
      v106[v107] = 0;
LABEL_202:
      v105 += 32LL;
      v103 += 4;
      if ( v126 == (unsigned __int64 *)v105 )
      {
        v88 = v118;
        goto LABEL_211;
      }
    }
    src = a5;
    v108 = sub_22409D0((__int64)v103, &v140, 0);
    a5 = src;
    *v103 = v108;
    v106 = (_BYTE *)v108;
    v103[2] = v140;
LABEL_208:
    memcpy(v106, a5, v107);
    v107 = v140;
    v106 = (_BYTE *)*v103;
    goto LABEL_201;
  }
LABEL_4:
  v149 = (__int64 *)v151;
  v150 = 0x400000000LL;
  v121 = a3 + 24;
  v129 = *(_QWORD *)(a3 + 32);
  if ( v129 != a3 + 24 )
  {
    while ( 1 )
    {
      if ( !v129 )
        BUG();
      v11 = *(_QWORD *)(v129 + 24);
      if ( v11 != v129 + 16 )
        break;
LABEL_27:
      v18 = (unsigned int)v150;
      v19 = (unsigned int)v150 + 1LL;
      if ( v19 > HIDWORD(v150) )
      {
        sub_C8D5F0((__int64)&v149, v151, v19, 8u, (__int64)a5, a6);
        v18 = (unsigned int)v150;
      }
      v149[v18] = v129 - 56;
      LODWORD(v150) = v150 + 1;
      v129 = *(_QWORD *)(v129 + 8);
      if ( v121 == v129 )
        goto LABEL_30;
    }
    while ( 1 )
    {
      if ( !v11 )
        BUG();
      v12 = *(_QWORD *)(v11 + 32);
      v133 = v11 + 24;
      if ( v12 != v11 + 24 )
        break;
LABEL_26:
      v11 = *(_QWORD *)(v11 + 8);
      if ( v129 + 16 == v11 )
        goto LABEL_27;
    }
    while ( 1 )
    {
      if ( !v12 )
LABEL_248:
        BUG();
      if ( *(_BYTE *)(v12 - 24) != 34 )
        goto LABEL_10;
      v159 = *(__m128i **)(v12 + 16);
      v13 = *(_QWORD *)(v12 - 88);
      v14 = *(_QWORD *)(v13 + 16);
      if ( !v14 )
        goto LABEL_10;
      while ( 1 )
      {
        v15 = *(_QWORD *)(v14 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v15 - 30) <= 0xAu )
          break;
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
          goto LABEL_10;
      }
LABEL_18:
      v16 = *(__int64 **)(v15 + 40);
      if ( !sub_AA5E90((__int64)v16) || v16 == (__int64 *)v159 )
      {
LABEL_16:
        while ( 1 )
        {
          v14 = *(_QWORD *)(v14 + 8);
          if ( !v14 )
            goto LABEL_10;
          v15 = *(_QWORD *)(v14 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v15 - 30) <= 0xAu )
            goto LABEL_18;
        }
      }
      v17 = (__int64 *)(v159[3].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
      if ( v17 == (__int64 *)&v159[3] || !v17 || (unsigned int)*((unsigned __int8 *)v17 - 24) - 30 > 0xA )
        goto LABEL_248;
      if ( *((_BYTE *)v17 - 24) != 34 )
        goto LABEL_16;
      v174 = (__int64 *)v176;
      v175 = 0x200000000LL;
      sub_F40790(v13, (__int64 **)&v159, 1, ".1", byte_4459056, (__int64)&v174, 0, 0, 0, 0);
      if ( v174 != (__int64 *)v176 )
      {
        _libc_free((unsigned __int64)v174);
        v12 = *(_QWORD *)(v12 + 8);
        if ( v133 == v12 )
          goto LABEL_26;
      }
      else
      {
LABEL_10:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v133 == v12 )
          goto LABEL_26;
      }
    }
  }
LABEL_30:
  v20 = v179;
  v21 = (unsigned int)v183;
  v22 = 0xAAAAAAAAAAAAAAABLL * (v179 - v178);
  v122 = -1431655765 * (v179 - v178);
  v23 = (unsigned int)v183 + (unsigned __int64)(unsigned int)v22;
  if ( v23 > v22 )
  {
    sub_25A3900((__int64)&v178, v23 - v22);
    v21 = (unsigned int)v183;
  }
  else if ( v23 < v22 )
  {
    v72 = 3 * v23;
    v73 = &v178[v72];
    if ( v179 != v73 )
    {
      v74 = &v178[v72];
      do
      {
        if ( *v74 )
          j_j___libc_free_0(*v74);
        v74 += 3;
      }
      while ( v20 != v74 );
      v179 = v73;
      v21 = (unsigned int)v183;
    }
  }
  v125 = v182;
  v120 = (_BYTE *)(v182 + 176 * v21);
  if ( (_BYTE *)v182 != v120 )
  {
    do
    {
      v134 = sub_BA8CB0(a3, *(_QWORD *)v125, *(_QWORD *)(v125 + 8));
      if ( !v134 )
        sub_C64ED0("Invalid function name specified in the input file", 0);
      v24 = *(_QWORD *)(v125 + 32);
      v25 = 32LL * *(unsigned int *)(v125 + 40);
      v130 = v24 + v25;
      if ( v24 != v24 + v25 )
      {
        v26 = *(_QWORD *)(v125 + 32);
        v127 = 3LL * v122;
        v27 = v134 + 72;
        do
        {
          v28 = *((_QWORD *)v134 + 10);
          if ( v27 == (_BYTE *)v28 )
            goto LABEL_240;
          while ( 1 )
          {
            v29 = v28 - 24;
            if ( !v28 )
              v29 = 0;
            v30 = *(_QWORD *)(v26 + 8);
            s2 = *(void **)v26;
            v31 = sub_BD5D20(v29);
            if ( v30 == v32 && (!v30 || !memcmp(v31, s2, v30)) )
              break;
            v28 = *(_QWORD *)(v28 + 8);
            if ( v27 == (_BYTE *)v28 )
              goto LABEL_240;
          }
          if ( v27 == (_BYTE *)v28 )
LABEL_240:
            sub_C64ED0("Invalid block name specified in the input file", 0);
          v33 = (__int64)&v178[v127];
          v174 = (__int64 *)v29;
          v34 = (_BYTE *)v178[v127 + 1];
          if ( v34 == (_BYTE *)v178[v127 + 2] )
          {
            sub_F38A10(v33, v34, &v174);
          }
          else
          {
            if ( v34 )
            {
              *(_QWORD *)v34 = v29;
              v34 = *(_BYTE **)(v33 + 8);
            }
            *(_QWORD *)(v33 + 8) = v34 + 8;
          }
          v26 += 32;
        }
        while ( v130 != v26 );
      }
      v125 += 176;
      ++v122;
    }
    while ( v120 != (_BYTE *)v125 );
  }
  v35 = (__int64 *)v178;
  s2a = 0;
  v36 = (__int64 *)v176;
  v37 = a3;
  v38 = &v174;
  v135 = (__int64 *)v179;
  if ( v178 != v179 )
  {
    do
    {
      v174 = v36;
      v175 = 0x2000000000LL;
      v39 = *v35;
      v40 = v35[1];
      if ( *v35 != v40 )
      {
        v131 = v35;
        v41 = (__int64)v38;
        v42 = v36;
        v43 = v40;
        v44 = v37;
        v45 = v39;
        do
        {
          v50 = *(_QWORD *)v45;
          if ( v44 != *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v45 + 72LL) + 40LL) )
            sub_C64ED0("Invalid basic block", 0);
          v51 = (unsigned int)v175;
          v52 = (unsigned int)v175 + 1LL;
          if ( v52 > HIDWORD(v175) )
          {
            sub_C8D5F0(v41, v42, v52, 8u, v39, v40);
            v51 = (unsigned int)v175;
          }
          v174[v51] = v50;
          v46 = (_QWORD *)(v50 + 48);
          v47 = (unsigned int)(v175 + 1);
          LODWORD(v175) = v175 + 1;
          v48 = *v46 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (_QWORD *)v48 == v46 || !v48 || (unsigned int)*(unsigned __int8 *)(v48 - 24) - 30 > 0xA )
            goto LABEL_248;
          if ( *(_BYTE *)(v48 - 24) == 34 )
          {
            v49 = *(_QWORD *)(v48 - 88);
            if ( v47 + 1 > (unsigned __int64)HIDWORD(v175) )
            {
              sub_C8D5F0(v41, v42, v47 + 1, 8u, v39, v40);
              v47 = (unsigned int)v175;
            }
            v174[v47] = v49;
            LODWORD(v175) = v175 + 1;
          }
          v45 += 8;
        }
        while ( v43 != v45 );
        v37 = v44;
        v36 = v42;
        v35 = v131;
        v38 = (__int64 **)v41;
        s2a = 1;
        v40 = *v131;
      }
      sub_29B4290(&v152, *(_QWORD *)(*(_QWORD *)v40 + 72LL));
      v146 = (char **)v148;
      sub_25A3700((__int64 *)&v146, byte_3F871B3, (__int64)byte_3F871B3);
      sub_29AFB10((unsigned int)&v159, (_DWORD)v174, v175, 0, 0, 0, 0, 0, 0, 0, 0, (__int64)&v146, 0);
      sub_29B77F0(&v159, &v152);
      if ( v173 != v38 )
        _libc_free((unsigned __int64)v173);
      sub_C7D6A0(v171[4], 8LL * v172, 8);
      if ( v170 != v171 )
        j_j___libc_free_0((unsigned __int64)v170);
      if ( v168 != &v169 )
        _libc_free((unsigned __int64)v168);
      if ( v167 != (unsigned __int64 *)&v168 )
        _libc_free((unsigned __int64)v167);
      sub_C7D6A0(v165, 8LL * v166, 8);
      if ( v146 != v148 )
        j_j___libc_free_0((unsigned __int64)v146);
      sub_C7D6A0(v157, 8LL * v158, 8);
      v53 = v156;
      if ( v156 )
      {
        v54 = v155;
        v132 = v35;
        v55 = v155 + 40LL * v156;
        do
        {
          if ( *(_QWORD *)v54 != -4096 && *(_QWORD *)v54 != -8192 )
            sub_C7D6A0(*(_QWORD *)(v54 + 16), 8LL * *(unsigned int *)(v54 + 32), 8);
          v54 += 40;
        }
        while ( v55 != v54 );
        v35 = v132;
        v53 = v156;
      }
      sub_C7D6A0(v155, 40 * v53, 8);
      if ( v152 != (unsigned __int64 *)v154 )
        _libc_free((unsigned __int64)v152);
      if ( v174 != v36 )
        _libc_free((unsigned __int64)v174);
      v35 += 3;
    }
    while ( v135 != v35 );
  }
  if ( v181 || (_BYTE)qword_4FEFAC8 )
  {
    v56 = v149;
    v57 = &v149[(unsigned int)v150];
    if ( v57 != v149 )
    {
      do
      {
        v58 = *v56;
        sub_B2CA40(*v56, 0);
        v59 = *(_BYTE *)(v58 + 32);
        *(_BYTE *)(v58 + 32) = v59 & 0xF0;
        if ( (v59 & 0x30) != 0 )
          *(_BYTE *)(v58 + 33) |= 0x40u;
        ++v56;
      }
      while ( v57 != v56 );
    }
    for ( i = *(_QWORD *)(a3 + 32); v121 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
      {
        MEMORY[0x20] &= 0xFFFFFFF0;
        BUG();
      }
      v61 = (*(_BYTE *)(i - 24) & 0x30) == 0;
      *(_BYTE *)(i - 24) &= 0xF0u;
      if ( !v61 )
        *(_BYTE *)(i - 23) |= 0x40u;
    }
    if ( v149 != (__int64 *)v151 )
      _libc_free((unsigned __int64)v149);
    v62 = v124 + 4;
    v63 = v124 + 10;
    goto LABEL_104;
  }
  if ( v149 != (__int64 *)v151 )
    _libc_free((unsigned __int64)v149);
  v62 = v124 + 4;
  v63 = v124 + 10;
  if ( s2a )
  {
LABEL_104:
    memset(v124, 0, 0x60u);
    v124[1] = v62;
    *((_DWORD *)v124 + 4) = 2;
    *((_BYTE *)v124 + 28) = 1;
    v124[7] = v63;
    *((_DWORD *)v124 + 16) = 2;
    *((_BYTE *)v124 + 76) = 1;
    goto LABEL_105;
  }
  v124[2] = 0x100000002LL;
  v124[1] = v62;
  v124[6] = 0;
  v124[7] = v63;
  v124[8] = 2;
  *((_DWORD *)v124 + 18) = 0;
  *((_BYTE *)v124 + 76) = 1;
  *((_DWORD *)v124 + 6) = 0;
  *((_BYTE *)v124 + 28) = 1;
  *v124 = 1;
  v124[4] = &qword_4F82400;
LABEL_105:
  v64 = v182;
  v65 = (unsigned __int64 *)(v182 + 176LL * (unsigned int)v183);
  if ( (unsigned __int64 *)v182 != v65 )
  {
    do
    {
      v66 = *((unsigned int *)v65 - 34);
      v67 = *(v65 - 18);
      v65 -= 22;
      v68 = (unsigned __int64 *)(v67 + 32 * v66);
      if ( (unsigned __int64 *)v67 != v68 )
      {
        do
        {
          v68 -= 4;
          if ( (unsigned __int64 *)*v68 != v68 + 2 )
            j_j___libc_free_0(*v68);
        }
        while ( (unsigned __int64 *)v67 != v68 );
        v67 = v65[4];
      }
      if ( (unsigned __int64 *)v67 != v65 + 6 )
        _libc_free(v67);
      if ( (unsigned __int64 *)*v65 != v65 + 2 )
        j_j___libc_free_0(*v65);
    }
    while ( (unsigned __int64 *)v64 != v65 );
    v65 = (unsigned __int64 *)v182;
  }
  if ( v65 != (unsigned __int64 *)v184 )
    _libc_free((unsigned __int64)v65);
  v69 = v179;
  v70 = v178;
  if ( v179 != v178 )
  {
    do
    {
      if ( *v70 )
        j_j___libc_free_0(*v70);
      v70 += 3;
    }
    while ( v69 != v70 );
    v70 = v178;
  }
  if ( v70 )
    j_j___libc_free_0((unsigned __int64)v70);
  return v124;
}
