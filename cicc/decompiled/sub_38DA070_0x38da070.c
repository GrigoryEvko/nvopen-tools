// Function: sub_38DA070
// Address: 0x38da070
//
__int64 *__fastcall sub_38DA070(
        __int64 *a1,
        char *a2,
        char *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5,
        int *a6,
        _BYTE *a7,
        _DWORD *a8)
{
  __int64 *v9; // r12
  int v11; // r9d
  size_t v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __m128i si128; // xmm0
  __m128i v16; // xmm0
  __m128i v17; // xmm0
  __m128i v18; // xmm0
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  char *v24; // rsi
  char **i; // r14
  bool v26; // zf
  _QWORD *v27; // r14
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  bool v34; // cc
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // r15
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  size_t v48; // rdx
  unsigned __int64 v49; // r8
  unsigned __int64 v50; // r8
  __int64 v51; // rax
  unsigned __int64 v52; // r9
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // r9
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __m128i v68; // xmm0
  __int64 v69; // rax
  unsigned __int64 v70; // rdx
  __m128i v71; // xmm0
  __m128i v72; // xmm0
  unsigned __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __m128i v77; // xmm0
  __m128i v78; // xmm0
  __m128i v79; // xmm0
  unsigned __int64 v80; // rax
  __int64 v81; // rdx
  _QWORD *v82; // rbx
  char *v83; // r12
  unsigned __int64 v84; // rax
  __int64 v85; // r8
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // r15
  char *v89; // r13
  unsigned __int64 v90; // rdx
  unsigned __int64 v91; // rax
  unsigned __int64 v92; // rax
  __int64 v93; // r8
  unsigned __int64 v94; // rax
  unsigned __int64 v95; // rax
  unsigned __int64 v96; // r15
  unsigned __int64 v97; // rdx
  unsigned __int64 v98; // rax
  unsigned __int64 v99; // rax
  __int64 v100; // r8
  unsigned __int64 v101; // rax
  unsigned __int64 v102; // rax
  unsigned __int64 v103; // r15
  unsigned __int64 v104; // rdx
  unsigned __int64 v105; // rax
  unsigned __int64 v106; // rax
  __int64 v107; // r8
  unsigned __int64 v108; // rax
  unsigned __int64 v109; // rax
  unsigned __int64 v110; // r13
  unsigned __int64 v111; // rdx
  unsigned __int64 v112; // rax
  int v113; // eax
  unsigned __int64 v114; // rax
  unsigned __int64 v115; // rdx
  unsigned __int64 v116; // r8
  void *v117; // rdx
  unsigned __int64 v118; // rax
  size_t v119; // rdx
  unsigned __int64 v120; // rax
  unsigned __int64 v121; // r13
  char *v122; // r15
  unsigned __int64 v123; // rdx
  unsigned __int64 v124; // rax
  unsigned __int64 v125; // rax
  __int64 v126; // r9
  unsigned __int64 v127; // rax
  unsigned __int64 v128; // rax
  unsigned __int64 v129; // r13
  unsigned __int64 v130; // rdx
  unsigned __int64 v131; // rax
  unsigned __int64 v132; // rax
  __int64 v133; // r9
  unsigned __int64 v134; // rax
  unsigned __int64 v135; // rax
  _OWORD *v136; // rax
  void *v137; // rdx
  __m128i v138; // xmm0
  __m128i v139; // xmm0
  _BYTE *v140; // rax
  __int64 v141; // rdx
  __int64 v142; // rax
  unsigned __int64 v143; // rdx
  __m128i v144; // xmm0
  __m128i v145; // xmm0
  __m128i v146; // xmm0
  unsigned __int64 v147; // rax
  __int64 v148; // rdx
  __int64 v149; // rax
  void *v150; // rdx
  __m128i v151; // xmm0
  __m128i v152; // xmm0
  __m128i v153; // xmm0
  __m128i v154; // xmm0
  _BYTE *v155; // rax
  __int64 v156; // rdx
  __int64 v157; // rax
  void *v158; // rdx
  __m128i v159; // xmm0
  __m128i v160; // xmm0
  __m128i v161; // xmm0
  _BYTE *v162; // rax
  __int64 v163; // rdx
  __int64 v164; // rax
  void *v165; // rdx
  __m128i v166; // xmm0
  __m128i v167; // xmm0
  _BYTE *v168; // rax
  __int64 v169; // rdx
  unsigned __int64 v170; // [rsp+8h] [rbp-128h]
  __int64 *v171; // [rsp+10h] [rbp-120h]
  __int64 v172; // [rsp+20h] [rbp-110h]
  __int64 v173; // [rsp+28h] [rbp-108h]
  _QWORD *s1; // [rsp+30h] [rbp-100h]
  _QWORD *s1a; // [rsp+30h] [rbp-100h]
  _QWORD *v176; // [rsp+38h] [rbp-F8h]
  _QWORD *v177; // [rsp+38h] [rbp-F8h]
  _QWORD *v178; // [rsp+38h] [rbp-F8h]
  _QWORD *v179; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v180; // [rsp+38h] [rbp-F8h]
  _QWORD *v181; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v182; // [rsp+38h] [rbp-F8h]
  char *v184[2]; // [rsp+50h] [rbp-E0h] BYREF
  char *v185; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int64 v186; // [rsp+68h] [rbp-C8h]
  void *v187; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v188; // [rsp+78h] [rbp-B8h]
  unsigned __int64 v189; // [rsp+80h] [rbp-B0h] BYREF
  unsigned __int64 v190; // [rsp+88h] [rbp-A8h]
  _BYTE v191[16]; // [rsp+90h] [rbp-A0h] BYREF
  _QWORD *v192; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v193; // [rsp+A8h] [rbp-88h]
  _BYTE v194[128]; // [rsp+B0h] [rbp-80h] BYREF

  v9 = a1;
  v184[0] = a2;
  v184[1] = a3;
  *a7 = 0;
  v192 = v194;
  v193 = 0x500000000LL;
  sub_16D2880(v184, (__int64)&v192, 44, -1, 1, (int)a6);
  if ( !(_DWORD)v193 )
  {
    *a4 = 0;
    v12 = 0;
    a4[1] = 0;
    s1 = 0;
    *a5 = 0;
    a5[1] = 0;
LABEL_3:
    v185 = 0;
    v186 = 0;
    goto LABEL_4;
  }
  v27 = v192;
  v28 = 0;
  v29 = sub_16D24E0(v192, byte_3F15413, 6, 0);
  v30 = v27[1];
  if ( v29 < v30 )
  {
    v28 = v30 - v29;
    v30 = v29;
  }
  v189 = *v27 + v30;
  v190 = v28;
  v31 = sub_16D2680((__int64 *)&v189, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v32 = v190;
  v33 = v31 + 1;
  v34 = v33 <= v190;
  *a4 = v189;
  if ( !v34 )
    v33 = v32;
  v35 = v32 - v28 + v33;
  if ( v35 > v32 )
    v35 = v32;
  v34 = (unsigned int)v193 <= 1;
  a4[1] = v35;
  if ( v34 )
  {
    *a5 = 0;
    a5[1] = 0;
LABEL_70:
    s1 = 0;
    v12 = 0;
    goto LABEL_3;
  }
  v36 = 0;
  v177 = v192;
  v37 = sub_16D24E0(v192 + 2, byte_3F15413, 6, 0);
  v38 = v177[3];
  if ( v37 < v38 )
  {
    v36 = v38 - v37;
    v38 = v37;
  }
  v189 = v177[2] + v38;
  v190 = v36;
  v39 = sub_16D2680((__int64 *)&v189, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v40 = v190;
  v41 = v39 + 1;
  v34 = v41 <= v190;
  *a5 = v189;
  if ( !v34 )
    v41 = v40;
  v42 = v40 - v36 + v41;
  if ( v42 > v40 )
    v42 = v40;
  v34 = (unsigned int)v193 <= 2;
  a5[1] = v42;
  if ( v34 )
    goto LABEL_70;
  v43 = 0;
  v178 = v192;
  v44 = sub_16D24E0(v192 + 4, byte_3F15413, 6, 0);
  v45 = v178[5];
  if ( v44 < v45 )
  {
    v43 = v45 - v44;
    v45 = v44;
  }
  v46 = v178[4] + v45;
  v190 = v43;
  v189 = v46;
  v47 = sub_16D2680((__int64 *)&v189, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v48 = v190;
  v49 = v47 + 1;
  if ( v47 + 1 > v190 )
    v49 = v190;
  v50 = v190 - v43 + v49;
  if ( v50 <= v190 )
    v48 = v50;
  s1 = (_QWORD *)v189;
  v12 = v48;
  if ( (unsigned int)v193 <= 3 )
    goto LABEL_3;
  v179 = v192;
  v51 = sub_16D24E0(v192 + 6, byte_3F15413, 6, 0);
  v52 = 0;
  v53 = v51;
  v54 = v179[7];
  if ( v53 < v54 )
  {
    v52 = v54 - v53;
    v54 = v53;
  }
  v55 = v179[6] + v54;
  v190 = v52;
  v180 = v52;
  v189 = v55;
  v56 = sub_16D2680((__int64 *)&v189, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
  v11 = v180;
  v185 = (char *)v189;
  if ( v56 > v190 )
    v56 = v190;
  v57 = v190 - v180 + v56;
  if ( v57 > v190 )
    v57 = v190;
  v186 = v57;
  if ( (unsigned int)v193 > 4 )
  {
    v181 = v192;
    v58 = sub_16D24E0(v192 + 8, byte_3F15413, 6, 0);
    v59 = 0;
    v60 = v58;
    v61 = v181[9];
    if ( v60 < v61 )
    {
      v59 = v61 - v60;
      v61 = v60;
    }
    v62 = v181[8] + v61;
    v190 = v59;
    v182 = v59;
    v189 = v62;
    v63 = sub_16D2680((__int64 *)&v189, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
    v64 = v190;
    v11 = v182;
    v65 = v63 + 1;
    if ( v65 > v190 )
      v65 = v190;
    v66 = v190 - v182 + v65;
    if ( v66 <= v190 )
      v64 = v66;
    v173 = v64;
    v172 = v189;
    goto LABEL_5;
  }
LABEL_4:
  v173 = 0;
  v172 = 0;
LABEL_5:
  v176 = a1 + 2;
  if ( a4[1] - 1 > 0xF )
  {
    *a1 = (__int64)(a1 + 2);
    v189 = 87;
    v13 = sub_22409D0((__int64)a1, &v189, 0);
    v14 = v189;
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    *a1 = v13;
    a1[2] = v14;
    *(__m128i *)v13 = si128;
    *(__m128i *)(v13 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F81B70);
    v16 = _mm_load_si128((const __m128i *)&xmmword_452E880);
LABEL_7:
    *(__m128i *)(v13 + 32) = v16;
    v17 = _mm_load_si128((const __m128i *)&xmmword_3F81BB0);
    *(_DWORD *)(v13 + 80) = 1952670066;
    *(__m128i *)(v13 + 48) = v17;
    v18 = _mm_load_si128((const __m128i *)&xmmword_3F81BC0);
    *(_WORD *)(v13 + 84) = 29285;
    *(_BYTE *)(v13 + 86) = 115;
    *(__m128i *)(v13 + 64) = v18;
    v19 = v189;
    v20 = *a1;
    a1[1] = v189;
    *(_BYTE *)(v20 + v19) = 0;
    goto LABEL_8;
  }
  v22 = a5[1];
  if ( !v22 )
  {
    v189 = 76;
    *a1 = (__int64)v176;
    v69 = sub_22409D0((__int64)a1, &v189, 0);
    v70 = v189;
    v71 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    *a1 = v69;
    a1[2] = v70;
    *(__m128i *)v69 = v71;
    v72 = _mm_load_si128((const __m128i *)&xmmword_3F81B70);
    qmemcpy((void *)(v69 + 64), "d by a comma", 12);
    *(__m128i *)(v69 + 16) = v72;
    *(__m128i *)(v69 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F81B80);
    *(__m128i *)(v69 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F81B90);
    v73 = v189;
    v74 = *a1;
    a1[1] = v189;
    *(_BYTE *)(v74 + v73) = 0;
    goto LABEL_8;
  }
  if ( v22 > 0x10 )
  {
    v189 = 87;
    *a1 = (__int64)v176;
    v13 = sub_22409D0((__int64)a1, &v189, 0);
    v67 = v189;
    v68 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    *a1 = v13;
    a1[2] = v67;
    *(__m128i *)v13 = v68;
    *(__m128i *)(v13 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F81B70);
    v16 = _mm_load_si128((const __m128i *)&xmmword_3F81BA0);
    goto LABEL_7;
  }
  *a6 = 0;
  *a8 = 0;
  if ( !v12 )
  {
    *((_BYTE *)a1 + 16) = 0;
    a1[1] = 0;
    *a1 = (__int64)v176;
    goto LABEL_8;
  }
  v23 = 7;
  v24 = "regular";
  for ( i = (char **)&off_49D9040; ; i += 16 )
  {
    if ( v23 == v12 && !memcmp(s1, v24, v12) )
    {
      v9 = a1;
      goto LABEL_23;
    }
    if ( (char *)v12 == i[5] && !memcmp(s1, i[4], v12) )
    {
      v9 = a1;
      i += 4;
      goto LABEL_23;
    }
    if ( (char *)v12 == i[9] && !memcmp(s1, i[8], v12) )
    {
      v9 = a1;
      i += 8;
      goto LABEL_23;
    }
    if ( (char *)v12 == i[13] && !memcmp(s1, i[12], v12) )
    {
      v9 = a1;
      i += 12;
LABEL_23:
      if ( i != (char **)&unk_49D9300 )
        goto LABEL_24;
LABEL_150:
      v189 = 53;
      *v9 = (__int64)v176;
      v142 = sub_22409D0((__int64)v9, &v189, 0);
      v143 = v189;
      v144 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
      *v9 = v142;
      v9[2] = v143;
      *(__m128i *)v142 = v144;
      v145 = _mm_load_si128((const __m128i *)&xmmword_3F81BD0);
      *(_DWORD *)(v142 + 48) = 1887007776;
      *(__m128i *)(v142 + 16) = v145;
      v146 = _mm_load_si128((const __m128i *)&xmmword_3F81BE0);
      *(_BYTE *)(v142 + 52) = 101;
      *(__m128i *)(v142 + 32) = v146;
      v147 = v189;
      v148 = *v9;
      v9[1] = v189;
      *(_BYTE *)(v148 + v147) = 0;
      goto LABEL_8;
    }
    if ( i + 16 == (char **)&unk_49D92C0 )
      break;
    v24 = i[16];
    v23 = (__int64)i[17];
  }
  v9 = a1;
  if ( i[17] == (char *)v12 && !memcmp(s1, i[16], v12) )
  {
    i = (char **)&unk_49D92C0;
    goto LABEL_24;
  }
  if ( v12 != 35 )
    goto LABEL_150;
  if ( *s1 ^ 0x6C5F646165726874LL | s1[1] ^ 0x696E695F6C61636FLL )
    goto LABEL_150;
  if ( s1[2] ^ 0x6974636E75665F74LL | s1[3] ^ 0x746E696F705F6E6FLL )
    goto LABEL_150;
  if ( *((_WORD *)s1 + 16) != 29285 )
    goto LABEL_150;
  i = &off_49D92E0;
  if ( *((_BYTE *)s1 + 34) != 115 )
    goto LABEL_150;
LABEL_24:
  v26 = v186 == 0;
  *a6 = ((char *)i - (char *)&off_49D9040) >> 5;
  *a7 = 1;
  if ( v26 )
  {
    v26 = *a6 == 8;
    *v9 = (__int64)v176;
    if ( v26 )
    {
      v189 = 73;
      v75 = sub_22409D0((__int64)v9, &v189, 0);
      v76 = v189;
      v77 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
      *v9 = v75;
      v9[2] = v76;
      *(__m128i *)v75 = v77;
      v78 = _mm_load_si128((const __m128i *)&xmmword_3F81BF0);
      *(_QWORD *)(v75 + 64) = 0x6569666963657073LL;
      *(__m128i *)(v75 + 16) = v78;
      v79 = _mm_load_si128((const __m128i *)&xmmword_3F81C00);
      *(_BYTE *)(v75 + 72) = 114;
      *(__m128i *)(v75 + 32) = v79;
      *(__m128i *)(v75 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F81C10);
      v80 = v189;
      v81 = *v9;
      v9[1] = v189;
      *(_BYTE *)(v81 + v80) = 0;
    }
    else
    {
      v9[1] = 0;
      *((_BYTE *)v9 + 16) = 0;
    }
    goto LABEL_8;
  }
  v189 = (unsigned __int64)v191;
  v190 = 0x100000000LL;
  sub_16D2880(&v185, (__int64)&v189, 43, -1, 0, v11);
  v82 = (_QWORD *)v189;
  s1a = (_QWORD *)(v189 + 16LL * (unsigned int)v190);
  if ( s1a == (_QWORD *)v189 )
  {
    v113 = *a6;
    goto LABEL_112;
  }
  v171 = v9;
  while ( 2 )
  {
    v83 = (char *)&unk_49D8E80;
    while ( 2 )
    {
      v110 = 0;
      v111 = sub_16D24E0(v82, byte_3F15413, 6, 0);
      v112 = v82[1];
      if ( v111 < v112 )
      {
        v110 = v112 - v111;
        v112 = v111;
      }
      v187 = (void *)(*v82 + v112);
      v188 = v110;
      v84 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v85 = *((_QWORD *)v83 + 2);
      v86 = v84 + 1;
      if ( v86 > v188 )
        v86 = v188;
      v87 = v188 - v110 + v86;
      if ( v87 > v188 )
        v87 = v188;
      if ( v85 == v87 && (!v85 || !memcmp(v187, *((const void **)v83 + 1), *((_QWORD *)v83 + 2))) )
        goto LABEL_109;
      v88 = 0;
      v89 = v83 + 40;
      v90 = sub_16D24E0(v82, byte_3F15413, 6, 0);
      v91 = v82[1];
      if ( v90 < v91 )
      {
        v88 = v91 - v90;
        v91 = v90;
      }
      v187 = (void *)(*v82 + v91);
      v188 = v88;
      v92 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v93 = *((_QWORD *)v83 + 7);
      v94 = v92 + 1;
      if ( v94 > v188 )
        v94 = v188;
      v95 = v188 - v88 + v94;
      if ( v95 > v188 )
        v95 = v188;
      if ( v93 == v95 && (!v93 || !memcmp(v187, *((const void **)v83 + 6), *((_QWORD *)v83 + 7))) )
        goto LABEL_119;
      v96 = 0;
      v89 = v83 + 80;
      v97 = sub_16D24E0(v82, byte_3F15413, 6, 0);
      v98 = v82[1];
      if ( v97 < v98 )
      {
        v96 = v98 - v97;
        v98 = v97;
      }
      v187 = (void *)(*v82 + v98);
      v188 = v96;
      v99 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v100 = *((_QWORD *)v83 + 12);
      v101 = v99 + 1;
      if ( v101 > v188 )
        v101 = v188;
      v102 = v188 - v96 + v101;
      if ( v102 > v188 )
        v102 = v188;
      if ( v100 == v102 && (!v100 || !memcmp(v187, *((const void **)v83 + 11), *((_QWORD *)v83 + 12))) )
        goto LABEL_119;
      v103 = 0;
      v89 = v83 + 120;
      v104 = sub_16D24E0(v82, byte_3F15413, 6, 0);
      v105 = v82[1];
      if ( v104 < v105 )
      {
        v103 = v105 - v104;
        v105 = v104;
      }
      v187 = (void *)(*v82 + v105);
      v188 = v103;
      v106 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v107 = *((_QWORD *)v83 + 17);
      v108 = v106 + 1;
      if ( v108 > v188 )
        v108 = v188;
      v109 = v188 - v103 + v108;
      if ( v109 > v188 )
        v109 = v188;
      if ( v107 == v109 && (!v107 || !memcmp(v187, *((const void **)v83 + 16), *((_QWORD *)v83 + 17))) )
      {
LABEL_119:
        v83 = v89;
        goto LABEL_109;
      }
      if ( v83 + 160 != (char *)&unk_49D8FC0 )
      {
        v83 += 160;
        continue;
      }
      break;
    }
    v114 = sub_16D24E0(v82, byte_3F15413, 6, 0);
    v115 = v82[1];
    v116 = 0;
    if ( v114 < v115 )
    {
      v116 = v115 - v114;
      v115 = v114;
    }
    v117 = (void *)(*v82 + v115);
    v188 = v116;
    v187 = v117;
    v170 = v116;
    v118 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
    v119 = *((_QWORD *)v83 + 22);
    if ( v118 > v188 )
      v118 = v188;
    v120 = v188 - v170 + v118;
    if ( v120 > v188 )
      v120 = v188;
    if ( v119 != v120 || v119 && memcmp(v187, *((const void **)v83 + 21), v119) )
    {
      v121 = 0;
      v122 = v83 + 200;
      v123 = sub_16D24E0(v82, byte_3F15413, 6, 0);
      v124 = v82[1];
      if ( v123 < v124 )
      {
        v121 = v124 - v123;
        v124 = v123;
      }
      v187 = (void *)(*v82 + v124);
      v188 = v121;
      v125 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v126 = *((_QWORD *)v83 + 27);
      v127 = v125 + 1;
      if ( v127 > v188 )
        v127 = v188;
      v128 = v188 - v121 + v127;
      if ( v128 > v188 )
        v128 = v188;
      if ( v126 != v128 || v126 && memcmp(v187, *((const void **)v83 + 26), *((_QWORD *)v83 + 27)) )
      {
        v129 = 0;
        v122 = v83 + 240;
        v130 = sub_16D24E0(v82, byte_3F15413, 6, 0);
        v131 = v82[1];
        if ( v130 < v131 )
        {
          v129 = v131 - v130;
          v131 = v130;
        }
        v187 = (void *)(*v82 + v131);
        v188 = v129;
        v132 = sub_16D2680((__int64 *)&v187, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
        v133 = *((_QWORD *)v83 + 32);
        v134 = v132 + 1;
        if ( v134 > v188 )
          v134 = v188;
        v135 = v188 - v129 + v134;
        if ( v135 > v188 )
          v135 = v188;
        if ( v133 != v135 || v133 && memcmp(v187, *((const void **)v83 + 31), *((_QWORD *)v83 + 32)) )
          goto LABEL_147;
      }
      v83 = v122;
LABEL_109:
      if ( v83 != (char *)&unk_49D9038 )
        goto LABEL_110;
LABEL_147:
      v9 = v171;
      v187 = (void *)46;
      *v171 = (__int64)v176;
      v136 = (_OWORD *)sub_22409D0((__int64)v171, (unsigned __int64 *)&v187, 0);
      v137 = v187;
      v138 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
      *v171 = (__int64)v136;
      v171[2] = (__int64)v137;
      *v136 = v138;
      v139 = _mm_load_si128((const __m128i *)&xmmword_3F81C20);
      qmemcpy(v136 + 2, "alid attribute", 14);
      v136[1] = v139;
      v140 = v187;
      v141 = *v171;
      v171[1] = (__int64)v187;
      v140[v141] = 0;
      goto LABEL_115;
    }
    v83 += 160;
LABEL_110:
    v82 += 2;
    v113 = *(_DWORD *)v83 | *a6;
    *a6 = v113;
    if ( s1a != v82 )
      continue;
    break;
  }
  v9 = v171;
LABEL_112:
  if ( v173 )
  {
    if ( (_BYTE)v113 == 8 )
    {
      if ( sub_16D2B80(v172, v173, 0, (unsigned __int64 *)&v187) || v187 != (void *)(unsigned int)v187 )
      {
        v187 = (void *)50;
        *v9 = (__int64)v176;
        v164 = sub_22409D0((__int64)v9, (unsigned __int64 *)&v187, 0);
        v165 = v187;
        v166 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
        *v9 = v164;
        v9[2] = (__int64)v165;
        *(__m128i *)v164 = v166;
        v167 = _mm_load_si128((const __m128i *)&xmmword_3F81C80);
        *(_WORD *)(v164 + 48) = 25978;
        *(__m128i *)(v164 + 16) = v167;
        *(__m128i *)(v164 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F81C90);
        v168 = v187;
        v169 = *v9;
        v9[1] = (__int64)v187;
        v168[v169] = 0;
      }
      else
      {
        *a8 = (_DWORD)v187;
        *v9 = (__int64)v176;
LABEL_114:
        v9[1] = 0;
        *((_BYTE *)v9 + 16) = 0;
      }
    }
    else
    {
      v187 = (void *)103;
      *v9 = (__int64)v176;
      v149 = sub_22409D0((__int64)v9, (unsigned __int64 *)&v187, 0);
      v150 = v187;
      v151 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
      *v9 = v149;
      v9[2] = (__int64)v150;
      *(__m128i *)v149 = v151;
      v152 = _mm_load_si128((const __m128i *)&xmmword_3F81C30);
      *(_DWORD *)(v149 + 96) = 1970565983;
      *(__m128i *)(v149 + 16) = v152;
      v153 = _mm_load_si128((const __m128i *)&xmmword_3F81C40);
      *(_WORD *)(v149 + 100) = 29538;
      *(__m128i *)(v149 + 32) = v153;
      v154 = _mm_load_si128((const __m128i *)&xmmword_3F81C50);
      *(_BYTE *)(v149 + 102) = 39;
      *(__m128i *)(v149 + 48) = v154;
      *(__m128i *)(v149 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F81C60);
      *(__m128i *)(v149 + 80) = _mm_load_si128((const __m128i *)&xmmword_3F81C70);
      v155 = v187;
      v156 = *v9;
      v9[1] = (__int64)v187;
      v155[v156] = 0;
    }
  }
  else
  {
    *v9 = (__int64)v176;
    if ( v113 != 8 )
      goto LABEL_114;
    v187 = (void *)73;
    v157 = sub_22409D0((__int64)v9, (unsigned __int64 *)&v187, 0);
    v158 = v187;
    v159 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    *v9 = v157;
    v9[2] = (__int64)v158;
    *(__m128i *)v157 = v159;
    v160 = _mm_load_si128((const __m128i *)&xmmword_3F81BF0);
    *(_QWORD *)(v157 + 64) = 0x6569666963657073LL;
    *(__m128i *)(v157 + 16) = v160;
    v161 = _mm_load_si128((const __m128i *)&xmmword_3F81C00);
    *(_BYTE *)(v157 + 72) = 114;
    *(__m128i *)(v157 + 32) = v161;
    *(__m128i *)(v157 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F81C10);
    v162 = v187;
    v163 = *v9;
    v9[1] = (__int64)v187;
    v162[v163] = 0;
  }
LABEL_115:
  if ( (_BYTE *)v189 != v191 )
    _libc_free(v189);
LABEL_8:
  if ( v192 != (_QWORD *)v194 )
    _libc_free((unsigned __int64)v192);
  return v9;
}
