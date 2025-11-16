// Function: sub_32C9810
// Address: 0x32c9810
//
unsigned __int64 __fastcall sub_32C9810(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  bool v5; // zf
  __int64 *v8; // rbx
  __int64 v9; // r12
  unsigned __int16 *v10; // rdx
  int v11; // eax
  __int64 v12; // rdx
  __int64 *v13; // r13
  unsigned int v14; // r15d
  int v15; // eax
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __m128i v18; // rax
  __int64 *v19; // r11
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int16 v24; // dx
  __int64 v25; // rsi
  __m128i v26; // xmm2
  unsigned __int64 v27; // r13
  __int64 v28; // rdi
  __int128 v29; // rax
  __int64 v30; // rdi
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // r12
  __int64 v33; // r13
  __int128 v34; // rax
  int v35; // r12d
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 *v38; // r11
  __int64 v39; // rdx
  int v40; // ebx
  unsigned __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // r13
  __int128 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rdx
  unsigned __int64 v48; // rcx
  int v49; // r14d
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  unsigned __int16 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int16 v61; // ax
  __int16 v62; // ax
  __int64 v63; // rsi
  __int64 v64; // r12
  __m128i v65; // xmm1
  __int64 (__fastcall *v66)(__int64, unsigned __int64, _QWORD, __int64, __m128i *); // r13
  __int64 *v67; // r8
  void *v68; // rax
  __int16 v69; // ax
  __int64 v70; // rcx
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // r11
  __int64 v74; // r12
  __int64 v75; // rdx
  __int64 v76; // r13
  __int64 (__fastcall *v77)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v78; // rsi
  __int64 *v79; // rcx
  unsigned int v80; // ebx
  __int128 v81; // rax
  int v82; // r9d
  __int64 v83; // r11
  __int128 v84; // rcx
  __int64 v85; // rdx
  __int16 v86; // ax
  __int64 v87; // rdx
  int v88; // esi
  __int64 v89; // rax
  __int64 v90; // rdx
  unsigned __int64 v91; // r13
  __int64 v92; // rdi
  __int128 v93; // rax
  __int128 v94; // rax
  int v95; // ecx
  __int64 *v96; // r14
  int v97; // ebx
  __int64 v98; // rax
  __int64 v99; // rdi
  __int64 v100; // r12
  __int128 v101; // rax
  __int128 v102; // rax
  __int128 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rdx
  __int64 v106; // rdx
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rdx
  int v110; // eax
  int v111; // r14d
  __int64 v112; // rdx
  __int64 v113; // rdx
  bool v114; // al
  __int64 v115; // rdx
  __int128 v116; // rax
  __int128 v117; // [rsp-30h] [rbp-240h]
  __int128 v118; // [rsp-20h] [rbp-230h]
  __int128 v119; // [rsp-20h] [rbp-230h]
  __int128 v120; // [rsp-20h] [rbp-230h]
  __int128 v121; // [rsp-20h] [rbp-230h]
  __int128 v122; // [rsp-10h] [rbp-220h]
  __int128 v123; // [rsp-10h] [rbp-220h]
  __int128 v124; // [rsp-10h] [rbp-220h]
  __int128 v125; // [rsp-10h] [rbp-220h]
  __int128 v126; // [rsp-10h] [rbp-220h]
  __int128 v127; // [rsp-10h] [rbp-220h]
  __int128 v128; // [rsp-10h] [rbp-220h]
  __int128 v129; // [rsp-10h] [rbp-220h]
  __int128 v130; // [rsp+0h] [rbp-210h]
  __int64 *v131; // [rsp+10h] [rbp-200h]
  __int128 v132; // [rsp+10h] [rbp-200h]
  __int128 v133; // [rsp+20h] [rbp-1F0h]
  __int128 v134; // [rsp+20h] [rbp-1F0h]
  __int64 v135; // [rsp+28h] [rbp-1E8h]
  __m128i v136; // [rsp+30h] [rbp-1E0h] BYREF
  __int64 v137; // [rsp+40h] [rbp-1D0h]
  unsigned int v138; // [rsp+4Ch] [rbp-1C4h]
  __int128 v139; // [rsp+50h] [rbp-1C0h]
  __int128 v140; // [rsp+60h] [rbp-1B0h]
  __int128 v141; // [rsp+70h] [rbp-1A0h]
  __int64 v142; // [rsp+80h] [rbp-190h]
  __int64 v143; // [rsp+88h] [rbp-188h]
  __int64 *v144; // [rsp+90h] [rbp-180h]
  unsigned __int64 v145; // [rsp+98h] [rbp-178h]
  __int64 v146; // [rsp+A0h] [rbp-170h]
  __int64 *v147; // [rsp+A8h] [rbp-168h]
  __int128 v148; // [rsp+B0h] [rbp-160h]
  __int64 v149; // [rsp+C0h] [rbp-150h]
  unsigned __int64 v150; // [rsp+C8h] [rbp-148h]
  __int64 v151; // [rsp+D0h] [rbp-140h]
  __int64 v152; // [rsp+D8h] [rbp-138h]
  __int64 v153; // [rsp+E0h] [rbp-130h]
  __int64 v154; // [rsp+E8h] [rbp-128h]
  __int64 v155; // [rsp+F0h] [rbp-120h]
  __int64 v156; // [rsp+F8h] [rbp-118h]
  __int64 *v157; // [rsp+100h] [rbp-110h]
  unsigned __int64 v158; // [rsp+108h] [rbp-108h]
  __int64 v159; // [rsp+110h] [rbp-100h]
  __int64 v160; // [rsp+118h] [rbp-F8h]
  __int64 v161; // [rsp+120h] [rbp-F0h]
  __int64 v162; // [rsp+128h] [rbp-E8h]
  __int64 v163; // [rsp+130h] [rbp-E0h]
  __int64 v164; // [rsp+138h] [rbp-D8h]
  __int64 v165; // [rsp+140h] [rbp-D0h]
  __int64 v166; // [rsp+148h] [rbp-C8h]
  __int64 v167; // [rsp+150h] [rbp-C0h]
  __int64 v168; // [rsp+158h] [rbp-B8h]
  __int64 v169; // [rsp+160h] [rbp-B0h]
  __int64 v170; // [rsp+168h] [rbp-A8h]
  __int64 v171; // [rsp+170h] [rbp-A0h]
  __int64 v172; // [rsp+178h] [rbp-98h]
  __int64 v173; // [rsp+180h] [rbp-90h]
  __int64 v174; // [rsp+188h] [rbp-88h]
  __int64 v175; // [rsp+190h] [rbp-80h]
  __int64 v176; // [rsp+198h] [rbp-78h]
  char v177; // [rsp+1ABh] [rbp-65h] BYREF
  int v178; // [rsp+1ACh] [rbp-64h] BYREF
  __m128i v179; // [rsp+1B0h] [rbp-60h] BYREF
  __int64 v180; // [rsp+1C0h] [rbp-50h] BYREF
  int v181; // [rsp+1C8h] [rbp-48h]
  __m128i v182; // [rsp+1D0h] [rbp-40h] BYREF

  v5 = *(_BYTE *)(a1 + 32) == 0;
  v150 = a2;
  *(_QWORD *)&v148 = a3;
  LODWORD(v147) = a5;
  LOBYTE(v149) = a5;
  if ( !v5 )
    return 0;
  v8 = (__int64 *)a1;
  v9 = (unsigned int)v148;
  LODWORD(v139) = v148;
  v10 = (unsigned __int16 *)(16LL * (unsigned int)v148 + *(_QWORD *)(v150 + 48));
  v142 = 16LL * (unsigned int)v148;
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v179.m128i_i16[0] = v11;
  v179.m128i_i64[1] = v12;
  if ( !(_WORD)v11 )
  {
    if ( !sub_30070B0((__int64)&v179) )
      goto LABEL_29;
    v61 = sub_3009970((__int64)&v179, a2, v51, v52, v53);
LABEL_42:
    if ( v61 == 11 )
      goto LABEL_6;
LABEL_29:
    v54 = v179.m128i_i16[0];
    v11 = v179.m128i_u16[0];
    if ( !v179.m128i_i16[0] )
    {
      if ( !sub_30070B0((__int64)&v179) )
        goto LABEL_31;
      v62 = sub_3009970((__int64)&v179, a2, v55, v56, v57);
LABEL_45:
      if ( v62 == 12 )
        goto LABEL_6;
      v54 = v179.m128i_i16[0];
LABEL_31:
      if ( !v54 )
      {
        if ( !sub_30070B0((__int64)&v179) )
          return 0;
        v54 = sub_3009970((__int64)&v179, a2, v58, v59, v60);
LABEL_39:
        if ( v54 == 13 )
          goto LABEL_6;
        return 0;
      }
LABEL_37:
      if ( (unsigned __int16)(v54 - 17) <= 0xD3u )
        v54 = word_4456580[v54 - 1];
      goto LABEL_39;
    }
LABEL_34:
    if ( (unsigned __int16)(v11 - 17) > 0xD3u )
    {
      if ( (_WORD)v11 == 12 )
        goto LABEL_6;
      v54 = v179.m128i_i16[0];
      goto LABEL_37;
    }
    v62 = word_4456580[v11 - 1];
    goto LABEL_45;
  }
  if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
  {
    v61 = word_4456580[v11 - 1];
    goto LABEL_42;
  }
  if ( (_WORD)v11 != 11 )
    goto LABEL_34;
LABEL_6:
  v13 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
  v14 = sub_2FEC360(*(_QWORD *)(a1 + 8), v179.m128i_u32[0], v179.m128i_i64[1], v13);
  if ( !v14 )
    return 0;
  v15 = sub_2FEC400(*(_QWORD *)(a1 + 8), v179.m128i_u32[0], v179.m128i_i64[1], v13);
  v16 = *(_QWORD *)(a1 + 8);
  v177 = 0;
  v178 = v15;
  v17 = *(__int64 (**)())(*(_QWORD *)v16 + 2568LL);
  if ( v17 == sub_325D4A0 )
    return 0;
  v18.m128i_i64[0] = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, _QWORD, int *, char *, _QWORD))v17)(
                       v16,
                       v150,
                       v148,
                       *v8,
                       v14,
                       &v178,
                       &v177,
                       (unsigned __int8)v147);
  v19 = (__int64 *)v18.m128i_i64[0];
  v136 = v18;
  if ( !v18.m128i_i64[0] )
    return 0;
  if ( *(_DWORD *)(v18.m128i_i64[0] + 24) != 328 )
  {
    v146 = v18.m128i_i64[0];
    v182.m128i_i64[0] = v18.m128i_i64[0];
    sub_32B3B20((__int64)(v8 + 71), v182.m128i_i64);
    v19 = (__int64 *)v146;
    if ( *(int *)(v146 + 88) < 0 )
    {
      *(_DWORD *)(v146 + 88) = *((_DWORD *)v8 + 12);
      v22 = *((unsigned int *)v8 + 12);
      if ( v22 + 1 > (unsigned __int64)*((unsigned int *)v8 + 13) )
      {
        sub_C8D5F0((__int64)(v8 + 5), v8 + 7, v22 + 1, 8u, v20, v21);
        v22 = *((unsigned int *)v8 + 12);
        v19 = (__int64 *)v146;
      }
      *(_QWORD *)(v8[5] + 8 * v22) = v19;
      ++*((_DWORD *)v8 + 12);
    }
  }
  v138 = v178;
  if ( v178 > 0 )
  {
    v23 = *(_QWORD *)(v150 + 48) + v142;
    v24 = *(_WORD *)v23;
    v25 = *(_QWORD *)(v150 + 80);
    v146 = *(_QWORD *)(v23 + 8);
    if ( v177 )
    {
      v26 = _mm_load_si128(&v136);
      v182.m128i_i64[0] = v25;
      v135 = v148;
      v141 = (__int128)v26;
      v149 = v24;
      if ( v25 )
      {
        v144 = v19;
        sub_B96E90((__int64)&v182, v25, 1);
        v19 = v144;
      }
      v27 = v150;
      v28 = *v8;
      v144 = v19;
      v182.m128i_i32[2] = *(_DWORD *)(v150 + 72);
      *(_QWORD *)&v29 = sub_33FE730(v28, &v182, (unsigned int)v149, v146, 0, 1.5);
      *(_QWORD *)&v133 = v27;
      v30 = *v8;
      v31 = *((_QWORD *)&v29 + 1);
      v139 = v29;
      *((_QWORD *)&v29 + 1) = v29;
      *(_QWORD *)&v29 = v9 | v135 & 0xFFFFFFFF00000000LL;
      v32 = v27;
      *((_QWORD *)&v133 + 1) = v29;
      v33 = v29;
      *((_QWORD *)&v122 + 1) = v29;
      *(_QWORD *)&v122 = v32;
      *(_QWORD *)&v34 = sub_3405C90(
                          v30,
                          98,
                          (unsigned int)&v182,
                          v149,
                          v146,
                          a4,
                          __PAIR128__(v31, *((unsigned __int64 *)&v29 + 1)),
                          v122);
      *((_QWORD *)&v123 + 1) = v33;
      *(_QWORD *)&v123 = v32;
      v35 = v146;
      v36 = *v8;
      v140 = v34;
      v37 = sub_3405C90(v36, 97, (unsigned int)&v182, v149, v146, a4, v34, v123);
      v38 = v144;
      v175 = v37;
      *(_QWORD *)&v140 = v37;
      v176 = v39;
      LODWORD(v143) = 0;
      v144 = v8;
      v40 = a4;
      v41 = *((_QWORD *)&v141 + 1);
      *((_QWORD *)&v140 + 1) = (unsigned int)v39 | *((_QWORD *)&v140 + 1) & 0xFFFFFFFF00000000LL;
      v42 = (__int64)v38;
      do
      {
        v43 = v42;
        *((_QWORD *)&v124 + 1) = v41;
        *(_QWORD *)&v124 = v42;
        *((_QWORD *)&v118 + 1) = v41;
        *(_QWORD *)&v118 = v42;
        *(_QWORD *)&v44 = sub_3405C90(*v144, 98, (unsigned int)&v182, v149, v35, v40, v118, v124);
        v141 = v44;
        v173 = sub_3405C90(*v144, 98, (unsigned int)&v182, v149, v35, v40, v140, v44);
        v174 = v45;
        *(_QWORD *)&v141 = v173;
        *(_QWORD *)&v125 = v173;
        *((_QWORD *)&v141 + 1) = (unsigned int)v45 | *((_QWORD *)&v141 + 1) & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v125 + 1) = *((_QWORD *)&v141 + 1);
        v171 = sub_3405C90(*v144, 97, (unsigned int)&v182, v149, v35, v40, v139, v125);
        v172 = v46;
        *((_QWORD *)&v126 + 1) = (unsigned int)v46 | *((_QWORD *)&v141 + 1) & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v126 = v171;
        *((_QWORD *)&v119 + 1) = v41;
        *(_QWORD *)&v119 = v43;
        v42 = sub_3405C90(*v144, 98, (unsigned int)&v182, v149, v35, v40, v119, v126);
        LODWORD(v143) = v143 + 1;
        v170 = v47;
        v48 = (unsigned int)v47 | v41 & 0xFFFFFFFF00000000LL;
        v169 = v42;
        v41 = v48;
      }
      while ( v138 != (_DWORD)v143 );
      v49 = v40;
      v19 = (__int64 *)v42;
      *(_QWORD *)&v141 = v42;
      *((_QWORD *)&v141 + 1) = v48;
      v8 = v144;
      if ( !(_BYTE)v147 )
      {
        v167 = sub_3405C90(*v144, 98, (unsigned int)&v182, v149, v146, v49, v141, v133);
        v19 = (__int64 *)v167;
        *(_QWORD *)&v141 = v167;
        v168 = v115;
        *((_QWORD *)&v141 + 1) = (unsigned int)v115 | *((_QWORD *)&v141 + 1) & 0xFFFFFFFF00000000LL;
      }
      if ( v182.m128i_i64[0] )
      {
        v149 = (__int64)v19;
        sub_B91220((__int64)&v182, v182.m128i_i64[0]);
        v19 = (__int64 *)v149;
      }
      v50 = *((_QWORD *)&v141 + 1);
    }
    else
    {
      v182.m128i_i64[0] = v25;
      v91 = v136.m128i_u64[1];
      *(_QWORD *)&v141 = v150;
      *((_QWORD *)&v141 + 1) = v148;
      v143 = v24;
      if ( v25 )
      {
        *(_QWORD *)&v140 = v19;
        sub_B96E90((__int64)&v182, v25, 1);
        v19 = (__int64 *)v140;
      }
      v92 = *v8;
      v131 = v19;
      v182.m128i_i32[2] = *(_DWORD *)(v150 + 72);
      *(_QWORD *)&v93 = sub_33FE730(v92, &v182, (unsigned int)v143, v146, 0, -3.0);
      v130 = v93;
      *(_QWORD *)&v94 = sub_33FE730(*v8, &v182, (unsigned int)v143, v146, 0, -0.5);
      v95 = a4;
      LODWORD(v140) = 0;
      v134 = v94;
      v96 = v8;
      v97 = v95;
      v137 = (unsigned int)v139;
      v98 = (__int64)v131;
      while ( 1 )
      {
        while ( 1 )
        {
          v99 = *v96;
          *((_QWORD *)&v128 + 1) = v91;
          *(_QWORD *)&v128 = v98;
          v100 = v98;
          *(_QWORD *)&v141 = v150;
          *((_QWORD *)&v141 + 1) = v137 | *((_QWORD *)&v141 + 1) & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v101 = sub_3405C90(
                               v99,
                               98,
                               (unsigned int)&v182,
                               v143,
                               v146,
                               v97,
                               __PAIR128__(*((unsigned __int64 *)&v141 + 1), v150),
                               v128);
          *((_QWORD *)&v129 + 1) = v91;
          *(_QWORD *)&v129 = v100;
          v132 = v101;
          *(_QWORD *)&v102 = sub_3405C90(*v96, 98, (unsigned int)&v182, v143, v146, v97, v101, v129);
          *(_QWORD *)&v103 = sub_3405C90(*v96, 96, (unsigned int)&v182, v143, v146, v97, v102, v130);
          LODWORD(v140) = v140 + 1;
          v139 = v103;
          v144 = 0;
          v145 &= 0xFFFFFFFF00000000LL;
          if ( !(_BYTE)v149 )
            break;
          *((_QWORD *)&v121 + 1) = v91;
          *(_QWORD *)&v121 = v100;
          v107 = sub_3405C90(*v96, 98, (unsigned int)&v182, v143, v146, v97, v121, v134);
          v166 = v108;
          v165 = v107;
          v144 = (__int64 *)v107;
          v145 = (unsigned int)v108 | v145 & 0xFFFFFFFF00000000LL;
          v98 = sub_3405C90(*v96, 98, (unsigned int)&v182, v143, v146, v97, __PAIR128__(v145, v107), v139);
          v164 = v109;
          v163 = v98;
          v91 = (unsigned int)v109 | v91 & 0xFFFFFFFF00000000LL;
          if ( (_DWORD)v140 == v138 )
          {
            v19 = (__int64 *)v98;
            v8 = v96;
            goto LABEL_68;
          }
        }
        if ( (unsigned int)v140 >= v138 )
          break;
        *((_QWORD *)&v120 + 1) = v91;
        *(_QWORD *)&v120 = v100;
        v104 = sub_3405C90(*v96, 98, (unsigned int)&v182, v143, v146, v97, v120, v134);
        v154 = v105;
        v153 = v104;
        v144 = (__int64 *)v104;
        v145 = (unsigned int)v105 | v145 & 0xFFFFFFFF00000000LL;
        v98 = sub_3405C90(*v96, 98, (unsigned int)&v182, v143, v146, v97, __PAIR128__(v145, v104), v139);
        v152 = v106;
        v151 = v98;
        v91 = (unsigned int)v106 | v91 & 0xFFFFFFFF00000000LL;
      }
      v110 = v97;
      v8 = v96;
      v111 = v110;
      v161 = sub_3405C90(*v8, 98, (unsigned int)&v182, v143, v146, v110, v132, v134);
      v144 = (__int64 *)v161;
      v162 = v112;
      v145 = (unsigned int)v112 | v145 & 0xFFFFFFFF00000000LL;
      v159 = sub_3405C90(*v8, 98, (unsigned int)&v182, v143, v146, v111, __PAIR128__(v145, v161), v139);
      v19 = (__int64 *)v159;
      v160 = v113;
      v91 = (unsigned int)v113 | v91 & 0xFFFFFFFF00000000LL;
LABEL_68:
      if ( v182.m128i_i64[0] )
      {
        v149 = (__int64)v19;
        sub_B91220((__int64)&v182, v182.m128i_i64[0]);
        v19 = (__int64 *)v149;
      }
      v50 = v91;
    }
    v158 = v50;
    v157 = v19;
    v136.m128i_i64[0] = (__int64)v19;
    v136.m128i_i64[1] = (unsigned int)v50 | v136.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  if ( !(_BYTE)v147 )
  {
    v63 = *(_QWORD *)(v150 + 80);
    v180 = v63;
    if ( v63 )
    {
      v149 = (__int64)v19;
      sub_B96E90((__int64)&v180, v63, 1);
      v19 = (__int64 *)v149;
    }
    v64 = v8[1];
    v147 = v19;
    v65 = _mm_loadu_si128(&v179);
    v181 = *(_DWORD *)(v150 + 72);
    v66 = *(__int64 (__fastcall **)(__int64, unsigned __int64, _QWORD, __int64, __m128i *))(*(_QWORD *)v64 + 2584LL);
    v67 = *(__int64 **)(*v8 + 40);
    v182 = v65;
    v149 = (__int64)v67;
    v68 = sub_300AC80((unsigned __int16 *)&v182, v150);
    v69 = sub_2E79010((__int64 *)v149, (__int64)v68);
    v70 = *v8;
    v182.m128i_i16[0] = v69;
    v71 = v66(v64, v150, v148, v70, &v182);
    v72 = v8[1];
    v73 = (__int64)v147;
    v74 = v71;
    v76 = v75;
    v149 = *v8;
    v77 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v72 + 2592LL);
    if ( v77 == sub_3030110 )
    {
      v78 = *(_QWORD *)(v150 + 80);
      v79 = *(__int64 **)(*(_QWORD *)(v150 + 48) + v142 + 8);
      v80 = *(unsigned __int16 *)(*(_QWORD *)(v150 + 48) + v142);
      v182.m128i_i64[0] = v78;
      if ( v78 )
      {
        v147 = v79;
        *(_QWORD *)&v148 = v73;
        sub_B96E90((__int64)&v182, v78, 1);
        v79 = v147;
        v73 = v148;
      }
      *(_QWORD *)&v148 = v73;
      v182.m128i_i32[2] = *(_DWORD *)(v150 + 72);
      *(_QWORD *)&v81 = sub_33FE730(v149, &v182, v80, v79, 0, 0.0);
      v83 = v148;
      v84 = v81;
      if ( v182.m128i_i64[0] )
      {
        v150 = v148;
        v148 = v81;
        sub_B91220((__int64)&v182, v182.m128i_i64[0]);
        v84 = v148;
        v83 = v150;
      }
    }
    else
    {
      *(_QWORD *)&v116 = v77(v72, v150, v148, v149);
      v83 = (__int64)v147;
      v84 = v116;
    }
    v85 = *(_QWORD *)(v74 + 48) + 16LL * (unsigned int)v76;
    v86 = *(_WORD *)v85;
    v87 = *(_QWORD *)(v85 + 8);
    v182.m128i_i16[0] = v86;
    v182.m128i_i64[1] = v87;
    if ( v86 )
    {
      v88 = ((unsigned __int16)(v86 - 17) < 0xD4u) + 205;
    }
    else
    {
      v148 = v84;
      v150 = v83;
      v114 = sub_30070B0((__int64)&v182);
      v84 = v148;
      v83 = v150;
      v88 = 205 - (!v114 - 1);
    }
    *((_QWORD *)&v127 + 1) = v136.m128i_i64[1];
    v136.m128i_i64[0] = v83;
    *(_QWORD *)&v127 = v83;
    *((_QWORD *)&v117 + 1) = v76;
    *(_QWORD *)&v117 = v74;
    v89 = sub_340F900(v149, v88, (unsigned int)&v180, v179.m128i_i32[0], v179.m128i_i32[2], v82, v117, v84, v127);
    v155 = v89;
    v19 = (__int64 *)v89;
    v136.m128i_i64[0] = v89;
    v156 = v90;
    v136.m128i_i64[1] = (unsigned int)v90 | v136.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v180 )
    {
      v150 = v89;
      sub_B91220((__int64)&v180, v180);
      return v150;
    }
  }
  return (unsigned __int64)v19;
}
