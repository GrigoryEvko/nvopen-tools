// Function: sub_8C0950
// Address: 0x8c0950
//
__int64 *__fastcall sub_8C0950(__int64 a1, __int64 *a2)
{
  char v2; // al
  __int64 v3; // r14
  __m128i *v4; // r13
  __int64 v5; // rcx
  __int64 v6; // rbx
  __m128i *i; // r15
  __m128i *v8; // rdi
  __int64 *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // r9
  _QWORD *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 *v20; // r15
  __int64 v21; // r12
  char v22; // al
  int v23; // r14d
  __int64 v24; // r13
  __int64 v25; // rsi
  _BOOL4 v26; // ecx
  __int64 v27; // r8
  _BOOL4 *v28; // rax
  int v29; // r11d
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // r12
  _DWORD *v34; // rax
  __int64 v35; // r8
  __int64 *v36; // r9
  __int64 v37; // r15
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // r12
  __int64 j; // rcx
  __int64 v42; // rax
  unsigned __int8 v43; // di
  char v44; // al
  __int64 *v45; // r13
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 *v49; // r9
  _QWORD *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 *v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 *v58; // r9
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 *v62; // r9
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 *v66; // r9
  __int64 *v68; // r15
  __m128i *v69; // rax
  _QWORD *k; // rdx
  _QWORD *v71; // rax
  __int64 v72; // rdi
  __int64 *v73; // rax
  _DWORD *v74; // rdx
  __int64 v75; // r8
  __int64 *v76; // r9
  __int64 *v77; // rdi
  __int64 v78; // r12
  _QWORD *v79; // rax
  __int64 *v80; // rbx
  __int64 v81; // r15
  char v82; // dl
  __int64 v83; // rdx
  __int64 **v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 *v88; // r9
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 *v92; // r9
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r8
  __int64 *v96; // r9
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 *v100; // r9
  __m128i *v101; // rax
  __int64 v102; // r13
  __int64 *v103; // rax
  __int64 *v104; // rdx
  __int64 v105; // rbx
  _QWORD *v106; // r15
  _QWORD *v107; // r14
  __int64 v108; // rax
  int v109; // r15d
  _QWORD *v110; // rax
  _QWORD *v111; // rax
  __int64 *v112; // r15
  __m128i *v113; // rax
  __m128i *v114; // rbx
  __int64 v115; // rax
  int v116; // r15d
  __int16 v117; // r14
  _BYTE *v118; // rax
  _QWORD *v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 *v122; // rax
  __int64 v123; // [rsp-10h] [rbp-1E0h]
  __int64 v124; // [rsp+0h] [rbp-1D0h]
  __int64 v125; // [rsp+8h] [rbp-1C8h]
  __int64 v126; // [rsp+8h] [rbp-1C8h]
  __int64 v127; // [rsp+18h] [rbp-1B8h]
  __int64 v128; // [rsp+28h] [rbp-1A8h]
  _BOOL4 v129; // [rsp+28h] [rbp-1A8h]
  __int64 v130; // [rsp+30h] [rbp-1A0h]
  __int64 v131; // [rsp+30h] [rbp-1A0h]
  _QWORD *v132; // [rsp+30h] [rbp-1A0h]
  __int64 v133; // [rsp+38h] [rbp-198h]
  __int64 v134; // [rsp+40h] [rbp-190h]
  _QWORD *v135; // [rsp+48h] [rbp-188h]
  __int64 *v136; // [rsp+58h] [rbp-178h]
  _QWORD *v137; // [rsp+58h] [rbp-178h]
  __int64 v138; // [rsp+60h] [rbp-170h]
  __int64 v139; // [rsp+68h] [rbp-168h]
  _QWORD *v140; // [rsp+70h] [rbp-160h]
  __int64 *v141; // [rsp+78h] [rbp-158h]
  __int16 v142; // [rsp+78h] [rbp-158h]
  __int64 v143; // [rsp+78h] [rbp-158h]
  _BYTE *v144; // [rsp+78h] [rbp-158h]
  _QWORD *v145; // [rsp+80h] [rbp-150h]
  __int64 v146; // [rsp+88h] [rbp-148h]
  unsigned int v147; // [rsp+94h] [rbp-13Ch] BYREF
  __m128i *v148; // [rsp+98h] [rbp-138h] BYREF
  __int64 *v149; // [rsp+A0h] [rbp-130h] BYREF
  __m128i *v150; // [rsp+A8h] [rbp-128h] BYREF
  __m128i *v151; // [rsp+B0h] [rbp-120h] BYREF
  __m128i *v152; // [rsp+B8h] [rbp-118h] BYREF
  __m128i *v153; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v154; // [rsp+C8h] [rbp-108h]
  __int64 v155; // [rsp+D0h] [rbp-100h]
  __int64 v156; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v157; // [rsp+E8h] [rbp-E8h]
  __int64 v158; // [rsp+F0h] [rbp-E0h]
  __int64 v159; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v160; // [rsp+108h] [rbp-C8h]
  __int64 v161; // [rsp+110h] [rbp-C0h]
  __int64 v162; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v163; // [rsp+128h] [rbp-A8h]
  __int64 v164; // [rsp+130h] [rbp-A0h]
  __m128i v165; // [rsp+140h] [rbp-90h] BYREF
  _QWORD *v166; // [rsp+150h] [rbp-80h]
  __int64 *v167; // [rsp+160h] [rbp-70h]
  __int64 *v168; // [rsp+178h] [rbp-58h]
  __int64 *v169; // [rsp+180h] [rbp-50h]

  v2 = *(_BYTE *)(a1 + 80);
  v136 = a2;
  v148 = 0;
  v3 = a1;
  switch ( v2 )
  {
    case 4:
    case 5:
      v138 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v138 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v138 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v138 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v140 = 0;
  v135 = 0;
  v133 = *(_QWORD *)(*(_QWORD *)(v138 + 176) + 88LL);
  v4 = *(__m128i **)(v133 + 160);
  v5 = a2[11];
  v139 = v5;
  if ( *((_BYTE *)a2 + 80) == 20 )
  {
    v135 = (_QWORD *)a2[11];
    v140 = **(_QWORD ***)(v5 + 328);
    v139 = *(_QWORD *)(v5 + 176);
  }
  v6 = *(_QWORD *)(v139 + 152);
  v134 = v6;
  sub_865900(0);
  for ( i = *(__m128i **)(v6 + 160); i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  v8 = i;
  v9 = 0;
  if ( !(unsigned int)sub_8D3E20(i) )
    goto LABEL_51;
  v141 = (__int64 *)(v3 + 48);
  v148 = sub_8A3C00((__int64)v140, 0, 0, 0);
  if ( !v140 )
  {
    v149 = 0;
    v147 = 0;
    goto LABEL_100;
  }
  if ( !(unsigned int)sub_8B3500(v4, (__int64)i, (__int64 *)&v148, (__int64)v140, 0) )
  {
    sub_725130(v148->m128i_i64);
    v148 = sub_8A3C00((__int64)v140, 0, 0, 0);
  }
  v149 = 0;
  v147 = 0;
  v127 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(i->m128i_i64[0] + 96) + 72LL) + 88LL);
  v14 = **(_QWORD ***)(v138 + 32);
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v15 = sub_823970(0);
  v154 = 0;
  v153 = (__m128i *)v15;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v16 = sub_823970(0);
  v157 = 0;
  v156 = v16;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v17 = sub_823970(0);
  v160 = 0;
  v159 = v17;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v18 = sub_823970(0);
  v163 = 0;
  v162 = v18;
  sub_892150(&v165);
  v169 = &v156;
  v19 = sub_896D70(v3, (__int64)v14, 1);
  v20 = (__int64 *)v148;
  v21 = (__int64)v140;
  v151 = (__m128i *)v19;
  if ( !v148 )
  {
    do
    {
LABEL_30:
      v30 = v21;
      v21 = *(_QWORD *)v21;
    }
    while ( v21 );
    v31 = *(int *)(v30 + 60);
    v131 = v31;
    if ( v31 > v155 )
    {
      LODWORD(v152) = 1;
      sub_89F5F0((__int64 *)&v153, v31, &v152);
    }
    goto LABEL_33;
  }
  v130 = v3;
  do
  {
    v22 = *((_BYTE *)v20 + 8);
    v23 = *(_DWORD *)(v21 + 60);
    if ( v22 == 3 )
      goto LABEL_13;
    v24 = v155;
    v25 = v23 - 1;
    if ( v25 >= v155 )
    {
      if ( v25 != v155 )
      {
        LODWORD(v152) = 1;
        sub_89F5F0((__int64 *)&v153, v25, &v152);
        v24 = v155;
        v22 = *((_BYTE *)v20 + 8);
      }
      if ( v22 != 1 || (v26 = 0, (v20[3] & 1) == 0) )
        v26 = v20[4] == 0;
      v27 = (__int64)v153;
      if ( v154 == v24 )
      {
        if ( v24 <= 1 )
        {
          v124 = 2;
          v72 = 8;
        }
        else
        {
          v124 = v24 + (v24 >> 1) + 1;
          v72 = 4 * v124;
        }
        v125 = (__int64)v153;
        v129 = v26;
        v73 = (__int64 *)sub_823970(v72);
        v75 = v125;
        v76 = v73;
        if ( v24 > 0 )
        {
          v74 = (_DWORD *)v125;
          v77 = (__int64 *)((char *)v73 + 4 * v24);
          do
          {
            if ( v73 )
              *(_DWORD *)v73 = *v74;
            v73 = (__int64 *)((char *)v73 + 4);
            ++v74;
          }
          while ( v77 != v73 );
        }
        v126 = (__int64)v76;
        sub_823A00(v75, 4 * v24, (__int64)v74, v129, v75, v76);
        v26 = v129;
        v153 = (__m128i *)v126;
        v27 = v126;
        v154 = v124;
      }
      v28 = (_BOOL4 *)(v27 + 4 * v24);
      if ( v28 )
        *v28 = v26;
      v155 = v24 + 1;
      v22 = *((_BYTE *)v20 + 8);
    }
    if ( v22 != 1 )
    {
      if ( !v20[4] )
        goto LABEL_13;
LABEL_28:
      sub_8A4F30((__int64)v20, v21, 0, 0, v151, (__int64)v14, v141, 0x2000u, (int *)&v147, &v165);
      goto LABEL_13;
    }
    if ( (v20[3] & 1) != 0 || v20[4] )
      goto LABEL_28;
LABEL_13:
    if ( (*(_BYTE *)(v21 + 56) & 0x10) == 0 )
      v21 = *(_QWORD *)v21;
    v20 = (__int64 *)*v20;
  }
  while ( v20 );
  v29 = v23;
  v3 = v130;
  if ( v21 )
    goto LABEL_30;
  v131 = v29;
LABEL_33:
  v32 = v158;
  v33 = v158;
  v128 = v158;
  v34 = (_DWORD *)sub_823970(4 * v158);
  v37 = (__int64)v34;
  v38 = (__int64)&v34[v33];
  if ( v32 > 0 )
  {
    do
    {
      if ( v34 )
        *v34 = 0;
      ++v34;
    }
    while ( v34 != (_DWORD *)v38 );
  }
  v39 = v156;
  v40 = (__int64)v14;
  for ( j = 1; ; j = 1 )
  {
    while ( v40 )
    {
      v42 = *(_DWORD *)(v40 + 60) - 1;
      if ( v42 >= v158 )
        break;
      v38 = *(unsigned int *)(v39 + 4 * v42);
      if ( (_DWORD)v38 )
      {
        *(_DWORD *)(v37 + 4 * v42) = 1;
        v38 = v156;
        *(_DWORD *)(v156 + 4 * v42) = 0;
        if ( (*(_BYTE *)(v40 + 56) & 1) != 0 )
        {
          v43 = 0;
          v44 = *(_BYTE *)(*(_QWORD *)(v40 + 8) + 80LL);
          if ( v44 != 3 )
            v43 = (v44 != 2) + 1;
          v45 = sub_725090(v43);
          sub_8AEEA0(v3, (__int64)v45, v40, v14);
          sub_8A4F30((__int64)v45, v40, 0, 0, v151, (__int64)v14, v141, 0x2000u, (int *)&v147, &v165);
          sub_725130(v45);
          v39 = v156;
          j = 0;
        }
        else
        {
          v39 = v156;
        }
      }
      v40 = *(_QWORD *)v40;
    }
    if ( (_DWORD)j )
      break;
    v40 = (__int64)v14;
  }
  sub_823A00(v39, 4 * v157, v38, j, v35, v36);
  v156 = v37;
  v152 = 0;
  v158 = v128;
  v157 = v128;
  sub_89F6E0(&v159, v131, &v152);
  sub_892DC0((__int64)v14, &v149, &v150, (__int64)&v156, 0);
  sub_8973E0((__int64 **)&v151, (__int64)v149, &v156);
  sub_8C0720(v3, (__int64)v149, v14, v151, 0x2000, (int *)&v147, &v165);
  if ( v147 )
    goto LABEL_50;
  v68 = v149;
  if ( v149 )
  {
    while ( (v68[7] & 0x10) == 0 )
    {
      v68 = (__int64 *)*v68;
      if ( !v68 )
        goto LABEL_79;
    }
    v69 = v148;
    for ( k = v140; v69; v69 = (__m128i *)v69->m128i_i64[0] )
    {
      if ( (v69[1].m128i_i8[8] & 0x10) != 0 )
        *(_QWORD *)(v159 + 8LL * (unsigned int)(*((_DWORD *)k + 15) - 1)) = v68;
      if ( (k[7] & 0x10) == 0 )
        k = (_QWORD *)*k;
    }
    v71 = v14;
    if ( v14 )
    {
      while ( (v71[7] & 0x10) == 0 )
      {
        v71 = (_QWORD *)*v71;
        if ( !v71 )
          goto LABEL_79;
      }
      v152 = 0;
      v132 = v71;
      sub_89F6E0(&v162, *((unsigned int *)v71 + 15), &v152);
      *(_QWORD *)(v162 + 8LL * (unsigned int)(*((_DWORD *)v132 + 15) - 1)) = v68;
    }
  }
LABEL_79:
  sub_892150(&v165);
  v168 = &v162;
  v166 = v14;
  v167 = v149;
  v78 = **(_QWORD **)(v127 + 32);
  if ( v148 )
  {
    v79 = v14;
    v80 = (__int64 *)v148;
    v81 = (__int64)v79;
    do
    {
      v82 = *((_BYTE *)v80 + 8);
      if ( v82 != 3 && (v82 == 1 && (v80[3] & 1) != 0 || v80[4]) )
      {
        sub_8A4F30((__int64)v80, v78, 0, 0, v151, v81, v141, 0x2000u, (int *)&v147, &v165);
        v49 = (__int64 *)v147;
        if ( v147 )
          goto LABEL_50;
        if ( (*(_BYTE *)(v78 + 56) & 0x10) == 0 )
          v78 = *(_QWORD *)v78;
      }
      v80 = (__int64 *)*v80;
    }
    while ( v80 );
  }
  sub_892DC0((__int64)v140, &v149, &v152, (__int64)&v153, 0);
  sub_8973E0((__int64 **)&v148, (__int64)v152, &v153);
  sub_8C0720(v3, (__int64)v152, v140, v148, 0x2000, (int *)&v147, &v165);
  v50 = v140;
  if ( v147 )
    goto LABEL_50;
  do
  {
    v83 = (unsigned int)(*((_DWORD *)v50 + 15) - 1);
    if ( v153->m128i_i32[v83] )
    {
      if ( (v50[7] & 0x10) != 0 )
        *(_QWORD *)(v159 + 8 * v83) = v152;
      v152 = (__m128i *)v152->m128i_i64[0];
    }
    v50 = (_QWORD *)*v50;
  }
  while ( v50 );
  sub_725130(v151->m128i_i64);
  sub_892150(&v165);
  v167 = v149;
  v166 = v140;
  v168 = &v159;
  v84 = sub_8A2270(v134, v148, (__int64)v140, v141, 540672, (int *)&v147, &v165);
  v47 = v123;
  v134 = (__int64)v84;
  if ( v147 )
  {
LABEL_50:
    v9 = 0;
    sub_823A00(0, 0, v46, v47, v48, v49);
    sub_823A00(v162, 8 * v163, v51, v52, v53, v54);
    sub_823A00(v159, 8 * v160, v55, v56, v57, v58);
    sub_823A00(v156, 4 * v157, v59, v60, v61, v62);
    v8 = v153;
    a2 = (__int64 *)(4 * v154);
    sub_823A00((__int64)v153, 4 * v154, v63, v64, v65, v66);
    goto LABEL_51;
  }
  for ( i = (__m128i *)v84[20]; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  sub_823A00(0, 0, v46, v123, v48, v49);
  sub_823A00(v162, 8 * v163, v85, v86, v87, v88);
  sub_823A00(v159, 8 * v160, v89, v90, v91, v92);
  sub_823A00(v156, 4 * v157, v93, v94, v95, v96);
  sub_823A00((__int64)v153, 4 * v154, v97, v98, v99, v100);
LABEL_100:
  v101 = sub_725FD0();
  v9 = v149;
  v102 = (__int64)v101;
  if ( !v149 )
  {
    a2 = (__int64 *)i;
    v8 = *(__m128i **)(v138 + 104);
    if ( !sub_8B5250(v8->m128i_i64, i) )
      goto LABEL_51;
    v122 = sub_87EBB0(0xBu, *v136, v141);
    v122[11] = v102;
    v9 = v122;
    goto LABEL_119;
  }
  v103 = sub_8907A0(v3, v133, (__int64)v136, v135[52], v3);
  v104 = v149;
  v9 = v103;
  v105 = v103[11];
  v103[6] = *(_QWORD *)(v3 + 48);
  **(_QWORD **)(v105 + 32) = v104;
  v146 = sub_699CB0(*(_QWORD *)(v138 + 104), (__int64)i);
  v106 = *(_QWORD **)(v135[41] + 32LL);
  if ( !v106 )
    goto LABEL_115;
  v107 = sub_727300();
  *v107 = *v106;
  v107[3] = v106[3];
  v107[4] = v106[4];
  v108 = v106[2];
  if ( !v108 )
    goto LABEL_105;
  v8 = *(__m128i **)v108;
  a2 = (__int64 *)v148;
  v142 = *(_WORD *)(v108 + 12);
  v109 = *(_DWORD *)(v108 + 8);
  v110 = sub_743530(*(__m128i **)v108, v148, (__int64)v140, 540676, (int *)&v147, v165.m128i_i64);
  v10 = v147;
  v137 = v110;
  if ( v147 )
    goto LABEL_51;
  v111 = sub_7272D0();
  *((_DWORD *)v111 + 2) = v109;
  *v111 = v137;
  *((_WORD *)v111 + 6) = v142;
  v107[2] = v111;
LABEL_105:
  v112 = v149;
  v113 = 0;
  if ( v149 )
  {
    v143 = v105;
    do
    {
      v114 = v113;
      v113 = sub_8992B0((__int64)v112);
      if ( v114 )
        v114[7].m128i_i64[0] = (__int64)v113;
      else
        v107[1] = v113;
      v112 = (__int64 *)*v112;
    }
    while ( v112 );
    v105 = v143;
  }
  *(_QWORD *)(*(_QWORD *)(v105 + 328) + 32LL) = v107;
  *(_QWORD *)(*(_QWORD *)(v105 + 104) + 176LL) = v107;
LABEL_115:
  v115 = *(_QWORD *)(v135[22] + 216LL);
  if ( !v115 )
  {
    v119 = sub_7272D0();
    *v119 = v146;
LABEL_118:
    *(_QWORD *)(v102 + 216) = v119;
    v120 = *(_QWORD *)(v138 + 104);
    *(_BYTE *)(v102 + 195) |= 8u;
    *(_QWORD *)(v102 + 176) = v120;
    v121 = *(_QWORD *)(v105 + 104);
    *(_QWORD *)(v105 + 176) = v102;
    *(_QWORD *)(v121 + 192) = v102;
LABEL_119:
    *(_QWORD *)(v102 + 152) = v134;
    sub_725ED0(v102, 7);
    *(_BYTE *)(v102 + 193) = *(_BYTE *)(v139 + 193) & 0x10 | *(_BYTE *)(v102 + 193) & 0xEF;
    sub_7362F0(v102, -1);
    v8 = (__m128i *)v9;
    a2 = (__int64 *)(v138 + 216);
    sub_87F0B0((__int64)v9, (__int64 *)(v138 + 216));
    goto LABEL_51;
  }
  v8 = *(__m128i **)v115;
  a2 = (__int64 *)v148;
  v116 = *(_DWORD *)(v115 + 8);
  v117 = *(_WORD *)(v115 + 12);
  v145 = sub_743530(*(__m128i **)v115, v148, (__int64)v140, 540676, (int *)&v147, v165.m128i_i64);
  if ( !v147 )
  {
    v118 = sub_726700(1);
    v118[27] |= 2u;
    v144 = v118;
    *(_QWORD *)v118 = sub_72C390();
    v144[56] = 87;
    *((_QWORD *)v144 + 9) = v145;
    v145[2] = v146;
    v119 = sub_7272D0();
    *((_DWORD *)v119 + 2) = v116;
    *v119 = v144;
    *((_WORD *)v119 + 6) = v117;
    goto LABEL_118;
  }
LABEL_51:
  sub_864110((__int64)v8, (__int64)a2, v10, v11, v12, v13);
  return v9;
}
