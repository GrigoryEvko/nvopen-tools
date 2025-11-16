// Function: sub_310FBD0
// Address: 0x310fbd0
//
__int64 __fastcall sub_310FBD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 i; // r14
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _BYTE *v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r13
  unsigned int *v16; // rax
  int v17; // ecx
  unsigned int *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned int *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // r13
  unsigned int v29; // eax
  _BYTE *v30; // rsi
  int v31; // r12d
  __int64 v32; // r12
  __int64 v33; // rdx
  _QWORD *v34; // rdi
  __int64 v35; // r9
  __int64 v36; // r13
  char *v37; // r12
  __int64 v38; // rax
  char *v39; // r13
  char *v40; // rbx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __m128i *v43; // r13
  __m128i *p_dest; // rbx
  unsigned __int64 v45; // r9
  int v46; // edx
  __m128i *v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // r12
  __int64 v53; // rbx
  __int64 v54; // rax
  char v55; // r14
  _QWORD *v56; // rax
  __int64 v57; // r13
  unsigned int *v58; // r12
  unsigned int *v59; // rbx
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rax
  __int64 v63; // rbx
  unsigned __int64 *v64; // r12
  unsigned __int64 v65; // rdi
  unsigned int v66; // r12d
  __int64 v68; // r13
  __int64 v69; // rsi
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r12
  unsigned int *v73; // rax
  int v74; // ecx
  unsigned int *v75; // rdx
  __int64 v76; // r12
  __int64 v77; // r14
  __int64 v78; // rax
  char v79; // al
  __int16 v80; // cx
  _QWORD *v81; // rax
  __int64 v82; // rbx
  unsigned int *v83; // r14
  unsigned int *v84; // r12
  __int64 v85; // rdx
  unsigned int v86; // esi
  __int64 v87; // r9
  __m128i *v88; // r12
  _QWORD *v89; // rax
  __m128i *v90; // r12
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 v93; // rsi
  _QWORD *v94; // rdx
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rbx
  unsigned __int64 *v98; // r12
  unsigned __int64 v99; // rdi
  size_t v100; // rdx
  char *v101; // r8
  size_t v102; // r10
  unsigned __int64 v103; // rdx
  bool v104; // cf
  unsigned __int64 v105; // rax
  char *v106; // r11
  unsigned __int64 v107; // rdx
  __int64 v108; // rax
  char *v109; // r9
  char *v110; // r15
  char *v111; // r10
  char *v112; // rdx
  char *v113; // rax
  char *v114; // rdi
  char *v115; // rax
  char *v116; // rbx
  __m128i *v117; // rdx
  _QWORD *v118; // rax
  int v119; // r14d
  unsigned __int64 v120; // rsi
  unsigned __int64 v121; // rbx
  char *v122; // rbx
  int v123; // r12d
  int v124; // ebx
  unsigned __int64 v125; // rsi
  unsigned __int64 v126; // rbx
  char *v127; // [rsp+8h] [rbp-288h]
  char *v128; // [rsp+18h] [rbp-278h]
  size_t v129; // [rsp+20h] [rbp-270h]
  char *v130; // [rsp+20h] [rbp-270h]
  __int64 v131; // [rsp+28h] [rbp-268h]
  char *v132; // [rsp+28h] [rbp-268h]
  char *v133; // [rsp+28h] [rbp-268h]
  __int64 v134; // [rsp+30h] [rbp-260h]
  __int64 *v135; // [rsp+30h] [rbp-260h]
  unsigned int v136; // [rsp+40h] [rbp-250h]
  __int64 v137; // [rsp+40h] [rbp-250h]
  unsigned __int64 v138; // [rsp+40h] [rbp-250h]
  unsigned __int64 v139; // [rsp+40h] [rbp-250h]
  __int64 *v140; // [rsp+58h] [rbp-238h]
  __int64 v141; // [rsp+60h] [rbp-230h]
  __int16 v142; // [rsp+78h] [rbp-218h]
  __int64 *v144; // [rsp+B8h] [rbp-1D8h]
  _QWORD *v145; // [rsp+C0h] [rbp-1D0h] BYREF
  size_t n; // [rsp+C8h] [rbp-1C8h]
  _QWORD src[2]; // [rsp+D0h] [rbp-1C0h] BYREF
  __int16 v148; // [rsp+E0h] [rbp-1B0h]
  void *dest; // [rsp+F0h] [rbp-1A0h] BYREF
  size_t v150; // [rsp+F8h] [rbp-198h]
  __m128i v151; // [rsp+100h] [rbp-190h] BYREF
  void *v152; // [rsp+110h] [rbp-180h]
  void *v153; // [rsp+118h] [rbp-178h]
  unsigned __int64 v154; // [rsp+120h] [rbp-170h]
  __m128i *v155; // [rsp+130h] [rbp-160h] BYREF
  __int64 v156; // [rsp+138h] [rbp-158h]
  _BYTE v157[16]; // [rsp+140h] [rbp-150h] BYREF
  __int16 v158; // [rsp+150h] [rbp-140h]
  _BYTE *v159; // [rsp+180h] [rbp-110h] BYREF
  __int64 v160; // [rsp+188h] [rbp-108h]
  _BYTE v161[64]; // [rsp+190h] [rbp-100h] BYREF
  unsigned int *v162; // [rsp+1D0h] [rbp-C0h] BYREF
  __int64 v163; // [rsp+1D8h] [rbp-B8h]
  _BYTE v164[32]; // [rsp+1E0h] [rbp-B0h] BYREF
  __int64 v165; // [rsp+200h] [rbp-90h]
  __int64 v166; // [rsp+208h] [rbp-88h]
  __int64 v167; // [rsp+210h] [rbp-80h]
  __int64 v168; // [rsp+218h] [rbp-78h]
  void **v169; // [rsp+220h] [rbp-70h]
  _QWORD *v170; // [rsp+228h] [rbp-68h]
  __int64 v171; // [rsp+230h] [rbp-60h]
  int v172; // [rsp+238h] [rbp-58h]
  __int16 v173; // [rsp+23Ch] [rbp-54h]
  char v174; // [rsp+23Eh] [rbp-52h]
  __int64 v175; // [rsp+240h] [rbp-50h]
  __int64 v176; // [rsp+248h] [rbp-48h]
  void *v177; // [rsp+250h] [rbp-40h] BYREF
  _QWORD v178[7]; // [rsp+258h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v159 = v161;
  v160 = 0x800000000LL;
  if ( v2 == a2 + 72 )
    return 0;
  v3 = 0x8000000000041LL;
  do
  {
    if ( !v2 )
      BUG();
    v4 = *(_QWORD *)(v2 + 32);
    for ( i = v2 + 24; i != v4; v4 = *(_QWORD *)(v4 + 8) )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        if ( (unsigned __int8)(*(_BYTE *)(v4 - 24) - 34) <= 0x33u
          && _bittest64(&v3, (unsigned int)*(unsigned __int8 *)(v4 - 24) - 34)
          && sub_B491E0(v4 - 24)
          && !(unsigned __int8)sub_A747A0((_QWORD *)(v4 + 48), "guard_nocf", 0xAu)
          && !(unsigned __int8)sub_B49590(v4 - 24, "guard_nocf", 0xAu) )
        {
          break;
        }
        v4 = *(_QWORD *)(v4 + 8);
        if ( i == v4 )
          goto LABEL_16;
      }
      v8 = (unsigned int)v160;
      v9 = (unsigned int)v160 + 1LL;
      if ( v9 > HIDWORD(v160) )
      {
        sub_C8D5F0((__int64)&v159, v161, v9, 8u, v6, v7);
        v8 = (unsigned int)v160;
      }
      *(_QWORD *)&v159[8 * v8] = v4 - 24;
      LODWORD(v160) = v160 + 1;
    }
LABEL_16:
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( a2 + 72 != v2 );
  v10 = v159;
  if ( !(_DWORD)v160 )
  {
    v66 = 0;
    goto LABEL_74;
  }
  v144 = (__int64 *)v159;
  v140 = (__int64 *)&v159[8 * (unsigned int)v160];
  if ( *(_DWORD *)(a1 + 24) != 1 )
  {
    while ( 1 )
    {
      v11 = *v144;
      v168 = sub_BD5C60(*v144);
      v162 = (unsigned int *)v164;
      v169 = &v177;
      v165 = 0;
      v170 = v178;
      v166 = 0;
      v163 = 0x200000000LL;
      v177 = &unk_49DA100;
      v171 = 0;
      v172 = 0;
      v173 = 512;
      v174 = 7;
      v175 = 0;
      v176 = 0;
      LOWORD(v167) = 0;
      v178[0] = &unk_49DA0B0;
      v165 = *(_QWORD *)(v11 + 40);
      v166 = v11 + 24;
      v12 = *(_QWORD *)sub_B46C60(v11);
      v155 = (__m128i *)v12;
      if ( !v12 )
        break;
      sub_B96E90((__int64)&v155, v12, 1);
      v15 = (__int64)v155;
      if ( !v155 )
        break;
      v16 = v162;
      v17 = v163;
      v18 = &v162[4 * (unsigned int)v163];
      if ( v162 == v18 )
      {
LABEL_120:
        if ( (unsigned int)v163 >= (unsigned __int64)HIDWORD(v163) )
        {
          v120 = (unsigned int)v163 + 1LL;
          v121 = v134 & 0xFFFFFFFF00000000LL;
          v134 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v163) < v120 )
          {
            sub_C8D5F0((__int64)&v162, v164, v120, 0x10u, v13, v14);
            v18 = &v162[4 * (unsigned int)v163];
          }
          *(_QWORD *)v18 = v121;
          *((_QWORD *)v18 + 1) = v15;
          v15 = (__int64)v155;
          LODWORD(v163) = v163 + 1;
        }
        else
        {
          if ( v18 )
          {
            *v18 = 0;
            *((_QWORD *)v18 + 1) = v15;
            v17 = v163;
            v15 = (__int64)v155;
          }
          LODWORD(v163) = v17 + 1;
        }
LABEL_118:
        if ( !v15 )
          goto LABEL_27;
        goto LABEL_26;
      }
      while ( 1 )
      {
        v13 = *v16;
        if ( !(_DWORD)v13 )
          break;
        v16 += 4;
        if ( v18 == v16 )
          goto LABEL_120;
      }
      *((_QWORD *)v16 + 1) = v155;
LABEL_26:
      sub_B91220((__int64)&v155, v15);
LABEL_27:
      v141 = *(_QWORD *)(v11 - 32);
      v155 = (__m128i *)v157;
      v156 = 0x100000000LL;
      if ( *(char *)(v11 + 7) >= 0 )
        goto LABEL_57;
      v19 = sub_BD2BC0(v11);
      v21 = v19 + v20;
      if ( *(char *)(v11 + 7) < 0 )
        v21 -= sub_BD2BC0(v11);
      v22 = v21 >> 4;
      if ( !(_DWORD)v22 )
        goto LABEL_57;
      v23 = 0;
      v24 = 16LL * (unsigned int)v22;
      while ( 1 )
      {
        v25 = 0;
        if ( *(char *)(v11 + 7) < 0 )
          v25 = sub_BD2BC0(v11);
        v26 = (unsigned int *)(v23 + v25);
        v27 = *(_QWORD *)v26;
        if ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) == 1 )
          break;
        v23 += 16;
        if ( v24 == v23 )
          goto LABEL_57;
      }
      v28 = v26[2];
      v29 = v26[3];
      v30 = (_BYTE *)(v27 + 16);
      v31 = *(_DWORD *)(v11 + 4);
      v151.m128i_i8[0] = 0;
      v136 = v29;
      dest = &v151;
      v32 = v31 & 0x7FFFFFF;
      v150 = 0;
      v152 = 0;
      v153 = 0;
      v154 = 0;
      v33 = *((_QWORD *)v30 - 2);
      v145 = src;
      sub_310FB20((__int64 *)&v145, v30, (__int64)&v30[v33]);
      v34 = dest;
      if ( v145 == src )
      {
        v100 = n;
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)dest = src[0];
          else
            memcpy(dest, src, n);
          v100 = n;
          v34 = dest;
        }
        v150 = v100;
        *((_BYTE *)v34 + v100) = 0;
        v34 = v145;
      }
      else
      {
        if ( dest == &v151 )
        {
          dest = v145;
          v150 = n;
          v151.m128i_i64[0] = src[0];
        }
        else
        {
          v35 = v151.m128i_i64[0];
          dest = v145;
          v150 = n;
          v151.m128i_i64[0] = src[0];
          if ( v34 )
          {
            v145 = v34;
            src[0] = v35;
            goto LABEL_40;
          }
        }
        v145 = src;
        v34 = src;
      }
LABEL_40:
      n = 0;
      *(_BYTE *)v34 = 0;
      if ( v145 != src )
        j_j___libc_free_0((unsigned __int64)v145);
      v36 = 32 * v28;
      v37 = (char *)(v11 + v36 - 32 * v32);
      v38 = 32LL * v136 - v36;
      v39 = (char *)v153;
      v40 = &v37[v38];
      if ( v37 == &v37[v38] )
        goto LABEL_48;
      v41 = v38 >> 5;
      if ( v41 > (__int64)(v154 - (_QWORD)v153) >> 3 )
      {
        v101 = (char *)v152;
        v102 = (_BYTE *)v153 - (_BYTE *)v152;
        v103 = ((_BYTE *)v153 - (_BYTE *)v152) >> 3;
        if ( v41 > 0xFFFFFFFFFFFFFFFLL - v103 )
          sub_4262D8((__int64)"vector::_M_range_insert");
        if ( v41 < v103 )
          v41 = ((_BYTE *)v153 - (_BYTE *)v152) >> 3;
        v104 = __CFADD__(v103, v41);
        v105 = v103 + v41;
        v106 = (char *)v105;
        if ( !v104 )
        {
          if ( v105 )
          {
            if ( v105 > 0xFFFFFFFFFFFFFFFLL )
              v105 = 0xFFFFFFFFFFFFFFFLL;
            v107 = 8 * v105;
LABEL_144:
            v138 = v107;
            v108 = sub_22077B0(v107);
            v109 = (char *)v153;
            v110 = (char *)v108;
            v101 = (char *)v152;
            v139 = v138 + v108;
            v102 = v39 - (_BYTE *)v152;
            v106 = (char *)((_BYTE *)v153 - v39);
          }
          else
          {
            v139 = 0;
            v109 = (char *)v153;
            v110 = 0;
          }
          if ( v39 != v101 )
          {
            v127 = v109;
            v128 = v106;
            v129 = v102;
            v132 = v101;
            memmove(v110, v101, v102);
            v109 = v127;
            v106 = v128;
            v102 = v129;
            v101 = v132;
          }
          v111 = &v110[v102];
          v112 = v37;
          v113 = v111;
          do
          {
            if ( v113 )
              *(_QWORD *)v113 = *(_QWORD *)v37;
            v37 += 32;
            v113 += 8;
          }
          while ( v40 != v37 );
          v114 = &v111[8 * ((unsigned __int64)(v40 - v112 - 32) >> 5) + 8];
          if ( v39 != v109 )
          {
            v130 = v101;
            v133 = v106;
            v115 = (char *)memcpy(v114, v39, (size_t)v106);
            v101 = v130;
            v106 = v133;
            v114 = v115;
          }
          v116 = &v106[(_QWORD)v114];
          if ( v101 )
            j_j___libc_free_0((unsigned __int64)v101);
          v152 = v110;
          v153 = v116;
          v154 = v139;
          goto LABEL_48;
        }
        v107 = 0x7FFFFFFFFFFFFFF8LL;
        goto LABEL_144;
      }
      do
      {
        if ( v39 )
          *(_QWORD *)v39 = *(_QWORD *)v37;
        v37 += 32;
        v39 += 8;
      }
      while ( v40 != v37 );
      v153 = (char *)v153 + 8 * v41;
LABEL_48:
      v42 = (unsigned int)v156;
      v43 = v155;
      p_dest = (__m128i *)&dest;
      v45 = (unsigned int)v156 + 1LL;
      v46 = v156;
      if ( v45 > HIDWORD(v156) )
      {
        if ( v155 > (__m128i *)&dest || &dest >= (void **)v155 + 7 * (unsigned int)v156 )
        {
          v43 = (__m128i *)sub_C8D7D0(
                             (__int64)&v155,
                             (__int64)v157,
                             (unsigned int)v156 + 1LL,
                             0x38u,
                             (unsigned __int64 *)&v145,
                             v45);
          sub_B56820((__int64)&v155, v43);
          v124 = (int)v145;
          if ( v155 != (__m128i *)v157 )
            _libc_free((unsigned __int64)v155);
          v42 = (unsigned int)v156;
          HIDWORD(v156) = v124;
          v155 = v43;
          p_dest = (__m128i *)&dest;
          v46 = v156;
        }
        else
        {
          v122 = (char *)((char *)&dest - (char *)v155);
          v43 = (__m128i *)sub_C8D7D0(
                             (__int64)&v155,
                             (__int64)v157,
                             (unsigned int)v156 + 1LL,
                             0x38u,
                             (unsigned __int64 *)&v145,
                             v45);
          sub_B56820((__int64)&v155, v43);
          v123 = (int)v145;
          if ( v155 != (__m128i *)v157 )
            _libc_free((unsigned __int64)v155);
          v42 = (unsigned int)v156;
          v155 = v43;
          p_dest = (__m128i *)&v122[(_QWORD)v43];
          HIDWORD(v156) = v123;
          v46 = v156;
        }
      }
      v47 = (__m128i *)((char *)v43 + 56 * v42);
      if ( v47 )
      {
        v47->m128i_i64[0] = (__int64)v47[1].m128i_i64;
        if ( (__m128i *)p_dest->m128i_i64[0] == &p_dest[1] )
        {
          v47[1] = _mm_loadu_si128(p_dest + 1);
        }
        else
        {
          v47->m128i_i64[0] = p_dest->m128i_i64[0];
          v47[1].m128i_i64[0] = p_dest[1].m128i_i64[0];
        }
        v48 = p_dest->m128i_i64[1];
        p_dest->m128i_i64[0] = (__int64)p_dest[1].m128i_i64;
        p_dest->m128i_i64[1] = 0;
        v47->m128i_i64[1] = v48;
        v49 = p_dest[2].m128i_i64[0];
        p_dest[1].m128i_i8[0] = 0;
        v47[2].m128i_i64[0] = v49;
        v50 = p_dest[2].m128i_i64[1];
        p_dest[2].m128i_i64[0] = 0;
        v47[2].m128i_i64[1] = v50;
        v51 = p_dest[3].m128i_i64[0];
        p_dest[2].m128i_i64[1] = 0;
        v47[3].m128i_i64[0] = v51;
        p_dest[3].m128i_i64[0] = 0;
        v46 = v156;
      }
      LODWORD(v156) = v46 + 1;
      if ( v152 )
        j_j___libc_free_0((unsigned __int64)v152);
      if ( dest != &v151 )
        j_j___libc_free_0((unsigned __int64)dest);
LABEL_57:
      v148 = 257;
      v52 = *(_QWORD *)(a1 + 40);
      v53 = *(_QWORD *)(a1 + 48);
      v54 = sub_AA4E30(v165);
      v55 = sub_AE5020(v54, v52);
      LOWORD(v152) = 257;
      v56 = sub_BD2C40(80, 1u);
      v57 = (__int64)v56;
      if ( v56 )
        sub_B4D190((__int64)v56, v52, v53, (__int64)&dest, 0, v55, 0, 0);
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD **, __int64, __int64))(*v170 + 16LL))(
        v170,
        v57,
        &v145,
        v166,
        v167);
      v58 = v162;
      v59 = &v162[4 * (unsigned int)v163];
      if ( v162 != v59 )
      {
        do
        {
          v60 = *((_QWORD *)v58 + 1);
          v61 = *v58;
          v58 += 4;
          sub_B99FD0(v57, v61, v60);
        }
        while ( v59 != v58 );
      }
      LOWORD(v152) = 257;
      v145 = (_QWORD *)v141;
      v62 = sub_B33530(
              &v162,
              *(_QWORD *)(a1 + 32),
              v57,
              (int)&v145,
              1,
              (__int64)&dest,
              (__int64)v155,
              (unsigned int)v156,
              0);
      *(_WORD *)(v62 + 2) = *(_WORD *)(v62 + 2) & 0xF003 | 0x4C;
      v63 = (__int64)v155;
      v64 = (unsigned __int64 *)v155 + 7 * (unsigned int)v156;
      if ( v155 != (__m128i *)v64 )
      {
        do
        {
          v65 = *(v64 - 3);
          v64 -= 7;
          if ( v65 )
            j_j___libc_free_0(v65);
          if ( (unsigned __int64 *)*v64 != v64 + 2 )
            j_j___libc_free_0(*v64);
        }
        while ( (unsigned __int64 *)v63 != v64 );
        v64 = (unsigned __int64 *)v155;
      }
      if ( v64 != (unsigned __int64 *)v157 )
        _libc_free((unsigned __int64)v64);
      nullsub_61();
      v177 = &unk_49DA100;
      nullsub_63();
      if ( v162 != (unsigned int *)v164 )
        _libc_free((unsigned __int64)v162);
      if ( v140 == ++v144 )
        goto LABEL_73;
    }
    sub_93FB40((__int64)&v162, 0);
    v15 = (__int64)v155;
    goto LABEL_118;
  }
  do
  {
    v68 = *v144;
    v168 = sub_BD5C60(*v144);
    v162 = (unsigned int *)v164;
    v169 = &v177;
    v165 = 0;
    v170 = v178;
    v166 = 0;
    v177 = &unk_49DA100;
    v163 = 0x200000000LL;
    v171 = 0;
    v172 = 0;
    v173 = 512;
    v174 = 7;
    v175 = 0;
    v176 = 0;
    LOWORD(v167) = 0;
    v178[0] = &unk_49DA0B0;
    v165 = *(_QWORD *)(v68 + 40);
    v166 = v68 + 24;
    v69 = *(_QWORD *)sub_B46C60(v68);
    v155 = (__m128i *)v69;
    if ( v69 && (sub_B96E90((__int64)&v155, v69, 1), (v72 = (__int64)v155) != 0) )
    {
      v73 = v162;
      v74 = v163;
      v75 = &v162[4 * (unsigned int)v163];
      if ( v162 != v75 )
      {
        while ( *v73 )
        {
          v73 += 4;
          if ( v75 == v73 )
            goto LABEL_127;
        }
        *((_QWORD *)v73 + 1) = v155;
LABEL_84:
        sub_B91220((__int64)&v155, v72);
        goto LABEL_85;
      }
LABEL_127:
      if ( (unsigned int)v163 >= (unsigned __int64)HIDWORD(v163) )
      {
        v125 = (unsigned int)v163 + 1LL;
        v126 = v131 & 0xFFFFFFFF00000000LL;
        v131 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v163) < v125 )
        {
          sub_C8D5F0((__int64)&v162, v164, v125, 0x10u, v70, v71);
          v75 = &v162[4 * (unsigned int)v163];
        }
        *(_QWORD *)v75 = v126;
        *((_QWORD *)v75 + 1) = v72;
        v72 = (__int64)v155;
        LODWORD(v163) = v163 + 1;
      }
      else
      {
        if ( v75 )
        {
          *v75 = 0;
          *((_QWORD *)v75 + 1) = v72;
          v74 = v163;
          v72 = (__int64)v155;
        }
        LODWORD(v163) = v74 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v162, 0);
      v72 = (__int64)v155;
    }
    if ( v72 )
      goto LABEL_84;
LABEL_85:
    v76 = *(_QWORD *)(*(_QWORD *)(v68 - 32) + 8LL);
    v137 = *(_QWORD *)(v68 - 32);
    LOWORD(v152) = 257;
    v77 = *(_QWORD *)(a1 + 48);
    v78 = sub_AA4E30(v165);
    v79 = sub_AE5020(v78, v76);
    HIBYTE(v80) = HIBYTE(v142);
    LOBYTE(v80) = v79;
    v158 = 257;
    v142 = v80;
    v81 = sub_BD2C40(80, 1u);
    v82 = (__int64)v81;
    if ( v81 )
      sub_B4D190((__int64)v81, v76, v77, (__int64)&v155, 0, v142, 0, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, void **, __int64, __int64))(*v170 + 16LL))(v170, v82, &dest, v166, v167);
    v83 = v162;
    v84 = &v162[4 * (unsigned int)v163];
    if ( v162 != v84 )
    {
      do
      {
        v85 = *((_QWORD *)v83 + 1);
        v86 = *v83;
        v83 += 4;
        sub_B99FD0(v82, v86, v85);
      }
      while ( v84 != v83 );
    }
    v155 = (__m128i *)v157;
    v156 = 0x100000000LL;
    sub_B56970(v68, (__int64)&v155);
    if ( (unsigned int)v156 >= HIDWORD(v156) )
    {
      v90 = (__m128i *)sub_C8D7D0((__int64)&v155, (__int64)v157, 0, 0x38u, (unsigned __int64 *)&v145, v87);
      dest = &v151;
      sub_310FB20((__int64 *)&dest, "cfguardtarget", (__int64)"");
      v117 = (__m128i *)((char *)v90 + 56 * (unsigned int)v156);
      if ( v117 )
      {
        v117->m128i_i64[0] = (__int64)v117[1].m128i_i64;
        if ( dest == &v151 )
        {
          v117[1] = _mm_load_si128(&v151);
        }
        else
        {
          v117->m128i_i64[0] = (__int64)dest;
          v117[1].m128i_i64[0] = v151.m128i_i64[0];
        }
        v135 = (__int64 *)v117;
        v117->m128i_i64[1] = v150;
        v150 = 0;
        dest = &v151;
        v151.m128i_i8[0] = 0;
        v117[2].m128i_i64[0] = 0;
        v117[2].m128i_i64[1] = 0;
        v117[3].m128i_i64[0] = 0;
        v118 = (_QWORD *)sub_22077B0(8u);
        v135[4] = (__int64)v118;
        v135[6] = (__int64)(v118 + 1);
        *v118 = v137;
        v135[5] = (__int64)(v118 + 1);
      }
      if ( dest != &v151 )
        j_j___libc_free_0((unsigned __int64)dest);
      sub_B56820((__int64)&v155, v90);
      v119 = (int)v145;
      if ( v155 != (__m128i *)v157 )
        _libc_free((unsigned __int64)v155);
      v155 = v90;
      HIDWORD(v156) = v119;
      v91 = (unsigned int)(v156 + 1);
      LODWORD(v156) = v156 + 1;
    }
    else
    {
      dest = &v151;
      sub_310FB20((__int64 *)&dest, "cfguardtarget", (__int64)"");
      v88 = (__m128i *)((char *)v155 + 56 * (unsigned int)v156);
      if ( v88 )
      {
        v88->m128i_i64[0] = (__int64)v88[1].m128i_i64;
        if ( dest == &v151 )
        {
          v88[1] = _mm_load_si128(&v151);
        }
        else
        {
          v88->m128i_i64[0] = (__int64)dest;
          v88[1].m128i_i64[0] = v151.m128i_i64[0];
        }
        v88->m128i_i64[1] = v150;
        v150 = 0;
        dest = &v151;
        v151.m128i_i8[0] = 0;
        v88[2].m128i_i64[0] = 0;
        v88[2].m128i_i64[1] = 0;
        v88[3].m128i_i64[0] = 0;
        v89 = (_QWORD *)sub_22077B0(8u);
        v88[2].m128i_i64[0] = (__int64)v89;
        v88[3].m128i_i64[0] = (__int64)(v89 + 1);
        *v89 = v137;
        v88[2].m128i_i64[1] = (__int64)(v89 + 1);
      }
      if ( dest != &v151 )
        j_j___libc_free_0((unsigned __int64)dest);
      v90 = v155;
      v91 = (unsigned int)(v156 + 1);
      LODWORD(v156) = v156 + 1;
    }
    v92 = sub_B4BA60((unsigned __int8 *)v68, (__int64)v90, v91, v68 + 24, 0);
    v93 = v92;
    if ( *(_QWORD *)(v92 - 32) )
    {
      v94 = *(_QWORD **)(v92 - 16);
      v95 = *(_QWORD *)(v92 - 24);
      *v94 = v95;
      if ( v95 )
        *(_QWORD *)(v95 + 16) = *(_QWORD *)(v93 - 16);
    }
    *(_QWORD *)(v93 - 32) = v82;
    if ( v82 )
    {
      v96 = *(_QWORD *)(v82 + 16);
      *(_QWORD *)(v93 - 24) = v96;
      if ( v96 )
        *(_QWORD *)(v96 + 16) = v93 - 24;
      *(_QWORD *)(v93 - 16) = v82 + 16;
      *(_QWORD *)(v82 + 16) = v93 - 32;
    }
    sub_BD84D0(v68, v93);
    sub_B43D60((_QWORD *)v68);
    v97 = (__int64)v155;
    v98 = (unsigned __int64 *)v155 + 7 * (unsigned int)v156;
    if ( v155 != (__m128i *)v98 )
    {
      do
      {
        v99 = *(v98 - 3);
        v98 -= 7;
        if ( v99 )
          j_j___libc_free_0(v99);
        if ( (unsigned __int64 *)*v98 != v98 + 2 )
          j_j___libc_free_0(*v98);
      }
      while ( (unsigned __int64 *)v97 != v98 );
      v98 = (unsigned __int64 *)v155;
    }
    if ( v98 != (unsigned __int64 *)v157 )
      _libc_free((unsigned __int64)v98);
    nullsub_61();
    v177 = &unk_49DA100;
    nullsub_63();
    if ( v162 != (unsigned int *)v164 )
      _libc_free((unsigned __int64)v162);
    ++v144;
  }
  while ( v140 != v144 );
LABEL_73:
  v10 = v159;
  v66 = 1;
LABEL_74:
  if ( v10 != v161 )
    _libc_free((unsigned __int64)v10);
  return v66;
}
