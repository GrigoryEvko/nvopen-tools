// Function: sub_2E404A0
// Address: 0x2e404a0
//
void __fastcall sub_2E404A0(__int64 a1)
{
  unsigned __int64 v1; // rdi
  unsigned __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // r13
  int v10; // r10d
  _QWORD *v11; // rdx
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  unsigned __int64 *v15; // rax
  int v16; // ecx
  __int64 v17; // rsi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  __int16 v23; // cx
  __int16 v24; // dx
  __int16 v25; // dx
  unsigned __int64 v26; // rax
  __int64 v27; // rbx
  int v28; // ecx
  __int64 v29; // rax
  __int64 v30; // r8
  int v31; // eax
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r13
  unsigned __int64 v34; // rbx
  __int64 v35; // rdi
  double v36; // xmm0_8
  __int64 v37; // r14
  _QWORD *v38; // rax
  __int64 v39; // r9
  _QWORD *v40; // r13
  _QWORD *v41; // rdx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // r12
  unsigned __int64 *v44; // rax
  _QWORD *v45; // r15
  _QWORD *i; // r13
  _BYTE *v47; // rsi
  __int64 v48; // rdi
  unsigned __int64 v49; // r13
  int v50; // r8d
  unsigned int v51; // r14d
  size_t v52; // rdx
  __int64 *v53; // r14
  __int64 v54; // rax
  unsigned __int64 j; // rax
  __int64 *v56; // rax
  unsigned int v57; // ecx
  _QWORD *v58; // r8
  __int64 v59; // r15
  int v60; // r9d
  __int64 v61; // rsi
  unsigned __int64 v62; // rdi
  unsigned __int64 *v63; // r14
  unsigned __int64 v64; // rbx
  unsigned __int64 v65; // rdi
  unsigned __int64 *k; // rbx
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 m; // rdi
  unsigned int v70; // ecx
  __int64 *v71; // r9
  __int64 v72; // r11
  __int64 v73; // r11
  __int64 v74; // rcx
  unsigned __int64 v75; // rdx
  int v76; // edx
  __int64 v77; // r9
  int v78; // esi
  unsigned int v79; // r11d
  __int64 *v80; // rdx
  __int64 v81; // r10
  __int64 v82; // rsi
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // rdx
  unsigned __int64 *v86; // rax
  unsigned __int64 v87; // rbx
  unsigned __int64 v88; // r15
  unsigned __int64 v89; // rax
  unsigned __int64 v90; // rdi
  unsigned __int16 v91; // cx
  unsigned __int64 v92; // rsi
  __int16 v93; // dx
  unsigned __int64 v94; // rax
  __int16 v95; // cx
  unsigned __int64 v96; // rdx
  unsigned __int16 v97; // dx
  unsigned __int64 v98; // rdi
  __int16 v99; // si
  __int16 v100; // dx
  unsigned __int64 v101; // rdi
  __int64 v102; // rax
  unsigned __int64 v103; // rax
  __int16 v104; // r9
  unsigned __int64 v105; // rdi
  unsigned __int16 v106; // dx
  unsigned __int64 v107; // rax
  __int64 *v108; // rax
  unsigned int v109; // ecx
  unsigned __int64 **v110; // rdx
  unsigned __int64 *v111; // r13
  unsigned __int64 *v112; // rbx
  unsigned __int64 v113; // rdx
  unsigned int v114; // eax
  char v115; // cl
  __int64 v116; // rax
  char *v117; // rsi
  __int64 v118; // rdi
  __int64 *v119; // rdi
  __int64 v120; // rdx
  int v121; // edx
  int v122; // ebx
  int v123; // r9d
  int v124; // r13d
  int v125; // r8d
  int v126; // r10d
  _QWORD *v127; // r9
  unsigned __int64 v128; // [rsp+8h] [rbp-228h]
  unsigned __int64 *v129; // [rsp+18h] [rbp-218h]
  unsigned __int64 v130; // [rsp+20h] [rbp-210h]
  __int16 v132; // [rsp+38h] [rbp-1F8h]
  unsigned __int64 v133; // [rsp+40h] [rbp-1F0h]
  unsigned __int16 v134; // [rsp+40h] [rbp-1F0h]
  __int64 v135; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v136; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v137; // [rsp+60h] [rbp-1D0h]
  __int64 v138; // [rsp+60h] [rbp-1D0h]
  __int64 v139; // [rsp+80h] [rbp-1B0h]
  __m128i v140; // [rsp+90h] [rbp-1A0h]
  unsigned __int16 v141; // [rsp+ACh] [rbp-184h] BYREF
  unsigned __int16 v142; // [rsp+AEh] [rbp-182h] BYREF
  __int64 v143; // [rsp+B0h] [rbp-180h] BYREF
  unsigned __int64 v144; // [rsp+B8h] [rbp-178h] BYREF
  __int128 v145; // [rsp+C0h] [rbp-170h] BYREF
  __int128 v146; // [rsp+D0h] [rbp-160h] BYREF
  unsigned __int64 v147; // [rsp+E0h] [rbp-150h] BYREF
  unsigned __int16 v148; // [rsp+E8h] [rbp-148h]
  unsigned __int64 v149; // [rsp+F0h] [rbp-140h] BYREF
  unsigned __int16 v150; // [rsp+F8h] [rbp-138h]
  unsigned __int64 v151; // [rsp+100h] [rbp-130h] BYREF
  unsigned __int64 v152; // [rsp+108h] [rbp-128h]
  __int64 v153; // [rsp+110h] [rbp-120h]
  unsigned __int64 *v154; // [rsp+120h] [rbp-110h] BYREF
  unsigned __int64 *v155; // [rsp+128h] [rbp-108h]
  __int64 v156; // [rsp+130h] [rbp-100h]
  __int64 v157; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v158; // [rsp+148h] [rbp-E8h]
  __int64 v159; // [rsp+150h] [rbp-E0h]
  unsigned int v160; // [rsp+158h] [rbp-D8h]
  void *v161; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v162; // [rsp+168h] [rbp-C8h]
  _BYTE v163[48]; // [rsp+170h] [rbp-C0h] BYREF
  int v164; // [rsp+1A0h] [rbp-90h]
  unsigned __int64 v165; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v166; // [rsp+1B8h] [rbp-78h]
  __int64 *v167; // [rsp+1C0h] [rbp-70h]
  __int64 *v168; // [rsp+1C8h] [rbp-68h]
  __int64 v169; // [rsp+1D0h] [rbp-60h]
  unsigned __int64 *v170; // [rsp+1D8h] [rbp-58h]
  __int64 *v171; // [rsp+1E0h] [rbp-50h]
  __int64 v172; // [rsp+1E8h] [rbp-48h]
  __int64 v173; // [rsp+1F0h] [rbp-40h]
  __int64 *v174; // [rsp+1F8h] [rbp-38h]

  v151 = 0;
  v152 = 0;
  v153 = 0;
  sub_2E3FB30(a1, (__int64)&v151);
  v1 = v152;
  v2 = v151;
  if ( v152 == v151 )
    goto LABEL_40;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v3 = (__int64)(v152 - v151) >> 3;
  v160 = 0;
  if ( v152 - v151 > 0x3FFFFFFFFFFFFFF8LL )
    goto LABEL_195;
  v4 = 16 * v3;
  if ( !v3 )
  {
    v7 = 0;
    v135 = 0;
    v145 = 0;
LABEL_9:
    v8 = v7;
    v9 = 0;
    while ( 1 )
    {
      v27 = *(_QWORD *)(v2 + 8 * v9);
      if ( v160 )
      {
        v10 = 1;
        v11 = 0;
        v12 = (v160 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v13 = (_QWORD *)(v158 + 16LL * v12);
        v14 = *v13;
        if ( v27 == *v13 )
        {
LABEL_11:
          v15 = v13 + 1;
          goto LABEL_12;
        }
        while ( v14 != -4096 )
        {
          if ( !v11 && v14 == -8192 )
            v11 = v13;
          v12 = (v160 - 1) & (v10 + v12);
          v13 = (_QWORD *)(v158 + 16LL * v12);
          v14 = *v13;
          if ( v27 == *v13 )
            goto LABEL_11;
          ++v10;
        }
        if ( !v11 )
          v11 = v13;
        ++v157;
        v28 = v159 + 1;
        if ( 4 * ((int)v159 + 1) < 3 * v160 )
        {
          if ( v160 - HIDWORD(v159) - v28 <= v160 >> 3 )
          {
            sub_2E3E470((__int64)&v157, v160);
            if ( !v160 )
            {
LABEL_201:
              LODWORD(v159) = v159 + 1;
              BUG();
            }
            v58 = 0;
            LODWORD(v59) = (v160 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v60 = 1;
            v28 = v159 + 1;
            v11 = (_QWORD *)(v158 + 16LL * (unsigned int)v59);
            v61 = *v11;
            if ( v27 != *v11 )
            {
              while ( v61 != -4096 )
              {
                if ( v61 == -8192 && !v58 )
                  v58 = v11;
                v59 = (v160 - 1) & ((_DWORD)v59 + v60);
                v11 = (_QWORD *)(v158 + 16 * v59);
                v61 = *v11;
                if ( v27 == *v11 )
                  goto LABEL_24;
                ++v60;
              }
              if ( v58 )
                v11 = v58;
            }
          }
          goto LABEL_24;
        }
      }
      else
      {
        ++v157;
      }
      sub_2E3E470((__int64)&v157, 2 * v160);
      if ( !v160 )
        goto LABEL_201;
      v28 = v159 + 1;
      LODWORD(v29) = (v160 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v11 = (_QWORD *)(v158 + 16LL * (unsigned int)v29);
      v30 = *v11;
      if ( v27 != *v11 )
      {
        v126 = 1;
        v127 = 0;
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v127 )
            v127 = v11;
          v29 = (v160 - 1) & ((_DWORD)v29 + v126);
          v11 = (_QWORD *)(v158 + 16 * v29);
          v30 = *v11;
          if ( v27 == *v11 )
            goto LABEL_24;
          ++v126;
        }
        if ( v127 )
          v11 = v127;
      }
LABEL_24:
      LODWORD(v159) = v28;
      if ( *v11 != -4096 )
        --HIDWORD(v159);
      *v11 = v27;
      v15 = v11 + 1;
      v11[1] = 0;
LABEL_12:
      *v15 = v9;
      v16 = *(_DWORD *)(a1 + 184);
      v17 = *(_QWORD *)(a1 + 168);
      if ( !v16 )
        goto LABEL_29;
      v18 = v16 - 1;
      v19 = v18 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v27 != *v20 )
      {
        v31 = 1;
        while ( v21 != -4096 )
        {
          v125 = v31 + 1;
          v19 = v18 & (v31 + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v27 == *v20 )
            goto LABEL_14;
          v31 = v125;
        }
LABEL_29:
        v22 = -1;
        goto LABEL_15;
      }
LABEL_14:
      v22 = *((_DWORD *)v20 + 2);
LABEL_15:
      LODWORD(v165) = v22;
      v139 = sub_FE8AC0(a1, (unsigned int *)&v165);
      *(_QWORD *)v8 = v139;
      v165 = v139;
      v161 = (void *)v145;
      v23 = WORD4(v145);
      *(_WORD *)(v8 + 8) = v24;
      LOWORD(v149) = v23;
      LOWORD(v154) = v24;
      v25 = sub_FDCA70((unsigned __int64 *)&v161, (unsigned __int16 *)&v149, &v165, (unsigned __int16 *)&v154);
      v26 = (unsigned __int64)v161 + v165;
      if ( __CFADD__(v161, v165) )
      {
        ++v25;
        v26 = (v26 >> 1) | 0x8000000000000000LL;
      }
      *(_QWORD *)&v145 = v26;
      WORD4(v145) = v25;
      if ( v25 > 0x3FFF )
      {
        *(_QWORD *)&v145 = -1;
        WORD4(v145) = 0x3FFF;
      }
      v2 = v151;
      ++v9;
      v8 += 16LL;
      if ( (__int64)(v152 - v151) >> 3 <= v9 )
        goto LABEL_43;
    }
  }
  v5 = sub_22077B0(16 * v3);
  v6 = v5 + v4;
  v7 = v5;
  v135 = v5 + v4;
  do
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = 0;
      *(_WORD *)(v5 + 8) = 0;
    }
    v5 += 16;
  }
  while ( v5 != v6 );
  v2 = v151;
  v145 = 0;
  if ( v151 != v152 )
    goto LABEL_9;
LABEL_43:
  if ( v7 != v135 )
  {
    v34 = v7;
    do
    {
      v35 = v34;
      v34 += 16LL;
      sub_FDE760(v35, (__int64)&v145);
    }
    while ( v135 != v34 );
  }
  v154 = 0;
  v155 = 0;
  v156 = 0;
  sub_2E3D060(a1, &v151, (__int64)&v157, (unsigned __int64 *)&v154);
  v36 = 1.0 / *(double *)&qword_4F8E288[8];
  if ( 1.0 / *(double *)&qword_4F8E288[8] >= 9.223372036854776e18 )
  {
    v161 = (void *)(unsigned int)(int)(v36 - 9.223372036854776e18);
    v161 = (void *)((unsigned __int64)v161 ^ 0x8000000000000000LL);
  }
  else
  {
    v161 = (void *)(unsigned int)(int)v36;
  }
  LOWORD(v162) = 0;
  v165 = 1;
  LOWORD(v166) = 0;
  v140 = _mm_loadu_si128((const __m128i *)sub_FDE760((__int64)&v165, (__int64)&v161));
  v128 = v135 - v7;
  v161 = (void *)v140.m128i_i64[0];
  v137 = (__int64)(v135 - v7) >> 4;
  LOWORD(v162) = v140.m128i_i16[4];
  v136 = v137 * LODWORD(qword_4F8E368[8]);
  if ( v128 > 0x5555555555555550LL )
LABEL_195:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v37 = 3 * v137;
  if ( v137 )
  {
    v38 = (_QWORD *)sub_22077B0(24 * v137);
    v40 = v38;
    v129 = &v38[v37];
    v41 = &v38[v37];
    do
    {
      if ( v38 )
      {
        *v38 = 0;
        v38[1] = 0;
        v38[2] = 0;
      }
      v38 += 3;
    }
    while ( v38 != v41 );
    v42 = 0;
    v165 = 0;
    v133 = v7;
    v43 = (unsigned __int64)v40;
    do
    {
      v44 = &v154[3 * v42];
      v45 = (_QWORD *)v44[1];
      for ( i = (_QWORD *)*v44; v45 != i; *(_QWORD *)(v48 + 8) = v47 + 8 )
      {
        while ( 1 )
        {
          v48 = v43 + 24LL * *i;
          v47 = *(_BYTE **)(v48 + 8);
          if ( v47 != *(_BYTE **)(v48 + 16) )
            break;
          i += 3;
          sub_9CA200(v48, v47, &v165);
          if ( v45 == i )
            goto LABEL_62;
        }
        if ( v47 )
        {
          *(_QWORD *)v47 = v165;
          v47 = *(_BYTE **)(v48 + 8);
        }
        i += 3;
      }
LABEL_62:
      v42 = v165 + 1;
      v165 = v42;
    }
    while ( v137 > v42 );
    v49 = v43;
    v7 = v133;
    v50 = v137;
    v51 = (unsigned int)(v137 + 63) >> 6;
    v161 = v163;
    v162 = 0x600000000LL;
    if ( v51 > 6 )
    {
      sub_C8D5F0((__int64)&v161, v163, v51, 8u, (unsigned int)v137, v39);
      memset(v161, 0, 8LL * v51);
      LODWORD(v162) = (unsigned int)(v137 + 63) >> 6;
      v50 = v137;
      goto LABEL_68;
    }
    if ( v51 )
    {
      v52 = 8LL * v51;
      if ( v52 )
      {
        memset(v163, 0, v52);
        v50 = v137;
      }
    }
  }
  else
  {
    v50 = 0;
    v49 = 0;
    v51 = 0;
    v161 = v163;
    HIDWORD(v162) = 6;
    v129 = 0;
  }
  LODWORD(v162) = v51;
LABEL_68:
  v164 = v50;
  v165 = 0;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v166 = 8;
  v165 = sub_22077B0(0x40u);
  v53 = (__int64 *)(v165 + ((4 * v166 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v54 = sub_22077B0(0x200u);
  v170 = (unsigned __int64 *)v53;
  *v53 = v54;
  v168 = (__int64 *)v54;
  v169 = v54 + 512;
  v174 = v53;
  v172 = v54;
  v173 = v54 + 512;
  v167 = (__int64 *)v54;
  v171 = (__int64 *)v54;
  v149 = 0;
  if ( v137 )
  {
    for ( j = 0; j < v137; v149 = j )
    {
      if ( (int)sub_D788E0(*(_QWORD *)(v7 + 16 * j), *(_WORD *)(v7 + 16 * j + 8), 0, 0) > 0 )
      {
        v56 = v171;
        if ( v171 == (__int64 *)(v173 - 8) )
        {
          sub_FE0450((__int64 *)&v165, &v149);
          v57 = v149;
        }
        else
        {
          v57 = v149;
          if ( v171 )
          {
            *v171 = v149;
            v56 = v171;
          }
          v171 = v56 + 1;
        }
        *((_QWORD *)v161 + (v57 >> 6)) |= 1LL << v57;
      }
      j = v149 + 1;
    }
    if ( v136 )
    {
      v138 = 1;
      v130 = v49;
      while ( 1 )
      {
        if ( v171 == v167 )
        {
LABEL_94:
          v49 = v130;
          goto LABEL_95;
        }
        v84 = *v167;
        v143 = *v167;
        if ( v167 == (__int64 *)(v169 - 8) )
        {
          j_j___libc_free_0((unsigned __int64)v168);
          LODWORD(v84) = v143;
          v120 = *++v170 + 512;
          v168 = (__int64 *)*v170;
          v169 = v120;
          v167 = v168;
        }
        else
        {
          ++v167;
        }
        *((_QWORD *)v161 + ((unsigned int)v84 >> 6)) &= ~(1LL << v84);
        v85 = v143;
        v148 = 0;
        v146 = 0;
        v147 = 1;
        v86 = &v154[3 * v143];
        v87 = v86[1];
        if ( *v86 != v87 )
        {
          v88 = *v86;
          while ( 1 )
          {
            if ( *(_QWORD *)v88 == v85 )
            {
              v95 = *(_WORD *)(v88 + 16);
              v96 = *(_QWORD *)(v88 + 8);
              v88 += 24LL;
              v147 = sub_FDCB20(v147, v148, v96, v95);
              v148 = v97;
              if ( v87 == v88 )
                goto LABEL_138;
            }
            else
            {
              v89 = v7 + 16LL * *(_QWORD *)v88;
              v90 = *(_QWORD *)v89;
              v91 = *(_WORD *)(v89 + 8);
              v149 = v90;
              v150 = v91;
              if ( v90 )
              {
                v92 = *(_QWORD *)(v88 + 8);
                if ( v92 )
                {
                  v104 = *(_WORD *)(v88 + 16);
                  if ( v90 > 0xFFFFFFFF || v92 > 0xFFFFFFFF )
                  {
                    v132 = *(_WORD *)(v88 + 16);
                    v134 = v91;
                    v107 = sub_F04140(v90, v92);
                    v104 = v132;
                    v91 = v134;
                    v105 = v107;
                  }
                  else
                  {
                    v105 = v92 * v90;
                    v106 = 0;
                  }
                  v149 = v105;
                  v150 = v106;
                  sub_D78C90((__int64)&v149, (__int16)(v91 + v104));
                  v91 = v150;
                  v90 = v149;
                }
                else
                {
                  v149 = 0;
                  v91 = *(_WORD *)(v88 + 16);
                  v90 = 0;
                  v150 = v91;
                }
              }
              v149 = v90;
              v142 = v91;
              v144 = v146;
              v141 = WORD4(v146);
              v93 = sub_FDCA70(&v144, &v141, &v149, &v142);
              v94 = v144 + v149;
              if ( __CFADD__(v144, v149) )
              {
                ++v93;
                v94 = (v94 >> 1) | 0x8000000000000000LL;
              }
              *(_QWORD *)&v146 = v94;
              WORD4(v146) = v93;
              if ( v93 > 0x3FFF )
              {
                *(_QWORD *)&v146 = -1;
                WORD4(v146) = 0x3FFF;
              }
              v88 += 24LL;
              if ( v87 == v88 )
              {
LABEL_138:
                v98 = v147;
                v99 = v148;
                goto LABEL_139;
              }
            }
            v85 = v143;
          }
        }
        v99 = 0;
        v98 = 1;
LABEL_139:
        if ( (unsigned int)sub_D788E0(v98, v99, 1u, 0) )
          sub_FDE760((__int64)&v146, (__int64)&v147);
        if ( (int)sub_D788E0(*(_QWORD *)(v7 + 16 * v143), *(_WORD *)(v7 + 16 * v143 + 8), v146, SWORD4(v146)) < 0 )
          v101 = sub_FDCB20(v146, WORD4(v146), *(_QWORD *)(v7 + 16 * v143), *(_WORD *)(v7 + 16 * v143 + 8));
        else
          v101 = sub_FDCB20(*(_QWORD *)(v7 + 16 * v143), *(_WORD *)(v7 + 16 * v143 + 8), v146, SWORD4(v146));
        if ( (int)sub_D788E0(v101, v100, v140.m128i_u64[0], v140.m128i_i16[4]) <= 0 )
          goto LABEL_144;
        v108 = v171;
        if ( v171 == (__int64 *)(v173 - 8) )
        {
          sub_FE0450((__int64 *)&v165, &v143);
          v109 = v143;
        }
        else
        {
          v109 = v143;
          if ( v171 )
          {
            *v171 = v143;
            v108 = v171;
          }
          v171 = v108 + 1;
        }
        *((_QWORD *)v161 + (v109 >> 6)) |= 1LL << v109;
        v102 = v143;
        v110 = (unsigned __int64 **)(v130 + 24 * v143);
        v111 = *v110;
        v112 = v110[1];
        if ( *v110 != v112 )
          break;
LABEL_145:
        v103 = v7 + 16 * v102;
        *(_QWORD *)v103 = v146;
        *(_WORD *)(v103 + 8) = WORD4(v146);
        if ( v138 == v136 )
          goto LABEL_94;
        ++v138;
      }
      do
      {
        v113 = *v111;
        v114 = *v111;
        v149 = v113;
        v115 = v113 & 0x3F;
        v116 = 8LL * (v114 >> 6);
        v117 = (char *)v161 + v116;
        v118 = *(_QWORD *)((char *)v161 + v116);
        if ( !_bittest64(&v118, v113) )
        {
          v119 = v171;
          if ( v171 == (__int64 *)(v173 - 8) )
          {
            sub_FE0450((__int64 *)&v165, &v149);
            v115 = v149 & 0x3F;
            v117 = (char *)v161 + 8 * ((unsigned int)v149 >> 6);
          }
          else
          {
            if ( v171 )
            {
              *v171 = v113;
              v119 = v171;
              v117 = (char *)v161 + v116;
            }
            v171 = v119 + 1;
          }
          *(_QWORD *)v117 |= 1LL << v115;
        }
        ++v111;
      }
      while ( v112 != v111 );
LABEL_144:
      v102 = v143;
      goto LABEL_145;
    }
  }
LABEL_95:
  v62 = v165;
  if ( v165 )
  {
    v63 = v170;
    v64 = (unsigned __int64)(v174 + 1);
    if ( v174 + 1 > (__int64 *)v170 )
    {
      do
      {
        v65 = *v63++;
        j_j___libc_free_0(v65);
      }
      while ( v64 > (unsigned __int64)v63 );
      v62 = v165;
    }
    j_j___libc_free_0(v62);
  }
  if ( v161 != v163 )
    _libc_free((unsigned __int64)v161);
  for ( k = (unsigned __int64 *)v49; v129 != k; k += 3 )
  {
    if ( *k )
      j_j___libc_free_0(*k);
  }
  if ( v49 )
    j_j___libc_free_0(v49);
  v67 = *(_QWORD *)(a1 + 128);
  v68 = *(_QWORD *)(v67 + 328);
  for ( m = v67 + 320; m != v68; v68 = *(_QWORD *)(v68 + 8) )
  {
    v76 = *(_DWORD *)(a1 + 184);
    v77 = *(_QWORD *)(a1 + 168);
    if ( v76 )
    {
      v78 = v76 - 1;
      v79 = (v76 - 1) & (((unsigned int)v68 >> 4) ^ ((unsigned int)v68 >> 9));
      v80 = (__int64 *)(v77 + 16LL * v79);
      v81 = *v80;
      if ( *v80 != v68 )
      {
        v121 = 1;
        while ( v81 != -4096 )
        {
          v122 = v121 + 1;
          v79 = v78 & (v121 + v79);
          v80 = (__int64 *)(v77 + 16LL * v79);
          v81 = *v80;
          if ( *v80 == v68 )
            goto LABEL_116;
          v121 = v122;
        }
        continue;
      }
LABEL_116:
      v82 = *((unsigned int *)v80 + 2);
      if ( (_DWORD)v82 != -1 )
      {
        if ( !v160 )
          goto LABEL_118;
        v70 = (v160 - 1) & (((unsigned int)v68 >> 4) ^ ((unsigned int)v68 >> 9));
        v71 = (__int64 *)(v158 + 16LL * v70);
        v72 = *v71;
        if ( *v71 != v68 )
        {
          v123 = 1;
          while ( v72 != -4096 )
          {
            v124 = v123 + 1;
            v70 = (v160 - 1) & (v123 + v70);
            v71 = (__int64 *)(v158 + 16LL * v70);
            v72 = *v71;
            if ( *v71 == v68 )
              goto LABEL_111;
            v123 = v124;
          }
LABEL_118:
          v73 = *(_QWORD *)(a1 + 8);
LABEL_119:
          v83 = v73 + 24 * v82;
          *(_QWORD *)v83 = 0;
          *(_WORD *)(v83 + 8) = 0;
          continue;
        }
LABEL_111:
        v73 = *(_QWORD *)(a1 + 8);
        v74 = v73 + 24 * v82;
        if ( v71 == (__int64 *)(v158 + 16LL * v160) )
          goto LABEL_119;
        v75 = v7 + 16 * v71[1];
        *(_QWORD *)v74 = *(_QWORD *)v75;
        *(_WORD *)(v74 + 8) = *(_WORD *)(v75 + 8);
      }
    }
  }
  v32 = v155;
  v33 = v154;
  if ( v155 != v154 )
  {
    do
    {
      if ( *v33 )
        j_j___libc_free_0(*v33);
      v33 += 3;
    }
    while ( v32 != v33 );
    v33 = v154;
  }
  if ( v33 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( v7 )
    j_j___libc_free_0(v7);
  sub_C7D6A0(v158, 16LL * v160, 8);
  v1 = v151;
LABEL_40:
  if ( v1 )
    j_j___libc_free_0(v1);
}
