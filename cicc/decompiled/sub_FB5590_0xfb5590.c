// Function: sub_FB5590
// Address: 0xfb5590
//
__int64 __fastcall sub_FB5590(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // r14
  unsigned __int8 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  unsigned __int64 v18; // r14
  _QWORD *v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  char v23; // al
  char *v24; // rax
  char *v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rsi
  char *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 *v36; // r9
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // r13
  unsigned __int8 v46; // di
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 v55; // r13
  unsigned __int64 v56; // rdx
  __int64 v57; // r13
  _QWORD *v58; // rdi
  __int64 v59; // rdx
  __int64 *v60; // rbx
  __int64 *v61; // r13
  __int64 *v62; // rdx
  __int64 v63; // r8
  unsigned __int8 *v64; // rax
  __int64 *v65; // r12
  __int64 v66; // r13
  __int64 v67; // r15
  _QWORD *v68; // rdi
  __int64 v69; // r9
  __int64 v70; // rax
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r12
  __int64 v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // rcx
  _QWORD *v79; // r15
  __int64 v80; // r12
  unsigned __int8 v81; // al
  _QWORD *v82; // rbx
  __int64 v83; // r13
  __int64 v84; // rbx
  _BYTE *v85; // r15
  __int64 v86; // rdx
  unsigned int v87; // esi
  int v88; // eax
  __int64 v89; // rsi
  __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // r15
  __int64 v95; // rdx
  __int64 v96; // rdx
  __int64 v97; // r13
  __int64 v98; // rcx
  __int64 v99; // rsi
  unsigned __int64 v100; // rdx
  const char **v101; // rdi
  __int64 v102; // rdx
  __int64 v103; // rax
  _QWORD *v104; // r15
  __int64 v105; // rcx
  const char **v106; // r13
  _QWORD *v107; // rax
  const char **v108; // r12
  _QWORD *v109; // r15
  const char *v110; // rdi
  __int64 v111; // rax
  __int64 v112; // rdx
  unsigned __int64 v113; // r12
  __int64 v114; // r14
  unsigned __int64 v115; // rax
  _QWORD *v116; // [rsp+0h] [rbp-250h]
  __int64 v117; // [rsp+10h] [rbp-240h]
  __int64 v118; // [rsp+20h] [rbp-230h]
  __int64 v119; // [rsp+20h] [rbp-230h]
  __int64 v120; // [rsp+20h] [rbp-230h]
  __int64 v121; // [rsp+28h] [rbp-228h]
  __int64 *v122; // [rsp+38h] [rbp-218h]
  __int64 v123; // [rsp+58h] [rbp-1F8h]
  unsigned __int8 v125; // [rsp+6Fh] [rbp-1E1h]
  __int64 v126; // [rsp+70h] [rbp-1E0h]
  __int64 *v127; // [rsp+70h] [rbp-1E0h]
  unsigned __int8 i; // [rsp+78h] [rbp-1D8h]
  __int64 v129; // [rsp+78h] [rbp-1D8h]
  unsigned __int8 v130; // [rsp+78h] [rbp-1D8h]
  __int64 v131; // [rsp+78h] [rbp-1D8h]
  __int64 v132; // [rsp+78h] [rbp-1D8h]
  __int64 v133; // [rsp+78h] [rbp-1D8h]
  __int64 v134; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v135; // [rsp+88h] [rbp-1C8h]
  __int64 v136; // [rsp+90h] [rbp-1C0h]
  __m128i v137[2]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int16 v138; // [rsp+C0h] [rbp-190h]
  __m128i v139; // [rsp+D0h] [rbp-180h] BYREF
  char v140[8]; // [rsp+E0h] [rbp-170h] BYREF
  char v141; // [rsp+E8h] [rbp-168h] BYREF
  __int16 v142; // [rsp+F0h] [rbp-160h]
  char v143; // [rsp+108h] [rbp-148h]
  char v144; // [rsp+110h] [rbp-140h]
  __int64 v145; // [rsp+120h] [rbp-130h] BYREF
  __int64 v146; // [rsp+128h] [rbp-128h]
  __int64 v147; // [rsp+130h] [rbp-120h]
  __int64 v148; // [rsp+138h] [rbp-118h]
  __int64 *v149; // [rsp+140h] [rbp-110h] BYREF
  __int64 v150; // [rsp+148h] [rbp-108h]
  _BYTE v151[64]; // [rsp+150h] [rbp-100h] BYREF
  _BYTE *v152; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v153; // [rsp+198h] [rbp-B8h]
  _BYTE v154[32]; // [rsp+1A0h] [rbp-B0h] BYREF
  __int64 v155; // [rsp+1C0h] [rbp-90h]
  __int64 v156; // [rsp+1C8h] [rbp-88h]
  __int64 v157; // [rsp+1D0h] [rbp-80h]
  __int64 v158; // [rsp+1D8h] [rbp-78h]
  void **v159; // [rsp+1E0h] [rbp-70h]
  _QWORD *v160; // [rsp+1E8h] [rbp-68h]
  __int64 v161; // [rsp+1F0h] [rbp-60h]
  int v162; // [rsp+1F8h] [rbp-58h]
  __int16 v163; // [rsp+1FCh] [rbp-54h]
  char v164; // [rsp+1FEh] [rbp-52h]
  __int64 v165; // [rsp+200h] [rbp-50h]
  __int64 v166; // [rsp+208h] [rbp-48h]
  void *v167; // [rsp+210h] [rbp-40h] BYREF
  _QWORD v168[7]; // [rsp+218h] [rbp-38h] BYREF

  v125 = *(_BYTE *)(*(_QWORD *)(a1 + 48) + 12LL);
  if ( v125 )
  {
    v4 = *(_QWORD *)(a2 + 40);
    v5 = a2;
    v6 = a2 + 24;
    sub_AA6320(v4);
    sub_B44570(a2);
    if ( *(_QWORD *)(v4 + 56) == a2 + 24 )
    {
      v8 = 0;
LABEL_14:
      i = v8;
      if ( v5 != v6 - 24 )
        return i;
    }
    else
    {
      for ( i = 0; ; i = v8 )
      {
        v7 = (*(_QWORD *)(v5 + 24) & 0xFFFFFFFFFFFFFFF8LL) - 24;
        if ( (*(_QWORD *)(v5 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          v7 = 0;
        v8 = sub_98CD80((char *)v7);
        if ( !v8 )
          break;
        sub_B44570(v7);
        a2 = sub_ACADE0(*(__int64 ***)(v7 + 8));
        sub_BD84D0(v7, a2);
        sub_B43D60((_QWORD *)v7);
        if ( *(_QWORD *)(v4 + 56) == v6 )
          goto LABEL_14;
      }
      v9 = *(_QWORD *)(v4 + 56);
      if ( !v9 || v5 != v9 - 24 )
        return i;
    }
    v136 = 0;
    v10 = *(_QWORD *)(v4 + 16);
    v134 = 0;
    v152 = (_BYTE *)v10;
    v135 = 0;
    sub_D4B000((__int64 *)&v152);
    v145 = 0;
    v149 = (__int64 *)v151;
    v146 = 0;
    v147 = 0;
    v148 = 0;
    v150 = 0x800000000LL;
    if ( !v152 )
      goto LABEL_50;
    v126 = v4;
    v11 = *((_QWORD *)v152 + 3);
    v12 = 0;
    v13 = (__int64)v152;
LABEL_19:
    v14 = *(_QWORD *)(v11 + 40);
    v139.m128i_i64[0] = v14;
    if ( v12 )
    {
      a2 = (__int64)&v145;
      sub_D6CB10((__int64)&v152, (__int64)&v145, v139.m128i_i64);
      if ( v154[16] )
      {
        v54 = (unsigned int)v150;
        v55 = v139.m128i_i64[0];
        v56 = (unsigned int)v150 + 1LL;
        if ( v56 > HIDWORD(v150) )
        {
          a2 = (__int64)v151;
          sub_C8D5F0((__int64)&v149, v151, v56, 8u, v52, v53);
          v54 = (unsigned int)v150;
        }
        v149[v54] = v55;
        LODWORD(v150) = v150 + 1;
      }
    }
    else
    {
      a2 = (__int64)&v149[(unsigned int)v150];
      if ( (_QWORD *)a2 == sub_F8ED40(v149, a2, v139.m128i_i64) )
      {
        if ( v16 + 1 > (unsigned __int64)HIDWORD(v150) )
        {
          sub_C8D5F0((__int64)&v149, v151, v16 + 1, 8u, v15, v16);
          a2 = (__int64)&v149[(unsigned int)v150];
        }
        *(_QWORD *)a2 = v14;
        v59 = (unsigned int)(v150 + 1);
        LODWORD(v150) = v59;
        if ( (unsigned int)v59 > 8 )
        {
          v123 = v13;
          v60 = v149;
          v61 = &v149[v59];
          do
          {
            v62 = v60;
            a2 = (__int64)&v145;
            ++v60;
            sub_D6CB10((__int64)&v152, (__int64)&v145, v62);
          }
          while ( v61 != v60 );
          v13 = v123;
        }
      }
    }
    while ( 1 )
    {
      v13 = *(_QWORD *)(v13 + 8);
      if ( !v13 )
        break;
      v11 = *(_QWORD *)(v13 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v11 - 30) <= 0xAu )
      {
        v12 = v147;
        goto LABEL_19;
      }
    }
    v4 = v126;
    v122 = &v149[(unsigned int)v150];
    if ( v149 == v122 )
    {
LABEL_50:
      v38 = *(_QWORD *)(a1 + 8);
      if ( v38 )
      {
        a2 = v134;
        sub_FFB3D0(v38, v134, (v135 - v134) >> 4);
      }
      v39 = *(_QWORD *)(v4 + 16);
      if ( v39 )
      {
        while ( (unsigned __int8)(**(_BYTE **)(v39 + 24) - 30) > 0xAu )
        {
          v39 = *(_QWORD *)(v39 + 8);
          if ( !v39 )
            goto LABEL_59;
        }
      }
      else
      {
LABEL_59:
        v40 = *(_QWORD *)(*(_QWORD *)(v4 + 72) + 80LL);
        if ( !v40 || v4 != v40 - 24 )
        {
          a2 = *(_QWORD *)(a1 + 8);
          sub_F34560(v4, a2, 0);
          i = v125;
        }
      }
      if ( v149 != (__int64 *)v151 )
        _libc_free(v149, a2);
      sub_C7D6A0(v146, 8LL * (unsigned int)v148, 8);
      if ( v134 )
        j_j___libc_free_0(v134, v136 - v134);
      return i;
    }
    v127 = v149;
    v121 = v4 | 4;
    while ( 1 )
    {
      v17 = *v127;
      v18 = *(_QWORD *)(*v127 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v18 == *v127 + 48 )
        goto LABEL_187;
      if ( !v18 )
        BUG();
      v19 = (_QWORD *)(v18 - 24);
      if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA )
      {
LABEL_187:
        v2 = sub_BD5C60(0);
        v164 = 7;
        v158 = v2;
        v159 = &v167;
        v160 = v168;
        v152 = v154;
        v153 = 0x200000000LL;
        v167 = &unk_49DA100;
        v161 = 0;
        v162 = 0;
        v168[0] = &unk_49DA0B0;
        v163 = 512;
        v165 = 0;
        v166 = 0;
        v155 = 0;
        v156 = 0;
        LOWORD(v157) = 0;
        sub_D5F1F0((__int64)&v152, 0);
        BUG();
      }
      v158 = sub_BD5C60(v18 - 24);
      v159 = &v167;
      v160 = v168;
      v152 = v154;
      v167 = &unk_49DA100;
      v153 = 0x200000000LL;
      LOWORD(v157) = 0;
      a2 = v18 - 24;
      v168[0] = &unk_49DA0B0;
      v161 = 0;
      v162 = 0;
      v163 = 512;
      v164 = 7;
      v165 = 0;
      v166 = 0;
      v155 = 0;
      v156 = 0;
      sub_D5F1F0((__int64)&v152, v18 - 24);
      v23 = *(_BYTE *)(v18 - 24);
      if ( v23 == 31 )
        break;
      switch ( v23 )
      {
        case ' ':
          v139.m128i_i64[0] = v18 - 24;
          v41 = 0;
          v143 = 0;
          v144 = 0;
          sub_B540B0(&v139);
          v42 = v139.m128i_i64[0];
          v43 = (*(_DWORD *)(v139.m128i_i64[0] + 4) & 0x7FFFFFFu) >> 1;
          v44 = v43 - 1;
          if ( v43 != 1 )
          {
            v118 = v17;
            v45 = v139.m128i_i64[0];
            v46 = i;
            do
            {
              while ( 1 )
              {
                v47 = 32;
                if ( (_DWORD)v41 != -2 )
                  v47 = 32LL * (unsigned int)(2 * v41 + 3);
                a2 = *(_QWORD *)(v45 - 8);
                v48 = *(_QWORD *)(a2 + v47);
                if ( v4 == v48 )
                {
                  if ( v48 )
                    break;
                }
                if ( ++v41 == v44 )
                  goto LABEL_71;
              }
              sub_AA5980(v4, *(_QWORD *)(v42 + 40), 0);
              a2 = v45;
              v49 = sub_B541A0((__int64)&v139, v45, v41);
              v42 = v139.m128i_i64[0];
              v46 = v125;
              v45 = v49;
              v41 = v50;
              v44 = ((*(_DWORD *)(v139.m128i_i64[0] + 4) & 0x7FFFFFFu) >> 1) - 1;
            }
            while ( v41 != v44 );
LABEL_71:
            i = v46;
            v17 = v118;
          }
          if ( *(_QWORD *)(a1 + 8) )
          {
            v51 = *(_QWORD *)(*(_QWORD *)(v18 - 32) + 32LL);
            if ( v4 != v51 || !v51 )
            {
              v137[0].m128i_i64[0] = v17;
              a2 = (__int64)v137;
              v137[0].m128i_i64[1] = v121;
              sub_F9E360((__int64)&v134, v137);
            }
          }
          if ( v144 )
          {
            v75 = v139.m128i_i64[0];
            v76 = sub_B53F50((__int64)&v139);
            a2 = 2;
            sub_B99FD0(v75, 2u, v76);
          }
          if ( v143 && (char *)v139.m128i_i64[1] != &v141 )
            _libc_free(v139.m128i_i64[1], a2);
          break;
        case '"':
          if ( v4 == *(_QWORD *)(v18 - 88) )
          {
            v63 = *(_QWORD *)(a1 + 8);
            if ( v63 )
            {
              sub_FFB3D0(*(_QWORD *)(a1 + 8), v134, (v135 - v134) >> 4);
              if ( v134 != v135 )
                v135 = v134;
              v63 = *(_QWORD *)(a1 + 8);
            }
            v64 = sub_F56CD0(*(const char **)(v18 + 16), v63, v20, v21, v63, v22);
            a2 = 41;
            v65 = (__int64 *)(v64 + 72);
            v66 = (__int64)v64;
            if ( (unsigned __int8)sub_A73ED0((_QWORD *)v64 + 9, 41) )
              goto LABEL_90;
            a2 = 41;
            if ( (unsigned __int8)sub_B49560(v66, 41) )
              goto LABEL_90;
            a2 = sub_BD5C60(v66);
            *(_QWORD *)(v66 + 72) = sub_A7A090(v65, (__int64 *)a2, -1, 41);
            i = v125;
          }
          break;
        case '\'':
          v71 = *(_QWORD *)(v18 - 32);
          v72 = *(_BYTE *)(v18 - 22) & 1;
          if ( (*(_BYTE *)(v18 - 22) & 1) != 0 )
          {
            v73 = *(_QWORD *)(v71 + 32);
            if ( v73 && v4 == v73 )
            {
              v74 = *(_QWORD *)(a1 + 8);
              if ( v74 )
              {
                v130 = *(_BYTE *)(v18 - 22) & 1;
                sub_FFB3D0(*(_QWORD *)(a1 + 8), v134, (v135 - v134) >> 4);
                v72 = v130;
                if ( v134 != v135 )
                  v135 = v134;
                v74 = *(_QWORD *)(a1 + 8);
              }
              a2 = v74;
              i = v72;
              sub_F56CD0(*(const char **)(v18 + 16), v74, v20, v71, v72, v22);
              sub_F94A20(&v152, a2);
              goto LABEL_49;
            }
            v77 = (_QWORD *)(v71 + 64);
          }
          else
          {
            v77 = (_QWORD *)(v71 + 32);
          }
          v78 = 32LL * (*(_DWORD *)(v18 - 20) & 0x7FFFFFF) + v71;
          if ( (_QWORD *)v78 != v77 )
          {
            v79 = v77;
            v80 = v4;
            v81 = i;
            v82 = (_QWORD *)v78;
            do
            {
              if ( v80 == *v79 )
              {
                a2 = (__int64)v79;
                v82 -= 4;
                sub_B4C570(v18 - 24, (__int64)v79);
                v81 = v125;
              }
              else
              {
                v79 += 4;
              }
            }
            while ( v82 != v79 );
            i = v81;
            v4 = v80;
            v19 = (_QWORD *)(v18 - 24);
          }
          if ( *(_QWORD *)(a1 + 8) )
          {
            a2 = (__int64)&v139;
            v139.m128i_i64[0] = v17;
            v139.m128i_i64[1] = v121;
            sub_F9E360((__int64)&v134, &v139);
          }
          v88 = *(_DWORD *)(v18 - 20);
          if ( (*(_BYTE *)(v18 - 22) & 1) != 0 )
          {
            if ( (v88 & 0x7FFFFFF) == 2 )
            {
              if ( !*(_QWORD *)(a1 + 8) )
                goto LABEL_154;
              v111 = sub_F92F30(v17);
              v120 = v112;
              v137[0].m128i_i64[0] = v111;
              if ( v112 != v111 )
              {
                v116 = v19;
                v113 = v18;
                do
                {
                  v114 = *(_QWORD *)(*(_QWORD *)(v111 + 24) + 40LL);
                  v115 = 0;
                  if ( (*(_BYTE *)(v113 - 22) & 1) != 0 )
                    v115 = *(_QWORD *)(*(_QWORD *)(v113 - 32) + 32LL) & 0xFFFFFFFFFFFFFFFBLL;
                  v139.m128i_i64[1] = v115;
                  v139.m128i_i64[0] = v114;
                  sub_F9E360((__int64)&v134, &v139);
                  v139.m128i_i64[0] = v114;
                  v139.m128i_i64[1] = v17 | 4;
                  sub_F9E360((__int64)&v134, &v139);
                  v137[0].m128i_i64[0] = *(_QWORD *)(v137[0].m128i_i64[0] + 8);
                  sub_D4B000(v137[0].m128i_i64);
                  v111 = v137[0].m128i_i64[0];
                }
                while ( v120 != v137[0].m128i_i64[0] );
                v18 = v113;
                v19 = v116;
              }
              if ( (*(_BYTE *)(v18 - 22) & 1) != 0 )
LABEL_154:
                v89 = *(_QWORD *)(*(_QWORD *)(v18 - 32) + 32LL);
              else
                v89 = 0;
              sub_BD84D0(v17, v89);
              goto LABEL_87;
            }
          }
          else if ( (v88 & 0x7FFFFFF) == 1 )
          {
            v90 = *(_QWORD *)(a1 + 8);
            if ( v90 )
            {
              sub_FFB3D0(v90, v134, (v135 - v134) >> 4);
              if ( v134 != v135 )
                v135 = v134;
            }
            v91 = sub_F92F30(v17);
            v94 = v91;
            v139.m128i_i64[0] = (__int64)v140;
            v117 = v91;
            v133 = v95;
            v139.m128i_i64[1] = 0x800000000LL;
            v137[0].m128i_i64[0] = v91;
            if ( v95 == v91 )
            {
              v97 = 0;
            }
            else
            {
              v96 = v91;
              v97 = 0;
              do
              {
                ++v97;
                v137[0].m128i_i64[0] = *(_QWORD *)(v96 + 8);
                sub_D4B000(v137[0].m128i_i64);
                v96 = v137[0].m128i_i64[0];
              }
              while ( v133 != v137[0].m128i_i64[0] );
            }
            v98 = v139.m128i_u32[2];
            v99 = v139.m128i_u32[3];
            v100 = v97 + v139.m128i_u32[2];
            if ( v100 > v139.m128i_u32[3] )
            {
              v99 = (__int64)v140;
              sub_C8D5F0((__int64)&v139, v140, v100, 8u, v92, v93);
              v98 = v139.m128i_u32[2];
            }
            v101 = (const char **)v139.m128i_i64[0];
            v137[0].m128i_i64[0] = v117;
            v102 = v139.m128i_i64[0] + 8 * v98;
            if ( v133 != v117 )
            {
              v103 = v94;
              v104 = (_QWORD *)(v139.m128i_i64[0] + 8 * v98);
              do
              {
                if ( v104 )
                  *v104 = *(_QWORD *)(*(_QWORD *)(v103 + 24) + 40LL);
                ++v104;
                v137[0].m128i_i64[0] = *(_QWORD *)(v103 + 8);
                sub_D4B000(v137[0].m128i_i64);
                v103 = v137[0].m128i_i64[0];
              }
              while ( v133 != v137[0].m128i_i64[0] );
              LODWORD(v98) = v139.m128i_i32[2];
              v101 = (const char **)v139.m128i_i64[0];
            }
            v139.m128i_i32[2] = v97 + v98;
            v105 = (unsigned int)(v97 + v98);
            v106 = &v101[v105];
            if ( v106 != v101 )
            {
              v107 = v19;
              v108 = v101;
              v109 = v107;
              do
              {
                v110 = *v108;
                v99 = *(_QWORD *)(a1 + 8);
                ++v108;
                sub_F56CD0(v110, v99, v102, v105, v92, v93);
              }
              while ( v106 != v108 );
              v101 = (const char **)v139.m128i_i64[0];
              v19 = v109;
            }
            if ( v101 != (const char **)v140 )
              _libc_free(v101, v99);
            goto LABEL_87;
          }
          break;
        case '%':
          if ( *(_QWORD *)(a1 + 8) )
          {
            v139.m128i_i64[0] = v17;
            v139.m128i_i64[1] = v121;
            sub_F9E360((__int64)&v134, &v139);
          }
LABEL_87:
          v57 = sub_BD5C60((__int64)v19);
          a2 = unk_3F148B8;
          v58 = sub_BD2C40(72, unk_3F148B8);
          if ( v58 )
          {
            a2 = v57;
            sub_B4C8A0((__int64)v58, v57, v18, 0);
          }
          sub_B43D60(v19);
LABEL_90:
          i = v125;
          break;
        default:
          break;
      }
LABEL_47:
      nullsub_61();
      v167 = &unk_49DA100;
      nullsub_63();
      if ( v152 != v154 )
        _libc_free(v152, a2);
LABEL_49:
      if ( v122 == ++v127 )
        goto LABEL_50;
    }
    if ( (*(_BYTE *)(v18 - 17) & 0x40) != 0 )
    {
      v24 = *(char **)(v18 - 32);
      v25 = &v24[32 * (*(_DWORD *)(v18 - 20) & 0x7FFFFFF)];
      if ( (*(_DWORD *)(v18 - 20) & 0x7FFFFFF) == 3 )
        v24 += 32;
    }
    else
    {
      v25 = (char *)(v18 - 24);
      v24 = (char *)&v19[-4 * (*(_DWORD *)(v18 - 20) & 0x7FFFFFF) + 4];
      if ( (*(_DWORD *)(v18 - 20) & 0x7FFFFFF) != 3 )
        v24 = (char *)&v19[-4 * (*(_DWORD *)(v18 - 20) & 0x7FFFFFF)];
    }
    v26 = (v25 - v24) >> 7;
    v27 = (v25 - v24) >> 5;
    if ( v26 > 0 )
    {
      v28 = &v24[128 * v26];
      while ( v4 == *(_QWORD *)v24 )
      {
        if ( v4 != *((_QWORD *)v24 + 4) )
        {
          v24 += 32;
          break;
        }
        if ( v4 != *((_QWORD *)v24 + 8) )
        {
          v24 += 64;
          break;
        }
        if ( v4 != *((_QWORD *)v24 + 12) )
        {
          v24 += 96;
          break;
        }
        v24 += 128;
        if ( v28 == v24 )
        {
          v27 = (v25 - v24) >> 5;
          goto LABEL_110;
        }
      }
LABEL_38:
      if ( v24 != v25 )
      {
        v29 = *(_QWORD *)(v18 - 56);
        v30 = *(_QWORD *)(v18 - 120);
        if ( v29 && v29 == v4 )
        {
          v138 = 257;
          v129 = sub_AD62B0(*(_QWORD *)(v30 + 8));
          v69 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v159 + 2))(v159, 30, v30, v129);
          if ( !v69 )
          {
            v142 = 257;
            v131 = sub_B504D0(30, v30, v129, (__int64)&v139, 0, 0);
            (*(void (__fastcall **)(_QWORD *, __int64, __m128i *, __int64, __int64))(*v160 + 16LL))(
              v160,
              v131,
              v137,
              v156,
              v157);
            v69 = v131;
            if ( v152 != &v152[16 * (unsigned int)v153] )
            {
              v132 = v17;
              v83 = v69;
              v119 = v4;
              v84 = (__int64)v152;
              v85 = &v152[16 * (unsigned int)v153];
              do
              {
                v86 = *(_QWORD *)(v84 + 8);
                v87 = *(_DWORD *)v84;
                v84 += 16;
                sub_B99FD0(v83, v87, v86);
              }
              while ( v85 != (_BYTE *)v84 );
              v69 = v83;
              v4 = v119;
              v17 = v132;
            }
          }
          v70 = sub_B33B40((__int64)&v152, v69, 0, 0);
          a2 = *(_QWORD *)(v18 - 88);
          v32 = v70;
          sub_F902B0((__int64 *)&v152, a2);
        }
        else
        {
          v31 = sub_B33B40((__int64)&v152, *(_QWORD *)(v18 - 120), 0, 0);
          a2 = *(_QWORD *)(v18 - 56);
          v32 = v31;
          sub_F902B0((__int64 *)&v152, a2);
        }
        v37 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL);
        if ( v37 )
        {
          a2 = v32;
          sub_CFEAE0(v37, v32, v33, v34, v35, v36);
        }
        sub_F91380((char *)(v18 - 24));
        goto LABEL_45;
      }
LABEL_113:
      v67 = sub_BD5C60(v18 - 24);
      a2 = unk_3F148B8;
      v68 = sub_BD2C40(72, unk_3F148B8);
      if ( v68 )
      {
        a2 = v67;
        sub_B4C8A0((__int64)v68, v67, v18, 0);
      }
      sub_B43D60((_QWORD *)(v18 - 24));
LABEL_45:
      if ( *(_QWORD *)(a1 + 8) )
      {
        a2 = (__int64)&v139;
        v139.m128i_i64[0] = v17;
        v139.m128i_i64[1] = v121;
        sub_F9E360((__int64)&v134, &v139);
        i = v125;
        goto LABEL_47;
      }
      goto LABEL_90;
    }
LABEL_110:
    if ( v27 != 2 )
    {
      if ( v27 != 3 )
      {
        if ( v27 != 1 )
          goto LABEL_113;
        goto LABEL_135;
      }
      if ( v4 != *(_QWORD *)v24 )
        goto LABEL_38;
      v24 += 32;
    }
    if ( v4 != *(_QWORD *)v24 )
      goto LABEL_38;
    v24 += 32;
LABEL_135:
    if ( v4 == *(_QWORD *)v24 )
      goto LABEL_113;
    goto LABEL_38;
  }
  return v125;
}
