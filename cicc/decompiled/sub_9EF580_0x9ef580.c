// Function: sub_9EF580
// Address: 0x9ef580
//
__int64 *__fastcall sub_9EF580(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r15
  __int64 v3; // rcx
  __int64 v4; // r15
  __int64 *v5; // r14
  __int64 *v6; // r13
  __int64 v7; // rbx
  __int64 *p_s; // r12
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  unsigned int v12; // r11d
  __int64 v13; // rsi
  __int64 v14; // r9
  int v15; // edx
  char v16; // al
  unsigned __int64 v18; // rax
  char v19; // si
  __int64 *v20; // rax
  unsigned __int64 v21; // rdi
  size_t v22; // rcx
  const char *v23; // rdx
  unsigned int v24; // r8d
  __int64 v25; // rax
  unsigned int v26; // r11d
  unsigned int v27; // edx
  char *v28; // rax
  bool v29; // dl
  __int64 v30; // rsi
  char v31; // dl
  char v32; // al
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rax
  __int64 *v35; // rcx
  __int64 v36; // r12
  __int64 v37; // rbx
  __m128i *v38; // rax
  __int64 v39; // rcx
  unsigned __int64 *v40; // rsi
  int v41; // edx
  int v42; // edx
  char v43; // al
  __int64 i; // rax
  __int64 v45; // rdi
  const void *v46; // rsi
  size_t v47; // rdx
  __int64 v48; // rax
  size_t v49; // rax
  __int64 v50; // rax
  __int64 v51; // rsi
  unsigned __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rcx
  int v55; // edx
  int v56; // edx
  char v57; // al
  int v58; // r13d
  __int64 v59; // r9
  __int64 v60; // rcx
  char *v61; // rdi
  __int64 v62; // r8
  _QWORD *v63; // rax
  __int64 *v64; // rsi
  __int64 v65; // rax
  char *v66; // r10
  int v67; // edx
  __int64 v68; // r9
  __int64 v69; // rsi
  unsigned int k; // eax
  _DWORD *v71; // rdx
  __int64 v72; // r14
  __int64 v73; // r13
  _QWORD *v74; // rbx
  _QWORD *v75; // r12
  unsigned int v76; // esi
  int v77; // ecx
  int v78; // r8d
  __int64 v79; // rdi
  unsigned int j; // edx
  __int64 v81; // rax
  int v82; // r9d
  unsigned __int64 *v83; // rax
  int v84; // r11d
  unsigned int m; // eax
  int v86; // eax
  unsigned int v87; // r11d
  unsigned int v88; // eax
  int v89; // eax
  unsigned int v90; // edx
  const char *v91; // rax
  __m128i *v92; // rax
  __int64 *v93; // [rsp+28h] [rbp-6D8h]
  unsigned __int64 v94; // [rsp+38h] [rbp-6C8h]
  unsigned __int64 v95; // [rsp+38h] [rbp-6C8h]
  unsigned __int64 v96; // [rsp+40h] [rbp-6C0h]
  unsigned __int64 v97; // [rsp+48h] [rbp-6B8h]
  __int64 *v98; // [rsp+78h] [rbp-688h]
  __int64 v99; // [rsp+80h] [rbp-680h]
  __int64 *v100; // [rsp+88h] [rbp-678h]
  char *v101; // [rsp+98h] [rbp-668h]
  int v102; // [rsp+B0h] [rbp-650h]
  unsigned int v103; // [rsp+B0h] [rbp-650h]
  unsigned int v104; // [rsp+B0h] [rbp-650h]
  unsigned int v105; // [rsp+B0h] [rbp-650h]
  unsigned int v106; // [rsp+B8h] [rbp-648h]
  __int64 *v107; // [rsp+B8h] [rbp-648h]
  unsigned int v108; // [rsp+B8h] [rbp-648h]
  unsigned int v109; // [rsp+B8h] [rbp-648h]
  unsigned int v110; // [rsp+B8h] [rbp-648h]
  unsigned int v111; // [rsp+B8h] [rbp-648h]
  unsigned int v112; // [rsp+B8h] [rbp-648h]
  unsigned int v113; // [rsp+C0h] [rbp-640h]
  __int64 v114; // [rsp+D0h] [rbp-630h] BYREF
  __int64 v115; // [rsp+D8h] [rbp-628h] BYREF
  __int64 v116; // [rsp+E0h] [rbp-620h] BYREF
  char v117; // [rsp+E8h] [rbp-618h]
  __int64 v118; // [rsp+F0h] [rbp-610h] BYREF
  char v119; // [rsp+F8h] [rbp-608h]
  unsigned __int64 v120; // [rsp+100h] [rbp-600h] BYREF
  char v121; // [rsp+108h] [rbp-5F8h]
  __int128 v122; // [rsp+110h] [rbp-5F0h]
  int v123; // [rsp+120h] [rbp-5E0h]
  char v124; // [rsp+130h] [rbp-5D0h] BYREF
  __int64 v125; // [rsp+150h] [rbp-5B0h] BYREF
  __int64 v126; // [rsp+158h] [rbp-5A8h]
  __int64 v127; // [rsp+160h] [rbp-5A0h]
  unsigned int v128; // [rsp+168h] [rbp-598h]
  unsigned __int64 v129; // [rsp+170h] [rbp-590h] BYREF
  char v130; // [rsp+178h] [rbp-588h]
  __int64 v131; // [rsp+190h] [rbp-570h] BYREF
  __int64 v132; // [rsp+198h] [rbp-568h]
  __m128i v133; // [rsp+1A0h] [rbp-560h] BYREF
  __int64 *v134; // [rsp+1B0h] [rbp-550h] BYREF
  char v135; // [rsp+1B8h] [rbp-548h]
  __int64 v136; // [rsp+1C0h] [rbp-540h] BYREF
  __int16 v137; // [rsp+1D0h] [rbp-530h]
  __m128i v138; // [rsp+1E0h] [rbp-520h] BYREF
  __m128i v139; // [rsp+1F0h] [rbp-510h] BYREF
  __int64 v140; // [rsp+200h] [rbp-500h]
  unsigned __int64 v141; // [rsp+210h] [rbp-4F0h] BYREF
  size_t v142; // [rsp+218h] [rbp-4E8h]
  __int64 v143; // [rsp+220h] [rbp-4E0h]
  char v144; // [rsp+228h] [rbp-4D8h] BYREF
  __int16 v145; // [rsp+230h] [rbp-4D0h]
  __int64 *v146; // [rsp+2B0h] [rbp-450h] BYREF
  __int64 v147; // [rsp+2B8h] [rbp-448h]
  _BYTE v148[512]; // [rsp+2C0h] [rbp-440h] BYREF
  char *s; // [rsp+4C0h] [rbp-240h] BYREF
  __int64 v150; // [rsp+4C8h] [rbp-238h]
  unsigned __int64 v151; // [rsp+4D0h] [rbp-230h] BYREF
  char v152[8]; // [rsp+4D8h] [rbp-228h] BYREF
  __int16 v153; // [rsp+4E0h] [rbp-220h]

  v2 = a1;
  sub_A4DCE0(&s, a2 + 24, 8, 0);
  if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL | 1;
    return v2;
  }
  v3 = (__int64)a1;
  v125 = 0;
  v4 = a2 + 24;
  v5 = a1;
  v6 = &v116;
  v146 = (__int64 *)v148;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v113 = 0;
  v147 = 0x4000000000LL;
  v7 = a2;
  p_s = (__int64 *)&s;
  do
  {
LABEL_3:
    sub_9CEA50((__int64)v6, v4, 0, v3);
    v11 = v117 & 1;
    v3 = (unsigned int)(2 * v11);
    v117 = (2 * v11) | v117 & 0xFD;
    if ( (_BYTE)v11 )
    {
      v2 = v5;
      sub_9C9090(v5, v6);
      goto LABEL_16;
    }
    v12 = HIDWORD(v116);
    if ( (_DWORD)v116 == 2 )
    {
      switch ( HIDWORD(v116) )
      {
        case 0:
          sub_9D23D0(p_s, (_QWORD *)v7);
          v18 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_73;
          goto LABEL_22;
        case 0x13:
          sub_A4DCE0(p_s, v4, 19, 0);
          v18 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_73;
          v142 = 0;
          s = (char *)&v151;
          v150 = 0x4000000000LL;
          v141 = (unsigned __int64)&v144;
          v143 = 128;
          v107 = 0;
          while ( 2 )
          {
            v40 = (unsigned __int64 *)v4;
            sub_9CEFB0((__int64)&v118, v4, 0, v39);
            v41 = v119 & 1;
            v3 = (unsigned int)(2 * v41);
            v119 = (2 * v41) | v119 & 0xFD;
            if ( (_BYTE)v41 )
            {
              v40 = (unsigned __int64 *)&v118;
              sub_9C9090(&v114, &v118);
              goto LABEL_128;
            }
            if ( (_DWORD)v118 == 1 )
            {
              v114 = 1;
              goto LABEL_132;
            }
            if ( (v118 & 0xFFFFFFFD) == 0 )
            {
              v40 = (unsigned __int64 *)v7;
              v138.m128i_i64[0] = (__int64)"Malformed block";
              LOWORD(v140) = 259;
              sub_9C81F0(&v114, v7, (__int64)&v138);
              goto LABEL_128;
            }
            LODWORD(v150) = 0;
            sub_A4B600(&v120, v4, HIDWORD(v118), p_s, 0);
            v42 = v121 & 1;
            v39 = (unsigned int)(2 * v42);
            v43 = (2 * v42) | v121 & 0xFD;
            v121 = v43;
            if ( (_BYTE)v42 )
            {
              v40 = &v120;
              sub_9C8CD0(&v114, (__int64 *)&v120);
              goto LABEL_183;
            }
            if ( (_DWORD)v120 == 1 )
            {
              v134 = *(__int64 **)s;
              if ( !(unsigned __int8)sub_9C3E90((__int64)s, (unsigned int)v150, 1u, &v141) )
              {
                v60 = 5;
                v61 = &v124;
                v62 = *(_QWORD *)(v7 + 424);
                while ( v60 )
                {
                  *(_DWORD *)v61 = 0;
                  v61 += 4;
                  --v60;
                }
                v107 = (__int64 *)sub_9C7D70(v62, (const void *)v141, v142, 0, v62, v59, (__m128i)0LL, 0);
                v99 = *v107;
                v63 = sub_9E2360(v7 + 480, &v134);
                v39 = (__int64)(v107 + 4);
                v63[1] = v99;
                *v63 = v107 + 4;
                v142 = 0;
LABEL_87:
                v43 = v121;
                if ( (v121 & 2) == 0 )
                {
LABEL_88:
                  if ( (v43 & 1) != 0 && v120 )
                    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v120 + 8LL))(v120);
                  if ( (v119 & 2) == 0 )
                  {
                    if ( (v119 & 1) != 0 && v118 )
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v118 + 8LL))(v118);
                    continue;
                  }
LABEL_151:
                  sub_9CEF10(&v118);
                }
LABEL_165:
                sub_9CE230(&v120);
              }
              BYTE1(v140) = 1;
              v91 = "Invalid record";
              goto LABEL_245;
            }
            break;
          }
          if ( (_DWORD)v120 != 2 )
            goto LABEL_88;
          if ( (unsigned int)v150 == 5 )
          {
            if ( v107 )
            {
              v39 = (__int64)s;
              for ( i = 0; i != 20; i += 4 )
                *(_DWORD *)((char *)v107 + i + 8) = *(_QWORD *)(v39 + 2 * i);
              v107 = 0;
              goto LABEL_87;
            }
            BYTE1(v140) = 1;
            v91 = "Invalid hash that does not follow a module path";
LABEL_245:
            v40 = (unsigned __int64 *)v7;
            v138.m128i_i64[0] = (__int64)v91;
            LOBYTE(v140) = 3;
            sub_9C81F0(&v114, v7, (__int64)&v138);
            goto LABEL_183;
          }
          v115 = (unsigned int)v150;
          v134 = &v115;
          v137 = 267;
          sub_CA0F50(&v129, &v134);
          v92 = (__m128i *)sub_2241130(&v129, 0, 0, "Invalid hash length ", 20);
          v131 = (__int64)&v133;
          if ( (__m128i *)v92->m128i_i64[0] == &v92[1] )
          {
            v133 = _mm_loadu_si128(v92 + 1);
          }
          else
          {
            v131 = v92->m128i_i64[0];
            v133.m128i_i64[0] = v92[1].m128i_i64[0];
          }
          v40 = (unsigned __int64 *)v7;
          v132 = v92->m128i_i64[1];
          v92->m128i_i64[0] = (__int64)v92[1].m128i_i64;
          v92->m128i_i64[1] = 0;
          v92[1].m128i_i8[0] = 0;
          v138.m128i_i64[0] = (__int64)&v131;
          LOWORD(v140) = 260;
          sub_9C81F0(&v114, v7, (__int64)&v138);
          sub_2240A30(&v131);
          sub_2240A30(&v129);
LABEL_183:
          if ( (v121 & 2) != 0 )
            goto LABEL_165;
          if ( (v121 & 1) != 0 && v120 )
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v120 + 8LL))(v120);
LABEL_128:
          if ( (v119 & 2) != 0 )
            goto LABEL_151;
          if ( (v119 & 1) != 0 && v118 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v118 + 8LL))(v118);
LABEL_132:
          if ( (char *)v141 != &v144 )
            _libc_free(v141, v40);
          if ( s != (char *)&v151 )
            _libc_free(s, v40);
          v18 = v114 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v114 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_73;
          goto LABEL_22;
        case 0x14:
        case 0x18:
          if ( *(_QWORD *)(v7 + 520) )
          {
            v45 = *(_QWORD *)(v7 + 424);
            v46 = *(const void **)(v7 + 544);
            v47 = *(_QWORD *)(v7 + 552);
            v108 = HIDWORD(v116);
            v123 = 0;
            v122 = 0;
            sub_9C7D70(v45, v46, v47, v3, v9, v10, (__m128i)0LL, 0);
            v12 = v108;
          }
          v30 = *(_QWORD *)(v7 + 440);
          if ( !v30 )
            goto LABEL_96;
          if ( *(_BYTE *)(v7 + 384) )
            goto LABEL_95;
          v106 = v12;
          sub_9CF0C0((__int64)&v129, v30, v4, v3);
          v12 = v106;
          v31 = v130 & 1;
          v32 = (2 * (v130 & 1)) | v130 & 0xFD;
          v130 = v32;
          if ( v31 )
          {
            v33 = v129;
            v129 = 0;
            v130 = v32 & 0xFD;
            v120 = v33 | 1;
            goto LABEL_49;
          }
          v97 = v129;
          sub_A4DCE0(p_s, v4, 14, 0);
          v12 = v106;
          if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            v120 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL | 1;
            goto LABEL_142;
          }
          s = (char *)&v151;
          v150 = 0x4000000000LL;
          v141 = (unsigned __int64)&v144;
          v142 = 0;
          v143 = 128;
          v100 = v5;
          v98 = v6;
          while ( 2 )
          {
            sub_9CEFB0((__int64)&v131, v4, 0, v54);
            v55 = v132 & 1;
            LOBYTE(v132) = (2 * v55) | v132 & 0xFD;
            if ( (_BYTE)v55 )
            {
              v64 = &v131;
              v5 = v100;
              v6 = v98;
              sub_9C9090((__int64 *)&v120, &v131);
              v12 = v106;
            }
            else if ( (_DWORD)v131 == 1 )
            {
              v64 = (__int64 *)v4;
              v5 = v100;
              v6 = v98;
              sub_9CDFE0(v138.m128i_i64, v4, v97, (unsigned int)(2 * v55));
              v12 = v106;
              v65 = v138.m128i_i64[0] | 1;
              if ( (v138.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
                v65 = 1;
              v120 = v65;
            }
            else if ( (v131 & 0xFFFFFFFD) != 0 )
            {
              LODWORD(v150) = 0;
              sub_A4B600(&v134, v4, HIDWORD(v131), p_s, 0);
              v56 = v135 & 1;
              v54 = (unsigned int)(2 * v56);
              v57 = (2 * v56) | v135 & 0xFD;
              v135 = v57;
              if ( (_BYTE)v56 )
              {
                v64 = (__int64 *)&v134;
                v5 = v100;
                v6 = v98;
                sub_9C8CD0((__int64 *)&v120, (__int64 *)&v134);
                v87 = v106;
              }
              else
              {
                v58 = (int)v134;
                if ( (_DWORD)v134 != 3 )
                {
                  if ( (_DWORD)v134 == 5 )
                  {
                    v72 = *(_QWORD *)(v7 + 424);
                    LODWORD(v118) = *(_QWORD *)s;
                    v120 = *((_QWORD *)s + 1);
                    v96 = v120;
                    if ( *(_BYTE *)(v72 + 343) )
                    {
                      v138.m128i_i64[0] = 0;
                    }
                    else
                    {
                      v138.m128i_i64[1] = 0;
                      v138.m128i_i64[0] = (__int64)byte_3F871B3;
                    }
                    v139 = 0u;
                    v140 = 0;
                    v94 = (unsigned __int64)(sub_9CA390((_QWORD *)v72, &v120, &v138) + 4);
                    if ( v139.m128i_i64[1] != v139.m128i_i64[0] )
                    {
                      v73 = v7;
                      v74 = (_QWORD *)v139.m128i_i64[0];
                      v93 = p_s;
                      v75 = (_QWORD *)v139.m128i_i64[1];
                      do
                      {
                        if ( *v74 )
                          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v74 + 8LL))(*v74);
                        ++v74;
                      }
                      while ( v75 != v74 );
                      v7 = v73;
                      p_s = v93;
                    }
                    if ( v139.m128i_i64[0] )
                      j_j___libc_free_0(v139.m128i_i64[0], v140 - v139.m128i_i64[0]);
                    v76 = *(_DWORD *)(v7 + 472);
                    v95 = *(unsigned __int8 *)(v72 + 343) | v94 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v76 )
                    {
                      v77 = v118;
                      v78 = 1;
                      v79 = 0;
                      for ( j = (v76 - 1) & (37 * v118); ; j = (v76 - 1) & v90 )
                      {
                        v81 = *(_QWORD *)(v7 + 456) + 24LL * j;
                        v82 = *(_DWORD *)v81;
                        if ( (_DWORD)v118 == *(_DWORD *)v81 )
                          break;
                        if ( v82 == -1 )
                        {
                          if ( v79 )
                            v81 = v79;
                          ++*(_QWORD *)(v7 + 448);
                          v138.m128i_i64[0] = v81;
                          v89 = *(_DWORD *)(v7 + 464) + 1;
                          if ( 4 * v89 >= 3 * v76 )
                            goto LABEL_233;
                          if ( v76 - *(_DWORD *)(v7 + 468) - v89 <= v76 >> 3 )
                            goto LABEL_234;
                          goto LABEL_229;
                        }
                        if ( v79 || v82 != -2 )
                          v81 = v79;
                        v90 = v78 + j;
                        v79 = v81;
                        ++v78;
                      }
                    }
                    else
                    {
                      ++*(_QWORD *)(v7 + 448);
                      v138.m128i_i64[0] = 0;
LABEL_233:
                      v76 *= 2;
LABEL_234:
                      sub_9E0BD0(v7 + 448, v76);
                      sub_9CCB10(v7 + 448, (int *)&v118, &v138);
                      v77 = v118;
LABEL_229:
                      v81 = v138.m128i_i64[0];
                      ++*(_DWORD *)(v7 + 464);
                      if ( *(_DWORD *)v81 != -1 )
                        --*(_DWORD *)(v7 + 468);
                      *(_DWORD *)v81 = v77;
                      *(_QWORD *)(v81 + 8) = 0;
                      *(_QWORD *)(v81 + 16) = 0;
                    }
                    v83 = (unsigned __int64 *)(v81 + 8);
                    *v83 = v95;
                    v54 = v96;
                    v83[1] = v96;
                  }
                  else
                  {
                    if ( (_DWORD)v134 != 1 )
                    {
LABEL_122:
                      if ( (v57 & 1) != 0 && v134 )
                        (*(void (__fastcall **)(__int64 *))(*v134 + 8))(v134);
                      if ( (v132 & 2) != 0 )
                        goto LABEL_181;
                      if ( (v132 & 1) != 0 && v131 )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v131 + 8LL))(v131);
                      continue;
                    }
                    if ( (unsigned __int8)sub_9C3E90((__int64)s, (unsigned int)v150, 1u, &v141) )
                      goto LABEL_219;
                    v69 = *(_QWORD *)s;
                    if ( v128 )
                    {
                      v68 = v128 - 1;
                      for ( k = v68 & (37 * v69); ; k = v68 & v86 )
                      {
                        v71 = (_DWORD *)(v126 + 8LL * k);
                        if ( (unsigned int)*(_QWORD *)s == *v71 )
                          break;
                        if ( *v71 == -1 )
                          goto LABEL_217;
                        v86 = v58 + k;
                        ++v58;
                      }
                    }
                    else
                    {
LABEL_217:
                      v71 = (_DWORD *)(v126 + 8LL * v128);
                    }
LABEL_193:
                    sub_9E0DB0(
                      v7,
                      v69,
                      (const void *)v141,
                      v142,
                      v71[1],
                      v68,
                      *(_QWORD *)(v7 + 512),
                      *(_QWORD *)(v7 + 520));
                    v142 = 0;
                  }
                  v57 = v135;
                  if ( (v135 & 2) != 0 )
                    sub_9CE230(&v134);
                  goto LABEL_122;
                }
                if ( !(unsigned __int8)sub_9C3E90((__int64)s, (unsigned int)v150, 2u, &v141) )
                {
                  v69 = *(_QWORD *)s;
                  if ( v128 )
                  {
                    v68 = v128 - 1;
                    v84 = 1;
                    for ( m = v68 & (37 * v69); ; m = v68 & v88 )
                    {
                      v71 = (_DWORD *)(v126 + 8LL * m);
                      if ( (unsigned int)*(_QWORD *)s == *v71 )
                        break;
                      if ( *v71 == -1 )
                        goto LABEL_222;
                      v88 = v84 + m;
                      ++v84;
                    }
                  }
                  else
                  {
LABEL_222:
                    v71 = (_DWORD *)(v126 + 8LL * v128);
                  }
                  goto LABEL_193;
                }
LABEL_219:
                v64 = (__int64 *)v7;
                v138.m128i_i64[0] = (__int64)"Invalid record";
                v5 = v100;
                LOWORD(v140) = 259;
                v6 = v98;
                sub_9C81F0((__int64 *)&v120, v7, (__int64)&v138);
                v87 = v106;
              }
              v105 = v87;
              sub_9CE2A0(&v134);
              v12 = v105;
            }
            else
            {
              v64 = (__int64 *)v7;
              v138.m128i_i64[0] = (__int64)"Malformed block";
              v5 = v100;
              LOWORD(v140) = 259;
              v6 = v98;
              sub_9C81F0((__int64 *)&v120, v7, (__int64)&v138);
              v12 = v106;
            }
            break;
          }
          if ( (v132 & 2) != 0 )
LABEL_181:
            sub_9CEF10(&v131);
          if ( (v132 & 1) != 0 && v131 )
          {
            v112 = v12;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v131 + 8LL))(v131);
            v12 = v112;
          }
          if ( (char *)v141 != &v144 )
          {
            v110 = v12;
            _libc_free(v141, v64);
            v12 = v110;
          }
          if ( s != (char *)&v151 )
          {
            v111 = v12;
            _libc_free(s, v64);
            v12 = v111;
          }
LABEL_142:
          if ( (v130 & 2) != 0 )
            sub_9CDF70(&v129);
          if ( (v130 & 1) != 0 && v129 )
          {
            v109 = v12;
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v129 + 8LL))(v129);
            v12 = v109;
          }
LABEL_49:
          v34 = v120 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v120 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *v5 = 0;
            v2 = v5;
            v120 = v34 | 1;
            sub_9C6670(v5, &v120);
            sub_9C66B0((__int64 *)&v120);
            goto LABEL_16;
          }
LABEL_95:
          *(_BYTE *)(v7 + 433) = 1;
LABEL_96:
          *(_BYTE *)(v7 + 432) = 1;
          sub_9EBD80(p_s, v7, v12);
          v18 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_73:
            v2 = v5;
            *v5 = v18 | 1;
            goto LABEL_16;
          }
          break;
        default:
          sub_9CE5C0(p_s, v4, v11, v3);
          v18 = (unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)s & 0xFFFFFFFFFFFFFFFELL) == 0 )
            goto LABEL_22;
          goto LABEL_73;
      }
      goto LABEL_22;
    }
    if ( (unsigned int)v116 <= 2 )
    {
      v2 = v5;
      if ( !(_DWORD)v116 )
      {
        s = "Malformed block";
        v153 = 259;
        sub_9C81F0(v5, v7, (__int64)p_s);
        goto LABEL_16;
      }
      *v5 = 1;
      goto LABEL_8;
    }
  }
  while ( (_DWORD)v116 != 3 );
  LODWORD(v147) = 0;
  sub_A4B600(&v131, v4, HIDWORD(v116), &v146, 0);
  v15 = v132 & 1;
  v3 = (unsigned int)(2 * v15);
  v16 = (2 * v15) | v132 & 0xFD;
  LOBYTE(v132) = v16;
  if ( (_BYTE)v15 )
  {
    v2 = v5;
    sub_9C8CD0(v5, &v131);
    goto LABEL_60;
  }
  switch ( (int)v131 )
  {
    case 0:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 9:
    case 10:
    case 11:
    case 12:
    case 15:
      v16 = v132;
      goto LABEL_27;
    case 1:
      sub_9C8860((__int64)p_s, v7, v146, (unsigned int)v147);
      sub_9C8CD0((__int64 *)&v141, p_s);
      if ( (v150 & 2) != 0 )
        sub_9CE230(p_s);
      if ( (v150 & 1) != 0 && s )
        (*(void (__fastcall **)(char *))(*(_QWORD *)s + 8LL))(s);
      if ( (v141 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v2 = v5;
        *v5 = v141 & 0xFFFFFFFFFFFFFFFELL | 1;
        goto LABEL_60;
      }
      v141 = 0;
      sub_9C66B0((__int64 *)&v141);
      goto LABEL_71;
    case 7:
    case 8:
    case 14:
      v19 = *(_BYTE *)(v7 + 384);
      v20 = v146;
      v21 = (unsigned int)v147;
      if ( v19 )
      {
        v22 = v146[1];
        if ( *v146 + v22 > *(_QWORD *)(v7 + 376) )
          goto LABEL_94;
        v23 = (const char *)(*(_QWORD *)(v7 + 368) + *v146);
        v21 = (unsigned int)v147 - 2LL;
        v20 = v146 + 2;
      }
      else
      {
        v22 = 0;
        v23 = byte_3F871B3;
      }
      if ( v21 <= 3 )
      {
LABEL_94:
        v2 = v5;
        s = "Invalid record";
        v153 = 259;
        sub_9C81F0(v5, v7, (__int64)p_s);
        goto LABEL_60;
      }
      v24 = 0;
      v25 = (unsigned int)v20[3] - 1;
      if ( (unsigned int)v25 <= 0x12 )
        v24 = dword_3F22240[v25];
      v26 = v113 + 1;
      if ( !v19 )
      {
        LODWORD(v141) = v113;
        if ( v128 )
        {
          v27 = (v128 - 1) & (37 * v113);
          v28 = (char *)(v126 + 8LL * v27);
          v3 = *(unsigned int *)v28;
          if ( v113 == (_DWORD)v3 )
          {
LABEL_39:
            *((_DWORD *)v28 + 1) = v24;
            goto LABEL_40;
          }
          v102 = 1;
          v66 = 0;
          while ( (_DWORD)v3 != -1 )
          {
            if ( (_DWORD)v3 == -2 && !v66 )
              v66 = v28;
            v27 = (v128 - 1) & (v102 + v27);
            ++v102;
            v28 = (char *)(v126 + 8LL * v27);
            v3 = *(unsigned int *)v28;
            if ( v113 == (_DWORD)v3 )
              goto LABEL_39;
          }
          if ( v66 )
            v28 = v66;
          ++v125;
          v67 = v127 + 1;
          s = v28;
          if ( 4 * ((int)v127 + 1) < 3 * v128 )
          {
            if ( v128 - HIDWORD(v127) - v67 <= v128 >> 3 )
            {
              v103 = v24;
              sub_9E1F80((__int64)&v125, v128);
              sub_9CCBB0((__int64)&v125, (int *)&v141, p_s);
              v26 = v113 + 1;
              v24 = v103;
              v113 = v141;
              v67 = v127 + 1;
              v28 = s;
            }
            goto LABEL_173;
          }
        }
        else
        {
          ++v125;
          s = 0;
        }
        v104 = v24;
        sub_9E1F80((__int64)&v125, 2 * v128);
        sub_9CCBB0((__int64)&v125, (int *)&v141, p_s);
        v24 = v104;
        v26 = v113 + 1;
        v113 = v141;
        v67 = v127 + 1;
        v28 = s;
LABEL_173:
        LODWORD(v127) = v67;
        if ( *(_DWORD *)v28 != -1 )
          --HIDWORD(v127);
        v3 = v113;
        *(_QWORD *)v28 = v113;
        goto LABEL_39;
      }
      sub_9E0DB0(v7, v113, v23, v22, v24, v14, *(_QWORD *)(v7 + 512), *(_QWORD *)(v7 + 520));
      v26 = v113 + 1;
LABEL_40:
      v16 = v132;
      v113 = v26;
      v29 = (v132 & 2) != 0;
      goto LABEL_41;
    case 13:
      v3 = (unsigned int)v147;
      if ( !(_DWORD)v147 )
        goto LABEL_94;
      *(_QWORD *)(v7 + 440) = *v146 - 1;
      goto LABEL_22;
    case 16:
      v150 = 0;
      s = v152;
      v151 = 128;
      if ( (unsigned __int8)sub_9C3E90((__int64)v146, (unsigned int)v147, 0, p_s) )
      {
        v2 = v5;
        v141 = (unsigned __int64)"Invalid record";
        v145 = 259;
        sub_9C81F0(v5, v7, (__int64)&v141);
        if ( s != v152 )
          _libc_free(s, v7);
        goto LABEL_60;
      }
      v48 = v150;
      if ( v150 + 1 > v151 )
      {
        sub_C8D290(p_s, v152, v150 + 1, 1);
        v48 = v150;
      }
      s[v48] = 0;
      v101 = s;
      v49 = strlen(s);
      sub_2241130(v7 + 512, 0, *(_QWORD *)(v7 + 520), v101, v49);
      if ( s != v152 )
        _libc_free(s, 0);
      goto LABEL_71;
    case 17:
      if ( (unsigned int)v147 == 5 )
      {
        v50 = sub_9C3450((_QWORD *)v7);
        v3 = (__int64)v146;
        v51 = v50;
        if ( &v146[(unsigned int)v147] != v146 )
        {
          v52 = 4 * ((8 * (unsigned __int64)(unsigned int)v147 - 8) >> 3) + 4;
          v53 = 0;
          do
          {
            *(_DWORD *)(v51 + v53 + 8) = *(_QWORD *)(v3 + 2 * v53);
            v53 += 4;
          }
          while ( v53 != v52 );
        }
LABEL_71:
        v16 = v132;
        v29 = (v132 & 2) != 0;
LABEL_41:
        if ( v29 )
          goto LABEL_42;
LABEL_27:
        if ( (v16 & 1) != 0 && v131 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v131 + 8LL))(v131);
LABEL_22:
        if ( (v117 & 2) != 0 )
          goto LABEL_93;
        if ( (v117 & 1) != 0 && v116 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v116 + 8LL))(v116);
        goto LABEL_3;
      }
      v2 = v5;
      v35 = p_s;
      v129 = (unsigned int)v147;
      v36 = v7;
      v145 = 267;
      v37 = (__int64)v35;
      v141 = (unsigned __int64)&v129;
      sub_CA0F50(&v134, &v141);
      v38 = (__m128i *)sub_2241130(&v134, 0, 0, "Invalid hash length ", 20);
      v138.m128i_i64[0] = (__int64)&v139;
      if ( (__m128i *)v38->m128i_i64[0] == &v38[1] )
      {
        v139 = _mm_loadu_si128(v38 + 1);
      }
      else
      {
        v138.m128i_i64[0] = v38->m128i_i64[0];
        v139.m128i_i64[0] = v38[1].m128i_i64[0];
      }
      v138.m128i_i64[1] = v38->m128i_i64[1];
      v38->m128i_i64[0] = (__int64)v38[1].m128i_i64;
      v38->m128i_i64[1] = 0;
      v38[1].m128i_i8[0] = 0;
      v153 = 260;
      s = (char *)&v138;
      sub_9C81F0(v5, v36, v37);
      if ( (__m128i *)v138.m128i_i64[0] != &v139 )
        j_j___libc_free_0(v138.m128i_i64[0], v139.m128i_i64[0] + 1);
      if ( v134 != &v136 )
        j_j___libc_free_0(v134, v136 + 1);
LABEL_60:
      if ( (v132 & 2) != 0 )
LABEL_42:
        sub_9CE230(&v131);
      if ( (v132 & 1) != 0 && v131 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v131 + 8LL))(v131);
LABEL_16:
      if ( (v117 & 2) != 0 )
LABEL_93:
        sub_9CEF10(v6);
      if ( (v117 & 1) != 0 && v116 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v116 + 8LL))(v116);
LABEL_8:
      v13 = 8LL * v128;
      sub_C7D6A0(v126, v13, 4);
      if ( v146 != (__int64 *)v148 )
        _libc_free(v146, v13);
      return v2;
    default:
      goto LABEL_27;
  }
}
