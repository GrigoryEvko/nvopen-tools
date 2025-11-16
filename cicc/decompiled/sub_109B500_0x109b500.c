// Function: sub_109B500
// Address: 0x109b500
//
__m128i *__fastcall sub_109B500(__m128i *a1, __int64 a2, size_t a3, unsigned __int64 a4, __int64 a5)
{
  size_t v5; // r13
  const char *v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r12
  size_t v13; // r14
  bool v14; // zf
  signed __int64 **v15; // r12
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  __int64 v20; // r8
  signed __int64 **v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // r9
  __int64 v24; // r15
  __int64 v25; // rbx
  char v26; // al
  size_t v27; // rbx
  _BYTE *v28; // rax
  __int64 v29; // rax
  __int64 *v30; // r12
  unsigned __int64 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  signed __int64 *v35; // rbx
  signed __int64 *v36; // r13
  char v37; // dl
  __int64 v38; // rdx
  const void *v39; // rsi
  int v40; // eax
  __int64 v41; // r12
  unsigned int v42; // r15d
  __int64 v43; // r15
  unsigned __int64 v44; // r15
  _QWORD *v45; // r12
  _QWORD *v46; // rdi
  int v47; // edi
  __int8 v48; // al
  __m128i v49; // xmm1
  int v50; // r10d
  __int8 v51; // al
  _BYTE *v52; // r13
  signed __int64 *v54; // rbx
  signed __int64 *v55; // r12
  _BYTE *v56; // r12
  _BYTE *v57; // rdi
  _BYTE *v58; // r14
  __int64 v59; // rbx
  __int64 v60; // rdi
  unsigned __int64 *v61; // rsi
  unsigned __int8 v62; // al
  _QWORD *v63; // rbx
  int v64; // eax
  __int64 v65; // rax
  size_t v66; // rdx
  size_t v67; // rbx
  __int64 v68; // rax
  __int64 *v69; // rax
  __int64 v70; // rax
  size_t v71; // rcx
  unsigned __int64 v72; // rdx
  __int64 *v73; // rax
  __int64 v74; // rax
  __int64 *v75; // rbx
  __int64 v76; // rax
  __int64 v77; // rsi
  const char *v78; // rax
  __int64 v79; // r12
  __int64 v80; // rbx
  signed __int64 **v81; // rbx
  signed __int64 **v82; // r12
  signed __int64 **v83; // rdi
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // r8
  signed __int64 **v87; // r15
  _BYTE **v88; // r13
  char *v89; // rbx
  __int64 v90; // r15
  __int64 v91; // rax
  signed __int64 **v92; // rax
  unsigned int v93; // r12d
  signed __int64 **v94; // r13
  int v95; // r13d
  char *v96; // r12
  signed __int64 **v97; // r15
  __int64 *v98; // r14
  __int64 v99; // rax
  _QWORD *v100; // rbx
  __int64 v101; // r12
  __int64 v102; // r13
  _QWORD *v103; // r14
  _BYTE *v104; // r11
  __int64 v105; // r10
  int v106; // eax
  __int64 v107; // rdx
  __int64 *v108; // rdi
  _BYTE *v109; // rsi
  __int64 v110; // rax
  __int64 v111; // rdi
  unsigned __int64 v112; // rdx
  _QWORD *v113; // r12
  _QWORD *v114; // rbx
  __int64 v115; // rax
  __m128i *v116; // rcx
  __int64 *v117; // rdi
  __m128i *v118; // rsi
  int v119; // eax
  __int64 v120; // rcx
  signed __int64 **v121; // [rsp+8h] [rbp-1D8h]
  __m128i *v122; // [rsp+20h] [rbp-1C0h]
  __int64 v123; // [rsp+40h] [rbp-1A0h]
  __int64 *v124; // [rsp+40h] [rbp-1A0h]
  __int64 v125; // [rsp+48h] [rbp-198h]
  __int64 v126; // [rsp+48h] [rbp-198h]
  __int64 v127; // [rsp+48h] [rbp-198h]
  int v128; // [rsp+48h] [rbp-198h]
  __int64 v130; // [rsp+58h] [rbp-188h]
  signed __int64 **v131; // [rsp+58h] [rbp-188h]
  __int64 v132; // [rsp+58h] [rbp-188h]
  _BYTE *v133; // [rsp+58h] [rbp-188h]
  __m128i *v134; // [rsp+58h] [rbp-188h]
  __int64 v135; // [rsp+58h] [rbp-188h]
  __int64 v136; // [rsp+60h] [rbp-180h]
  __int64 v137; // [rsp+60h] [rbp-180h]
  __int64 v138; // [rsp+60h] [rbp-180h]
  __int64 v139; // [rsp+70h] [rbp-170h]
  unsigned __int64 v140; // [rsp+78h] [rbp-168h]
  __int64 v141; // [rsp+78h] [rbp-168h]
  __int64 v142; // [rsp+78h] [rbp-168h]
  __int64 v143; // [rsp+78h] [rbp-168h]
  __int64 *v144; // [rsp+78h] [rbp-168h]
  unsigned __int64 v145; // [rsp+80h] [rbp-160h]
  __int64 v146; // [rsp+88h] [rbp-158h]
  __int64 v147; // [rsp+90h] [rbp-150h] BYREF
  size_t v148; // [rsp+98h] [rbp-148h]
  __int64 v149; // [rsp+A8h] [rbp-138h] BYREF
  signed __int64 **v150; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v151; // [rsp+B8h] [rbp-128h]
  signed __int64 *v152; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v153; // [rsp+C8h] [rbp-118h]
  _BYTE v154[32]; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v155; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v156; // [rsp+F8h] [rbp-E8h]
  __int64 v157[2]; // [rsp+100h] [rbp-E0h] BYREF
  char v158; // [rsp+110h] [rbp-D0h] BYREF
  unsigned __int64 v159; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v160; // [rsp+128h] [rbp-B8h]
  _QWORD v161[2]; // [rsp+130h] [rbp-B0h] BYREF
  char v162; // [rsp+140h] [rbp-A0h]
  char v163; // [rsp+141h] [rbp-9Fh]
  char v164; // [rsp+148h] [rbp-98h] BYREF
  unsigned __int8 v165; // [rsp+150h] [rbp-90h]
  __m128i v166; // [rsp+160h] [rbp-80h] BYREF
  __int64 v167; // [rsp+170h] [rbp-70h] BYREF
  __int64 v168; // [rsp+178h] [rbp-68h]
  _BYTE v169[96]; // [rsp+180h] [rbp-60h] BYREF

  v147 = a2;
  v6 = "?*[{\\";
  v148 = a3;
  v145 = a4;
  v146 = a5;
  v166 = 0u;
  v167 = (__int64)v169;
  v168 = 0x100000000LL;
  v7 = sub_C934D0(&v147, "?*[{\\", 5, 0);
  v10 = v148;
  v11 = v147;
  v12 = v148;
  v166.m128i_i64[0] = v147;
  if ( v7 <= v148 )
    v12 = v7;
  v166.m128i_i64[1] = v12;
  if ( v7 != -1 )
  {
    v153 = 0x100000000LL;
    v13 = v148 - v12;
    v14 = v147 + v12 == 0;
    v15 = (signed __int64 **)(v147 + v12);
    v152 = (signed __int64 *)v154;
    v147 = (__int64)v15;
    v148 = v13;
    v140 = v145;
    if ( v14 )
    {
      v17 = v161;
      LOBYTE(v161[0]) = 0;
      v159 = (unsigned __int64)v161;
      v16 = (unsigned __int64)v161;
      v160 = 0;
    }
    else
    {
      v159 = (unsigned __int64)v161;
      sub_1099480((__int64 *)&v159, v15, v11 + v10);
      v16 = v159;
      v17 = (_QWORD *)(v159 + v160);
    }
    v156 = 0x100000000LL;
    v155 = (__int64)v157;
    v157[0] = (__int64)&v158;
    sub_10990A0(v157, (_BYTE *)v16, (__int64)v17);
    LODWORD(v156) = v156 + 1;
    if ( (_QWORD *)v159 != v161 )
    {
      v16 = v161[0] + 1LL;
      j_j___libc_free_0(v159, v161[0] + 1LL);
    }
    if ( !(_BYTE)v146 || !v13 || (v16 = 123, v21 = v15, (v22 = memchr(v15, 123, v13)) == 0) || v22 - (_BYTE *)v15 == -1 )
    {
      v23 = (unsigned int)v156;
      v165 = v165 & 0xFC | 2;
      v159 = (unsigned __int64)v161;
      v160 = 0x100000000LL;
      if ( !(_DWORD)v156 )
        goto LABEL_26;
      v16 = (unsigned __int64)&v155;
      sub_1099B30((__int64)&v159, (__int64)&v155, v18, v19, v20, (unsigned int)v156);
LABEL_116:
      v30 = (__int64 *)v155;
      v75 = (__int64 *)(v155 + 32LL * (unsigned int)v156);
      if ( (__int64 *)v155 == v75 )
      {
LABEL_27:
        if ( v30 != v157 )
          _libc_free(v30, v16);
        v31 = v159;
        v32 = v165 & 1;
        v33 = (unsigned int)(2 * v32);
        v165 = (2 * v32) | v165 & 0xFD;
        if ( (_BYTE)v32
          || (v61 = &v159,
              sub_1099B30((__int64)&v152, (__int64)&v159, v32, v33, v20, v23),
              v62 = v165,
              v31 = v159,
              v32 = v165 & 0xFD,
              v165 &= ~2u,
              (v62 & 1) != 0) )
        {
          v155 = v31 | 1;
        }
        else
        {
          v155 = 1;
          v63 = (_QWORD *)(v159 + 32LL * (unsigned int)v160);
          if ( v63 != (_QWORD *)v159 )
          {
            do
            {
              v63 -= 4;
              if ( (_QWORD *)*v63 != v63 + 2 )
              {
                v61 = (unsigned __int64 *)(v63[2] + 1LL);
                j_j___libc_free_0(*v63, v61);
              }
            }
            while ( v63 != (_QWORD *)v31 );
            v31 = v159;
          }
          if ( (_QWORD *)v31 != v161 )
            _libc_free(v31, v61);
        }
        v34 = v155 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v155 & 0xFFFFFFFFFFFFFFFELL) == 0 )
        {
          v35 = v152;
          v36 = &v152[4 * (unsigned int)v153];
          if ( v36 == v152 )
          {
LABEL_51:
            v6 = (const char *)a1;
            v47 = v168;
            v48 = a1[4].m128i_i8[8];
            *a1 = _mm_load_si128(&v166);
            a1[4].m128i_i8[8] = v48 & 0xFC | 2;
            a1[1].m128i_i64[0] = (__int64)a1[2].m128i_i64;
            a1[1].m128i_i64[1] = 0x100000000LL;
            if ( v47 )
            {
              v6 = (const char *)&v167;
              sub_109AD70((__int64)a1[1].m128i_i64, (__int64)&v167, v32, v33, v20, v23);
            }
            goto LABEL_62;
          }
          while ( 1 )
          {
            sub_109A760((__int64)&v159, *v35, v35[1], v33, v20, v23);
            v37 = v164 & 1;
            v164 = (2 * (v164 & 1)) | v164 & 0xFD;
            if ( v37 )
              break;
            v38 = (unsigned int)v168;
            v20 = (__int64)&v159;
            v33 = v167;
            v39 = (const void *)((unsigned int)v168 + 1LL);
            v40 = v168;
            if ( (unsigned __int64)v39 > HIDWORD(v168) )
            {
              if ( v167 > (unsigned __int64)&v159
                || (unsigned __int64)&v159 >= v167 + 40 * (unsigned __int64)(unsigned int)v168 )
              {
                sub_F31630((__int64)&v167, (unsigned __int64)v39, (unsigned int)v168, v167, (__int64)&v159, v23);
                v38 = (unsigned int)v168;
                v33 = v167;
                v20 = (__int64)&v159;
                v40 = v168;
              }
              else
              {
                v96 = (char *)&v159 - v167;
                sub_F31630((__int64)&v167, (unsigned __int64)v39, (unsigned int)v168, v167, (__int64)&v159, v23);
                v33 = v167;
                v38 = (unsigned int)v168;
                v20 = (__int64)&v96[v167];
                v40 = v168;
              }
            }
            v32 = 5 * v38;
            v41 = v33 + 8 * v32;
            if ( v41 )
            {
              *(_QWORD *)(v41 + 8) = 0;
              v23 = v41 + 16;
              *(_QWORD *)v41 = v41 + 16;
              v42 = *(_DWORD *)(v20 + 8);
              if ( v42 && v41 != v20 )
              {
                v142 = v20;
                sub_F30F50(v33 + 8 * v32, v42, v32, v33, v20, v23);
                v20 = v142;
                v76 = *(_QWORD *)v41;
                v23 = v41 + 16;
                v32 = *(_QWORD *)v142;
                v33 = *(_QWORD *)v142 + 80LL * *(unsigned int *)(v142 + 8);
                if ( *(_QWORD *)v142 != v33 )
                {
                  do
                  {
                    if ( v76 )
                    {
                      v77 = *(_QWORD *)v32;
                      *(_DWORD *)(v76 + 16) = 0;
                      *(_DWORD *)(v76 + 20) = 6;
                      *(_QWORD *)v76 = v77;
                      *(_QWORD *)(v76 + 8) = v76 + 24;
                      if ( *(_DWORD *)(v32 + 16) )
                      {
                        v123 = v33;
                        v125 = v23;
                        v130 = v20;
                        v139 = v32;
                        v143 = v76;
                        sub_1098FC0(v76 + 8, v32 + 8, v32, v33, v20, v23);
                        v33 = v123;
                        v23 = v125;
                        v20 = v130;
                        v32 = v139;
                        v76 = v143;
                      }
                      *(_DWORD *)(v76 + 72) = *(_DWORD *)(v32 + 72);
                    }
                    v32 += 80;
                    v76 += 80;
                  }
                  while ( v33 != v32 );
                }
                *(_DWORD *)(v41 + 8) = v42;
              }
              *(_QWORD *)(v41 + 24) = 0;
              v39 = (const void *)(v41 + 40);
              *(_QWORD *)(v41 + 16) = v41 + 40;
              *(_QWORD *)(v41 + 32) = 0;
              v43 = *(_QWORD *)(v20 + 24);
              if ( v43 )
              {
                v141 = v20;
                if ( v23 != v20 + 16 )
                {
                  sub_C8D290(v23, v39, v43, 1u, v20, v23);
                  v20 = v141;
                  v32 = *(_QWORD *)(v141 + 24);
                  if ( v32 )
                  {
                    v39 = *(const void **)(v141 + 16);
                    memcpy(*(void **)(v41 + 16), v39, v32);
                  }
                  *(_QWORD *)(v41 + 24) = v43;
                }
              }
              v40 = v168;
            }
            LODWORD(v168) = v40 + 1;
            if ( (v164 & 2) != 0 )
              sub_109A360(&v159, (__int64)v39);
            if ( (v164 & 1) != 0 )
            {
              if ( v159 )
                (*(void (__fastcall **)(unsigned __int64, const void *, __int64, __int64, __int64))(*(_QWORD *)v159 + 8LL))(
                  v159,
                  v39,
                  v32,
                  v33,
                  v20);
            }
            else
            {
              if ( (char *)v161[0] != &v164 )
                _libc_free(v161[0], v39);
              v44 = v159;
              v45 = (_QWORD *)(v159 + 80LL * (unsigned int)v160);
              if ( (_QWORD *)v159 != v45 )
              {
                do
                {
                  v45 -= 10;
                  v46 = (_QWORD *)v45[1];
                  if ( v46 != v45 + 3 )
                    _libc_free(v46, v39);
                }
                while ( (_QWORD *)v44 != v45 );
                v45 = (_QWORD *)v159;
              }
              if ( v45 != v161 )
                _libc_free(v45, v39);
            }
            v35 += 4;
            if ( v36 == v35 )
              goto LABEL_51;
          }
          v34 = v159 & 0xFFFFFFFFFFFFFFFELL;
        }
        v6 = (const char *)a1;
        a1[4].m128i_i8[8] |= 3u;
        a1->m128i_i64[0] = v34;
LABEL_62:
        v54 = v152;
        v55 = &v152[4 * (unsigned int)v153];
        if ( v152 != v55 )
        {
          do
          {
            v55 -= 4;
            if ( (signed __int64 *)*v55 != v55 + 2 )
            {
              v6 = (const char *)(v55[2] + 1);
              j_j___libc_free_0(*v55, v6);
            }
          }
          while ( v54 != v55 );
          v55 = v152;
        }
        if ( v55 != (signed __int64 *)v154 )
          _libc_free(v55, v6);
        goto LABEL_69;
      }
      do
      {
        v75 -= 4;
        if ( (__int64 *)*v75 != v75 + 2 )
        {
          v16 = v75[2] + 1;
          j_j___libc_free_0(*v75, v16);
        }
      }
      while ( v30 != v75 );
LABEL_26:
      v30 = (__int64 *)v155;
      goto LABEL_27;
    }
    v24 = 0;
    v25 = 0;
    v151 = 0;
    v150 = &v152;
    do
    {
      v26 = *((_BYTE *)v15 + v25);
      switch ( v26 )
      {
        case '[':
          v27 = v25 + 2;
          if ( v27 >= v13
            || (v21 = (signed __int64 **)((char *)v15 + v27),
                v16 = 93,
                (v28 = memchr((char *)v15 + v27, 93, v13 - v27)) == 0)
            || (v29 = v28 - (_BYTE *)v15, v29 == -1) )
          {
            v16 = (unsigned __int64)"invalid glob pattern, unmatched '['";
            sub_10995A0(&v149, "invalid glob pattern, unmatched '['", 22, v19, v20);
            v165 |= 3u;
            v159 = v149 & 0xFFFFFFFFFFFFFFFELL;
            goto LABEL_140;
          }
          v25 = v29 + 1;
          break;
        case '{':
          if ( v24 )
          {
            v163 = 1;
            v78 = "nested brace expansions are not supported";
            goto LABEL_137;
          }
          v64 = v151;
          if ( HIDWORD(v151) <= (unsigned int)v151 )
          {
            v84 = (__int64)&v152;
            v21 = (signed __int64 **)&v150;
            v136 = sub_C8D7D0((__int64)&v150, (__int64)&v152, 0, 0x40u, &v159, v23);
            v20 = (unsigned __int64)(unsigned int)v151 << 6;
            v85 = v20 + v136;
            if ( v20 + v136 )
            {
              v18 = v85 + 32;
              v84 = 0x200000000LL;
              *(_OWORD *)v85 = 0;
              v86 = (unsigned int)v151;
              *(_QWORD *)(v85 + 16) = v85 + 32;
              *(_QWORD *)(v85 + 24) = 0x200000000LL;
              v20 = v86 << 6;
              *(_OWORD *)(v85 + 32) = 0;
              *(_OWORD *)(v85 + 48) = 0;
            }
            v87 = (signed __int64 **)((char *)v150 + v20);
            if ( v150 != (signed __int64 **)((char *)v150 + v20) )
            {
              v23 = v136;
              v126 = v25;
              v88 = (_BYTE **)(v150 + 4);
              v89 = (char *)v150 + v20;
              v131 = v15;
              v90 = v136;
              while ( 1 )
              {
                if ( v90 )
                {
                  *(_QWORD *)v90 = *(v88 - 4);
                  v91 = (__int64)*(v88 - 3);
                  *(_DWORD *)(v90 + 24) = 0;
                  *(_QWORD *)(v90 + 8) = v91;
                  v92 = (signed __int64 **)(v90 + 32);
                  *(_QWORD *)(v90 + 16) = v90 + 32;
                  *(_DWORD *)(v90 + 28) = 2;
                  v93 = *((_DWORD *)v88 - 2);
                  if ( v93 )
                  {
                    v21 = (signed __int64 **)(v90 + 16);
                    v18 = (__int64)(v88 - 2);
                    if ( (_BYTE **)(v90 + 16) != v88 - 2 )
                    {
                      v18 = (__int64)*(v88 - 2);
                      if ( (_BYTE **)v18 == v88 )
                      {
                        v84 = (__int64)v88;
                        v18 = 16LL * v93;
                        if ( v93 <= 2
                          || (sub_C8D5F0((__int64)v21, (const void *)(v90 + 32), v93, 0x10u, v20, v23),
                              v92 = *(signed __int64 ***)(v90 + 16),
                              v84 = (__int64)*(v88 - 2),
                              (v18 = 16LL * *((unsigned int *)v88 - 2)) != 0) )
                        {
                          v21 = v92;
                          memcpy(v92, (const void *)v84, v18);
                        }
                        *(_DWORD *)(v90 + 24) = v93;
                        *((_DWORD *)v88 - 2) = 0;
                      }
                      else
                      {
                        *(_QWORD *)(v90 + 16) = v18;
                        *(_DWORD *)(v90 + 24) = *((_DWORD *)v88 - 2);
                        *(_DWORD *)(v90 + 28) = *((_DWORD *)v88 - 1);
                        *(v88 - 2) = v88;
                        *((_DWORD *)v88 - 1) = 0;
                        *((_DWORD *)v88 - 2) = 0;
                      }
                    }
                  }
                }
                v90 += 64;
                if ( v89 == (char *)(v88 + 4) )
                  break;
                v88 += 8;
              }
              v94 = v150;
              v15 = v131;
              v25 = v126;
              v20 = (unsigned __int64)(unsigned int)v151 << 6;
              v87 = (signed __int64 **)((char *)v150 + v20);
              if ( v150 != (signed __int64 **)((char *)v150 + v20) )
              {
                do
                {
                  v87 -= 8;
                  v21 = (signed __int64 **)v87[2];
                  if ( v21 != v87 + 4 )
                    _libc_free(v21, v84);
                }
                while ( v94 != v87 );
                v87 = v150;
              }
            }
            v95 = v159;
            if ( v87 != &v152 )
            {
              v21 = v87;
              _libc_free(v87, v84);
            }
            v16 = v136;
            HIDWORD(v151) = v95;
            v150 = (signed __int64 **)v136;
            LODWORD(v151) = v151 + 1;
            v24 = v136 + ((unsigned __int64)(unsigned int)v151 << 6) - 64;
          }
          else
          {
            v19 = (unsigned __int64)v150;
            v18 = (__int64)&v150[8 * (unsigned __int64)(unsigned int)v151];
            if ( v18 )
            {
              *(_QWORD *)(v18 + 16) = v18 + 32;
              *(_OWORD *)v18 = 0;
              v19 = (unsigned __int64)v150;
              *(_QWORD *)(v18 + 24) = 0x200000000LL;
              v64 = v151;
              *(_OWORD *)(v18 + 32) = 0;
              *(_OWORD *)(v18 + 48) = 0;
            }
            v65 = (unsigned int)(v64 + 1);
            LODWORD(v151) = v65;
            v24 = v19 + (v65 << 6) - 64;
          }
          v5 = v25 + 1;
          *(_QWORD *)v24 = v25++;
          break;
        case ',':
          v20 = v25 + 1;
          if ( v24 )
          {
            v66 = v13;
            v19 = *(unsigned int *)(v24 + 28);
            if ( v5 <= v13 )
              v66 = v5;
            v67 = v25 - v5;
            v23 = (__int64)v15 + v66;
            if ( v67 > v13 - v66 )
              v67 = v13 - v66;
            v68 = *(unsigned int *)(v24 + 24);
            v18 = v68 + 1;
            if ( v68 + 1 > v19 )
            {
              v16 = v24 + 32;
              v21 = (signed __int64 **)(v24 + 16);
              v132 = v23;
              v137 = v20;
              sub_C8D5F0(v24 + 16, (const void *)(v24 + 32), v18, 0x10u, v20, v23);
              v68 = *(unsigned int *)(v24 + 24);
              v23 = v132;
              v20 = v137;
            }
            v69 = (__int64 *)(*(_QWORD *)(v24 + 16) + 16 * v68);
            v5 = v20;
            v69[1] = v67;
            v25 = v20;
            *v69 = v23;
            ++*(_DWORD *)(v24 + 24);
          }
          else
          {
            ++v25;
          }
          break;
        case '}':
          if ( v24 )
          {
            v70 = *(unsigned int *)(v24 + 24);
            if ( !(_DWORD)v70 )
            {
              v163 = 1;
              v78 = "empty or singleton brace expansions are not supported";
              goto LABEL_137;
            }
            v71 = v13;
            if ( v5 <= v13 )
              v71 = v5;
            v20 = v25 - v5;
            v72 = v13 - v71;
            v23 = (__int64)v15 + v71;
            v19 = *(unsigned int *)(v24 + 28);
            if ( v25 - v5 > v72 )
              v20 = v72;
            v18 = v70 + 1;
            if ( v70 + 1 > v19 )
            {
              v16 = v24 + 32;
              v21 = (signed __int64 **)(v24 + 16);
              v135 = v20;
              v138 = v23;
              sub_C8D5F0(v24 + 16, (const void *)(v24 + 32), v18, 0x10u, v20, v23);
              v70 = *(unsigned int *)(v24 + 24);
              v20 = v135;
              v23 = v138;
            }
            ++v25;
            v73 = (__int64 *)(*(_QWORD *)(v24 + 16) + 16 * v70);
            *v73 = v23;
            v73[1] = v20;
            v74 = v25 - *(_QWORD *)v24;
            ++*(_DWORD *)(v24 + 24);
            *(_QWORD *)(v24 + 8) = v74;
            v24 = 0;
          }
          else
          {
            ++v25;
          }
          break;
        default:
          v18 = v25 + 1;
          if ( v26 == 92 )
          {
            if ( v13 == v18 )
            {
              v16 = (unsigned __int64)"invalid glob pattern, stray '\\'";
              sub_1099530(&v149, "invalid glob pattern, stray '\\'", 22, v19, v20);
              v165 |= 3u;
              v159 = v149 & 0xFFFFFFFFFFFFFFFELL;
              goto LABEL_140;
            }
            v25 += 2;
          }
          else
          {
            ++v25;
          }
          break;
      }
    }
    while ( v13 != v25 );
    if ( v24 )
    {
      v163 = 1;
      v78 = "incomplete brace expansion";
LABEL_137:
      v159 = (unsigned __int64)v78;
      v162 = 3;
      v79 = sub_2241E50(v21, v16, v18, v19, v20);
      v80 = sub_22077B0(64);
      if ( v80 )
      {
        v16 = (unsigned __int64)&v159;
        sub_C63EB0(v80, (__int64)&v159, 22, v79);
      }
      v165 |= 3u;
      v159 = v80 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_140;
    }
    v16 = (unsigned __int64)v150;
    v121 = v150;
    v97 = &v150[8 * (unsigned __int64)(unsigned int)v151];
    if ( v150 == v97 )
    {
      if ( v140 )
        goto LABEL_204;
    }
    else
    {
      v16 = *((unsigned int *)v150 + 6);
      v21 = v150 + 8;
      v19 = 1;
      while ( 1 )
      {
        v19 *= v16;
        if ( v97 == v21 )
          break;
        v16 = *((unsigned int *)v21 + 6);
        v18 = (*((unsigned int *)v21 + 6) * (unsigned __int128)v19) >> 64;
        v21 += 8;
        if ( v18 )
        {
          v19 = -1;
          break;
        }
      }
      if ( v140 >= v19 )
      {
        do
        {
          v16 = (unsigned __int64)&v159;
          v159 = (unsigned __int64)v161;
          v160 = 0x100000000LL;
          sub_109A010((__int64)&v155, (__int64)&v159, v18, v19, v20, v23);
          v98 = *(v97 - 6);
          v23 = v159;
          v124 = &v98[2 * *((unsigned int *)v97 - 10)];
          if ( v98 == v124 )
          {
            v113 = (_QWORD *)(v159 + 32LL * (unsigned int)v160);
          }
          else
          {
            v144 = *(v97 - 6);
            v99 = (unsigned int)v160;
            do
            {
              v100 = (_QWORD *)(v23 + 32 * v99);
              v101 = *v144;
              v102 = v144[1];
              if ( v100 == (_QWORD *)v23 )
              {
                v113 = (_QWORD *)v23;
              }
              else
              {
                v103 = (_QWORD *)v23;
                do
                {
                  v104 = (_BYTE *)*v103;
                  v105 = v103[1];
                  v106 = v156;
                  if ( HIDWORD(v156) <= (unsigned int)v156 )
                  {
                    v127 = v103[1];
                    v133 = (_BYTE *)*v103;
                    v115 = sub_C8D7D0((__int64)&v155, (__int64)v157, 0, 0x20u, (unsigned __int64 *)&v149, v23);
                    v116 = (__m128i *)v115;
                    v117 = (__int64 *)(v115 + 32LL * (unsigned int)v156);
                    if ( v117 )
                    {
                      v122 = (__m128i *)v115;
                      *v117 = (__int64)(v117 + 2);
                      sub_1099480(v117, v133, (__int64)&v133[v127]);
                      v116 = v122;
                    }
                    v118 = v116;
                    v134 = v116;
                    sub_1099A70((__int64)&v155, v116);
                    v119 = v149;
                    v120 = (__int64)v134;
                    if ( (__int64 *)v155 != v157 )
                    {
                      v128 = v149;
                      _libc_free(v155, v118);
                      v119 = v128;
                      v120 = (__int64)v134;
                    }
                    HIDWORD(v156) = v119;
                    v155 = v120;
                    LODWORD(v156) = v156 + 1;
                    v111 = v120 + 32LL * (unsigned int)v156 - 32;
                  }
                  else
                  {
                    v107 = v155;
                    v108 = (__int64 *)(v155 + 32LL * (unsigned int)v156);
                    if ( v108 )
                    {
                      v109 = (_BYTE *)*v103;
                      *v108 = (__int64)(v108 + 2);
                      sub_1099480(v108, v109, (__int64)&v104[v105]);
                      v106 = v156;
                      v107 = v155;
                    }
                    v110 = (unsigned int)(v106 + 1);
                    LODWORD(v156) = v110;
                    v111 = v107 + 32 * v110 - 32;
                  }
                  v16 = (unsigned __int64)*(v97 - 8);
                  v112 = *(_QWORD *)(v111 + 8) - v16;
                  if ( (unsigned __int64)*(v97 - 7) <= v112 )
                    v112 = (unsigned __int64)*(v97 - 7);
                  if ( v16 > *(_QWORD *)(v111 + 8) )
                    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
                  v103 += 4;
                  sub_2241130(v111, v16, v112, v101, v102);
                }
                while ( v100 != v103 );
                v99 = (unsigned int)v160;
                v23 = v159;
                v18 = 32LL * (unsigned int)v160;
                v113 = (_QWORD *)(v159 + v18);
              }
              v144 += 2;
              v19 = (unsigned __int64)v144;
            }
            while ( v124 != v144 );
          }
          if ( (_QWORD *)v23 != v113 )
          {
            v114 = (_QWORD *)v23;
            do
            {
              v113 -= 4;
              if ( (_QWORD *)*v113 != v113 + 2 )
              {
                v16 = v113[2] + 1LL;
                j_j___libc_free_0(*v113, v16);
              }
            }
            while ( v113 != v114 );
            v113 = (_QWORD *)v159;
          }
          if ( v113 != v161 )
            _libc_free(v113, v16);
          v97 -= 8;
        }
        while ( v121 != v97 );
LABEL_204:
        v20 = (unsigned int)v156;
        v165 = v165 & 0xFC | 2;
        v159 = (unsigned __int64)v161;
        v160 = 0x100000000LL;
        if ( (_DWORD)v156 )
        {
          v16 = (unsigned __int64)&v155;
          sub_1099B30((__int64)&v159, (__int64)&v155, v18, v19, (unsigned int)v156, v23);
        }
LABEL_140:
        v81 = v150;
        v82 = &v150[8 * (unsigned __int64)(unsigned int)v151];
        if ( v150 != v82 )
        {
          do
          {
            v82 -= 8;
            v83 = (signed __int64 **)v82[2];
            if ( v83 != v82 + 4 )
              _libc_free(v83, v16);
          }
          while ( v81 != v82 );
          v82 = v150;
        }
        if ( v82 != &v152 )
          _libc_free(v82, v16);
        goto LABEL_116;
      }
    }
    v163 = 1;
    v78 = "too many brace expansions";
    goto LABEL_137;
  }
  v49 = _mm_load_si128(&v166);
  v50 = v168;
  v51 = a1[4].m128i_i8[8];
  a1[1].m128i_i64[1] = 0x100000000LL;
  *a1 = v49;
  a1[4].m128i_i8[8] = v51 & 0xFC | 2;
  a1[1].m128i_i64[0] = (__int64)a1[2].m128i_i64;
  if ( !v50 )
  {
LABEL_56:
    v52 = (_BYTE *)v167;
    goto LABEL_57;
  }
  v6 = (const char *)&v167;
  sub_109AD70((__int64)a1[1].m128i_i64, (__int64)&v167, v10, (__int64)a1, v8, v9);
LABEL_69:
  v52 = (_BYTE *)v167;
  v56 = (_BYTE *)(v167 + 40LL * (unsigned int)v168);
  if ( (_BYTE *)v167 != v56 )
  {
    do
    {
      v56 -= 40;
      v57 = (_BYTE *)*((_QWORD *)v56 + 2);
      if ( v57 != v56 + 40 )
        _libc_free(v57, v6);
      v58 = *(_BYTE **)v56;
      v59 = *(_QWORD *)v56 + 80LL * *((unsigned int *)v56 + 2);
      if ( *(_QWORD *)v56 != v59 )
      {
        do
        {
          v59 -= 80;
          v60 = *(_QWORD *)(v59 + 8);
          if ( v60 != v59 + 24 )
            _libc_free(v60, v6);
        }
        while ( v58 != (_BYTE *)v59 );
        v58 = *(_BYTE **)v56;
      }
      if ( v58 != v56 + 16 )
        _libc_free(v58, v6);
    }
    while ( v52 != v56 );
    goto LABEL_56;
  }
LABEL_57:
  if ( v52 != v169 )
    _libc_free(v52, v6);
  return a1;
}
