// Function: sub_18D9200
// Address: 0x18d9200
//
__int64 __fastcall sub_18D9200(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  double v15; // xmm4_8
  double v16; // xmm5_8
  double v17; // xmm4_8
  double v18; // xmm5_8
  int v19; // eax
  int v20; // edx
  _QWORD *v21; // r13
  __int64 v22; // r11
  _QWORD *v23; // r10
  _QWORD *v24; // r15
  _QWORD *v25; // rbx
  _QWORD *v26; // r12
  _QWORD *ii; // r13
  _QWORD *v28; // rax
  _QWORD *v29; // rbx
  __int64 v30; // r13
  unsigned __int64 v31; // rax
  int v32; // r12d
  __int64 i; // rdi
  int v34; // edi
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // r15
  unsigned __int8 v38; // al
  int v39; // eax
  bool v40; // bl
  __int64 v41; // rax
  void *v42; // rdi
  void *v43; // rsi
  __int64 *v44; // rdx
  __int64 v45; // r12
  __int64 *v46; // rax
  __int64 *v47; // rcx
  int v48; // edi
  __int64 j; // rdi
  int v50; // edi
  __int64 v51; // rax
  __int64 v52; // r14
  unsigned __int8 v53; // al
  __int64 v54; // rdi
  int v55; // eax
  _QWORD *v56; // rdx
  _QWORD *v57; // rax
  _QWORD *v58; // rdx
  _QWORD *v59; // r8
  _QWORD *v60; // r15
  __int64 v61; // rbx
  __int64 v62; // r12
  _QWORD *m; // r14
  _QWORD *v64; // rax
  __int64 v65; // rdi
  __int64 v66; // r8
  __int64 v67; // rbx
  _QWORD *v68; // rax
  __int64 v69; // rdi
  int v70; // eax
  __int64 n; // rbx
  _QWORD *v72; // rax
  int v73; // eax
  double v74; // xmm4_8
  double v75; // xmm5_8
  _QWORD *v76; // r9
  __int64 v77; // r8
  __int64 v78; // rax
  __int64 v79; // rdi
  unsigned __int8 v80; // al
  __int64 v81; // r10
  __int64 v82; // rsi
  __int64 v83; // rax
  __int64 v84; // rdi
  __int64 *v85; // r10
  void *v86; // rdi
  void *v87; // rsi
  __int64 v88; // r14
  unsigned int v89; // eax
  unsigned int v90; // eax
  double v91; // xmm4_8
  double v92; // xmm5_8
  __int64 v93; // rax
  void *v94; // rdi
  void *v95; // r8
  unsigned int v96; // eax
  unsigned int v97; // eax
  __int64 v98; // rbx
  double v99; // xmm4_8
  double v100; // xmm5_8
  __int64 v101; // rbx
  unsigned int v102; // eax
  unsigned int v103; // eax
  __int64 v104; // rax
  __int64 *v105; // rdx
  __int64 *v106; // rax
  __int64 *v107; // rcx
  int v108; // edi
  __int64 k; // rdi
  int v110; // edi
  __int64 v111; // rax
  __int64 v112; // rbx
  unsigned __int8 v113; // al
  __int64 v114; // rdi
  char *v115; // rcx
  __int64 v116; // rdx
  __int64 *v117; // rax
  __int64 *v118; // rsi
  int v119; // eax
  __int64 v120; // rdx
  __int64 v121; // r11
  _QWORD *v122; // rax
  __int64 v123; // r15
  __int64 v124; // r11
  _QWORD *v125; // rax
  __int64 v126; // r15
  __int64 *v127; // rax
  __int64 v128; // r15
  __int64 v129; // rax
  __int64 *v130; // rax
  __int64 v131; // r15
  __int64 v132; // rax
  _QWORD *v133; // [rsp+38h] [rbp-108h]
  bool v134; // [rsp+38h] [rbp-108h]
  __int64 v135; // [rsp+40h] [rbp-100h]
  _QWORD *v136; // [rsp+40h] [rbp-100h]
  __int64 v137; // [rsp+40h] [rbp-100h]
  __int64 v138; // [rsp+40h] [rbp-100h]
  _QWORD *v139; // [rsp+48h] [rbp-F8h]
  __int64 v140; // [rsp+48h] [rbp-F8h]
  __int64 v141; // [rsp+48h] [rbp-F8h]
  __int64 *v142; // [rsp+48h] [rbp-F8h]
  __int64 *v143; // [rsp+48h] [rbp-F8h]
  __int64 v144; // [rsp+48h] [rbp-F8h]
  __int64 v145; // [rsp+50h] [rbp-F0h]
  _QWORD *v146; // [rsp+50h] [rbp-F0h]
  __int64 v147; // [rsp+50h] [rbp-F0h]
  __int64 v148; // [rsp+50h] [rbp-F0h]
  __int64 v149; // [rsp+50h] [rbp-F0h]
  __int64 v150; // [rsp+50h] [rbp-F0h]
  __int64 v151; // [rsp+50h] [rbp-F0h]
  __int64 *v152; // [rsp+50h] [rbp-F0h]
  __int64 *v153; // [rsp+50h] [rbp-F0h]
  __int64 v155; // [rsp+58h] [rbp-E8h]
  int v157; // [rsp+68h] [rbp-D8h]
  unsigned __int8 v158; // [rsp+6Fh] [rbp-D1h]
  __int64 *v159; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v160; // [rsp+78h] [rbp-C8h]
  void *s; // [rsp+80h] [rbp-C0h]
  __int128 v162; // [rsp+88h] [rbp-B8h]
  _BYTE v163[40]; // [rsp+98h] [rbp-A8h] BYREF
  __int64 v164; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v165; // [rsp+C8h] [rbp-78h]
  void *v166; // [rsp+D0h] [rbp-70h]
  __int128 v167; // [rsp+D8h] [rbp-68h]
  _BYTE v168[88]; // [rsp+E8h] [rbp-58h] BYREF

  v158 = byte_4F99CA8[0];
  if ( byte_4F99CA8[0] )
  {
    v10 = a1;
    v158 = *(_BYTE *)(a1 + 344);
    if ( v158 )
    {
      v12 = *(__int64 **)(a1 + 8);
      *(_BYTE *)(a1 + 153) = 0;
      v13 = *v12;
      v14 = v12[1];
      if ( v13 == v14 )
LABEL_250:
        BUG();
      while ( *(_UNKNOWN **)v13 != &unk_4F96DB4 )
      {
        v13 += 16;
        if ( v14 == v13 )
          goto LABEL_250;
      }
      *(_QWORD *)(a1 + 160) = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
                                            *(_QWORD *)(v13 + 8),
                                            &unk_4F96DB4)
                                        + 160);
      sub_18CED20(a1, a2, a3, a4, a5, a6, v15, v16, a9, a10);
      v19 = *(_DWORD *)(a1 + 348);
      LOBYTE(v20) = v19;
      if ( (v19 & 0x7F000) == 0 )
      {
LABEL_9:
        if ( (v19 & 0xB) != 0 && (v19 & 0x10) != 0 )
        {
          while ( (unsigned __int8)sub_18D7250(v10, a2, a3, a4, a5, a6, v17, v18, a9, a10) )
            ;
          v20 = *(_DWORD *)(v10 + 348);
        }
        if ( (v20 & 0x60) == 0 )
          return *(unsigned __int8 *)(v10 + 153);
        if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) != 15 )
          return *(unsigned __int8 *)(v10 + 153);
        v30 = *(_QWORD *)(a2 + 80);
        v159 = 0;
        v160 = (__int64)v163;
        s = v163;
        v165 = (__int64)v168;
        v166 = v168;
        *(_QWORD *)&v162 = 4;
        DWORD2(v162) = 0;
        v164 = 0;
        *(_QWORD *)&v167 = 4;
        DWORD2(v167) = 0;
        if ( v30 == a2 + 72 )
          return *(unsigned __int8 *)(v10 + 153);
        v135 = v10;
        v145 = v10 + 160;
LABEL_44:
        if ( !v30 )
          BUG();
        v31 = *(_QWORD *)(v30 + 16) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v31 )
          BUG();
        if ( *(_BYTE *)(v31 - 8) != 25 )
          goto LABEL_167;
        v32 = v31 - 24;
        for ( i = *(_QWORD *)(v31 - 24LL * (*(_DWORD *)(v31 - 4) & 0xFFFFFFF) - 24);
              ;
              i = *(_QWORD *)(v37 - 24LL * (*(_DWORD *)(v37 + 20) & 0xFFFFFFF)) )
        {
          v36 = sub_1649C60(i);
          v34 = 23;
          v37 = v36;
          v38 = *(_BYTE *)(v36 + 16);
          if ( v38 <= 0x17u )
            goto LABEL_49;
          if ( v38 != 78 )
            break;
          v34 = 21;
          if ( *(_BYTE *)(*(_QWORD *)(v37 - 24) + 16LL) )
            goto LABEL_49;
          v39 = sub_1438F00(*(_QWORD *)(v37 - 24));
          v35 = sub_1439C90(v39);
          if ( !v35 )
          {
LABEL_55:
            v40 = v35;
            sub_18DCE30(0, v37, v30 - 24, v32, (unsigned int)&v159, (unsigned int)&v164, v145);
            v41 = DWORD1(v162);
            v42 = s;
            v43 = (void *)v160;
            if ( DWORD1(v162) - DWORD2(v162) != 1 )
              goto LABEL_173;
            if ( s != (void *)v160 )
              v41 = (unsigned int)v162;
            v44 = (__int64 *)((char *)s + 8 * v41);
            v45 = *(_QWORD *)s;
            if ( s != v44 )
            {
              v46 = (__int64 *)s;
              while ( 1 )
              {
                v45 = *v46;
                v47 = v46;
                if ( (unsigned __int64)*v46 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v44 == ++v46 )
                {
                  v45 = v47[1];
                  break;
                }
              }
            }
            if ( v45 )
            {
              if ( *(_BYTE *)(v45 + 16) == 78 )
              {
                v48 = 21;
                if ( !*(_BYTE *)(*(_QWORD *)(v45 - 24) + 16LL) )
                  v48 = sub_1438F00(*(_QWORD *)(v45 - 24));
                if ( sub_1439C80(v48) )
                {
                  for ( j = *(_QWORD *)(v45 - 24LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF));
                        ;
                        j = *(_QWORD *)(v52 - 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF)) )
                  {
                    v51 = sub_1649C60(j);
                    v50 = 23;
                    v52 = v51;
                    v53 = *(_BYTE *)(v51 + 16);
                    if ( v53 > 0x17u )
                    {
                      if ( v53 == 78 )
                      {
                        v50 = 21;
                        if ( !*(_BYTE *)(*(_QWORD *)(v52 - 24) + 16LL) )
                          v50 = sub_1438F00(*(_QWORD *)(v52 - 24));
                      }
                      else
                      {
                        v50 = 2 * (v53 != 29) + 21;
                      }
                    }
                    if ( !(unsigned __int8)sub_1439C90(v50) )
                      break;
                  }
                  v42 = s;
                  v43 = (void *)v160;
                  if ( v37 != v52 )
                    v45 = 0;
                  goto LABEL_174;
                }
                v42 = s;
                v43 = (void *)v160;
              }
LABEL_173:
              v45 = 0;
            }
LABEL_174:
            v159 = (__int64 *)((char *)v159 + 1);
            if ( v43 == v42 )
            {
LABEL_179:
              *(_QWORD *)((char *)&v162 + 4) = 0;
            }
            else
            {
              v102 = 4 * (DWORD1(v162) - DWORD2(v162));
              if ( v102 < 0x20 )
                v102 = 32;
              if ( (unsigned int)v162 <= v102 )
              {
                memset(v42, -1, 8LL * (unsigned int)v162);
                goto LABEL_179;
              }
              sub_16CC920((__int64)&v159);
            }
            ++v164;
            if ( v166 == (void *)v165 )
            {
LABEL_185:
              *(_QWORD *)((char *)&v167 + 4) = 0;
            }
            else
            {
              v103 = 4 * (DWORD1(v167) - DWORD2(v167));
              if ( v103 < 0x20 )
                v103 = 32;
              if ( (unsigned int)v167 <= v103 )
              {
                memset(v166, -1, 8LL * (unsigned int)v167);
                goto LABEL_185;
              }
              sub_16CC920((__int64)&v164);
            }
            if ( v45 )
            {
              sub_18DCE30(2, v37, *(_QWORD *)(v45 + 40), v45, (unsigned int)&v159, (unsigned int)&v164, v145);
              v104 = DWORD1(v162);
              v87 = (void *)v160;
              v86 = s;
              if ( DWORD1(v162) - DWORD2(v162) != 1 )
                goto LABEL_135;
              if ( s != (void *)v160 )
                v104 = (unsigned int)v162;
              v105 = (__int64 *)((char *)s + 8 * v104);
              v88 = *(_QWORD *)s;
              if ( s != v105 )
              {
                v106 = (__int64 *)s;
                while ( 1 )
                {
                  v88 = *v106;
                  v107 = v106;
                  if ( (unsigned __int64)*v106 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v105 == ++v106 )
                  {
                    v88 = v107[1];
                    break;
                  }
                }
              }
              if ( v88 )
              {
                if ( *(_BYTE *)(v88 + 16) == 78 )
                {
                  v108 = 21;
                  if ( !*(_BYTE *)(*(_QWORD *)(v88 - 24) + 16LL) )
                    v108 = sub_1438F00(*(_QWORD *)(v88 - 24));
                  if ( sub_1439C70(v108) )
                  {
                    v134 = v40;
                    for ( k = *(_QWORD *)(v88 - 24LL * (*(_DWORD *)(v88 + 20) & 0xFFFFFFF));
                          ;
                          k = *(_QWORD *)(v112 - 24LL * (*(_DWORD *)(v112 + 20) & 0xFFFFFFF)) )
                    {
                      v111 = sub_1649C60(k);
                      v110 = 23;
                      v112 = v111;
                      v113 = *(_BYTE *)(v111 + 16);
                      if ( v113 > 0x17u )
                      {
                        if ( v113 == 78 )
                        {
                          v110 = 21;
                          if ( !*(_BYTE *)(*(_QWORD *)(v112 - 24) + 16LL) )
                            v110 = sub_1438F00(*(_QWORD *)(v112 - 24));
                        }
                        else
                        {
                          v110 = 2 * (v113 != 29) + 21;
                        }
                      }
                      if ( !(unsigned __int8)sub_1439C90(v110) )
                        break;
                    }
                    v120 = v112;
                    v40 = v134;
                    v86 = s;
                    v87 = (void *)v160;
                    if ( v37 != v120 )
                      v88 = 0;
                    goto LABEL_136;
                  }
                  v86 = s;
                  v87 = (void *)v160;
                }
LABEL_135:
                v88 = 0;
              }
LABEL_136:
              v159 = (__int64 *)((char *)v159 + 1);
              if ( v87 == v86 )
              {
LABEL_141:
                *(_QWORD *)((char *)&v162 + 4) = 0;
              }
              else
              {
                v89 = 4 * (DWORD1(v162) - DWORD2(v162));
                if ( v89 < 0x20 )
                  v89 = 32;
                if ( (unsigned int)v162 <= v89 )
                {
                  memset(v86, -1, 8LL * (unsigned int)v162);
                  goto LABEL_141;
                }
                sub_16CC920((__int64)&v159);
              }
              ++v164;
              if ( v166 == (void *)v165 )
              {
LABEL_147:
                *(_QWORD *)((char *)&v167 + 4) = 0;
              }
              else
              {
                v90 = 4 * (DWORD1(v167) - DWORD2(v167));
                if ( v90 < 0x20 )
                  v90 = 32;
                if ( (unsigned int)v167 <= v90 )
                {
                  memset(v166, -1, 8LL * (unsigned int)v167);
                  goto LABEL_147;
                }
                sub_16CC920((__int64)&v164);
              }
              if ( v88 )
              {
                sub_18DCE30(2, v37, *(_QWORD *)(v88 + 40), v88, (unsigned int)&v159, (unsigned int)&v164, v145);
                v93 = DWORD1(v162);
                v94 = s;
                v95 = (void *)v160;
                if ( DWORD1(v162) - DWORD2(v162) == 1 )
                {
                  if ( s != (void *)v160 )
                    v93 = (unsigned int)v162;
                  v115 = (char *)s + 8 * v93;
                  v116 = *(_QWORD *)s;
                  if ( s != v115 )
                  {
                    v117 = (__int64 *)s;
                    while ( 1 )
                    {
                      v116 = *v117;
                      v118 = v117;
                      if ( (unsigned __int64)*v117 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v115 == (char *)++v117 )
                      {
                        v116 = v118[1];
                        break;
                      }
                    }
                  }
                  if ( v116 )
                  {
                    if ( *(_BYTE *)(v116 + 16) == 78 && v37 == v116 )
                    {
                      v40 = v158;
                      if ( !*(_BYTE *)(*(_QWORD *)(v37 - 24) + 16LL) )
                      {
                        v119 = sub_1438F00(*(_QWORD *)(v37 - 24));
                        v94 = s;
                        v95 = (void *)v160;
                        v40 = (unsigned int)(v119 - 21) <= 1;
                      }
                    }
                  }
                }
                v159 = (__int64 *)((char *)v159 + 1);
                if ( v95 == v94 )
                {
LABEL_155:
                  *(_QWORD *)((char *)&v162 + 4) = 0;
                }
                else
                {
                  v96 = 4 * (DWORD1(v162) - DWORD2(v162));
                  if ( v96 < 0x20 )
                    v96 = 32;
                  if ( (unsigned int)v162 <= v96 )
                  {
                    memset(v94, -1, 8LL * (unsigned int)v162);
                    goto LABEL_155;
                  }
                  sub_16CC920((__int64)&v159);
                }
                ++v164;
                if ( v166 == (void *)v165 )
                {
LABEL_161:
                  *(_QWORD *)((char *)&v167 + 4) = 0;
                }
                else
                {
                  v97 = 4 * (DWORD1(v167) - DWORD2(v167));
                  if ( v97 < 0x20 )
                    v97 = 32;
                  if ( (unsigned int)v167 <= v97 )
                  {
                    memset(v166, -1, 8LL * (unsigned int)v167);
                    goto LABEL_161;
                  }
                  sub_16CC920((__int64)&v164);
                }
                if ( v40 )
                {
                  *(_BYTE *)(v135 + 153) = 1;
                  v98 = *(_QWORD *)(v88 - 24LL * (*(_DWORD *)(v88 + 20) & 0xFFFFFFF));
                  if ( *(_QWORD *)(v88 + 8) )
                  {
                    sub_164D160(
                      v88,
                      *(_QWORD *)(v88 - 24LL * (*(_DWORD *)(v88 + 20) & 0xFFFFFFF)),
                      a3,
                      a4,
                      a5,
                      a6,
                      v91,
                      v92,
                      a9,
                      a10);
                    sub_15F20C0((_QWORD *)v88);
                  }
                  else
                  {
                    sub_15F20C0((_QWORD *)v88);
                    sub_1AEB370(v98, 0);
                  }
                  v101 = *(_QWORD *)(v45 - 24LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF));
                  if ( *(_QWORD *)(v45 + 8) )
                  {
                    sub_164D160(
                      v45,
                      *(_QWORD *)(v45 - 24LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF)),
                      a3,
                      a4,
                      a5,
                      a6,
                      v99,
                      v100,
                      a9,
                      a10);
                    sub_15F20C0((_QWORD *)v45);
                  }
                  else
                  {
                    sub_15F20C0((_QWORD *)v45);
                    sub_1AEB370(v101, 0);
                  }
                }
              }
            }
LABEL_167:
            v30 = *(_QWORD *)(v30 + 8);
            if ( a2 + 72 == v30 )
            {
              v10 = v135;
              if ( v166 != (void *)v165 )
                _libc_free((unsigned __int64)v166);
              if ( (void *)v160 != s )
                _libc_free((unsigned __int64)s);
              return *(unsigned __int8 *)(v10 + 153);
            }
            goto LABEL_44;
          }
LABEL_50:
          ;
        }
        v34 = 2 * (v38 != 29) + 21;
LABEL_49:
        v35 = sub_1439C90(v34);
        if ( !v35 )
          goto LABEL_55;
        goto LABEL_50;
      }
      v21 = *(_QWORD **)(a2 + 80);
      v22 = a2 + 72;
      if ( (_QWORD *)(a2 + 72) != v21 )
      {
        if ( !v21 )
          BUG();
        v23 = *(_QWORD **)(a2 + 80);
        v24 = (_QWORD *)v21[3];
        if ( v24 == v21 + 2 )
        {
          do
          {
            v23 = (_QWORD *)v23[1];
            if ( (_QWORD *)v22 == v23 )
              goto LABEL_31;
            if ( !v23 )
              BUG();
            v24 = (_QWORD *)v23[3];
          }
          while ( v24 == v23 + 2 );
        }
        if ( (_QWORD *)v22 == v23 )
        {
LABEL_31:
          while ( 1 )
          {
            v29 = (_QWORD *)v21[3];
            if ( v29 != v21 + 2 )
              break;
            v21 = (_QWORD *)v21[1];
            if ( (_QWORD *)v22 == v21 )
              goto LABEL_33;
            if ( !v21 )
              BUG();
          }
          if ( v21 != (_QWORD *)v22 )
          {
            v155 = v10;
            v62 = v22;
LABEL_95:
            for ( m = (_QWORD *)v29[1]; ; m = (_QWORD *)v21[3] )
            {
              v64 = v21 - 3;
              if ( !v21 )
                v64 = 0;
              if ( m != v64 + 5 )
              {
                if ( *((_BYTE *)v29 - 8) == 78 )
                {
                  v65 = *(v29 - 6);
                  if ( !*(_BYTE *)(v65 + 16) && (unsigned int)sub_1438F00(v65) == 18 )
                  {
LABEL_109:
                    v66 = v29[-3 * (*((_DWORD *)v29 - 1) & 0xFFFFFFF) - 3];
                    if ( *(_BYTE *)(v66 + 16) == 53 )
                    {
                      v67 = *(_QWORD *)(v66 + 8);
                      if ( v67 )
                      {
                        while ( 1 )
                        {
                          v147 = v66;
                          v68 = sub_1648700(v67);
                          if ( *((_BYTE *)v68 + 16) != 78 )
                            break;
                          v69 = *(v68 - 3);
                          if ( *(_BYTE *)(v69 + 16) )
                            break;
                          v70 = sub_1438F00(v69);
                          v66 = v147;
                          if ( v70 > 14 )
                          {
                            if ( v70 != 18 )
                              break;
                          }
                          else if ( v70 <= 12 )
                          {
                            break;
                          }
                          v67 = *(_QWORD *)(v67 + 8);
                          if ( !v67 )
                            goto LABEL_116;
                        }
                      }
                      else
                      {
LABEL_116:
                        *(_BYTE *)(v155 + 153) = 1;
                        for ( n = *(_QWORD *)(v66 + 8); n; v66 = v149 )
                        {
                          v140 = v66;
                          v72 = sub_1648700(n);
                          n = *(_QWORD *)(n + 8);
                          v148 = (__int64)v72;
                          v73 = sub_1438F00(*(v72 - 3));
                          v76 = (_QWORD *)v148;
                          v77 = v140;
                          if ( v73 <= 14 )
                          {
                            sub_164D160(
                              v148,
                              *(_QWORD *)(v148 + 24 * (1LL - (*(_DWORD *)(v148 + 20) & 0xFFFFFFF))),
                              a3,
                              a4,
                              a5,
                              a6,
                              v74,
                              v75,
                              a9,
                              a10);
                            v77 = v140;
                            v76 = (_QWORD *)v148;
                          }
                          v149 = v77;
                          sub_15F20C0(v76);
                        }
                        sub_15F20C0((_QWORD *)v66);
                      }
                    }
                  }
                }
                if ( (_QWORD *)v62 == v21 )
                  goto LABEL_106;
                v29 = m;
                goto LABEL_95;
              }
              v21 = (_QWORD *)v21[1];
              if ( (_QWORD *)v62 == v21 )
                break;
              if ( !v21 )
                BUG();
            }
            if ( *((_BYTE *)v29 - 8) == 78 )
            {
              v114 = *(v29 - 6);
              if ( !*(_BYTE *)(v114 + 16) && (unsigned int)sub_1438F00(v114) == 18 )
                goto LABEL_109;
            }
LABEL_106:
            v10 = v155;
            v19 = *(_DWORD *)(v155 + 348);
            goto LABEL_34;
          }
        }
        else
        {
          v25 = v23;
          v26 = (_QWORD *)(a2 + 72);
          while ( 1 )
          {
            for ( ii = (_QWORD *)v24[1]; ; ii = (_QWORD *)v25[3] )
            {
              v28 = v25 - 3;
              if ( !v25 )
                v28 = 0;
              if ( ii != v28 + 5 )
                break;
              v25 = (_QWORD *)v25[1];
              if ( v26 == v25 )
                break;
              if ( !v25 )
                BUG();
            }
            if ( *((_BYTE *)v24 - 8) == 78 )
            {
              v54 = *(v24 - 6);
              if ( !*(_BYTE *)(v54 + 16) )
              {
                v55 = sub_1438F00(v54);
                v157 = v55;
                if ( v55 == 15 || v55 == 12 )
                {
                  v133 = v24 - 3;
                  if ( v55 == 15 && !*(v24 - 2) )
                  {
                    sub_15F20C0(v133);
                    goto LABEL_36;
                  }
                  v56 = ii;
                  v57 = v25;
                  while ( 1 )
                  {
                    if ( v26 != v57 )
                    {
                      if ( !v57 )
                        BUG();
                      v59 = (_QWORD *)v57[3];
                      if ( v56 != v59 )
                        break;
                    }
                    v57 = (_QWORD *)(*v57 & 0xFFFFFFFFFFFFFFF8LL);
                    v58 = v57 - 3;
                    if ( !v57 )
                      v58 = 0;
                    v56 = v58 + 5;
                  }
                  if ( (_QWORD *)(*v56 & 0xFFFFFFFFFFFFFFF8LL) != v59 )
                  {
                    v146 = v25;
                    v136 = (_QWORD *)v57[3];
                    v139 = v24;
                    v60 = (_QWORD *)(*v56 & 0xFFFFFFFFFFFFFFF8LL);
                    while ( 1 )
                    {
                      v61 = (*v60 & 0xFFFFFFFFFFFFFFF8LL) - 24;
                      if ( (*v60 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                        v61 = 0;
                      switch ( (unsigned int)sub_14399D0(v61) )
                      {
                        case 7u:
                        case 0x14u:
                        case 0x17u:
                        case 0x18u:
                          goto LABEL_123;
                        case 0xCu:
                        case 0xFu:
                          v83 = *(_QWORD *)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF));
                          v159 = (__int64 *)v133[-3 * (*((_DWORD *)v139 - 1) & 0xFFFFFFF)];
                          v84 = *(_QWORD *)(a1 + 160);
                          v165 = -1;
                          v164 = v83;
                          v166 = 0;
                          v167 = 0u;
                          v160 = -1;
                          s = 0;
                          v162 = 0u;
                          v80 = sub_134CB50(v84, (__int64)&v159, (__int64)&v164);
                          if ( v80 <= 2u )
                            goto LABEL_121;
                          if ( v80 != 3 )
                            goto LABEL_123;
                          v85 = (__int64 *)v61;
                          v25 = v146;
                          *(_BYTE *)(a1 + 153) = 1;
                          if ( v157 == 12 )
                          {
                            v124 = *(_QWORD *)(a1 + 256);
                            if ( !v124 )
                            {
                              v143 = v85;
                              v152 = **(__int64 ***)(a1 + 232);
                              v127 = (__int64 *)sub_1643330(v152);
                              v159 = (__int64 *)sub_1646BA0(v127, 0);
                              v128 = sub_1644EA0(v159, &v159, 1, 0);
                              v164 = 0;
                              v164 = sub_1563AB0(&v164, v152, -1, 30);
                              v129 = sub_1632080(*(_QWORD *)(a1 + 232), (__int64)"objc_retain", 11, v128, v164);
                              v85 = v143;
                              v124 = v129;
                              *(_QWORD *)(a1 + 256) = v129;
                            }
                            v159 = v85;
                            LOWORD(v166) = 257;
                            v142 = v85;
                            v138 = v124;
                            v151 = *(_QWORD *)(*(_QWORD *)v124 + 24LL);
                            v125 = sub_1648AB0(72, 2u, 0);
                            v85 = v142;
                            v126 = (__int64)v125;
                            if ( v125 )
                            {
                              sub_15F1EA0(
                                (__int64)v125,
                                **(_QWORD **)(v151 + 16),
                                54,
                                (__int64)(v125 - 6),
                                2,
                                (__int64)v133);
                              *(_QWORD *)(v126 + 56) = 0;
                              sub_15F5B40(v126, v151, v138, (__int64 *)&v159, 1, (__int64)&v164, 0, 0);
                              v85 = v142;
                            }
                            *(_WORD *)(v126 + 18) = *(_WORD *)(v126 + 18) & 0xFFFC | 1;
                          }
                          v82 = (__int64)v85;
                          goto LABEL_129;
                        case 0xDu:
                        case 0xEu:
                          v78 = *(_QWORD *)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF));
                          v159 = (__int64 *)v133[-3 * (*((_DWORD *)v139 - 1) & 0xFFFFFFF)];
                          v79 = *(_QWORD *)(a1 + 160);
                          v165 = -1;
                          v164 = v78;
                          v166 = 0;
                          v167 = 0u;
                          v160 = -1;
                          s = 0;
                          v162 = 0u;
                          v80 = sub_134CB50(v79, (__int64)&v159, (__int64)&v164);
                          if ( v80 <= 2u )
                          {
LABEL_121:
                            if ( v80 )
                              goto LABEL_122;
                          }
                          else if ( v80 == 3 )
                          {
                            v81 = v61;
                            v25 = v146;
                            *(_BYTE *)(a1 + 153) = 1;
                            if ( v157 == 12 )
                            {
                              v121 = *(_QWORD *)(a1 + 256);
                              if ( !v121 )
                              {
                                v144 = v81;
                                v153 = **(__int64 ***)(a1 + 232);
                                v130 = (__int64 *)sub_1643330(v153);
                                v159 = (__int64 *)sub_1646BA0(v130, 0);
                                v131 = sub_1644EA0(v159, &v159, 1, 0);
                                v164 = 0;
                                v164 = sub_1563AB0(&v164, v153, -1, 30);
                                v132 = sub_1632080(*(_QWORD *)(a1 + 232), (__int64)"objc_retain", 11, v131, v164);
                                v81 = v144;
                                v121 = v132;
                                *(_QWORD *)(a1 + 256) = v132;
                              }
                              v159 = (__int64 *)v81;
                              LOWORD(v166) = 257;
                              v141 = v81;
                              v137 = v121;
                              v150 = *(_QWORD *)(*(_QWORD *)v121 + 24LL);
                              v122 = sub_1648AB0(72, 2u, 0);
                              v81 = v141;
                              v123 = (__int64)v122;
                              if ( v122 )
                              {
                                sub_15F1EA0(
                                  (__int64)v122,
                                  **(_QWORD **)(v150 + 16),
                                  54,
                                  (__int64)(v122 - 6),
                                  2,
                                  (__int64)v133);
                                *(_QWORD *)(v123 + 56) = 0;
                                sub_15F5B40(v123, v150, v137, (__int64 *)&v159, 1, (__int64)&v164, 0, 0);
                                v81 = v141;
                              }
                              *(_WORD *)(v123 + 18) = *(_WORD *)(v123 + 18) & 0xFFFC | 1;
                            }
                            v82 = *(_QWORD *)(v81 + 24 * (1LL - (*(_DWORD *)(v81 + 20) & 0xFFFFFFF)));
LABEL_129:
                            sub_164D160((__int64)v133, v82, a3, a4, a5, a6, v17, v18, a9, a10);
                            sub_15F20C0(v133);
                            goto LABEL_36;
                          }
LABEL_123:
                          v60 = (_QWORD *)(*v60 & 0xFFFFFFFFFFFFFFF8LL);
                          if ( v60 == v136 )
                          {
LABEL_122:
                            v25 = v146;
                            goto LABEL_36;
                          }
                          break;
                        default:
                          goto LABEL_122;
                      }
                    }
                  }
                }
              }
            }
LABEL_36:
            if ( v26 == v25 )
              break;
            v24 = ii;
          }
          v22 = a2 + 72;
          v10 = a1;
          v21 = *(_QWORD **)(a2 + 80);
          if ( v21 != (_QWORD *)(a2 + 72) )
          {
            if ( !v21 )
              BUG();
            goto LABEL_31;
          }
        }
LABEL_33:
        v19 = *(_DWORD *)(v10 + 348);
      }
LABEL_34:
      LOBYTE(v20) = v19;
      goto LABEL_9;
    }
  }
  return v158;
}
