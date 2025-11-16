// Function: sub_1D96F70
// Address: 0x1d96f70
//
__int64 __fastcall sub_1D96F70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        int a6,
        char a7,
        char a8,
        char a9,
        char a10)
{
  __int64 result; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  void *v16; // r10
  __int64 v17; // rax
  size_t v18; // r9
  __int64 v19; // rax
  unsigned __int64 v20; // r15
  __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // rax
  unsigned int v24; // ebx
  unsigned int v25; // eax
  __int64 v26; // r13
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 *v29; // r15
  unsigned int v30; // esi
  char *v31; // rbx
  __int64 v32; // r14
  __int64 *v33; // r13
  size_t v34; // r12
  __int64 v35; // r8
  __int64 *v36; // rax
  unsigned __int64 v37; // rsi
  __int64 v38; // rcx
  char *v39; // r13
  char *v40; // r12
  char *v41; // r15
  unsigned __int64 *v42; // rsi
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rbx
  int v45; // ecx
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // rax
  __int64 i; // rbx
  _QWORD *v50; // r13
  unsigned __int64 *v51; // rcx
  unsigned __int64 v52; // rdx
  __int64 v53; // rsi
  unsigned __int64 v54; // r12
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  __int64 v57; // rax
  __int64 j; // r12
  __int64 v59; // rdi
  __int64 (*v60)(); // rax
  char *v61; // rbx
  __int64 v62; // rbx
  __int64 v63; // rdi
  __int64 v64; // rbx
  __int64 v65; // rdi
  unsigned __int64 v66; // rbx
  __int16 v67; // ax
  __int64 k; // rbx
  char v69; // al
  __int64 v70; // rax
  _BYTE *v71; // rdi
  int v72; // r9d
  __int64 v73; // rbx
  __int64 v74; // r8
  __int64 v75; // r14
  __int64 v76; // rbx
  unsigned int v77; // r12d
  _DWORD *v78; // rax
  _DWORD *v79; // rdx
  unsigned int *v80; // r12
  unsigned int *v81; // r14
  unsigned int *v82; // rdi
  __int64 v83; // rdx
  _DWORD *v84; // rax
  _DWORD *v85; // rcx
  __int64 v86; // rax
  int *v87; // rdi
  __int64 v88; // rsi
  __int64 v89; // rcx
  __int64 v90; // rcx
  int v91; // ebx
  __int64 v92; // rsi
  __int64 v93; // r15
  int v94; // eax
  __int64 v95; // rax
  int *v96; // rsi
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // rdx
  __int64 v101; // r15
  int v102; // eax
  __int64 v103; // r14
  __int64 v104; // r13
  __int64 v105; // rdi
  __int64 (*v106)(); // rax
  __int64 v107; // rdi
  __int64 (*v108)(); // rax
  _BYTE *v109; // rax
  _BYTE **v110; // [rsp+18h] [rbp-228h]
  _BYTE **v111; // [rsp+20h] [rbp-220h]
  __int64 v112; // [rsp+28h] [rbp-218h]
  __int64 v113; // [rsp+30h] [rbp-210h]
  int v114; // [rsp+38h] [rbp-208h]
  __int64 *v115; // [rsp+40h] [rbp-200h]
  size_t n; // [rsp+48h] [rbp-1F8h]
  size_t na; // [rsp+48h] [rbp-1F8h]
  _QWORD *src; // [rsp+50h] [rbp-1F0h]
  void *srca; // [rsp+50h] [rbp-1F0h]
  char v120; // [rsp+5Bh] [rbp-1E5h]
  unsigned __int64 v121; // [rsp+60h] [rbp-1E0h]
  __int64 v122; // [rsp+68h] [rbp-1D8h]
  __int64 v123; // [rsp+68h] [rbp-1D8h]
  size_t v124; // [rsp+68h] [rbp-1D8h]
  __int64 v127; // [rsp+80h] [rbp-1C0h]
  __int64 v128; // [rsp+88h] [rbp-1B8h]
  unsigned int v129; // [rsp+9Ch] [rbp-1A4h] BYREF
  unsigned int *v130; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v131; // [rsp+A8h] [rbp-198h]
  _BYTE v132[16]; // [rsp+B0h] [rbp-190h] BYREF
  _BYTE *v133; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v134; // [rsp+C8h] [rbp-178h]
  _BYTE v135[24]; // [rsp+D0h] [rbp-170h] BYREF
  int v136; // [rsp+E8h] [rbp-158h] BYREF
  __int64 v137; // [rsp+F0h] [rbp-150h]
  int *v138; // [rsp+F8h] [rbp-148h]
  int *v139; // [rsp+100h] [rbp-140h]
  __int64 v140; // [rsp+108h] [rbp-138h]
  _BYTE *v141; // [rsp+110h] [rbp-130h] BYREF
  __int64 v142; // [rsp+118h] [rbp-128h]
  _BYTE v143[24]; // [rsp+120h] [rbp-120h] BYREF
  int v144; // [rsp+138h] [rbp-108h] BYREF
  __int64 v145; // [rsp+140h] [rbp-100h]
  int *v146; // [rsp+148h] [rbp-F8h]
  int *v147; // [rsp+150h] [rbp-F0h]
  __int64 v148; // [rsp+158h] [rbp-E8h]
  _BYTE *v149; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v150; // [rsp+168h] [rbp-D8h]
  _BYTE v151[208]; // [rsp+170h] [rbp-D0h] BYREF

  v127 = a3;
  v128 = a4;
  if ( (*(_BYTE *)a3 & 1) != 0
    || (result = *(_BYTE *)a4 & 1, (*(_BYTE *)a4 & 1) != 0)
    || (v12 = *(_QWORD *)(a3 + 16), (unsigned int)((__int64)(*(_QWORD *)(v12 + 72) - *(_QWORD *)(v12 + 64)) >> 3) > 1)
    || (v13 = *(_QWORD *)(a4 + 16), (unsigned int)((__int64)(*(_QWORD *)(v13 + 72) - *(_QWORD *)(v13 + 64)) >> 3) > 1) )
  {
    *(_BYTE *)a2 &= ~4u;
    *(_BYTE *)a3 &= ~4u;
    *(_BYTE *)v128 &= ~4u;
    return 0;
  }
  if ( !*(_BYTE *)(v12 + 181) )
  {
    v120 = *(_BYTE *)(v13 + 181);
    if ( !v120 )
    {
      v16 = *(void **)(a2 + 40);
      v17 = *(unsigned int *)(a2 + 48);
      v149 = v151;
      v150 = 0x400000000LL;
      v19 = 40 * v17;
      v18 = v19;
      v20 = 0xCCCCCCCCCCCCCCCDLL * (v19 >> 3);
      if ( (unsigned __int64)v19 > 0xA0 )
      {
        na = v19;
        srca = v16;
        sub_16CD150((__int64)&v149, v151, 0xCCCCCCCCCCCCCCCDLL * (v19 >> 3), 40, (int)&v149, v19);
        v16 = srca;
        v18 = na;
        v71 = &v149[40 * (unsigned int)v150];
      }
      else
      {
        if ( !v19 )
        {
LABEL_9:
          LODWORD(v150) = v19 + v20;
          (*(void (__fastcall **)(_QWORD, _BYTE **))(**(_QWORD **)(a1 + 544) + 624LL))(*(_QWORD *)(a1 + 544), &v149);
          v111 = (_BYTE **)(a2 + 40);
          if ( a8 == 1 || (v110 = &v149, !a7) )
          {
            if ( a8 || a7 || (v21 = *(unsigned int *)(v128 + 4), *(_DWORD *)(v127 + 4) <= (unsigned int)v21) )
            {
              v21 = v127;
              v111 = &v149;
              v110 = (_BYTE **)(a2 + 40);
              v70 = v128;
              v128 = v127;
              v127 = v70;
            }
            else
            {
              v110 = &v149;
            }
          }
          *(_DWORD *)(a2 + 4) -= (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64, _BYTE **))(**(_QWORD **)(a1 + 544) + 280LL))(
                                   *(_QWORD *)(a1 + 544),
                                   *(_QWORD *)(a2 + 16),
                                   0,
                                   v21,
                                   &v149);
          n = *(_QWORD *)(v128 + 16);
          v22 = *(_QWORD **)(v127 + 16);
          *(_DWORD *)(a1 + 592) = 0;
          src = v22;
          v122 = a1 + 576;
          v23 = *(_QWORD *)(a1 + 552);
          *(_QWORD *)(a1 + 576) = v23;
          v24 = *(_DWORD *)(v23 + 16);
          v25 = *(_DWORD *)(a1 + 640);
          if ( v24 < v25 >> 2 || v24 > v25 )
          {
            _libc_free(*(_QWORD *)(a1 + 632));
            v26 = (__int64)_libc_calloc(v24, 1u);
            if ( !v26 )
            {
              if ( v24 )
                sub_16BD1C0("Allocation failed", 1u);
              else
                v26 = sub_13A3880(1u);
            }
            *(_QWORD *)(a1 + 632) = v26;
            *(_DWORD *)(a1 + 640) = v24;
          }
          if ( (**(_BYTE **)(**(_QWORD **)(a1 + 568) + 352LL) & 4) != 0 )
          {
            sub_1DC2B10(v122, n);
            sub_1DC2B10(v122, src);
          }
          v27 = sub_1DD6100(n);
          v28 = sub_1DD6100(src);
          *(_DWORD *)(v128 + 4) -= a5;
          *(_DWORD *)(v127 + 4) -= a5;
          if ( a5 )
          {
            v29 = (__int64 *)(n + 24);
            v30 = 0;
            do
            {
              if ( (__int64 *)v27 == v29 )
                goto LABEL_28;
              v30 -= ((unsigned __int16)(**(_WORD **)(v27 + 16) - 12) < 2u) - 1;
              if ( (*(_BYTE *)v27 & 4) == 0 && (*(_BYTE *)(v27 + 46) & 8) != 0 )
              {
                do
                  v27 = *(_QWORD *)(v27 + 8);
                while ( (*(_BYTE *)(v27 + 46) & 8) != 0 );
              }
              v27 = *(_QWORD *)(v27 + 8);
            }
            while ( a5 > v30 );
            v29 = (__int64 *)v27;
            while ( 1 )
            {
LABEL_28:
              if ( !v28 )
                BUG();
              if ( (*(_BYTE *)v28 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v28 + 46) & 8) != 0 )
                  v28 = *(_QWORD *)(v28 + 8);
              }
              v28 = *(_QWORD *)(v28 + 8);
              v31 = (char *)(src + 3);
              if ( (_QWORD *)v28 == src + 3 )
                break;
              if ( (unsigned __int16)(**(_WORD **)(v28 + 16) - 12) > 1u && !--a5 )
                goto LABEL_33;
            }
          }
          else
          {
            v29 = (__int64 *)v27;
LABEL_33:
            v31 = (char *)v28;
          }
          if ( (**(_BYTE **)(**(_QWORD **)(a1 + 568) + 352LL) & 4) != 0 )
          {
            if ( *(__int64 **)(n + 32) == v29 )
            {
              v34 = n + 16;
LABEL_57:
              v39 = (char *)(src + 2);
              if ( (char *)src[4] != v31 )
              {
                v124 = v34;
                v40 = (char *)src[4];
                do
                {
                  v41 = v40;
                  v40 = (char *)*((_QWORD *)v40 + 1);
                  sub_1DD5BC0(v39, v41);
                  v42 = (unsigned __int64 *)*((_QWORD *)v41 + 1);
                  v43 = *(_QWORD *)v41 & 0xFFFFFFFFFFFFFFF8LL;
                  *v42 = v43 | *v42 & 7;
                  *(_QWORD *)(v43 + 8) = v42;
                  *(_QWORD *)v41 &= 7uLL;
                  *((_QWORD *)v41 + 1) = 0;
                  sub_1DD5C20(v39, v41);
                }
                while ( v40 != v31 );
                v34 = v124;
              }
              *(_DWORD *)(v128 + 4) -= (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 544)
                                                                                         + 280LL))(
                                         *(_QWORD *)(a1 + 544),
                                         *(_QWORD *)(v128 + 16),
                                         0);
              v113 = n + 24;
              if ( a6 )
              {
                v44 = n + 24;
                v45 = 0;
                do
                {
                  v46 = (_QWORD *)(*(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL);
                  v47 = v46;
                  if ( !v46 )
                    BUG();
                  v44 = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL;
                  v48 = *v46;
                  if ( (v48 & 4) == 0 && (*((_BYTE *)v47 + 46) & 4) != 0 )
                  {
                    for ( i = v48; ; i = *(_QWORD *)v44 )
                    {
                      v44 = i & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (*(_BYTE *)(v44 + 46) & 4) == 0 )
                        break;
                    }
                  }
                  v45 -= ((unsigned __int16)(**(_WORD **)(v44 + 16) - 12) < 2u) - 1;
                }
                while ( a6 != v45 );
                while ( v113 != v44 )
                {
                  v50 = (_QWORD *)v44;
                  v44 = *(_QWORD *)(v44 + 8);
                  sub_1DD5BC0(v34, v50);
                  v51 = (unsigned __int64 *)v50[1];
                  v52 = *v50 & 0xFFFFFFFFFFFFFFF8LL;
                  *v51 = v52 | *v51 & 7;
                  *(_QWORD *)(v52 + 8) = v51;
                  *v50 &= 7uLL;
                  v50[1] = 0;
                  sub_1DD5C20(v34, v50);
                }
              }
              v53 = *(_QWORD *)(v127 + 16);
              v54 = v53 + 24;
              if ( a9 )
              {
                *(_DWORD *)(v127 + 4) -= (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 544)
                                                                                            + 280LL))(
                                           *(_QWORD *)(a1 + 544),
                                           v53,
                                           0);
LABEL_74:
                if ( a6 )
                {
                  do
                  {
                    v55 = (_QWORD *)(*(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL);
                    v56 = v55;
                    if ( !v55 )
                      BUG();
                    v54 = *(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL;
                    v57 = *v55;
                    if ( (v57 & 4) == 0 && (*((_BYTE *)v56 + 46) & 4) != 0 )
                    {
                      for ( j = v57; ; j = *(_QWORD *)v54 )
                      {
                        v54 = j & 0xFFFFFFFFFFFFFFF8LL;
                        if ( (*(_BYTE *)(v54 + 46) & 4) == 0 )
                          break;
                      }
                    }
                  }
                  while ( (unsigned __int16)(**(_WORD **)(v54 + 16) - 12) <= 1u || --a6 );
                }
                v144 = 0;
                v133 = v135;
                v138 = &v136;
                v139 = &v136;
                v141 = v143;
                v146 = &v144;
                v147 = &v144;
                v142 = 0x400000000LL;
                v59 = *(_QWORD *)(a1 + 544);
                v134 = 0x400000000LL;
                v145 = 0;
                v148 = 0;
                v136 = 0;
                v137 = 0;
                v140 = 0;
                v60 = *(__int64 (**)())(*(_QWORD *)v59 + 352LL);
                if ( v60 == sub_1D918A0 )
                  goto LABEL_84;
                if ( !((unsigned __int8 (__fastcall *)(__int64, size_t, _QWORD *))v60)(v59, n, src) )
                  goto LABEL_84;
                v73 = src[4];
                if ( v73 == v54 )
                  goto LABEL_84;
                v121 = v54;
LABEL_124:
                if ( (unsigned __int16)(**(_WORD **)(v73 + 16) - 12) <= 1u )
                  goto LABEL_125;
                v74 = *(_QWORD *)(v73 + 32);
                v130 = (unsigned int *)v132;
                v131 = 0x400000000LL;
                v75 = v74 + 40LL * *(unsigned int *)(v73 + 40);
                if ( v74 == v75 )
                  goto LABEL_125;
                v112 = v73;
                v76 = v74;
                while ( 1 )
                {
                  if ( !*(_BYTE *)v76 )
                  {
                    v77 = *(_DWORD *)(v76 + 8);
                    if ( v77 )
                    {
                      if ( (*(_BYTE *)(v76 + 3) & 0x10) != 0 )
                      {
                        v97 = (unsigned int)v131;
                        if ( (unsigned int)v131 >= HIDWORD(v131) )
                        {
                          sub_16CD150((__int64)&v130, v132, 0, 4, v74, v72);
                          v97 = (unsigned int)v131;
                        }
                        v130[v97] = v77;
                        LODWORD(v131) = v131 + 1;
                      }
                      else
                      {
                        if ( !v140 )
                        {
                          v78 = v133;
                          v79 = &v133[4 * (unsigned int)v134];
                          if ( v133 != (_BYTE *)v79 )
                          {
                            while ( v77 != *v78 )
                            {
                              if ( v79 == ++v78 )
                                goto LABEL_177;
                            }
                            if ( v78 != v79 )
                              goto LABEL_141;
                          }
LABEL_177:
                          v98 = *(_QWORD *)(a1 + 552);
                          if ( !v98 )
                            BUG();
                          v99 = *(unsigned int *)(*(_QWORD *)(v98 + 8) + 24LL * v77 + 4);
                          v100 = *(_QWORD *)(v98 + 56) + 2 * v99;
LABEL_180:
                          v101 = v100;
                          while ( v101 )
                          {
                            v101 += 2;
                            v129 = (unsigned __int16)v77;
                            sub_1D041C0((__int64)&v141, &v129, v100, v99, v74);
                            v102 = *(unsigned __int16 *)(v101 - 2);
                            v100 = 0;
                            v77 += v102;
                            if ( !(_WORD)v102 )
                              goto LABEL_180;
                          }
                          goto LABEL_141;
                        }
                        v95 = v137;
                        if ( !v137 )
                          goto LABEL_177;
                        v96 = &v136;
                        do
                        {
                          if ( v77 > *(_DWORD *)(v95 + 32) )
                          {
                            v95 = *(_QWORD *)(v95 + 24);
                          }
                          else
                          {
                            v96 = (int *)v95;
                            v95 = *(_QWORD *)(v95 + 16);
                          }
                        }
                        while ( v95 );
                        if ( v96 == &v136 || v77 < v96[8] )
                          goto LABEL_177;
                      }
                    }
                  }
LABEL_141:
                  v76 += 40;
                  if ( v75 == v76 )
                  {
                    v80 = v130;
                    v73 = v112;
                    v81 = &v130[(unsigned int)v131];
                    v82 = v81;
                    if ( v130 != v81 )
                    {
                      while ( 1 )
                      {
                        v83 = *v80;
                        if ( v148 )
                        {
                          v86 = v145;
                          if ( v145 )
                          {
                            v87 = &v144;
                            do
                            {
                              while ( 1 )
                              {
                                v88 = *(_QWORD *)(v86 + 16);
                                v89 = *(_QWORD *)(v86 + 24);
                                if ( (unsigned int)v83 <= *(_DWORD *)(v86 + 32) )
                                  break;
                                v86 = *(_QWORD *)(v86 + 24);
                                if ( !v89 )
                                  goto LABEL_158;
                              }
                              v87 = (int *)v86;
                              v86 = *(_QWORD *)(v86 + 16);
                            }
                            while ( v88 );
LABEL_158:
                            if ( v87 != &v144 && (unsigned int)v83 >= v87[8] )
                              goto LABEL_149;
                          }
                        }
                        else
                        {
                          v84 = v141;
                          v85 = &v141[4 * (unsigned int)v142];
                          if ( v141 != (_BYTE *)v85 )
                          {
                            while ( (_DWORD)v83 != *v84 )
                            {
                              if ( v85 == ++v84 )
                                goto LABEL_160;
                            }
                            if ( v84 != v85 )
                              goto LABEL_149;
                          }
                        }
LABEL_160:
                        v90 = *(_QWORD *)(a1 + 552);
                        if ( !v90 )
                          BUG();
                        v91 = *v80;
                        v92 = *(_QWORD *)(v90 + 56)
                            + 2LL * *(unsigned int *)(*(_QWORD *)(v90 + 8) + 24LL * (unsigned int)v83 + 4);
                        while ( 1 )
                        {
                          v93 = v92;
                          if ( !v92 )
                            break;
                          while ( 1 )
                          {
                            v93 += 2;
                            v129 = (unsigned __int16)v91;
                            sub_1D041C0((__int64)&v133, &v129, v83, v90, v74);
                            v94 = *(unsigned __int16 *)(v93 - 2);
                            v92 = 0;
                            v83 = (unsigned int)(v94 + v91);
                            if ( !(_WORD)v94 )
                              break;
                            v91 += v94;
                            if ( !v93 )
                              goto LABEL_149;
                          }
                        }
LABEL_149:
                        if ( v81 == ++v80 )
                        {
                          v73 = v112;
                          v82 = v130;
                          break;
                        }
                      }
                    }
                    if ( v82 != (unsigned int *)v132 )
                      _libc_free((unsigned __int64)v82);
LABEL_125:
                    if ( (*(_BYTE *)v73 & 4) == 0 )
                    {
                      while ( (*(_BYTE *)(v73 + 46) & 8) != 0 )
                        v73 = *(_QWORD *)(v73 + 8);
                    }
                    v73 = *(_QWORD *)(v73 + 8);
                    if ( v121 != v73 )
                      goto LABEL_124;
                    v54 = v121;
LABEL_84:
                    sub_1D959A0(a1, v128, v113, (__int64)v110, (__int64)&v133);
                    v61 = (char *)(src + 3);
                    if ( src + 3 != (_QWORD *)(src[3] & 0xFFFFFFFFFFFFFFF8LL) && v61 == (char *)v54 )
                    {
                      v103 = sub_1DD5EE0(n);
                      v104 = sub_1DD5EE0(src);
                      if ( v103 != v113 )
                      {
                        v105 = *(_QWORD *)(a1 + 544);
                        v106 = *(__int64 (**)())(*(_QWORD *)v105 + 656LL);
                        if ( v106 != sub_1D918C0 )
                          v120 = ((__int64 (__fastcall *)(__int64, __int64))v106)(v105, v103);
                      }
                      if ( v61 != (char *)v104 )
                      {
                        v107 = *(_QWORD *)(a1 + 544);
                        v108 = *(__int64 (**)())(*(_QWORD *)v107 + 656LL);
                        if ( (v108 == sub_1D918C0
                           || !((unsigned __int8 (__fastcall *)(__int64, __int64))v108)(v107, v104))
                          && (v120 || (*(_BYTE *)v127 & 0x10) == 0) )
                        {
                          v109 = (_BYTE *)(*(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL);
                          if ( !v109 )
                            BUG();
                          v54 = *(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL;
                          if ( (*v109 & 4) == 0 && (v109[46] & 4) != 0 )
                          {
                            do
                              v54 = *(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL;
                            while ( (*(_BYTE *)(v54 + 46) & 4) != 0 );
                          }
                        }
                      }
                    }
                    sub_1D959A0(a1, v127, v54, (__int64)v111, 0);
                    sub_1D96690(a1, (char *)a2, v128, a10);
                    sub_1D96690(a1, (char *)a2, v127, a10);
                    v62 = v145;
                    while ( v62 )
                    {
                      sub_1D924A0(*(_QWORD *)(v62 + 24));
                      v63 = v62;
                      v62 = *(_QWORD *)(v62 + 16);
                      j_j___libc_free_0(v63, 40);
                    }
                    if ( v141 != v143 )
                      _libc_free((unsigned __int64)v141);
                    v64 = v137;
                    while ( v64 )
                    {
                      sub_1D924A0(*(_QWORD *)(v64 + 24));
                      v65 = v64;
                      v64 = *(_QWORD *)(v64 + 16);
                      j_j___libc_free_0(v65, 40);
                    }
                    if ( v133 != v135 )
                      _libc_free((unsigned __int64)v133);
                    if ( v149 != v151 )
                      _libc_free((unsigned __int64)v149);
                    return 1;
                  }
                }
              }
              if ( v54 == src[4] )
                goto LABEL_74;
              while ( 1 )
              {
                v66 = *(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v66 )
                  BUG();
                v67 = *(_WORD *)(v66 + 46);
                if ( (*(_QWORD *)v66 & 4) != 0 )
                {
                  if ( (v67 & 4) != 0 )
                  {
LABEL_111:
                    v69 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v66 + 16) + 8LL) >> 7;
                    goto LABEL_106;
                  }
                }
                else if ( (v67 & 4) != 0 )
                {
                  for ( k = *(_QWORD *)v66; ; k = *(_QWORD *)v66 )
                  {
                    v66 = k & 0xFFFFFFFFFFFFFFF8LL;
                    v67 = *(_WORD *)(v66 + 46);
                    if ( (v67 & 4) == 0 )
                      break;
                  }
                }
                if ( (v67 & 8) == 0 )
                  goto LABEL_111;
                v69 = sub_1E15D00(v66, 128, 1);
LABEL_106:
                if ( v69 || (unsigned __int16)(**(_WORD **)(v66 + 16) - 12) <= 1u )
                {
                  v54 = v66;
                  if ( src[4] != v66 )
                    continue;
                }
                goto LABEL_74;
              }
            }
            v114 = a6;
            v32 = *(_QWORD *)(n + 32);
            do
            {
              v141 = v143;
              v142 = 0x400000000LL;
              sub_1DC2290(v122, v32, &v141);
              if ( v141 != v143 )
                _libc_free((unsigned __int64)v141);
              if ( !v32 )
                BUG();
              if ( (*(_BYTE *)v32 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v32 + 46) & 8) != 0 )
                  v32 = *(_QWORD *)(v32 + 8);
              }
              v32 = *(_QWORD *)(v32 + 8);
            }
            while ( v29 != (__int64 *)v32 );
            a6 = v114;
          }
          v33 = *(__int64 **)(n + 32);
          v34 = n + 16;
          if ( v33 != v29 )
          {
            v35 = *(_QWORD *)(a2 + 16);
            v36 = (__int64 *)(v35 + 24);
            if ( (__int64 *)(v35 + 24) != v29 )
            {
              if ( v35 + 16 != v34 )
              {
                v115 = (__int64 *)(v35 + 24);
                v123 = *(_QWORD *)(a2 + 16);
                sub_1DD5C00(v35 + 16, n + 16, *(_QWORD *)(n + 32), v29);
                v36 = v115;
                v35 = v123;
              }
              if ( v36 != v29 && v29 != v33 )
              {
                v37 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)((*v33 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v29;
                *v29 = *v29 & 7 | *v33 & 0xFFFFFFFFFFFFFFF8LL;
                v38 = *(_QWORD *)(v35 + 24);
                *(_QWORD *)(v37 + 8) = v36;
                v38 &= 0xFFFFFFFFFFFFFFF8LL;
                *v33 = v38 | *v33 & 7;
                *(_QWORD *)(v38 + 8) = v33;
                *(_QWORD *)(v35 + 24) = v37 | *(_QWORD *)(v35 + 24) & 7LL;
              }
            }
          }
          goto LABEL_57;
        }
        v71 = v151;
      }
      memcpy(v71, v16, v18);
      LODWORD(v19) = v150;
      goto LABEL_9;
    }
  }
  return result;
}
