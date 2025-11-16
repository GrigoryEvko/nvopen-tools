// Function: sub_3875200
// Address: 0x3875200
//
__int64 __fastcall sub_3875200(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // r8
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdi
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rdi
  int v22; // esi
  int v23; // r9d
  unsigned int v24; // eax
  __int64 v25; // r8
  int v26; // eax
  int v27; // r8d
  __int64 v28; // r10
  int v29; // r11d
  unsigned int v30; // eax
  __int64 v31; // r9
  __int64 v32; // rcx
  int v33; // r8d
  int v34; // r9d
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  _BYTE *v39; // r9
  __int64 *v40; // r10
  _QWORD *v41; // r11
  __int64 v42; // rax
  unsigned int v43; // edx
  __int64 v44; // r11
  __int64 *v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  char v49; // al
  __int64 *v50; // r12
  __int64 v51; // r11
  unsigned int v52; // esi
  int v53; // eax
  int v54; // eax
  __int64 *v55; // r13
  _BYTE *v56; // rax
  __int64 v59; // rax
  __int64 *v60; // r8
  _BOOL4 v61; // r14d
  bool v62; // zf
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned int v65; // edx
  bool v66; // al
  _BYTE *v67; // rdi
  __int64 **v68; // rdx
  int v69; // eax
  __int64 v70; // r14
  _BYTE *v71; // rsi
  __int64 v72; // rbx
  __int64 v73; // rax
  int i; // eax
  __int64 *v75; // r9
  __int64 *v76; // rbx
  __int64 *v77; // r12
  char v78; // dl
  __int64 v79; // rax
  __int64 v80; // rax
  bool v81; // al
  __int64 v82; // r13
  __int64 *v83; // rax
  __int64 *v84; // rsi
  __int64 *v85; // rcx
  __int64 v86; // rax
  __int64 **v87; // rax
  int v88; // edx
  __int64 v89; // rax
  __int64 v90; // rax
  unsigned int v91; // ebx
  int v92; // eax
  bool v93; // al
  unsigned __int64 v94; // rdi
  __int64 **v95; // rax
  __int64 v96; // r8
  __int64 *v97; // r9
  char v98; // dl
  __int64 v99; // rax
  __int64 v100; // rax
  int v101; // eax
  bool v102; // al
  __int64 *v103; // rsi
  char v104; // dl
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned int v107; // ebx
  __int64 **v108; // rdi
  __int64 **v109; // rcx
  __int64 *v110; // rdx
  __int64 *v111; // rsi
  __int64 **v112; // rdi
  __int64 **v113; // rsi
  unsigned int v114; // esi
  __int64 v115; // rax
  _QWORD *v116; // rax
  __int64 **v117; // rax
  __int64 v118; // rax
  _QWORD *v119; // rax
  __int64 v120; // rax
  _QWORD *v121; // rax
  __int64 v122; // rax
  int v123; // r10d
  _BYTE *v124; // [rsp+8h] [rbp-168h]
  __int64 v125; // [rsp+10h] [rbp-160h]
  __int64 v126; // [rsp+10h] [rbp-160h]
  __int64 *v127; // [rsp+18h] [rbp-158h]
  __int64 v128; // [rsp+18h] [rbp-158h]
  __int64 v129; // [rsp+18h] [rbp-158h]
  __int64 v130; // [rsp+20h] [rbp-150h]
  int v131; // [rsp+20h] [rbp-150h]
  _BYTE *v132; // [rsp+20h] [rbp-150h]
  __int64 v133; // [rsp+28h] [rbp-148h]
  __int64 *v134; // [rsp+30h] [rbp-140h]
  _QWORD *v135; // [rsp+30h] [rbp-140h]
  __int64 v136; // [rsp+30h] [rbp-140h]
  __int64 v137; // [rsp+30h] [rbp-140h]
  int v138; // [rsp+30h] [rbp-140h]
  __int64 **v139; // [rsp+30h] [rbp-140h]
  __int64 v140; // [rsp+30h] [rbp-140h]
  __int64 v141; // [rsp+30h] [rbp-140h]
  __int64 v142; // [rsp+30h] [rbp-140h]
  __int64 *v143; // [rsp+48h] [rbp-128h] BYREF
  __int64 *v144[6]; // [rsp+50h] [rbp-120h] BYREF
  char *v145; // [rsp+80h] [rbp-F0h] BYREF
  _BYTE *v146; // [rsp+88h] [rbp-E8h] BYREF
  __int64 v147; // [rsp+90h] [rbp-E0h]
  _BYTE v148[64]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v149; // [rsp+D8h] [rbp-98h] BYREF
  __int64 **v150; // [rsp+E0h] [rbp-90h]
  __int64 **v151; // [rsp+E8h] [rbp-88h]
  __int64 v152; // [rsp+F0h] [rbp-80h]
  int v153; // [rsp+F8h] [rbp-78h]
  _QWORD v154[14]; // [rsp+100h] [rbp-70h] BYREF

  v6 = (__int64)a1;
  v7 = a1[35];
  v8 = *a1;
  v9 = *(_QWORD *)(v8 + 64);
  if ( v7 )
    v7 -= 24;
  v10 = 0;
  v11 = *(_DWORD *)(v9 + 24);
  if ( v11 )
  {
    v12 = *(_QWORD *)(v6 + 272);
    v13 = *(_QWORD *)(v9 + 8);
    v14 = v11 - 1;
    v15 = (v11 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( v12 != *v16 )
    {
      for ( i = 1; ; i = v123 )
      {
        if ( v17 == -8 )
        {
          v10 = 0;
          goto LABEL_8;
        }
        v123 = i + 1;
        v15 = v14 & (i + v15);
        v16 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
          break;
      }
    }
    v10 = v16[1];
  }
LABEL_8:
  while ( sub_146CEE0(v8, a2, v10) )
  {
    if ( !v10 )
      goto LABEL_27;
    v18 = sub_13FC520(v10);
    if ( v18 )
    {
      v7 = sub_157EBA0(v18);
    }
    else
    {
      v7 = sub_157EE30(**(_QWORD **)(v10 + 32));
      if ( v7 )
        v7 -= 24;
    }
    v10 = *(_QWORD *)v10;
    v8 = *(_QWORD *)v6;
  }
  if ( v10 )
  {
    if ( sub_146D100(*(_QWORD *)v6, a2, v10) )
    {
      v61 = sub_1498DE0(v6 + 152, v10);
      if ( !v61 )
      {
        v62 = *(_WORD *)(a2 + 24) == 6;
        LOBYTE(v143) = 0;
        v146 = v148;
        v147 = 0x800000000LL;
        v150 = (__int64 **)v154;
        v151 = (__int64 **)v154;
        v145 = (char *)&v143;
        v144[0] = (__int64 *)a2;
        v152 = 0x100000008LL;
        v153 = 0;
        v154[0] = a2;
        v149 = 1;
        if ( v62
          && ((v63 = *(_QWORD *)(a2 + 40), *(_WORD *)(v63 + 24))
           || ((v64 = *(_QWORD *)(v63 + 32), v65 = *(_DWORD *)(v64 + 32), v65 <= 0x40)
             ? (v66 = *(_QWORD *)(v64 + 24) == 0)
             : (v66 = v65 == (unsigned int)sub_16A57B0(v64 + 24)),
               v66)) )
        {
          LOBYTE(v143) = 1;
          v67 = v148;
          v68 = &v143;
        }
        else
        {
          sub_1458920((__int64)&v146, v144);
          v68 = (__int64 **)v145;
          v67 = v146;
          v61 = v147;
        }
        v128 = v7;
        v69 = v61;
        v70 = a2;
        v126 = v6;
LABEL_65:
        v71 = &v67[8 * v69];
        while ( 2 )
        {
          if ( v69 && !*(_BYTE *)v68 )
          {
            v72 = *((_QWORD *)v71 - 1);
            LODWORD(v147) = --v69;
            switch ( *(_WORD *)(v72 + 24) )
            {
              case 0:
              case 0xA:
                v71 -= 8;
                continue;
              case 1:
              case 2:
              case 3:
                v60 = *(__int64 **)(v72 + 32);
                v87 = v150;
                v144[0] = v60;
                if ( v151 != v150 )
                  goto LABEL_100;
                v112 = &v150[HIDWORD(v152)];
                if ( v150 == v112 )
                  goto LABEL_191;
                v113 = 0;
                do
                {
                  if ( v60 == *v87 )
                    goto LABEL_108;
                  if ( *v87 == (__int64 *)-2LL )
                    v113 = v87;
                  ++v87;
                }
                while ( v112 != v87 );
                if ( !v113 )
                {
LABEL_191:
                  if ( HIDWORD(v152) >= (unsigned int)v152 )
                  {
LABEL_100:
                    sub_16CCBA0((__int64)&v149, (__int64)v60);
                    LODWORD(v60) = v88;
                    v68 = (__int64 **)v145;
                    if ( !(_BYTE)v60 )
                      goto LABEL_108;
                  }
                  else
                  {
                    ++HIDWORD(v152);
                    *v112 = v60;
                    v68 = (__int64 **)v145;
                    ++v149;
                  }
                }
                else
                {
                  *v113 = v60;
                  v68 = (__int64 **)v145;
                  --v153;
                  ++v149;
                }
                if ( *((_WORD *)v144[0] + 12) == 6 )
                {
                  v89 = v144[0][5];
                  if ( *(_WORD *)(v89 + 24)
                    || ((v90 = *(_QWORD *)(v89 + 32), v91 = *(_DWORD *)(v90 + 32), v91 <= 0x40)
                      ? (v93 = *(_QWORD *)(v90 + 24) == 0)
                      : (v139 = v68, v92 = sub_16A57B0(v90 + 24), v68 = v139, v93 = v91 == v92),
                        v93) )
                  {
                    *(_BYTE *)v68 = 1;
                    goto LABEL_107;
                  }
                }
                sub_1458920((__int64)&v146, v144);
                v68 = (__int64 **)v145;
                goto LABEL_108;
              case 4:
              case 5:
              case 7:
              case 8:
              case 9:
                v75 = *(__int64 **)(v72 + 32);
                v76 = &v75[*(_QWORD *)(v72 + 40)];
                if ( v75 == v76 )
                  goto LABEL_65;
                v77 = v75;
                break;
              case 6:
                v94 = (unsigned __int64)v151;
                v95 = v150;
                v96 = *(_QWORD *)(v72 + 32);
                if ( v151 != v150 )
                  goto LABEL_111;
                v97 = (__int64 *)&v151[HIDWORD(v152)];
                if ( v151 == (__int64 **)v97 )
                  goto LABEL_184;
                v110 = (__int64 *)v151;
                v111 = 0;
                while ( v96 != *v110 )
                {
                  if ( *v110 == -2 )
                    v111 = v110;
                  if ( v97 == ++v110 )
                  {
                    if ( v111 )
                    {
                      *v111 = v96;
                      --v153;
                      ++v149;
                    }
                    else
                    {
LABEL_184:
                      if ( HIDWORD(v152) >= (unsigned int)v152 )
                      {
LABEL_111:
                        v140 = *(_QWORD *)(v72 + 32);
                        sub_16CCBA0((__int64)&v149, v140);
                        v94 = (unsigned __int64)v151;
                        v95 = v150;
                        v96 = v140;
                        if ( !v98 )
                          break;
                      }
                      else
                      {
                        ++HIDWORD(v152);
                        *v97 = v96;
                        ++v149;
                      }
                    }
                    if ( *(_WORD *)(v96 + 24) == 6
                      && ((v99 = *(_QWORD *)(v96 + 40), *(_WORD *)(v99 + 24))
                       || ((v100 = *(_QWORD *)(v99 + 32), *(_DWORD *)(v100 + 32) <= 0x40u)
                         ? (v102 = *(_QWORD *)(v100 + 24) == 0)
                         : (v131 = *(_DWORD *)(v100 + 32),
                            v141 = v96,
                            v101 = sub_16A57B0(v100 + 24),
                            v96 = v141,
                            v102 = v131 == v101),
                           v102)) )
                    {
                      *v145 = 1;
                      v94 = (unsigned __int64)v151;
                      v95 = v150;
                    }
                    else
                    {
                      v122 = (unsigned int)v147;
                      if ( (unsigned int)v147 >= HIDWORD(v147) )
                      {
                        v142 = v96;
                        sub_16CD150((__int64)&v146, v148, 0, 8, v96, (int)v97);
                        v122 = (unsigned int)v147;
                        v96 = v142;
                      }
                      *(_QWORD *)&v146[8 * v122] = v96;
                      v94 = (unsigned __int64)v151;
                      LODWORD(v147) = v147 + 1;
                      v95 = v150;
                    }
                    break;
                  }
                }
                v103 = *(__int64 **)(v72 + 40);
                v144[0] = v103;
                if ( (__int64 **)v94 != v95 )
                  goto LABEL_119;
                v108 = &v95[HIDWORD(v152)];
                LODWORD(v60) = HIDWORD(v152);
                if ( v108 == v95 )
                {
LABEL_187:
                  if ( HIDWORD(v152) >= (unsigned int)v152 )
                  {
LABEL_119:
                    sub_16CCBA0((__int64)&v149, (__int64)v103);
                    if ( !v104 )
                      goto LABEL_107;
                  }
                  else
                  {
                    LODWORD(v60) = ++HIDWORD(v152);
                    *v108 = v103;
                    ++v149;
                  }
LABEL_120:
                  if ( *((_WORD *)v144[0] + 12) == 6 )
                  {
                    v105 = v144[0][5];
                    if ( *(_WORD *)(v105 + 24) )
                      goto LABEL_124;
                    v106 = *(_QWORD *)(v105 + 32);
                    v107 = *(_DWORD *)(v106 + 32);
                    if ( v107 <= 0x40 )
                    {
                      if ( !*(_QWORD *)(v106 + 24) )
                      {
LABEL_124:
                        *v145 = 1;
                        goto LABEL_107;
                      }
                    }
                    else if ( v107 == (unsigned int)sub_16A57B0(v106 + 24) )
                    {
                      goto LABEL_124;
                    }
                  }
                  sub_1458920((__int64)&v146, v144);
                }
                else
                {
                  v109 = 0;
                  while ( v103 != *v95 )
                  {
                    if ( *v95 == (__int64 *)-2LL )
                      v109 = v95;
                    if ( v108 == ++v95 )
                    {
                      if ( !v109 )
                        goto LABEL_187;
                      *v109 = v103;
                      --v153;
                      ++v149;
                      goto LABEL_120;
                    }
                  }
                }
LABEL_107:
                v68 = (__int64 **)v145;
LABEL_108:
                v67 = v146;
                v69 = v147;
                goto LABEL_65;
            }
            while ( 1 )
            {
              v82 = *v77;
              v83 = (__int64 *)v150;
              if ( v151 != v150 )
                goto LABEL_79;
              v84 = (__int64 *)&v150[HIDWORD(v152)];
              if ( v150 != (__int64 **)v84 )
              {
                v85 = 0;
                while ( v82 != *v83 )
                {
                  if ( *v83 == -2 )
                    v85 = v83;
                  if ( v84 == ++v83 )
                  {
                    if ( !v85 )
                      goto LABEL_139;
                    *v85 = v82;
                    --v153;
                    ++v149;
                    if ( *(_WORD *)(v82 + 24) == 6 )
                      goto LABEL_81;
                    goto LABEL_96;
                  }
                }
                goto LABEL_86;
              }
LABEL_139:
              if ( HIDWORD(v152) < (unsigned int)v152 )
              {
                ++HIDWORD(v152);
                *v84 = v82;
                ++v149;
              }
              else
              {
LABEL_79:
                sub_16CCBA0((__int64)&v149, *v77);
                if ( !v78 )
                  goto LABEL_86;
              }
              if ( *(_WORD *)(v82 + 24) != 6 )
                goto LABEL_96;
LABEL_81:
              v79 = *(_QWORD *)(v82 + 40);
              if ( !*(_WORD *)(v79 + 24) )
              {
                v80 = *(_QWORD *)(v79 + 32);
                if ( *(_DWORD *)(v80 + 32) <= 0x40u )
                {
                  v81 = *(_QWORD *)(v80 + 24) == 0;
                }
                else
                {
                  v138 = *(_DWORD *)(v80 + 32);
                  v81 = v138 == (unsigned int)sub_16A57B0(v80 + 24);
                }
                if ( !v81 )
                {
LABEL_96:
                  v86 = (unsigned int)v147;
                  if ( (unsigned int)v147 >= HIDWORD(v147) )
                  {
                    sub_16CD150((__int64)&v146, v148, 0, 8, (int)v60, (int)v75);
                    v86 = (unsigned int)v147;
                  }
                  *(_QWORD *)&v146[8 * v86] = v82;
                  LODWORD(v147) = v147 + 1;
                  goto LABEL_86;
                }
              }
              *v145 = 1;
LABEL_86:
              if ( v76 == ++v77 )
                goto LABEL_107;
            }
          }
          break;
        }
        v7 = v128;
        v6 = v126;
        a2 = v70;
        if ( v151 != v150 )
        {
          _libc_free((unsigned __int64)v151);
          v67 = v146;
        }
        if ( v67 != v148 )
          _libc_free((unsigned __int64)v67);
        if ( !(_BYTE)v143 )
        {
          v7 = sub_157EE30(**(_QWORD **)(v10 + 32));
          if ( v7 )
            v7 -= 24;
        }
      }
    }
  }
  v19 = *(_QWORD *)(v6 + 280);
  if ( v19 != v7 + 24 )
  {
    v20 = *(_DWORD *)(v6 + 80);
    v21 = *(_QWORD *)(v6 + 64);
    v22 = v20 - 1;
    do
    {
      if ( v20 )
      {
        v23 = 1;
        v24 = v22 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v25 = *(_QWORD *)(v21 + 8LL * v24);
        if ( v25 == v7 )
          goto LABEL_19;
        while ( v25 != -8 )
        {
          v24 = v22 & (v23 + v24);
          v25 = *(_QWORD *)(v21 + 8LL * v24);
          if ( v25 == v7 )
            goto LABEL_19;
          ++v23;
        }
      }
      v26 = *(_DWORD *)(v6 + 112);
      if ( !v26 )
        goto LABEL_26;
      v27 = v26 - 1;
      v28 = *(_QWORD *)(v6 + 96);
      v29 = 1;
      v30 = (v26 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v31 = *(_QWORD *)(v28 + 8LL * v30);
      if ( v7 != v31 )
      {
        while ( v31 != -8 )
        {
          v30 = v27 & (v29 + v30);
          v31 = *(_QWORD *)(v28 + 8LL * v30);
          if ( v31 == v7 )
            goto LABEL_19;
          ++v29;
        }
LABEL_26:
        if ( *(_BYTE *)(v7 + 16) != 78 )
          break;
        v59 = *(_QWORD *)(v7 - 24);
        if ( *(_BYTE *)(v59 + 16)
          || (*(_BYTE *)(v59 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v59 + 36) - 35) > 3 )
        {
          break;
        }
      }
LABEL_19:
      v7 = *(_QWORD *)(v7 + 32);
      if ( v7 )
        v7 -= 24;
    }
    while ( v19 != v7 + 24 );
  }
LABEL_27:
  v145 = (char *)a2;
  v146 = (_BYTE *)v7;
  v33 = sub_3872DE0(v6 + 24, (__int64 *)&v145, v144);
  if ( (_BYTE)v33 )
  {
    v32 = 5LL * *(unsigned int *)(v6 + 48);
    if ( v144[0] != (__int64 *)(*(_QWORD *)(v6 + 32) + 40LL * *(unsigned int *)(v6 + 48)) )
      return v144[0][4];
  }
  v134 = (__int64 *)(v6 + 264);
  sub_38701C0(v144, (__int64 *)(v6 + 264), v6, v32, v33, v34);
  sub_17050D0((__int64 *)(v6 + 264), v7);
  v35 = (_QWORD *)sub_3872670(v6, a2, v7);
  v39 = v35;
  v40 = (__int64 *)v36;
  v41 = v35;
  if ( v35 )
  {
    if ( v36 )
    {
      v42 = *v35;
      v133 = v42;
      if ( *(_BYTE *)(v42 + 8) == 15 )
      {
        v43 = *(_DWORD *)(v36 + 32);
        v44 = **(_QWORD **)(v42 + 16);
        v45 = (__int64 *)v40[3];
        if ( v43 <= 0x40 )
          v46 = (__int64)((_QWORD)v45 << (64 - (unsigned __int8)v43)) >> (64 - (unsigned __int8)v43);
        else
          v46 = *v45;
        v124 = v39;
        v125 = v46;
        v127 = v40;
        v130 = v44;
        v47 = sub_1456C90(*(_QWORD *)v6, v44);
        if ( 8 * v125 % v47 )
        {
          v129 = sub_159C580(*v127, -v125);
          v114 = *(_DWORD *)(v133 + 8);
          v115 = *(_QWORD *)v6;
          LOWORD(v147) = 257;
          v116 = (_QWORD *)sub_15E0530(*(_QWORD *)(v115 + 24));
          v117 = (__int64 **)sub_16471D0(v116, v114 >> 8);
          v132 = sub_38723F0(v134, 47, (__int64)v124, v117, (__int64 *)&v145);
          v118 = *(_QWORD *)v6;
          v145 = "uglygep";
          LOWORD(v147) = 259;
          v119 = (_QWORD *)sub_15E0530(*(_QWORD *)(v118 + 24));
          v120 = sub_1643330(v119);
          v121 = sub_3871660((__int64)v134, v120, v132, v129, (__int64 *)&v145);
          LOWORD(v147) = 257;
          v41 = sub_38723F0(v134, 47, (__int64)v121, (__int64 **)v133, (__int64 *)&v145);
        }
        else
        {
          v48 = sub_159C580(*v127, -8 * v125 / v47);
          v145 = "scevgep";
          LOWORD(v147) = 259;
          v41 = sub_3871660((__int64)v134, v130, v124, v48, (__int64 *)&v145);
        }
      }
      else
      {
        LOWORD(v147) = 257;
        v41 = sub_38718D0(v134, (__int64)v41, v36, (__int64 *)&v145, 0, 0, a3, a4, a5);
      }
    }
  }
  else
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
        v41 = *(_QWORD **)(a2 + 32);
        break;
      case 1:
        v41 = (_QWORD *)sub_38762F0(v6, a2, v36, v37, v38, 0);
        break;
      case 2:
        v41 = (_QWORD *)sub_3876480(v6, a2, v36, v37, v38, 0);
        break;
      case 3:
        v41 = (_QWORD *)sub_3876610(v6, a2, v36, v37, v38, 0);
        break;
      case 4:
        v41 = (_QWORD *)sub_3878580(v6, a2, v36, v37, v38, 0);
        break;
      case 5:
        v41 = (_QWORD *)sub_3876AE0(v6, a2, v36, v37, v38, 0);
        break;
      case 6:
        v41 = (_QWORD *)sub_3876200(v6, a2, v36, v37, v38, 0);
        break;
      case 7:
        v41 = (_QWORD *)sub_387C970(v6, a2, v36, v37, v38, 0);
        break;
      case 8:
        v41 = (_QWORD *)sub_387BA10(v6, a2, v36, v37, v38, 0);
        break;
      case 9:
        v41 = (_QWORD *)sub_387B3F0(v6, a2, v36, v37, v38, 0);
        break;
      case 0xA:
        v41 = *(_QWORD **)(a2 - 8);
        break;
      default:
        BUG();
    }
  }
  v146 = (_BYTE *)v7;
  v135 = v41;
  v145 = (char *)a2;
  v49 = sub_3872DE0(v6 + 24, (__int64 *)&v145, &v143);
  v50 = v143;
  v51 = (__int64)v135;
  if ( !v49 )
  {
    v52 = *(_DWORD *)(v6 + 48);
    v53 = *(_DWORD *)(v6 + 40);
    ++*(_QWORD *)(v6 + 24);
    v54 = v53 + 1;
    if ( 4 * v54 >= 3 * v52 )
    {
      v52 *= 2;
    }
    else if ( v52 - *(_DWORD *)(v6 + 44) - v54 > v52 >> 3 )
    {
LABEL_38:
      *(_DWORD *)(v6 + 40) = v54;
      if ( *v50 != -8 || v50[1] != -8 )
        --*(_DWORD *)(v6 + 44);
      v55 = v50 + 2;
      *v50 = (__int64)v145;
      v56 = v146;
      v50[2] = 6;
      v50[1] = (__int64)v56;
      v50[3] = 0;
      v50[4] = 0;
      if ( !v51 )
        goto LABEL_44;
      goto LABEL_41;
    }
    sub_3874EC0(v6 + 24, v52);
    sub_3872DE0(v6 + 24, (__int64 *)&v145, &v143);
    v50 = v143;
    v51 = (__int64)v135;
    v54 = *(_DWORD *)(v6 + 40) + 1;
    goto LABEL_38;
  }
  v73 = v143[4];
  v55 = v143 + 2;
  if ( v135 != (_QWORD *)v73 )
  {
    if ( v73 != 0 && v73 != -8 && v73 != -16 )
    {
      sub_1649B30(v143 + 2);
      v51 = (__int64)v135;
    }
LABEL_41:
    v50[4] = v51;
    if ( v51 != 0 && v51 != -8 && v51 != -16 )
    {
      v136 = v51;
      sub_164C220((__int64)v55);
      v51 = v136;
    }
  }
LABEL_44:
  v137 = v51;
  sub_3870260((__int64)v144);
  return v137;
}
