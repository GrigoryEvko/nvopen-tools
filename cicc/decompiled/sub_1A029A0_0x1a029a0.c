// Function: sub_1A029A0
// Address: 0x1a029a0
//
__int64 __fastcall sub_1A029A0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned __int64 *v5; // r8
  unsigned __int64 v6; // r9
  __int64 v7; // rdi
  __int64 *v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rbx
  unsigned __int8 v19; // si
  int v20; // edx
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // r15d
  __int64 v24; // rdx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rsi
  __m128i *v28; // rcx
  __m128i *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rdi
  char v32; // dl
  _QWORD *v33; // r11
  unsigned __int64 *v34; // rdi
  unsigned int v35; // ecx
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdx
  int v40; // r10d
  unsigned int v41; // esi
  unsigned __int64 *v42; // rax
  unsigned __int64 v43; // rcx
  unsigned int v44; // esi
  __int64 v45; // rax
  bool v46; // zf
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // rax
  _QWORD *v50; // r11
  unsigned int v51; // r9d
  _QWORD *v52; // rdx
  _QWORD *m; // rax
  _QWORD *v54; // rdx
  _QWORD *v55; // rbx
  __int64 v56; // rcx
  int v57; // esi
  __int64 v58; // r8
  int v59; // esi
  __int64 v60; // r10
  int v61; // r15d
  __int64 *v62; // r12
  __int64 v63; // rdi
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdi
  unsigned int n; // eax
  __int64 *v67; // rdi
  __int64 v68; // r9
  int v69; // esi
  int v70; // ecx
  int v71; // esi
  int v72; // r10d
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // r8
  unsigned __int64 v75; // rdi
  unsigned int ii; // edi
  __int64 v77; // rax
  int v78; // ecx
  __int64 v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // r11
  unsigned int v82; // r9d
  int v83; // r10d
  _QWORD *v84; // rdx
  _QWORD *j; // rax
  _QWORD *v86; // rdx
  _QWORD *v87; // rbx
  __int64 v88; // rcx
  int v89; // esi
  __int64 v90; // r8
  int v91; // esi
  __int64 v92; // r11
  int v93; // r15d
  __int64 *v94; // r12
  __int64 v95; // rdi
  unsigned __int64 v96; // rdi
  unsigned __int64 v97; // rdi
  unsigned int k; // eax
  __int64 *v99; // rdi
  __int64 v100; // r9
  int v101; // esi
  int v102; // edi
  int v103; // esi
  int v104; // r10d
  unsigned int v105; // edi
  unsigned int v106; // edi
  __int64 v107; // rax
  unsigned int v108; // edi
  __int64 v109; // rax
  _QWORD *v110; // rcx
  _QWORD *v111; // rax
  __int64 v112; // rax
  _QWORD *v113; // rcx
  _QWORD *v114; // rax
  unsigned int v115; // eax
  unsigned int v116; // eax
  __int64 v117; // [rsp+0h] [rbp-3A0h]
  _QWORD *v119; // [rsp+10h] [rbp-390h]
  __int64 v120; // [rsp+18h] [rbp-388h]
  unsigned int v121; // [rsp+20h] [rbp-380h]
  int v122; // [rsp+20h] [rbp-380h]
  unsigned int v123; // [rsp+20h] [rbp-380h]
  _QWORD *v124; // [rsp+28h] [rbp-378h]
  unsigned __int64 v125; // [rsp+28h] [rbp-378h]
  unsigned int v126; // [rsp+28h] [rbp-378h]
  unsigned __int64 v127; // [rsp+28h] [rbp-378h]
  unsigned int v128; // [rsp+30h] [rbp-370h]
  unsigned __int64 v129; // [rsp+30h] [rbp-370h]
  _QWORD *v130; // [rsp+30h] [rbp-370h]
  unsigned __int64 v131; // [rsp+30h] [rbp-370h]
  int v132; // [rsp+30h] [rbp-370h]
  __int64 v133; // [rsp+40h] [rbp-360h]
  __int64 v134; // [rsp+58h] [rbp-348h]
  __int64 v135; // [rsp+60h] [rbp-340h]
  __int64 v136; // [rsp+68h] [rbp-338h]
  unsigned int v137; // [rsp+74h] [rbp-32Ch]
  __int64 v138; // [rsp+78h] [rbp-328h]
  __m128i v139; // [rsp+80h] [rbp-320h] BYREF
  _QWORD *i; // [rsp+90h] [rbp-310h] BYREF
  __int64 v141; // [rsp+98h] [rbp-308h]
  _QWORD v142[8]; // [rsp+A0h] [rbp-300h] BYREF
  _BYTE *v143; // [rsp+E0h] [rbp-2C0h] BYREF
  __int64 v144; // [rsp+E8h] [rbp-2B8h]
  _BYTE v145[64]; // [rsp+F0h] [rbp-2B0h] BYREF
  __m128i *v146; // [rsp+130h] [rbp-270h] BYREF
  __int64 v147; // [rsp+138h] [rbp-268h]
  _BYTE v148[512]; // [rsp+140h] [rbp-260h] BYREF
  __int64 v149; // [rsp+340h] [rbp-60h] BYREF
  int v150; // [rsp+348h] [rbp-58h] BYREF
  __int64 v151; // [rsp+350h] [rbp-50h]
  int *v152; // [rsp+358h] [rbp-48h]
  int *v153; // [rsp+360h] [rbp-40h]
  __int64 v154; // [rsp+368h] [rbp-38h]

  result = *a2;
  v120 = a2[1];
  v117 = *a2;
  while ( v117 != v120 )
  {
    v3 = *(_QWORD *)(v120 - 8);
    v133 = v3 + 40;
    v135 = *(_QWORD *)(v3 + 48);
    if ( v135 == v3 + 40 )
      goto LABEL_55;
    do
    {
      v4 = v135 - 24;
      if ( !v135 )
        v4 = 0;
      if ( !(unsigned __int8)sub_15F34B0(v4) )
        goto LABEL_54;
      v7 = *(_QWORD *)(v4 + 8);
      if ( v7 )
      {
        if ( !*(_QWORD *)(v7 + 8) && *(_BYTE *)(v4 + 16) == *((_BYTE *)sub_1648700(v7) + 16) )
          goto LABEL_54;
      }
      if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
        v8 = *(__int64 **)(v4 - 8);
      else
        v8 = (__int64 *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
      v11 = *v8;
      v9 = v8[3];
      v10 = v142;
      v142[0] = v11;
      LODWORD(v11) = 2;
      v142[1] = v9;
      v141 = 0x800000002LL;
      v143 = v145;
      v144 = 0x800000000LL;
      v12 = 0;
      for ( i = v142; ; v10 = i )
      {
        v17 = (unsigned int)v11;
        v11 = (unsigned int)(v11 - 1);
        v18 = v10[v17 - 1];
        LODWORD(v141) = v11;
        v19 = *(_BYTE *)(v18 + 16);
        if ( v19 <= 0x17u )
          break;
        if ( v19 != *(_BYTE *)(v4 + 16) )
          break;
        v13 = *(_QWORD *)(v18 + 8);
        if ( !v13 || *(_QWORD *)(v13 + 8) )
          break;
        if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
        {
          v14 = *(__int64 **)(v18 - 8);
          v15 = *v14;
          if ( v18 == *v14 )
            goto LABEL_19;
          if ( (unsigned int)v11 < HIDWORD(v141) )
            goto LABEL_17;
LABEL_59:
          sub_16CD150((__int64)&i, v142, 0, 8, (int)v5, v6);
          v10 = i;
          v11 = (unsigned int)v141;
          goto LABEL_17;
        }
        v14 = (__int64 *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
        v15 = *v14;
        if ( v18 == *v14 )
          goto LABEL_19;
        if ( (unsigned int)v11 >= HIDWORD(v141) )
          goto LABEL_59;
LABEL_17:
        v10[v11] = v15;
        v11 = (unsigned int)(v141 + 1);
        LODWORD(v141) = v141 + 1;
        if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
          v14 = *(__int64 **)(v18 - 8);
        else
          v14 = (__int64 *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
LABEL_19:
        v16 = v14[3];
        if ( v18 == v16 )
        {
          v12 = v144;
        }
        else
        {
          if ( HIDWORD(v141) <= (unsigned int)v11 )
          {
            sub_16CD150((__int64)&i, v142, 0, 8, (int)v5, v6);
            v11 = (unsigned int)v141;
          }
          i[v11] = v16;
          LODWORD(v11) = v141 + 1;
          v12 = v144;
          LODWORD(v141) = v141 + 1;
        }
        if ( !(_DWORD)v11 )
          goto LABEL_31;
LABEL_24:
        if ( v12 > 0xA )
          goto LABEL_50;
      }
      if ( HIDWORD(v144) <= v12 )
        sub_16CD150((__int64)&v143, v145, 0, 8, (int)v5, v6);
      *(_QWORD *)&v143[8 * (unsigned int)v144] = v18;
      LODWORD(v11) = v141;
      v12 = v144 + 1;
      LODWORD(v144) = v144 + 1;
      if ( (_DWORD)v141 )
        goto LABEL_24;
LABEL_31:
      v137 = v11;
      if ( v12 > 0xA )
        goto LABEL_50;
      v20 = *(unsigned __int8 *)(v4 + 16);
      v150 = 0;
      v146 = (__m128i *)v148;
      v21 = (unsigned int)(v20 - 35);
      v147 = 0x2000000000LL;
      v151 = 0;
      v152 = &v150;
      v153 = &v150;
      v154 = 0;
      if ( v12 == 1 )
        goto LABEL_50;
      v136 = 0;
      v134 = a1 + 32 * v21 + 176;
      while ( 2 )
      {
        ++v137;
        v22 = 8 * v136;
        v136 = v137;
        v23 = v137;
        v24 = v137;
        v138 = v22;
        if ( v137 >= v12 )
          goto LABEL_45;
        while ( 2 )
        {
          v25 = *(_QWORD *)&v143[8 * v24];
          v26 = *(_QWORD *)&v143[v138];
          if ( v25 < v26 )
          {
            v26 = *(_QWORD *)&v143[8 * v24];
            v25 = *(_QWORD *)&v143[v138];
          }
          v139.m128i_i64[0] = v26;
          v139.m128i_i64[1] = v25;
          if ( v154 )
          {
            sub_1A02830(&v149, &v139);
            if ( !v32 )
              goto LABEL_44;
LABEL_62:
            LODWORD(v6) = *(_DWORD *)(v134 + 24);
            v33 = *(_QWORD **)(v134 + 8);
            if ( !(_DWORD)v6 )
              goto LABEL_76;
          }
          else
          {
            v27 = (unsigned __int64)v146;
            v28 = &v146[(unsigned int)v147];
            if ( v146 != v28 )
            {
              v29 = v146;
              while ( v26 != v29->m128i_i64[0] || v25 != v29->m128i_i64[1] )
              {
                if ( v28 == ++v29 )
                  goto LABEL_71;
              }
              if ( v28 != v29 )
                goto LABEL_44;
            }
LABEL_71:
            v45 = (unsigned int)v147;
            if ( (unsigned int)v147 <= 0x1FuLL )
            {
              if ( (unsigned int)v147 >= HIDWORD(v147) )
              {
                sub_16CD150((__int64)&v146, v148, 0, 16, (int)v5, v6);
                v28 = &v146[(unsigned int)v147];
              }
              *v28 = _mm_load_si128(&v139);
              LODWORD(v147) = v147 + 1;
              goto LABEL_62;
            }
            while ( 1 )
            {
              sub_1A02830(&v149, (const __m128i *)(v27 + 16 * v45 - 16));
              v46 = (_DWORD)v147 == 1;
              v45 = (unsigned int)(v147 - 1);
              LODWORD(v147) = v147 - 1;
              if ( v46 )
                break;
              v27 = (unsigned __int64)v146;
            }
            sub_1A02830(&v149, &v139);
            LODWORD(v6) = *(_DWORD *)(v134 + 24);
            v33 = *(_QWORD **)(v134 + 8);
            if ( !(_DWORD)v6 )
            {
LABEL_76:
              ++*(_QWORD *)v134;
              goto LABEL_77;
            }
          }
          LODWORD(v5) = 1;
          v34 = 0;
          v35 = (unsigned int)v25 >> 9;
          v36 = (((v35 ^ ((unsigned int)v25 >> 4)
                 | ((unsigned __int64)(((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(v35 ^ ((unsigned int)v25 >> 4)) << 32)) >> 22)
              ^ ((v35 ^ ((unsigned int)v25 >> 4)
                | ((unsigned __int64)(((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(v35 ^ ((unsigned int)v25 >> 4)) << 32));
          v37 = ((9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13)))) >> 15)
              ^ (9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13))));
          v38 = ((v37 - 1 - (v37 << 27)) >> 31) ^ (v37 - 1 - (v37 << 27));
          v39 = (unsigned int)(v6 - 1);
          v40 = v38;
          v41 = v38 & (v6 - 1);
          while ( 2 )
          {
            v42 = &v33[3 * v41];
            v43 = *v42;
            if ( v26 == *v42 && v25 == v42[1] )
            {
              ++*((_DWORD *)v42 + 4);
              goto LABEL_44;
            }
            if ( v43 != -8 )
            {
              if ( v43 == -16 && v42[1] == -16 && !v34 )
                v34 = &v33[3 * v41];
              goto LABEL_70;
            }
            if ( v42[1] != -8 )
            {
LABEL_70:
              v44 = (_DWORD)v5 + v41;
              LODWORD(v5) = (_DWORD)v5 + 1;
              v41 = v39 & v44;
              continue;
            }
            break;
          }
          v78 = *(_DWORD *)(v134 + 16);
          if ( v34 )
            v42 = v34;
          ++*(_QWORD *)v134;
          v70 = v78 + 1;
          if ( 4 * v70 < (unsigned int)(3 * v6) )
          {
            if ( (int)v6 - *(_DWORD *)(v134 + 20) - v70 > (unsigned int)v6 >> 3 )
              goto LABEL_128;
            v122 = v40;
            v126 = v6;
            v130 = v33;
            v79 = ((((((((v39 >> 1) | v39 | (((v39 >> 1) | v39) >> 2)) >> 4)
                     | (v39 >> 1)
                     | v39
                     | (((v39 >> 1) | v39) >> 2)) >> 8)
                   | (((v39 >> 1) | v39 | (((v39 >> 1) | v39) >> 2)) >> 4)
                   | (v39 >> 1)
                   | v39
                   | (((v39 >> 1) | v39) >> 2)) >> 16)
                 | (((((v39 >> 1) | v39 | (((v39 >> 1) | v39) >> 2)) >> 4) | (v39 >> 1)
                                                                           | v39
                                                                           | (((v39 >> 1) | v39) >> 2)) >> 8)
                 | (((v39 >> 1) | v39 | (((v39 >> 1) | v39) >> 2)) >> 4)
                 | (v39 >> 1)
                 | v39
                 | (((v39 >> 1) | v39) >> 2))
                + 1;
            if ( (unsigned int)v79 < 0x40 )
              LODWORD(v79) = 64;
            *(_DWORD *)(v134 + 24) = v79;
            v80 = sub_22077B0(24LL * (unsigned int)v79);
            v81 = v130;
            v82 = v126;
            *(_QWORD *)(v134 + 8) = v80;
            v83 = v122;
            v84 = (_QWORD *)v80;
            if ( v130 )
            {
              *(_QWORD *)(v134 + 16) = 0;
              for ( j = (_QWORD *)(v80 + 24LL * *(unsigned int *)(v134 + 24)); j != v84; v84 += 3 )
              {
                if ( v84 )
                {
                  *v84 = -8;
                  v84[1] = -8;
                }
              }
              v127 = v25;
              v86 = v130;
              v123 = v23;
              v119 = v130;
              v131 = v26;
              v87 = &v81[3 * v82];
              while ( 1 )
              {
                while ( 1 )
                {
                  v88 = *v86;
                  if ( *v86 != -8 )
                    break;
                  if ( v86[1] == -8 )
                  {
                    v86 += 3;
                    if ( v87 == v86 )
                      goto LABEL_154;
                  }
                  else
                  {
LABEL_143:
                    v89 = *(_DWORD *)(v134 + 24);
                    if ( !v89 )
                    {
                      MEMORY[0] = *v86;
                      BUG();
                    }
                    v90 = v86[1];
                    v91 = v89 - 1;
                    v93 = 1;
                    v94 = 0;
                    v95 = ((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4);
                    v96 = (((v95 | ((unsigned __int64)(((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4)) << 32))
                          - 1
                          - (v95 << 32)) >> 22)
                        ^ ((v95 | ((unsigned __int64)(((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4)) << 32))
                         - 1
                         - (v95 << 32));
                    v97 = 9 * (((v96 - 1 - (v96 << 13)) >> 8) ^ (v96 - 1 - (v96 << 13)));
                    for ( k = v91
                            & ((((v97 ^ (v97 >> 15)) - 1 - ((v97 ^ (v97 >> 15)) << 27)) >> 31)
                             ^ ((v97 ^ (v97 >> 15)) - 1 - (((unsigned int)v97 ^ (unsigned int)(v97 >> 15)) << 27)));
                          ;
                          k = v91 & v116 )
                    {
                      v92 = *(_QWORD *)(v134 + 8);
                      v99 = (__int64 *)(v92 + 24LL * k);
                      v100 = *v99;
                      if ( v88 == *v99 && v99[1] == v90 )
                        goto LABEL_168;
                      if ( v100 == -8 )
                        break;
                      if ( v100 == -16 && v99[1] == -16 && !v94 )
                        v94 = (__int64 *)(v92 + 24LL * k);
LABEL_195:
                      v116 = v93 + k;
                      ++v93;
                    }
                    if ( v99[1] != -8 )
                      goto LABEL_195;
                    if ( v94 )
                      v99 = v94;
LABEL_168:
                    *v99 = v88;
                    v107 = v86[1];
                    v86 += 3;
                    v99[1] = v107;
                    *((_DWORD *)v99 + 4) = *((_DWORD *)v86 - 2);
                    ++*(_DWORD *)(v134 + 16);
                    if ( v87 == v86 )
                    {
LABEL_154:
                      v26 = v131;
                      v132 = v83;
                      v25 = v127;
                      v23 = v123;
                      j___libc_free_0(v119);
                      v84 = *(_QWORD **)(v134 + 8);
                      v101 = *(_DWORD *)(v134 + 24);
                      v83 = v132;
                      v70 = *(_DWORD *)(v134 + 16) + 1;
                      goto LABEL_155;
                    }
                  }
                }
                if ( v88 != -16 || v86[1] != -16 )
                  goto LABEL_143;
                v86 += 3;
                if ( v87 == v86 )
                  goto LABEL_154;
              }
            }
            v109 = *(unsigned int *)(v134 + 24);
            *(_QWORD *)(v134 + 16) = 0;
            v101 = v109;
            v110 = &v84[3 * v109];
            if ( v84 != v110 )
            {
              v111 = v84;
              do
              {
                if ( v111 )
                {
                  *v111 = -8;
                  v111[1] = -8;
                }
                v111 += 3;
              }
              while ( v110 != v111 );
            }
            v70 = 1;
LABEL_155:
            if ( !v101 )
            {
LABEL_202:
              ++*(_DWORD *)(v134 + 16);
              BUG();
            }
            v102 = v83;
            v103 = v101 - 1;
            v104 = 1;
            v5 = 0;
            v105 = v103 & v102;
            while ( 2 )
            {
              v42 = &v84[3 * v105];
              v6 = *v42;
              if ( v26 == *v42 )
              {
                if ( v25 == v42[1] )
                  goto LABEL_128;
                if ( v6 == -8 )
                  goto LABEL_176;
LABEL_159:
                if ( v6 == -16 && v42[1] == -16 && !v5 )
                  v5 = &v84[3 * v105];
              }
              else
              {
                if ( v6 != -8 )
                  goto LABEL_159;
LABEL_176:
                if ( v42[1] == -8 )
                  goto LABEL_177;
              }
              v106 = v104 + v105;
              ++v104;
              v105 = v103 & v106;
              continue;
            }
          }
LABEL_77:
          v124 = v33;
          v128 = v6;
          v47 = ((((((((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v6 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v6 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 8)
               | (((((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v6 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v6 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 16;
          v48 = (v47
               | (((((((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v6 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v6 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 8)
               | (((((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v6 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v6 - 1) | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v6 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v6 - 1) >> 1))
              + 1;
          if ( (unsigned int)v48 < 0x40 )
            LODWORD(v48) = 64;
          *(_DWORD *)(v134 + 24) = v48;
          v49 = sub_22077B0(24LL * (unsigned int)v48);
          v50 = v124;
          v51 = v128;
          *(_QWORD *)(v134 + 8) = v49;
          v52 = (_QWORD *)v49;
          if ( v124 )
          {
            *(_QWORD *)(v134 + 16) = 0;
            for ( m = (_QWORD *)(v49 + 24LL * *(unsigned int *)(v134 + 24)); m != v52; v52 += 3 )
            {
              if ( v52 )
              {
                *v52 = -8;
                v52[1] = -8;
              }
            }
            v54 = v124;
            if ( v124 != &v124[3 * v128] )
            {
              v125 = v25;
              v121 = v23;
              v129 = v26;
              v55 = &v50[3 * v51];
              while ( 1 )
              {
                while ( 1 )
                {
                  v56 = *v54;
                  if ( *v54 != -8 )
                    break;
                  if ( v54[1] == -8 )
                  {
                    v54 += 3;
                    if ( v55 == v54 )
                      goto LABEL_105;
                  }
                  else
                  {
LABEL_88:
                    v57 = *(_DWORD *)(v134 + 24);
                    if ( !v57 )
                    {
                      MEMORY[0] = *v54;
                      BUG();
                    }
                    v58 = v54[1];
                    v59 = v57 - 1;
                    v60 = *(_QWORD *)(v134 + 8);
                    v61 = 1;
                    v62 = 0;
                    v63 = ((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4);
                    v64 = (((v63 | ((unsigned __int64)(((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4)) << 32))
                          - 1
                          - (v63 << 32)) >> 22)
                        ^ ((v63 | ((unsigned __int64)(((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4)) << 32))
                         - 1
                         - (v63 << 32));
                    v65 = ((9 * (((v64 - 1 - (v64 << 13)) >> 8) ^ (v64 - 1 - (v64 << 13)))) >> 15)
                        ^ (9 * (((v64 - 1 - (v64 << 13)) >> 8) ^ (v64 - 1 - (v64 << 13))));
                    for ( n = v59 & (((v65 - 1 - (v65 << 27)) >> 31) ^ (v65 - 1 - ((_DWORD)v65 << 27))); ; n = v59 & v115 )
                    {
                      v67 = (__int64 *)(v60 + 24LL * n);
                      v68 = *v67;
                      if ( v56 == *v67 && v67[1] == v58 )
                        goto LABEL_120;
                      if ( v68 == -8 )
                        break;
                      if ( v68 == -16 && v67[1] == -16 && !v62 )
                        v62 = (__int64 *)(v60 + 24LL * n);
LABEL_193:
                      v115 = v61 + n;
                      ++v61;
                    }
                    if ( v67[1] != -8 )
                      goto LABEL_193;
                    if ( v62 )
                      v67 = v62;
LABEL_120:
                    *v67 = v56;
                    v77 = v54[1];
                    v54 += 3;
                    v67[1] = v77;
                    *((_DWORD *)v67 + 4) = *((_DWORD *)v54 - 2);
                    ++*(_DWORD *)(v134 + 16);
                    if ( v55 == v54 )
                    {
LABEL_105:
                      v26 = v129;
                      v25 = v125;
                      v23 = v121;
                      goto LABEL_106;
                    }
                  }
                }
                if ( v56 != -16 || v54[1] != -16 )
                  goto LABEL_88;
                v54 += 3;
                if ( v55 == v54 )
                  goto LABEL_105;
              }
            }
LABEL_106:
            j___libc_free_0(v50);
            v52 = *(_QWORD **)(v134 + 8);
            v69 = *(_DWORD *)(v134 + 24);
            v70 = *(_DWORD *)(v134 + 16) + 1;
          }
          else
          {
            v112 = *(unsigned int *)(v134 + 24);
            *(_QWORD *)(v134 + 16) = 0;
            v69 = v112;
            v113 = &v52[3 * v112];
            if ( v52 != v113 )
            {
              v114 = v52;
              do
              {
                if ( v114 )
                {
                  *v114 = -8;
                  v114[1] = -8;
                }
                v114 += 3;
              }
              while ( v113 != v114 );
            }
            v70 = 1;
          }
          if ( !v69 )
            goto LABEL_202;
          v71 = v69 - 1;
          v72 = 1;
          v73 = (((((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)
                 | ((unsigned __int64)(((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)) << 32)) >> 22)
              ^ ((((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)
                | ((unsigned __int64)(((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)) << 32));
          v74 = ((9 * (((v73 - 1 - (v73 << 13)) >> 8) ^ (v73 - 1 - (v73 << 13)))) >> 15)
              ^ (9 * (((v73 - 1 - (v73 << 13)) >> 8) ^ (v73 - 1 - (v73 << 13))));
          v75 = v74 - 1 - (v74 << 27);
          v5 = 0;
          for ( ii = v71 & ((v75 >> 31) ^ v75); ; ii = v71 & v108 )
          {
            v42 = &v52[3 * ii];
            v6 = *v42;
            if ( v26 == *v42 )
              break;
            if ( v6 == -8 )
              goto LABEL_172;
LABEL_111:
            if ( v6 == -16 && v42[1] == -16 && !v5 )
              v5 = &v52[3 * ii];
LABEL_173:
            v108 = v72 + ii;
            ++v72;
          }
          if ( v25 == v42[1] )
            goto LABEL_128;
          if ( v6 != -8 )
            goto LABEL_111;
LABEL_172:
          if ( v42[1] != -8 )
            goto LABEL_173;
LABEL_177:
          if ( v5 )
            v42 = v5;
LABEL_128:
          *(_DWORD *)(v134 + 16) = v70;
          if ( *v42 != -8 || v42[1] != -8 )
            --*(_DWORD *)(v134 + 20);
          *v42 = v26;
          v42[1] = v25;
          *((_DWORD *)v42 + 4) = 1;
LABEL_44:
          v12 = v144;
          v24 = v23 + 1;
          v23 = v24;
          if ( (unsigned int)v24 < (unsigned int)v144 )
            continue;
          break;
        }
LABEL_45:
        if ( (unsigned __int64)v12 - 1 > v137 )
          continue;
        break;
      }
      v30 = v151;
      while ( v30 )
      {
        sub_19FF0E0(*(_QWORD *)(v30 + 24));
        v31 = v30;
        v30 = *(_QWORD *)(v30 + 16);
        j_j___libc_free_0(v31, 48);
      }
      if ( v146 != (__m128i *)v148 )
        _libc_free((unsigned __int64)v146);
LABEL_50:
      if ( v143 != v145 )
        _libc_free((unsigned __int64)v143);
      if ( i != v142 )
        _libc_free((unsigned __int64)i);
LABEL_54:
      v135 = *(_QWORD *)(v135 + 8);
    }
    while ( v133 != v135 );
LABEL_55:
    v120 -= 8;
    result = v120;
  }
  return result;
}
