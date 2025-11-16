// Function: sub_2D1E1A0
// Address: 0x2d1e1a0
//
__int64 __fastcall sub_2D1E1A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // r15
  unsigned int v4; // ebx
  __int64 v5; // r14
  __int64 v6; // rdi
  bool v7; // al
  __int64 v8; // r15
  __int64 i; // r14
  unsigned __int8 v10; // bl
  unsigned __int64 v11; // rsi
  char v12; // al
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *j; // rdx
  __int64 v20; // r12
  __int64 v21; // r10
  int v22; // r13d
  __int64 v23; // r8
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 v28; // rbx
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // ecx
  int v33; // eax
  __int64 *v34; // rdx
  __int64 v35; // rdi
  _QWORD *v36; // rbx
  int *v37; // r12
  int *v38; // rax
  _QWORD *v39; // r14
  _QWORD *v40; // rdx
  int *v41; // r8
  __int64 v42; // rsi
  __int64 v43; // rcx
  int *v44; // rax
  _QWORD *v45; // r15
  _QWORD *v46; // rdi
  int *v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rax
  _QWORD *v51; // r15
  int *v52; // rax
  int *v53; // r8
  __int64 v54; // rsi
  __int64 v55; // rcx
  _QWORD *v56; // rax
  _BYTE *v57; // rsi
  __int64 v58; // rsi
  unsigned __int64 v59; // r8
  unsigned __int64 *v60; // rdx
  _QWORD *v61; // rax
  _QWORD *v62; // rbx
  char v63; // bl
  __int64 v65; // r13
  _QWORD *v66; // rdi
  int *v67; // rax
  _QWORD *v68; // r14
  __int64 v69; // rax
  __int64 v70; // rdx
  _QWORD *v71; // rax
  _BYTE *m; // rsi
  unsigned __int64 v73; // rdx
  bool v74; // al
  int *v75; // rax
  __int64 v76; // r8
  __int64 v77; // rcx
  __int64 v78; // rdx
  int v79; // eax
  int v80; // ecx
  int v81; // ecx
  __int64 v82; // rdi
  __int64 *v83; // r8
  int v84; // r9d
  unsigned int v85; // r14d
  __int64 v86; // rsi
  unsigned int v87; // ecx
  unsigned int v88; // eax
  _QWORD *v89; // rdi
  int v90; // ebx
  _QWORD *v91; // rax
  __int64 v92; // rax
  int v93; // r14d
  int *v94; // rax
  unsigned __int64 v95; // rcx
  __int64 v96; // rsi
  __int64 v97; // rdi
  __int64 v98; // rdx
  __int64 v99; // rax
  int *v100; // rax
  int *v101; // r9
  int *v102; // rdx
  __int64 v103; // rsi
  __int64 v104; // rcx
  __int64 v105; // r14
  _QWORD *v106; // r11
  int *v107; // rsi
  __int64 v108; // rcx
  __int64 v109; // rdx
  __int64 v110; // rax
  unsigned __int64 v111; // r8
  unsigned __int64 *v112; // rdx
  __int64 v113; // r12
  _QWORD *v114; // r13
  __int64 v115; // rsi
  unsigned __int64 v116; // rax
  unsigned __int64 v117; // rax
  __int64 v118; // rax
  _QWORD *v119; // rax
  __int64 v120; // rdx
  _QWORD *k; // rdx
  __int64 v122; // r9
  int v123; // r14d
  __int64 *v124; // r9
  int *v125; // [rsp+8h] [rbp-198h]
  __int64 v126; // [rsp+18h] [rbp-188h]
  unsigned __int64 v127; // [rsp+20h] [rbp-180h]
  __int64 v128; // [rsp+30h] [rbp-170h]
  __int64 v129; // [rsp+38h] [rbp-168h]
  __int64 v130; // [rsp+48h] [rbp-158h]
  _QWORD *v131; // [rsp+48h] [rbp-158h]
  __int64 v132; // [rsp+50h] [rbp-150h]
  unsigned __int8 v133; // [rsp+5Fh] [rbp-141h]
  __int64 v134; // [rsp+60h] [rbp-140h]
  _QWORD *v135; // [rsp+60h] [rbp-140h]
  char v137; // [rsp+68h] [rbp-138h]
  __int64 v139; // [rsp+70h] [rbp-130h]
  __int64 v140; // [rsp+70h] [rbp-130h]
  int v141; // [rsp+70h] [rbp-130h]
  __int64 v142; // [rsp+70h] [rbp-130h]
  __int64 v143; // [rsp+78h] [rbp-128h]
  _QWORD *v144; // [rsp+78h] [rbp-128h]
  char v145; // [rsp+8Fh] [rbp-111h] BYREF
  _QWORD *v146; // [rsp+90h] [rbp-110h] BYREF
  _QWORD *v147; // [rsp+98h] [rbp-108h] BYREF
  unsigned __int64 v148; // [rsp+A0h] [rbp-100h] BYREF
  unsigned __int64 *v149; // [rsp+A8h] [rbp-F8h] BYREF
  _QWORD *v150; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned __int64 **v151; // [rsp+B8h] [rbp-E8h] BYREF
  unsigned __int64 v152; // [rsp+C0h] [rbp-E0h] BYREF
  _BYTE *v153; // [rsp+C8h] [rbp-D8h]
  _BYTE *v154; // [rsp+D0h] [rbp-D0h]
  __int64 v155; // [rsp+E0h] [rbp-C0h] BYREF
  int v156; // [rsp+E8h] [rbp-B8h] BYREF
  int *v157; // [rsp+F0h] [rbp-B0h]
  int *v158; // [rsp+F8h] [rbp-A8h]
  int *v159; // [rsp+100h] [rbp-A0h]
  __int64 v160; // [rsp+108h] [rbp-98h]
  __int64 v161; // [rsp+110h] [rbp-90h] BYREF
  int v162; // [rsp+118h] [rbp-88h] BYREF
  int *v163; // [rsp+120h] [rbp-80h]
  int *v164; // [rsp+128h] [rbp-78h]
  int *v165; // [rsp+130h] [rbp-70h]
  __int64 v166; // [rsp+138h] [rbp-68h]
  unsigned __int64 v167; // [rsp+140h] [rbp-60h] BYREF
  int v168; // [rsp+148h] [rbp-58h] BYREF
  int *v169; // [rsp+150h] [rbp-50h]
  int *v170; // [rsp+158h] [rbp-48h]
  int *v171; // [rsp+160h] [rbp-40h]
  __int64 v172; // [rsp+168h] [rbp-38h]

  sub_2D1B830(a1[10]);
  a1[10] = 0;
  a1[11] = a1 + 9;
  a1[12] = a1 + 9;
  a1[13] = 0;
  v2 = *(_QWORD *)(a2 + 80);
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v143 = a2 + 72;
  if ( v2 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    v3 = a1 + 8;
    do
    {
      if ( !v2 )
      {
        v167 = 0;
        BUG();
      }
      v4 = 0;
      v167 = v2 - 24;
      v5 = *(_QWORD *)(v2 + 32);
      if ( v5 != v2 + 24 )
      {
        do
        {
          v6 = v5 - 24;
          if ( !v5 )
            v6 = 0;
          v7 = sub_CEA640(v6);
          v5 = *(_QWORD *)(v5 + 8);
          v4 -= !v7 - 1;
        }
        while ( v2 + 24 != v5 );
        if ( v4 > 1 )
        {
          if ( SLODWORD(qword_5016428[8]) > 1 )
          {
            v57 = v153;
            if ( v153 == v154 )
            {
              sub_9319A0((__int64)&v152, v153, &v167);
            }
            else
            {
              if ( v153 )
              {
                *(_QWORD *)v153 = v167;
                v57 = v153;
              }
              v153 = v57 + 8;
            }
          }
          sub_22DC880(v3, &v167);
        }
      }
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v143 != v2 );
    v8 = (__int64)a1;
    if ( !a1[13] )
    {
      v13 = v152;
      v133 = 0;
      goto LABEL_97;
    }
    v133 = 0;
    for ( i = *(_QWORD *)(a2 + 80); v2 != i; i = *(_QWORD *)(a2 + 80) )
    {
      v10 = 0;
      do
      {
        v11 = i - 24;
        if ( !i )
          v11 = 0;
        v12 = sub_2D1C160(a1, v11);
        i = *(_QWORD *)(i + 8);
        v10 |= v12;
      }
      while ( v2 != i );
      if ( !v10 )
        break;
      v133 = v10;
    }
    v13 = v152;
    if ( SLODWORD(qword_5016428[8]) > 1 )
    {
      v14 = (__int64)&v153[-v152] >> 3;
      if ( (_DWORD)v14 )
      {
        v132 = 0;
        v128 = (__int64)(a1 + 14);
        v129 = 8LL * (unsigned int)v14;
        do
        {
          v15 = *(_QWORD *)(v13 + v132);
          ++*(_QWORD *)(v8 + 112);
          v134 = v15;
          v16 = *(_DWORD *)(v8 + 128);
          if ( v16 )
          {
            v87 = 4 * v16;
            v17 = *(unsigned int *)(v8 + 136);
            if ( (unsigned int)(4 * v16) < 0x40 )
              v87 = 64;
            if ( (unsigned int)v17 <= v87 )
            {
LABEL_26:
              v18 = *(_QWORD **)(v8 + 120);
              for ( j = &v18[2 * v17]; j != v18; v18 += 2 )
                *v18 = -4096;
              *(_QWORD *)(v8 + 128) = 0;
              goto LABEL_29;
            }
            v88 = v16 - 1;
            if ( !v88 )
            {
              v89 = *(_QWORD **)(v8 + 120);
              v90 = 64;
LABEL_209:
              sub_C7D6A0((__int64)v89, 16LL * *(unsigned int *)(v8 + 136), 8);
              v116 = (4 * v90 / 3u + 1) | ((unsigned __int64)(4 * v90 / 3u + 1) >> 1);
              v117 = (((v116 >> 2) | v116) >> 4) | (v116 >> 2) | v116;
              v118 = ((((v117 >> 8) | v117) >> 16) | (v117 >> 8) | v117) + 1;
              *(_DWORD *)(v8 + 136) = v118;
              v119 = (_QWORD *)sub_C7D670(16 * v118, 8);
              v120 = *(unsigned int *)(v8 + 136);
              *(_QWORD *)(v8 + 128) = 0;
              *(_QWORD *)(v8 + 120) = v119;
              for ( k = &v119[2 * v120]; k != v119; v119 += 2 )
              {
                if ( v119 )
                  *v119 = -4096;
              }
              goto LABEL_29;
            }
            _BitScanReverse(&v88, v88);
            v89 = *(_QWORD **)(v8 + 120);
            v90 = 1 << (33 - (v88 ^ 0x1F));
            if ( v90 < 64 )
              v90 = 64;
            if ( (_DWORD)v17 != v90 )
              goto LABEL_209;
            *(_QWORD *)(v8 + 128) = 0;
            v91 = &v89[2 * (unsigned int)v17];
            do
            {
              if ( v89 )
                *v89 = -4096;
              v89 += 2;
            }
            while ( v91 != v89 );
          }
          else if ( *(_DWORD *)(v8 + 132) )
          {
            v17 = *(unsigned int *)(v8 + 136);
            if ( (unsigned int)v17 <= 0x40 )
              goto LABEL_26;
            sub_C7D6A0(*(_QWORD *)(v8 + 120), 16LL * *(unsigned int *)(v8 + 136), 8);
            *(_QWORD *)(v8 + 120) = 0;
            *(_QWORD *)(v8 + 128) = 0;
            *(_DWORD *)(v8 + 136) = 0;
          }
LABEL_29:
          v20 = *(_QWORD *)(v134 + 56);
          v21 = v134 + 48;
          v144 = (_QWORD *)(v134 + 48);
          if ( v20 != v134 + 48 )
          {
            v22 = 0;
            while ( 1 )
            {
              v27 = *(_DWORD *)(v8 + 136);
              v28 = v20 - 24;
              if ( !v20 )
                v28 = 0;
              ++v22;
              if ( !v27 )
                break;
              v23 = *(_QWORD *)(v8 + 120);
              v24 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
              v25 = (__int64 *)(v23 + 16LL * v24);
              v26 = *v25;
              if ( v28 == *v25 )
              {
LABEL_32:
                *((_DWORD *)v25 + 2) = v22;
                v20 = *(_QWORD *)(v20 + 8);
                if ( v21 == v20 )
                  goto LABEL_42;
              }
              else
              {
                v141 = 1;
                v34 = 0;
                while ( v26 != -4096 )
                {
                  if ( v26 == -8192 && !v34 )
                    v34 = v25;
                  v24 = (v27 - 1) & (v141 + v24);
                  v25 = (__int64 *)(v23 + 16LL * v24);
                  v26 = *v25;
                  if ( v28 == *v25 )
                    goto LABEL_32;
                  ++v141;
                }
                if ( !v34 )
                  v34 = v25;
                v79 = *(_DWORD *)(v8 + 128);
                ++*(_QWORD *)(v8 + 112);
                v33 = v79 + 1;
                if ( 4 * v33 < 3 * v27 )
                {
                  if ( v27 - *(_DWORD *)(v8 + 132) - v33 <= v27 >> 3 )
                  {
                    v142 = v21;
                    sub_9BAAD0(v128, v27);
                    v80 = *(_DWORD *)(v8 + 136);
                    if ( !v80 )
                    {
LABEL_239:
                      ++*(_DWORD *)(v8 + 128);
                      BUG();
                    }
                    v81 = v80 - 1;
                    v82 = *(_QWORD *)(v8 + 120);
                    v83 = 0;
                    v84 = 1;
                    v85 = v81 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                    v21 = v142;
                    v33 = *(_DWORD *)(v8 + 128) + 1;
                    v34 = (__int64 *)(v82 + 16LL * v85);
                    v86 = *v34;
                    if ( v28 != *v34 )
                    {
                      while ( v86 != -4096 )
                      {
                        if ( !v83 && v86 == -8192 )
                          v83 = v34;
                        v85 = v81 & (v84 + v85);
                        v34 = (__int64 *)(v82 + 16LL * v85);
                        v86 = *v34;
                        if ( v28 == *v34 )
                          goto LABEL_39;
                        ++v84;
                      }
                      if ( v83 )
                        v34 = v83;
                    }
                  }
                  goto LABEL_39;
                }
LABEL_37:
                v139 = v21;
                sub_9BAAD0(v128, 2 * v27);
                v29 = *(_DWORD *)(v8 + 136);
                if ( !v29 )
                  goto LABEL_239;
                v30 = v29 - 1;
                v31 = *(_QWORD *)(v8 + 120);
                v21 = v139;
                v32 = v30 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                v33 = *(_DWORD *)(v8 + 128) + 1;
                v34 = (__int64 *)(v31 + 16LL * v32);
                v35 = *v34;
                if ( v28 != *v34 )
                {
                  v123 = 1;
                  v124 = 0;
                  while ( v35 != -4096 )
                  {
                    if ( v35 == -8192 && !v124 )
                      v124 = v34;
                    v32 = v30 & (v123 + v32);
                    v34 = (__int64 *)(v31 + 16LL * v32);
                    v35 = *v34;
                    if ( v28 == *v34 )
                      goto LABEL_39;
                    ++v123;
                  }
                  if ( v124 )
                    v34 = v124;
                }
LABEL_39:
                *(_DWORD *)(v8 + 128) = v33;
                if ( *v34 != -4096 )
                  --*(_DWORD *)(v8 + 132);
                *v34 = v28;
                *((_DWORD *)v34 + 2) = 0;
                *((_DWORD *)v34 + 2) = v22;
                v20 = *(_QWORD *)(v20 + 8);
                if ( v21 == v20 )
                {
LABEL_42:
                  v36 = *(_QWORD **)(v134 + 56);
                  goto LABEL_43;
                }
              }
            }
            ++*(_QWORD *)(v8 + 112);
            goto LABEL_37;
          }
          v36 = (_QWORD *)(v134 + 48);
LABEL_43:
          v156 = 0;
          v37 = &v156;
          v164 = &v162;
          v157 = 0;
          v158 = &v156;
          v159 = &v156;
          v160 = 0;
          v162 = 0;
          v163 = 0;
          v165 = &v162;
          v166 = 0;
          v168 = 0;
          v169 = 0;
          v170 = &v168;
          v171 = &v168;
          v172 = 0;
          v146 = 0;
          if ( v36 != v144 )
          {
            v140 = 0;
            v38 = 0;
            v39 = 0;
            v130 = v8;
            while ( 1 )
            {
              v40 = v36 - 3;
              v41 = &v156;
              if ( !v36 )
                v40 = 0;
              v150 = v40;
              if ( !v38 )
                goto LABEL_54;
              do
              {
                while ( 1 )
                {
                  v42 = *((_QWORD *)v38 + 2);
                  v43 = *((_QWORD *)v38 + 3);
                  if ( *((_QWORD *)v38 + 4) >= (unsigned __int64)v40 )
                    break;
                  v38 = (int *)*((_QWORD *)v38 + 3);
                  if ( !v43 )
                    goto LABEL_52;
                }
                v41 = v38;
                v38 = (int *)*((_QWORD *)v38 + 2);
              }
              while ( v42 );
LABEL_52:
              if ( v41 == &v156 || *((_QWORD *)v41 + 4) > (unsigned __int64)v40 )
              {
LABEL_54:
                v151 = &v150;
                v41 = (int *)sub_2D1BF10(&v155, (__int64)v41, (unsigned __int64 **)&v151);
              }
              v44 = v163;
              v45 = v146;
              *((_QWORD *)v41 + 5) = v39;
              if ( !v44 )
                break;
              v46 = v150;
              v47 = &v162;
              do
              {
                while ( 1 )
                {
                  v48 = *((_QWORD *)v44 + 2);
                  v49 = *((_QWORD *)v44 + 3);
                  if ( *((_QWORD *)v44 + 4) >= (unsigned __int64)v150 )
                    break;
                  v44 = (int *)*((_QWORD *)v44 + 3);
                  if ( !v49 )
                    goto LABEL_60;
                }
                v47 = v44;
                v44 = (int *)*((_QWORD *)v44 + 2);
              }
              while ( v48 );
LABEL_60:
              if ( v47 == &v162 || *((_QWORD *)v47 + 4) > (unsigned __int64)v150 )
                goto LABEL_62;
LABEL_63:
              *((_QWORD *)v47 + 5) = v45;
              if ( sub_CEA640((__int64)v46) )
                v39 = v150;
              v51 = v150;
              if ( (unsigned __int8)sub_B46490((__int64)v150) )
              {
                if ( v146 )
                {
                  v52 = v169;
                  v53 = &v168;
                  if ( !v169 )
                    goto LABEL_74;
                  do
                  {
                    while ( 1 )
                    {
                      v54 = *((_QWORD *)v52 + 2);
                      v55 = *((_QWORD *)v52 + 3);
                      if ( *((_QWORD *)v52 + 4) >= (unsigned __int64)v146 )
                        break;
                      v52 = (int *)*((_QWORD *)v52 + 3);
                      if ( !v55 )
                        goto LABEL_72;
                    }
                    v53 = v52;
                    v52 = (int *)*((_QWORD *)v52 + 2);
                  }
                  while ( v54 );
LABEL_72:
                  if ( v53 == &v168 || (v56 = v51, *((_QWORD *)v53 + 4) > (unsigned __int64)v146) )
                  {
LABEL_74:
                    v151 = &v146;
                    v53 = (int *)sub_2D1BF10(&v167, (__int64)v53, (unsigned __int64 **)&v151);
                    v56 = v150;
                  }
                  *((_QWORD *)v53 + 5) = v51;
                  v51 = v56;
                }
                else
                {
                  v140 = (__int64)v51;
                }
                v146 = v51;
              }
              v36 = (_QWORD *)v36[1];
              if ( v36 == v144 )
              {
                v8 = v130;
                v144 = *(_QWORD **)(v134 + 56);
                goto LABEL_102;
              }
              v38 = v157;
            }
            v47 = &v162;
LABEL_62:
            v151 = &v150;
            v50 = sub_2D1BF10(&v161, (__int64)v47, (unsigned __int64 **)&v151);
            v46 = v150;
            v47 = (int *)v50;
            goto LABEL_63;
          }
          v140 = 0;
          v39 = 0;
LABEL_102:
          v65 = v134;
          v145 = 0;
          v62 = (_QWORD *)(*(_QWORD *)(v134 + 48) & 0xFFFFFFFFFFFFFFF8LL);
          if ( v144 == v62 )
          {
            sub_2D1BA00((unsigned __int64)v169);
            sub_2D1BA00((unsigned __int64)v163);
            sub_2D1BA00((unsigned __int64)v157);
            goto LABEL_96;
          }
          v135 = v39;
          while ( 1 )
          {
            v66 = v62 - 3;
            if ( !v62 )
              v66 = 0;
            v147 = v66;
            if ( sub_CEA640((__int64)v66) )
              goto LABEL_90;
            v67 = v157;
            if ( v157 )
            {
              v58 = (__int64)v37;
              do
              {
                if ( *((_QWORD *)v67 + 4) < (unsigned __int64)v147 )
                {
                  v67 = (int *)*((_QWORD *)v67 + 3);
                }
                else
                {
                  v58 = (__int64)v67;
                  v67 = (int *)*((_QWORD *)v67 + 2);
                }
              }
              while ( v67 );
              if ( (int *)v58 != v37 && *(_QWORD *)(v58 + 32) <= (unsigned __int64)v147 )
                goto LABEL_86;
            }
            else
            {
              v58 = (__int64)v37;
            }
            v151 = &v147;
            v58 = sub_2D1BF10(&v155, v58, (unsigned __int64 **)&v151);
LABEL_86:
            v59 = *(_QWORD *)(v58 + 40);
            if ( v59 )
            {
              v60 = *(unsigned __int64 **)(v59 + 32);
              if ( v60 )
                v60 -= 3;
              sub_2D1DCF0(v8, (__int64)v147, v60, &v145, v59, v65, v140, &v155, &v161, &v167);
              goto LABEL_90;
            }
            if ( SLODWORD(qword_5016428[8]) <= 2 || (v68 = v147, *(_BYTE *)v147 == 84) )
            {
LABEL_92:
              v137 = v145;
              goto LABEL_93;
            }
            if ( (unsigned __int8)sub_B46490((__int64)v147) )
              goto LABEL_90;
            v69 = v68[2];
            if ( !v69 )
              goto LABEL_90;
            do
            {
              v70 = *(_QWORD *)(v69 + 24);
              if ( *(_BYTE *)v70 > 0x1Cu && v68[5] == *(_QWORD *)(v70 + 40) )
                goto LABEL_90;
              v69 = *(_QWORD *)(v69 + 8);
            }
            while ( v69 );
            v71 = v135;
            v149 = 0;
            v148 = (unsigned __int64)v135;
            if ( !v135 )
              goto LABEL_90;
            for ( m = v68; ; m = v147 )
            {
              v73 = v71[4];
              if ( v73 )
                v73 -= 24LL;
              v149 = (unsigned __int64 *)v73;
              v74 = sub_2D1D770(v8, m, v73, v140, &v161, &v167);
              if ( v74 )
                break;
              v75 = v157;
              if ( !v157 )
              {
                v76 = (__int64)v37;
LABEL_132:
                v151 = (unsigned __int64 **)&v148;
                v76 = sub_2D1BF10(&v155, v76, (unsigned __int64 **)&v151);
                goto LABEL_133;
              }
              v76 = (__int64)v37;
              do
              {
                while ( 1 )
                {
                  v77 = *((_QWORD *)v75 + 2);
                  v78 = *((_QWORD *)v75 + 3);
                  if ( *((_QWORD *)v75 + 4) >= v148 )
                    break;
                  v75 = (int *)*((_QWORD *)v75 + 3);
                  if ( !v78 )
                    goto LABEL_130;
                }
                v76 = (__int64)v75;
                v75 = (int *)*((_QWORD *)v75 + 2);
              }
              while ( v77 );
LABEL_130:
              if ( (int *)v76 == v37 || *(_QWORD *)(v76 + 32) > v148 )
                goto LABEL_132;
LABEL_133:
              v71 = *(_QWORD **)(v76 + 40);
              v148 = (unsigned __int64)v71;
              if ( !v71 )
                goto LABEL_90;
            }
            v137 = v74;
            if ( v148 )
              break;
LABEL_90:
            v61 = *(_QWORD **)(v65 + 56);
LABEL_91:
            v62 = (_QWORD *)(*v62 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v62 == v61 )
              goto LABEL_92;
          }
          v62 = (_QWORD *)v62[1];
          v92 = v126;
          LOWORD(v92) = 0;
          v126 = v92;
          sub_B444E0(v147, (__int64)(v149 + 3), v92);
          v150 = v147;
          v151 = (unsigned __int64 **)v149;
          v93 = *(_DWORD *)sub_DA65E0(v128, (__int64 *)&v151);
          *(_DWORD *)sub_DA65E0(v128, (__int64 *)&v150) = v93;
          v94 = v157;
          v95 = v148;
          if ( !v157 )
          {
            v96 = (__int64)v37;
            goto LABEL_174;
          }
          v96 = (__int64)v37;
          do
          {
            while ( 1 )
            {
              v97 = *((_QWORD *)v94 + 2);
              v98 = *((_QWORD *)v94 + 3);
              if ( *((_QWORD *)v94 + 4) >= (unsigned __int64)v147 )
                break;
              v94 = (int *)*((_QWORD *)v94 + 3);
              if ( !v98 )
                goto LABEL_172;
            }
            v96 = (__int64)v94;
            v94 = (int *)*((_QWORD *)v94 + 2);
          }
          while ( v97 );
LABEL_172:
          if ( (int *)v96 == v37 || *(_QWORD *)(v96 + 32) > (unsigned __int64)v147 )
          {
LABEL_174:
            v127 = v148;
            v151 = &v147;
            v99 = sub_2D1BF10(&v155, v96, (unsigned __int64 **)&v151);
            v95 = v127;
            v96 = v99;
          }
          v100 = v163;
          *(_QWORD *)(v96 + 40) = v95;
          if ( v100 )
          {
            v101 = &v162;
            v102 = v100;
            do
            {
              while ( 1 )
              {
                v103 = *((_QWORD *)v102 + 2);
                v104 = *((_QWORD *)v102 + 3);
                if ( *((_QWORD *)v102 + 4) >= (unsigned __int64)v149 )
                  break;
                v102 = (int *)*((_QWORD *)v102 + 3);
                if ( !v104 )
                  goto LABEL_180;
              }
              v101 = v102;
              v102 = (int *)*((_QWORD *)v102 + 2);
            }
            while ( v103 );
LABEL_180:
            if ( v101 != &v162 && *((_QWORD *)v101 + 4) <= (unsigned __int64)v149 )
            {
              v105 = *((_QWORD *)v101 + 5);
LABEL_183:
              v106 = v147;
              v107 = &v162;
              do
              {
                while ( 1 )
                {
                  v108 = *((_QWORD *)v100 + 2);
                  v109 = *((_QWORD *)v100 + 3);
                  if ( *((_QWORD *)v100 + 4) >= (unsigned __int64)v147 )
                    break;
                  v100 = (int *)*((_QWORD *)v100 + 3);
                  if ( !v109 )
                    goto LABEL_187;
                }
                v107 = v100;
                v100 = (int *)*((_QWORD *)v100 + 2);
              }
              while ( v108 );
LABEL_187:
              if ( v107 == &v162 || *((_QWORD *)v107 + 4) > (unsigned __int64)v147 )
                goto LABEL_189;
              goto LABEL_190;
            }
          }
          else
          {
            v101 = &v162;
          }
          v151 = &v149;
          v122 = sub_2D1BF10(&v161, (__int64)v101, (unsigned __int64 **)&v151);
          v100 = v163;
          v105 = *(_QWORD *)(v122 + 40);
          if ( v163 )
            goto LABEL_183;
          v107 = &v162;
LABEL_189:
          v151 = &v147;
          v110 = sub_2D1BF10(&v161, (__int64)v107, (unsigned __int64 **)&v151);
          v106 = v147;
          v107 = (int *)v110;
LABEL_190:
          *((_QWORD *)v107 + 5) = v105;
          v111 = v148;
          if ( v106 )
          {
            v106 += 3;
            if ( !v148 )
            {
              v131 = 0;
              goto LABEL_193;
            }
            v131 = (_QWORD *)(v148 + 24);
            if ( v106 != (_QWORD *)(v148 + 24) )
            {
LABEL_193:
              v112 = v149;
              v125 = v37;
              v113 = v65;
              v114 = v106;
              while ( 1 )
              {
                v115 = (__int64)(v114 - 3);
                if ( !v114 )
                  v115 = 0;
                v149 = sub_2D1DCF0(v8, v115, v112, &v145, v111, v113, v140, &v155, &v161, &v167);
                v112 = v149;
                v114 = (_QWORD *)(*v114 & 0xFFFFFFFFFFFFFFF8LL);
                if ( v131 == v114 )
                  break;
                v111 = v148;
              }
              v65 = v113;
              v37 = v125;
            }
          }
          else
          {
            v131 = (_QWORD *)(v148 + 24);
            if ( v148 )
              goto LABEL_193;
          }
          v61 = *(_QWORD **)(v65 + 56);
          v145 = 1;
          if ( v62 != v61 )
            goto LABEL_91;
LABEL_93:
          sub_2D1BA00((unsigned __int64)v169);
          sub_2D1BA00((unsigned __int64)v163);
          sub_2D1BA00((unsigned __int64)v157);
          v63 = v133;
          if ( v137 )
            v63 = v137;
          v133 = v63;
LABEL_96:
          v132 += 8;
          v13 = v152;
        }
        while ( v129 != v132 );
      }
    }
LABEL_97:
    if ( v13 )
      j_j___libc_free_0(v13);
  }
  return v133;
}
