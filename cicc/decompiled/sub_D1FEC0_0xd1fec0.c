// Function: sub_D1FEC0
// Address: 0xd1fec0
//
__int64 __fastcall sub_D1FEC0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rsi
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // esi
  int v8; // r11d
  __int64 v9; // r9
  __int64 *v10; // rdx
  unsigned int v11; // r8d
  _QWORD *v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // rbx
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // r13
  __int64 v18; // rax
  __int64 *v19; // r9
  __int64 v20; // rax
  char *v21; // r14
  __int64 *v22; // r14
  __int64 *j; // r13
  __int64 v24; // rsi
  __int64 v25; // rcx
  int v26; // eax
  int v27; // edx
  unsigned int v28; // eax
  __int64 *v29; // rbx
  __int64 v30; // rdi
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // rsi
  __int64 *v33; // r14
  unsigned __int64 v34; // r15
  __int64 v35; // r8
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rdx
  __int64 i; // r15
  unsigned __int64 *v41; // r14
  __int64 v42; // rbx
  unsigned __int64 *v43; // rcx
  _BYTE *v44; // r14
  char v45; // al
  __int64 v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // r8
  unsigned __int64 v49; // rbx
  void *v50; // r13
  _QWORD *v51; // rdx
  _QWORD *v52; // rax
  char v53; // al
  __int64 v54; // r14
  __int64 v55; // rdi
  __int64 v56; // rax
  const void *v57; // rsi
  size_t v58; // rdx
  __int64 *v59; // rdx
  __int64 v60; // rbx
  unsigned __int64 v61; // r13
  __int64 *v63; // r14
  __int64 *v64; // r13
  __int64 v65; // rsi
  __int64 v66; // rcx
  int v67; // eax
  int v68; // edx
  unsigned int v69; // eax
  __int64 *v70; // rbx
  __int64 v71; // rdi
  unsigned __int64 v72; // r12
  int v73; // r8d
  __int64 v74; // rsi
  unsigned __int64 v75; // r12
  char v76; // al
  unsigned __int64 v77; // rbx
  unsigned int v78; // r12d
  int v79; // r11d
  __int64 v80; // r9
  _QWORD *v81; // rax
  unsigned int v82; // r8d
  _QWORD *v83; // rdx
  __int64 v84; // rdi
  __int64 *v85; // r14
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // r13
  unsigned int v88; // esi
  __int64 v89; // r13
  int v90; // esi
  int v91; // esi
  __int64 v92; // r9
  __int64 v93; // rcx
  int v94; // edx
  __int64 v95; // rdi
  int v96; // edi
  int v97; // esi
  int v98; // esi
  _QWORD *v99; // r8
  __int64 v100; // r9
  int v101; // r11d
  __int64 v102; // rcx
  __int64 v103; // rdi
  _QWORD *v104; // rax
  unsigned __int64 v105; // rcx
  void *v106; // r9
  _QWORD *v107; // rdx
  _QWORD *v108; // rax
  char v109; // al
  __int64 v110; // rdx
  __int64 v111; // rdi
  __int64 v112; // rax
  const void *v113; // rsi
  size_t v114; // rdx
  __int64 v115; // rax
  __int64 v116; // r8
  __int64 v117; // r14
  unsigned __int64 v118; // r13
  unsigned __int64 v119; // rax
  __int64 v120; // rcx
  unsigned __int64 v121; // rcx
  __int64 v122; // rax
  __int64 *v123; // rax
  __int64 *v124; // rsi
  _QWORD *v125; // r9
  _QWORD *v126; // rax
  _QWORD *v127; // rdi
  __int64 v128; // rcx
  __int64 v129; // rdx
  __int64 *v130; // rax
  __int64 v131; // rdx
  __int64 v132; // rcx
  __int64 v133; // rdi
  __int64 *v134; // rcx
  char v135; // al
  unsigned __int64 v136; // r13
  unsigned __int64 v137; // r12
  __int64 v138; // rcx
  int v139; // r8d
  unsigned __int64 v140; // rdx
  __int64 v141; // rax
  int v142; // eax
  int v143; // edi
  int v144; // r9d
  int v145; // r9d
  __int64 v146; // r10
  unsigned int v147; // eax
  __int64 v148; // rbx
  int v149; // r8d
  __int64 *v150; // rsi
  int v151; // r8d
  int v152; // r8d
  __int64 v153; // r10
  int v154; // esi
  unsigned int v155; // r13d
  __int64 *v156; // rax
  __int64 v157; // r9
  int v158; // r11d
  __int64 v160; // [rsp+10h] [rbp-E0h]
  void *v161; // [rsp+10h] [rbp-E0h]
  __int64 v162; // [rsp+18h] [rbp-D8h]
  unsigned __int64 src; // [rsp+28h] [rbp-C8h]
  char *srca; // [rsp+28h] [rbp-C8h]
  __int64 v165; // [rsp+30h] [rbp-C0h]
  void *v166; // [rsp+30h] [rbp-C0h]
  int v167; // [rsp+30h] [rbp-C0h]
  __int64 v168; // [rsp+30h] [rbp-C0h]
  unsigned __int64 *v169; // [rsp+38h] [rbp-B8h]
  unsigned int v170; // [rsp+38h] [rbp-B8h]
  _QWORD *v171; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v172; // [rsp+38h] [rbp-B8h]
  _QWORD *v173; // [rsp+38h] [rbp-B8h]
  __int64 v174; // [rsp+38h] [rbp-B8h]
  __int64 v175; // [rsp+40h] [rbp-B0h]
  __int64 *v176; // [rsp+40h] [rbp-B0h]
  __int64 v177; // [rsp+40h] [rbp-B0h]
  __int64 *v178; // [rsp+48h] [rbp-A8h]
  __int64 v179; // [rsp+48h] [rbp-A8h]
  __int64 v180; // [rsp+48h] [rbp-A8h]
  __int64 v181; // [rsp+48h] [rbp-A8h]
  __int64 v182; // [rsp+48h] [rbp-A8h]
  int v183; // [rsp+48h] [rbp-A8h]
  _QWORD v184[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v185; // [rsp+60h] [rbp-90h]
  __int64 v186; // [rsp+68h] [rbp-88h]
  __int64 v187; // [rsp+70h] [rbp-80h]
  __int64 v188; // [rsp+78h] [rbp-78h]
  __int64 v189; // [rsp+80h] [rbp-70h]
  __int64 v190; // [rsp+88h] [rbp-68h]
  __int64 *v191; // [rsp+90h] [rbp-60h]
  __int64 *v192; // [rsp+98h] [rbp-58h]
  __int64 v193; // [rsp+A0h] [rbp-50h]
  __int64 v194; // [rsp+A8h] [rbp-48h]
  __int64 v195; // [rsp+B0h] [rbp-40h]
  __int64 v196; // [rsp+B8h] [rbp-38h]

  v3 = a2[7];
  v184[0] = 0;
  v184[1] = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v189 = 0;
  v190 = 0;
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  sub_D126D0((__int64)v184, v3);
  sub_D12BD0((__int64)v184);
  v4 = v191;
  if ( v191 == v192 )
    goto LABEL_86;
  do
  {
    v5 = *v4;
    v6 = *(_QWORD *)(v5 + 8);
    if ( !v6 || (unsigned __int8)sub_B2FC00(*(_BYTE **)(v5 + 8)) )
    {
      v63 = v192;
      v64 = v191;
      if ( v191 == v192 )
        goto LABEL_85;
      while ( 1 )
      {
        v65 = *(_QWORD *)(a1 + 280);
        v66 = *(_QWORD *)(*v64 + 8);
        v67 = *(_DWORD *)(a1 + 296);
        if ( v67 )
        {
          v68 = v67 - 1;
          v69 = (v67 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v70 = (__int64 *)(v65 + 16LL * v69);
          v71 = *v70;
          if ( v66 == *v70 )
          {
LABEL_99:
            v72 = v70[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( v72 )
            {
              if ( (*(_BYTE *)(v72 + 8) & 1) == 0 )
                sub_C7D6A0(*(_QWORD *)(v72 + 16), 16LL * *(unsigned int *)(v72 + 24), 8);
              j_j___libc_free_0(v72, 272);
            }
            *v70 = -8192;
            --*(_DWORD *)(a1 + 288);
            ++*(_DWORD *)(a1 + 292);
          }
          else
          {
            v73 = 1;
            while ( v71 != -4096 )
            {
              v69 = v68 & (v73 + v69);
              v70 = (__int64 *)(v65 + 16LL * v69);
              v71 = *v70;
              if ( v66 == *v70 )
                goto LABEL_99;
              ++v73;
            }
          }
        }
        if ( v63 == ++v64 )
          goto LABEL_85;
      }
    }
    v7 = *(_DWORD *)(a1 + 296);
    v162 = a1 + 272;
    if ( v7 )
    {
      v8 = 1;
      v9 = *(_QWORD *)(a1 + 280);
      v10 = 0;
      v11 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v12 = (_QWORD *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v6 == *v12 )
      {
LABEL_6:
        v14 = v12 + 1;
        goto LABEL_7;
      }
      while ( v13 != -4096 )
      {
        if ( !v10 && v13 == -8192 )
          v10 = v12;
        v11 = (v7 - 1) & (v8 + v11);
        v12 = (_QWORD *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v6 == *v12 )
          goto LABEL_6;
        ++v8;
      }
      if ( !v10 )
        v10 = v12;
      v142 = *(_DWORD *)(a1 + 288);
      ++*(_QWORD *)(a1 + 272);
      v143 = v142 + 1;
      if ( 4 * (v142 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a1 + 292) - v143 <= v7 >> 3 )
        {
          sub_D1E430(v162, v7);
          v151 = *(_DWORD *)(a1 + 296);
          if ( !v151 )
          {
LABEL_305:
            ++*(_DWORD *)(a1 + 288);
            BUG();
          }
          v152 = v151 - 1;
          v153 = *(_QWORD *)(a1 + 280);
          v154 = 1;
          v155 = v152 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v143 = *(_DWORD *)(a1 + 288) + 1;
          v156 = 0;
          v10 = (__int64 *)(v153 + 16LL * v155);
          v157 = *v10;
          if ( v6 != *v10 )
          {
            while ( v157 != -4096 )
            {
              if ( v157 == -8192 && !v156 )
                v156 = v10;
              v155 = v152 & (v154 + v155);
              v10 = (__int64 *)(v153 + 16LL * v155);
              v157 = *v10;
              if ( v6 == *v10 )
                goto LABEL_251;
              ++v154;
            }
            if ( v156 )
              v10 = v156;
          }
        }
        goto LABEL_251;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 272);
    }
    sub_D1E430(v162, 2 * v7);
    v144 = *(_DWORD *)(a1 + 296);
    if ( !v144 )
      goto LABEL_305;
    v145 = v144 - 1;
    v146 = *(_QWORD *)(a1 + 280);
    v147 = v145 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v143 = *(_DWORD *)(a1 + 288) + 1;
    v10 = (__int64 *)(v146 + 16LL * v147);
    v148 = *v10;
    if ( v6 != *v10 )
    {
      v149 = 1;
      v150 = 0;
      while ( v148 != -4096 )
      {
        if ( !v150 && v148 == -8192 )
          v150 = v10;
        v147 = v145 & (v149 + v147);
        v10 = (__int64 *)(v146 + 16LL * v147);
        v148 = *v10;
        if ( v6 == *v10 )
          goto LABEL_251;
        ++v149;
      }
      if ( v150 )
        v10 = v150;
    }
LABEL_251:
    *(_DWORD *)(a1 + 288) = v143;
    if ( *v10 != -4096 )
      --*(_DWORD *)(a1 + 292);
    *v10 = v6;
    v14 = (unsigned __int64 *)(v10 + 1);
    v10[1] = 0;
LABEL_7:
    v15 = *(_QWORD *)(a1 + 336);
    v16 = (_QWORD *)sub_22077B0(64);
    v16[3] = 2;
    v17 = v16;
    v16[4] = 0;
    v16[5] = v6;
    if ( v6 != -4096 && v6 != -8192 )
      sub_BD73F0((__int64)(v16 + 3));
    v17[6] = a1;
    v17[7] = 0;
    v17[2] = &unk_49DDE50;
    sub_2208C80(v17, v15);
    v18 = *(_QWORD *)(a1 + 336);
    ++*(_QWORD *)(a1 + 352);
    *(_QWORD *)(v18 + 56) = v18;
    v19 = v191;
    v178 = v192;
    v20 = v192 - v191;
    if ( !(_DWORD)v20 )
    {
LABEL_33:
      v32 = *v14;
      if ( v178 != v19 )
      {
        v160 = a1;
        v33 = v19;
        v34 = *v14;
        while ( 1 )
        {
          v36 = *v33;
          v35 = v34 & 7;
          if ( (v34 & 3) == 3 )
          {
LABEL_113:
            v32 = v34;
            a1 = v160;
            goto LABEL_114;
          }
          if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v36 + 8), 48) )
            goto LABEL_35;
          v37 = *(_QWORD *)(v36 + 8);
          v38 = *(_QWORD *)(v37 + 80);
          if ( v37 + 72 == v38 )
          {
            v176 = v33;
            v41 = v14;
            v42 = v37 + 72;
            i = 0;
          }
          else
          {
            if ( !v38 )
              BUG();
            while ( 1 )
            {
              v39 = *(_QWORD *)(v38 + 32);
              if ( v39 != v38 + 24 )
                break;
              v38 = *(_QWORD *)(v38 + 8);
              if ( v37 + 72 == v38 )
                break;
              if ( !v38 )
                BUG();
            }
            v176 = v33;
            i = v39;
            v41 = v14;
            v42 = v37 + 72;
          }
          if ( v38 == v42 )
          {
LABEL_112:
            v14 = v41;
            v34 = *v41;
            v33 = v176 + 1;
            v35 = v34 & 7;
            if ( v178 == v176 + 1 )
              goto LABEL_113;
          }
          else
          {
            v43 = v41;
            while ( 2 )
            {
              v44 = (_BYTE *)(i - 24);
              if ( !i )
                v44 = 0;
              v35 = *v43 & 7;
              if ( (*v43 & 3) == 3 )
              {
                v33 = v176;
                v14 = v43;
                v34 = *v43;
                goto LABEL_36;
              }
              if ( (unsigned __int8)(*v44 - 34) > 0x33u
                || (v74 = 0x8000000000041LL, !_bittest64(&v74, (unsigned int)(unsigned __int8)*v44 - 34)) )
              {
                v169 = v43;
                src = *v43;
                v165 = *v43 & 7;
                if ( (unsigned __int8)sub_B46420((__int64)v44) )
                  *v169 = v165 | 1 | src & 0xFFFFFFFFFFFFFFF8LL;
                v45 = sub_B46490((__int64)v44);
                v43 = v169;
                if ( v45 )
                  *v169 |= 2u;
              }
              for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v38 + 32) )
              {
                v46 = v38 - 24;
                if ( !v38 )
                  v46 = 0;
                if ( i != v46 + 48 )
                  break;
                v38 = *(_QWORD *)(v38 + 8);
                if ( v42 == v38 )
                {
                  v41 = v43;
                  goto LABEL_112;
                }
                if ( !v38 )
                  BUG();
              }
              if ( v38 != v42 )
                continue;
              break;
            }
            v33 = v176;
            v14 = v43;
LABEL_35:
            v34 = *v14;
            v35 = *v14 & 7;
LABEL_36:
            if ( v178 == ++v33 )
              goto LABEL_113;
          }
        }
      }
      v35 = *v14 & 7;
LABEL_114:
      v60 = v35;
      v75 = v32 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v32 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v180 = v35;
        v47 = (_QWORD *)sub_22077B0(272);
        v48 = v180;
        v49 = (unsigned __int64)v47;
        if ( v47 )
        {
          v50 = v47 + 2;
          *v47 = 0;
          v51 = v47 + 34;
          v47[1] = 1;
          v52 = v47 + 2;
          do
          {
            if ( v52 )
              *v52 = -4096;
            v52 += 2;
          }
          while ( v52 != v51 );
          if ( (*(_BYTE *)(v49 + 8) & 1) == 0 )
          {
            sub_C7D6A0(*(_QWORD *)(v49 + 16), 16LL * *(unsigned int *)(v49 + 24), 8);
            v48 = v180;
          }
          v53 = *(_BYTE *)(v49 + 8) | 1;
          *(_BYTE *)(v49 + 8) = v53;
          if ( (*(_BYTE *)(v75 + 8) & 1) == 0 && *(_DWORD *)(v75 + 24) > 0x10u )
          {
            *(_BYTE *)(v49 + 8) = v53 & 0xFE;
            if ( (*(_BYTE *)(v75 + 8) & 1) != 0 )
            {
              v55 = 256;
              LODWORD(v54) = 16;
            }
            else
            {
              v54 = *(unsigned int *)(v75 + 24);
              v55 = 16 * v54;
            }
            v181 = v48;
            v56 = sub_C7D670(v55, 8);
            *(_DWORD *)(v49 + 24) = v54;
            v48 = v181;
            *(_QWORD *)(v49 + 16) = v56;
          }
          *(_DWORD *)(v49 + 8) = *(_DWORD *)(v75 + 8) & 0xFFFFFFFE | *(_DWORD *)(v49 + 8) & 1;
          *(_DWORD *)(v49 + 12) = *(_DWORD *)(v75 + 12);
          if ( (*(_BYTE *)(v49 + 8) & 1) == 0 )
            v50 = *(void **)(v49 + 16);
          v57 = (const void *)(v75 + 16);
          if ( (*(_BYTE *)(v75 + 8) & 1) == 0 )
            v57 = *(const void **)(v75 + 16);
          v58 = 256;
          if ( (*(_BYTE *)(v49 + 8) & 1) == 0 )
            v58 = 16LL * *(unsigned int *)(v49 + 24);
          v182 = v48;
          memcpy(v50, v57, v58);
          v48 = v182;
        }
        v59 = v191;
        v60 = v48 | v49;
        v183 = v192 - v191;
        if ( v183 == 1 )
        {
          v61 = v60 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_84:
          if ( v61 )
          {
            if ( (*(_BYTE *)(v61 + 8) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v61 + 16), 16LL * *(unsigned int *)(v61 + 24), 8);
            j_j___libc_free_0(v61, 272);
          }
          goto LABEL_85;
        }
      }
      else
      {
        v59 = v191;
        v183 = v192 - v191;
        if ( v183 == 1 )
          goto LABEL_85;
      }
      v76 = v60;
      v77 = v60 & 0xFFFFFFFFFFFFFFF8LL;
      v78 = 1;
      v177 = v76 & 7;
      while ( 1 )
      {
        v88 = *(_DWORD *)(a1 + 296);
        v89 = *(_QWORD *)(v59[v78] + 8);
        if ( !v88 )
          break;
        v79 = 1;
        v80 = *(_QWORD *)(a1 + 280);
        v81 = 0;
        v82 = (v88 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
        v83 = (_QWORD *)(v80 + 16LL * v82);
        v84 = *v83;
        if ( v89 != *v83 )
        {
          while ( v84 != -4096 )
          {
            if ( v81 || v84 != -8192 )
              v83 = v81;
            v82 = (v88 - 1) & (v79 + v82);
            v84 = *(_QWORD *)(v80 + 16LL * v82);
            if ( v89 == v84 )
            {
              v83 = (_QWORD *)(v80 + 16LL * v82);
              goto LABEL_118;
            }
            ++v79;
            v81 = v83;
            v83 = (_QWORD *)(v80 + 16LL * v82);
          }
          v96 = *(_DWORD *)(a1 + 288);
          if ( !v81 )
            v81 = v83;
          ++*(_QWORD *)(a1 + 272);
          v94 = v96 + 1;
          if ( 4 * (v96 + 1) < 3 * v88 )
          {
            if ( v88 - *(_DWORD *)(a1 + 292) - v94 <= v88 >> 3 )
            {
              v170 = ((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4);
              sub_D1E430(v162, v88);
              v97 = *(_DWORD *)(a1 + 296);
              if ( !v97 )
              {
LABEL_307:
                ++*(_DWORD *)(a1 + 288);
                BUG();
              }
              v98 = v97 - 1;
              v99 = 0;
              v100 = *(_QWORD *)(a1 + 280);
              v101 = 1;
              LODWORD(v102) = v98 & v170;
              v94 = *(_DWORD *)(a1 + 288) + 1;
              v81 = (_QWORD *)(v100 + 16LL * (v98 & v170));
              v103 = *v81;
              if ( v89 != *v81 )
              {
                while ( v103 != -4096 )
                {
                  if ( !v99 && v103 == -8192 )
                    v99 = v81;
                  v102 = v98 & (unsigned int)(v102 + v101);
                  v81 = (_QWORD *)(v100 + 16 * v102);
                  v103 = *v81;
                  if ( v89 == *v81 )
                    goto LABEL_129;
                  ++v101;
                }
LABEL_141:
                if ( v99 )
                  v81 = v99;
              }
            }
LABEL_129:
            *(_DWORD *)(a1 + 288) = v94;
            if ( *v81 != -4096 )
              --*(_DWORD *)(a1 + 292);
            *v81 = v89;
            v85 = v81 + 1;
            v81[1] = 0;
            goto LABEL_122;
          }
LABEL_127:
          sub_D1E430(v162, 2 * v88);
          v90 = *(_DWORD *)(a1 + 296);
          if ( !v90 )
            goto LABEL_307;
          v91 = v90 - 1;
          v92 = *(_QWORD *)(a1 + 280);
          LODWORD(v93) = v91 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
          v94 = *(_DWORD *)(a1 + 288) + 1;
          v81 = (_QWORD *)(v92 + 16LL * (unsigned int)v93);
          v95 = *v81;
          if ( v89 != *v81 )
          {
            v158 = 1;
            v99 = 0;
            while ( v95 != -4096 )
            {
              if ( !v99 && v95 == -8192 )
                v99 = v81;
              v93 = v91 & (unsigned int)(v93 + v158);
              v81 = (_QWORD *)(v92 + 16 * v93);
              v95 = *v81;
              if ( v89 == *v81 )
                goto LABEL_129;
              ++v158;
            }
            goto LABEL_141;
          }
          goto LABEL_129;
        }
LABEL_118:
        v85 = v83 + 1;
        v86 = v83[1] & 0xFFFFFFFFFFFFFFF8LL;
        v87 = v86;
        if ( v86 )
        {
          if ( (*(_BYTE *)(v86 + 8) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v86 + 16), 16LL * *(unsigned int *)(v86 + 24), 8);
          j_j___libc_free_0(v87, 272);
        }
LABEL_122:
        v61 = v77;
        *v85 = v177;
        if ( v77 )
        {
          v104 = (_QWORD *)sub_22077B0(272);
          v105 = (unsigned __int64)v104;
          if ( v104 )
          {
            v106 = v104 + 2;
            *v104 = 0;
            v107 = v104 + 34;
            v104[1] = 1;
            v108 = v104 + 2;
            do
            {
              if ( v108 )
                *v108 = -4096;
              v108 += 2;
            }
            while ( v108 != v107 );
            if ( (*(_BYTE *)(v105 + 8) & 1) == 0 )
            {
              v166 = v106;
              v171 = (_QWORD *)v105;
              sub_C7D6A0(*(_QWORD *)(v105 + 16), 16LL * *(unsigned int *)(v105 + 24), 8);
              v106 = v166;
              v105 = (unsigned __int64)v171;
            }
            v109 = *(_BYTE *)(v105 + 8) | 1;
            *(_BYTE *)(v105 + 8) = v109;
            if ( (*(_BYTE *)(v77 + 8) & 1) == 0 && *(_DWORD *)(v77 + 24) > 0x10u )
            {
              *(_BYTE *)(v105 + 8) = v109 & 0xFE;
              if ( (*(_BYTE *)(v77 + 8) & 1) != 0 )
              {
                v111 = 256;
                LODWORD(v110) = 16;
              }
              else
              {
                v110 = *(unsigned int *)(v77 + 24);
                v111 = 16 * v110;
              }
              v161 = v106;
              v167 = v110;
              v172 = v105;
              v112 = sub_C7D670(v111, 8);
              v105 = v172;
              v106 = v161;
              *(_QWORD *)(v172 + 16) = v112;
              *(_DWORD *)(v172 + 24) = v167;
            }
            *(_DWORD *)(v105 + 8) = *(_DWORD *)(v77 + 8) & 0xFFFFFFFE | *(_DWORD *)(v105 + 8) & 1;
            *(_DWORD *)(v105 + 12) = *(_DWORD *)(v77 + 12);
            if ( (*(_BYTE *)(v105 + 8) & 1) == 0 )
              v106 = *(void **)(v105 + 16);
            if ( (*(_BYTE *)(v77 + 8) & 1) != 0 )
              v113 = (const void *)(v77 + 16);
            else
              v113 = *(const void **)(v77 + 16);
            v114 = 256;
            if ( (*(_BYTE *)(v105 + 8) & 1) == 0 )
              v114 = 16LL * *(unsigned int *)(v105 + 24);
            v173 = (_QWORD *)v105;
            memcpy(v106, v113, v114);
            v105 = (unsigned __int64)v173;
          }
          *v85 = *v85 & 7 | v105;
        }
        if ( ++v78 == v183 )
          goto LABEL_84;
        v59 = v191;
      }
      ++*(_QWORD *)(a1 + 272);
      goto LABEL_127;
    }
    v175 = a1;
    v21 = 0;
    v179 = 8LL * (unsigned int)v20;
    while ( 2 )
    {
      while ( sub_B2FC80(v6) || (unsigned __int8)sub_B2D610(v6, 48) )
      {
        if ( sub_B2DCC0(v6) )
        {
LABEL_31:
          v21 += 8;
          if ( (char *)v179 == v21 )
            goto LABEL_32;
        }
        else if ( (unsigned __int8)sub_B2DCE0(v6) )
        {
          *v14 |= 1u;
          if ( sub_B2DD40(v6)
            || sub_B2FC80(v6) && (unsigned __int8)sub_B2D610(v6, 39) && (unsigned __int8)sub_B2D610(v6, 24) )
          {
            goto LABEL_31;
          }
          *v14 |= 4u;
          v21 += 8;
          if ( (char *)v179 == v21 )
            goto LABEL_32;
        }
        else
        {
          *v14 |= 3u;
          if ( !sub_B2DD40(v6) )
            *v14 |= 4u;
          if ( !sub_B2FC80(v6) || !(unsigned __int8)sub_B2D610(v6, 39) || !(unsigned __int8)sub_B2D610(v6, 24) )
          {
LABEL_19:
            a1 = v175;
            goto LABEL_20;
          }
          v21 += 8;
          if ( (char *)v179 == v21 )
          {
LABEL_32:
            a1 = v175;
            v19 = v191;
            v178 = v192;
            goto LABEL_33;
          }
        }
      }
      v115 = *(_QWORD *)&v21[(_QWORD)v191];
      v116 = *(_QWORD *)(v115 + 16);
      v174 = *(_QWORD *)(v115 + 24);
      if ( v174 == v116 )
        goto LABEL_31;
      if ( !*(_QWORD *)(*(_QWORD *)(v116 + 32) + 8LL) )
        goto LABEL_19;
      srca = v21;
      a1 = v175;
      v117 = v116 + 40;
      v168 = v6;
      v118 = *(_QWORD *)(*(_QWORD *)(v116 + 32) + 8LL);
      while ( 2 )
      {
        v123 = sub_D1B8E0(v175, v118);
        v124 = v123;
        if ( v123 )
        {
          v119 = *v14 & 0xFFFFFFFFFFFFFFF8LL | *v14 & 7 | *v123 & 3;
          *v14 = v119;
          v120 = *v124;
          if ( (*v124 & 4) != 0 )
          {
            *v14 = v119 | 4;
            v120 = *v124;
          }
          v121 = v120 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v121 )
            goto LABEL_177;
          v135 = *(_BYTE *)(v121 + 8) & 1;
          if ( *(_DWORD *)(v121 + 8) >> 1 )
          {
            if ( v135 )
            {
              v136 = v121 + 16;
              v137 = v121 + 272;
              do
              {
LABEL_202:
                if ( *(_QWORD *)v136 != -8192 && *(_QWORD *)v136 != -4096 )
                  break;
                v136 += 16LL;
              }
              while ( v136 != v137 );
            }
            else
            {
              v136 = *(_QWORD *)(v121 + 16);
              v138 = 16LL * *(unsigned int *)(v121 + 24);
              v137 = v136 + v138;
              if ( v136 != v136 + v138 )
                goto LABEL_202;
            }
          }
          else
          {
            if ( v135 )
            {
              v140 = v121 + 16;
              v141 = 256;
            }
            else
            {
              v140 = *(_QWORD *)(v121 + 16);
              v141 = 16LL * *(unsigned int *)(v121 + 24);
            }
            v136 = v140 + v141;
            v137 = v140 + v141;
          }
          if ( v137 != v136 )
          {
LABEL_206:
            sub_D1E0D0(v14, *(_QWORD *)v136, *(_BYTE *)(v136 + 8));
            while ( 1 )
            {
              v136 += 16LL;
              if ( v136 == v137 )
                break;
              if ( *(_QWORD *)v136 != -8192 && *(_QWORD *)v136 != -4096 )
              {
                if ( v136 != v137 )
                  goto LABEL_206;
                break;
              }
            }
          }
LABEL_177:
          if ( v174 == v117 )
            goto LABEL_197;
LABEL_178:
          v122 = *(_QWORD *)(v117 + 32);
          v117 += 40;
          v118 = *(_QWORD *)(v122 + 8);
          if ( !v118 )
            goto LABEL_20;
          continue;
        }
        break;
      }
      v125 = a2 + 2;
      v126 = (_QWORD *)a2[3];
      if ( v126 )
      {
        v127 = a2 + 2;
        do
        {
          while ( 1 )
          {
            v128 = v126[2];
            v129 = v126[3];
            if ( v126[4] >= v118 )
              break;
            v126 = (_QWORD *)v126[3];
            if ( !v129 )
              goto LABEL_185;
          }
          v127 = v126;
          v126 = (_QWORD *)v126[2];
        }
        while ( v128 );
LABEL_185:
        if ( v125 != v127 && v127[4] <= v118 )
          v125 = v127;
      }
      v130 = v191;
      v131 = v125[5];
      v132 = ((char *)v192 - (char *)v191) >> 5;
      v133 = v192 - v191;
      if ( v132 <= 0 )
        goto LABEL_230;
      v134 = &v191[4 * v132];
      do
      {
        if ( v131 == *v130 )
          goto LABEL_195;
        if ( v131 == v130[1] )
        {
          if ( v192 == v130 + 1 )
            goto LABEL_20;
          goto LABEL_196;
        }
        if ( v131 == v130[2] )
        {
          if ( v192 == v130 + 2 )
            goto LABEL_20;
          goto LABEL_196;
        }
        if ( v131 == v130[3] )
        {
          if ( v192 == v130 + 3 )
            goto LABEL_20;
          goto LABEL_196;
        }
        v130 += 4;
      }
      while ( v134 != v130 );
      v133 = v192 - v130;
LABEL_230:
      if ( v133 != 2 )
      {
        if ( v133 != 3 )
        {
          if ( v133 == 1 && v131 == *v130 )
            goto LABEL_195;
          break;
        }
        if ( v131 == *v130 )
          goto LABEL_195;
        ++v130;
      }
      if ( v131 != *v130 && v131 != *++v130 )
        break;
LABEL_195:
      if ( v192 != v130 )
      {
LABEL_196:
        if ( v174 == v117 )
        {
LABEL_197:
          v6 = v168;
          v21 = srca + 8;
          if ( (char *)v179 == srca + 8 )
            goto LABEL_32;
          continue;
        }
        goto LABEL_178;
      }
      break;
    }
LABEL_20:
    v22 = v192;
    for ( j = v191; v22 != j; ++j )
    {
      v24 = *(_QWORD *)(a1 + 280);
      v25 = *(_QWORD *)(*j + 8);
      v26 = *(_DWORD *)(a1 + 296);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = (v26 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = (__int64 *)(v24 + 16LL * v28);
        v30 = *v29;
        if ( *v29 == v25 )
        {
LABEL_26:
          v31 = v29[1] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v31 )
          {
            if ( (*(_BYTE *)(v31 + 8) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v31 + 16), 16LL * *(unsigned int *)(v31 + 24), 8);
            j_j___libc_free_0(v31, 272);
          }
          *v29 = -8192;
          --*(_DWORD *)(a1 + 288);
          ++*(_DWORD *)(a1 + 292);
        }
        else
        {
          v139 = 1;
          while ( v30 != -4096 )
          {
            v28 = v27 & (v139 + v28);
            v29 = (__int64 *)(v24 + 16LL * v28);
            v30 = *v29;
            if ( v25 == *v29 )
              goto LABEL_26;
            ++v139;
          }
        }
      }
    }
LABEL_85:
    sub_D12BD0((__int64)v184);
    v4 = v191;
  }
  while ( v191 != v192 );
LABEL_86:
  if ( v194 )
    j_j___libc_free_0(v194, v196 - v194);
  if ( v191 )
    j_j___libc_free_0(v191, v193 - (_QWORD)v191);
  if ( v188 )
    j_j___libc_free_0(v188, v190 - v188);
  return sub_C7D6A0(v185, 16LL * (unsigned int)v187, 8);
}
