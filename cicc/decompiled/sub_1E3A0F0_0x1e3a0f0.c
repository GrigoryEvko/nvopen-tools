// Function: sub_1E3A0F0
// Address: 0x1e3a0f0
//
__int64 __fastcall sub_1E3A0F0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rcx
  unsigned int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rbx
  int v8; // r12d
  unsigned int v9; // esi
  unsigned int v10; // r10d
  __int64 v11; // r9
  unsigned int v12; // r8d
  int *v13; // rax
  int v14; // edi
  __int64 v15; // r13
  int v16; // r14d
  unsigned int v17; // eax
  __int64 v18; // rax
  int v19; // ebx
  _DWORD *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 *v25; // r9
  int v26; // r8d
  unsigned int v27; // esi
  __int64 v28; // rdi
  unsigned int v29; // ecx
  int *v30; // rax
  int v31; // edx
  __int64 v32; // rax
  unsigned int v33; // esi
  int v34; // r8d
  __int64 v35; // r12
  __int64 v36; // rdi
  unsigned int v37; // ecx
  int *v38; // rax
  int v39; // edx
  __int64 v40; // rax
  unsigned int v41; // esi
  int *v42; // r12
  __int64 v43; // r8
  unsigned int v44; // ecx
  int *v45; // rax
  int v46; // edi
  __int64 v47; // rax
  int v48; // r11d
  unsigned int v49; // r14d
  int i; // r13d
  __int64 v51; // rax
  __int64 v52; // r13
  unsigned int v53; // esi
  __int64 v54; // rdi
  unsigned int v55; // ecx
  int *v56; // rax
  int v57; // edx
  __int64 v58; // rsi
  int v59; // eax
  int v61; // r10d
  int *v62; // r9
  int v63; // edx
  int v64; // edx
  int v65; // ecx
  int v66; // ecx
  __int64 v67; // r8
  __int64 v68; // rsi
  int v69; // edi
  int v70; // r10d
  int *v71; // r9
  int v72; // ecx
  int v73; // ecx
  __int64 v74; // rdi
  int v75; // r9d
  __int64 v76; // r14
  int *v77; // r8
  int v78; // esi
  int v79; // r11d
  int *v80; // r10
  int v81; // edx
  int v82; // edx
  int v83; // edx
  int v84; // r11d
  int *v85; // r10
  int v86; // edx
  int v87; // edx
  __int64 v88; // rax
  int v89; // ecx
  int v90; // ecx
  __int64 v91; // r9
  unsigned int v92; // edi
  int v93; // r8d
  int v94; // r14d
  int *v95; // r11
  int v96; // ecx
  int v97; // ecx
  __int64 v98; // r9
  unsigned int v99; // esi
  int v100; // edi
  int v101; // r14d
  int *v102; // r11
  int v103; // r13d
  int *v104; // r11
  int v105; // eax
  int v106; // eax
  int v107; // edx
  int v108; // edx
  __int64 v109; // rdi
  unsigned int v110; // esi
  int v111; // ecx
  int v112; // r9d
  int *v113; // r8
  int v114; // edx
  int v115; // edx
  __int64 v116; // rdi
  int v117; // r9d
  unsigned int v118; // esi
  int v119; // ecx
  int v120; // ecx
  int v121; // ecx
  __int64 v122; // r9
  int v123; // r14d
  unsigned int v124; // edi
  int v125; // r8d
  int v126; // ecx
  int v127; // ecx
  __int64 v128; // rdi
  int v129; // r11d
  unsigned int v130; // r14d
  int *v131; // r10
  int v132; // esi
  int *v133; // r14
  int v134; // edx
  int v135; // edx
  int v136; // r14d
  int v137; // r14d
  __int64 v138; // rdi
  unsigned int v139; // ecx
  int v140; // esi
  int *v141; // r10
  int v142; // r14d
  int v143; // r14d
  __int64 v144; // r10
  unsigned int v145; // ecx
  int v146; // esi
  int *v147; // rdi
  int *v148; // r14
  __int64 v149; // [rsp+8h] [rbp-68h]
  _DWORD *v150; // [rsp+10h] [rbp-60h]
  int v151; // [rsp+18h] [rbp-58h]
  int v152; // [rsp+20h] [rbp-50h]
  int v153; // [rsp+24h] [rbp-4Ch]
  int v154; // [rsp+24h] [rbp-4Ch]
  int v155; // [rsp+24h] [rbp-4Ch]
  int v156; // [rsp+24h] [rbp-4Ch]
  __int64 v157; // [rsp+28h] [rbp-48h]
  unsigned int v159; // [rsp+34h] [rbp-3Ch]
  __int64 v160; // [rsp+38h] [rbp-38h]

  v159 = a3;
  if ( !a3 )
    return 0;
  v160 = 0;
  v3 = *(_QWORD *)(a1 + 24);
  v157 = 4LL * a2;
  while ( 2 )
  {
    v5 = *(_DWORD *)(a1 + 276);
    if ( v5 )
    {
LABEL_4:
      v6 = 4LL * *(unsigned int *)(a1 + 272);
      goto LABEL_5;
    }
    while ( 1 )
    {
      *(_DWORD *)(a1 + 272) = a2;
      v6 = v157;
LABEL_5:
      v7 = *(_QWORD *)(a1 + 264);
      v8 = *(_DWORD *)(v3 + v6);
      v9 = *(_DWORD *)(v7 + 24);
      if ( v9 )
        break;
LABEL_29:
      v51 = sub_145CBF0((__int64 *)(a1 + 40), 80, 8);
      *(_QWORD *)v51 = 0;
      v52 = v51;
      *(_QWORD *)(v51 + 8) = 0;
      *(_QWORD *)(v51 + 16) = 0;
      *(_DWORD *)(v51 + 24) = 0;
      *(_BYTE *)(v51 + 32) = 1;
      *(_DWORD *)(v51 + 48) = -1;
      *(_DWORD *)(v51 + 36) = a2;
      *(_QWORD *)(v51 + 40) = a1 + 256;
      *(_QWORD *)(v51 + 56) = 0;
      *(_QWORD *)(v51 + 64) = v7;
      *(_QWORD *)(v51 + 72) = 0;
      v53 = *(_DWORD *)(v7 + 24);
      if ( v53 )
      {
        v54 = *(_QWORD *)(v7 + 8);
        v55 = (v53 - 1) & (37 * v8);
        v56 = (int *)(v54 + 16LL * v55);
        v57 = *v56;
        if ( v8 == *v56 )
          goto LABEL_31;
        v61 = 1;
        v62 = 0;
        while ( v57 != -1 )
        {
          if ( v57 == -2 && !v62 )
            v62 = v56;
          v55 = (v53 - 1) & (v61 + v55);
          v56 = (int *)(v54 + 16LL * v55);
          v57 = *v56;
          if ( v8 == *v56 )
            goto LABEL_31;
          ++v61;
        }
        v63 = *(_DWORD *)(v7 + 16);
        if ( v62 )
          v56 = v62;
        ++*(_QWORD *)v7;
        v64 = v63 + 1;
        if ( 4 * v64 < 3 * v53 )
        {
          if ( v53 - *(_DWORD *)(v7 + 20) - v64 <= v53 >> 3 )
          {
            sub_1E37A30(v7, v53);
            v72 = *(_DWORD *)(v7 + 24);
            if ( !v72 )
            {
LABEL_232:
              ++*(_DWORD *)(v7 + 16);
              BUG();
            }
            v73 = v72 - 1;
            v74 = *(_QWORD *)(v7 + 8);
            v75 = 1;
            LODWORD(v76) = v73 & (37 * v8);
            v77 = 0;
            v64 = *(_DWORD *)(v7 + 16) + 1;
            v56 = (int *)(v74 + 16LL * (unsigned int)v76);
            v78 = *v56;
            if ( v8 != *v56 )
            {
              while ( v78 != -1 )
              {
                if ( !v77 && v78 == -2 )
                  v77 = v56;
                v76 = v73 & (unsigned int)(v76 + v75);
                v56 = (int *)(v74 + 16 * v76);
                v78 = *v56;
                if ( v8 == *v56 )
                  goto LABEL_45;
                ++v75;
              }
              if ( v77 )
                v56 = v77;
            }
          }
          goto LABEL_45;
        }
      }
      else
      {
        ++*(_QWORD *)v7;
      }
      sub_1E37A30(v7, 2 * v53);
      v65 = *(_DWORD *)(v7 + 24);
      if ( !v65 )
        goto LABEL_232;
      v66 = v65 - 1;
      v67 = *(_QWORD *)(v7 + 8);
      LODWORD(v68) = v66 & (37 * v8);
      v64 = *(_DWORD *)(v7 + 16) + 1;
      v56 = (int *)(v67 + 16LL * (unsigned int)v68);
      v69 = *v56;
      if ( v8 != *v56 )
      {
        v70 = 1;
        v71 = 0;
        while ( v69 != -1 )
        {
          if ( !v71 && v69 == -2 )
            v71 = v56;
          v68 = v66 & (unsigned int)(v68 + v70);
          v56 = (int *)(v67 + 16 * v68);
          v69 = *v56;
          if ( v8 == *v56 )
            goto LABEL_45;
          ++v70;
        }
        if ( v71 )
          v56 = v71;
      }
LABEL_45:
      *(_DWORD *)(v7 + 16) = v64;
      if ( *v56 != -1 )
        --*(_DWORD *)(v7 + 20);
      *v56 = v8;
      *((_QWORD *)v56 + 1) = 0;
LABEL_31:
      *((_QWORD *)v56 + 1) = v52;
      v47 = *(_QWORD *)(a1 + 264);
      if ( !v160 )
        goto LABEL_22;
      v58 = v160;
      --v159;
      v160 = 0;
      *(_QWORD *)(v58 + 56) = v47;
      if ( *(_DWORD *)(v47 + 36) != -1 )
      {
LABEL_23:
        *(_QWORD *)(a1 + 264) = *(_QWORD *)(v47 + 56);
LABEL_24:
        if ( !v159 )
          return 0;
        goto LABEL_25;
      }
LABEL_33:
      v59 = *(_DWORD *)(a1 + 276);
      if ( !v59 )
        goto LABEL_24;
      *(_DWORD *)(a1 + 276) = v59 - 1;
      *(_DWORD *)(a1 + 272) = a2 + 1 - v159;
      if ( !v159 )
        return 0;
LABEL_25:
      v5 = *(_DWORD *)(a1 + 276);
      v3 = *(_QWORD *)(a1 + 24);
      if ( v5 )
        goto LABEL_4;
    }
    v10 = v9 - 1;
    v11 = *(_QWORD *)(v7 + 8);
    v153 = 37 * v8;
    v12 = (v9 - 1) & (37 * v8);
    v13 = (int *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
      v15 = *((_QWORD *)v13 + 1);
      goto LABEL_8;
    }
    v48 = *v13;
    v49 = (v9 - 1) & (37 * v8);
    for ( i = 1; ; ++i )
    {
      if ( v48 == -1 )
        goto LABEL_29;
      v49 = v10 & (i + v49);
      v48 = *(_DWORD *)(v11 + 16LL * v49);
      if ( v8 == v48 )
        break;
    }
    v103 = 1;
    v104 = 0;
    while ( v14 != -1 )
    {
      if ( v14 != -2 || v104 )
        v13 = v104;
      v12 = v10 & (v103 + v12);
      v148 = (int *)(v11 + 16LL * v12);
      v14 = *v148;
      if ( v8 == *v148 )
      {
        v15 = *((_QWORD *)v148 + 1);
        goto LABEL_8;
      }
      ++v103;
      v104 = v13;
      v13 = (int *)(v11 + 16LL * v12);
    }
    if ( !v104 )
      v104 = v13;
    v105 = *(_DWORD *)(v7 + 16);
    ++*(_QWORD *)v7;
    v106 = v105 + 1;
    if ( 4 * v106 >= 3 * v9 )
    {
      sub_1E37A30(v7, 2 * v9);
      v107 = *(_DWORD *)(v7 + 24);
      if ( !v107 )
        goto LABEL_233;
      v108 = v107 - 1;
      v109 = *(_QWORD *)(v7 + 8);
      v110 = v108 & v153;
      v106 = *(_DWORD *)(v7 + 16) + 1;
      v104 = (int *)(v109 + 16LL * (v108 & (unsigned int)v153));
      v111 = *v104;
      if ( v8 != *v104 )
      {
        v112 = 1;
        v113 = 0;
        while ( v111 != -1 )
        {
          if ( v111 == -2 && !v113 )
            v113 = v104;
          v110 = v108 & (v112 + v110);
          v104 = (int *)(v109 + 16LL * v110);
          v111 = *v104;
          if ( v8 == *v104 )
            goto LABEL_107;
          ++v112;
        }
LABEL_114:
        if ( v113 )
          v104 = v113;
      }
    }
    else if ( v9 - *(_DWORD *)(v7 + 20) - v106 <= v9 >> 3 )
    {
      sub_1E37A30(v7, v9);
      v114 = *(_DWORD *)(v7 + 24);
      if ( !v114 )
      {
LABEL_233:
        ++*(_DWORD *)(v7 + 16);
        BUG();
      }
      v115 = v114 - 1;
      v116 = *(_QWORD *)(v7 + 8);
      v117 = 1;
      v113 = 0;
      v118 = v115 & v153;
      v106 = *(_DWORD *)(v7 + 16) + 1;
      v104 = (int *)(v116 + 16LL * (v115 & (unsigned int)v153));
      v119 = *v104;
      if ( v8 != *v104 )
      {
        while ( v119 != -1 )
        {
          if ( v119 == -2 && !v113 )
            v113 = v104;
          v118 = v115 & (v117 + v118);
          v104 = (int *)(v116 + 16LL * v118);
          v119 = *v104;
          if ( v8 == *v104 )
            goto LABEL_107;
          ++v117;
        }
        goto LABEL_114;
      }
    }
LABEL_107:
    *(_DWORD *)(v7 + 16) = v106;
    if ( *v104 != -1 )
      --*(_DWORD *)(v7 + 20);
    *v104 = v8;
    v15 = 0;
    *((_QWORD *)v104 + 1) = 0;
    v5 = *(_DWORD *)(a1 + 276);
    v3 = *(_QWORD *)(a1 + 24);
LABEL_8:
    v16 = *(_DWORD *)(v15 + 36);
    if ( v16 == -1 )
    {
      v17 = 0;
LABEL_37:
      *(_DWORD *)(a1 + 272) += v17;
      *(_DWORD *)(a1 + 276) = v5;
      *(_QWORD *)(a1 + 264) = v15;
      continue;
    }
    break;
  }
  v17 = **(_DWORD **)(v15 + 40) - v16 + 1;
  if ( v17 <= v5 )
  {
    v5 -= v17;
    goto LABEL_37;
  }
  v18 = v16 + v5;
  if ( *(_DWORD *)(v3 + 4 * v18) != *(_DWORD *)(v3 + v157) )
  {
    v152 = *(_DWORD *)(v3 + v157);
    v19 = v18 - 1;
    v149 = *(_QWORD *)(a1 + 264);
    v20 = (_DWORD *)sub_145CBF0((__int64 *)(a1 + 152), 4, 8);
    *v20 = v19;
    v150 = v20;
    v21 = sub_145CBF0((__int64 *)(a1 + 40), 80, 8);
    v22 = v149;
    v23 = v21;
    v24 = *(_QWORD *)(a1 + 144);
    v25 = (__int64 *)(a1 + 40);
    *(_QWORD *)v23 = 0;
    v26 = v152;
    *(_QWORD *)(v23 + 8) = 0;
    *(_QWORD *)(v23 + 16) = 0;
    *(_DWORD *)(v23 + 24) = 0;
    *(_BYTE *)(v23 + 32) = 1;
    *(_DWORD *)(v23 + 36) = v16;
    *(_QWORD *)(v23 + 40) = v150;
    *(_DWORD *)(v23 + 48) = -1;
    *(_QWORD *)(v23 + 56) = v24;
    *(_QWORD *)(v23 + 64) = v149;
    *(_QWORD *)(v23 + 72) = 0;
    if ( v149 )
    {
      v27 = *(_DWORD *)(v149 + 24);
      if ( v27 )
      {
        v28 = *(_QWORD *)(v149 + 8);
        v29 = (v27 - 1) & v153;
        v30 = (int *)(v28 + 16LL * v29);
        v31 = *v30;
        if ( v8 == *v30 )
          goto LABEL_14;
        v151 = 1;
        v133 = 0;
        while ( v31 != -1 )
        {
          if ( v31 == -2 && !v133 )
            v133 = v30;
          v29 = (v27 - 1) & (v151 + v29);
          v30 = (int *)(v28 + 16LL * v29);
          v31 = *v30;
          if ( v8 == *v30 )
            goto LABEL_14;
          ++v151;
        }
        v134 = *(_DWORD *)(v149 + 16);
        if ( v133 )
          v30 = v133;
        ++*(_QWORD *)v149;
        v135 = v134 + 1;
        if ( 4 * v135 < 3 * v27 )
        {
          if ( v27 - *(_DWORD *)(v149 + 20) - v135 <= v27 >> 3 )
          {
            sub_1E37A30(v149, v27);
            v22 = v149;
            v142 = *(_DWORD *)(v149 + 24);
            if ( !v142 )
            {
LABEL_230:
              ++*(_DWORD *)(v22 + 16);
              BUG();
            }
            v143 = v142 - 1;
            v144 = *(_QWORD *)(v149 + 8);
            v26 = v152;
            v145 = v143 & v153;
            v25 = (__int64 *)(a1 + 40);
            v135 = *(_DWORD *)(v149 + 16) + 1;
            v30 = (int *)(v144 + 16LL * (v143 & (unsigned int)v153));
            v146 = *v30;
            if ( v8 != *v30 )
            {
              v156 = 1;
              v147 = 0;
              while ( v146 != -1 )
              {
                if ( v146 == -2 && !v147 )
                  v147 = v30;
                v145 = v143 & (v156 + v145);
                v30 = (int *)(v144 + 16LL * v145);
                v146 = *v30;
                if ( v8 == *v30 )
                  goto LABEL_145;
                ++v156;
              }
              if ( v147 )
                v30 = v147;
            }
          }
          goto LABEL_145;
        }
      }
      else
      {
        ++*(_QWORD *)v149;
      }
      sub_1E37A30(v149, 2 * v27);
      v22 = v149;
      v136 = *(_DWORD *)(v149 + 24);
      if ( !v136 )
        goto LABEL_230;
      v137 = v136 - 1;
      v138 = *(_QWORD *)(v149 + 8);
      v26 = v152;
      v139 = v137 & v153;
      v25 = (__int64 *)(a1 + 40);
      v135 = *(_DWORD *)(v149 + 16) + 1;
      v30 = (int *)(v138 + 16LL * (v137 & (unsigned int)v153));
      v140 = *v30;
      if ( v8 != *v30 )
      {
        v155 = 1;
        v141 = 0;
        while ( v140 != -1 )
        {
          if ( !v141 && v140 == -2 )
            v141 = v30;
          v139 = v137 & (v155 + v139);
          v30 = (int *)(v138 + 16LL * v139);
          v140 = *v30;
          if ( v8 == *v30 )
            goto LABEL_145;
          ++v155;
        }
        if ( v141 )
          v30 = v141;
      }
LABEL_145:
      *(_DWORD *)(v22 + 16) = v135;
      if ( *v30 != -1 )
        --*(_DWORD *)(v22 + 20);
      *v30 = v8;
      *((_QWORD *)v30 + 1) = 0;
LABEL_14:
      *((_QWORD *)v30 + 1) = v23;
    }
    v154 = v26;
    v32 = sub_145CBF0(v25, 80, 8);
    v33 = *(_DWORD *)(v23 + 24);
    v34 = v154;
    *(_QWORD *)v32 = 0;
    v35 = v32;
    *(_QWORD *)(v32 + 8) = 0;
    *(_QWORD *)(v32 + 16) = 0;
    *(_DWORD *)(v32 + 24) = 0;
    *(_BYTE *)(v32 + 32) = 1;
    *(_DWORD *)(v32 + 48) = -1;
    *(_DWORD *)(v32 + 36) = a2;
    *(_QWORD *)(v32 + 40) = a1 + 256;
    *(_QWORD *)(v32 + 56) = 0;
    *(_QWORD *)(v32 + 64) = v23;
    *(_QWORD *)(v32 + 72) = 0;
    if ( v33 )
    {
      v36 = *(_QWORD *)(v23 + 8);
      v37 = (v33 - 1) & (37 * v154);
      v38 = (int *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( v154 == *v38 )
        goto LABEL_17;
      v84 = 1;
      v85 = 0;
      while ( v39 != -1 )
      {
        if ( v39 == -2 && !v85 )
          v85 = v38;
        v37 = (v33 - 1) & (v84 + v37);
        v38 = (int *)(v36 + 16LL * v37);
        v39 = *v38;
        if ( v154 == *v38 )
          goto LABEL_17;
        ++v84;
      }
      v86 = *(_DWORD *)(v23 + 16);
      if ( v85 )
        v38 = v85;
      ++*(_QWORD *)v23;
      v87 = v86 + 1;
      if ( 4 * v87 < 3 * v33 )
      {
        if ( v33 - *(_DWORD *)(v23 + 20) - v87 <= v33 >> 3 )
        {
          sub_1E37A30(v23, v33);
          v126 = *(_DWORD *)(v23 + 24);
          if ( !v126 )
          {
LABEL_231:
            ++*(_DWORD *)(v23 + 16);
            BUG();
          }
          v127 = v126 - 1;
          v128 = *(_QWORD *)(v23 + 8);
          v129 = 1;
          v130 = v127 & (37 * v154);
          v34 = v154;
          v131 = 0;
          v87 = *(_DWORD *)(v23 + 16) + 1;
          v38 = (int *)(v128 + 16LL * v130);
          v132 = *v38;
          if ( v154 != *v38 )
          {
            while ( v132 != -1 )
            {
              if ( v132 == -2 && !v131 )
                v131 = v38;
              v130 = v127 & (v129 + v130);
              v38 = (int *)(v128 + 16LL * v130);
              v132 = *v38;
              if ( v154 == *v38 )
                goto LABEL_77;
              ++v129;
            }
            if ( v131 )
              v38 = v131;
          }
        }
        goto LABEL_77;
      }
    }
    else
    {
      ++*(_QWORD *)v23;
    }
    sub_1E37A30(v23, 2 * v33);
    v96 = *(_DWORD *)(v23 + 24);
    if ( !v96 )
      goto LABEL_231;
    v34 = v154;
    v97 = v96 - 1;
    v98 = *(_QWORD *)(v23 + 8);
    v87 = *(_DWORD *)(v23 + 16) + 1;
    v99 = v97 & (37 * v154);
    v38 = (int *)(v98 + 16LL * v99);
    v100 = *v38;
    if ( v154 != *v38 )
    {
      v101 = 1;
      v102 = 0;
      while ( v100 != -1 )
      {
        if ( v100 == -2 && !v102 )
          v102 = v38;
        v99 = v97 & (v101 + v99);
        v38 = (int *)(v98 + 16LL * v99);
        v100 = *v38;
        if ( v154 == *v38 )
          goto LABEL_77;
        ++v101;
      }
      if ( v102 )
        v38 = v102;
    }
LABEL_77:
    *(_DWORD *)(v23 + 16) = v87;
    if ( *v38 != -1 )
      --*(_DWORD *)(v23 + 20);
    *v38 = v34;
    *((_QWORD *)v38 + 1) = 0;
LABEL_17:
    *((_QWORD *)v38 + 1) = v35;
    v40 = (unsigned int)(*(_DWORD *)(v15 + 36) + *(_DWORD *)(a1 + 276));
    *(_QWORD *)(v15 + 64) = v23;
    *(_DWORD *)(v15 + 36) = v40;
    v41 = *(_DWORD *)(v23 + 24);
    v42 = (int *)(*(_QWORD *)(a1 + 24) + 4 * v40);
    if ( v41 )
    {
      v43 = *(_QWORD *)(v23 + 8);
      v44 = (v41 - 1) & (37 * *v42);
      v45 = (int *)(v43 + 16LL * v44);
      v46 = *v45;
      if ( *v45 == *v42 )
      {
LABEL_19:
        *((_QWORD *)v45 + 1) = v15;
        if ( v160 )
          *(_QWORD *)(v160 + 56) = v23;
        v160 = v23;
        v47 = *(_QWORD *)(a1 + 264);
LABEL_22:
        --v159;
        if ( *(_DWORD *)(v47 + 36) != -1 )
          goto LABEL_23;
        goto LABEL_33;
      }
      v79 = 1;
      v80 = 0;
      while ( v46 != -1 )
      {
        if ( !v80 && v46 == -2 )
          v80 = v45;
        v44 = (v41 - 1) & (v79 + v44);
        v45 = (int *)(v43 + 16LL * v44);
        v46 = *v45;
        if ( *v42 == *v45 )
          goto LABEL_19;
        ++v79;
      }
      v81 = *(_DWORD *)(v23 + 16);
      if ( v80 )
        v45 = v80;
      ++*(_QWORD *)v23;
      v82 = v81 + 1;
      if ( 4 * v82 < 3 * v41 )
      {
        if ( v41 - *(_DWORD *)(v23 + 20) - v82 <= v41 >> 3 )
        {
          sub_1E37A30(v23, v41);
          v120 = *(_DWORD *)(v23 + 24);
          if ( !v120 )
          {
LABEL_234:
            ++*(_DWORD *)(v23 + 16);
            BUG();
          }
          v121 = v120 - 1;
          v122 = *(_QWORD *)(v23 + 8);
          v95 = 0;
          v123 = 1;
          v82 = *(_DWORD *)(v23 + 16) + 1;
          v124 = v121 & (37 * *v42);
          v45 = (int *)(v122 + 16LL * v124);
          v125 = *v45;
          if ( *v42 != *v45 )
          {
            while ( v125 != -1 )
            {
              if ( v125 == -2 && !v95 )
                v95 = v45;
              v124 = v121 & (v123 + v124);
              v45 = (int *)(v122 + 16LL * v124);
              v125 = *v45;
              if ( *v42 == *v45 )
                goto LABEL_68;
              ++v123;
            }
            goto LABEL_89;
          }
        }
        goto LABEL_68;
      }
    }
    else
    {
      ++*(_QWORD *)v23;
    }
    sub_1E37A30(v23, 2 * v41);
    v89 = *(_DWORD *)(v23 + 24);
    if ( !v89 )
      goto LABEL_234;
    v90 = v89 - 1;
    v91 = *(_QWORD *)(v23 + 8);
    v82 = *(_DWORD *)(v23 + 16) + 1;
    v92 = v90 & (37 * *v42);
    v45 = (int *)(v91 + 16LL * v92);
    v93 = *v45;
    if ( *v45 != *v42 )
    {
      v94 = 1;
      v95 = 0;
      while ( v93 != -1 )
      {
        if ( !v95 && v93 == -2 )
          v95 = v45;
        v92 = v90 & (v94 + v92);
        v45 = (int *)(v91 + 16LL * v92);
        v93 = *v45;
        if ( *v42 == *v45 )
          goto LABEL_68;
        ++v94;
      }
LABEL_89:
      if ( v95 )
        v45 = v95;
    }
LABEL_68:
    *(_DWORD *)(v23 + 16) = v82;
    if ( *v45 != -1 )
      --*(_DWORD *)(v23 + 20);
    v83 = *v42;
    *((_QWORD *)v45 + 1) = 0;
    *v45 = v83;
    goto LABEL_19;
  }
  if ( v160 )
  {
    v88 = *(_QWORD *)(a1 + 264);
    if ( *(_DWORD *)(v88 + 36) != -1 )
      *(_QWORD *)(v160 + 56) = v88;
  }
  *(_DWORD *)(a1 + 276) = v5 + 1;
  return v159;
}
