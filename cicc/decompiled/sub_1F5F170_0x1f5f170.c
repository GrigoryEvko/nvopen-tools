// Function: sub_1F5F170
// Address: 0x1f5f170
//
void __fastcall sub_1F5F170(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rax
  char v11; // dl
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rcx
  __int64 *v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // esi
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rcx
  unsigned int v23; // edi
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rcx
  __int64 *v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // esi
  unsigned int v36; // edi
  __int64 *v37; // rax
  __int64 v38; // rcx
  unsigned __int64 v39; // rcx
  unsigned int v40; // esi
  unsigned int v41; // edi
  __int64 v42; // rcx
  unsigned __int64 v43; // rcx
  unsigned int v44; // edi
  unsigned int v45; // esi
  unsigned int i; // r9d
  __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // r9
  int v52; // r11d
  unsigned int v53; // edi
  __int64 *v54; // r10
  __int64 v55; // r8
  unsigned __int64 v56; // r8
  unsigned int v57; // edi
  int v58; // ecx
  int v59; // ecx
  __int64 v60; // r11
  int v61; // r10d
  unsigned int v62; // edi
  __int64 *v63; // r9
  __int64 v64; // r8
  unsigned __int64 v65; // r8
  unsigned int v66; // edi
  int v67; // esi
  int v68; // esi
  __int64 v69; // r11
  int v70; // r10d
  unsigned int v71; // edi
  __int64 *v72; // r9
  __int64 v73; // r8
  unsigned __int64 v74; // r8
  unsigned int v75; // edi
  int v76; // ecx
  __int64 v77; // r9
  int v78; // r11d
  unsigned int v79; // edi
  __int64 v80; // r8
  unsigned __int64 v81; // r8
  unsigned int v82; // edi
  int v83; // ecx
  __int64 v84; // rdx
  int v85; // edi
  int v86; // ecx
  int v87; // edx
  int v88; // edx
  __int64 v89; // r9
  int v90; // r10d
  __int64 *v91; // rdi
  unsigned int v92; // r8d
  __int64 v93; // rsi
  unsigned __int64 v94; // rsi
  unsigned int v95; // r8d
  unsigned int v96; // edi
  unsigned int v97; // r9d
  int v98; // ecx
  int v99; // ecx
  int v100; // ecx
  __int64 v101; // r10
  int v102; // r11d
  unsigned int v103; // r8d
  __int64 v104; // rdi
  unsigned __int64 v105; // rdi
  unsigned int v106; // r8d
  int v107; // ecx
  int v108; // edx
  int v109; // edx
  int v110; // r10d
  unsigned __int64 v111; // r11
  unsigned int v112; // r9d
  __int64 v113; // rsi
  unsigned __int64 v114; // rsi
  unsigned int v115; // r9d
  int v116; // ecx
  int v117; // edx
  int v118; // edx
  int v119; // r10d
  __int64 *v120; // rdi
  unsigned __int64 v121; // r11
  unsigned int v122; // r9d
  __int64 v123; // rsi
  unsigned __int64 v124; // rsi
  unsigned int v125; // r9d
  int v126; // [rsp+4h] [rbp-4Ch]
  int v127; // [rsp+4h] [rbp-4Ch]
  int v128; // [rsp+4h] [rbp-4Ch]
  int v129; // [rsp+4h] [rbp-4Ch]
  __int64 *v130; // [rsp+8h] [rbp-48h]
  __int64 *v131; // [rsp+8h] [rbp-48h]
  __int64 *v132; // [rsp+8h] [rbp-48h]
  __int64 *v133; // [rsp+8h] [rbp-48h]
  unsigned __int64 v134; // [rsp+8h] [rbp-48h]
  unsigned __int64 v135; // [rsp+8h] [rbp-48h]
  unsigned __int64 v136; // [rsp+10h] [rbp-40h]
  unsigned __int64 v137; // [rsp+10h] [rbp-40h]
  unsigned __int64 v138; // [rsp+10h] [rbp-40h]
  unsigned __int64 v139; // [rsp+10h] [rbp-40h]
  unsigned __int64 v140; // [rsp+10h] [rbp-40h]
  unsigned __int64 v141; // [rsp+10h] [rbp-40h]
  __int64 v143; // [rsp+18h] [rbp-38h]
  unsigned __int64 v144; // [rsp+18h] [rbp-38h]
  unsigned __int64 v145; // [rsp+18h] [rbp-38h]
  __int64 *v146; // [rsp+18h] [rbp-38h]
  __int64 v147; // [rsp+18h] [rbp-38h]

  v2 = a1 + 72;
  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 == a1 + 72 )
    return;
  do
  {
    v5 = v3 - 24;
    if ( !v3 )
      v5 = 0;
    v6 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v5) + 16) - 34;
    if ( (unsigned int)v6 > 0x36 )
      goto LABEL_6;
    v7 = 0x40018000000001LL;
    if ( !_bittest64(&v7, v6) )
      goto LABEL_6;
    v26 = sub_157ED20(v5);
    if ( *(_BYTE *)(v26 + 16) != 74 )
      goto LABEL_6;
    v27 = *(_QWORD *)(v26 - 24);
    if ( (*(_BYTE *)(v27 + 18) & 1) == 0 )
      goto LABEL_6;
    v28 = (*(_BYTE *)(v27 + 23) & 0x40) != 0 ? *(_QWORD *)(v27 - 8) : v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
    v29 = *(_QWORD *)(v28 + 24);
    if ( !v29 )
      goto LABEL_6;
    v30 = sub_157ED20(*(_QWORD *)(v28 + 24));
    if ( *(_BYTE *)(v30 + 16) != 34 )
    {
      v45 = *(_DWORD *)(a2 + 24);
      if ( !v45 )
      {
        ++*(_QWORD *)a2;
        goto LABEL_85;
      }
      v129 = 1;
      v133 = 0;
      v139 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      for ( i = (v45 - 1) & (37 * v5); ; i = (v45 - 1) & v97 )
      {
        v37 = (__int64 *)(*(_QWORD *)(a2 + 8) + 16LL * i);
        v47 = *v37;
        if ( !((v5 >> 2) & 1) == !((*v37 >> 2) & 1) )
        {
          v48 = v47 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v5 >> 2) & 1) != 0 )
          {
            if ( v48 == v139 )
              goto LABEL_123;
            goto LABEL_156;
          }
          if ( v48 == v139 )
            goto LABEL_123;
        }
        else
        {
          if ( ((*v37 >> 2) & 1) != 0 )
            goto LABEL_156;
          v48 = v47 & 0xFFFFFFFFFFFFFFF8LL;
        }
        if ( v48 == -8 )
        {
          if ( v133 )
            v37 = v133;
          v98 = *(_DWORD *)(a2 + 16);
          ++*(_QWORD *)a2;
          v85 = v98 + 1;
          if ( 4 * (v98 + 1) < 3 * v45 )
          {
            if ( v45 - *(_DWORD *)(a2 + 20) - v85 > v45 >> 3 )
              goto LABEL_127;
            sub_1F5EF30(a2, v45);
            v99 = *(_DWORD *)(a2 + 24);
            v37 = 0;
            if ( !v99 )
              goto LABEL_126;
            v100 = v99 - 1;
            v101 = *(_QWORD *)(a2 + 8);
            v102 = 1;
            v63 = 0;
            v135 = v5 & 0xFFFFFFFFFFFFFFF8LL;
            v103 = v100 & (37 * v5);
            while ( 2 )
            {
              v37 = (__int64 *)(v101 + 16LL * v103);
              v104 = *v37;
              if ( !((v5 >> 2) & 1) == !((*v37 >> 2) & 1) )
              {
                v105 = v104 & 0xFFFFFFFFFFFFFFF8LL;
                if ( ((v5 >> 2) & 1) != 0 )
                {
                  if ( v105 == v135 )
                    goto LABEL_126;
                }
                else
                {
                  if ( v105 == v135 )
                    goto LABEL_126;
LABEL_176:
                  if ( v105 == -8 )
                    goto LABEL_124;
                  if ( v105 == -16 && !v63 )
                    v63 = (__int64 *)(v101 + 16LL * v103);
                }
              }
              else if ( ((*v37 >> 2) & 1) == 0 )
              {
                v105 = v104 & 0xFFFFFFFFFFFFFFF8LL;
                goto LABEL_176;
              }
              v106 = v102 + v103;
              ++v102;
              v103 = v100 & v106;
              continue;
            }
          }
LABEL_85:
          sub_1F5EF30(a2, 2 * v45);
          v58 = *(_DWORD *)(a2 + 24);
          v37 = 0;
          if ( !v58 )
            goto LABEL_126;
          v59 = v58 - 1;
          v60 = *(_QWORD *)(a2 + 8);
          v61 = 1;
          v62 = v59 & (37 * v5);
          v63 = 0;
          v140 = v5 & 0xFFFFFFFFFFFFFFF8LL;
          while ( 2 )
          {
            v37 = (__int64 *)(v60 + 16LL * v62);
            v64 = *v37;
            if ( !((v5 >> 2) & 1) == !((*v37 >> 2) & 1) )
            {
              v65 = v64 & 0xFFFFFFFFFFFFFFF8LL;
              if ( ((v5 >> 2) & 1) != 0 )
              {
                if ( v65 == v140 )
                  goto LABEL_126;
              }
              else
              {
                if ( v65 == v140 )
                  goto LABEL_126;
LABEL_90:
                if ( v65 == -8 )
                {
LABEL_124:
                  if ( v63 )
                    v37 = v63;
LABEL_126:
                  v85 = *(_DWORD *)(a2 + 16) + 1;
LABEL_127:
                  *(_DWORD *)(a2 + 16) = v85;
                  v84 = *v37;
                  if ( (*v37 & 4) != 0 )
                    goto LABEL_121;
                  goto LABEL_128;
                }
                if ( v65 == -16 && !v63 )
                  v63 = (__int64 *)(v60 + 16LL * v62);
              }
            }
            else if ( ((*v37 >> 2) & 1) == 0 )
            {
              v65 = v64 & 0xFFFFFFFFFFFFFFF8LL;
              goto LABEL_90;
            }
            v66 = v61 + v62;
            ++v61;
            v62 = v59 & v66;
            continue;
          }
        }
        if ( !v133 )
        {
          if ( v48 != -16 )
            v37 = 0;
          v133 = v37;
        }
LABEL_156:
        v97 = v129 + i;
        ++v129;
      }
    }
    if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
      v31 = *(_QWORD *)(v30 - 8);
    else
      v31 = v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF);
    v32 = (__int64 *)(v31 + 24);
    v33 = (__int64 *)(v31 + 48);
    if ( (*(_BYTE *)(v30 + 18) & 1) == 0 )
      v33 = v32;
    v34 = sub_1523720(*v33);
    v35 = *(_DWORD *)(a2 + 24);
    v29 = v34;
    if ( !v35 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_101;
    }
    v127 = 1;
    v131 = 0;
    v36 = (v35 - 1) & (37 * v5);
    v137 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v37 = (__int64 *)(*(_QWORD *)(a2 + 8) + 16LL * v36);
      v38 = *v37;
      if ( !((v5 >> 2) & 1) != !((*v37 >> 2) & 1) )
        break;
      v39 = v38 & 0xFFFFFFFFFFFFFFF8LL;
      if ( ((v5 >> 2) & 1) == 0 )
      {
        if ( v39 == v137 )
          goto LABEL_123;
        goto LABEL_132;
      }
      if ( v39 == v137 )
        goto LABEL_123;
LABEL_154:
      v96 = v127 + v36;
      ++v127;
      v36 = (v35 - 1) & v96;
    }
    v39 = v38 & 0xFFFFFFFFFFFFFFF8LL;
    if ( ((*v37 >> 2) & 1) != 0 )
      goto LABEL_154;
LABEL_132:
    if ( v39 != -8 )
    {
      if ( v39 == -16 )
      {
        if ( v131 )
          v37 = v131;
        v131 = v37;
      }
      goto LABEL_154;
    }
    if ( v131 )
      v37 = v131;
    v86 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v83 = v86 + 1;
    if ( 4 * v83 < 3 * v35 )
    {
      if ( v35 - *(_DWORD *)(a2 + 20) - v83 > v35 >> 3 )
        goto LABEL_120;
      sub_1F5EF30(a2, v35);
      v87 = *(_DWORD *)(a2 + 24);
      v37 = 0;
      if ( !v87 )
        goto LABEL_119;
      v88 = v87 - 1;
      v89 = *(_QWORD *)(a2 + 8);
      v90 = 1;
      v91 = 0;
      v134 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v92 = v88 & (37 * v5);
      while ( 2 )
      {
        v37 = (__int64 *)(v89 + 16LL * v92);
        v93 = *v37;
        if ( !((v5 >> 2) & 1) == !((*v37 >> 2) & 1) )
        {
          v94 = v93 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v5 >> 2) & 1) != 0 )
          {
            if ( v94 == v134 )
              goto LABEL_119;
            goto LABEL_145;
          }
          if ( v94 == v134 )
            goto LABEL_119;
        }
        else
        {
          v94 = v93 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((*v37 >> 2) & 1) != 0 )
          {
LABEL_145:
            v95 = v90 + v92;
            ++v90;
            v92 = v88 & v95;
            continue;
          }
        }
        break;
      }
      if ( v94 == -8 )
      {
        if ( v91 )
          v37 = v91;
        goto LABEL_119;
      }
      if ( v94 == -16 && !v91 )
        v91 = (__int64 *)(v89 + 16LL * v92);
      goto LABEL_145;
    }
LABEL_101:
    sub_1F5EF30(a2, 2 * v35);
    v67 = *(_DWORD *)(a2 + 24);
    v37 = 0;
    if ( !v67 )
      goto LABEL_119;
    v68 = v67 - 1;
    v69 = *(_QWORD *)(a2 + 8);
    v70 = 1;
    v71 = v68 & (37 * v5);
    v72 = 0;
    v141 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 2 )
    {
      v37 = (__int64 *)(v69 + 16LL * v71);
      v73 = *v37;
      if ( !((v5 >> 2) & 1) == !((*v37 >> 2) & 1) )
      {
        v74 = v73 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((v5 >> 2) & 1) != 0 )
        {
          if ( v74 == v141 )
            goto LABEL_119;
          goto LABEL_105;
        }
        if ( v74 == v141 )
          goto LABEL_119;
      }
      else
      {
        v74 = v73 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((*v37 >> 2) & 1) != 0 )
        {
LABEL_105:
          v75 = v70 + v71;
          ++v70;
          v71 = v68 & v75;
          continue;
        }
      }
      break;
    }
    if ( v74 != -8 )
    {
      if ( v74 == -16 && !v72 )
        v72 = (__int64 *)(v69 + 16LL * v71);
      goto LABEL_105;
    }
    if ( v72 )
      v37 = v72;
LABEL_119:
    v83 = *(_DWORD *)(a2 + 16) + 1;
LABEL_120:
    *(_DWORD *)(a2 + 16) = v83;
    v84 = *v37;
    if ( (*v37 & 4) != 0 )
    {
LABEL_121:
      --*(_DWORD *)(a2 + 20);
      goto LABEL_122;
    }
LABEL_128:
    if ( (v84 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
      goto LABEL_121;
LABEL_122:
    *v37 = v5;
    v37[1] = 0;
LABEL_123:
    v37[1] = v29;
LABEL_6:
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v2 != v3 );
  v8 = *(_QWORD *)(a1 + 80);
  if ( v3 == v8 )
    return;
  do
  {
    v9 = v8 - 24;
    if ( !v8 )
      v9 = 0;
    v10 = sub_157EBA0(v9);
    v11 = *(_BYTE *)(v10 + 16);
    if ( v11 == 29 )
    {
      v12 = *(_QWORD *)(v10 - 24);
    }
    else
    {
      if ( v11 != 32 || (*(_BYTE *)(v10 + 18) & 1) == 0 )
        goto LABEL_35;
      v12 = *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
    }
    if ( !v12 )
      goto LABEL_35;
    v13 = sub_157ED20(v12);
    v143 = a2 + 32;
    if ( *(_BYTE *)(v13 + 16) != 34 )
    {
      v40 = *(_DWORD *)(a2 + 56);
      if ( !v40 )
      {
        ++*(_QWORD *)(a2 + 32);
        goto LABEL_73;
      }
      v128 = 1;
      v132 = 0;
      v41 = (v40 - 1) & (37 * v9);
      v138 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v20 = (__int64 *)(*(_QWORD *)(a2 + 40) + 16LL * v41);
        v42 = *v20;
        if ( !((v9 >> 2) & 1) == !((*v20 >> 2) & 1) )
        {
          v43 = v42 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v9 >> 2) & 1) != 0 )
          {
            if ( v138 == v43 )
              goto LABEL_34;
            goto LABEL_64;
          }
          if ( v138 == v43 )
            goto LABEL_34;
        }
        else
        {
          v43 = v42 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((*v20 >> 2) & 1) != 0 )
            goto LABEL_64;
        }
        if ( v43 == -8 )
        {
          if ( v132 )
            v20 = v132;
          v116 = *(_DWORD *)(a2 + 48);
          ++*(_QWORD *)(a2 + 32);
          v25 = v116 + 1;
          if ( 4 * v25 < 3 * v40 )
          {
            if ( v40 - *(_DWORD *)(a2 + 52) - v25 > v40 >> 3 )
              goto LABEL_31;
            sub_1F5EF30(v143, v40);
            v117 = *(_DWORD *)(a2 + 56);
            v20 = 0;
            if ( !v117 )
              goto LABEL_30;
            v118 = v117 - 1;
            v119 = 1;
            v120 = 0;
            v121 = v9 & 0xFFFFFFFFFFFFFFF8LL;
            v147 = *(_QWORD *)(a2 + 40);
            v122 = v118 & (37 * v9);
            while ( 2 )
            {
              v20 = (__int64 *)(v147 + 16LL * v122);
              v123 = *v20;
              if ( !((v9 >> 2) & 1) == !((*v20 >> 2) & 1) )
              {
                v124 = v123 & 0xFFFFFFFFFFFFFFF8LL;
                if ( ((v9 >> 2) & 1) != 0 )
                {
                  if ( v124 == v121 )
                    goto LABEL_30;
                  goto LABEL_204;
                }
                if ( v124 == v121 )
                  goto LABEL_30;
              }
              else
              {
                v124 = v123 & 0xFFFFFFFFFFFFFFF8LL;
                if ( ((*v20 >> 2) & 1) != 0 )
                {
LABEL_204:
                  v125 = v119 + v122;
                  ++v119;
                  v122 = v118 & v125;
                  continue;
                }
              }
              break;
            }
            if ( v124 == -8 )
            {
              if ( v120 )
                v20 = v120;
              goto LABEL_30;
            }
            if ( v124 == -16 && !v120 )
              v120 = (__int64 *)(v147 + 16LL * v122);
            goto LABEL_204;
          }
LABEL_73:
          sub_1F5EF30(v143, 2 * v40);
          v49 = *(_DWORD *)(a2 + 56);
          v20 = 0;
          if ( !v49 )
            goto LABEL_30;
          v50 = v49 - 1;
          v51 = *(_QWORD *)(a2 + 40);
          v52 = 1;
          v53 = v50 & (37 * v9);
          v54 = 0;
          v144 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          while ( 2 )
          {
            v20 = (__int64 *)(v51 + 16LL * v53);
            v55 = *v20;
            if ( !((v9 >> 2) & 1) == !((*v20 >> 2) & 1) )
            {
              v56 = v55 & 0xFFFFFFFFFFFFFFF8LL;
              if ( ((v9 >> 2) & 1) != 0 )
              {
                if ( v56 == v144 )
                  goto LABEL_30;
                goto LABEL_77;
              }
              if ( v56 == v144 )
                goto LABEL_30;
            }
            else
            {
              v56 = v55 & 0xFFFFFFFFFFFFFFF8LL;
              if ( ((*v20 >> 2) & 1) != 0 )
              {
LABEL_77:
                v57 = v52 + v53;
                ++v52;
                v53 = v50 & v57;
                continue;
              }
            }
            break;
          }
          if ( v56 == -8 )
          {
LABEL_219:
            if ( v54 )
              v20 = v54;
LABEL_30:
            v25 = *(_DWORD *)(a2 + 48) + 1;
            goto LABEL_31;
          }
          if ( !v54 && v56 == -16 )
            v54 = (__int64 *)(v51 + 16LL * v53);
          goto LABEL_77;
        }
        if ( v43 == -16 )
        {
          if ( v132 )
            v20 = v132;
          v132 = v20;
        }
LABEL_64:
        v44 = v128 + v41;
        ++v128;
        v41 = (v40 - 1) & v44;
      }
    }
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      v14 = *(_QWORD *)(v13 - 8);
    else
      v14 = v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
    v15 = (__int64 *)(v14 + 24);
    v16 = (__int64 *)(v14 + 48);
    if ( (*(_BYTE *)(v13 + 18) & 1) == 0 )
      v16 = v15;
    v17 = sub_1523720(*v16);
    v18 = *(_DWORD *)(a2 + 56);
    v12 = v17;
    if ( !v18 )
    {
      ++*(_QWORD *)(a2 + 32);
LABEL_29:
      sub_1F5EF30(v143, 2 * v18);
      v24 = *(_DWORD *)(a2 + 56);
      v20 = 0;
      if ( !v24 )
        goto LABEL_30;
      v76 = v24 - 1;
      v77 = *(_QWORD *)(a2 + 40);
      v78 = 1;
      v79 = v76 & (37 * v9);
      v54 = 0;
      v145 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v20 = (__int64 *)(v77 + 16LL * v79);
        v80 = *v20;
        if ( !((v9 >> 2) & 1) == !((*v20 >> 2) & 1) )
        {
          v81 = v80 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v9 >> 2) & 1) != 0 )
          {
            if ( v145 == v81 )
              goto LABEL_30;
            goto LABEL_116;
          }
          if ( v145 == v81 )
            goto LABEL_30;
        }
        else
        {
          if ( ((*v20 >> 2) & 1) != 0 )
            goto LABEL_116;
          v81 = v80 & 0xFFFFFFFFFFFFFFF8LL;
        }
        if ( v81 == -8 )
          goto LABEL_219;
        if ( v81 == -16 && !v54 )
          v54 = (__int64 *)(v77 + 16LL * v79);
LABEL_116:
        v82 = v78 + v79;
        ++v78;
        v79 = v76 & v82;
      }
    }
    v126 = 1;
    v130 = 0;
    v19 = (v18 - 1) & (37 * v9);
    v136 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v20 = (__int64 *)(*(_QWORD *)(a2 + 40) + 16LL * v19);
      v21 = *v20;
      if ( !((v9 >> 2) & 1) != !((*v20 >> 2) & 1) )
        break;
      v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
      if ( ((v9 >> 2) & 1) == 0 )
      {
        if ( v136 == v22 )
          goto LABEL_34;
        goto LABEL_22;
      }
      if ( v136 == v22 )
        goto LABEL_34;
LABEL_27:
      v23 = v126 + v19;
      ++v126;
      v19 = (v18 - 1) & v23;
    }
    v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    if ( ((*v20 >> 2) & 1) != 0 )
      goto LABEL_27;
LABEL_22:
    if ( v22 != -8 )
    {
      if ( v22 == -16 )
      {
        if ( v130 )
          v20 = v130;
        v130 = v20;
      }
      goto LABEL_27;
    }
    if ( v130 )
      v20 = v130;
    v107 = *(_DWORD *)(a2 + 48);
    ++*(_QWORD *)(a2 + 32);
    v25 = v107 + 1;
    if ( 4 * v25 >= 3 * v18 )
      goto LABEL_29;
    if ( v18 - *(_DWORD *)(a2 + 52) - v25 <= v18 >> 3 )
    {
      sub_1F5EF30(v143, v18);
      v108 = *(_DWORD *)(a2 + 56);
      v20 = 0;
      if ( !v108 )
        goto LABEL_30;
      v109 = v108 - 1;
      v146 = 0;
      v110 = 1;
      v111 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      v112 = v109 & (37 * v9);
      while ( 2 )
      {
        v20 = (__int64 *)(*(_QWORD *)(a2 + 40) + 16LL * v112);
        v113 = *v20;
        if ( !((v9 >> 2) & 1) == !((*v20 >> 2) & 1) )
        {
          v114 = v113 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v9 >> 2) & 1) != 0 )
          {
            if ( v111 == v114 )
              goto LABEL_30;
          }
          else
          {
            if ( v111 == v114 )
              goto LABEL_30;
LABEL_190:
            if ( v114 == -8 )
            {
              if ( v146 )
                v20 = v146;
              goto LABEL_30;
            }
            if ( v114 == -16 )
            {
              if ( v146 )
                v20 = v146;
              v146 = v20;
            }
          }
        }
        else if ( ((*v20 >> 2) & 1) == 0 )
        {
          v114 = v113 & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_190;
        }
        v115 = v110 + v112;
        ++v110;
        v112 = v109 & v115;
        continue;
      }
    }
LABEL_31:
    *(_DWORD *)(a2 + 48) = v25;
    if ( (*v20 & 4) != 0 || (*v20 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
      --*(_DWORD *)(a2 + 52);
    *v20 = v9;
    v20[1] = 0;
LABEL_34:
    v20[1] = v12;
LABEL_35:
    v8 = *(_QWORD *)(v8 + 8);
  }
  while ( v3 != v8 );
}
