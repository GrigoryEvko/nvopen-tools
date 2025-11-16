// Function: sub_34F7340
// Address: 0x34f7340
//
__int64 *__fastcall sub_34F7340(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  unsigned __int64 *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 *v8; // r11
  unsigned __int64 v9; // rbx
  __int64 *v10; // rbx
  __int64 *result; // rax
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 *v14; // rdi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r14
  unsigned __int64 *v18; // r13
  __int64 v19; // r11
  __int64 v20; // rcx
  unsigned __int64 *v21; // r12
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // r10
  _QWORD *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r8
  unsigned __int64 v30; // rsi
  __int64 v31; // rcx
  unsigned __int64 v32; // rax
  __int64 i; // rdi
  __int16 v34; // dx
  unsigned int v35; // esi
  __int64 v36; // r10
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  unsigned __int64 v41; // rdi
  unsigned __int64 j; // rax
  __int64 k; // r9
  __int16 v44; // cx
  __int64 *v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned int v50; // eax
  __int64 v51; // rbx
  unsigned int v52; // esi
  __int64 v53; // r9
  unsigned int v54; // edi
  _QWORD *v55; // rax
  __int64 v56; // rcx
  __int64 *v57; // rax
  unsigned __int64 *v58; // rax
  _QWORD *v59; // rdi
  int v60; // eax
  int v61; // ecx
  __int64 v62; // rdx
  unsigned int v63; // eax
  __int64 v64; // r14
  __int64 v65; // r9
  unsigned int v66; // r8d
  _QWORD *v67; // rax
  __int64 v68; // rdi
  unsigned __int64 *v69; // rax
  unsigned int v70; // edi
  int v71; // ecx
  int v72; // edx
  int v73; // eax
  int v74; // r8d
  __int64 v75; // r10
  unsigned int v76; // eax
  __int64 v77; // rsi
  _QWORD *v78; // r9
  int v79; // eax
  int v80; // r8d
  __int64 v81; // r10
  unsigned int v82; // eax
  __int64 v83; // rsi
  _QWORD *v84; // rdx
  int v85; // eax
  int v86; // eax
  _QWORD *v87; // rdx
  int v88; // eax
  int v89; // eax
  int v90; // r14d
  int v91; // r14d
  __int64 v92; // r9
  unsigned int v93; // r10d
  __int64 v94; // rdi
  int v95; // esi
  _QWORD *v96; // rcx
  int v97; // eax
  int v98; // r8d
  __int64 v99; // r10
  unsigned int v100; // r9d
  __int64 v101; // rdi
  int v102; // esi
  _QWORD *v103; // rcx
  int v104; // eax
  int v105; // r9d
  __int64 v106; // r10
  unsigned int v107; // r14d
  int v108; // esi
  __int64 v109; // rdi
  int v110; // eax
  int v111; // r8d
  __int64 v112; // r10
  int v113; // esi
  __int64 v114; // r9
  __int64 v115; // rdi
  int v116; // [rsp+8h] [rbp-58h]
  __int64 v117; // [rsp+8h] [rbp-58h]
  unsigned int v118; // [rsp+10h] [rbp-50h]
  __int64 v119; // [rsp+10h] [rbp-50h]
  __int64 v120; // [rsp+10h] [rbp-50h]
  __int64 v121; // [rsp+10h] [rbp-50h]
  __int64 v122; // [rsp+10h] [rbp-50h]
  __int64 v123; // [rsp+10h] [rbp-50h]
  __int64 v124; // [rsp+10h] [rbp-50h]
  int v125; // [rsp+10h] [rbp-50h]
  unsigned int v126; // [rsp+18h] [rbp-48h]
  __int64 v127; // [rsp+18h] [rbp-48h]
  __int64 v128; // [rsp+18h] [rbp-48h]
  __int64 v129; // [rsp+18h] [rbp-48h]
  int v130; // [rsp+18h] [rbp-48h]
  int v131; // [rsp+18h] [rbp-48h]
  int v132; // [rsp+18h] [rbp-48h]
  int v133; // [rsp+18h] [rbp-48h]
  __int64 v134; // [rsp+18h] [rbp-48h]
  __int64 v135; // [rsp+18h] [rbp-48h]
  __int64 v136; // [rsp+18h] [rbp-48h]
  int v137; // [rsp+18h] [rbp-48h]

  v5 = a2;
  v6 = *(unsigned __int64 **)(a2 + 8);
  if ( *(_BYTE *)(a2 + 28) )
    v7 = *(unsigned int *)(a2 + 20);
  else
    v7 = *(unsigned int *)(a2 + 16);
  v8 = &v6[v7];
  if ( v6 != v8 )
  {
    while ( 1 )
    {
      v9 = *v6;
      if ( *v6 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v8 == ++v6 )
        goto LABEL_6;
    }
    if ( v8 != v6 )
    {
      v17 = *(_QWORD *)(v9 + 24);
      v18 = v8;
      v19 = a1;
      v20 = *(_QWORD *)(a1 + 32);
      v21 = v6;
      if ( !v17 )
        goto LABEL_63;
LABEL_19:
      v22 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
      v23 = *(_DWORD *)(v17 + 24) + 1;
LABEL_20:
      v24 = 0;
      if ( *(_DWORD *)(v20 + 32) > v23 )
        v24 = *(_QWORD *)(*(_QWORD *)(v20 + 24) + 8 * v22);
      v25 = *(_DWORD *)(a4 + 24);
      if ( v25 )
      {
        v26 = *(_QWORD *)(a4 + 8);
        v126 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
        v118 = (v25 - 1) & v126;
        v27 = (_QWORD *)(v26 + 16LL * v118);
        v28 = *v27;
        if ( v24 == *v27 )
          goto LABEL_24;
        v116 = 1;
        v59 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 != -8192 || v59 )
            v27 = v59;
          v118 = (v25 - 1) & (v118 + v116);
          v28 = *(_QWORD *)(v26 + 16LL * v118);
          if ( v24 == v28 )
          {
            v27 = (_QWORD *)(v26 + 16LL * v118);
LABEL_24:
            v29 = v27[1];
            if ( v29 )
            {
              v30 = v27[1];
              v31 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 32LL);
              v32 = v30;
              if ( (*(_DWORD *)(v29 + 44) & 4) != 0 )
              {
                do
                  v32 = *(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL;
                while ( (*(_BYTE *)(v32 + 44) & 4) != 0 );
              }
              if ( (*(_DWORD *)(v29 + 44) & 8) != 0 )
              {
                do
                  v30 = *(_QWORD *)(v30 + 8);
                while ( (*(_BYTE *)(v30 + 44) & 8) != 0 );
              }
              for ( i = *(_QWORD *)(v30 + 8); i != v32; v32 = *(_QWORD *)(v32 + 8) )
              {
                v34 = *(_WORD *)(v32 + 68);
                if ( (unsigned __int16)(v34 - 14) > 4u && v34 != 24 )
                  break;
              }
              v35 = *(_DWORD *)(v31 + 144);
              v36 = *(_QWORD *)(v31 + 128);
              if ( v35 )
              {
                v37 = (v35 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                v38 = (__int64 *)(v36 + 16LL * v37);
                v39 = *v38;
                if ( *v38 == v32 )
                  goto LABEL_35;
                v72 = 1;
                while ( v39 != -4096 )
                {
                  v37 = (v35 - 1) & (v72 + v37);
                  v137 = v72 + 1;
                  v38 = (__int64 *)(v36 + 16LL * v37);
                  v39 = *v38;
                  if ( *v38 == v32 )
                    goto LABEL_35;
                  v72 = v137;
                }
              }
              v38 = (__int64 *)(v36 + 16LL * v35);
LABEL_35:
              v40 = v38[1];
              v41 = v9;
              for ( j = v9; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
                ;
              if ( (*(_DWORD *)(v9 + 44) & 8) != 0 )
              {
                do
                  v41 = *(_QWORD *)(v41 + 8);
                while ( (*(_BYTE *)(v41 + 44) & 8) != 0 );
              }
              for ( k = *(_QWORD *)(v41 + 8); k != j; j = *(_QWORD *)(j + 8) )
              {
                v44 = *(_WORD *)(j + 68);
                if ( (unsigned __int16)(v44 - 14) > 4u && v44 != 24 )
                  break;
              }
              if ( v35 )
              {
                k = v35 - 1;
                v45 = (__int64 *)(v36 + 16LL * ((unsigned int)k & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4))));
                v127 = *v45;
                if ( *v45 == j )
                  goto LABEL_45;
                v70 = k & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
                v71 = 1;
                while ( v127 != -4096 )
                {
                  v70 = k & (v71 + v70);
                  v125 = v71 + 1;
                  v45 = (__int64 *)(v36 + 16LL * v70);
                  v127 = *v45;
                  if ( *v45 == j )
                    goto LABEL_45;
                  v71 = v125;
                }
              }
              v45 = (__int64 *)(v36 + 16LL * v35);
LABEL_45:
              if ( (*(_DWORD *)((v45[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v45[1] >> 1) & 3) <= (*(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v40 >> 1) & 3) )
              {
                v46 = v29;
                v29 = v9;
                v9 = v46;
              }
              v47 = *(unsigned int *)(a3 + 8);
              if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
              {
                v119 = v19;
                v128 = v29;
                sub_C8D5F0(a3, (const void *)(a3 + 16), v47 + 1, 8u, v29, k);
                v19 = v119;
                v29 = v128;
                v47 = *(unsigned int *)(a3 + 8);
              }
              *(_QWORD *)(*(_QWORD *)a3 + 8 * v47) = v9;
              ++*(_DWORD *)(a3 + 8);
              v48 = *(_QWORD *)(v19 + 32);
              if ( v17 )
              {
                v49 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
                v50 = *(_DWORD *)(v17 + 24) + 1;
              }
              else
              {
                v49 = 0;
                v50 = 0;
              }
              v51 = 0;
              if ( v50 < *(_DWORD *)(v48 + 32) )
                v51 = *(_QWORD *)(*(_QWORD *)(v48 + 24) + 8 * v49);
              v52 = *(_DWORD *)(a4 + 24);
              if ( v52 )
              {
                v53 = *(_QWORD *)(a4 + 8);
                v54 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                v55 = (_QWORD *)(v53 + 16LL * v54);
                v56 = *v55;
                if ( v51 == *v55 )
                {
LABEL_55:
                  v57 = v55 + 1;
                  goto LABEL_56;
                }
                v132 = 1;
                v84 = 0;
                while ( v56 != -4096 )
                {
                  if ( v56 == -8192 && !v84 )
                    v84 = v55;
                  v54 = (v52 - 1) & (v132 + v54);
                  v55 = (_QWORD *)(v53 + 16LL * v54);
                  v56 = *v55;
                  if ( v51 == *v55 )
                    goto LABEL_55;
                  ++v132;
                }
                if ( !v84 )
                  v84 = v55;
                v85 = *(_DWORD *)(a4 + 16);
                ++*(_QWORD *)a4;
                v86 = v85 + 1;
                if ( 4 * v86 < 3 * v52 )
                {
                  if ( v52 - *(_DWORD *)(a4 + 20) - v86 > v52 >> 3 )
                  {
LABEL_113:
                    *(_DWORD *)(a4 + 16) = v86;
                    if ( *v84 != -4096 )
                      --*(_DWORD *)(a4 + 20);
                    *v84 = v51;
                    v57 = v84 + 1;
                    v84[1] = 0;
LABEL_56:
                    *v57 = v29;
LABEL_57:
                    v58 = v21 + 1;
                    if ( v21 + 1 != v18 )
                    {
                      while ( 1 )
                      {
                        v9 = *v58;
                        v21 = v58;
                        if ( *v58 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( v18 == ++v58 )
                          goto LABEL_60;
                      }
                      if ( v18 != v58 )
                      {
                        v17 = *(_QWORD *)(v9 + 24);
                        v20 = *(_QWORD *)(v19 + 32);
                        if ( v17 )
                          goto LABEL_19;
LABEL_63:
                        v22 = 0;
                        v23 = 0;
                        goto LABEL_20;
                      }
                    }
LABEL_60:
                    v5 = a2;
                    goto LABEL_6;
                  }
                  v123 = v19;
                  v136 = v29;
                  sub_34F7160(a4, v52);
                  v104 = *(_DWORD *)(a4 + 24);
                  if ( v104 )
                  {
                    v105 = v104 - 1;
                    v106 = *(_QWORD *)(a4 + 8);
                    v96 = 0;
                    v107 = (v104 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                    v29 = v136;
                    v19 = v123;
                    v108 = 1;
                    v86 = *(_DWORD *)(a4 + 16) + 1;
                    v84 = (_QWORD *)(v106 + 16LL * v107);
                    v109 = *v84;
                    if ( v51 == *v84 )
                      goto LABEL_113;
                    while ( v109 != -4096 )
                    {
                      if ( !v96 && v109 == -8192 )
                        v96 = v84;
                      v107 = v105 & (v108 + v107);
                      v84 = (_QWORD *)(v106 + 16LL * v107);
                      v109 = *v84;
                      if ( v51 == *v84 )
                        goto LABEL_113;
                      ++v108;
                    }
                    goto LABEL_130;
                  }
                  goto LABEL_193;
                }
              }
              else
              {
                ++*(_QWORD *)a4;
              }
              v122 = v19;
              v134 = v29;
              sub_34F7160(a4, 2 * v52);
              v90 = *(_DWORD *)(a4 + 24);
              if ( v90 )
              {
                v91 = v90 - 1;
                v92 = *(_QWORD *)(a4 + 8);
                v29 = v134;
                v19 = v122;
                v93 = v91 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                v86 = *(_DWORD *)(a4 + 16) + 1;
                v84 = (_QWORD *)(v92 + 16LL * v93);
                v94 = *v84;
                if ( v51 == *v84 )
                  goto LABEL_113;
                v95 = 1;
                v96 = 0;
                while ( v94 != -4096 )
                {
                  if ( !v96 && v94 == -8192 )
                    v96 = v84;
                  v93 = v91 & (v95 + v93);
                  v84 = (_QWORD *)(v92 + 16LL * v93);
                  v94 = *v84;
                  if ( v51 == *v84 )
                    goto LABEL_113;
                  ++v95;
                }
LABEL_130:
                if ( v96 )
                  v84 = v96;
                goto LABEL_113;
              }
LABEL_193:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
LABEL_73:
            if ( v17 )
            {
              v62 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
              v63 = *(_DWORD *)(v17 + 24) + 1;
            }
            else
            {
              v62 = 0;
              v63 = 0;
            }
            v64 = 0;
            if ( v63 < *(_DWORD *)(v20 + 32) )
              v64 = *(_QWORD *)(*(_QWORD *)(v20 + 24) + 8 * v62);
            if ( v25 )
            {
              v65 = *(_QWORD *)(a4 + 8);
              v66 = (v25 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
              v67 = (_QWORD *)(v65 + 16LL * v66);
              v68 = *v67;
              if ( v64 == *v67 )
              {
LABEL_79:
                v69 = v67 + 1;
LABEL_80:
                *v69 = v9;
                goto LABEL_57;
              }
              v133 = 1;
              v87 = 0;
              while ( v68 != -4096 )
              {
                if ( !v87 && v68 == -8192 )
                  v87 = v67;
                v66 = (v25 - 1) & (v133 + v66);
                v67 = (_QWORD *)(v65 + 16LL * v66);
                v68 = *v67;
                if ( v64 == *v67 )
                  goto LABEL_79;
                ++v133;
              }
              if ( !v87 )
                v87 = v67;
              v88 = *(_DWORD *)(a4 + 16);
              ++*(_QWORD *)a4;
              v89 = v88 + 1;
              if ( 4 * v89 < 3 * v25 )
              {
                if ( v25 - *(_DWORD *)(a4 + 20) - v89 > v25 >> 3 )
                {
LABEL_122:
                  *(_DWORD *)(a4 + 16) = v89;
                  if ( *v87 != -4096 )
                    --*(_DWORD *)(a4 + 20);
                  *v87 = v64;
                  v69 = v87 + 1;
                  v87[1] = 0;
                  goto LABEL_80;
                }
                v124 = v19;
                sub_34F7160(a4, v25);
                v110 = *(_DWORD *)(a4 + 24);
                if ( v110 )
                {
                  v111 = v110 - 1;
                  v112 = *(_QWORD *)(a4 + 8);
                  v113 = 1;
                  v19 = v124;
                  v89 = *(_DWORD *)(a4 + 16) + 1;
                  v103 = 0;
                  v114 = v111 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                  v87 = (_QWORD *)(v112 + 16 * v114);
                  v115 = *v87;
                  if ( v64 == *v87 )
                    goto LABEL_122;
                  while ( v115 != -4096 )
                  {
                    if ( !v103 && v115 == -8192 )
                      v103 = v87;
                    LODWORD(v114) = v111 & (v113 + v114);
                    v87 = (_QWORD *)(v112 + 16LL * (unsigned int)v114);
                    v115 = *v87;
                    if ( v64 == *v87 )
                      goto LABEL_122;
                    ++v113;
                  }
                  goto LABEL_142;
                }
                goto LABEL_192;
              }
            }
            else
            {
              ++*(_QWORD *)a4;
            }
            v135 = v19;
            sub_34F7160(a4, 2 * v25);
            v97 = *(_DWORD *)(a4 + 24);
            if ( v97 )
            {
              v98 = v97 - 1;
              v99 = *(_QWORD *)(a4 + 8);
              v19 = v135;
              v100 = (v97 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
              v89 = *(_DWORD *)(a4 + 16) + 1;
              v87 = (_QWORD *)(v99 + 16LL * v100);
              v101 = *v87;
              if ( v64 == *v87 )
                goto LABEL_122;
              v102 = 1;
              v103 = 0;
              while ( v101 != -4096 )
              {
                if ( !v103 && v101 == -8192 )
                  v103 = v87;
                v100 = v98 & (v102 + v100);
                v87 = (_QWORD *)(v99 + 16LL * v100);
                v101 = *v87;
                if ( v64 == *v87 )
                  goto LABEL_122;
                ++v102;
              }
LABEL_142:
              if ( v103 )
                v87 = v103;
              goto LABEL_122;
            }
LABEL_192:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          ++v116;
          v59 = v27;
          v27 = (_QWORD *)(v26 + 16LL * v118);
        }
        if ( !v59 )
          v59 = v27;
        v60 = *(_DWORD *)(a4 + 16);
        ++*(_QWORD *)a4;
        v61 = v60 + 1;
        if ( 4 * (v60 + 1) < 3 * v25 )
        {
          if ( v25 - *(_DWORD *)(a4 + 20) - v61 > v25 >> 3 )
          {
LABEL_70:
            *(_DWORD *)(a4 + 16) = v61;
            if ( *v59 != -4096 )
              --*(_DWORD *)(a4 + 20);
            *v59 = v24;
            v59[1] = 0;
            v20 = *(_QWORD *)(v19 + 32);
            v25 = *(_DWORD *)(a4 + 24);
            goto LABEL_73;
          }
          v117 = v19;
          v121 = v24;
          sub_34F7160(a4, v25);
          v79 = *(_DWORD *)(a4 + 24);
          if ( v79 )
          {
            v80 = v79 - 1;
            v81 = *(_QWORD *)(a4 + 8);
            v19 = v117;
            v82 = v80 & v126;
            v61 = *(_DWORD *)(a4 + 16) + 1;
            v24 = v121;
            v59 = (_QWORD *)(v81 + 16LL * (v80 & v126));
            v83 = *v59;
            if ( v121 == *v59 )
              goto LABEL_70;
            v131 = 1;
            v78 = 0;
            while ( v83 != -4096 )
            {
              if ( v83 == -8192 && !v78 )
                v78 = v59;
              v82 = v80 & (v131 + v82);
              v59 = (_QWORD *)(v81 + 16LL * v82);
              v83 = *v59;
              if ( v121 == *v59 )
                goto LABEL_70;
              ++v131;
            }
            goto LABEL_95;
          }
          goto LABEL_194;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v120 = v19;
      v129 = v24;
      sub_34F7160(a4, 2 * v25);
      v73 = *(_DWORD *)(a4 + 24);
      if ( v73 )
      {
        v24 = v129;
        v74 = v73 - 1;
        v75 = *(_QWORD *)(a4 + 8);
        v19 = v120;
        v76 = (v73 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
        v61 = *(_DWORD *)(a4 + 16) + 1;
        v59 = (_QWORD *)(v75 + 16LL * v76);
        v77 = *v59;
        if ( v129 == *v59 )
          goto LABEL_70;
        v130 = 1;
        v78 = 0;
        while ( v77 != -4096 )
        {
          if ( v77 == -8192 && !v78 )
            v78 = v59;
          v76 = v74 & (v130 + v76);
          v59 = (_QWORD *)(v75 + 16LL * v76);
          v77 = *v59;
          if ( v24 == *v59 )
            goto LABEL_70;
          ++v130;
        }
LABEL_95:
        if ( v78 )
          v59 = v78;
        goto LABEL_70;
      }
LABEL_194:
      ++*(_DWORD *)(a4 + 16);
      BUG();
    }
  }
LABEL_6:
  v10 = *(__int64 **)a3;
  result = (__int64 *)*(unsigned int *)(a3 + 8);
  v12 = *(_QWORD *)a3 + 8LL * (_QWORD)result;
  if ( v12 != *(_QWORD *)a3 )
  {
    do
    {
      v13 = *v10;
      if ( *(_BYTE *)(v5 + 28) )
      {
        v14 = *(__int64 **)(v5 + 8);
        v15 = &v14[*(unsigned int *)(v5 + 20)];
        result = v14;
        if ( v14 != v15 )
        {
          while ( v13 != *result )
          {
            if ( v15 == ++result )
              goto LABEL_13;
          }
          v16 = (unsigned int)(*(_DWORD *)(v5 + 20) - 1);
          *(_DWORD *)(v5 + 20) = v16;
          *result = v14[v16];
          ++*(_QWORD *)v5;
        }
      }
      else
      {
        result = sub_C8CA60(v5, v13);
        if ( result )
        {
          *result = -2;
          ++*(_DWORD *)(v5 + 24);
          ++*(_QWORD *)v5;
        }
      }
LABEL_13:
      ++v10;
    }
    while ( (__int64 *)v12 != v10 );
  }
  return result;
}
