// Function: sub_1C2BFA0
// Address: 0x1c2bfa0
//
void __fastcall sub_1C2BFA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v5; // r15
  __int64 v6; // rcx
  unsigned int v7; // ebx
  unsigned int v8; // edi
  __int64 *v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // esi
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r11
  int v18; // eax
  __int64 v19; // rbx
  size_t v20; // r14
  __int64 v21; // rax
  __int64 v22; // r11
  char *v23; // r13
  const void *v24; // rsi
  __int64 v25; // rbx
  __int64 v26; // r10
  unsigned int v27; // esi
  __int64 v28; // r13
  unsigned int v29; // edi
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 *v33; // r13
  __int64 v34; // r8
  int v35; // esi
  __int64 v36; // rdx
  unsigned int v37; // edi
  __int64 *v38; // rcx
  __int64 v39; // r10
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rbx
  int v43; // esi
  unsigned int v44; // esi
  __int64 v45; // r10
  __int64 v46; // r9
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // rdi
  unsigned int v50; // esi
  __int64 v51; // rdx
  __int64 v52; // r10
  __int64 v53; // r9
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // rdi
  int v57; // ecx
  int v58; // eax
  int v59; // eax
  _QWORD *v60; // rdi
  int v61; // eax
  int v62; // eax
  _QWORD *v63; // rdi
  int v64; // eax
  int v65; // r11d
  int v66; // r11d
  unsigned int v67; // esi
  int v68; // r10d
  _QWORD *v69; // rax
  int v70; // esi
  int v71; // esi
  int v72; // eax
  _QWORD *v73; // r11
  int v74; // esi
  int v75; // esi
  int v76; // eax
  int v77; // r11d
  int v78; // r11d
  int v79; // r10d
  unsigned int v80; // esi
  int v81; // edx
  int v82; // r9d
  __int64 v83; // rdx
  int v84; // eax
  int v85; // eax
  int v86; // eax
  int v87; // eax
  __int64 v88; // rdi
  unsigned int v89; // ebx
  __int64 v90; // rsi
  int v91; // r10d
  int v92; // eax
  int v93; // eax
  int v94; // eax
  __int64 v95; // rdi
  unsigned int v96; // ebx
  __int64 v97; // rsi
  int v98; // esi
  int v99; // esi
  __int64 v100; // rdi
  int v101; // r14d
  int v102; // ecx
  __int64 v103; // rdi
  unsigned int v104; // r14d
  __int64 v105; // rsi
  int v106; // edi
  __int64 *v107; // rax
  int v108; // edi
  __int64 *v109; // rax
  __int64 *v110; // r11
  __int64 v111; // rax
  __int64 v112; // [rsp+8h] [rbp-78h]
  __int64 v113; // [rsp+8h] [rbp-78h]
  __int64 v114; // [rsp+8h] [rbp-78h]
  __int64 v115; // [rsp+10h] [rbp-70h]
  unsigned int v116; // [rsp+18h] [rbp-68h]
  unsigned int v117; // [rsp+18h] [rbp-68h]
  unsigned int v118; // [rsp+18h] [rbp-68h]
  int v119; // [rsp+18h] [rbp-68h]
  __int64 v120; // [rsp+18h] [rbp-68h]
  __int64 v121; // [rsp+18h] [rbp-68h]
  __int64 v123; // [rsp+28h] [rbp-58h]
  __int64 v124; // [rsp+28h] [rbp-58h]
  __int64 v125; // [rsp+28h] [rbp-58h]
  __int64 *v126; // [rsp+28h] [rbp-58h]
  __int64 v127; // [rsp+28h] [rbp-58h]
  char *v128; // [rsp+30h] [rbp-50h] BYREF
  __int64 v129; // [rsp+38h] [rbp-48h]
  int v130; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(unsigned int *)(v2 + 48);
  if ( (_DWORD)v3 )
  {
    v5 = **(_QWORD **)(a2 + 32);
    v6 = *(_QWORD *)(v2 + 32);
    v7 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
    v8 = (v3 - 1) & v7;
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v5 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 *)(v6 + 16 * v3) && v9[1] )
      {
        v11 = *(_DWORD *)(a1 + 136);
        v115 = a1 + 112;
        if ( v11 )
        {
          LODWORD(v12) = v11 - 1;
          v13 = *(_QWORD *)(a1 + 120);
          v14 = (v11 - 1) & v7;
          v15 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v15;
          if ( v5 == *v15 )
          {
            v17 = v15[1];
            goto LABEL_8;
          }
          v91 = 1;
          v6 = 0;
          while ( v16 != -8 )
          {
            if ( v16 != -16 || v6 )
              v15 = (__int64 *)v6;
            v6 = (unsigned int)(v91 + 1);
            v14 = v12 & (v91 + v14);
            v110 = (__int64 *)(v13 + 16LL * v14);
            v16 = *v110;
            if ( v5 == *v110 )
            {
              v17 = v110[1];
              goto LABEL_8;
            }
            ++v91;
            v6 = (__int64)v15;
            v15 = (__int64 *)(v13 + 16LL * v14);
          }
          if ( !v6 )
            v6 = (__int64)v15;
          v92 = *(_DWORD *)(a1 + 128);
          ++*(_QWORD *)(a1 + 112);
          v16 = (unsigned int)(v92 + 1);
          if ( 4 * (int)v16 < 3 * v11 )
          {
            if ( v11 - *(_DWORD *)(a1 + 132) - (unsigned int)v16 > v11 >> 3 )
              goto LABEL_108;
            sub_1C29D90(v115, v11);
            v93 = *(_DWORD *)(a1 + 136);
            if ( v93 )
            {
              v94 = v93 - 1;
              v95 = *(_QWORD *)(a1 + 120);
              LODWORD(v12) = 1;
              v13 = 0;
              v96 = v94 & v7;
              v16 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
              v6 = v95 + 16LL * v96;
              v97 = *(_QWORD *)v6;
              if ( v5 != *(_QWORD *)v6 )
              {
                while ( v97 != -8 )
                {
                  if ( !v13 && v97 == -16 )
                    v13 = v6;
                  v96 = v94 & (v12 + v96);
                  v6 = v95 + 16LL * v96;
                  v97 = *(_QWORD *)v6;
                  if ( v5 == *(_QWORD *)v6 )
                    goto LABEL_108;
                  LODWORD(v12) = v12 + 1;
                }
LABEL_124:
                if ( v13 )
                  v6 = v13;
                goto LABEL_108;
              }
              goto LABEL_108;
            }
            goto LABEL_197;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 112);
        }
        sub_1C29D90(v115, 2 * v11);
        v86 = *(_DWORD *)(a1 + 136);
        if ( v86 )
        {
          v87 = v86 - 1;
          v88 = *(_QWORD *)(a1 + 120);
          v89 = v87 & v7;
          v16 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
          v6 = v88 + 16LL * v89;
          v90 = *(_QWORD *)v6;
          if ( v5 != *(_QWORD *)v6 )
          {
            LODWORD(v12) = 1;
            v13 = 0;
            while ( v90 != -8 )
            {
              if ( v90 == -16 && !v13 )
                v13 = v6;
              v89 = v87 & (v12 + v89);
              v6 = v88 + 16LL * v89;
              v90 = *(_QWORD *)v6;
              if ( v5 == *(_QWORD *)v6 )
                goto LABEL_108;
              LODWORD(v12) = v12 + 1;
            }
            goto LABEL_124;
          }
LABEL_108:
          *(_DWORD *)(a1 + 128) = v16;
          if ( *(_QWORD *)v6 != -8 )
            --*(_DWORD *)(a1 + 132);
          *(_QWORD *)v6 = v5;
          v17 = 0;
          *(_QWORD *)(v6 + 8) = 0;
LABEL_8:
          v18 = *(_DWORD *)(v17 + 40);
          v128 = 0;
          v129 = 0;
          v130 = v18;
          if ( v18 )
          {
            v123 = v17;
            v19 = (unsigned int)(v18 + 63) >> 6;
            v20 = 8 * v19;
            v21 = malloc(8 * v19);
            v22 = v123;
            v23 = (char *)v21;
            if ( !v21 )
            {
              if ( v20 || (v111 = malloc(1u), v22 = v123, !v111) )
              {
                v127 = v22;
                sub_16BD1C0("Allocation failed", 1u);
                v22 = v127;
              }
              else
              {
                v23 = (char *)v111;
              }
            }
            v24 = *(const void **)(v22 + 24);
            v124 = v22;
            v128 = v23;
            v129 = v19;
            memcpy(v23, v24, v20);
            v17 = v124;
          }
          v25 = *(_QWORD *)(v5 + 48);
          v26 = v5 + 40;
          v125 = a1 + 56;
          if ( v25 == v5 + 40 )
            goto LABEL_19;
          while ( 1 )
          {
            if ( !v25 )
              BUG();
            if ( *(_BYTE *)(v25 - 8) != 77 )
              goto LABEL_19;
            v27 = *(_DWORD *)(a1 + 80);
            v28 = v25 - 24;
            if ( !v27 )
              break;
            LODWORD(v12) = v27 - 1;
            v13 = *(_QWORD *)(a1 + 64);
            v29 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v30 = v13 + 16LL * v29;
            v31 = *(_QWORD *)v30;
            if ( v28 == *(_QWORD *)v30 )
            {
              v6 = *(unsigned int *)(v30 + 8);
              goto LABEL_17;
            }
            v119 = 1;
            v83 = 0;
            while ( 1 )
            {
              if ( v31 == -8 )
              {
                if ( !v83 )
                  v83 = v30;
                v84 = *(_DWORD *)(a1 + 72);
                ++*(_QWORD *)(a1 + 56);
                v85 = v84 + 1;
                if ( 4 * v85 < 3 * v27 )
                {
                  v6 = v27 - *(_DWORD *)(a1 + 76) - v85;
                  if ( (unsigned int)v6 > v27 >> 3 )
                  {
LABEL_102:
                    *(_DWORD *)(a1 + 72) = v85;
                    if ( *(_QWORD *)v83 != -8 )
                      --*(_DWORD *)(a1 + 76);
                    *(_QWORD *)v83 = v28;
                    v32 = 0;
                    *(_DWORD *)(v83 + 8) = 0;
                    v16 = -2;
                    goto LABEL_18;
                  }
                  v114 = v17;
                  v121 = v26;
                  sub_1BFE340(v125, v27);
                  v102 = *(_DWORD *)(a1 + 80);
                  if ( v102 )
                  {
                    v6 = (unsigned int)(v102 - 1);
                    v103 = *(_QWORD *)(a1 + 64);
                    v13 = 0;
                    v104 = v6 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                    v26 = v121;
                    v17 = v114;
                    LODWORD(v12) = 1;
                    v85 = *(_DWORD *)(a1 + 72) + 1;
                    v83 = v103 + 16LL * v104;
                    v105 = *(_QWORD *)v83;
                    if ( v28 != *(_QWORD *)v83 )
                    {
                      while ( v105 != -8 )
                      {
                        if ( !v13 && v105 == -16 )
                          v13 = v83;
                        v104 = v6 & (v12 + v104);
                        v83 = v103 + 16LL * v104;
                        v105 = *(_QWORD *)v83;
                        if ( v28 == *(_QWORD *)v83 )
                          goto LABEL_102;
                        LODWORD(v12) = v12 + 1;
                      }
                      if ( v13 )
                        v83 = v13;
                    }
                    goto LABEL_102;
                  }
LABEL_198:
                  ++*(_DWORD *)(a1 + 72);
                  BUG();
                }
LABEL_128:
                v113 = v17;
                v120 = v26;
                sub_1BFE340(v125, 2 * v27);
                v98 = *(_DWORD *)(a1 + 80);
                if ( v98 )
                {
                  v99 = v98 - 1;
                  v13 = *(_QWORD *)(a1 + 64);
                  v26 = v120;
                  v17 = v113;
                  v6 = v99 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                  v85 = *(_DWORD *)(a1 + 72) + 1;
                  v83 = v13 + 16 * v6;
                  v100 = *(_QWORD *)v83;
                  if ( v28 != *(_QWORD *)v83 )
                  {
                    v101 = 1;
                    v12 = 0;
                    while ( v100 != -8 )
                    {
                      if ( v100 == -16 && !v12 )
                        v12 = v83;
                      v6 = v99 & (unsigned int)(v101 + v6);
                      v83 = v13 + 16LL * (unsigned int)v6;
                      v100 = *(_QWORD *)v83;
                      if ( v28 == *(_QWORD *)v83 )
                        goto LABEL_102;
                      ++v101;
                    }
                    if ( v12 )
                      v83 = v12;
                  }
                  goto LABEL_102;
                }
                goto LABEL_198;
              }
              if ( v83 || v31 != -16 )
                v30 = v83;
              v29 = v12 & (v119 + v29);
              v112 = v13 + 16LL * v29;
              v31 = *(_QWORD *)v112;
              if ( v28 == *(_QWORD *)v112 )
                break;
              ++v119;
              v83 = v30;
              v30 = v13 + 16LL * v29;
            }
            v6 = *(unsigned int *)(v112 + 8);
LABEL_17:
            v32 = 8LL * ((unsigned int)v6 >> 6);
            v16 = ~(1LL << v6);
LABEL_18:
            *(_QWORD *)&v128[v32] &= v16;
            v25 = *(_QWORD *)(v25 + 8);
            if ( v26 == v25 )
            {
LABEL_19:
              sub_1C28EA0(v17 + 48, (__int64)&v128, v16, v6, v13, v12);
              v33 = *(__int64 **)(a2 + 32);
              v126 = *(__int64 **)(a2 + 40);
              if ( v33 == v126 )
              {
LABEL_29:
                _libc_free((unsigned __int64)v128);
                return;
              }
              while ( 2 )
              {
                v41 = *(_QWORD *)(a1 + 8);
                v42 = *v33;
                v43 = *(_DWORD *)(v41 + 24);
                if ( !v43 )
                {
LABEL_27:
                  if ( v42 != v5 )
                    BUG();
                  goto LABEL_25;
                }
                v34 = *(_QWORD *)(v41 + 8);
                v35 = v43 - 1;
                v36 = ((unsigned int)v42 >> 4) ^ ((unsigned int)v42 >> 9);
                v37 = v35 & (((unsigned int)v42 >> 4) ^ ((unsigned int)v42 >> 9));
                v38 = (__int64 *)(v34 + 16LL * v37);
                v39 = *v38;
                if ( v42 != *v38 )
                {
                  v57 = 1;
                  while ( v39 != -8 )
                  {
                    v58 = v57 + 1;
                    v37 = v35 & (v57 + v37);
                    v38 = (__int64 *)(v34 + 16LL * v37);
                    v39 = *v38;
                    if ( v42 == *v38 )
                      goto LABEL_22;
                    v57 = v58;
                  }
                  goto LABEL_27;
                }
LABEL_22:
                if ( v42 == v5 || (v40 = v38[1], a2 != v40) && v42 != **(_QWORD **)(v40 + 32) )
                {
LABEL_25:
                  if ( v126 == ++v33 )
                    goto LABEL_29;
                  continue;
                }
                break;
              }
              v44 = *(_DWORD *)(a1 + 136);
              if ( v44 )
              {
                v45 = *(_QWORD *)(a1 + 120);
                LODWORD(v46) = (v44 - 1) & v36;
                v47 = v45 + 16LL * (unsigned int)v46;
                v48 = *(_QWORD *)v47;
                if ( v42 == *(_QWORD *)v47 )
                {
                  v49 = *(_QWORD *)(v47 + 8);
                  goto LABEL_34;
                }
                v62 = 1;
                v63 = 0;
                while ( v48 != -8 )
                {
                  if ( v63 || v48 != -16 )
                    v47 = (__int64)v63;
                  v108 = v62 + 1;
                  LODWORD(v46) = (v44 - 1) & (v62 + v46);
                  v109 = (__int64 *)(v45 + 16LL * (unsigned int)v46);
                  v48 = *v109;
                  if ( v42 == *v109 )
                  {
                    v49 = v109[1];
                    goto LABEL_34;
                  }
                  v62 = v108;
                  v63 = (_QWORD *)v47;
                  v47 = v45 + 16LL * (unsigned int)v46;
                }
                v64 = *(_DWORD *)(a1 + 128);
                if ( !v63 )
                  v63 = (_QWORD *)v47;
                ++*(_QWORD *)(a1 + 112);
                v47 = (unsigned int)(v64 + 1);
                if ( 4 * (int)v47 < 3 * v44 )
                {
                  LODWORD(v48) = v44 - *(_DWORD *)(a1 + 132) - v47;
                  LODWORD(v46) = v44 >> 3;
                  if ( (unsigned int)v48 <= v44 >> 3 )
                  {
                    v118 = ((unsigned int)v42 >> 4) ^ ((unsigned int)v42 >> 9);
                    sub_1C29D90(v115, v44);
                    v77 = *(_DWORD *)(a1 + 136);
                    if ( !v77 )
                      goto LABEL_197;
                    v36 = v118;
                    v78 = v77 - 1;
                    v46 = *(_QWORD *)(a1 + 120);
                    v79 = 1;
                    v80 = v78 & v118;
                    v47 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
                    v69 = 0;
                    v63 = (_QWORD *)(v46 + 16LL * (v78 & v118));
                    v48 = *v63;
                    if ( v42 != *v63 )
                    {
                      while ( v48 != -8 )
                      {
                        if ( !v69 && v48 == -16 )
                          v69 = v63;
                        v80 = v78 & (v79 + v80);
                        v63 = (_QWORD *)(v46 + 16LL * v80);
                        v48 = *v63;
                        if ( v42 == *v63 )
                          goto LABEL_57;
                        ++v79;
                      }
                      goto LABEL_89;
                    }
                  }
                  goto LABEL_57;
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 112);
              }
              v117 = ((unsigned int)v42 >> 4) ^ ((unsigned int)v42 >> 9);
              sub_1C29D90(v115, 2 * v44);
              v65 = *(_DWORD *)(a1 + 136);
              if ( !v65 )
                goto LABEL_197;
              v36 = v117;
              v66 = v65 - 1;
              v46 = *(_QWORD *)(a1 + 120);
              v67 = v66 & v117;
              v47 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
              v63 = (_QWORD *)(v46 + 16LL * (v66 & v117));
              v48 = *v63;
              if ( v42 != *v63 )
              {
                v68 = 1;
                v69 = 0;
                while ( v48 != -8 )
                {
                  if ( v48 == -16 && !v69 )
                    v69 = v63;
                  v67 = v66 & (v68 + v67);
                  v63 = (_QWORD *)(v46 + 16LL * v67);
                  v48 = *v63;
                  if ( v42 == *v63 )
                    goto LABEL_57;
                  ++v68;
                }
LABEL_89:
                if ( v69 )
                  v63 = v69;
              }
LABEL_57:
              *(_DWORD *)(a1 + 128) = v47;
              if ( *v63 != -8 )
                --*(_DWORD *)(a1 + 132);
              *v63 = v42;
              v63[1] = 0;
              v49 = 0;
LABEL_34:
              v116 = v36;
              sub_1C28EA0(v49 + 24, (__int64)&v128, v36, v47, v48, v46);
              v50 = *(_DWORD *)(a1 + 136);
              v51 = v116;
              if ( v50 )
              {
                v52 = *(_QWORD *)(a1 + 120);
                LODWORD(v53) = (v50 - 1) & v116;
                v54 = v52 + 16LL * (unsigned int)v53;
                v55 = *(_QWORD *)v54;
                if ( v42 == *(_QWORD *)v54 )
                {
                  v56 = *(_QWORD *)(v54 + 8);
                  goto LABEL_37;
                }
                v59 = 1;
                v60 = 0;
                while ( v55 != -8 )
                {
                  if ( v60 || v55 != -16 )
                    v54 = (__int64)v60;
                  v106 = v59 + 1;
                  LODWORD(v53) = (v50 - 1) & (v59 + v53);
                  v107 = (__int64 *)(v52 + 16LL * (unsigned int)v53);
                  v55 = *v107;
                  if ( v42 == *v107 )
                  {
                    v56 = v107[1];
                    goto LABEL_37;
                  }
                  v59 = v106;
                  v60 = (_QWORD *)v54;
                  v54 = v52 + 16LL * (unsigned int)v53;
                }
                v61 = *(_DWORD *)(a1 + 128);
                if ( !v60 )
                  v60 = (_QWORD *)v54;
                ++*(_QWORD *)(a1 + 112);
                v54 = (unsigned int)(v61 + 1);
                if ( 4 * (int)v54 < 3 * v50 )
                {
                  LODWORD(v55) = v50 - *(_DWORD *)(a1 + 132) - v54;
                  LODWORD(v53) = v50 >> 3;
                  if ( (unsigned int)v55 <= v50 >> 3 )
                  {
                    sub_1C29D90(v115, v50);
                    v74 = *(_DWORD *)(a1 + 136);
                    if ( !v74 )
                      goto LABEL_197;
                    v75 = v74 - 1;
                    v53 = *(_QWORD *)(a1 + 120);
                    v73 = 0;
                    v51 = v75 & v116;
                    v54 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
                    v76 = 1;
                    v60 = (_QWORD *)(v53 + 16 * v51);
                    v55 = *v60;
                    if ( v42 != *v60 )
                    {
                      while ( v55 != -8 )
                      {
                        if ( !v73 && v55 == -16 )
                          v73 = v60;
                        v51 = v75 & (unsigned int)(v76 + v51);
                        v60 = (_QWORD *)(v53 + 16LL * (unsigned int)v51);
                        v55 = *v60;
                        if ( v42 == *v60 )
                          goto LABEL_48;
                        ++v76;
                      }
                      goto LABEL_83;
                    }
                  }
                  goto LABEL_48;
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 112);
              }
              sub_1C29D90(v115, 2 * v50);
              v70 = *(_DWORD *)(a1 + 136);
              if ( !v70 )
                goto LABEL_197;
              v71 = v70 - 1;
              v53 = *(_QWORD *)(a1 + 120);
              v51 = v71 & v116;
              v54 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
              v60 = (_QWORD *)(v53 + 16 * v51);
              v55 = *v60;
              if ( v42 != *v60 )
              {
                v72 = 1;
                v73 = 0;
                while ( v55 != -8 )
                {
                  if ( v55 == -16 && !v73 )
                    v73 = v60;
                  v51 = v71 & (unsigned int)(v72 + v51);
                  v60 = (_QWORD *)(v53 + 16LL * (unsigned int)v51);
                  v55 = *v60;
                  if ( v42 == *v60 )
                    goto LABEL_48;
                  ++v72;
                }
LABEL_83:
                if ( v73 )
                  v60 = v73;
              }
LABEL_48:
              *(_DWORD *)(a1 + 128) = v54;
              if ( *v60 != -8 )
                --*(_DWORD *)(a1 + 132);
              *v60 = v42;
              v60[1] = 0;
              v56 = 0;
LABEL_37:
              sub_1C28EA0(v56 + 48, (__int64)&v128, v51, v54, v55, v53);
              goto LABEL_25;
            }
          }
          ++*(_QWORD *)(a1 + 56);
          goto LABEL_128;
        }
LABEL_197:
        ++*(_DWORD *)(a1 + 128);
        BUG();
      }
    }
    else
    {
      v81 = 1;
      while ( v10 != -8 )
      {
        v82 = v81 + 1;
        v8 = (v3 - 1) & (v81 + v8);
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( v5 == *v9 )
          goto LABEL_3;
        v81 = v82;
      }
    }
  }
}
