// Function: sub_3954A10
// Address: 0x3954a10
//
__int64 __fastcall sub_3954A10(__int64 *a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // r13
  _BYTE *v4; // rbx
  _BYTE *v5; // r12
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // r14
  __int64 result; // rax
  __int64 v10; // r10
  __int64 v11; // r9
  unsigned int v12; // esi
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // ecx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // r13
  __int64 v21; // r10
  __int64 v22; // r9
  __int64 v23; // r15
  unsigned __int64 v24; // r12
  unsigned int v25; // edx
  __int64 v26; // rcx
  __int64 v27; // rax
  int v28; // edi
  _BYTE *v29; // rsi
  _BYTE *v30; // rsi
  __int64 v31; // rcx
  unsigned __int64 v32; // rsi
  unsigned int v33; // edx
  _QWORD *v34; // rax
  __int64 v35; // rdi
  int v36; // r11d
  _QWORD *v37; // r8
  _QWORD *v38; // r8
  int v39; // ebx
  __int64 v40; // rdi
  unsigned int v41; // ecx
  __int64 *v42; // rax
  __int64 v43; // r8
  int v44; // r11d
  _QWORD *v45; // r8
  int v46; // edi
  int v47; // edi
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // r15
  _QWORD *k; // rax
  unsigned __int64 m; // rax
  __int64 v56; // r11
  int v57; // esi
  int v58; // esi
  __int64 v59; // r8
  unsigned int v60; // ecx
  __int64 *v61; // rdx
  __int64 v62; // rdi
  int v63; // r8d
  int v64; // r8d
  unsigned int v65; // esi
  int v66; // r12d
  _QWORD *v67; // r11
  unsigned __int64 v68; // rax
  __int64 v69; // rax
  _QWORD *v70; // rax
  _QWORD *v71; // rdx
  __int64 v72; // rax
  unsigned __int64 v73; // r15
  _QWORD *i; // rax
  unsigned __int64 v75; // rax
  __int64 v76; // r11
  int v77; // esi
  int v78; // esi
  __int64 v79; // r8
  unsigned int v80; // ecx
  __int64 *v81; // rdx
  __int64 v82; // rdi
  int v83; // r8d
  int v84; // r8d
  int v85; // r12d
  _QWORD *v86; // r11
  unsigned int v87; // esi
  __int64 v88; // r15
  int v89; // r12d
  __int64 v90; // r8
  int v91; // r12d
  __int64 v92; // r15
  int v93; // edx
  unsigned int v94; // ecx
  int v95; // r13d
  __int64 *v96; // rsi
  int v97; // r12d
  __int64 *v98; // r13
  int v99; // ecx
  __int64 v100; // rcx
  _QWORD *n; // rcx
  __int64 v102; // rcx
  _QWORD *j; // rcx
  int v104; // r12d
  int v105; // r12d
  __int64 v106; // r15
  int v107; // r13d
  unsigned int v108; // ecx
  __int64 v109; // [rsp+10h] [rbp-80h]
  __int64 v110; // [rsp+18h] [rbp-78h]
  __int64 v111; // [rsp+20h] [rbp-70h]
  __int64 v112; // [rsp+20h] [rbp-70h]
  __int64 v113; // [rsp+20h] [rbp-70h]
  __int64 v114; // [rsp+20h] [rbp-70h]
  int v115; // [rsp+20h] [rbp-70h]
  int v116; // [rsp+20h] [rbp-70h]
  __int64 v117; // [rsp+28h] [rbp-68h]
  __int64 v118; // [rsp+30h] [rbp-60h]
  __int64 v119; // [rsp+30h] [rbp-60h]
  __int64 v120; // [rsp+30h] [rbp-60h]
  __int64 v121; // [rsp+30h] [rbp-60h]
  __int64 v122; // [rsp+30h] [rbp-60h]
  __int64 *v123; // [rsp+30h] [rbp-60h]
  __int64 *v124; // [rsp+30h] [rbp-60h]
  __int64 v125; // [rsp+40h] [rbp-50h]
  __int64 v126; // [rsp+40h] [rbp-50h]
  __int64 v127; // [rsp+40h] [rbp-50h]
  int v128; // [rsp+48h] [rbp-48h]
  __int64 v129; // [rsp+48h] [rbp-48h]
  __int64 v130; // [rsp+48h] [rbp-48h]
  __int64 v131; // [rsp+48h] [rbp-48h]
  __int64 v132; // [rsp+48h] [rbp-48h]
  _QWORD v133[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = (__int64)a1;
  v3 = *a1;
  if ( (*(_BYTE *)(*a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(v3, a2);
    v4 = *(_BYTE **)(v3 + 88);
    v2 = (__int64)a1;
    v5 = &v4[40 * *(_QWORD *)(v3 + 96)];
    if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
    {
      sub_15E08E0(v3, a2);
      v4 = *(_BYTE **)(v3 + 88);
      v2 = (__int64)a1;
    }
  }
  else
  {
    v4 = *(_BYTE **)(v3 + 88);
    v5 = &v4[40 * *(_QWORD *)(v3 + 96)];
  }
  v6 = v2;
  if ( v5 != v4 )
  {
    do
    {
      while ( !sub_3953820(v6, v4) )
      {
        v4 += 40;
        if ( v5 == v4 )
          goto LABEL_8;
      }
      v7 = (__int64)v4;
      v4 += 40;
      sub_39547E0(v6, v7);
    }
    while ( v5 != v4 );
LABEL_8:
    v2 = v6;
  }
  v8 = v2;
  result = *(_QWORD *)v2 + 72LL;
  v110 = result;
  v117 = *(_QWORD *)(*(_QWORD *)v2 + 80LL);
  if ( v117 != result )
  {
    while ( 1 )
    {
      if ( !v117 )
        BUG();
      v10 = v117 - 24;
      v109 = v8 + 56;
      v11 = *(_QWORD *)(v117 + 24);
      if ( v11 != v117 + 16 )
        break;
LABEL_29:
      result = *(_QWORD *)(v117 + 8);
      v117 = result;
      if ( v110 == result )
        return result;
    }
    while ( 1 )
    {
      if ( !v11 )
        BUG();
      if ( *(_BYTE *)(v11 - 8) == 77 )
        break;
LABEL_16:
      v17 = 24LL * (*(_DWORD *)(v11 - 4) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v11 - 1) & 0x40) != 0 )
      {
        v18 = *(_QWORD *)(v11 - 32);
        v19 = v18 + v17;
      }
      else
      {
        v19 = v11 - 24;
        v18 = v11 - 24 - v17;
      }
      if ( v18 != v19 )
      {
        v20 = v10;
        v21 = v11;
        v22 = v19;
        while ( 1 )
        {
          v27 = *(_QWORD *)v18;
          if ( *(_BYTE *)(*(_QWORD *)v18 + 16LL) <= 0x17u )
            goto LABEL_22;
          if ( v20 != *(_QWORD *)(v27 + 40) || *(_BYTE *)(v21 - 8) == 77 )
          {
            v23 = *(unsigned int *)(v8 + 80);
            v24 = *(_QWORD *)(v8 + 64);
            v133[0] = *(_QWORD *)v18;
            if ( !(_DWORD)v23 )
              goto LABEL_33;
            v25 = (v23 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v26 = *(_QWORD *)(v24 + 16LL * v25);
            if ( v27 != v26 )
            {
              v28 = 1;
              while ( v26 != -8 )
              {
                v25 = (v23 - 1) & (v28 + v25);
                v26 = *(_QWORD *)(v24 + 16LL * v25);
                if ( v27 == v26 )
                  goto LABEL_22;
                ++v28;
              }
LABEL_33:
              v29 = *(_BYTE **)(v8 + 96);
              if ( v29 == *(_BYTE **)(v8 + 104) )
              {
                v118 = v22;
                v129 = v21;
                sub_1287830(v8 + 88, v29, v133);
                v30 = *(_BYTE **)(v8 + 96);
                v24 = *(_QWORD *)(v8 + 64);
                v23 = *(unsigned int *)(v8 + 80);
                v22 = v118;
                v21 = v129;
              }
              else
              {
                if ( v29 )
                {
                  *(_QWORD *)v29 = v27;
                  v29 = *(_BYTE **)(v8 + 96);
                  v24 = *(_QWORD *)(v8 + 64);
                  v23 = *(unsigned int *)(v8 + 80);
                }
                v30 = v29 + 8;
                *(_QWORD *)(v8 + 96) = v30;
              }
              v128 = ((__int64)&v30[-*(_QWORD *)(v8 + 88)] >> 3) - 1;
              if ( (_DWORD)v23 )
              {
                v31 = v133[0];
                v32 = (unsigned int)(v23 - 1);
                v33 = v32 & ((LODWORD(v133[0]) >> 9) ^ (LODWORD(v133[0]) >> 4));
                v34 = (_QWORD *)(v24 + 16LL * v33);
                v35 = *v34;
                if ( v133[0] == *v34 )
                {
LABEL_39:
                  *((_DWORD *)v34 + 2) = v128;
                  goto LABEL_22;
                }
                v44 = 1;
                v45 = 0;
                while ( v35 != -8 )
                {
                  if ( v35 == -16 && !v45 )
                    v45 = v34;
                  v33 = v32 & (v44 + v33);
                  v34 = (_QWORD *)(v24 + 16LL * v33);
                  v35 = *v34;
                  if ( v133[0] == *v34 )
                    goto LABEL_39;
                  ++v44;
                }
                v46 = *(_DWORD *)(v8 + 72);
                if ( v45 )
                  v34 = v45;
                ++*(_QWORD *)(v8 + 56);
                v47 = v46 + 1;
                if ( 4 * v47 < (unsigned int)(3 * v23) )
                {
                  if ( (int)v23 - (v47 + *(_DWORD *)(v8 + 76)) <= (unsigned int)v23 >> 3 )
                  {
                    v113 = v22;
                    v121 = v21;
                    v68 = (v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2);
                    v69 = ((((((v68 >> 4) | v68) >> 8) | (v68 >> 4) | v68) >> 16)
                         | (((v68 >> 4) | v68) >> 8)
                         | (v68 >> 4)
                         | v68)
                        + 1;
                    if ( (unsigned int)v69 < 0x40 )
                      LODWORD(v69) = 64;
                    *(_DWORD *)(v8 + 80) = v69;
                    v70 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v69);
                    v21 = v121;
                    v22 = v113;
                    *(_QWORD *)(v8 + 64) = v70;
                    v71 = v70;
                    if ( v24 )
                    {
                      v72 = *(unsigned int *)(v8 + 80);
                      *(_QWORD *)(v8 + 72) = 0;
                      v73 = v24 + 16 * v23;
                      for ( i = &v71[2 * v72]; i != v71; v71 += 2 )
                      {
                        if ( v71 )
                          *v71 = -8;
                      }
                      v75 = v24;
                      do
                      {
                        v76 = *(_QWORD *)v75;
                        if ( *(_QWORD *)v75 != -8 && v76 != -16 )
                        {
                          v77 = *(_DWORD *)(v8 + 80);
                          if ( !v77 )
                          {
                            MEMORY[0] = *(_QWORD *)v75;
                            BUG();
                          }
                          v78 = v77 - 1;
                          v79 = *(_QWORD *)(v8 + 64);
                          v80 = v78 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                          v81 = (__int64 *)(v79 + 16LL * v80);
                          v82 = *v81;
                          if ( *v81 != v76 )
                          {
                            v116 = 1;
                            v124 = 0;
                            while ( v82 != -8 )
                            {
                              if ( !v124 )
                              {
                                if ( v82 != -16 )
                                  v81 = 0;
                                v124 = v81;
                              }
                              v80 = v78 & (v116 + v80);
                              v81 = (__int64 *)(v79 + 16LL * v80);
                              v82 = *v81;
                              if ( v76 == *v81 )
                                goto LABEL_96;
                              ++v116;
                            }
                            if ( v124 )
                              v81 = v124;
                          }
LABEL_96:
                          *v81 = v76;
                          *((_DWORD *)v81 + 2) = *(_DWORD *)(v75 + 8);
                          ++*(_DWORD *)(v8 + 72);
                        }
                        v75 += 16LL;
                      }
                      while ( v73 != v75 );
                      v114 = v22;
                      v122 = v21;
                      j___libc_free_0(v24);
                      v71 = *(_QWORD **)(v8 + 64);
                      v83 = *(_DWORD *)(v8 + 80);
                      v21 = v122;
                      v22 = v114;
                      v47 = *(_DWORD *)(v8 + 72) + 1;
                    }
                    else
                    {
                      v102 = *(unsigned int *)(v8 + 80);
                      *(_QWORD *)(v8 + 72) = 0;
                      v83 = v102;
                      for ( j = &v70[2 * v102]; j != v70; v70 += 2 )
                      {
                        if ( v70 )
                          *v70 = -8;
                      }
                      v47 = 1;
                    }
                    if ( !v83 )
                      goto LABEL_196;
                    v31 = v133[0];
                    v84 = v83 - 1;
                    v85 = 1;
                    v86 = 0;
                    v87 = v84 & ((LODWORD(v133[0]) >> 9) ^ (LODWORD(v133[0]) >> 4));
                    v34 = &v71[2 * v87];
                    v88 = *v34;
                    if ( *v34 != v133[0] )
                    {
                      while ( v88 != -8 )
                      {
                        if ( !v86 && v88 == -16 )
                          v86 = v34;
                        v87 = v84 & (v85 + v87);
                        v34 = &v71[2 * v87];
                        v88 = *v34;
                        if ( v133[0] == *v34 )
                          goto LABEL_58;
                        ++v85;
                      }
                      if ( v86 )
                        v34 = v86;
                    }
                  }
LABEL_58:
                  *(_DWORD *)(v8 + 72) = v47;
                  if ( *v34 != -8 )
                    --*(_DWORD *)(v8 + 76);
                  *v34 = v31;
                  *((_DWORD *)v34 + 2) = 0;
                  goto LABEL_39;
                }
              }
              else
              {
                ++*(_QWORD *)(v8 + 56);
              }
              v111 = v22;
              v119 = v21;
              v48 = ((((((((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                       | (unsigned int)(2 * v23 - 1)
                       | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 4)
                     | (((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v23 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 8)
                   | (((((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v23 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 4)
                   | (((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v23 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 16;
              v49 = (v48
                   | (((((((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                       | (unsigned int)(2 * v23 - 1)
                       | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 4)
                     | (((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v23 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 8)
                   | (((((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v23 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 4)
                   | (((unsigned int)(2 * v23 - 1) | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v23 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v23 - 1) >> 1))
                  + 1;
              if ( (unsigned int)v49 < 0x40 )
                LODWORD(v49) = 64;
              *(_DWORD *)(v8 + 80) = v49;
              v50 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v49);
              v21 = v119;
              v22 = v111;
              *(_QWORD *)(v8 + 64) = v50;
              v51 = v50;
              if ( v24 )
              {
                v52 = *(unsigned int *)(v8 + 80);
                *(_QWORD *)(v8 + 72) = 0;
                v53 = v24 + 16 * v23;
                for ( k = &v51[2 * v52]; k != v51; v51 += 2 )
                {
                  if ( v51 )
                    *v51 = -8;
                }
                for ( m = v24; v53 != m; m += 16LL )
                {
                  v56 = *(_QWORD *)m;
                  if ( *(_QWORD *)m != -16 && v56 != -8 )
                  {
                    v57 = *(_DWORD *)(v8 + 80);
                    if ( !v57 )
                    {
                      MEMORY[0] = *(_QWORD *)m;
                      BUG();
                    }
                    v58 = v57 - 1;
                    v59 = *(_QWORD *)(v8 + 64);
                    v60 = v58 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                    v61 = (__int64 *)(v59 + 16LL * v60);
                    v62 = *v61;
                    if ( *v61 != v56 )
                    {
                      v115 = 1;
                      v123 = 0;
                      while ( v62 != -8 )
                      {
                        if ( v62 == -16 )
                        {
                          if ( v123 )
                            v61 = v123;
                          v123 = v61;
                        }
                        v60 = v58 & (v115 + v60);
                        v61 = (__int64 *)(v59 + 16LL * v60);
                        v62 = *v61;
                        if ( v56 == *v61 )
                          goto LABEL_74;
                        ++v115;
                      }
                      if ( v123 )
                        v61 = v123;
                    }
LABEL_74:
                    *v61 = v56;
                    *((_DWORD *)v61 + 2) = *(_DWORD *)(m + 8);
                    ++*(_DWORD *)(v8 + 72);
                  }
                }
                v112 = v22;
                v120 = v21;
                j___libc_free_0(v24);
                v51 = *(_QWORD **)(v8 + 64);
                v63 = *(_DWORD *)(v8 + 80);
                v21 = v120;
                v22 = v112;
                v47 = *(_DWORD *)(v8 + 72) + 1;
              }
              else
              {
                v100 = *(unsigned int *)(v8 + 80);
                *(_QWORD *)(v8 + 72) = 0;
                v63 = v100;
                for ( n = &v50[2 * v100]; n != v50; v50 += 2 )
                {
                  if ( v50 )
                    *v50 = -8;
                }
                v47 = 1;
              }
              if ( !v63 )
              {
LABEL_196:
                ++*(_DWORD *)(v8 + 72);
                BUG();
              }
              v64 = v63 - 1;
              v65 = v64 & ((LODWORD(v133[0]) >> 9) ^ (LODWORD(v133[0]) >> 4));
              v34 = &v51[2 * v65];
              v31 = *v34;
              if ( v133[0] != *v34 )
              {
                v66 = 1;
                v67 = 0;
                while ( v31 != -8 )
                {
                  if ( !v67 && v31 == -16 )
                    v67 = v34;
                  v65 = v64 & (v66 + v65);
                  v34 = &v51[2 * v65];
                  v31 = *v34;
                  if ( v133[0] == *v34 )
                    goto LABEL_58;
                  ++v66;
                }
                v31 = v133[0];
                if ( v67 )
                  v34 = v67;
              }
              goto LABEL_58;
            }
LABEL_22:
            v18 += 24;
            if ( v22 == v18 )
              goto LABEL_27;
          }
          else
          {
            v18 += 24;
            if ( v22 == v18 )
            {
LABEL_27:
              v11 = v21;
              v10 = v20;
              break;
            }
          }
        }
      }
      v11 = *(_QWORD *)(v11 + 8);
      if ( v117 + 16 == v11 )
        goto LABEL_29;
    }
    v12 = *(_DWORD *)(v8 + 80);
    v13 = v11 - 24;
    v14 = *(_QWORD *)(v8 + 64);
    v133[0] = v11 - 24;
    if ( v12 )
    {
      v15 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v16 = *(_QWORD *)(v14 + 16LL * v15);
      if ( v13 == v16 )
        goto LABEL_16;
      v36 = 1;
      while ( v16 != -8 )
      {
        v15 = (v12 - 1) & (v36 + v15);
        v16 = *(_QWORD *)(v14 + 16LL * v15);
        if ( v13 == v16 )
          goto LABEL_16;
        ++v36;
      }
    }
    v37 = *(_QWORD **)(v8 + 96);
    if ( v37 == *(_QWORD **)(v8 + 104) )
    {
      v125 = v11;
      v130 = v10;
      sub_1287830(v8 + 88, *(_BYTE **)(v8 + 96), v133);
      v12 = *(_DWORD *)(v8 + 80);
      v14 = *(_QWORD *)(v8 + 64);
      v11 = v125;
      v10 = v130;
      v39 = ((__int64)(*(_QWORD *)(v8 + 96) - *(_QWORD *)(v8 + 88)) >> 3) - 1;
      if ( v12 )
        goto LABEL_50;
    }
    else
    {
      if ( v37 )
      {
        *v37 = v13;
        v37 = *(_QWORD **)(v8 + 96);
        v14 = *(_QWORD *)(v8 + 64);
        v12 = *(_DWORD *)(v8 + 80);
      }
      v38 = v37 + 1;
      *(_QWORD *)(v8 + 96) = v38;
      v39 = (((__int64)v38 - *(_QWORD *)(v8 + 88)) >> 3) - 1;
      if ( v12 )
      {
LABEL_50:
        v40 = v133[0];
        v41 = (v12 - 1) & ((LODWORD(v133[0]) >> 9) ^ (LODWORD(v133[0]) >> 4));
        v42 = (__int64 *)(v14 + 16LL * v41);
        v43 = *v42;
        if ( v133[0] == *v42 )
        {
LABEL_51:
          *((_DWORD *)v42 + 2) = v39;
          goto LABEL_16;
        }
        v97 = 1;
        v98 = 0;
        while ( v43 != -8 )
        {
          if ( v43 == -16 && !v98 )
            v98 = v42;
          v41 = (v12 - 1) & (v97 + v41);
          v42 = (__int64 *)(v14 + 16LL * v41);
          v43 = *v42;
          if ( v133[0] == *v42 )
            goto LABEL_51;
          ++v97;
        }
        v99 = *(_DWORD *)(v8 + 72);
        if ( v98 )
          v42 = v98;
        ++*(_QWORD *)(v8 + 56);
        v93 = v99 + 1;
        if ( 4 * (v99 + 1) < 3 * v12 )
        {
          if ( v12 - (v93 + *(_DWORD *)(v8 + 76)) > v12 >> 3 )
            goto LABEL_120;
          v127 = v11;
          v132 = v10;
          sub_1BFE340(v109, v12);
          v104 = *(_DWORD *)(v8 + 80);
          if ( !v104 )
          {
LABEL_195:
            ++*(_DWORD *)(v8 + 72);
            BUG();
          }
          v90 = v133[0];
          v105 = v104 - 1;
          v106 = *(_QWORD *)(v8 + 64);
          v107 = 1;
          v10 = v132;
          v11 = v127;
          v93 = *(_DWORD *)(v8 + 72) + 1;
          v96 = 0;
          v108 = v105 & ((LODWORD(v133[0]) >> 9) ^ (LODWORD(v133[0]) >> 4));
          v42 = (__int64 *)(v106 + 16LL * v108);
          v40 = *v42;
          if ( v133[0] == *v42 )
            goto LABEL_120;
          while ( v40 != -8 )
          {
            if ( !v96 && v40 == -16 )
              v96 = v42;
            v108 = v105 & (v107 + v108);
            v42 = (__int64 *)(v106 + 16LL * v108);
            v40 = *v42;
            if ( v133[0] == *v42 )
              goto LABEL_120;
            ++v107;
          }
          goto LABEL_111;
        }
LABEL_107:
        v126 = v11;
        v131 = v10;
        sub_1BFE340(v109, 2 * v12);
        v89 = *(_DWORD *)(v8 + 80);
        if ( !v89 )
          goto LABEL_195;
        v90 = v133[0];
        v91 = v89 - 1;
        v92 = *(_QWORD *)(v8 + 64);
        v10 = v131;
        v11 = v126;
        v93 = *(_DWORD *)(v8 + 72) + 1;
        v94 = v91 & ((LODWORD(v133[0]) >> 9) ^ (LODWORD(v133[0]) >> 4));
        v42 = (__int64 *)(v92 + 16LL * v94);
        v40 = *v42;
        if ( v133[0] == *v42 )
          goto LABEL_120;
        v95 = 1;
        v96 = 0;
        while ( v40 != -8 )
        {
          if ( v40 == -16 && !v96 )
            v96 = v42;
          v94 = v91 & (v95 + v94);
          v42 = (__int64 *)(v92 + 16LL * v94);
          v40 = *v42;
          if ( v133[0] == *v42 )
            goto LABEL_120;
          ++v95;
        }
LABEL_111:
        v40 = v90;
        if ( v96 )
          v42 = v96;
LABEL_120:
        *(_DWORD *)(v8 + 72) = v93;
        if ( *v42 != -8 )
          --*(_DWORD *)(v8 + 76);
        *v42 = v40;
        *((_DWORD *)v42 + 2) = 0;
        goto LABEL_51;
      }
    }
    ++*(_QWORD *)(v8 + 56);
    goto LABEL_107;
  }
  return result;
}
