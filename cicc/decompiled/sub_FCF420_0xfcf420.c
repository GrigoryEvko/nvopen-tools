// Function: sub_FCF420
// Address: 0xfcf420
//
__int64 __fastcall sub_FCF420(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  _BYTE *v6; // rbx
  _BYTE *v7; // r14
  int v8; // eax
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // eax
  _BYTE *v12; // rdx
  int v13; // edi
  _BYTE *v14; // rsi
  _BYTE *v15; // rsi
  __int64 v16; // r15
  __int64 result; // rax
  __int64 v18; // r10
  __int64 v19; // r11
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdi
  int v23; // ecx
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rax
  _BYTE **v27; // rbx
  _BYTE **v28; // r9
  __int64 v29; // r13
  __int64 v30; // r14
  _BYTE **v31; // r12
  int v32; // eax
  __int64 v33; // rdi
  int v34; // ecx
  unsigned int v35; // eax
  _BYTE *v36; // rdx
  _BYTE *v37; // rsi
  int v38; // r8d
  _BYTE *v39; // rsi
  _BYTE *v40; // rsi
  unsigned int v41; // r8d
  __int64 v42; // r11
  int v43; // r10d
  __int64 v44; // rcx
  unsigned int v45; // edx
  _QWORD *v46; // rax
  __int64 v47; // rdi
  _DWORD *v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rdx
  int v51; // r8d
  _BYTE *v52; // rsi
  _BYTE *v53; // rsi
  unsigned int v54; // r8d
  int v55; // ebx
  __int64 v56; // rsi
  __int64 v57; // r9
  int v58; // r13d
  __int64 *v59; // rax
  unsigned int v60; // ecx
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 *v63; // rsi
  int v64; // eax
  int v65; // edx
  unsigned __int64 v66; // rdx
  unsigned __int64 v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // r11
  __int64 v70; // r8
  __int64 v71; // rdx
  __int64 v72; // r9
  _QWORD *i; // rdx
  __int64 v74; // rax
  __int64 v75; // r14
  __int64 v76; // rdx
  int v77; // edi
  int v78; // edi
  __int64 v79; // r13
  unsigned int v80; // esi
  __int64 *v81; // rcx
  __int64 v82; // r8
  int v83; // r8d
  int v84; // r8d
  unsigned int v85; // edi
  __int64 *v86; // r11
  int v87; // edi
  int v88; // edi
  __int64 v89; // r9
  unsigned int v90; // eax
  __int64 v91; // r8
  __int64 *v92; // r11
  _QWORD *v93; // rax
  _QWORD *v94; // rdx
  __int64 *v95; // rax
  int v96; // ecx
  int v97; // edx
  _DWORD *v98; // rax
  __int64 v99; // rcx
  _QWORD *v100; // rcx
  _QWORD *v101; // rdx
  int v102; // r12d
  __int64 v103; // r8
  int v104; // r12d
  __int64 v105; // r13
  unsigned int v106; // ecx
  int v107; // r9d
  __int64 *v108; // rdi
  int v109; // r12d
  int v110; // r12d
  __int64 v111; // r13
  int v112; // r9d
  unsigned int v113; // ecx
  __int64 v114; // rdx
  __int64 v115; // rcx
  _QWORD *v116; // rax
  _QWORD *v117; // rdx
  int v118; // [rsp+4h] [rbp-7Ch]
  __int64 v119; // [rsp+8h] [rbp-78h]
  __int64 *v120; // [rsp+8h] [rbp-78h]
  int v121; // [rsp+10h] [rbp-70h]
  __int64 v122; // [rsp+10h] [rbp-70h]
  int v123; // [rsp+10h] [rbp-70h]
  __int64 v124; // [rsp+10h] [rbp-70h]
  __int64 v125; // [rsp+10h] [rbp-70h]
  __int64 v126; // [rsp+10h] [rbp-70h]
  __int64 v127; // [rsp+10h] [rbp-70h]
  int v128; // [rsp+18h] [rbp-68h]
  unsigned int v129; // [rsp+18h] [rbp-68h]
  __int64 v130; // [rsp+18h] [rbp-68h]
  int v131; // [rsp+18h] [rbp-68h]
  int v132; // [rsp+18h] [rbp-68h]
  int v133; // [rsp+18h] [rbp-68h]
  __int64 v134; // [rsp+18h] [rbp-68h]
  __int64 v135; // [rsp+18h] [rbp-68h]
  __int64 v136; // [rsp+18h] [rbp-68h]
  __int64 v137; // [rsp+18h] [rbp-68h]
  __int64 v138; // [rsp+20h] [rbp-60h]
  __int64 v139; // [rsp+28h] [rbp-58h]
  __int64 v140; // [rsp+30h] [rbp-50h]
  int v141; // [rsp+38h] [rbp-48h]
  __int64 v142[7]; // [rsp+48h] [rbp-38h] BYREF

  v5 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 2LL) & 1) != 0 )
  {
    sub_B2C6D0(*(_QWORD *)a1, a2, a3, a4);
    v6 = *(_BYTE **)(v5 + 96);
    v7 = &v6[40 * *(_QWORD *)(v5 + 104)];
    if ( (*(_BYTE *)(v5 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v5, a2, v114, v115);
      v6 = *(_BYTE **)(v5 + 96);
    }
  }
  else
  {
    v6 = *(_BYTE **)(v5 + 96);
    v7 = &v6[40 * *(_QWORD *)(v5 + 104)];
  }
  if ( v6 != v7 )
  {
    while ( 1 )
    {
      if ( !sub_FCDB90(a1, (__int64)v6) )
        goto LABEL_5;
      v8 = *(_DWORD *)(a1 + 80);
      v9 = *(_QWORD *)(a1 + 64);
      v142[0] = (__int64)v6;
      if ( v8 )
      {
        v10 = v8 - 1;
        v11 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v12 = *(_BYTE **)(v9 + 16LL * v11);
        if ( v6 == v12 )
          goto LABEL_5;
        v13 = 1;
        while ( v12 != (_BYTE *)-4096LL )
        {
          v11 = v10 & (v13 + v11);
          v12 = *(_BYTE **)(v9 + 16LL * v11);
          if ( v12 == v6 )
            goto LABEL_5;
          ++v13;
        }
      }
      if ( *v6 == 22 )
      {
        if ( *(_BYTE *)(a1 + 204) )
        {
          v116 = *(_QWORD **)(a1 + 184);
          v117 = &v116[*(unsigned int *)(a1 + 196)];
          if ( v116 == v117 )
            goto LABEL_12;
          while ( (_BYTE *)*v116 != v6 )
          {
            if ( v117 == ++v116 )
              goto LABEL_12;
          }
        }
        else if ( !sub_C8CA60(a1 + 176, (__int64)v6) )
        {
          goto LABEL_12;
        }
LABEL_5:
        v6 += 40;
        if ( v7 == v6 )
          break;
      }
      else
      {
LABEL_12:
        v14 = *(_BYTE **)(a1 + 96);
        if ( v14 == *(_BYTE **)(a1 + 104) )
        {
          sub_9281F0(a1 + 88, v14, v142);
          v15 = *(_BYTE **)(a1 + 96);
        }
        else
        {
          if ( v14 )
          {
            *(_QWORD *)v14 = v142[0];
            v14 = *(_BYTE **)(a1 + 96);
          }
          v15 = v14 + 8;
          *(_QWORD *)(a1 + 96) = v15;
        }
        v6 += 40;
        v141 = ((__int64)&v15[-*(_QWORD *)(a1 + 88)] >> 3) - 1;
        *(_DWORD *)sub_FCF1D0(a1 + 56, v142) = v141;
        if ( v7 == v6 )
          break;
      }
    }
  }
  v16 = a1;
  result = *(_QWORD *)a1 + 72LL;
  v138 = result;
  v140 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( v140 != result )
  {
    while ( 1 )
    {
      if ( !v140 )
        BUG();
      v18 = *(_QWORD *)(v140 + 32);
      v19 = v140 - 24;
      v139 = v16 + 176;
      if ( v18 != v140 + 24 )
        break;
LABEL_37:
      result = *(_QWORD *)(v140 + 8);
      v140 = result;
      if ( v138 == result )
        return result;
    }
    while ( 1 )
    {
      if ( !v18 )
        BUG();
      if ( *(_BYTE *)(v18 - 24) == 84 )
      {
        v20 = *(_DWORD *)(v16 + 80);
        v21 = v18 - 24;
        v22 = *(_QWORD *)(v16 + 64);
        v142[0] = v18 - 24;
        if ( v20 )
        {
          v23 = v20 - 1;
          v24 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v25 = *(_QWORD *)(v22 + 16LL * v24);
          if ( v21 == v25 )
            goto LABEL_24;
          v51 = 1;
          while ( v25 != -4096 )
          {
            v24 = v23 & (v51 + v24);
            v25 = *(_QWORD *)(v22 + 16LL * v24);
            if ( v21 == v25 )
              goto LABEL_24;
            ++v51;
          }
        }
        if ( *(_BYTE *)(v18 - 24) != 22 )
          goto LABEL_63;
        if ( !*(_BYTE *)(v16 + 204) )
        {
          v125 = v18;
          v135 = v19;
          v95 = sub_C8CA60(v139, v21);
          v19 = v135;
          v18 = v125;
          if ( v95 )
            goto LABEL_24;
LABEL_63:
          v52 = *(_BYTE **)(v16 + 96);
          if ( v52 == *(_BYTE **)(v16 + 104) )
          {
            v124 = v18;
            v134 = v19;
            sub_9281F0(v16 + 88, v52, v142);
            v53 = *(_BYTE **)(v16 + 96);
            v18 = v124;
            v19 = v134;
          }
          else
          {
            if ( v52 )
            {
              *(_QWORD *)v52 = v142[0];
              v52 = *(_BYTE **)(v16 + 96);
            }
            v53 = v52 + 8;
            *(_QWORD *)(v16 + 96) = v53;
          }
          v54 = *(_DWORD *)(v16 + 80);
          v55 = ((__int64)&v53[-*(_QWORD *)(v16 + 88)] >> 3) - 1;
          if ( v54 )
          {
            v56 = v142[0];
            v57 = *(_QWORD *)(v16 + 64);
            v58 = 1;
            v59 = 0;
            v60 = (v54 - 1) & ((LODWORD(v142[0]) >> 9) ^ (LODWORD(v142[0]) >> 4));
            v61 = v57 + 16LL * v60;
            v62 = *(_QWORD *)v61;
            if ( v142[0] == *(_QWORD *)v61 )
            {
LABEL_69:
              *(_DWORD *)(v61 + 8) = v55;
              goto LABEL_24;
            }
            while ( v62 != -4096 )
            {
              if ( v62 == -8192 && !v59 )
                v59 = (__int64 *)v61;
              v60 = (v54 - 1) & (v58 + v60);
              v61 = v57 + 16LL * v60;
              v62 = *(_QWORD *)v61;
              if ( v142[0] == *(_QWORD *)v61 )
                goto LABEL_69;
              ++v58;
            }
            v96 = *(_DWORD *)(v16 + 72);
            if ( !v59 )
              v59 = (__int64 *)v61;
            ++*(_QWORD *)(v16 + 56);
            v97 = v96 + 1;
            if ( 4 * (v96 + 1) < 3 * v54 )
            {
              if ( v54 - *(_DWORD *)(v16 + 76) - v97 <= v54 >> 3 )
              {
                v127 = v18;
                v137 = v19;
                sub_CE2410(v16 + 56, v54);
                v109 = *(_DWORD *)(v16 + 80);
                if ( !v109 )
                {
LABEL_206:
                  ++*(_DWORD *)(v16 + 72);
                  BUG();
                }
                v103 = v142[0];
                v110 = v109 - 1;
                v111 = *(_QWORD *)(v16 + 64);
                v112 = 1;
                v19 = v137;
                v18 = v127;
                v97 = *(_DWORD *)(v16 + 72) + 1;
                v108 = 0;
                v113 = v110 & ((LODWORD(v142[0]) >> 9) ^ (LODWORD(v142[0]) >> 4));
                v59 = (__int64 *)(v111 + 16LL * v113);
                v56 = *v59;
                if ( v142[0] != *v59 )
                {
                  while ( v56 != -4096 )
                  {
                    if ( v56 == -8192 && !v108 )
                      v108 = v59;
                    v113 = v110 & (v112 + v113);
                    v59 = (__int64 *)(v111 + 16LL * v113);
                    v56 = *v59;
                    if ( v142[0] == *v59 )
                      goto LABEL_132;
                    ++v112;
                  }
                  goto LABEL_158;
                }
              }
              goto LABEL_132;
            }
          }
          else
          {
            ++*(_QWORD *)(v16 + 56);
          }
          v126 = v18;
          v136 = v19;
          sub_CE2410(v16 + 56, 2 * v54);
          v102 = *(_DWORD *)(v16 + 80);
          if ( !v102 )
            goto LABEL_206;
          v103 = v142[0];
          v104 = v102 - 1;
          v105 = *(_QWORD *)(v16 + 64);
          v19 = v136;
          v18 = v126;
          v97 = *(_DWORD *)(v16 + 72) + 1;
          v106 = v104 & ((LODWORD(v142[0]) >> 9) ^ (LODWORD(v142[0]) >> 4));
          v59 = (__int64 *)(v105 + 16LL * v106);
          v56 = *v59;
          if ( v142[0] != *v59 )
          {
            v107 = 1;
            v108 = 0;
            while ( v56 != -4096 )
            {
              if ( !v108 && v56 == -8192 )
                v108 = v59;
              v106 = v104 & (v107 + v106);
              v59 = (__int64 *)(v105 + 16LL * v106);
              v56 = *v59;
              if ( v142[0] == *v59 )
                goto LABEL_132;
              ++v107;
            }
LABEL_158:
            v56 = v103;
            if ( v108 )
              v59 = v108;
          }
LABEL_132:
          *(_DWORD *)(v16 + 72) = v97;
          if ( *v59 != -4096 )
            --*(_DWORD *)(v16 + 76);
          *v59 = v56;
          v98 = v59 + 1;
          *v98 = 0;
          *v98 = v55;
          goto LABEL_24;
        }
        v93 = *(_QWORD **)(v16 + 184);
        v94 = &v93[*(unsigned int *)(v16 + 196)];
        if ( v93 == v94 )
          goto LABEL_63;
        while ( v21 != *v93 )
        {
          if ( v94 == ++v93 )
            goto LABEL_63;
        }
      }
LABEL_24:
      v26 = 4LL * (*(_DWORD *)(v18 - 20) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v18 - 17) & 0x40) != 0 )
      {
        v27 = *(_BYTE ***)(v18 - 32);
        v28 = &v27[v26];
      }
      else
      {
        v28 = (_BYTE **)(v18 - 24);
        v27 = (_BYTE **)(v18 - 24 - v26 * 8);
      }
      v29 = v19;
      v30 = v18;
      v31 = v28;
      if ( v27 != v28 )
      {
        while ( 1 )
        {
          v37 = *v27;
          if ( **v27 <= 0x1Cu )
            goto LABEL_30;
          if ( v29 != *((_QWORD *)v37 + 5) || *(_BYTE *)(v30 - 24) == 84 )
          {
            v32 = *(_DWORD *)(v16 + 80);
            v33 = *(_QWORD *)(v16 + 64);
            v142[0] = (__int64)*v27;
            if ( !v32 )
              goto LABEL_41;
            v34 = v32 - 1;
            v35 = (v32 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
            v36 = *(_BYTE **)(v33 + 16LL * v35);
            if ( v37 != v36 )
            {
              v38 = 1;
              while ( v36 != (_BYTE *)-4096LL )
              {
                v35 = v34 & (v38 + v35);
                v36 = *(_BYTE **)(v33 + 16LL * v35);
                if ( v37 == v36 )
                  goto LABEL_30;
                ++v38;
              }
LABEL_41:
              if ( *v37 == 22 )
              {
                if ( *(_BYTE *)(v16 + 204) )
                {
                  v49 = *(_QWORD **)(v16 + 184);
                  v50 = &v49[*(unsigned int *)(v16 + 196)];
                  if ( v49 != v50 )
                  {
                    while ( v37 != (_BYTE *)*v49 )
                    {
                      if ( v50 == ++v49 )
                        goto LABEL_42;
                    }
                    goto LABEL_30;
                  }
                }
                else if ( sub_C8CA60(v139, (__int64)v37) )
                {
                  goto LABEL_30;
                }
              }
LABEL_42:
              v39 = *(_BYTE **)(v16 + 96);
              if ( v39 == *(_BYTE **)(v16 + 104) )
              {
                sub_9281F0(v16 + 88, v39, v142);
                v40 = *(_BYTE **)(v16 + 96);
              }
              else
              {
                if ( v39 )
                {
                  *(_QWORD *)v39 = v142[0];
                  v39 = *(_BYTE **)(v16 + 96);
                }
                v40 = v39 + 8;
                *(_QWORD *)(v16 + 96) = v40;
              }
              v41 = *(_DWORD *)(v16 + 80);
              v42 = *(_QWORD *)(v16 + 64);
              v43 = ((__int64)&v40[-*(_QWORD *)(v16 + 88)] >> 3) - 1;
              if ( v41 )
              {
                v44 = v142[0];
                v45 = (v41 - 1) & ((LODWORD(v142[0]) >> 9) ^ (LODWORD(v142[0]) >> 4));
                v46 = (_QWORD *)(v42 + 16LL * v45);
                v47 = *v46;
                if ( *v46 == v142[0] )
                {
LABEL_48:
                  v48 = v46 + 1;
LABEL_49:
                  *v48 = v43;
                  goto LABEL_30;
                }
                v128 = 1;
                v63 = 0;
                while ( v47 != -4096 )
                {
                  if ( !v63 && v47 == -8192 )
                    v63 = v46;
                  v45 = (v41 - 1) & (v128 + v45);
                  v46 = (_QWORD *)(v42 + 16LL * v45);
                  v47 = *v46;
                  if ( v142[0] == *v46 )
                    goto LABEL_48;
                  ++v128;
                }
                if ( !v63 )
                  v63 = v46;
                v64 = *(_DWORD *)(v16 + 72);
                ++*(_QWORD *)(v16 + 56);
                v65 = v64 + 1;
                if ( 4 * (v64 + 1) < 3 * v41 )
                {
                  if ( v41 - *(_DWORD *)(v16 + 76) - v65 <= v41 >> 3 )
                  {
                    v132 = v43;
                    sub_CE2410(v16 + 56, v41);
                    v87 = *(_DWORD *)(v16 + 80);
                    if ( !v87 )
                      goto LABEL_204;
                    v44 = v142[0];
                    v88 = v87 - 1;
                    v89 = *(_QWORD *)(v16 + 64);
                    v43 = v132;
                    v90 = v88 & ((LODWORD(v142[0]) >> 9) ^ (LODWORD(v142[0]) >> 4));
                    v65 = *(_DWORD *)(v16 + 72) + 1;
                    v63 = (__int64 *)(v89 + 16LL * v90);
                    v91 = *v63;
                    if ( *v63 != v142[0] )
                    {
                      v133 = 1;
                      v92 = 0;
                      while ( v91 != -4096 )
                      {
                        if ( !v92 && v91 == -8192 )
                          v92 = v63;
                        v90 = v88 & (v133 + v90);
                        v63 = (__int64 *)(v89 + 16LL * v90);
                        v91 = *v63;
                        if ( v142[0] == *v63 )
                          goto LABEL_78;
                        ++v133;
                      }
                      if ( v92 )
                        v63 = v92;
                    }
                  }
LABEL_78:
                  *(_DWORD *)(v16 + 72) = v65;
                  if ( *v63 != -4096 )
                    --*(_DWORD *)(v16 + 76);
                  *v63 = v44;
                  v48 = v63 + 1;
                  *((_DWORD *)v63 + 2) = 0;
                  goto LABEL_49;
                }
              }
              else
              {
                ++*(_QWORD *)(v16 + 56);
              }
              v119 = v42;
              v121 = v43;
              v129 = v41;
              v66 = ((((((((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                       | (2 * v41 - 1)
                       | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 4)
                     | (((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                     | (2 * v41 - 1)
                     | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 8)
                   | (((((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                     | (2 * v41 - 1)
                     | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 4)
                   | (((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                   | (2 * v41 - 1)
                   | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 16;
              v67 = (v66
                   | (((((((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                       | (2 * v41 - 1)
                       | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 4)
                     | (((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                     | (2 * v41 - 1)
                     | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 8)
                   | (((((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                     | (2 * v41 - 1)
                     | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 4)
                   | (((2 * v41 - 1) | ((unsigned __int64)(2 * v41 - 1) >> 1)) >> 2)
                   | (2 * v41 - 1)
                   | ((unsigned __int64)(2 * v41 - 1) >> 1))
                  + 1;
              if ( (unsigned int)v67 < 0x40 )
                LODWORD(v67) = 64;
              *(_DWORD *)(v16 + 80) = v67;
              v68 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v67, 8);
              v69 = v119;
              v70 = v129;
              *(_QWORD *)(v16 + 64) = v68;
              v43 = v121;
              if ( v119 )
              {
                v71 = *(unsigned int *)(v16 + 80);
                *(_QWORD *)(v16 + 72) = 0;
                v130 = 16LL * v129;
                v72 = v119 + 16 * v70;
                for ( i = &v68[2 * v71]; i != v68; v68 += 2 )
                {
                  if ( v68 )
                    *v68 = -4096;
                }
                v74 = v119;
                if ( v119 != v72 )
                {
                  v122 = v30;
                  v75 = v29;
                  do
                  {
                    v76 = *(_QWORD *)v74;
                    if ( *(_QWORD *)v74 != -8192 && v76 != -4096 )
                    {
                      v77 = *(_DWORD *)(v16 + 80);
                      if ( !v77 )
                      {
                        MEMORY[0] = *(_QWORD *)v74;
                        BUG();
                      }
                      v78 = v77 - 1;
                      v79 = *(_QWORD *)(v16 + 64);
                      v80 = v78 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                      v81 = (__int64 *)(v79 + 16LL * v80);
                      v82 = *v81;
                      if ( v76 != *v81 )
                      {
                        v118 = 1;
                        v120 = 0;
                        while ( v82 != -4096 )
                        {
                          if ( !v120 )
                          {
                            if ( v82 != -8192 )
                              v81 = 0;
                            v120 = v81;
                          }
                          v80 = v78 & (v118 + v80);
                          v81 = (__int64 *)(v79 + 16LL * v80);
                          v82 = *v81;
                          if ( v76 == *v81 )
                            goto LABEL_95;
                          ++v118;
                        }
                        if ( v120 )
                          v81 = v120;
                      }
LABEL_95:
                      *v81 = v76;
                      *((_DWORD *)v81 + 2) = *(_DWORD *)(v74 + 8);
                      ++*(_DWORD *)(v16 + 72);
                    }
                    v74 += 16;
                  }
                  while ( v72 != v74 );
                  v29 = v75;
                  v30 = v122;
                }
                v123 = v43;
                sub_C7D6A0(v69, v130, 8);
                v68 = *(_QWORD **)(v16 + 64);
                v83 = *(_DWORD *)(v16 + 80);
                v43 = v123;
                v65 = *(_DWORD *)(v16 + 72) + 1;
              }
              else
              {
                v99 = *(unsigned int *)(v16 + 80);
                *(_QWORD *)(v16 + 72) = 0;
                v83 = v99;
                v100 = &v68[2 * v99];
                if ( v68 != v100 )
                {
                  v101 = v68;
                  do
                  {
                    if ( v101 )
                      *v101 = -4096;
                    v101 += 2;
                  }
                  while ( v100 != v101 );
                }
                v65 = 1;
              }
              if ( !v83 )
              {
LABEL_204:
                ++*(_DWORD *)(v16 + 72);
                BUG();
              }
              v84 = v83 - 1;
              v85 = v84 & ((LODWORD(v142[0]) >> 9) ^ (LODWORD(v142[0]) >> 4));
              v63 = &v68[2 * v85];
              v44 = *v63;
              if ( v142[0] != *v63 )
              {
                v131 = 1;
                v86 = 0;
                while ( v44 != -4096 )
                {
                  if ( v44 == -8192 && !v86 )
                    v86 = v63;
                  v85 = v84 & (v131 + v85);
                  v63 = &v68[2 * v85];
                  v44 = *v63;
                  if ( v142[0] == *v63 )
                    goto LABEL_78;
                  ++v131;
                }
                v44 = v142[0];
                if ( v86 )
                  v63 = v86;
              }
              goto LABEL_78;
            }
LABEL_30:
            v27 += 4;
            if ( v31 == v27 )
              goto LABEL_35;
          }
          else
          {
            v27 += 4;
            if ( v31 == v27 )
            {
LABEL_35:
              v19 = v29;
              v18 = v30;
              break;
            }
          }
        }
      }
      v18 = *(_QWORD *)(v18 + 8);
      if ( v140 + 24 == v18 )
        goto LABEL_37;
    }
  }
  return result;
}
