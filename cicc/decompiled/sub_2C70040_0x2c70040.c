// Function: sub_2C70040
// Address: 0x2c70040
//
__int64 __fastcall sub_2C70040(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r9
  int v6; // r10d
  int v7; // ebx
  unsigned int i; // edx
  __int64 **v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 *v13; // r10
  __int64 **v14; // r11
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r8
  unsigned __int64 v29; // rdi
  int v30; // r14d
  unsigned int v31; // eax
  __int64 v32; // r13
  __int64 *v33; // r12
  __int64 v34; // r15
  __int64 v35; // rdx
  __int64 v36; // rax
  _QWORD *v37; // rdi
  int v38; // ebx
  _QWORD *v39; // rcx
  __int64 *v40; // rdx
  int v41; // r8d
  __int64 **v42; // rsi
  unsigned int k; // edx
  __int64 *v44; // r9
  unsigned int v45; // edx
  unsigned int v46; // edx
  int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // r12
  __int64 v50; // rbx
  __int64 v51; // r14
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 v54; // r15
  int v55; // edi
  unsigned int m; // esi
  __int64 **v57; // rax
  unsigned int v58; // esi
  __int64 v59; // rax
  unsigned int v60; // edx
  __int64 *v61; // rdi
  __int64 v62; // rdx
  __int64 *v63; // rcx
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 *v66; // rax
  __int64 *v67; // rdx
  __int64 *v68; // r8
  __int64 **v69; // r10
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  unsigned int v73; // eax
  __int64 v74; // rbx
  unsigned int v75; // eax
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // rsi
  int v82; // ecx
  int v83; // ecx
  __int64 n; // rax
  __int64 *v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned __int64 v88; // rdi
  int v89; // r14d
  unsigned int v90; // eax
  __int64 v91; // r15
  __int64 v92; // rdx
  __int64 v93; // rax
  _QWORD *v94; // rdi
  int v95; // ebx
  _QWORD *v96; // rcx
  __int64 *v97; // rdx
  int v98; // edi
  unsigned int j; // ebx
  __int64 **v100; // rcx
  __int64 *v101; // r8
  unsigned int v102; // ebx
  __int64 v103; // [rsp+8h] [rbp-398h]
  __int64 v104; // [rsp+10h] [rbp-390h]
  __int64 *v105; // [rsp+18h] [rbp-388h]
  __int64 **v106; // [rsp+20h] [rbp-380h]
  __int64 *v107; // [rsp+30h] [rbp-370h]
  __int64 **v108; // [rsp+38h] [rbp-368h]
  __int64 v109; // [rsp+38h] [rbp-368h]
  unsigned __int8 v110; // [rsp+47h] [rbp-359h]
  __int64 *v111; // [rsp+48h] [rbp-358h]
  __int64 v112; // [rsp+50h] [rbp-350h]
  __int64 *v113; // [rsp+58h] [rbp-348h]
  __int64 v114; // [rsp+60h] [rbp-340h]
  __int64 v115; // [rsp+68h] [rbp-338h]
  __int64 v116; // [rsp+70h] [rbp-330h]
  __int64 v117; // [rsp+78h] [rbp-328h]
  __int64 v118; // [rsp+80h] [rbp-320h]
  __int64 v119; // [rsp+88h] [rbp-318h]
  __int64 *v120; // [rsp+88h] [rbp-318h]
  __int64 *v121; // [rsp+90h] [rbp-310h]
  __int64 **v123; // [rsp+A0h] [rbp-300h]
  __int64 v124; // [rsp+A8h] [rbp-2F8h]
  __int64 v125; // [rsp+A8h] [rbp-2F8h]
  __int64 **v126; // [rsp+B0h] [rbp-2F0h]
  __int64 v127; // [rsp+B0h] [rbp-2F0h]
  __int64 v128; // [rsp+B0h] [rbp-2F0h]
  __int64 *v129; // [rsp+B0h] [rbp-2F0h]
  __int64 v130; // [rsp+B0h] [rbp-2F0h]
  __int64 v131; // [rsp+B0h] [rbp-2F0h]
  __int64 *v132; // [rsp+B8h] [rbp-2E8h]
  __int64 v133; // [rsp+B8h] [rbp-2E8h]
  __int64 *v134; // [rsp+B8h] [rbp-2E8h]
  __int64 v135; // [rsp+C0h] [rbp-2E0h] BYREF
  __int64 v136; // [rsp+C8h] [rbp-2D8h]
  __int64 v137; // [rsp+D0h] [rbp-2D0h]
  __int64 v138; // [rsp+D8h] [rbp-2C8h]
  _QWORD v139[4]; // [rsp+E0h] [rbp-2C0h] BYREF
  __int64 v140; // [rsp+100h] [rbp-2A0h]
  _QWORD v141[8]; // [rsp+120h] [rbp-280h] BYREF
  __int64 *v142; // [rsp+160h] [rbp-240h] BYREF
  __int64 v143; // [rsp+168h] [rbp-238h]
  __int64 v144; // [rsp+170h] [rbp-230h] BYREF
  __int64 v145; // [rsp+178h] [rbp-228h]
  int v146; // [rsp+1A0h] [rbp-200h]

  v1 = a1[6];
  v2 = (__int64 *)a1[5];
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v107 = (__int64 *)v1;
  if ( v2 == (__int64 *)v1 )
  {
    v110 = 0;
    v23 = 0;
    v24 = 0;
    goto LABEL_37;
  }
  v111 = v2;
  v110 = 0;
  do
  {
    v3 = *v111;
    v113 = (__int64 *)*v111;
    sub_2C6EB40(v139, *v111, *a1);
    v116 = v139[1];
    v121 = (__int64 *)v139[2];
    v115 = v139[3];
    v117 = v140;
    if ( v140 == v139[0] )
      goto LABEL_35;
    v4 = v139[0];
    v112 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
    do
    {
LABEL_5:
      if ( !(_DWORD)v138 )
      {
        ++v135;
        goto LABEL_54;
      }
      v5 = 0;
      v6 = 1;
      v7 = ((0xBF58476D1CE4E5B9LL
           * (v112 | ((unsigned __int64)(((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4)) << 32))) >> 31)
         ^ (484763065 * v112);
      for ( i = v7 & (v138 - 1); ; i = (v138 - 1) & v46 )
      {
        v9 = (__int64 **)(v136 + 16LL * i);
        v10 = *v9;
        if ( *v9 == v121 && v113 == v9[1] )
          goto LABEL_15;
        if ( v10 == (__int64 *)-4096LL )
          break;
        if ( v10 == (__int64 *)-8192LL && v9[1] == (__int64 *)-8192LL && !v5 )
          v5 = v136 + 16LL * i;
LABEL_64:
        v46 = v6 + i;
        ++v6;
      }
      if ( v9[1] != (__int64 *)-4096LL )
        goto LABEL_64;
      if ( v5 )
        v9 = (__int64 **)v5;
      ++v135;
      v47 = v137 + 1;
      if ( 4 * ((int)v137 + 1) < (unsigned int)(3 * v138) )
      {
        if ( (int)v138 - HIDWORD(v137) - v47 > (unsigned int)v138 >> 3 )
          goto LABEL_69;
        sub_2C6FD80((__int64)&v135, v138);
        if ( (_DWORD)v138 )
        {
          v98 = 1;
          v9 = 0;
          for ( j = (v138 - 1) & v7; ; j = (v138 - 1) & v102 )
          {
            v100 = (__int64 **)(v136 + 16LL * j);
            v101 = *v100;
            if ( *v100 == v121 && v113 == v100[1] )
            {
              v47 = v137 + 1;
              v9 = (__int64 **)(v136 + 16LL * j);
              goto LABEL_69;
            }
            if ( v101 == (__int64 *)-4096LL )
            {
              if ( v100[1] == (__int64 *)-4096LL )
              {
                if ( !v9 )
                  v9 = (__int64 **)(v136 + 16LL * j);
                v47 = v137 + 1;
                goto LABEL_69;
              }
            }
            else if ( v101 == (__int64 *)-8192LL && v100[1] == (__int64 *)-8192LL && !v9 )
            {
              v9 = (__int64 **)(v136 + 16LL * j);
            }
            v102 = v98 + j;
            ++v98;
          }
        }
LABEL_183:
        LODWORD(v137) = v137 + 1;
        BUG();
      }
LABEL_54:
      sub_2C6FD80((__int64)&v135, 2 * v138);
      if ( !(_DWORD)v138 )
        goto LABEL_183;
      v41 = 1;
      v42 = 0;
      for ( k = (v138 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (v112 | ((unsigned __int64)(((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4)) << 32))) >> 31)
               ^ (484763065 * v112)); ; k = (v138 - 1) & v45 )
      {
        v9 = (__int64 **)(v136 + 16LL * k);
        v44 = *v9;
        if ( *v9 == v121 && v113 == v9[1] )
          break;
        if ( v44 == (__int64 *)-4096LL )
        {
          if ( v9[1] == (__int64 *)-4096LL )
          {
            if ( v42 )
              v9 = v42;
            v47 = v137 + 1;
            goto LABEL_69;
          }
        }
        else if ( v44 == (__int64 *)-8192LL && v9[1] == (__int64 *)-8192LL && !v42 )
        {
          v42 = (__int64 **)(v136 + 16LL * k);
        }
        v45 = v41 + k;
        ++v41;
      }
      v47 = v137 + 1;
LABEL_69:
      LODWORD(v137) = v47;
      if ( *v9 != (__int64 *)-4096LL || v9[1] != (__int64 *)-4096LL )
        --HIDWORD(v137);
      *v9 = v121;
      v9[1] = v113;
      v48 = sub_2C6EE90(v121, v113, (__int64)a1);
      v49 = *a1;
      v50 = (__int64)v48;
      v120 = v48;
      v128 = sub_2C6ED70((__int64)a1, v48);
      sub_2C6EB40(v141, v50, v49);
      v5 = v128;
      v51 = v141[0];
      v52 = v141[1];
      v114 = v4;
      v134 = (__int64 *)v141[2];
      v53 = v128;
      v54 = v141[3];
      v125 = v141[4];
      v118 = ((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4);
LABEL_72:
      if ( v51 == v125 )
        goto LABEL_109;
      while ( 2 )
      {
        if ( !(_DWORD)v138 )
          goto LABEL_90;
        v55 = 1;
        for ( m = (v138 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (v118 | ((unsigned __int64)(((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v118)); ; m = (v138 - 1) & v58 )
        {
          v57 = (__int64 **)(v136 + 16LL * m);
          if ( *v57 == v134 && v120 == v57[1] )
            break;
          if ( *v57 == (__int64 *)-4096LL && v57[1] == (__int64 *)-4096LL )
            goto LABEL_90;
          v58 = v55 + m;
          ++v55;
        }
        v78 = sub_2C6ED70((__int64)a1, v134);
        v142 = &v144;
        v143 = 0x600000000LL;
        v81 = *(unsigned int *)(v78 + 16);
        if ( (_DWORD)v81 )
        {
          v130 = v78;
          sub_2C6DEF0((__int64)&v142, v78 + 8, v78, v79, v80, v5);
          v85 = v142;
          v81 = (unsigned int)v143;
          v61 = &v142[(unsigned int)v143];
          v82 = *(_DWORD *)(v130 + 72);
          v146 = v82;
          if ( v61 != v142 )
          {
            do
            {
              *v85 = ~*v85;
              ++v85;
            }
            while ( v61 != v85 );
            LOBYTE(v82) = v146;
            v61 = v142;
            v81 = (unsigned int)v143;
          }
        }
        else
        {
          v82 = *(_DWORD *)(v78 + 72);
          v61 = &v144;
          v146 = v82;
        }
        v83 = v82 & 0x3F;
        if ( v83 )
        {
          v61[v81 - 1] &= ~(-1LL << v83);
          LODWORD(v81) = v143;
          v61 = v142;
        }
        v60 = v81;
        if ( *(_DWORD *)(v53 + 16) <= (unsigned int)v81 )
          v60 = *(_DWORD *)(v53 + 16);
        if ( v60 )
        {
          for ( n = 0; ; ++n )
          {
            v61[n] &= *(_QWORD *)(*(_QWORD *)(v53 + 8) + n * 8);
            v61 = v142;
            if ( v60 - 1 == n )
              break;
          }
        }
        while ( (_DWORD)v81 != v60 )
        {
          v59 = v60++;
          v61[v59] = 0;
          v61 = v142;
        }
        v62 = 8LL * (unsigned int)v143;
        v63 = &v61[(unsigned __int64)v62 / 8];
        v64 = v62 >> 3;
        v65 = v62 >> 5;
        if ( !v65 )
        {
          v66 = v61;
LABEL_138:
          if ( v64 == 2 )
            goto LABEL_145;
          if ( v64 != 3 )
          {
            if ( v64 == 1 )
              goto LABEL_141;
            goto LABEL_88;
          }
          if ( *v66 )
            goto LABEL_87;
          ++v66;
LABEL_145:
          if ( *v66 )
            goto LABEL_87;
          ++v66;
LABEL_141:
          if ( *v66 )
          {
LABEL_87:
            if ( v63 != v66 )
              goto LABEL_132;
          }
LABEL_88:
          if ( v61 != &v144 )
            _libc_free((unsigned __int64)v61);
LABEL_90:
          v51 = *(_QWORD *)(v51 + 8);
          v68 = &v144;
          v69 = &v142;
          if ( !v51 )
          {
LABEL_108:
            v51 = 0;
            if ( v125 )
              continue;
LABEL_109:
            v4 = v114;
            goto LABEL_15;
          }
          while ( 2 )
          {
            v70 = *(_QWORD *)(v51 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v70 - 30) <= 0xAu )
            {
              v71 = *(_QWORD *)(v70 + 40);
              if ( v71 )
              {
                v72 = (unsigned int)(*(_DWORD *)(v71 + 44) + 1);
                v73 = *(_DWORD *)(v71 + 44) + 1;
              }
              else
              {
                v72 = 0;
                v73 = 0;
              }
              if ( v73 < *(_DWORD *)(v54 + 32) )
              {
                v74 = *(_QWORD *)(*(_QWORD *)(v54 + 24) + 8 * v72);
                if ( v74 )
                {
                  if ( v52 != v74 && v52 && v74 != *(_QWORD *)(v52 + 8) )
                  {
                    if ( v52 == *(_QWORD *)(v74 + 8) || *(_DWORD *)(v74 + 16) >= *(_DWORD *)(v52 + 16) )
                      goto LABEL_115;
                    if ( !*(_BYTE *)(v54 + 112) )
                    {
                      v75 = *(_DWORD *)(v54 + 116) + 1;
                      *(_DWORD *)(v54 + 116) = v75;
                      if ( v75 <= 0x20 )
                      {
                        v76 = v52;
                        do
                        {
                          v77 = v76;
                          v76 = *(_QWORD *)(v76 + 8);
                        }
                        while ( v76 && *(_DWORD *)(v74 + 16) <= *(_DWORD *)(v76 + 16) );
                        if ( v74 != v77 )
                        {
LABEL_115:
                          v134 = (__int64 *)v74;
                          goto LABEL_72;
                        }
                        goto LABEL_107;
                      }
                      v142 = v68;
                      HIDWORD(v143) = 32;
                      v86 = *(_QWORD *)(v54 + 96);
                      if ( v86 )
                      {
                        v87 = *(_QWORD *)(v86 + 24);
                        v88 = (unsigned __int64)v68;
                        v5 = v74;
                        v144 = *(_QWORD *)(v54 + 96);
                        v145 = v87;
                        LODWORD(v143) = 1;
                        v109 = v51;
                        v89 = 1;
                        *(_DWORD *)(v86 + 72) = 0;
                        v90 = 1;
                        v131 = v54;
                        do
                        {
                          v95 = v89++;
                          v96 = (_QWORD *)(v88 + 16LL * v90 - 16);
                          v97 = (__int64 *)v96[1];
                          if ( v97 == (__int64 *)(*(_QWORD *)(*v96 + 24LL) + 8LL * *(unsigned int *)(*v96 + 32LL)) )
                          {
                            --v90;
                            *(_DWORD *)(*v96 + 76LL) = v95;
                            LODWORD(v143) = v90;
                          }
                          else
                          {
                            v91 = *v97;
                            v96[1] = v97 + 1;
                            v92 = (unsigned int)v143;
                            v93 = *(_QWORD *)(v91 + 24);
                            if ( (unsigned __int64)(unsigned int)v143 + 1 > HIDWORD(v143) )
                            {
                              v103 = v5;
                              v104 = *(_QWORD *)(v91 + 24);
                              v105 = v68;
                              v106 = v69;
                              sub_C8D5F0((__int64)v69, v68, (unsigned int)v143 + 1LL, 0x10u, (__int64)v68, v5);
                              v88 = (unsigned __int64)v142;
                              v92 = (unsigned int)v143;
                              v5 = v103;
                              v93 = v104;
                              v68 = v105;
                              v69 = v106;
                            }
                            v94 = (_QWORD *)(16 * v92 + v88);
                            *v94 = v91;
                            v94[1] = v93;
                            v90 = v143 + 1;
                            LODWORD(v143) = v143 + 1;
                            *(_DWORD *)(v91 + 72) = v95;
                            v88 = (unsigned __int64)v142;
                          }
                        }
                        while ( v90 );
                        v54 = v131;
                        v51 = v109;
                        v74 = v5;
                        *(_DWORD *)(v131 + 116) = 0;
                        *(_BYTE *)(v131 + 112) = 1;
                        if ( (__int64 *)v88 != v68 )
                        {
                          v108 = v69;
                          v129 = v68;
                          _libc_free(v88);
                          v68 = v129;
                          v69 = v108;
                        }
                      }
                    }
                    if ( *(_DWORD *)(v52 + 72) < *(_DWORD *)(v74 + 72) || *(_DWORD *)(v52 + 76) > *(_DWORD *)(v74 + 76) )
                      goto LABEL_115;
                  }
                }
              }
            }
LABEL_107:
            v51 = *(_QWORD *)(v51 + 8);
            if ( !v51 )
              goto LABEL_108;
            continue;
          }
        }
        break;
      }
      v66 = v61;
      v67 = &v61[4 * v65];
      while ( 1 )
      {
        if ( *v66 )
          goto LABEL_87;
        if ( v66[1] )
          break;
        if ( v66[2] )
        {
          v66 += 2;
          goto LABEL_87;
        }
        if ( v66[3] )
        {
          v66 += 3;
          goto LABEL_87;
        }
        v66 += 4;
        if ( v67 == v66 )
        {
          v64 = v63 - v66;
          goto LABEL_138;
        }
      }
      if ( v63 == v66 + 1 )
        goto LABEL_88;
LABEL_132:
      v4 = v114;
      if ( v61 != &v144 )
        _libc_free((unsigned __int64)v61);
      v110 = 1;
LABEL_15:
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
      {
LABEL_34:
        if ( v117 == v4 )
          break;
        goto LABEL_5;
      }
      v11 = v116;
      v12 = v115;
      v13 = &v144;
      v14 = &v142;
      while ( 2 )
      {
        v15 = *(_QWORD *)(v4 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v15 - 30) > 0xAu
          || ((v16 = *(_QWORD *)(v15 + 40)) == 0
            ? (v17 = 0, v18 = 0)
            : (v17 = (unsigned int)(*(_DWORD *)(v16 + 44) + 1), v18 = *(_DWORD *)(v16 + 44) + 1),
              v18 >= *(_DWORD *)(v12 + 32)
           || (v19 = *(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v17)) == 0
           || v11 == v19
           || !v11
           || v19 == *(_QWORD *)(v11 + 8)) )
        {
LABEL_33:
          v4 = *(_QWORD *)(v4 + 8);
          if ( !v4 )
            goto LABEL_34;
          continue;
        }
        break;
      }
      if ( v11 == *(_QWORD *)(v19 + 8) || *(_DWORD *)(v19 + 16) >= *(_DWORD *)(v11 + 16) )
        goto LABEL_43;
      if ( *(_BYTE *)(v12 + 112) )
        goto LABEL_41;
      v20 = *(_DWORD *)(v12 + 116) + 1;
      *(_DWORD *)(v12 + 116) = v20;
      if ( v20 > 0x20 )
      {
        v142 = v13;
        HIDWORD(v143) = 32;
        v26 = *(_QWORD *)(v12 + 96);
        if ( v26 )
        {
          v27 = *(_QWORD *)(v26 + 24);
          v28 = 1;
          v144 = *(_QWORD *)(v12 + 96);
          v29 = (unsigned __int64)v13;
          LODWORD(v143) = 1;
          v145 = v27;
          v133 = v11;
          v30 = 1;
          *(_DWORD *)(v26 + 72) = 0;
          v31 = 1;
          v127 = v4;
          v32 = v19;
          v124 = v12;
          v33 = v13;
          do
          {
            v38 = v30++;
            v39 = (_QWORD *)(v29 + 16LL * v31 - 16);
            v40 = (__int64 *)v39[1];
            if ( v40 == (__int64 *)(*(_QWORD *)(*v39 + 24LL) + 8LL * *(unsigned int *)(*v39 + 32LL)) )
            {
              --v31;
              *(_DWORD *)(*v39 + 76LL) = v38;
              LODWORD(v143) = v31;
            }
            else
            {
              v34 = *v40;
              v39[1] = v40 + 1;
              v35 = (unsigned int)v143;
              v36 = *(_QWORD *)(v34 + 24);
              if ( (unsigned __int64)(unsigned int)v143 + 1 > HIDWORD(v143) )
              {
                v119 = *(_QWORD *)(v34 + 24);
                v123 = v14;
                sub_C8D5F0((__int64)v14, v33, (unsigned int)v143 + 1LL, 0x10u, v28, v5);
                v29 = (unsigned __int64)v142;
                v35 = (unsigned int)v143;
                v36 = v119;
                v14 = v123;
              }
              v37 = (_QWORD *)(16 * v35 + v29);
              *v37 = v34;
              v37[1] = v36;
              v31 = v143 + 1;
              LODWORD(v143) = v143 + 1;
              *(_DWORD *)(v34 + 72) = v38;
              v29 = (unsigned __int64)v142;
            }
          }
          while ( v31 );
          v13 = v33;
          v12 = v124;
          v19 = v32;
          v11 = v133;
          v4 = v127;
          *(_DWORD *)(v124 + 116) = 0;
          *(_BYTE *)(v124 + 112) = 1;
          if ( (__int64 *)v29 != v13 )
          {
            v126 = v14;
            v132 = v13;
            _libc_free(v29);
            v13 = v132;
            v14 = v126;
          }
        }
LABEL_41:
        if ( *(_DWORD *)(v11 + 72) < *(_DWORD *)(v19 + 72) || *(_DWORD *)(v11 + 76) > *(_DWORD *)(v19 + 76) )
          goto LABEL_43;
        goto LABEL_33;
      }
      v21 = v11;
      do
      {
        v22 = v21;
        v21 = *(_QWORD *)(v21 + 8);
      }
      while ( v21 && *(_DWORD *)(v19 + 16) <= *(_DWORD *)(v21 + 16) );
      if ( v19 == v22 )
        goto LABEL_33;
LABEL_43:
      v121 = (__int64 *)v19;
    }
    while ( v117 != v4 );
LABEL_35:
    ++v111;
  }
  while ( v107 != v111 );
  v23 = v136;
  v24 = 16LL * (unsigned int)v138;
LABEL_37:
  sub_C7D6A0(v23, v24, 8);
  return v110;
}
