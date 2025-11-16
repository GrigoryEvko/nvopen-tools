// Function: sub_27D0D10
// Address: 0x27d0d10
//
void __fastcall sub_27D0D10(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  unsigned __int8 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rax
  unsigned int v12; // ecx
  int *v13; // rdx
  int *v14; // r9
  int *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rsi
  int *v18; // r9
  __int64 v19; // r13
  char v20; // al
  char *v21; // rax
  int *v22; // r8
  unsigned int v23; // edx
  int *v24; // rax
  unsigned int v25; // edx
  __int64 v26; // rsi
  __int64 v27; // rcx
  __int64 *v28; // rsi
  int v29; // eax
  __int64 *v30; // r13
  __int64 v31; // r14
  char v32; // al
  unsigned int **v33; // r8
  unsigned int **v34; // rbx
  unsigned int **i; // r12
  unsigned int *v36; // rax
  __int64 *v37; // r15
  __int64 *v38; // rbx
  unsigned int v39; // esi
  __int64 *v40; // rdx
  __int64 v41; // r10
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // rdi
  int v46; // r8d
  unsigned int v47; // r14d
  int *v48; // r9
  int *v49; // rdx
  int *v50; // rax
  int *v51; // r10
  __int64 v52; // rsi
  __int64 v53; // rcx
  int *v54; // rsi
  __int64 *v55; // r14
  __int64 *v56; // rbx
  __int64 v57; // r13
  __int64 v58; // rdi
  __int64 v59; // rax
  int v60; // esi
  __int64 v61; // rax
  int *v62; // r13
  unsigned int v63; // edx
  int *v64; // rax
  unsigned int v65; // edx
  __int64 v66; // rsi
  __int64 v67; // rcx
  __int64 v68; // rax
  int *v69; // rdx
  bool v70; // r9
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  int *v74; // rax
  int *v75; // r9
  __int64 v76; // rcx
  __int64 v77; // rdx
  int *v78; // rax
  int *v79; // r8
  __int64 v80; // rsi
  __int64 v81; // rcx
  int *v82; // rsi
  __int64 v83; // rcx
  __int64 v84; // rax
  int *v85; // rax
  int *v86; // rsi
  __int64 v87; // rcx
  __int64 v88; // rdx
  char *v89; // rax
  int v90; // edx
  int v91; // ecx
  unsigned int v92; // edi
  __int64 v93; // r8
  unsigned int v94; // esi
  char **v95; // rcx
  char *v96; // r9
  int *v97; // rax
  int *v98; // rsi
  int v99; // ecx
  int v100; // edx
  unsigned __int8 v102; // [rsp+1Dh] [rbp-133h]
  unsigned __int8 v103; // [rsp+1Eh] [rbp-132h]
  char v104; // [rsp+1Fh] [rbp-131h]
  unsigned __int8 v105; // [rsp+1Fh] [rbp-131h]
  int v106; // [rsp+20h] [rbp-130h]
  __int64 v107; // [rsp+28h] [rbp-128h]
  int *v108; // [rsp+30h] [rbp-120h]
  unsigned __int8 v109; // [rsp+38h] [rbp-118h]
  char v110; // [rsp+38h] [rbp-118h]
  unsigned __int8 v111; // [rsp+40h] [rbp-110h]
  char v112; // [rsp+40h] [rbp-110h]
  int v113; // [rsp+48h] [rbp-108h]
  int v114[4]; // [rsp+4Ch] [rbp-104h] BYREF
  unsigned int v115; // [rsp+5Ch] [rbp-F4h] BYREF
  __int64 v116; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int *v117; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v118; // [rsp+74h] [rbp-DCh]
  int v119; // [rsp+7Ch] [rbp-D4h]
  unsigned int **v120; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int **v121; // [rsp+88h] [rbp-C8h]
  unsigned int **v122; // [rsp+90h] [rbp-C0h]
  __int64 *v123; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 *v124; // [rsp+A8h] [rbp-A8h]
  __int64 *v125; // [rsp+B0h] [rbp-A0h]
  __int64 v126; // [rsp+C0h] [rbp-90h] BYREF
  int v127; // [rsp+C8h] [rbp-88h] BYREF
  int *v128; // [rsp+D0h] [rbp-80h]
  int *v129; // [rsp+D8h] [rbp-78h]
  int *v130; // [rsp+E0h] [rbp-70h]
  __int64 j; // [rsp+E8h] [rbp-68h]
  char v132[8]; // [rsp+F0h] [rbp-60h] BYREF
  int v133; // [rsp+F8h] [rbp-58h] BYREF
  unsigned __int64 v134; // [rsp+100h] [rbp-50h]
  int *v135; // [rsp+108h] [rbp-48h]
  int *v136; // [rsp+110h] [rbp-40h]
  __int64 v137; // [rsp+118h] [rbp-38h]

  v114[0] = a3;
  v118 = sub_DF98E0(a2);
  v119 = v4;
  if ( !(_BYTE)v4 )
    return;
  v5 = *(_QWORD *)(a1 + 80);
  v120 = 0;
  v6 = v4;
  v121 = 0;
  v113 = v118;
  v122 = 0;
  v104 = BYTE4(v118);
  v129 = &v127;
  v130 = &v127;
  v135 = &v133;
  v136 = &v133;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v127 = 0;
  v128 = 0;
  j = 0;
  v133 = 0;
  v134 = 0;
  v137 = 0;
  v109 = 0;
  v111 = 0;
  if ( v5 == a1 + 72 )
  {
    if ( BYTE4(v118) )
      goto LABEL_60;
    goto LABEL_24;
  }
  while ( 1 )
  {
    if ( !v5 )
      BUG();
    v7 = v5 + 24;
    if ( *(_QWORD *)(v5 + 32) != v5 + 24 )
      break;
LABEL_19:
    v5 = *(_QWORD *)(v5 + 8);
    if ( a1 + 72 == v5 )
    {
      if ( v109 || v111 )
        goto LABEL_60;
      if ( (!v137 || v137 == 1 && v135[8] == v113) && v104 )
      {
        v34 = v120;
        for ( i = v121; i != v34; *(_DWORD *)sub_27D0AC0(a4, (__int64 *)&v117) = v113 )
        {
          v36 = *v34++;
          v117 = v36;
        }
        goto LABEL_60;
      }
LABEL_24:
      v106 = v114[0];
      v12 = v114[0];
      v13 = v128;
      if ( !v128 )
        goto LABEL_76;
      v14 = &v127;
      v15 = v128;
      do
      {
        while ( 1 )
        {
          v16 = *((_QWORD *)v15 + 2);
          v17 = *((_QWORD *)v15 + 3);
          if ( v114[0] <= (unsigned int)v15[8] )
            break;
          v15 = (int *)*((_QWORD *)v15 + 3);
          if ( !v17 )
            goto LABEL_29;
        }
        v14 = v15;
        v15 = (int *)*((_QWORD *)v15 + 2);
      }
      while ( v16 );
LABEL_29:
      if ( v14 == &v127 || v114[0] < (unsigned int)v14[8] )
      {
LABEL_76:
        v105 = 0;
LABEL_77:
        v102 = v6;
        v103 = v105 ^ 1;
        while ( 1 )
        {
          v37 = (__int64 *)v120;
          v38 = (__int64 *)v121;
          if ( v121 == v120 )
            goto LABEL_60;
          v112 = 0;
          do
          {
            while ( 1 )
            {
              v42 = *v37;
              v43 = *(unsigned int *)(a4 + 24);
              v44 = *(_QWORD *)(a4 + 8);
              v116 = *v37;
              if ( (_DWORD)v43 )
              {
                v39 = (v43 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
                v40 = (__int64 *)(v44 + 16LL * v39);
                v41 = *v40;
                if ( v42 == *v40 )
                {
LABEL_81:
                  if ( v40 != (__int64 *)(v44 + 16 * v43) )
                    goto LABEL_82;
                }
                else
                {
                  v90 = 1;
                  while ( v41 != -4096 )
                  {
                    v91 = v90 + 1;
                    v39 = (v43 - 1) & (v90 + v39);
                    v40 = (__int64 *)(v44 + 16LL * v39);
                    v41 = *v40;
                    if ( v42 == *v40 )
                      goto LABEL_81;
                    v90 = v91;
                  }
                }
              }
              v45 = *(_QWORD *)(v42 - 32);
              v46 = v114[0];
              v47 = *(_DWORD *)(*(_QWORD *)(v45 + 8) + 8LL) >> 8;
              v115 = v47;
              if ( v114[0] == v47 )
                break;
              v48 = v128;
              v49 = v128;
              if ( v113 != v47 )
                goto LABEL_86;
LABEL_132:
              if ( !v48 )
                goto LABEL_204;
              v78 = v48;
              v79 = &v127;
              do
              {
                while ( 1 )
                {
                  v80 = *((_QWORD *)v78 + 2);
                  v81 = *((_QWORD *)v78 + 3);
                  if ( v47 <= v78[8] )
                    break;
                  v78 = (int *)*((_QWORD *)v78 + 3);
                  if ( !v81 )
                    goto LABEL_137;
                }
                v79 = v78;
                v78 = (int *)*((_QWORD *)v78 + 2);
              }
              while ( v80 );
LABEL_137:
              if ( v79 == &v127 || (v82 = &v127, v47 < v79[8]) )
              {
LABEL_204:
                if ( v103 | (v106 == v47) )
                {
                  v112 = v103 | (v106 == v47);
                  *(_DWORD *)sub_27D0AC0(a4, &v116) = v47;
                }
                goto LABEL_82;
              }
              do
              {
                while ( 1 )
                {
                  v83 = *((_QWORD *)v49 + 2);
                  v84 = *((_QWORD *)v49 + 3);
                  if ( v47 <= v49[8] )
                    break;
                  v49 = (int *)*((_QWORD *)v49 + 3);
                  if ( !v84 )
                    goto LABEL_144;
                }
                v82 = v49;
                v49 = (int *)*((_QWORD *)v49 + 2);
              }
              while ( v83 );
LABEL_144:
              if ( v82 == &v127 || v47 < v82[8] )
              {
                v117 = &v115;
                if ( *(_QWORD *)(sub_27D0740(&v126, (__int64)v82, &v117) + 80) != 1 )
                  goto LABEL_82;
                v85 = v128;
                if ( !v128 )
                {
                  v86 = &v127;
LABEL_154:
                  v117 = &v115;
                  v86 = (int *)sub_27D0740(&v126, (__int64)v86, &v117);
                  goto LABEL_155;
                }
              }
              else
              {
                if ( *((_QWORD *)v82 + 10) != 1 )
                  goto LABEL_82;
                v85 = v48;
              }
              v86 = &v127;
              do
              {
                while ( 1 )
                {
                  v87 = *((_QWORD *)v85 + 2);
                  v88 = *((_QWORD *)v85 + 3);
                  if ( v85[8] >= v115 )
                    break;
                  v85 = (int *)*((_QWORD *)v85 + 3);
                  if ( !v88 )
                    goto LABEL_152;
                }
                v86 = v85;
                v85 = (int *)*((_QWORD *)v85 + 2);
              }
              while ( v87 );
LABEL_152:
              if ( v86 == &v127 || v115 < v86[8] )
                goto LABEL_154;
LABEL_155:
              if ( *(_DWORD *)(*((_QWORD *)v86 + 8) + 32LL) != v47 )
                goto LABEL_82;
              if ( v106 == v47 )
              {
LABEL_129:
                *(_DWORD *)sub_27D0AC0(a4, &v116) = v47;
                v112 = v102;
                goto LABEL_82;
              }
LABEL_128:
              if ( !v105 )
                goto LABEL_129;
LABEL_82:
              if ( v38 == ++v37 )
                goto LABEL_95;
            }
            v47 = *(_DWORD *)(*((_QWORD *)sub_27CDC30((char *)v45, v114[0]) + 1) + 8LL) >> 8;
            v115 = v47;
            if ( v46 == v47 )
              goto LABEL_82;
            v48 = v128;
            v49 = v128;
            if ( v113 == v47 )
              goto LABEL_132;
LABEL_86:
            if ( v48 )
            {
              v50 = v48;
              v51 = &v127;
              do
              {
                while ( 1 )
                {
                  v52 = *((_QWORD *)v50 + 2);
                  v53 = *((_QWORD *)v50 + 3);
                  if ( v50[8] >= v47 )
                    break;
                  v50 = (int *)*((_QWORD *)v50 + 3);
                  if ( !v53 )
                    goto LABEL_91;
                }
                v51 = v50;
                v50 = (int *)*((_QWORD *)v50 + 2);
              }
              while ( v52 );
LABEL_91:
              if ( v51 != &v127 )
              {
                v54 = &v127;
                if ( v51[8] <= v47 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v72 = *((_QWORD *)v49 + 2);
                      v73 = *((_QWORD *)v49 + 3);
                      if ( v47 <= v49[8] )
                        break;
                      v49 = (int *)*((_QWORD *)v49 + 3);
                      if ( !v73 )
                        goto LABEL_115;
                    }
                    v54 = v49;
                    v49 = (int *)*((_QWORD *)v49 + 2);
                  }
                  while ( v72 );
LABEL_115:
                  if ( v54 == &v127 || v47 < v54[8] )
                  {
                    v117 = &v115;
                    if ( *(_QWORD *)(sub_27D0740(&v126, (__int64)v54, &v117) + 80) != 1 )
                      goto LABEL_82;
                    v74 = v128;
                    if ( v128 )
                      goto LABEL_119;
                    v75 = &v127;
LABEL_125:
                    v117 = &v115;
                    v75 = (int *)sub_27D0740(&v126, (__int64)v75, &v117);
                  }
                  else
                  {
                    if ( *((_QWORD *)v54 + 10) != 1 )
                      goto LABEL_82;
                    v74 = v48;
LABEL_119:
                    v75 = &v127;
                    do
                    {
                      while ( 1 )
                      {
                        v76 = *((_QWORD *)v74 + 2);
                        v77 = *((_QWORD *)v74 + 3);
                        if ( v74[8] >= v115 )
                          break;
                        v74 = (int *)*((_QWORD *)v74 + 3);
                        if ( !v77 )
                          goto LABEL_123;
                      }
                      v75 = v74;
                      v74 = (int *)*((_QWORD *)v74 + 2);
                    }
                    while ( v76 );
LABEL_123:
                    if ( v75 == &v127 || v115 < v75[8] )
                      goto LABEL_125;
                  }
                  v47 = *(_DWORD *)(*((_QWORD *)v75 + 8) + 32LL);
                  if ( v114[0] == v47 )
                    goto LABEL_82;
                  if ( v47 == v106 )
                    goto LABEL_129;
                  goto LABEL_128;
                }
              }
            }
            if ( (v105 & (v46 != v106)) == 0 )
              goto LABEL_82;
            ++v37;
            v112 = v105 & (v46 != v106);
            *(_DWORD *)sub_27D0AC0(a4, &v116) = v106;
          }
          while ( v38 != v37 );
LABEL_95:
          if ( !v112 )
            goto LABEL_60;
          sub_27CDE90(v128);
          v55 = v124;
          v56 = v123;
          v128 = 0;
          v129 = &v127;
          v130 = &v127;
          for ( j = 0; v55 != v56; ++v56 )
          {
            v57 = *v56;
            v58 = *(_QWORD *)(*v56 - 64);
            v59 = *(_QWORD *)(v58 + 8);
            if ( *(_BYTE *)(v59 + 8) == 14 )
            {
              v60 = *(_DWORD *)(v59 + 8) >> 8;
              v115 = v60;
              if ( v114[0] == v60 )
              {
                v89 = sub_27CDC30((char *)v58, v60);
                if ( *v89 == 61 )
                {
                  v117 = (unsigned int *)v89;
                  if ( v60 == *(_DWORD *)(*((_QWORD *)v89 + 1) + 8LL) >> 8 )
                  {
                    v92 = *(_DWORD *)(a4 + 24);
                    v93 = *(_QWORD *)(a4 + 8);
                    if ( v92 )
                    {
                      v94 = (v92 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
                      v95 = (char **)(v93 + 16LL * v94);
                      v96 = *v95;
                      if ( v89 == *v95 )
                      {
LABEL_175:
                        if ( v95 != (char **)(v93 + 16LL * v92) )
                          v115 = *(_DWORD *)sub_27D0AC0(a4, (__int64 *)&v117);
                      }
                      else
                      {
                        v99 = 1;
                        while ( v96 != (char *)-4096LL )
                        {
                          v100 = v99 + 1;
                          v94 = (v92 - 1) & (v99 + v94);
                          v95 = (char **)(v93 + 16LL * v94);
                          v96 = *v95;
                          if ( v89 == *v95 )
                            goto LABEL_175;
                          v99 = v100;
                        }
                      }
                    }
                  }
                  else
                  {
                    v115 = *(_DWORD *)(*((_QWORD *)v89 + 1) + 8LL) >> 8;
                  }
                }
              }
              v61 = *(_QWORD *)(v57 - 32);
              v62 = &v127;
              v63 = *(_DWORD *)(*(_QWORD *)(v61 + 8) + 8LL);
              v64 = v128;
              v65 = v63 >> 8;
              LODWORD(v116) = v65;
              if ( !v128 )
                goto LABEL_106;
              do
              {
                while ( 1 )
                {
                  v66 = *((_QWORD *)v64 + 2);
                  v67 = *((_QWORD *)v64 + 3);
                  if ( v65 <= v64[8] )
                    break;
                  v64 = (int *)*((_QWORD *)v64 + 3);
                  if ( !v67 )
                    goto LABEL_104;
                }
                v62 = v64;
                v64 = (int *)*((_QWORD *)v64 + 2);
              }
              while ( v66 );
LABEL_104:
              if ( v62 == &v127 || v65 < v62[8] )
              {
LABEL_106:
                v117 = (unsigned int *)&v116;
                v62 = (int *)sub_27D0810(&v126, (__int64)v62, &v117);
              }
              v68 = sub_B996D0((__int64)(v62 + 10), &v115);
              if ( v69 )
              {
                v70 = 1;
                if ( !v68 && v62 + 12 != v69 )
                  v70 = v115 < v69[8];
                v108 = v69;
                v110 = v70;
                v71 = sub_22077B0(0x28u);
                *(_DWORD *)(v71 + 32) = v115;
                sub_220F040(v110, v71, v108, (_QWORD *)v62 + 6);
                ++*((_QWORD *)v62 + 10);
              }
            }
          }
        }
      }
      v18 = &v127;
      do
      {
        if ( v114[0] > (unsigned int)v13[8] )
        {
          v13 = (int *)*((_QWORD *)v13 + 3);
        }
        else
        {
          v18 = v13;
          v13 = (int *)*((_QWORD *)v13 + 2);
        }
      }
      while ( v13 );
      if ( v18 == &v127 || v114[0] < (unsigned int)v18[8] )
      {
        v117 = (unsigned int *)v114;
        if ( *(_QWORD *)(sub_27D0740(&v126, (__int64)v18, &v117) + 80) != 1 )
          goto LABEL_60;
        v97 = v128;
        if ( !v128 )
        {
          v98 = &v127;
LABEL_193:
          v117 = (unsigned int *)v114;
          v98 = (int *)sub_27D0740(&v126, (__int64)v98, &v117);
LABEL_194:
          v105 = v6;
          v106 = *(_DWORD *)(*((_QWORD *)v98 + 8) + 32LL);
          goto LABEL_77;
        }
        v12 = v114[0];
      }
      else
      {
        if ( *((_QWORD *)v18 + 10) != 1 )
          goto LABEL_60;
        v97 = v128;
      }
      v98 = &v127;
      do
      {
        if ( v97[8] < v12 )
        {
          v97 = (int *)*((_QWORD *)v97 + 3);
        }
        else
        {
          v98 = v97;
          v97 = (int *)*((_QWORD *)v97 + 2);
        }
      }
      while ( v97 );
      if ( v98 != &v127 && v12 >= v98[8] )
        goto LABEL_194;
      goto LABEL_193;
    }
  }
  v107 = v5;
  v8 = *(_QWORD *)(v5 + 32);
  while ( 1 )
  {
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      v10 = *(_BYTE *)(v8 - 24);
      if ( v10 == 76 )
      {
        v109 = v6;
        goto LABEL_10;
      }
      if ( v10 == 85 )
      {
        v11 = *(_QWORD *)(v8 - 56);
        if ( v11
          && !*(_BYTE *)v11
          && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v8 + 56)
          && (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
        {
          goto LABEL_37;
        }
        goto LABEL_17;
      }
      if ( (unsigned __int8)(v10 - 65) > 1u )
        break;
LABEL_17:
      v8 = *(_QWORD *)(v8 + 8);
      v111 = v6;
      if ( v7 == v8 )
      {
LABEL_18:
        v5 = v107;
        goto LABEL_19;
      }
    }
    if ( v10 == 61 )
    {
      v117 = (unsigned int *)(v8 - 24);
      v9 = *(_QWORD *)(v8 - 16);
      if ( *(_BYTE *)(v9 + 8) == 14 && v114[0] == *(_DWORD *)(v9 + 8) >> 8 )
      {
        v33 = v121;
        if ( v121 == v122 )
        {
          sub_27D05B0((__int64)&v120, v121, &v117);
        }
        else
        {
          if ( v121 )
          {
            *v121 = (unsigned int *)(v8 - 24);
            v33 = v121;
          }
          v121 = v33 + 1;
        }
      }
      goto LABEL_10;
    }
LABEL_37:
    if ( *(_BYTE *)(v8 - 24) == 62 )
    {
      v116 = v8 - 24;
      v19 = *((_QWORD *)sub_27CDC30(*(char **)(v8 - 88), v114[0]) + 1);
      v20 = *(_BYTE *)(v19 + 8);
      if ( v20 == 14 )
      {
        v21 = sub_27CDC30(*(char **)(v8 - 56), v114[0]);
        v22 = &v127;
        v23 = *(_DWORD *)(*((_QWORD *)v21 + 1) + 8LL);
        v24 = v128;
        v25 = v23 >> 8;
        v115 = v25;
        if ( !v128 )
          goto LABEL_46;
        do
        {
          while ( 1 )
          {
            v26 = *((_QWORD *)v24 + 2);
            v27 = *((_QWORD *)v24 + 3);
            if ( v25 <= v24[8] )
              break;
            v24 = (int *)*((_QWORD *)v24 + 3);
            if ( !v27 )
              goto LABEL_44;
          }
          v22 = v24;
          v24 = (int *)*((_QWORD *)v24 + 2);
        }
        while ( v26 );
LABEL_44:
        if ( v22 == &v127 || v25 < v22[8] )
        {
LABEL_46:
          v117 = &v115;
          v22 = (int *)sub_27D0810(&v126, (__int64)v22, &v117);
        }
        LODWORD(v117) = *(_DWORD *)(v19 + 8) >> 8;
        sub_B99770((__int64)(v22 + 10), (unsigned int *)&v117);
        v28 = v124;
        if ( v124 == v125 )
        {
          sub_278FF40((__int64)&v123, v124, &v116);
        }
        else
        {
          if ( v124 )
          {
            *v124 = v116;
            v28 = v124;
          }
          v124 = v28 + 1;
        }
        LODWORD(v117) = *(_DWORD *)(v19 + 8) >> 8;
        sub_B99770((__int64)&v132, (unsigned int *)&v117);
        goto LABEL_10;
      }
      if ( v20 == 15 )
      {
        v29 = *(_DWORD *)(v19 + 12);
        if ( v29 )
          break;
      }
    }
LABEL_10:
    v8 = *(_QWORD *)(v8 + 8);
    if ( v7 == v8 )
      goto LABEL_18;
  }
  v30 = *(__int64 **)(v19 + 16);
  v31 = (__int64)&v30[(unsigned int)(v29 - 1) + 1];
  while ( 1 )
  {
    v32 = *(_BYTE *)(*v30 + 8);
    if ( v32 == 14 || v32 == 15 && (unsigned __int8)sub_27CDFF0(*v30) )
      break;
    if ( (__int64 *)v31 == ++v30 )
      goto LABEL_10;
  }
LABEL_60:
  sub_27CDCC0(v134);
  sub_27CDE90(v128);
  if ( v123 )
    j_j___libc_free_0((unsigned __int64)v123);
  if ( v120 )
    j_j___libc_free_0((unsigned __int64)v120);
}
