// Function: sub_CFA400
// Address: 0xcfa400
//
void __fastcall sub_CFA400(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // rdx
  unsigned int v8; // r12d
  __int64 v9; // r15
  bool v10; // al
  __int64 v11; // rax
  _QWORD *v12; // r13
  unsigned int v13; // esi
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 *v16; // rdi
  int v17; // r9d
  unsigned __int64 v18; // rax
  unsigned int m; // eax
  __int64 *v20; // rcx
  __int64 v21; // r8
  unsigned int v22; // eax
  unsigned int v23; // esi
  __int64 v24; // r10
  __int64 v25; // r12
  int v26; // r15d
  __int64 *v27; // rcx
  unsigned int v28; // r8d
  __int64 *v29; // rax
  __int64 v30; // rdi
  _QWORD *v31; // rdx
  unsigned int v32; // eax
  int v33; // r10d
  unsigned __int64 v34; // rax
  unsigned int v35; // eax
  __int64 *v36; // rdx
  __int64 v37; // rdi
  unsigned int v38; // eax
  unsigned int v39; // esi
  int v40; // r13d
  __int64 v41; // rdx
  unsigned int v42; // eax
  __int64 *v43; // rdi
  int v44; // r8d
  int v45; // esi
  unsigned int i; // eax
  __int64 *v47; // r13
  __int64 v48; // rcx
  unsigned int v49; // eax
  unsigned int v50; // eax
  int v51; // r11d
  __int64 *v52; // r10
  unsigned __int64 v53; // rax
  unsigned int j; // eax
  __int64 v55; // rdi
  unsigned int v56; // eax
  int v57; // eax
  int v58; // eax
  unsigned int v59; // esi
  __int64 v60; // r9
  __int64 v61; // r12
  int v62; // r11d
  __int64 *v63; // rdx
  unsigned int v64; // edi
  __int64 *v65; // rax
  __int64 v66; // rcx
  _QWORD *v67; // rax
  int v68; // edx
  int v69; // edx
  int v70; // edx
  __int64 v71; // r8
  __int64 v72; // rsi
  __int64 v73; // rdi
  int v74; // r11d
  __int64 *v75; // r10
  int v76; // eax
  int v77; // eax
  int v78; // esi
  int v79; // esi
  __int64 v80; // r9
  int v81; // r11d
  __int64 v82; // rdi
  __int64 v83; // r8
  int v84; // ecx
  int v85; // ecx
  int v86; // ecx
  __int64 v87; // r8
  unsigned int v88; // esi
  __int64 v89; // rdi
  int v90; // r11d
  __int64 *v91; // r9
  __int64 v92; // rdx
  unsigned int v93; // eax
  int v94; // r9d
  unsigned int n; // eax
  __int64 v96; // r8
  unsigned int v97; // eax
  int v98; // ecx
  int v99; // ecx
  __int64 v100; // r8
  int v101; // r11d
  unsigned int v102; // esi
  __int64 v103; // rdi
  int v104; // r13d
  __int64 v105; // rcx
  unsigned int v106; // eax
  int v107; // r8d
  int v108; // esi
  unsigned int k; // eax
  __int64 v110; // rdx
  unsigned int v111; // eax
  __int64 v112; // [rsp+8h] [rbp-78h]
  unsigned int v113; // [rsp+10h] [rbp-70h]
  unsigned int v114; // [rsp+10h] [rbp-70h]
  __int64 v115; // [rsp+10h] [rbp-70h]
  __int64 v116; // [rsp+18h] [rbp-68h]
  unsigned __int64 v117; // [rsp+18h] [rbp-68h]
  int v118; // [rsp+20h] [rbp-60h]
  __int64 v119; // [rsp+20h] [rbp-60h]
  __int64 v120; // [rsp+20h] [rbp-60h]
  __int64 *v121; // [rsp+20h] [rbp-60h]
  int v122; // [rsp+20h] [rbp-60h]
  unsigned __int64 v123; // [rsp+20h] [rbp-60h]
  unsigned int v124; // [rsp+2Ch] [rbp-54h]
  __int64 v125; // [rsp+30h] [rbp-50h]
  unsigned int v127[13]; // [rsp+4Ch] [rbp-34h] BYREF

  if ( *(char *)(a1 + 7) >= 0 )
    return;
  v3 = sub_BD2BC0(a1);
  v125 = v4 + v3;
  v5 = *(char *)(a1 + 7) >= 0 ? 0LL : sub_BD2BC0(a1);
  if ( v125 == v5 )
    return;
  v124 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  while ( 1 )
  {
LABEL_6:
    v6 = sub_A6E860(*(_QWORD *)v5 + 16LL, **(_QWORD **)v5);
    v7 = *(unsigned int *)(v5 + 8);
    v8 = v6;
    if ( *(_DWORD *)(v5 + 12) == (_DWORD)v7 )
    {
      v10 = 1;
      v9 = 0;
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 32 * ((unsigned int)v7 - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
      v10 = v9 == 0;
    }
    if ( !v8 && v10 )
      goto LABEL_33;
    if ( (unsigned int)(*(_DWORD *)(v5 + 12) - v7) > 1 )
      break;
    v39 = *(_DWORD *)(a2 + 24);
    if ( !v39 )
    {
      ++*(_QWORD *)a2;
LABEL_46:
      sub_CF9EC0(a2, 2 * v39);
      v40 = *(_DWORD *)(a2 + 24);
      if ( v40 )
      {
        v41 = *(_QWORD *)(a2 + 8);
        v127[0] = v8;
        v120 = v41;
        v42 = sub_CF97C0(v127);
        v43 = 0;
        v44 = 1;
        v45 = v40 - 1;
        for ( i = (v40 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (v42 | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v42)); ; i = v45 & v49 )
        {
          v47 = (__int64 *)(v120 + 48LL * i);
          v48 = *v47;
          if ( v9 == *v47 && v8 == *((_DWORD *)v47 + 2) )
            break;
          if ( v48 == -4096 )
          {
            if ( *((_DWORD *)v47 + 2) == 100 )
              goto LABEL_181;
          }
          else if ( v48 == -8192 && *((_DWORD *)v47 + 2) == 101 && !v43 )
          {
            v43 = (__int64 *)(v120 + 48LL * i);
          }
          v49 = v44 + i;
          ++v44;
        }
LABEL_175:
        v84 = *(_DWORD *)(a2 + 16) + 1;
        goto LABEL_125;
      }
      goto LABEL_200;
    }
    v114 = *(_DWORD *)(a2 + 24);
    v116 = *(_QWORD *)(a2 + 8);
    v127[0] = v8;
    v50 = sub_CF97C0(v127);
    v51 = 1;
    v52 = 0;
    v53 = 0xBF58476D1CE4E5B9LL * (((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32) | v50);
    v39 = v114;
    for ( j = (v114 - 1) & ((v53 >> 31) ^ v53); ; j = (v114 - 1) & v56 )
    {
      v47 = (__int64 *)(v116 + 48LL * j);
      v55 = *v47;
      if ( v9 == *v47 && v8 == *((_DWORD *)v47 + 2) )
      {
        v59 = *((_DWORD *)v47 + 10);
        v60 = v47[3];
        v61 = (__int64)(v47 + 2);
        if ( !v59 )
          goto LABEL_128;
        v62 = 1;
        v63 = 0;
        v64 = (v59 - 1) & v124;
        v65 = (__int64 *)(v60 + 24LL * v64);
        v66 = *v65;
        if ( a1 == *v65 )
        {
LABEL_80:
          v67 = v65 + 1;
          goto LABEL_81;
        }
        while ( v66 != -4096 )
        {
          if ( !v63 && v66 == -8192 )
            v63 = v65;
          v64 = (v59 - 1) & (v62 + v64);
          v65 = (__int64 *)(v60 + 24LL * v64);
          v66 = *v65;
          if ( a1 == *v65 )
            goto LABEL_80;
          ++v62;
        }
        if ( !v63 )
          v63 = v65;
        v76 = *((_DWORD *)v47 + 8);
        ++v47[2];
        v77 = v76 + 1;
        if ( 4 * v77 >= 3 * v59 )
          goto LABEL_129;
        if ( v59 - *((_DWORD *)v47 + 9) - v77 > v59 >> 3 )
          goto LABEL_109;
        sub_CFA200((__int64)(v47 + 2), v59);
        v98 = *((_DWORD *)v47 + 10);
        if ( v98 )
        {
          v99 = v98 - 1;
          v100 = v47[3];
          v101 = 1;
          v91 = 0;
          v102 = v99 & v124;
          v63 = (__int64 *)(v100 + 24LL * (v99 & v124));
          v103 = *v63;
          v77 = *((_DWORD *)v47 + 8) + 1;
          if ( a1 != *v63 )
          {
            while ( v103 != -4096 )
            {
              if ( v103 == -8192 && !v91 )
                v91 = v63;
              v102 = v99 & (v101 + v102);
              v63 = (__int64 *)(v100 + 24LL * v102);
              v103 = *v63;
              if ( a1 == *v63 )
                goto LABEL_109;
              ++v101;
            }
LABEL_133:
            if ( v91 )
              v63 = v91;
          }
          goto LABEL_109;
        }
LABEL_197:
        ++*(_DWORD *)(v61 + 16);
        BUG();
      }
      if ( v55 == -4096 )
        break;
      if ( v55 == -8192 && *((_DWORD *)v47 + 2) == 101 && !v52 )
        v52 = (__int64 *)(v116 + 48LL * j);
LABEL_62:
      v56 = v51 + j;
      ++v51;
    }
    if ( *((_DWORD *)v47 + 2) != 100 )
      goto LABEL_62;
    if ( v52 )
      v47 = v52;
    ++*(_QWORD *)a2;
    v84 = *(_DWORD *)(a2 + 16) + 1;
    if ( 4 * v84 >= 3 * v114 )
      goto LABEL_46;
    if ( v114 - *(_DWORD *)(a2 + 20) - v84 <= v114 >> 3 )
    {
      sub_CF9EC0(a2, v114);
      v104 = *(_DWORD *)(a2 + 24);
      if ( v104 )
      {
        v105 = *(_QWORD *)(a2 + 8);
        v127[0] = v8;
        v115 = v105;
        v106 = sub_CF97C0(v127);
        v43 = 0;
        v107 = 1;
        v108 = v104 - 1;
        v123 = (unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32;
        for ( k = (v104 - 1) & (((0xBF58476D1CE4E5B9LL * (v123 | v106)) >> 31) ^ (484763065 * (v123 | v106)));
              ;
              k = v108 & v111 )
        {
          v47 = (__int64 *)(v115 + 48LL * k);
          v110 = *v47;
          if ( v9 == *v47 && v8 == *((_DWORD *)v47 + 2) )
            break;
          if ( v110 == -4096 )
          {
            if ( *((_DWORD *)v47 + 2) == 100 )
            {
LABEL_181:
              if ( v43 )
                v47 = v43;
              v84 = *(_DWORD *)(a2 + 16) + 1;
              goto LABEL_125;
            }
          }
          else if ( v110 == -8192 && *((_DWORD *)v47 + 2) == 101 && !v43 )
          {
            v43 = (__int64 *)(v115 + 48LL * k);
          }
          v111 = v107 + k;
          ++v107;
        }
        goto LABEL_175;
      }
LABEL_200:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
LABEL_125:
    *(_DWORD *)(a2 + 16) = v84;
    if ( *v47 != -4096 || *((_DWORD *)v47 + 2) != 100 )
      --*(_DWORD *)(a2 + 20);
    *v47 = v9;
    v47[2] = 0;
    v47[3] = 0;
    v47[4] = 0;
    *((_DWORD *)v47 + 10) = 0;
    *((_DWORD *)v47 + 2) = v8;
    v61 = (__int64)(v47 + 2);
LABEL_128:
    ++*(_QWORD *)v61;
    v59 = 0;
LABEL_129:
    sub_CFA200(v61, 2 * v59);
    v85 = *(_DWORD *)(v61 + 24);
    if ( !v85 )
      goto LABEL_197;
    v86 = v85 - 1;
    v87 = *(_QWORD *)(v61 + 8);
    v88 = v86 & v124;
    v63 = (__int64 *)(v87 + 24LL * (v86 & v124));
    v89 = *v63;
    v77 = *(_DWORD *)(v61 + 16) + 1;
    if ( a1 != *v63 )
    {
      v90 = 1;
      v91 = 0;
      while ( v89 != -4096 )
      {
        if ( v89 == -8192 && !v91 )
          v91 = v63;
        v88 = v86 & (v90 + v88);
        v63 = (__int64 *)(v87 + 24LL * v88);
        v89 = *v63;
        if ( a1 == *v63 )
          goto LABEL_109;
        ++v90;
      }
      goto LABEL_133;
    }
LABEL_109:
    *(_DWORD *)(v61 + 16) = v77;
    if ( *v63 != -4096 )
      --*(_DWORD *)(v61 + 20);
    *v63 = a1;
    v67 = v63 + 1;
    v63[1] = 0;
    v63[2] = 0;
LABEL_81:
    *v67 = 0;
    v5 += 16;
    v67[1] = 0;
    if ( v125 == v5 )
      return;
  }
  v11 = *(_QWORD *)(a1 + 32 * (v7 + 1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v11 != 17 )
    goto LABEL_33;
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  v13 = *(_DWORD *)(a2 + 24);
  if ( !v13 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_16;
  }
  v113 = *(_DWORD *)(a2 + 24);
  v119 = *(_QWORD *)(a2 + 8);
  v127[0] = v8;
  v32 = sub_CF97C0(v127);
  v13 = v113;
  v33 = 1;
  v20 = 0;
  v34 = 0xBF58476D1CE4E5B9LL * (((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32) | v32);
  v35 = (v113 - 1) & ((v34 >> 31) ^ v34);
  while ( 2 )
  {
    v36 = (__int64 *)(v119 + 48LL * v35);
    v37 = *v36;
    if ( v9 == *v36 && v8 == *((_DWORD *)v36 + 2) )
    {
      v23 = *((_DWORD *)v36 + 10);
      v24 = v36[3];
      v25 = (__int64)(v36 + 2);
      if ( !v23 )
        goto LABEL_91;
      v26 = 1;
      v27 = 0;
      v28 = (v23 - 1) & v124;
      v29 = (__int64 *)(v24 + 24LL * v28);
      v30 = *v29;
      if ( a1 == *v29 )
      {
LABEL_28:
        v31 = v12;
        if ( v29[1] <= (unsigned __int64)v12 )
          v31 = (_QWORD *)v29[1];
        if ( v29[2] >= (unsigned __int64)v12 )
          v12 = (_QWORD *)v29[2];
        v29[1] = (__int64)v31;
        v29[2] = (__int64)v12;
LABEL_33:
        v5 += 16;
        if ( v125 == v5 )
          return;
        goto LABEL_6;
      }
      while ( v30 != -4096 )
      {
        if ( !v27 && v30 == -8192 )
          v27 = v29;
        v28 = (v23 - 1) & (v26 + v28);
        v29 = (__int64 *)(v24 + 24LL * v28);
        v30 = *v29;
        if ( a1 == *v29 )
          goto LABEL_28;
        ++v26;
      }
      if ( !v27 )
        v27 = v29;
      v57 = *((_DWORD *)v36 + 8);
      ++v36[2];
      v58 = v57 + 1;
      if ( 4 * v58 >= 3 * v23 )
        goto LABEL_92;
      if ( v23 - *((_DWORD *)v36 + 9) - v58 > v23 >> 3 )
        goto LABEL_73;
      v121 = v36;
      sub_CFA200((__int64)(v36 + 2), v23);
      v78 = *((_DWORD *)v121 + 10);
      if ( v78 )
      {
        v79 = v78 - 1;
        v80 = v121[3];
        v81 = 1;
        v75 = 0;
        LODWORD(v82) = v79 & v124;
        v27 = (__int64 *)(v80 + 24LL * (v79 & v124));
        v83 = *v27;
        v58 = *((_DWORD *)v121 + 8) + 1;
        if ( *v27 != a1 )
        {
          while ( v83 != -4096 )
          {
            if ( v83 == -8192 && !v75 )
              v75 = v27;
            v82 = v79 & (unsigned int)(v82 + v81);
            v27 = (__int64 *)(v80 + 24 * v82);
            v83 = *v27;
            if ( a1 == *v27 )
              goto LABEL_73;
            ++v81;
          }
LABEL_96:
          if ( v75 )
            v27 = v75;
        }
        goto LABEL_73;
      }
LABEL_198:
      ++*(_DWORD *)(v25 + 16);
      BUG();
    }
    if ( v37 != -4096 )
    {
      if ( v37 == -8192 && *((_DWORD *)v36 + 2) == 101 && !v20 )
        v20 = (__int64 *)(v119 + 48LL * v35);
      goto LABEL_43;
    }
    if ( *((_DWORD *)v36 + 2) != 100 )
    {
LABEL_43:
      v38 = v33 + v35;
      ++v33;
      v35 = (v113 - 1) & v38;
      continue;
    }
    break;
  }
  if ( !v20 )
    v20 = (__int64 *)(v119 + 48LL * v35);
  ++*(_QWORD *)a2;
  v68 = *(_DWORD *)(a2 + 16) + 1;
  if ( 4 * v68 >= 3 * v113 )
  {
LABEL_16:
    sub_CF9EC0(a2, 2 * v13);
    v118 = *(_DWORD *)(a2 + 24);
    if ( v118 )
    {
      v14 = *(_QWORD *)(a2 + 8);
      v127[0] = v8;
      v15 = sub_CF97C0(v127);
      v16 = 0;
      v17 = 1;
      v18 = 0xBF58476D1CE4E5B9LL * (v15 | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32));
      for ( m = (v118 - 1) & ((v18 >> 31) ^ v18); ; m = (v118 - 1) & v22 )
      {
        v20 = (__int64 *)(v14 + 48LL * m);
        v21 = *v20;
        if ( v9 == *v20 && v8 == *((_DWORD *)v20 + 2) )
          break;
        if ( v21 == -4096 )
        {
          if ( *((_DWORD *)v20 + 2) == 100 )
            goto LABEL_137;
        }
        else if ( v21 == -8192 && *((_DWORD *)v20 + 2) == 101 && !v16 )
        {
          v16 = (__int64 *)(v14 + 48LL * m);
        }
        v22 = v17 + m;
        ++v17;
      }
LABEL_171:
      v68 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_88;
    }
LABEL_199:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( v113 - *(_DWORD *)(a2 + 20) - v68 <= v113 >> 3 )
  {
    sub_CF9EC0(a2, v113);
    v122 = *(_DWORD *)(a2 + 24);
    if ( v122 )
    {
      v92 = *(_QWORD *)(a2 + 8);
      v127[0] = v8;
      v112 = v92;
      v93 = sub_CF97C0(v127);
      v16 = 0;
      v94 = 1;
      v117 = (unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32;
      for ( n = (v122 - 1) & (((0xBF58476D1CE4E5B9LL * (v93 | v117)) >> 31) ^ (484763065 * (v93 | v117)));
            ;
            n = (v122 - 1) & v97 )
      {
        v20 = (__int64 *)(v112 + 48LL * n);
        v96 = *v20;
        if ( v9 == *v20 && v8 == *((_DWORD *)v20 + 2) )
          break;
        if ( v96 == -4096 )
        {
          if ( *((_DWORD *)v20 + 2) == 100 )
          {
LABEL_137:
            if ( v16 )
              v20 = v16;
            v68 = *(_DWORD *)(a2 + 16) + 1;
            goto LABEL_88;
          }
        }
        else if ( v96 == -8192 && *((_DWORD *)v20 + 2) == 101 && !v16 )
        {
          v16 = (__int64 *)(v112 + 48LL * n);
        }
        v97 = v94 + n;
        ++v94;
      }
      goto LABEL_171;
    }
    goto LABEL_199;
  }
LABEL_88:
  *(_DWORD *)(a2 + 16) = v68;
  if ( *v20 != -4096 || *((_DWORD *)v20 + 2) != 100 )
    --*(_DWORD *)(a2 + 20);
  *v20 = v9;
  v20[2] = 0;
  v20[3] = 0;
  v20[4] = 0;
  *((_DWORD *)v20 + 10) = 0;
  *((_DWORD *)v20 + 2) = v8;
  v25 = (__int64)(v20 + 2);
LABEL_91:
  ++*(_QWORD *)v25;
  v23 = 0;
LABEL_92:
  sub_CFA200(v25, 2 * v23);
  v69 = *(_DWORD *)(v25 + 24);
  if ( !v69 )
    goto LABEL_198;
  v70 = v69 - 1;
  v71 = *(_QWORD *)(v25 + 8);
  LODWORD(v72) = v70 & v124;
  v27 = (__int64 *)(v71 + 24LL * (v70 & v124));
  v73 = *v27;
  v58 = *(_DWORD *)(v25 + 16) + 1;
  if ( a1 != *v27 )
  {
    v74 = 1;
    v75 = 0;
    while ( v73 != -4096 )
    {
      if ( !v75 && v73 == -8192 )
        v75 = v27;
      v72 = v70 & (unsigned int)(v72 + v74);
      v27 = (__int64 *)(v71 + 24 * v72);
      v73 = *v27;
      if ( a1 == *v27 )
        goto LABEL_73;
      ++v74;
    }
    goto LABEL_96;
  }
LABEL_73:
  *(_DWORD *)(v25 + 16) = v58;
  if ( *v27 != -4096 )
    --*(_DWORD *)(v25 + 20);
  *v27 = a1;
  v5 += 16;
  v27[1] = (__int64)v12;
  v27[2] = (__int64)v12;
  if ( v125 != v5 )
    goto LABEL_6;
}
