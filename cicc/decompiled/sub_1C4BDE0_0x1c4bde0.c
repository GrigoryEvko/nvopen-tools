// Function: sub_1C4BDE0
// Address: 0x1c4bde0
//
__int64 __fastcall sub_1C4BDE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  __int64 v4; // r15
  __int64 **v5; // rdx
  __int64 v6; // r15
  unsigned __int64 v7; // r13
  __int64 *v8; // r14
  char v9; // al
  __int64 v10; // rax
  char v11; // cl
  int v12; // eax
  __int64 *v13; // r13
  char v14; // r15
  __int64 v15; // rdx
  __int64 result; // rax
  __int64 *v17; // r10
  __int64 *v18; // r9
  unsigned int i; // r8d
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // r11
  __int64 v23; // rdx
  __int64 v24; // rax
  char v25; // al
  int v26; // edx
  unsigned int v27; // edi
  int v28; // r14d
  _QWORD *v29; // r12
  __int64 v30; // rbx
  __int64 *v31; // rdx
  __int64 v32; // r8
  unsigned __int8 v33; // al
  bool v34; // dl
  char v35; // si
  int v36; // eax
  _QWORD *v37; // r8
  bool v38; // dl
  int v39; // r9d
  int v40; // eax
  __int64 v41; // r15
  bool v42; // r13
  unsigned int v43; // ebx
  _QWORD *v44; // rdx
  char v45; // r14
  _QWORD *v46; // r12
  int v47; // ecx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int8 v51; // al
  bool v52; // al
  char v53; // r10
  bool v54; // dl
  char v55; // al
  int v56; // r15d
  __int64 *v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __int16 v60; // dx
  __int64 v61; // rax
  __int64 v62; // rdx
  char v63; // r8
  unsigned int v64; // edx
  unsigned int v65; // ecx
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rdx
  bool v70; // si
  char v71; // di
  __int64 v72; // rcx
  __int64 v73; // rax
  bool v74; // al
  __int64 v75; // rdx
  __int64 v76; // r15
  _QWORD *v77; // rax
  _QWORD *v78; // r13
  unsigned int v79; // esi
  int v80; // edi
  unsigned int v81; // r14d
  _QWORD *v82; // rdx
  unsigned int k; // ecx
  _QWORD *v84; // rax
  _QWORD *v85; // r8
  int v86; // ecx
  unsigned int n; // esi
  __int64 v88; // r8
  unsigned int v89; // ecx
  __int64 v90; // rdx
  __int64 v91; // rax
  _QWORD *v92; // r11
  __int64 v93; // rax
  __int64 v94; // rax
  int v95; // eax
  unsigned int j; // eax
  __int64 v97; // rcx
  int v98; // esi
  int v99; // esi
  int v100; // edi
  unsigned int v101; // edx
  _QWORD *v102; // rcx
  __int64 v103; // r8
  unsigned int v104; // edx
  int v105; // esi
  int v106; // esi
  int v107; // edi
  unsigned int m; // edx
  __int64 v109; // r8
  unsigned int v110; // edx
  unsigned int v111; // ecx
  _QWORD *v112; // r8
  _QWORD *v113; // r15
  unsigned int v114; // ebx
  __int64 v115; // r14
  _QWORD *v116; // rax
  unsigned int v117; // r15d
  _QWORD *v118; // rbx
  _QWORD *v119; // r8
  _QWORD *v120; // rax
  _QWORD *v121; // r13
  unsigned int v122; // esi
  _QWORD *v123; // r10
  int v124; // ecx
  unsigned int ii; // edx
  _QWORD *v126; // rax
  _QWORD *v127; // rdi
  int v128; // ecx
  _QWORD *v129; // r15
  int v130; // ecx
  int v131; // ecx
  int v132; // esi
  unsigned int jj; // edx
  __int64 v134; // rdi
  int v135; // ecx
  int v136; // ecx
  int v137; // esi
  unsigned int kk; // edx
  __int64 v139; // rdi
  unsigned int v140; // edx
  unsigned int v141; // edx
  _QWORD *v142; // rax
  unsigned int v143; // edx
  unsigned int v144; // [rsp+4h] [rbp-9Ch]
  int v145; // [rsp+8h] [rbp-98h]
  unsigned int v146; // [rsp+8h] [rbp-98h]
  __int64 *v147; // [rsp+8h] [rbp-98h]
  _QWORD *v148; // [rsp+10h] [rbp-90h]
  __int64 *v149; // [rsp+10h] [rbp-90h]
  __int64 *v150; // [rsp+10h] [rbp-90h]
  int v151; // [rsp+18h] [rbp-88h]
  __int64 *v152; // [rsp+18h] [rbp-88h]
  __int64 v153; // [rsp+18h] [rbp-88h]
  bool v154; // [rsp+20h] [rbp-80h]
  int v155; // [rsp+20h] [rbp-80h]
  __int64 v156; // [rsp+20h] [rbp-80h]
  __int64 v157; // [rsp+28h] [rbp-78h]
  __int64 *v158; // [rsp+28h] [rbp-78h]
  int v159; // [rsp+38h] [rbp-68h]
  int v161; // [rsp+38h] [rbp-68h]
  __int64 v162; // [rsp+40h] [rbp-60h]
  __int64 *v163; // [rsp+40h] [rbp-60h]
  int v164; // [rsp+48h] [rbp-58h]
  unsigned __int8 v165; // [rsp+48h] [rbp-58h]
  _QWORD *v166; // [rsp+48h] [rbp-58h]
  unsigned int v167; // [rsp+48h] [rbp-58h]
  _QWORD *v168; // [rsp+50h] [rbp-50h] BYREF
  __int64 v169; // [rsp+58h] [rbp-48h]
  _QWORD v170[8]; // [rsp+60h] [rbp-40h] BYREF

  v2 = a2;
  v3 = (_QWORD *)a1;
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(__int64 ***)a1;
  if ( (_DWORD)v4 )
  {
    v6 = 8 * v4;
    v7 = 0;
    while ( 1 )
    {
      v8 = v5[v7 / 8];
      v9 = *((_BYTE *)v8 + 16);
      switch ( v9 )
      {
        case 'M':
          return 0;
        case 'N':
          v10 = *(v8 - 3);
          v11 = *(_BYTE *)(v10 + 16);
          if ( v11 )
          {
            if ( v11 == 20 && *(_BYTE *)(v10 + 96) )
              return 0;
          }
          else if ( (*(_BYTE *)(v10 + 33) & 0x20) != 0 )
          {
            v12 = *(_DWORD *)(v10 + 36);
            if ( v12 == 4175 || v12 == 4178 )
              return 0;
          }
          v7 += 8LL;
          if ( v6 == v7 )
            goto LABEL_10;
          break;
        case '6':
        case '7':
          if ( sub_15F32D0((__int64)v5[v7 / 8]) || (*((_BYTE *)v8 + 18) & 1) != 0 )
            return 0;
          v7 += 8LL;
          v5 = *(__int64 ***)a1;
          if ( v6 == v7 )
            goto LABEL_10;
          break;
        case 'V':
        case ';':
          return 0;
        default:
          v7 += 8LL;
          if ( v6 == v7 )
            goto LABEL_10;
          break;
      }
    }
  }
LABEL_10:
  v13 = *v5;
  v14 = *((_BYTE *)*v5 + 16);
  v164 = *((_DWORD *)*v5 + 5) & 0xFFFFFFF;
  v162 = **v5;
  if ( v14 == 78 )
  {
    v15 = *(v13 - 3);
    if ( *(_BYTE *)(v15 + 16) )
      return 0;
    if ( (unsigned int)dword_4FBBEA0 <= 1 )
      __asm { jmp     rax }
    v17 = v13;
  }
  else
  {
    v17 = 0;
  }
  v168 = v170;
  v169 = 0x200000000LL;
  if ( v14 == 54 || (v18 = 0, v14 == 55) )
  {
    v18 = (__int64 *)*(v13 - 3);
    if ( v18 )
    {
      v170[0] = *(v13 - 3);
      LODWORD(v169) = 1;
    }
  }
  v159 = *(_DWORD *)(a1 + 8);
  if ( v159 != 1 )
  {
    for ( i = 1; v159 != i; ++i )
    {
      v20 = i;
      v21 = *(_QWORD *)(*v3 + 8LL * i);
      if ( v164 != (*(_DWORD *)(v21 + 20) & 0xFFFFFFF) || v162 != *(_QWORD *)v21 || v14 != *(_BYTE *)(v21 + 16) )
        goto LABEL_81;
      if ( v14 == 54 || v14 == 55 )
      {
        v22 = *(_QWORD *)(v21 - 24);
        v23 = (unsigned int)v169;
        if ( (unsigned int)v169 >= HIDWORD(v169) )
        {
          v144 = i;
          v147 = v17;
          v150 = v18;
          v153 = i;
          v156 = *(_QWORD *)(v21 - 24);
          sub_16CD150((__int64)&v168, v170, 0, 8, i, (int)v18);
          v23 = (unsigned int)v169;
          v22 = v156;
          i = v144;
          v17 = v147;
          v18 = v150;
          v20 = v153;
        }
        v168[v23] = v22;
        LODWORD(v169) = v169 + 1;
      }
      if ( v17 )
      {
        v24 = *(_QWORD *)(*v3 + 8 * v20);
        if ( *(_BYTE *)(v24 + 16) != 78 || *(v17 - 3) != *(_QWORD *)(v24 - 24) )
          goto LABEL_81;
      }
      v25 = *((_BYTE *)v13 + 16);
      if ( (unsigned __int8)(v25 - 75) <= 1u )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v21 + 16) - 75) > 1u )
          goto LABEL_81;
        v26 = *((unsigned __int16 *)v13 + 9);
        BYTE1(v26) &= ~0x80u;
        v27 = *(_WORD *)(v21 + 18) & 0x7FFF;
        if ( v26 == v27 )
          continue;
        v146 = i;
        v149 = v17;
        v152 = v18;
        if ( v26 != (unsigned int)sub_15FF5D0(v27) )
          goto LABEL_81;
        sub_15FF6F0(v21);
        v60 = *((_WORD *)v13 + 9);
        i = v146;
        v17 = v149;
        HIBYTE(v60) &= ~0x80u;
        v18 = v152;
        *(_WORD *)(v21 + 18) = v60 | *(_WORD *)(v21 + 18) & 0x8000;
        v25 = *((_BYTE *)v13 + 16);
      }
      if ( v25 == 83 )
      {
        if ( *(_BYTE *)(v21 + 16) != 83 )
          goto LABEL_81;
        v61 = *(v13 - 3);
        if ( *(_BYTE *)(v61 + 16) <= 0x10u )
        {
          v62 = *(_QWORD *)(v21 - 24);
          if ( *(_BYTE *)(v62 + 16) <= 0x10u && v61 != v62 )
            goto LABEL_81;
        }
      }
      else if ( v25 == 56 && (*(_BYTE *)(v21 + 16) != 56 || v13[7] != *(_QWORD *)(v21 + 56)) )
      {
        goto LABEL_81;
      }
    }
  }
  if ( !v164 )
    goto LABEL_106;
  v163 = v18;
  v28 = 0;
  v29 = v3;
  v30 = 0;
  do
  {
    if ( (*((_BYTE *)v13 + 23) & 0x40) != 0 )
      v31 = (__int64 *)*(v13 - 1);
    else
      v31 = &v13[-3 * (*((_DWORD *)v13 + 5) & 0xFFFFFFF)];
    v32 = *(__int64 *)((char *)v31 + v30);
    v33 = *(_BYTE *)(v32 + 16);
    if ( v33 <= 0x10u )
    {
      v35 = 1;
      v34 = 0;
    }
    else
    {
      v34 = v33 == 53;
      v35 = 0;
    }
    v154 = v34;
    v157 = v32;
    v36 = sub_1648EF0(v32);
    v37 = (_QWORD *)v157;
    v38 = v154;
    v39 = v36;
    if ( *((_DWORD *)v29 + 2) == 1 )
      goto LABEL_104;
    v40 = *((_DWORD *)v29 + 2);
    v158 = v13;
    v41 = v30;
    v155 = v28;
    v42 = v38;
    v43 = 1;
    v44 = v29;
    v45 = 1;
    v46 = v37;
    v47 = v40;
    do
    {
      v48 = *(_QWORD *)(*v44 + 8LL * v43);
      if ( (*(_BYTE *)(v48 + 23) & 0x40) != 0 )
        v49 = *(_QWORD *)(v48 - 8);
      else
        v49 = v48 - 24LL * (*(_DWORD *)(v48 + 20) & 0xFFFFFFF);
      v50 = *(_QWORD *)(v49 + v41);
      if ( v46 != (_QWORD *)v50 )
      {
        if ( *(_QWORD *)v50 != *v46 )
          goto LABEL_81;
        v45 = 0;
      }
      v51 = *(_BYTE *)(v50 + 16);
      if ( v51 <= 0x10u )
      {
        if ( v42 )
          goto LABEL_81;
      }
      else
      {
        if ( v42 )
        {
          if ( v51 != 53 )
            goto LABEL_81;
          v145 = v47;
          v148 = v44;
          v151 = v39;
          v52 = sub_1648CD0(v50, v39);
          v39 = v151;
          v44 = v148;
          v47 = v145;
          if ( !v52 )
            goto LABEL_81;
        }
        else if ( v51 == 53 )
        {
          goto LABEL_81;
        }
        v35 = 0;
      }
      ++v43;
    }
    while ( v47 != v43 );
    v53 = v45;
    v29 = v44;
    v28 = v155;
    v54 = v42;
    v30 = v41;
    v13 = v158;
    if ( v53 )
      goto LABEL_104;
    v55 = *((_BYTE *)v158 + 16);
    if ( v155 != 2 )
    {
      if ( v55 != 56 )
        goto LABEL_70;
      if ( !v155 )
      {
        if ( !v54 )
          goto LABEL_81;
LABEL_117:
        if ( !v155 )
          goto LABEL_70;
      }
      if ( v35 )
        goto LABEL_81;
      goto LABEL_70;
    }
    if ( v55 == 85 )
      goto LABEL_81;
    if ( v55 == 56 )
      goto LABEL_117;
LABEL_70:
    v56 = *((_DWORD *)v29 + 2);
    if ( v56 )
    {
      v57 = (__int64 *)*v29;
      while ( 1 )
      {
        v59 = *v57;
        v58 = (*(_BYTE *)(*v57 + 23) & 0x40) != 0
            ? *(_QWORD *)(v59 - 8)
            : v59 - 24LL * (*(_DWORD *)(v59 + 20) & 0xFFFFFFF);
        if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v58 + v30) + 16LL) - 18) <= 5u )
          break;
        if ( (__int64 *)(*v29 + 8LL * (unsigned int)(v56 - 1) + 8) == ++v57 )
          goto LABEL_104;
      }
LABEL_81:
      result = 0;
      goto LABEL_82;
    }
LABEL_104:
    ++v28;
    v30 += 24;
  }
  while ( v164 != v28 );
  v3 = v29;
  v18 = v163;
  v2 = a2;
LABEL_106:
  v63 = *((_BYTE *)v13 + 16);
  if ( v63 == 87 )
  {
    v64 = 1;
    v65 = *((_DWORD *)v13 + 16);
    while ( 1 )
    {
      if ( *((_DWORD *)v3 + 2) == v64 )
        goto LABEL_124;
      v66 = *(_QWORD *)(*v3 + 8LL * v64);
      if ( v65 != *(_DWORD *)(v66 + 64) )
        goto LABEL_81;
      v67 = *(_QWORD *)(v66 + 56);
      if ( v65 )
        break;
LABEL_123:
      ++v64;
    }
    v68 = 0;
    while ( *(_DWORD *)(v67 + 4 * v68) == *(_DWORD *)(v13[7] + 4 * v68) )
    {
      if ( v65 <= (unsigned int)++v68 )
        goto LABEL_123;
    }
    goto LABEL_81;
  }
LABEL_124:
  if ( !v18 )
    goto LABEL_136;
  v69 = 0;
  v70 = 0;
  v71 = 1;
  while ( (_DWORD)v169 != (_DWORD)v69 )
  {
    v72 = v168[v69];
    if ( (__int64 *)v72 != v18 )
      v71 = 0;
    if ( *(_BYTE *)(v72 + 16) == 56 )
    {
      v73 = **(_QWORD **)(v72 - 24LL * (*(_DWORD *)(v72 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v73 + 8) == 16 )
        v73 = **(_QWORD **)(v73 + 16);
      v74 = *(_DWORD *)(v73 + 8) >> 8 == 5 || *(_DWORD *)(v73 + 8) >> 8 == 0;
      if ( v74 )
        v70 = v74;
    }
    ++v69;
  }
  if ( v71 )
    goto LABEL_136;
  if ( v70 )
    goto LABEL_81;
  if ( v63 != 55 )
    goto LABEL_136;
  v94 = *v18;
  if ( *(_BYTE *)(*v18 + 8) != 15 )
    BUG();
  if ( *(_BYTE *)(*(_QWORD *)(v94 + 24) + 8LL) == 13 && ((v95 = *(_DWORD *)(v94 + 8) >> 8, v95 == 5) || !v95) )
  {
    v75 = *v3;
    v161 = *((_DWORD *)v3 + 2);
    for ( j = 1; j != v161; ++j )
    {
      v97 = *(_QWORD *)(v75 + 8LL * j);
      if ( *(_BYTE *)(v97 + 16) != 55 )
        BUG();
      if ( *(v13 - 6) != *(_QWORD *)(v97 - 48) )
        goto LABEL_81;
    }
  }
  else
  {
LABEL_136:
    v75 = *v3;
    v161 = *((_DWORD *)v3 + 2);
  }
  v76 = *(_QWORD *)(*(_QWORD *)v75 + 8LL);
  while ( 2 )
  {
    if ( v76 )
    {
      v77 = sub_1648700(v76);
      v78 = v77;
      if ( *((_BYTE *)v77 + 16) == 77 )
      {
        v79 = *(_DWORD *)(v2 + 24);
        if ( v79 )
        {
          v80 = 1;
          v81 = ((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4);
          v82 = 0;
          for ( k = (v79 - 1) & v81; ; k = (v79 - 1) & v111 )
          {
            v84 = (_QWORD *)(*(_QWORD *)(v2 + 8) + 8LL * k);
            v85 = (_QWORD *)*v84;
            if ( v78 == (_QWORD *)*v84 )
              goto LABEL_138;
            if ( v85 == (_QWORD *)-8LL )
              break;
            if ( v85 != (_QWORD *)-16LL || v82 )
              v84 = v82;
            v111 = v80 + k;
            v82 = v84;
            ++v80;
          }
          v86 = *(_DWORD *)(v2 + 16);
          if ( v82 )
            v84 = v82;
          ++*(_QWORD *)v2;
          if ( 4 * (v86 + 1) < 3 * v79 )
          {
            if ( v79 - *(_DWORD *)(v2 + 20) - (v86 + 1) <= v79 >> 3 )
            {
              sub_176FD40(v2, v79);
              v98 = *(_DWORD *)(v2 + 24);
              if ( !v98 )
                goto LABEL_244;
              v99 = v98 - 1;
              v100 = 1;
              v101 = v99 & v81;
              v102 = 0;
              while ( 1 )
              {
                v84 = (_QWORD *)(*(_QWORD *)(v2 + 8) + 8LL * v101);
                v103 = *v84;
                if ( v78 == (_QWORD *)*v84 )
                  break;
                if ( v103 == -8 )
                  goto LABEL_179;
                if ( v103 != -16 || v102 )
                  v84 = v102;
                v104 = v100 + v101;
                v102 = v84;
                ++v100;
                v101 = v99 & v104;
              }
            }
            goto LABEL_149;
          }
        }
        else
        {
          ++*(_QWORD *)v2;
        }
        sub_176FD40(v2, 2 * v79);
        v105 = *(_DWORD *)(v2 + 24);
        if ( !v105 )
        {
LABEL_244:
          ++*(_DWORD *)(v2 + 16);
          BUG();
        }
        v106 = v105 - 1;
        v107 = 1;
        v102 = 0;
        for ( m = v106 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4)); ; m = v106 & v110 )
        {
          v84 = (_QWORD *)(*(_QWORD *)(v2 + 8) + 8LL * m);
          v109 = *v84;
          if ( v78 == (_QWORD *)*v84 )
            break;
          if ( v109 == -8 )
          {
LABEL_179:
            if ( v102 )
              v84 = v102;
            break;
          }
          if ( v109 != -16 || v102 )
            v84 = v102;
          v110 = v107 + m;
          v102 = v84;
          ++v107;
        }
LABEL_149:
        ++*(_DWORD *)(v2 + 16);
        if ( *v84 != -8 )
          --*(_DWORD *)(v2 + 20);
        *v84 = v78;
        for ( n = 1; v161 != n; ++n )
        {
          v88 = *(_QWORD *)(*v3 + 8LL * n);
          v89 = *((_DWORD *)v78 + 5) & 0xFFFFFFF;
          if ( !v89 )
            goto LABEL_81;
          v90 = 0;
          v91 = 24LL * *((unsigned int *)v78 + 14) + 8;
          while ( 1 )
          {
            v92 = &v78[-3 * v89];
            if ( (*((_BYTE *)v78 + 23) & 0x40) != 0 )
              v92 = (_QWORD *)*(v78 - 1);
            if ( *(_QWORD *)(v88 + 40) == *(_QWORD *)((char *)v92 + v91) )
              break;
            ++v90;
            v91 += 8;
            if ( v89 == (_DWORD)v90 )
              goto LABEL_81;
          }
          v93 = v92[3 * v90];
          if ( !v93 || v88 != v93 )
            goto LABEL_81;
        }
      }
LABEL_138:
      v76 = *(_QWORD *)(v76 + 8);
      continue;
    }
    break;
  }
  v112 = 0;
  v113 = v3;
  v114 = 1;
LABEL_200:
  if ( v161 == v114 )
  {
    result = 1;
  }
  else
  {
    v115 = *(_QWORD *)(*(_QWORD *)(*v113 + 8LL * v114) + 8LL);
    v116 = v113;
    v117 = v114;
    v118 = v112;
    v119 = v116;
    while ( 1 )
    {
      if ( !v115 )
      {
        v142 = v119;
        v112 = v118;
        v114 = v117 + 1;
        v113 = v142;
        goto LABEL_200;
      }
      v166 = v119;
      v120 = sub_1648700(v115);
      v119 = v166;
      v121 = v120;
      if ( *((_BYTE *)v120 + 16) == 77 )
        break;
LABEL_202:
      v115 = *(_QWORD *)(v115 + 8);
    }
    v122 = *(_DWORD *)(v2 + 24);
    if ( v122 )
    {
      v123 = 0;
      v124 = 1;
      v167 = ((unsigned int)v120 >> 9) ^ ((unsigned int)v120 >> 4);
      for ( ii = (v122 - 1) & v167; ; ii = (v122 - 1) & v143 )
      {
        v126 = (_QWORD *)(*(_QWORD *)(v2 + 8) + 8LL * ii);
        v127 = (_QWORD *)*v126;
        if ( v121 == (_QWORD *)*v126 )
          goto LABEL_202;
        if ( v127 == (_QWORD *)-8LL )
          break;
        if ( v127 != (_QWORD *)-16LL || v123 )
          v126 = v123;
        v143 = v124 + ii;
        v123 = v126;
        ++v124;
      }
      v128 = *(_DWORD *)(v2 + 16);
      v129 = v118;
      if ( v123 )
        v126 = v123;
      ++*(_QWORD *)v2;
      if ( 4 * (v128 + 1) < 3 * v122 )
      {
        if ( v122 - (*(_DWORD *)(v2 + 20) + v128 + 1) > v122 >> 3 )
          goto LABEL_213;
        sub_176FD40(v2, v122);
        v130 = *(_DWORD *)(v2 + 24);
        if ( v130 )
        {
          v131 = v130 - 1;
          v132 = 1;
          for ( jj = v131 & v167; ; jj = v131 & v141 )
          {
            v126 = (_QWORD *)(*(_QWORD *)(v2 + 8) + 8LL * jj);
            v134 = *v126;
            if ( v121 == (_QWORD *)*v126 )
              break;
            if ( v134 == -8 )
              goto LABEL_221;
            if ( v129 || v134 != -16 )
              v126 = v129;
            v141 = v132 + jj;
            v129 = v126;
            ++v132;
          }
          goto LABEL_213;
        }
LABEL_242:
        ++*(_DWORD *)(v2 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v2;
      v129 = v118;
    }
    sub_176FD40(v2, 2 * v122);
    v135 = *(_DWORD *)(v2 + 24);
    if ( !v135 )
      goto LABEL_242;
    v136 = v135 - 1;
    v137 = 1;
    for ( kk = v136 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4)); ; kk = v136 & v140 )
    {
      v126 = (_QWORD *)(*(_QWORD *)(v2 + 8) + 8LL * kk);
      v139 = *v126;
      if ( v121 == (_QWORD *)*v126 )
        break;
      if ( v139 == -8 )
      {
LABEL_221:
        if ( v129 )
          v126 = v129;
        break;
      }
      if ( v129 || v139 != -16 )
        v126 = v129;
      v140 = v137 + kk;
      v129 = v126;
      ++v137;
    }
LABEL_213:
    ++*(_DWORD *)(v2 + 16);
    if ( *v126 != -8 )
      --*(_DWORD *)(v2 + 20);
    *v126 = v121;
    result = 0;
  }
LABEL_82:
  if ( v168 != v170 )
  {
    v165 = result;
    _libc_free((unsigned __int64)v168);
    return v165;
  }
  return result;
}
