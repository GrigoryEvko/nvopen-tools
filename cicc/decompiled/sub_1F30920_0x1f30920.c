// Function: sub_1F30920
// Address: 0x1f30920
//
__int64 __fastcall sub_1F30920(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v6; // r15
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned int v9; // ebx
  __int64 v10; // r13
  void **v11; // r14
  void **i; // r12
  void *v13; // rax
  int v14; // r13d
  size_t v15; // rdx
  unsigned int v16; // ebx
  char *v17; // r13
  char *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // r8
  __int64 v25; // rax
  unsigned __int64 v26; // r9
  int v27; // eax
  int v28; // ecx
  unsigned int v29; // esi
  __int64 v30; // rax
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rcx
  int v35; // edx
  unsigned int v36; // eax
  unsigned int v37; // r10d
  unsigned int v38; // esi
  __int64 v39; // r11
  int v40; // ecx
  unsigned __int64 v41; // r9
  int v42; // ecx
  __int64 v43; // rdx
  int v46; // eax
  char v47; // al
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rdi
  unsigned int v51; // esi
  __int64 v52; // rax
  __int64 v53; // rsi
  unsigned int v54; // edx
  unsigned __int64 *v55; // r13
  __int64 v56; // rax
  float *v57; // rax
  __int64 v59; // rbx
  __int64 v60; // rax
  __int64 v61; // r12
  __int64 v62; // r13
  __int64 v63; // rcx
  int v64; // edx
  unsigned int v65; // eax
  unsigned int v66; // r11d
  unsigned int v67; // edi
  int v68; // ecx
  unsigned __int64 v69; // r8
  int v70; // ecx
  __int64 v71; // rdx
  __int64 v74; // rdx
  __int64 v75; // rcx
  _DWORD *v76; // rdi
  __int64 v77; // rdx
  __int64 v78; // rcx
  char *v79; // rax
  char *v80; // rbx
  __int64 v81; // r8
  __int64 v82; // rbx
  __int64 v83; // r12
  int v84; // esi
  __int64 v85; // rax
  __int64 v86; // rcx
  __int64 v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 *v92; // rdi
  __int64 v93; // rdx
  __int64 j; // r14
  __int64 v95; // rax
  __int64 v96; // r8
  _DWORD *v97; // r9
  __int64 v98; // rax
  __int64 v99; // r8
  __int64 v100; // rcx
  int v101; // esi
  int v102; // edi
  _QWORD *v103; // r13
  __int64 v104; // rbx
  __int64 v105; // rdi
  __int64 (*v106)(); // rax
  __int64 v107; // rax
  __int64 v108; // rdi
  __int64 v109; // r15
  __int64 (__fastcall *v110)(__int64, __int64, __int64, int *); // r8
  __int64 (*v111)(); // rax
  __int64 *v112; // r14
  __int64 *v113; // rbx
  __int64 v114; // rdi
  int v115; // eax
  __int64 v116; // r11
  __int64 v117; // r12
  int k; // eax
  int v119; // edx
  unsigned int v120; // eax
  __int64 *v121; // rcx
  int v122; // edx
  unsigned int v123; // edi
  unsigned int v124; // esi
  __int64 v125; // r8
  int v126; // ecx
  unsigned __int64 v127; // r10
  int v128; // ecx
  __int64 v129; // rdx
  unsigned __int64 v130; // r10
  unsigned __int64 *v133; // rbx
  unsigned __int64 *v134; // r12
  int v136; // r8d
  int v137; // r9d
  __int64 v138; // rax
  int v139; // r11d
  __int64 v140; // r12
  __int64 v141; // rdi
  __int64 (__fastcall *v142)(__int64, __int64, __int64, int *); // rax
  __int64 (*v143)(); // rax
  int v144; // eax
  int v145; // r11d
  bool v146; // zf
  int v147; // r8d
  int v148; // r9d
  __int64 v149; // rax
  __int64 v150; // rdi
  int v151; // r8d
  int v152; // r9d
  __int64 v153; // rax
  __int64 v154; // [rsp+10h] [rbp-390h]
  __int64 v156; // [rsp+20h] [rbp-380h]
  __int64 v157; // [rsp+20h] [rbp-380h]
  __int64 v158; // [rsp+28h] [rbp-378h]
  char *v159; // [rsp+38h] [rbp-368h]
  unsigned __int8 v160; // [rsp+53h] [rbp-34Dh]
  int v161; // [rsp+54h] [rbp-34Ch]
  int v162; // [rsp+54h] [rbp-34Ch]
  int v163; // [rsp+54h] [rbp-34Ch]
  __int64 v164; // [rsp+60h] [rbp-340h]
  __int64 v165; // [rsp+68h] [rbp-338h]
  char v166; // [rsp+68h] [rbp-338h]
  __int64 v167; // [rsp+70h] [rbp-330h]
  __int64 v168; // [rsp+70h] [rbp-330h]
  unsigned int v169; // [rsp+78h] [rbp-328h]
  __int64 v170; // [rsp+78h] [rbp-328h]
  int v171; // [rsp+80h] [rbp-320h] BYREF
  int v172; // [rsp+84h] [rbp-31Ch] BYREF
  int v173; // [rsp+88h] [rbp-318h] BYREF
  int v174; // [rsp+8Ch] [rbp-314h] BYREF
  void *src; // [rsp+90h] [rbp-310h] BYREF
  __int64 v176; // [rsp+98h] [rbp-308h]
  _BYTE v177[32]; // [rsp+A0h] [rbp-300h] BYREF
  void *v178; // [rsp+C0h] [rbp-2E0h] BYREF
  __int64 v179; // [rsp+C8h] [rbp-2D8h]
  _BYTE v180[64]; // [rsp+D0h] [rbp-2D0h] BYREF
  void *v181; // [rsp+110h] [rbp-290h] BYREF
  __int64 v182; // [rsp+118h] [rbp-288h]
  _BYTE v183[64]; // [rsp+120h] [rbp-280h] BYREF
  unsigned __int64 *v184; // [rsp+160h] [rbp-240h] BYREF
  __int64 v185; // [rsp+168h] [rbp-238h]
  _BYTE v186[560]; // [rsp+170h] [rbp-230h] BYREF

  v6 = a1;
  v7 = a1[30];
  v8 = *(_QWORD *)(v7 + 16) - *(_QWORD *)(v7 + 8);
  v179 = 0x1000000000LL;
  v9 = -858993459 * (v8 >> 3) - *(_DWORD *)(v7 + 32);
  v178 = v180;
  v10 = 4LL * v9;
  if ( v9 <= 0x10uLL )
  {
    LODWORD(v179) = v9;
    if ( v180 != &v180[v10] )
      memset(v180, 255, 4LL * v9);
    v182 = v9 | 0x1000000000LL;
    v181 = v183;
    if ( v183 != &v183[v10] )
      memset(v183, 0, 4LL * v9);
    HIDWORD(v185) = 16;
    src = v177;
    v176 = 0x400000000LL;
    v184 = (unsigned __int64 *)v186;
    v11 = (void **)v186;
    goto LABEL_7;
  }
  sub_16CD150((__int64)&v178, v180, v9, 4, a5, a6);
  LODWORD(v179) = v9;
  if ( v178 != (char *)v178 + v10 )
    memset(v178, 255, 4LL * v9);
  v181 = v183;
  v182 = 0x1000000000LL;
  sub_16CD150((__int64)&v181, v183, v9, 4, v151, v152);
  LODWORD(v182) = v9;
  if ( v181 == (char *)v181 + v10 )
  {
    src = v177;
    v176 = 0x400000000LL;
    v184 = (unsigned __int64 *)v186;
    v185 = 0x1000000000LL;
  }
  else
  {
    memset(v181, 0, 4LL * v9);
    src = v177;
    v176 = 0x400000000LL;
    v184 = (unsigned __int64 *)v186;
    v185 = 0x1000000000LL;
    if ( v9 <= 0x10uLL )
    {
      v11 = (void **)v186;
      goto LABEL_7;
    }
  }
  sub_1E47E20((__int64)&v184, v9);
  v11 = (void **)v184;
LABEL_7:
  LODWORD(v185) = v9;
  for ( i = &v11[4 * v9]; i != v11; v11 += 4 )
  {
    while ( 1 )
    {
      if ( v11 )
      {
        v13 = v11 + 2;
        *((_DWORD *)v11 + 2) = 0;
        *v11 = v11 + 2;
        *((_DWORD *)v11 + 3) = 4;
        v14 = v176;
        if ( v11 != &src )
        {
          if ( (_DWORD)v176 )
            break;
        }
      }
      v11 += 4;
      if ( i == v11 )
        goto LABEL_16;
    }
    v15 = 4LL * (unsigned int)v176;
    if ( (unsigned int)v176 <= 4
      || (sub_16CD150((__int64)v11, v11 + 2, (unsigned int)v176, 4, v176, a6),
          v13 = *v11,
          (v15 = 4LL * (unsigned int)v176) != 0) )
    {
      memcpy(v13, src, v15);
    }
    *((_DWORD *)v11 + 2) = v14;
  }
LABEL_16:
  if ( src != v177 )
    _libc_free((unsigned __int64)src);
  v16 = (v9 + 63) >> 6;
  v159 = (char *)malloc(8LL * v16);
  if ( !v159 )
  {
    if ( 8LL * v16 || (v153 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed", 1u);
    else
      v159 = (char *)v153;
  }
  if ( v16 )
    memset(v159, 0, 8LL * v16);
  v17 = (char *)a1[34];
  v18 = (char *)a1[33];
  v19 = v17 - v18;
  v20 = (v17 - v18) >> 3;
  v21 = v20;
  if ( !(_DWORD)v20 )
  {
    v160 = 0;
    goto LABEL_83;
  }
  v160 = 0;
  v158 = 0;
  v154 = 8LL * (unsigned int)(v20 - 1);
  while ( 1 )
  {
    v22 = v6[30];
    v23 = *(_QWORD *)&v18[v158];
    v24 = *(_QWORD *)(v22 + 8);
    v161 = *(_DWORD *)(v23 + 112) - 0x40000000;
    v156 = *(unsigned __int8 *)(v24 + 40LL * (unsigned int)(*(_DWORD *)(v22 + 32) + v161) + 23);
    v164 = 24 * v156;
    v25 = v6[229] + 24 * v156;
    v26 = *(_QWORD *)v25;
    if ( !byte_4FCAE20 )
    {
      v27 = *(_DWORD *)(v25 + 16);
      if ( v27 )
      {
        v28 = -v27;
        v29 = (unsigned int)(v27 - 1) >> 6;
        v30 = 0;
        while ( 1 )
        {
          _RDX = *(_QWORD *)(v26 + 8 * v30);
          if ( v29 == (_DWORD)v30 )
            _RDX = (0xFFFFFFFFFFFFFFFFLL >> v28) & *(_QWORD *)(v26 + 8 * v30);
          if ( _RDX )
            break;
          if ( v29 + 1 == ++v30 )
            goto LABEL_30;
        }
        __asm { tzcnt   rdx, rdx }
        v169 = ((_DWORD)v30 << 6) + _RDX;
        if ( v169 != -1 )
        {
          while ( 1 )
          {
            v167 = (int)v169;
            v59 = v6[237] + 48LL * (int)v169;
            v165 = 48LL * (int)v169;
            v60 = *(unsigned int *)(v59 + 8);
            if ( !(_DWORD)v60 )
              break;
            v61 = 8 * v60;
            v62 = 0;
            while ( !*(_DWORD *)(v23 + 8)
                 || !(unsigned __int8)sub_1DB3D00(*(__int64 ***)(*(_QWORD *)v59 + v62), v23, *(__int64 **)v23) )
            {
              v62 += 8;
              if ( v61 == v62 )
                goto LABEL_63;
            }
            v63 = v6[229] + v164;
            v64 = *(_DWORD *)(v63 + 16);
            v65 = v169 + 1;
            v26 = *(_QWORD *)v63;
            if ( v64 != v169 + 1 )
            {
              v66 = v65 >> 6;
              v67 = (unsigned int)(v64 - 1) >> 6;
              if ( v65 >> 6 <= v67 )
              {
                v68 = 64 - (v65 & 0x3F);
                v69 = 0xFFFFFFFFFFFFFFFFLL >> v68;
                if ( v68 == 64 )
                  v69 = 0;
                v70 = -v64;
                v71 = v66;
                v24 = ~v69;
                while ( 1 )
                {
                  _RAX = *(_QWORD *)(v26 + 8 * v71);
                  if ( v66 == (_DWORD)v71 )
                    _RAX = v24 & *(_QWORD *)(v26 + 8 * v71);
                  if ( v67 == (_DWORD)v71 )
                    _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v70;
                  if ( _RAX )
                    break;
                  if ( v67 < (unsigned int)++v71 )
                    goto LABEL_30;
                }
                __asm { tzcnt   rax, rax }
                v169 = ((_DWORD)v71 << 6) + _RAX;
                if ( v169 != -1 )
                  continue;
              }
            }
            goto LABEL_30;
          }
LABEL_63:
          if ( *(_BYTE *)(*(_QWORD *)(v6[30] + 8LL) + 40LL * (*(_DWORD *)(v6[30] + 32LL) + v169) + 23) == *(_BYTE *)(*(_QWORD *)(v6[30] + 8LL) + 40LL * (unsigned int)(v161 + *(_DWORD *)(v6[30] + 32LL)) + 23) )
          {
            v47 = 1;
            v157 = 1LL << v169;
            v33 = 8LL * (v169 >> 6);
            goto LABEL_43;
          }
          v26 = *(_QWORD *)(v6[229] + 24 * v156);
        }
      }
    }
LABEL_30:
    v32 = v156;
    v169 = *(_DWORD *)(v6[226] + 4 * v156);
    v33 = 8LL * (v169 >> 6);
    v157 = 1LL << v169;
    *(_QWORD *)(v26 + v33) |= 1LL << v169;
    v34 = v6[218] + v164;
    v24 = v6[226] + 4 * v32;
    v35 = *(_DWORD *)(v34 + 16);
    v36 = *(_DWORD *)v24 + 1;
    if ( v35 == v36 || (v37 = v36 >> 6, v38 = (unsigned int)(v35 - 1) >> 6, v36 >> 6 > v38) )
    {
LABEL_56:
      v46 = -1;
    }
    else
    {
      v39 = *(_QWORD *)v34;
      v40 = 64 - (v36 & 0x3F);
      v41 = 0xFFFFFFFFFFFFFFFFLL >> v40;
      if ( v40 == 64 )
        v41 = 0;
      v42 = -v35;
      v43 = v37;
      v26 = ~v41;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v39 + 8 * v43);
        if ( v37 == (_DWORD)v43 )
          _RAX = v26 & *(_QWORD *)(v39 + 8 * v43);
        if ( v38 == (_DWORD)v43 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v42;
        if ( _RAX )
          break;
        if ( v38 < (unsigned int)++v43 )
          goto LABEL_56;
      }
      __asm { tzcnt   rax, rax }
      v46 = ((_DWORD)v43 << 6) + _RAX;
    }
    *(_DWORD *)v24 = v46;
    v167 = (int)v169;
    v47 = 0;
    v165 = 48LL * (int)v169;
LABEL_43:
    v48 = v6[237] + v165;
    v49 = *(unsigned int *)(v48 + 8);
    if ( (unsigned int)v49 >= *(_DWORD *)(v48 + 12) )
    {
      v150 = v6[237] + v165;
      v166 = v47;
      sub_16CD150(v150, (const void *)(v48 + 16), 0, 8, v24, v26);
      v49 = *(unsigned int *)(v48 + 8);
      v47 = v166;
    }
    *(_QWORD *)(*(_QWORD *)v48 + 8 * v49) = v23;
    ++*(_DWORD *)(v48 + 8);
    v50 = v6[30];
    v51 = *(_DWORD *)(v6[198] + 4LL * v161);
    if ( !v47 )
    {
      *(_DWORD *)(*(_QWORD *)(v50 + 8) + 40LL * (*(_DWORD *)(v50 + 32) + v169) + 16) = v51;
      sub_1E08740(v50, v51);
      v54 = *(_DWORD *)(v6[208] + 4LL * v161);
      v52 = *(_QWORD *)(v6[30] + 8LL) + 40LL * (*(_DWORD *)(v6[30] + 32LL) + v169);
LABEL_49:
      *(_QWORD *)(v52 + 8) = v54;
      goto LABEL_50;
    }
    v52 = *(_QWORD *)(v50 + 8) + 40LL * (*(_DWORD *)(v50 + 32) + v169);
    if ( v51 <= *(_DWORD *)(v52 + 16) )
    {
      v53 = *(unsigned int *)(v6[208] + 4LL * v161);
      v54 = *(_DWORD *)(v6[208] + 4LL * v161);
    }
    else
    {
      *(_DWORD *)(v52 + 16) = v51;
      sub_1E08740(v50, v51);
      v53 = *(unsigned int *)(v6[208] + 4LL * v161);
      v54 = *(_DWORD *)(v6[208] + 4LL * v161);
      v52 = *(_QWORD *)(v6[30] + 8LL) + 40LL * (*(_DWORD *)(v6[30] + 32LL) + v169);
    }
    if ( v53 > *(_QWORD *)(v52 + 8) )
      goto LABEL_49;
LABEL_50:
    *((_DWORD *)v178 + v161) = v169;
    v55 = &v184[4 * v167];
    v56 = *((unsigned int *)v55 + 2);
    if ( (unsigned int)v56 >= *((_DWORD *)v55 + 3) )
    {
      sub_16CD150((__int64)&v184[4 * v167], v55 + 2, 0, 4, v24, v26);
      v56 = *((unsigned int *)v55 + 2);
    }
    *(_DWORD *)(*v55 + 4 * v56) = v161;
    v57 = (float *)v181;
    ++*((_DWORD *)v55 + 2);
    v57[v167] = v57[v167] + *(float *)(v23 + 116);
    *(_QWORD *)&v159[v33] |= v157;
    v160 |= v161 != v169;
    if ( v154 == v158 )
      break;
    v18 = (char *)v6[33];
    v158 += 8;
  }
  v17 = (char *)v6[34];
  v18 = (char *)v6[33];
  v19 = v17 - v18;
  v74 = (v17 - v18) >> 3;
  v21 = v74;
  if ( (_DWORD)v74 )
  {
    v75 = (unsigned int)(v74 - 1);
    v76 = v181;
    v77 = 0;
    v78 = 8 * v75;
    while ( 1 )
    {
      *(_DWORD *)(*(_QWORD *)&v18[v77] + 116LL) = v76[*(_DWORD *)(*(_QWORD *)&v18[v77] + 112LL) - 0x40000000];
      v18 = (char *)v6[33];
      if ( v78 == v77 )
        break;
      v77 += 8;
    }
    v17 = (char *)v6[34];
    v19 = v17 - v18;
    v21 = (v17 - v18) >> 3;
  }
LABEL_83:
  if ( v19 <= 0 )
  {
LABEL_207:
    v80 = 0;
    sub_1F2F820(v18, v17);
    v81 = 0;
  }
  else
  {
    while ( 1 )
    {
      v79 = (char *)sub_2207800(8 * v21, &unk_435FF63);
      v80 = v79;
      if ( v79 )
        break;
      v21 >>= 1;
      if ( !v21 )
        goto LABEL_207;
    }
    sub_1F30850(v18, v17, v79, (char *)v21);
    v81 = 8 * v21;
  }
  j_j___libc_free_0(v80, v81);
  if ( v160 )
  {
    v82 = *((unsigned int *)v6 + 74);
    v83 = 0;
    if ( (_DWORD)v82 )
    {
      do
      {
        while ( 1 )
        {
          v84 = *((_DWORD *)v178 + v83);
          if ( v84 != -1 && v84 != (_DWORD)v83 )
          {
            v85 = sub_1EB39F0(*(__int64 **)(a2 + 376), v84);
            v86 = v6[36] + 80 * v83;
            v87 = v85;
            v88 = *(unsigned int *)(v86 + 8);
            if ( (_DWORD)v88 )
              break;
          }
          if ( v82 == ++v83 )
            goto LABEL_96;
        }
        v89 = 8 * v88;
        v90 = v87 | 4;
        v91 = 0;
        do
        {
          v92 = *(__int64 **)(*(_QWORD *)v86 + v91);
          v91 += 8;
          *v92 = v90;
        }
        while ( v89 != v91 );
        ++v83;
      }
      while ( v82 != v83 );
    }
LABEL_96:
    v168 = *(_QWORD *)(a2 + 328);
    if ( a2 + 320 != v168 )
    {
      while ( 1 )
      {
        v93 = *(_QWORD *)(v168 + 32);
        for ( j = v168 + 24; j != v93; v93 = *(_QWORD *)(v93 + 8) )
        {
          v95 = *(unsigned int *)(v93 + 40);
          if ( (_DWORD)v95 )
          {
            v96 = 5 * v95;
            v97 = v178;
            v98 = 0;
            v99 = 8 * v96;
            do
            {
              v100 = v98 + *(_QWORD *)(v93 + 32);
              if ( *(_BYTE *)v100 == 5 )
              {
                v101 = *(_DWORD *)(v100 + 24);
                if ( v101 >= 0 )
                {
                  v102 = v97[v101];
                  if ( v102 != -1 && v101 != v102 )
                    *(_DWORD *)(v100 + 24) = v102;
                }
              }
              v98 += 40;
            }
            while ( v99 != v98 );
          }
          if ( (*(_BYTE *)v93 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v93 + 46) & 8) != 0 )
              v93 = *(_QWORD *)(v93 + 8);
          }
        }
        v103 = v6;
        src = v177;
        v176 = 0x400000000LL;
        v104 = *(_QWORD *)(v168 + 32);
        if ( j != v104 )
          break;
LABEL_126:
        v168 = *(_QWORD *)(v168 + 8);
        if ( a2 + 320 == v168 )
          goto LABEL_127;
      }
      while ( 2 )
      {
        if ( dword_4FCAD40 != -1 && dword_4CD4AB8 >= dword_4FCAD40 )
        {
LABEL_121:
          v112 = (__int64 *)src;
          v6 = v103;
          v113 = (__int64 *)((char *)src + 8 * (unsigned int)v176);
          if ( src != v113 )
          {
            do
            {
              v114 = *v112++;
              sub_1E16240(v114);
            }
            while ( v113 != v112 );
            v113 = (__int64 *)src;
          }
          if ( v113 != (__int64 *)v177 )
            _libc_free((unsigned __int64)v113);
          goto LABEL_126;
        }
        v105 = v103[31];
        v106 = *(__int64 (**)())(*(_QWORD *)v105 + 112LL);
        if ( v106 == sub_1F2E880
          || !((unsigned __int8 (__fastcall *)(__int64, __int64, int *, int *))v106)(v105, v104, &v171, &v172)
          || v172 != v171
          || v171 == -1 )
        {
          if ( !v104 )
            BUG();
          v107 = v104;
          if ( (*(_BYTE *)v104 & 4) == 0 && (*(_BYTE *)(v104 + 46) & 8) != 0 )
          {
            do
              v107 = *(_QWORD *)(v107 + 8);
            while ( (*(_BYTE *)(v107 + 46) & 8) != 0 );
          }
          v108 = v103[31];
          v109 = *(_QWORD *)(v107 + 8);
          v173 = 0;
          v174 = 0;
          v110 = *(__int64 (__fastcall **)(__int64, __int64, __int64, int *))(*(_QWORD *)v108 + 56LL);
          if ( v110 == sub_1F2EAA0 )
          {
            v111 = *(__int64 (**)())(*(_QWORD *)v108 + 48LL);
            if ( v111 == sub_1E1C810 )
              goto LABEL_118;
            v139 = ((__int64 (__fastcall *)(__int64, __int64, int *))v111)(v108, v104, &v171);
          }
          else
          {
            v139 = v110(v108, v104, (__int64)&v171, &v173);
          }
          if ( v139 && j != v109 )
          {
            v140 = v104;
            while ( (unsigned __int16)(**(_WORD **)(v109 + 16) - 12) <= 1u )
            {
              if ( (*(_BYTE *)v109 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v109 + 46) & 8) != 0 )
                  v109 = *(_QWORD *)(v109 + 8);
              }
              v109 = *(_QWORD *)(v109 + 8);
              if ( !v140 )
                BUG();
              if ( (*(_BYTE *)v140 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v140 + 46) & 8) != 0 )
                  v140 = *(_QWORD *)(v140 + 8);
              }
              v140 = *(_QWORD *)(v140 + 8);
              if ( j == v109 )
                goto LABEL_176;
            }
            if ( j == v109 )
              goto LABEL_176;
            v141 = v103[31];
            v142 = *(__int64 (__fastcall **)(__int64, __int64, __int64, int *))(*(_QWORD *)v141 + 88LL);
            if ( v142 == sub_1F2EA70 )
            {
              v174 = 0;
              v143 = *(__int64 (**)())(*(_QWORD *)v141 + 80LL);
              if ( v143 == sub_1EBAF80 )
                goto LABEL_176;
              v162 = v139;
              v144 = ((__int64 (__fastcall *)(__int64, __int64, int *))v143)(v141, v109, &v172);
              v145 = v162;
            }
            else
            {
              v163 = v139;
              v144 = v142(v141, v109, (__int64)&v172, &v174);
              v145 = v163;
            }
            if ( !v144 || v145 != v144 || v172 != v171 || v171 == -1 || v173 != v174 )
            {
LABEL_176:
              v104 = v140;
              goto LABEL_118;
            }
            v146 = (unsigned int)sub_1E165A0(v109, v145, 1, 0) == -1;
            v149 = (unsigned int)v176;
            if ( !v146 )
            {
              if ( (unsigned int)v176 >= HIDWORD(v176) )
              {
                sub_16CD150((__int64)&src, v177, 0, 8, v147, v148);
                v149 = (unsigned int)v176;
              }
              *((_QWORD *)src + v149) = v104;
              v149 = (unsigned int)(v176 + 1);
              LODWORD(v176) = v176 + 1;
            }
            if ( HIDWORD(v176) <= (unsigned int)v149 )
            {
              sub_16CD150((__int64)&src, v177, 0, 8, v147, v148);
              v149 = (unsigned int)v176;
            }
            *((_QWORD *)src + v149) = v109;
            LODWORD(v176) = v176 + 1;
            if ( !v140 )
              BUG();
            if ( (*(_BYTE *)v140 & 4) != 0 )
            {
              v104 = *(_QWORD *)(v140 + 8);
            }
            else
            {
              while ( (*(_BYTE *)(v140 + 46) & 8) != 0 )
                v140 = *(_QWORD *)(v140 + 8);
              v104 = *(_QWORD *)(v140 + 8);
            }
          }
        }
        else
        {
          v138 = (unsigned int)v176;
          if ( (unsigned int)v176 >= HIDWORD(v176) )
          {
            sub_16CD150((__int64)&src, v177, 0, 8, v136, v137);
            v138 = (unsigned int)v176;
          }
          *((_QWORD *)src + v138) = v104;
          LODWORD(v176) = v176 + 1;
        }
LABEL_118:
        if ( !v104 )
          BUG();
        if ( (*(_BYTE *)v104 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v104 + 46) & 8) != 0 )
            v104 = *(_QWORD *)(v104 + 8);
        }
        v104 = *(_QWORD *)(v104 + 8);
        if ( j == v104 )
          goto LABEL_121;
        continue;
      }
    }
LABEL_127:
    v115 = *((_DWORD *)v6 + 438);
    if ( v115 )
    {
      v116 = 0;
      v117 = 0;
      v170 = 4LL * (unsigned int)(v115 - 1) + 4;
      do
      {
        for ( k = *(_DWORD *)(v6[226] + v117); k != -1; k = ((_DWORD)v129 << 6) + _RAX )
        {
          v119 = k;
          v120 = k + 1;
          *(_QWORD *)(*(_QWORD *)(v6[30] + 8LL) + 40LL * (unsigned int)(*(_DWORD *)(v6[30] + 32LL) + v119) + 8) = -1;
          v121 = (__int64 *)(v116 + v6[218]);
          v122 = *((_DWORD *)v121 + 4);
          if ( v122 == v120 )
            break;
          v123 = v120 >> 6;
          v124 = (unsigned int)(v122 - 1) >> 6;
          if ( v120 >> 6 > v124 )
            break;
          v125 = *v121;
          v126 = 64 - (v120 & 0x3F);
          v127 = 0xFFFFFFFFFFFFFFFFLL >> v126;
          if ( v126 == 64 )
            v127 = 0;
          v128 = -v122;
          v129 = v123;
          v130 = ~v127;
          while ( 1 )
          {
            _RAX = *(_QWORD *)(v125 + 8 * v129);
            if ( v123 == (_DWORD)v129 )
              _RAX = v130 & *(_QWORD *)(v125 + 8 * v129);
            if ( v124 == (_DWORD)v129 )
              _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v128;
            if ( _RAX )
              break;
            if ( v124 < (unsigned int)++v129 )
              goto LABEL_142;
          }
          __asm { tzcnt   rax, rax }
        }
LABEL_142:
        v117 += 4;
        v116 += 24;
      }
      while ( v170 != v117 );
    }
  }
  _libc_free((unsigned __int64)v159);
  v133 = v184;
  v134 = &v184[4 * (unsigned int)v185];
  if ( v184 != v134 )
  {
    do
    {
      v134 -= 4;
      if ( (unsigned __int64 *)*v134 != v134 + 2 )
        _libc_free(*v134);
    }
    while ( v133 != v134 );
    v134 = v184;
  }
  if ( v134 != (unsigned __int64 *)v186 )
    _libc_free((unsigned __int64)v134);
  if ( v181 != v183 )
    _libc_free((unsigned __int64)v181);
  if ( v178 != v180 )
    _libc_free((unsigned __int64)v178);
  return v160;
}
