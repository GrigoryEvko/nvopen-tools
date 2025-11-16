// Function: sub_D17E50
// Address: 0xd17e50
//
__int64 __fastcall sub_D17E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  unsigned int v8; // eax
  __int64 v9; // rdx
  int v10; // r13d
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r15
  unsigned int v14; // eax
  __int64 v15; // r12
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 j; // r15
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned int v24; // eax
  __int64 *v25; // rcx
  __int64 v26; // rdi
  __int64 v27; // rcx
  int v28; // edx
  char v29; // r12
  __int64 *v30; // rbx
  __int64 *v31; // rdi
  __int64 *v32; // r15
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rdx
  _QWORD *v36; // rax
  unsigned int v38; // eax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rdx
  __int64 v41; // rax
  int v42; // r13d
  __int64 v43; // r10
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // eax
  unsigned int v50; // eax
  __int64 v51; // rcx
  __int64 v52; // rdx
  int v53; // eax
  __int64 **v54; // rax
  __int64 **v55; // rdx
  __int64 **v56; // rax
  __int64 v57; // rcx
  bool v58; // al
  unsigned int v59; // esi
  int v60; // r11d
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdi
  unsigned int v64; // eax
  unsigned int v65; // ebx
  bool v66; // al
  __int64 *v67; // rax
  int v68; // ecx
  __int64 v69; // r13
  __int64 v70; // r12
  __int64 v71; // rdi
  int v72; // edx
  __int64 v73; // rax
  __int64 *v74; // r12
  __int64 v75; // rcx
  __int64 *v76; // rbx
  __int64 v77; // rdi
  __int64 v78; // rdx
  _QWORD *v79; // rax
  __int64 v80; // rax
  unsigned int v81; // eax
  unsigned int v82; // esi
  unsigned __int64 v83; // rdx
  __int64 v84; // rax
  int v85; // r11d
  unsigned int v86; // edx
  __int64 v87; // rdi
  __int64 *v88; // rax
  __int64 *v89; // rax
  __int64 v90; // rdi
  int v91; // eax
  int v92; // r10d
  __int64 v93; // r11
  int v94; // edx
  unsigned int v95; // esi
  __int64 *v96; // rdi
  unsigned int v97; // eax
  __int64 v98; // rdi
  unsigned int v99; // edx
  __int64 v100; // rax
  __int64 v101; // rcx
  int v102; // edi
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // rdi
  int v106; // ecx
  int v107; // edi
  int v108; // edx
  bool v109; // cc
  const void *v110; // rdx
  int v111; // edi
  int v112; // eax
  int v113; // r10d
  __int64 v114; // r11
  __int64 *v115; // rdi
  unsigned int v116; // esi
  __int64 v117; // rdx
  int v118; // ebx
  unsigned int v119; // eax
  _QWORD *v120; // rdi
  unsigned __int64 v121; // rdx
  unsigned __int64 v122; // rax
  _QWORD *v123; // rax
  __int64 v124; // rdx
  _QWORD *i; // rdx
  int v126; // eax
  int v127; // r13d
  __int64 v128; // r10
  unsigned int v129; // esi
  __int64 v130; // rdi
  int v131; // edx
  int v132; // r13d
  __int64 v133; // rdi
  __int64 v134; // r10
  unsigned int v135; // esi
  int v136; // r8d
  unsigned int v137; // ebx
  __int64 v138; // rdi
  __int64 v139; // rsi
  int v140; // r8d
  __int64 v141; // r11
  int v142; // edi
  __int64 v143; // rsi
  _QWORD *v144; // rax
  int v145; // r11d
  int v146; // r11d
  __int64 v147; // r10
  __int64 v148; // rdi
  __int64 v149; // rsi
  int v150; // r10d
  int v151; // r10d
  unsigned int v152; // r11d
  __int64 v153; // rsi
  unsigned int v154; // r11d
  unsigned int v155; // r11d
  __int64 *v156; // [rsp+8h] [rbp-1A8h]
  __int64 v157; // [rsp+10h] [rbp-1A0h]
  __int64 v158; // [rsp+18h] [rbp-198h]
  __int64 v159; // [rsp+20h] [rbp-190h]
  __int64 v160; // [rsp+28h] [rbp-188h]
  unsigned int v161; // [rsp+28h] [rbp-188h]
  unsigned int v162; // [rsp+28h] [rbp-188h]
  __int64 v163; // [rsp+28h] [rbp-188h]
  unsigned int v164; // [rsp+28h] [rbp-188h]
  __int64 v165; // [rsp+28h] [rbp-188h]
  unsigned int v166; // [rsp+28h] [rbp-188h]
  unsigned int v167; // [rsp+30h] [rbp-180h]
  __int64 v168; // [rsp+38h] [rbp-178h]
  __int64 v169; // [rsp+48h] [rbp-168h]
  char v170; // [rsp+57h] [rbp-159h] BYREF
  __int64 v171; // [rsp+58h] [rbp-158h] BYREF
  const void *v172; // [rsp+60h] [rbp-150h] BYREF
  unsigned int v173; // [rsp+68h] [rbp-148h]
  unsigned __int64 v174; // [rsp+70h] [rbp-140h] BYREF
  unsigned int v175; // [rsp+78h] [rbp-138h]
  unsigned __int64 v176; // [rsp+80h] [rbp-130h] BYREF
  unsigned int v177; // [rsp+88h] [rbp-128h]
  __int64 v178; // [rsp+90h] [rbp-120h] BYREF
  unsigned int v179; // [rsp+98h] [rbp-118h]
  __int64 v180; // [rsp+A0h] [rbp-110h]
  unsigned int v181; // [rsp+A8h] [rbp-108h]
  __int64 v182; // [rsp+B0h] [rbp-100h] BYREF
  unsigned int v183; // [rsp+B8h] [rbp-F8h]
  __int64 v184; // [rsp+C0h] [rbp-F0h]
  unsigned int v185; // [rsp+C8h] [rbp-E8h]
  __int64 v186; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v187; // [rsp+D8h] [rbp-D8h]
  __int64 v188; // [rsp+E0h] [rbp-D0h]
  __int64 v189; // [rsp+E8h] [rbp-C8h]
  _BYTE *v190; // [rsp+F0h] [rbp-C0h]
  __int64 v191; // [rsp+F8h] [rbp-B8h]
  _BYTE v192[176]; // [rsp+100h] [rbp-B0h] BYREF

  ++*(_QWORD *)(a1 + 32);
  v7 = *(_BYTE *)(a1 + 60) == 0;
  *(_BYTE *)(a1 + 24) = 1;
  v169 = a1 + 32;
  if ( v7 )
  {
    v8 = 4 * (*(_DWORD *)(a1 + 52) - *(_DWORD *)(a1 + 56));
    v9 = *(unsigned int *)(a1 + 48);
    if ( v8 < 0x20 )
      v8 = 32;
    if ( (unsigned int)v9 > v8 )
    {
      sub_C8C990(v169, a2);
      goto LABEL_7;
    }
    a2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 40), -1, 8 * v9);
  }
  *(_QWORD *)(a1 + 52) = 0;
LABEL_7:
  v10 = *(_DWORD *)(a1 + 336);
  ++*(_QWORD *)(a1 + 320);
  v157 = a1 + 320;
  if ( !v10 && !*(_DWORD *)(a1 + 340) )
    goto LABEL_21;
  v11 = *(_QWORD *)(a1 + 328);
  v12 = *(unsigned int *)(a1 + 344);
  v13 = 24 * v12;
  v14 = 4 * v10;
  v15 = v11 + 24 * v12;
  if ( (unsigned int)(4 * v10) < 0x40 )
    v14 = 64;
  if ( (unsigned int)v12 <= v14 )
  {
    for ( ; v11 != v15; v11 += 24 )
    {
      if ( *(_QWORD *)v11 != -4096 )
      {
        if ( *(_QWORD *)v11 != -8192 && *(_DWORD *)(v11 + 16) > 0x40u )
        {
          v16 = *(_QWORD *)(v11 + 8);
          if ( v16 )
            j_j___libc_free_0_0(v16);
        }
        *(_QWORD *)v11 = -4096;
      }
    }
    goto LABEL_20;
  }
  do
  {
    while ( *(_QWORD *)v11 == -8192 )
    {
LABEL_213:
      v11 += 24;
      if ( v11 == v15 )
        goto LABEL_268;
    }
    if ( *(_QWORD *)v11 != -4096 )
    {
      if ( *(_DWORD *)(v11 + 16) > 0x40u )
      {
        v105 = *(_QWORD *)(v11 + 8);
        if ( v105 )
          j_j___libc_free_0_0(v105);
      }
      goto LABEL_213;
    }
    v11 += 24;
  }
  while ( v11 != v15 );
LABEL_268:
  v117 = *(unsigned int *)(a1 + 344);
  if ( !v10 )
  {
    if ( (_DWORD)v117 )
    {
      a2 = v13;
      sub_C7D6A0(*(_QWORD *)(a1 + 328), v13, 8);
      *(_QWORD *)(a1 + 328) = 0;
      *(_QWORD *)(a1 + 336) = 0;
      *(_DWORD *)(a1 + 344) = 0;
      goto LABEL_21;
    }
LABEL_20:
    *(_QWORD *)(a1 + 336) = 0;
    goto LABEL_21;
  }
  v118 = 64;
  if ( v10 != 1 )
  {
    _BitScanReverse(&v119, v10 - 1);
    v118 = 1 << (33 - (v119 ^ 0x1F));
    if ( v118 < 64 )
      v118 = 64;
  }
  v120 = *(_QWORD **)(a1 + 328);
  if ( (_DWORD)v117 == v118 )
  {
    *(_QWORD *)(a1 + 336) = 0;
    v144 = &v120[3 * v117];
    do
    {
      if ( v120 )
        *v120 = -4096;
      v120 += 3;
    }
    while ( v144 != v120 );
  }
  else
  {
    sub_C7D6A0((__int64)v120, v13, 8);
    a2 = 8;
    v121 = ((((((((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
              | (4 * v118 / 3u + 1)
              | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 4)
            | (((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
            | (4 * v118 / 3u + 1)
            | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
            | (4 * v118 / 3u + 1)
            | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 4)
          | (((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
          | (4 * v118 / 3u + 1)
          | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 16;
    v122 = (v121
          | (((((((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
              | (4 * v118 / 3u + 1)
              | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 4)
            | (((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
            | (4 * v118 / 3u + 1)
            | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
            | (4 * v118 / 3u + 1)
            | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 4)
          | (((4 * v118 / 3u + 1) | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1)) >> 2)
          | (4 * v118 / 3u + 1)
          | ((unsigned __int64)(4 * v118 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(a1 + 344) = v122;
    v123 = (_QWORD *)sub_C7D670(24 * v122, 8);
    v124 = *(unsigned int *)(a1 + 344);
    *(_QWORD *)(a1 + 336) = 0;
    *(_QWORD *)(a1 + 328) = v123;
    for ( i = &v123[3 * v124]; i != v123; v123 += 3 )
    {
      if ( v123 )
        *v123 = -4096;
    }
  }
LABEL_21:
  ++*(_QWORD *)(a1 + 352);
  v158 = a1 + 352;
  if ( *(_BYTE *)(a1 + 380) )
  {
LABEL_26:
    *(_QWORD *)(a1 + 372) = 0;
  }
  else
  {
    v17 = 4 * (*(_DWORD *)(a1 + 372) - *(_DWORD *)(a1 + 376));
    v18 = *(unsigned int *)(a1 + 368);
    if ( v17 < 0x20 )
      v17 = 32;
    if ( (unsigned int)v18 <= v17 )
    {
      a2 = 0xFFFFFFFFLL;
      memset(*(void **)(a1 + 360), -1, 8 * v18);
      goto LABEL_26;
    }
    sub_C8C990(v158, a2);
  }
  v186 = 0;
  v190 = v192;
  v191 = 0x1000000000LL;
  v19 = *(_QWORD *)a1;
  v187 = 0;
  v20 = *(_QWORD *)(v19 + 80);
  v188 = 0;
  v189 = 0;
  if ( v19 + 72 == v20 )
  {
    j = 0;
  }
  else
  {
    if ( !v20 )
      BUG();
    while ( 1 )
    {
      j = *(_QWORD *)(v20 + 32);
      if ( j != v20 + 24 )
        break;
      v20 = *(_QWORD *)(v20 + 8);
      if ( v19 + 72 == v20 )
        goto LABEL_33;
      if ( !v20 )
        BUG();
    }
  }
  v69 = v19 + 72;
  if ( v19 + 72 == v20 )
    goto LABEL_33;
  while ( 2 )
  {
    v70 = j - 24;
    if ( !j )
      v70 = 0;
    if ( !(unsigned __int8)sub_D14860((unsigned __int8 *)v70) )
      goto LABEL_167;
    v71 = *(_QWORD *)(v70 + 8);
    v72 = *(unsigned __int8 *)(v71 + 8);
    if ( (unsigned int)(v72 - 17) <= 1 )
      LOBYTE(v72) = *(_BYTE *)(**(_QWORD **)(v71 + 16) + 8LL);
    if ( (_BYTE)v72 == 12 )
    {
      v97 = sub_BCB060(v71);
      a2 = *(unsigned int *)(a1 + 344);
      v167 = v97;
      if ( (_DWORD)a2 )
      {
        a6 = 1;
        a5 = *(_QWORD *)(a1 + 328);
        v98 = 0;
        v99 = (a2 - 1) & (((unsigned int)v70 >> 4) ^ ((unsigned int)v70 >> 9));
        v100 = a5 + 24LL * v99;
        v101 = *(_QWORD *)v100;
        if ( v70 == *(_QWORD *)v100 )
          goto LABEL_167;
        while ( v101 != -4096 )
        {
          if ( !v98 && v101 == -8192 )
            v98 = v100;
          v99 = (a2 - 1) & (a6 + v99);
          v100 = a5 + 24LL * v99;
          v101 = *(_QWORD *)v100;
          if ( v70 == *(_QWORD *)v100 )
            goto LABEL_167;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v98 )
          v100 = v98;
        v102 = *(_DWORD *)(a1 + 336);
        ++*(_QWORD *)(a1 + 320);
        v103 = (unsigned int)(v102 + 1);
        if ( 4 * (int)v103 < (unsigned int)(3 * a2) )
        {
          v104 = (unsigned int)(a2 - *(_DWORD *)(a1 + 340) - v103);
          if ( (unsigned int)v104 > (unsigned int)a2 >> 3 )
            goto LABEL_205;
          v166 = ((unsigned int)v70 >> 4) ^ ((unsigned int)v70 >> 9);
          sub_D175E0(v157, a2);
          v150 = *(_DWORD *)(a1 + 344);
          if ( v150 )
          {
            v151 = v150 - 1;
            v104 = 0;
            a6 = *(_QWORD *)(a1 + 328);
            a5 = 1;
            v152 = v151 & v166;
            v103 = (unsigned int)(*(_DWORD *)(a1 + 336) + 1);
            v100 = a6 + 24LL * (v151 & v166);
            v153 = *(_QWORD *)v100;
            if ( v70 != *(_QWORD *)v100 )
            {
              while ( v153 != -4096 )
              {
                if ( v153 == -8192 && !v104 )
                  v104 = v100;
                v152 = v151 & (a5 + v152);
                v100 = a6 + 24LL * v152;
                v153 = *(_QWORD *)v100;
                if ( v70 == *(_QWORD *)v100 )
                  goto LABEL_205;
                a5 = (unsigned int)(a5 + 1);
              }
              if ( v104 )
                v100 = v104;
            }
            goto LABEL_205;
          }
          goto LABEL_375;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 320);
      }
      sub_D175E0(v157, 2 * a2);
      v145 = *(_DWORD *)(a1 + 344);
      if ( v145 )
      {
        v146 = v145 - 1;
        v147 = *(_QWORD *)(a1 + 328);
        v104 = v146 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
        v103 = (unsigned int)(*(_DWORD *)(a1 + 336) + 1);
        v100 = v147 + 24 * v104;
        v148 = *(_QWORD *)v100;
        if ( v70 != *(_QWORD *)v100 )
        {
          a6 = 1;
          v149 = 0;
          while ( v148 != -4096 )
          {
            if ( !v149 && v148 == -8192 )
              v149 = v100;
            a5 = (unsigned int)(a6 + 1);
            v104 = v146 & (unsigned int)(a6 + v104);
            v100 = v147 + 24LL * (unsigned int)v104;
            v148 = *(_QWORD *)v100;
            if ( v70 == *(_QWORD *)v100 )
              goto LABEL_205;
            a6 = (unsigned int)a5;
          }
          if ( v149 )
            v100 = v149;
        }
LABEL_205:
        *(_DWORD *)(a1 + 336) = v103;
        if ( *(_QWORD *)v100 != -4096 )
          --*(_DWORD *)(a1 + 340);
        *(_QWORD *)v100 = v70;
        *(_DWORD *)(v100 + 16) = v167;
        if ( v167 > 0x40 )
          sub_C43690(v100 + 8, 0, 0);
        else
          *(_QWORD *)(v100 + 8) = 0;
        a2 = (__int64)&v182;
        v182 = v70;
        sub_D17810((__int64)&v186, &v182, v103, v104, a5, a6);
        goto LABEL_167;
      }
LABEL_375:
      ++*(_DWORD *)(a1 + 336);
      BUG();
    }
    v73 = 4LL * (*(_DWORD *)(v70 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v70 + 7) & 0x40) != 0 )
    {
      v74 = *(__int64 **)(v70 - 8);
      v75 = (__int64)&v74[v73];
    }
    else
    {
      v75 = v70;
      v74 = (__int64 *)(v70 - v73 * 8);
    }
    if ( v74 == (__int64 *)v75 )
      goto LABEL_167;
    v165 = v20;
    v76 = (__int64 *)v75;
    while ( 2 )
    {
      a2 = *v74;
      if ( *(_BYTE *)*v74 <= 0x1Cu )
        goto LABEL_165;
      v178 = *v74;
      v77 = *(_QWORD *)(a2 + 8);
      v78 = *(unsigned __int8 *)(v77 + 8);
      if ( (unsigned int)(v78 - 17) <= 1 )
        v78 = *(unsigned __int8 *)(**(_QWORD **)(v77 + 16) + 8LL);
      if ( (_BYTE)v78 == 12 )
      {
        v81 = sub_BCB060(v77);
        v183 = v81;
        if ( v81 > 0x40 )
        {
          sub_C43690((__int64)&v182, -1, 1);
          v82 = *(_DWORD *)(a1 + 344);
          if ( v82 )
            goto LABEL_178;
        }
        else
        {
          v82 = *(_DWORD *)(a1 + 344);
          v83 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v81;
          v7 = v81 == 0;
          v84 = 0;
          if ( !v7 )
            v84 = v83;
          v182 = v84;
          if ( v82 )
          {
LABEL_178:
            v75 = v178;
            a6 = *(_QWORD *)(a1 + 328);
            v85 = 1;
            v86 = (v82 - 1) & (((unsigned int)v178 >> 9) ^ ((unsigned int)v178 >> 4));
            v87 = a6 + 24LL * v86;
            v88 = 0;
            a5 = *(_QWORD *)v87;
            if ( v178 == *(_QWORD *)v87 )
            {
LABEL_179:
              v89 = (__int64 *)(v87 + 8);
              if ( *(_DWORD *)(v87 + 16) > 0x40u )
              {
                v90 = *(_QWORD *)(v87 + 8);
                if ( v90 )
                {
                  v156 = v89;
                  j_j___libc_free_0_0(v90);
                  v89 = v156;
                }
              }
LABEL_182:
              *v89 = v182;
              v78 = v183;
              *((_DWORD *)v89 + 2) = v183;
              goto LABEL_164;
            }
            while ( a5 != -4096 )
            {
              if ( a5 == -8192 && !v88 )
                v88 = (__int64 *)v87;
              v86 = (v82 - 1) & (v85 + v86);
              v87 = a6 + 24LL * v86;
              a5 = *(_QWORD *)v87;
              if ( v178 == *(_QWORD *)v87 )
                goto LABEL_179;
              ++v85;
            }
            if ( !v88 )
              v88 = (__int64 *)v87;
            v111 = *(_DWORD *)(a1 + 336);
            ++*(_QWORD *)(a1 + 320);
            v94 = v111 + 1;
            if ( 4 * (v111 + 1) < 3 * v82 )
            {
              a5 = v82 >> 3;
              if ( v82 - *(_DWORD *)(a1 + 340) - v94 <= (unsigned int)a5 )
              {
                sub_D175E0(v157, v82);
                v112 = *(_DWORD *)(a1 + 344);
                if ( !v112 )
                {
LABEL_377:
                  ++*(_DWORD *)(a1 + 336);
                  BUG();
                }
                a5 = v178;
                v113 = v112 - 1;
                v114 = *(_QWORD *)(a1 + 328);
                a6 = 1;
                v94 = *(_DWORD *)(a1 + 336) + 1;
                v115 = 0;
                v116 = (v112 - 1) & (((unsigned int)v178 >> 9) ^ ((unsigned int)v178 >> 4));
                v88 = (__int64 *)(v114 + 24LL * v116);
                v75 = *v88;
                if ( v178 != *v88 )
                {
                  while ( v75 != -4096 )
                  {
                    if ( v75 == -8192 && !v115 )
                      v115 = v88;
                    v116 = v113 & (a6 + v116);
                    v88 = (__int64 *)(v114 + 24LL * v116);
                    v75 = *v88;
                    if ( v178 == *v88 )
                      goto LABEL_257;
                    a6 = (unsigned int)(a6 + 1);
                  }
                  v75 = v178;
                  if ( v115 )
                    v88 = v115;
                }
              }
              goto LABEL_257;
            }
LABEL_190:
            sub_D175E0(v157, 2 * v82);
            v91 = *(_DWORD *)(a1 + 344);
            if ( !v91 )
              goto LABEL_377;
            v75 = v178;
            v92 = v91 - 1;
            v93 = *(_QWORD *)(a1 + 328);
            v94 = *(_DWORD *)(a1 + 336) + 1;
            v95 = (v91 - 1) & (((unsigned int)v178 >> 9) ^ ((unsigned int)v178 >> 4));
            v88 = (__int64 *)(v93 + 24LL * v95);
            a6 = *v88;
            if ( *v88 != v178 )
            {
              a5 = 1;
              v96 = 0;
              while ( a6 != -4096 )
              {
                if ( !v96 && a6 == -8192 )
                  v96 = v88;
                v95 = v92 & (a5 + v95);
                v88 = (__int64 *)(v93 + 24LL * v95);
                a6 = *v88;
                if ( v178 == *v88 )
                  goto LABEL_257;
                a5 = (unsigned int)(a5 + 1);
              }
              if ( v96 )
                v88 = v96;
            }
LABEL_257:
            *(_DWORD *)(a1 + 336) = v94;
            if ( *v88 != -4096 )
              --*(_DWORD *)(a1 + 340);
            *v88 = v75;
            v89 = v88 + 1;
            *((_DWORD *)v89 + 2) = 1;
            *v89 = 0;
            goto LABEL_182;
          }
        }
        ++*(_QWORD *)(a1 + 320);
        goto LABEL_190;
      }
      if ( !*(_BYTE *)(a1 + 60) )
        goto LABEL_183;
      v79 = *(_QWORD **)(a1 + 40);
      v75 = *(unsigned int *)(a1 + 52);
      v78 = (__int64)&v79[v75];
      if ( v79 == (_QWORD *)v78 )
      {
LABEL_184:
        if ( (unsigned int)v75 < *(_DWORD *)(a1 + 48) )
        {
          v75 = (unsigned int)(v75 + 1);
          *(_DWORD *)(a1 + 52) = v75;
          *(_QWORD *)v78 = a2;
          ++*(_QWORD *)(a1 + 32);
          goto LABEL_164;
        }
LABEL_183:
        sub_C8CC70(v169, a2, v78, v75, a5, a6);
        goto LABEL_164;
      }
      while ( a2 != *v79 )
      {
        if ( (_QWORD *)v78 == ++v79 )
          goto LABEL_184;
      }
LABEL_164:
      a2 = (__int64)&v178;
      sub_D17810((__int64)&v186, &v178, v78, v75, a5, a6);
LABEL_165:
      v74 += 4;
      if ( v76 != v74 )
        continue;
      break;
    }
    v20 = v165;
LABEL_167:
    for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v20 + 32) )
    {
      v80 = v20 - 24;
      if ( !v20 )
        v80 = 0;
      if ( j != v80 + 48 )
        break;
      v20 = *(_QWORD *)(v20 + 8);
      if ( v69 == v20 )
        goto LABEL_33;
      if ( !v20 )
        BUG();
    }
    if ( v69 != v20 )
      continue;
    break;
  }
LABEL_33:
  v22 = (unsigned int)v191;
  if ( !(_DWORD)v191 )
    goto LABEL_73;
  while ( 2 )
  {
    a2 = v187;
    v23 = *(_QWORD *)&v190[8 * v22 - 8];
    if ( (_DWORD)v189 )
    {
      v24 = (v189 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v25 = (__int64 *)(v187 + 8LL * v24);
      v26 = *v25;
      if ( v23 == *v25 )
      {
LABEL_36:
        *v25 = -8192;
        LODWORD(v188) = v188 - 1;
        ++HIDWORD(v188);
      }
      else
      {
        v68 = 1;
        while ( v26 != -4096 )
        {
          a5 = (unsigned int)(v68 + 1);
          v24 = (v189 - 1) & (v68 + v24);
          v25 = (__int64 *)(v187 + 8LL * v24);
          v26 = *v25;
          if ( v23 == *v25 )
            goto LABEL_36;
          v68 = a5;
        }
      }
    }
    LODWORD(v191) = v191 - 1;
    v173 = 1;
    v27 = *(_QWORD *)(v23 + 8);
    v172 = 0;
    v28 = *(unsigned __int8 *)(v27 + 8);
    if ( (unsigned int)(v28 - 17) <= 1 )
      LOBYTE(v28) = *(_BYTE *)(**(_QWORD **)(v27 + 16) + 8LL);
    if ( (_BYTE)v28 != 12 )
    {
LABEL_40:
      v29 = 0;
      goto LABEL_41;
    }
    v59 = *(_DWORD *)(a1 + 344);
    if ( !v59 )
    {
      ++*(_QWORD *)(a1 + 320);
LABEL_301:
      sub_D175E0(v157, 2 * v59);
      v140 = *(_DWORD *)(a1 + 344);
      if ( !v140 )
      {
LABEL_378:
        ++*(_DWORD *)(a1 + 336);
        BUG();
      }
      a5 = (unsigned int)(v140 - 1);
      a6 = *(_QWORD *)(a1 + 328);
      v27 = (unsigned int)a5 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v108 = *(_DWORD *)(a1 + 336) + 1;
      v62 = a6 + 24 * v27;
      v141 = *(_QWORD *)v62;
      if ( *(_QWORD *)v62 != v23 )
      {
        v142 = 1;
        v143 = 0;
        while ( v141 != -4096 )
        {
          if ( v141 == -8192 && !v143 )
            v143 = v62;
          v27 = (unsigned int)a5 & (v142 + (_DWORD)v27);
          v62 = a6 + 24LL * (unsigned int)v27;
          v141 = *(_QWORD *)v62;
          if ( v23 == *(_QWORD *)v62 )
            goto LABEL_241;
          ++v142;
        }
        if ( v143 )
          v62 = v143;
      }
      goto LABEL_241;
    }
    a6 = v59 - 1;
    v60 = 1;
    a5 = *(_QWORD *)(a1 + 328);
    v27 = (unsigned int)a6 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v61 = a5 + 24 * v27;
    v62 = 0;
    v63 = *(_QWORD *)v61;
    if ( *(_QWORD *)v61 == v23 )
    {
LABEL_132:
      v64 = *(_DWORD *)(v61 + 16);
      a2 = v61 + 8;
      if ( v64 > 0x40 )
        goto LABEL_133;
LABEL_245:
      v110 = *(const void **)a2;
      v173 = v64;
      v172 = v110;
LABEL_246:
      v66 = v110 == 0;
      goto LABEL_135;
    }
    while ( v63 != -4096 )
    {
      if ( !v62 && v63 == -8192 )
        v62 = v61;
      v27 = (unsigned int)a6 & (v60 + (_DWORD)v27);
      v61 = a5 + 24LL * (unsigned int)v27;
      v63 = *(_QWORD *)v61;
      if ( v23 == *(_QWORD *)v61 )
        goto LABEL_132;
      ++v60;
    }
    v107 = *(_DWORD *)(a1 + 336);
    if ( !v62 )
      v62 = v61;
    ++*(_QWORD *)(a1 + 320);
    v108 = v107 + 1;
    if ( 4 * (v107 + 1) >= 3 * v59 )
      goto LABEL_301;
    v27 = v59 - *(_DWORD *)(a1 + 340) - v108;
    if ( (unsigned int)v27 <= v59 >> 3 )
    {
      sub_D175E0(v157, v59);
      v136 = *(_DWORD *)(a1 + 344);
      if ( !v136 )
        goto LABEL_378;
      a5 = (unsigned int)(v136 - 1);
      a6 = *(_QWORD *)(a1 + 328);
      v27 = 1;
      v137 = a5 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v108 = *(_DWORD *)(a1 + 336) + 1;
      v138 = 0;
      v62 = a6 + 24LL * v137;
      v139 = *(_QWORD *)v62;
      if ( v23 != *(_QWORD *)v62 )
      {
        while ( v139 != -4096 )
        {
          if ( !v138 && v139 == -8192 )
            v138 = v62;
          v137 = a5 & (v27 + v137);
          v62 = a6 + 24LL * v137;
          v139 = *(_QWORD *)v62;
          if ( v23 == *(_QWORD *)v62 )
            goto LABEL_241;
          v27 = (unsigned int)(v27 + 1);
        }
        if ( v138 )
          v62 = v138;
      }
    }
LABEL_241:
    *(_DWORD *)(a1 + 336) = v108;
    if ( *(_QWORD *)v62 != -4096 )
      --*(_DWORD *)(a1 + 340);
    *(_QWORD *)v62 = v23;
    a2 = v62 + 8;
    *(_QWORD *)(v62 + 8) = 0;
    v109 = v173 <= 0x40;
    *(_DWORD *)(v62 + 16) = 1;
    if ( v109 )
    {
      v64 = 1;
      goto LABEL_245;
    }
LABEL_133:
    sub_C43990((__int64)&v172, a2);
    v65 = v173;
    if ( v173 <= 0x40 )
    {
      v110 = v172;
      goto LABEL_246;
    }
    v66 = v65 == (unsigned int)sub_C444A0((__int64)&v172);
LABEL_135:
    if ( !v66 )
      goto LABEL_40;
    v29 = sub_D14860((unsigned __int8 *)v23) ^ 1;
LABEL_41:
    v179 = 1;
    v178 = 0;
    v181 = 1;
    v180 = 0;
    v183 = 1;
    v182 = 0;
    v185 = 1;
    v184 = 0;
    v170 = 0;
    if ( (*(_BYTE *)(v23 + 7) & 0x40) != 0 )
    {
      v30 = *(__int64 **)(v23 - 8);
      v31 = &v30[4 * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF)];
    }
    else
    {
      v31 = (__int64 *)v23;
      v30 = (__int64 *)(v23 - 32LL * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF));
    }
    v32 = v31;
    if ( v30 == v31 )
      goto LABEL_60;
    v168 = v23;
    while ( 1 )
    {
LABEL_45:
      a2 = *v30;
      if ( *(_BYTE *)*v30 <= 0x1Cu )
      {
        v171 = 0;
        v33 = *v30;
        a2 = 0;
        if ( *(_BYTE *)*v30 != 22 )
        {
          v30 += 4;
          if ( v32 == v30 )
            goto LABEL_57;
          continue;
        }
      }
      else
      {
        v171 = *v30;
        v33 = a2;
      }
      v34 = *(_QWORD *)(v33 + 8);
      v35 = *(unsigned __int8 *)(v34 + 8);
      if ( (unsigned int)(v35 - 17) <= 1 )
        v35 = *(unsigned __int8 *)(**(_QWORD **)(v34 + 16) + 8LL);
      if ( (_BYTE)v35 == 12 )
        break;
      if ( !a2 )
        goto LABEL_56;
      if ( !*(_BYTE *)(a1 + 60) )
        goto LABEL_79;
      v36 = *(_QWORD **)(a1 + 40);
      v27 = *(unsigned int *)(a1 + 52);
      v35 = (__int64)&v36[v27];
      if ( v36 != (_QWORD *)v35 )
      {
        while ( *v36 != a2 )
        {
          if ( (_QWORD *)v35 == ++v36 )
            goto LABEL_103;
        }
        goto LABEL_56;
      }
LABEL_103:
      if ( (unsigned int)v27 < *(_DWORD *)(a1 + 48) )
      {
        v27 = (unsigned int)(v27 + 1);
        *(_DWORD *)(a1 + 52) = v27;
        *(_QWORD *)v35 = a2;
        ++*(_QWORD *)(a1 + 32);
      }
      else
      {
LABEL_79:
        sub_C8CC70(v169, a2, v35, v27, a5, a6);
        if ( !(_BYTE)v35 )
          goto LABEL_56;
      }
      a2 = (__int64)&v171;
      v30 += 4;
      sub_D17810((__int64)&v186, &v171, v35, v27, a5, a6);
      if ( v32 == v30 )
        goto LABEL_57;
    }
    v38 = sub_BCB060(v34);
    v175 = v38;
    if ( v38 > 0x40 )
    {
      v161 = v38;
      sub_C43690((__int64)&v174, -1, 1);
      if ( !v29 )
        goto LABEL_111;
      a2 = 0;
      v177 = v161;
      sub_C43690((__int64)&v176, 0, 0);
      if ( v175 > 0x40 && v174 )
        j_j___libc_free_0_0(v174);
      v40 = v176;
      v38 = v177;
LABEL_87:
      v174 = v40;
      v175 = v38;
    }
    else
    {
      v39 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v38;
      if ( !v38 )
        v39 = 0;
      v174 = v39;
      if ( v29 )
      {
        v40 = 0;
        goto LABEL_87;
      }
LABEL_111:
      v50 = sub_BD2910((__int64)v30);
      a2 = v168;
      sub_D15BF0(a1, v168, *v30, v50, (__int64)&v172, &v174, (__int64)&v178, (__int64)&v182, (__int64)&v170);
      v52 = v175;
      if ( v175 <= 0x40 )
      {
        if ( v174 )
          goto LABEL_121;
LABEL_113:
        if ( *(_BYTE *)(a1 + 380) )
        {
          v54 = *(__int64 ***)(a1 + 360);
          v51 = *(unsigned int *)(a1 + 372);
          v52 = (__int64)&v54[v51];
          if ( v54 != (__int64 **)v52 )
          {
            while ( *v54 != v30 )
            {
              if ( (__int64 **)v52 == ++v54 )
                goto LABEL_117;
            }
            goto LABEL_88;
          }
LABEL_117:
          if ( (unsigned int)v51 < *(_DWORD *)(a1 + 368) )
          {
            *(_DWORD *)(a1 + 372) = v51 + 1;
            *(_QWORD *)v52 = v30;
            ++*(_QWORD *)(a1 + 352);
            goto LABEL_88;
          }
        }
        a2 = (__int64)v30;
        sub_C8CC70(v158, (__int64)v30, v52, v51, a5, a6);
        goto LABEL_88;
      }
      v162 = v175;
      v53 = sub_C444A0((__int64)&v174);
      v52 = v162;
      if ( v162 == v53 )
        goto LABEL_113;
LABEL_121:
      if ( *(_BYTE *)(a1 + 380) )
      {
        a2 = *(_QWORD *)(a1 + 360);
        v55 = (__int64 **)(a2 + 8LL * *(unsigned int *)(a1 + 372));
        v56 = (__int64 **)a2;
        if ( (__int64 **)a2 != v55 )
        {
          while ( *v56 != v30 )
          {
            if ( v55 == ++v56 )
              goto LABEL_88;
          }
          v57 = (unsigned int)(*(_DWORD *)(a1 + 372) - 1);
          *(_DWORD *)(a1 + 372) = v57;
          *v56 = *(__int64 **)(a2 + 8 * v57);
          ++*(_QWORD *)(a1 + 352);
        }
      }
      else
      {
        a2 = (__int64)v30;
        v67 = sub_C8CA60(v158, (__int64)v30);
        if ( v67 )
        {
          *v67 = -2;
          ++*(_DWORD *)(a1 + 376);
          ++*(_QWORD *)(a1 + 352);
        }
      }
    }
LABEL_88:
    v41 = v171;
    if ( !v171 )
    {
LABEL_98:
      v27 = v175;
      goto LABEL_99;
    }
    a2 = *(unsigned int *)(a1 + 344);
    if ( !(_DWORD)a2 )
    {
      ++*(_QWORD *)(a1 + 320);
      goto LABEL_279;
    }
    a6 = (unsigned int)(a2 - 1);
    a5 = *(_QWORD *)(a1 + 328);
    v42 = 1;
    v43 = 0;
    LODWORD(v44) = a6 & (((unsigned int)v171 >> 9) ^ ((unsigned int)v171 >> 4));
    v45 = a5 + 24LL * (unsigned int)v44;
    v46 = *(_QWORD *)v45;
    if ( v171 != *(_QWORD *)v45 )
    {
      while ( v46 != -4096 )
      {
        if ( !v43 && v46 == -8192 )
          v43 = v45;
        v44 = (unsigned int)a6 & ((_DWORD)v44 + v42);
        v45 = a5 + 24 * v44;
        v46 = *(_QWORD *)v45;
        if ( v171 == *(_QWORD *)v45 )
          goto LABEL_91;
        ++v42;
      }
      v106 = *(_DWORD *)(a1 + 336);
      if ( v43 )
        v45 = v43;
      ++*(_QWORD *)(a1 + 320);
      v27 = (unsigned int)(v106 + 1);
      if ( 4 * (int)v27 < (unsigned int)(3 * a2) )
      {
        a5 = (unsigned int)a2 >> 3;
        if ( (int)a2 - *(_DWORD *)(a1 + 340) - (int)v27 <= (unsigned int)a5 )
        {
          sub_D175E0(v157, a2);
          v131 = *(_DWORD *)(a1 + 344);
          if ( !v131 )
          {
LABEL_376:
            ++*(_DWORD *)(a1 + 336);
            BUG();
          }
          v41 = v171;
          v132 = v131 - 1;
          v133 = 0;
          v134 = *(_QWORD *)(a1 + 328);
          a5 = 1;
          v27 = (unsigned int)(*(_DWORD *)(a1 + 336) + 1);
          v135 = (v131 - 1) & (((unsigned int)v171 >> 9) ^ ((unsigned int)v171 >> 4));
          v45 = v134 + 24LL * v135;
          a6 = *(_QWORD *)v45;
          if ( *(_QWORD *)v45 != v171 )
          {
            while ( a6 != -4096 )
            {
              if ( !v133 && a6 == -8192 )
                v133 = v45;
              v155 = a5 + 1;
              a5 = v135 + (unsigned int)a5;
              v135 = v132 & a5;
              v45 = v134 + 24LL * (v132 & (unsigned int)a5);
              a6 = *(_QWORD *)v45;
              if ( v171 == *(_QWORD *)v45 )
                goto LABEL_228;
              a5 = v155;
            }
            if ( v133 )
              v45 = v133;
          }
        }
        goto LABEL_228;
      }
LABEL_279:
      sub_D175E0(v157, 2 * a2);
      v126 = *(_DWORD *)(a1 + 344);
      if ( !v126 )
        goto LABEL_376;
      a5 = v171;
      v127 = v126 - 1;
      v128 = *(_QWORD *)(a1 + 328);
      v27 = (unsigned int)(*(_DWORD *)(a1 + 336) + 1);
      v129 = (v126 - 1) & (((unsigned int)v171 >> 9) ^ ((unsigned int)v171 >> 4));
      v45 = v128 + 24LL * v129;
      v41 = *(_QWORD *)v45;
      if ( v171 != *(_QWORD *)v45 )
      {
        a6 = 1;
        v130 = 0;
        while ( v41 != -4096 )
        {
          if ( !v130 && v41 == -8192 )
            v130 = v45;
          v154 = a6 + 1;
          a6 = v129 + (unsigned int)a6;
          v129 = v127 & a6;
          v45 = v128 + 24LL * (v127 & (unsigned int)a6);
          v41 = *(_QWORD *)v45;
          if ( v171 == *(_QWORD *)v45 )
            goto LABEL_228;
          a6 = v154;
        }
        v41 = v171;
        if ( v130 )
          v45 = v130;
      }
LABEL_228:
      *(_DWORD *)(a1 + 336) = v27;
      if ( *(_QWORD *)v45 != -4096 )
        --*(_DWORD *)(a1 + 340);
      *(_QWORD *)v45 = v41;
      *(_DWORD *)(v45 + 16) = 1;
      *(_QWORD *)(v45 + 8) = 0;
LABEL_97:
      a2 = (__int64)&v171;
      *(_QWORD *)(v45 + 8) = v174;
      v49 = v175;
      v175 = 0;
      *(_DWORD *)(v45 + 16) = v49;
      sub_D17810((__int64)&v186, &v171, v45, v27, a5, a6);
      goto LABEL_98;
    }
LABEL_91:
    if ( v175 <= 0x40 )
    {
      v27 = *(_QWORD *)(v45 + 8);
      v47 = v27 | v174;
      v174 |= v27;
      goto LABEL_93;
    }
    v159 = v45;
    v163 = v45 + 8;
    sub_C43BD0(&v174, (__int64 *)(v45 + 8));
    a2 = v163;
    v45 = v159;
    if ( v175 <= 0x40 )
    {
      v47 = v174;
      v27 = *(_QWORD *)(v159 + 8);
LABEL_93:
      if ( v27 == v47 )
        goto LABEL_98;
LABEL_94:
      if ( *(_DWORD *)(v45 + 16) > 0x40u )
      {
        v48 = *(_QWORD *)(v45 + 8);
        if ( v48 )
        {
          v160 = v45;
          j_j___libc_free_0_0(v48);
          v45 = v160;
        }
      }
      goto LABEL_97;
    }
    v164 = v175;
    v58 = sub_C43C50((__int64)&v174, (const void **)a2);
    v27 = v164;
    if ( !v58 )
    {
      v45 = v159;
      goto LABEL_94;
    }
LABEL_99:
    if ( (unsigned int)v27 <= 0x40 || !v174 )
    {
LABEL_56:
      v30 += 4;
      if ( v32 == v30 )
        goto LABEL_57;
      goto LABEL_45;
    }
    j_j___libc_free_0_0(v174);
    v30 += 4;
    if ( v32 != v30 )
      goto LABEL_45;
LABEL_57:
    if ( v185 > 0x40 && v184 )
      j_j___libc_free_0_0(v184);
LABEL_60:
    if ( v183 > 0x40 && v182 )
      j_j___libc_free_0_0(v182);
    if ( v181 > 0x40 && v180 )
      j_j___libc_free_0_0(v180);
    if ( v179 > 0x40 && v178 )
      j_j___libc_free_0_0(v178);
    if ( v173 > 0x40 && v172 )
      j_j___libc_free_0_0(v172);
    v22 = (unsigned int)v191;
    if ( (_DWORD)v191 )
      continue;
    break;
  }
LABEL_73:
  if ( v190 != v192 )
    _libc_free(v190, a2);
  return sub_C7D6A0(v187, 8LL * (unsigned int)v189, 8);
}
