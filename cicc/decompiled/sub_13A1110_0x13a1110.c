// Function: sub_13A1110
// Address: 0x13a1110
//
void __fastcall sub_13A1110(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r15
  void *v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  int v6; // r13d
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r12
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r10
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // r13
  __int64 *v17; // rax
  __int64 *v18; // r12
  unsigned int v19; // r14d
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rsi
  int v23; // r8d
  unsigned int v24; // edx
  __int64 *v25; // r14
  __int64 v26; // rcx
  unsigned int v27; // ecx
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 v34; // rbx
  _QWORD *v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  char v40; // al
  char v41; // r8
  __int64 v42; // rdi
  __int64 *v43; // r14
  __int64 v44; // rcx
  __int64 v45; // rdi
  unsigned int v46; // eax
  __int64 v47; // rax
  bool v48; // al
  _QWORD *v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 *v53; // rsi
  unsigned int v54; // edi
  __int64 *v55; // rcx
  unsigned int v56; // esi
  __int64 v57; // r8
  unsigned int v58; // edi
  __int64 *v59; // rdx
  __int64 v60; // rcx
  unsigned int v61; // eax
  __int64 *v62; // rsi
  int v63; // r11d
  __int64 *v64; // r9
  int v65; // eax
  int v66; // eax
  int v67; // eax
  int v68; // esi
  __int64 v69; // rdi
  __int64 v70; // rdx
  __int64 v71; // rcx
  int v72; // r9d
  __int64 *v73; // r8
  __int64 j; // r12
  __int64 v75; // r9
  __int64 v76; // r15
  __int64 v77; // r13
  char v78; // al
  unsigned int v79; // esi
  unsigned int v80; // r14d
  __int64 v81; // rdi
  unsigned int v82; // edx
  __int64 *v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rsi
  __int64 v89; // r13
  __int64 v90; // r15
  __int64 v91; // r12
  unsigned int v92; // esi
  __int64 v93; // rdi
  unsigned int v94; // edx
  __int64 *v95; // r14
  __int64 v96; // rcx
  __int64 v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rbx
  unsigned int v100; // ecx
  int v101; // eax
  int v102; // r9d
  __int64 v103; // r10
  __int64 v104; // rax
  int v105; // edx
  __int64 v106; // rsi
  int v107; // r8d
  __int64 *v108; // rcx
  int v109; // ecx
  int v110; // ecx
  int v111; // edi
  __int64 *v112; // rsi
  __int64 v113; // r9
  int v114; // edx
  __int64 v115; // r8
  __int64 v116; // rdi
  int v117; // r10d
  __int64 *v118; // r9
  int v119; // edi
  unsigned __int64 v120; // rdx
  int v121; // edx
  int v122; // r9d
  __int64 v123; // r10
  int v124; // r8d
  __int64 v125; // rax
  __int64 v126; // rsi
  __int64 v127; // rdx
  int v128; // ebx
  unsigned int v129; // eax
  _QWORD *v130; // rdi
  unsigned __int64 v131; // rdx
  unsigned __int64 v132; // rax
  _QWORD *v133; // rax
  __int64 v134; // rdx
  _QWORD *i; // rdx
  __int64 *v136; // r10
  int v137; // edi
  int v138; // edx
  __int64 v139; // rax
  int v140; // r10d
  __int64 *v141; // rax
  int v142; // edi
  int v143; // edx
  bool v144; // cc
  int v145; // r8d
  int v146; // r8d
  __int64 v147; // r9
  unsigned int v148; // ecx
  __int64 v149; // r11
  int v150; // edi
  __int64 *v151; // rsi
  int v152; // eax
  __int64 v153; // r10
  unsigned int v154; // ecx
  __int64 v155; // r8
  int v156; // edi
  __int64 *v157; // rsi
  int v158; // r8d
  int v159; // r8d
  __int64 v160; // r9
  int v161; // ecx
  unsigned int v162; // ebx
  __int64 *v163; // rdi
  __int64 v164; // rsi
  int v165; // r10d
  int v166; // r10d
  __int64 *v167; // rcx
  unsigned int v168; // r8d
  __int64 v169; // rdi
  int v170; // esi
  _QWORD *v171; // rax
  int v172; // r10d
  __int64 v173; // rdi
  __int64 v174; // [rsp+8h] [rbp-508h]
  __int64 v175; // [rsp+10h] [rbp-500h]
  __int64 v176; // [rsp+20h] [rbp-4F0h]
  unsigned int v177; // [rsp+28h] [rbp-4E8h]
  __int64 v178; // [rsp+28h] [rbp-4E8h]
  unsigned int v179; // [rsp+28h] [rbp-4E8h]
  int v180; // [rsp+28h] [rbp-4E8h]
  __int64 v181; // [rsp+28h] [rbp-4E8h]
  __int64 v182; // [rsp+28h] [rbp-4E8h]
  __int64 *v183; // [rsp+30h] [rbp-4E0h]
  __int64 v184; // [rsp+38h] [rbp-4D8h]
  unsigned int v185; // [rsp+38h] [rbp-4D8h]
  __int64 v186; // [rsp+38h] [rbp-4D8h]
  __int64 v187; // [rsp+38h] [rbp-4D8h]
  __int64 v188; // [rsp+38h] [rbp-4D8h]
  int v189; // [rsp+38h] [rbp-4D8h]
  __int64 v190; // [rsp+38h] [rbp-4D8h]
  unsigned __int64 v191; // [rsp+40h] [rbp-4D0h] BYREF
  unsigned int v192; // [rsp+48h] [rbp-4C8h]
  unsigned __int64 v193; // [rsp+50h] [rbp-4C0h] BYREF
  unsigned int v194; // [rsp+58h] [rbp-4B8h]
  unsigned __int64 v195; // [rsp+60h] [rbp-4B0h] BYREF
  unsigned int v196; // [rsp+68h] [rbp-4A8h]
  unsigned __int64 v197; // [rsp+70h] [rbp-4A0h] BYREF
  unsigned int v198; // [rsp+78h] [rbp-498h]
  unsigned __int64 v199; // [rsp+80h] [rbp-490h] BYREF
  unsigned int v200; // [rsp+88h] [rbp-488h]
  __int64 v201; // [rsp+90h] [rbp-480h] BYREF
  __int64 v202; // [rsp+98h] [rbp-478h]
  __int64 v203; // [rsp+A0h] [rbp-470h]
  __int64 v204; // [rsp+A8h] [rbp-468h]
  unsigned __int64 v205; // [rsp+B0h] [rbp-460h] BYREF
  __int64 v206; // [rsp+B8h] [rbp-458h]
  __int64 v207; // [rsp+C0h] [rbp-450h]
  __int64 v208; // [rsp+C8h] [rbp-448h]
  _BYTE *v209; // [rsp+D0h] [rbp-440h] BYREF
  __int64 v210; // [rsp+D8h] [rbp-438h]
  _BYTE v211[1072]; // [rsp+E0h] [rbp-430h] BYREF

  v1 = a1 + 32;
  v2 = a1;
  *(_BYTE *)(a1 + 24) = 1;
  ++*(_QWORD *)(a1 + 32);
  v3 = *(void **)(a1 + 48);
  v176 = v1;
  if ( v3 != *(void **)(v2 + 40) )
  {
    v4 = 4 * (*(_DWORD *)(v2 + 60) - *(_DWORD *)(v2 + 64));
    v5 = *(unsigned int *)(v2 + 56);
    if ( v4 < 0x20 )
      v4 = 32;
    if ( (unsigned int)v5 > v4 )
    {
      sub_16CC920(v176);
      goto LABEL_7;
    }
    memset(v3, -1, 8 * v5);
  }
  *(_QWORD *)(v2 + 60) = 0;
LABEL_7:
  v6 = *(_DWORD *)(v2 + 344);
  ++*(_QWORD *)(v2 + 328);
  v174 = v2 + 328;
  if ( !v6 && !*(_DWORD *)(v2 + 348) )
    goto LABEL_21;
  v7 = *(_QWORD *)(v2 + 336);
  v8 = *(unsigned int *)(v2 + 352);
  v9 = v7 + 24 * v8;
  v10 = 4 * v6;
  if ( (unsigned int)(4 * v6) < 0x40 )
    v10 = 64;
  if ( (unsigned int)v8 <= v10 )
  {
    for ( ; v7 != v9; v7 += 24 )
    {
      if ( *(_QWORD *)v7 != -8 )
      {
        if ( *(_QWORD *)v7 != -16 && *(_DWORD *)(v7 + 16) > 0x40u )
        {
          v11 = *(_QWORD *)(v7 + 8);
          if ( v11 )
            j_j___libc_free_0_0(v11);
        }
        *(_QWORD *)v7 = -8;
      }
    }
    goto LABEL_20;
  }
  do
  {
    while ( *(_QWORD *)v7 == -8 )
    {
LABEL_220:
      v7 += 24;
      if ( v7 == v9 )
        goto LABEL_246;
    }
    if ( *(_QWORD *)v7 != -16 )
    {
      if ( *(_DWORD *)(v7 + 16) > 0x40u )
      {
        v116 = *(_QWORD *)(v7 + 8);
        if ( v116 )
          j_j___libc_free_0_0(v116);
      }
      goto LABEL_220;
    }
    v7 += 24;
  }
  while ( v7 != v9 );
LABEL_246:
  v127 = *(unsigned int *)(v2 + 352);
  if ( !v6 )
  {
    if ( (_DWORD)v127 )
    {
      j___libc_free_0(*(_QWORD *)(v2 + 336));
      *(_QWORD *)(v2 + 336) = 0;
      *(_QWORD *)(v2 + 344) = 0;
      *(_DWORD *)(v2 + 352) = 0;
      goto LABEL_21;
    }
LABEL_20:
    *(_QWORD *)(v2 + 344) = 0;
    goto LABEL_21;
  }
  v128 = 64;
  if ( v6 != 1 )
  {
    _BitScanReverse(&v129, v6 - 1);
    v128 = 1 << (33 - (v129 ^ 0x1F));
    if ( v128 < 64 )
      v128 = 64;
  }
  v130 = *(_QWORD **)(v2 + 336);
  if ( (_DWORD)v127 == v128 )
  {
    *(_QWORD *)(v2 + 344) = 0;
    v171 = &v130[3 * v127];
    do
    {
      if ( v130 )
        *v130 = -8;
      v130 += 3;
    }
    while ( v171 != v130 );
  }
  else
  {
    j___libc_free_0(v130);
    v131 = ((((((((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
              | (4 * v128 / 3u + 1)
              | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 4)
            | (((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
            | (4 * v128 / 3u + 1)
            | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
            | (4 * v128 / 3u + 1)
            | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 4)
          | (((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
          | (4 * v128 / 3u + 1)
          | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 16;
    v132 = (v131
          | (((((((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
              | (4 * v128 / 3u + 1)
              | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 4)
            | (((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
            | (4 * v128 / 3u + 1)
            | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
            | (4 * v128 / 3u + 1)
            | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 4)
          | (((4 * v128 / 3u + 1) | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1)) >> 2)
          | (4 * v128 / 3u + 1)
          | ((unsigned __int64)(4 * v128 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(v2 + 352) = v132;
    v133 = (_QWORD *)sub_22077B0(24 * v132);
    v134 = *(unsigned int *)(v2 + 352);
    *(_QWORD *)(v2 + 344) = 0;
    *(_QWORD *)(v2 + 336) = v133;
    for ( i = &v133[3 * v134]; i != v133; v133 += 3 )
    {
      if ( v133 )
        *v133 = -8;
    }
  }
LABEL_21:
  v209 = v211;
  v210 = 0x8000000000LL;
  v12 = *(_QWORD *)(*(_QWORD *)v2 + 80LL);
  v13 = *(_QWORD *)v2 + 72LL;
  if ( v13 == v12 )
  {
    v14 = 0;
  }
  else
  {
    if ( !v12 )
      BUG();
    while ( 1 )
    {
      v14 = *(_QWORD *)(v12 + 24);
      if ( v14 != v12 + 16 )
        break;
      v12 = *(_QWORD *)(v12 + 8);
      if ( v13 == v12 )
        goto LABEL_27;
      if ( !v12 )
        BUG();
    }
  }
  if ( v13 != v12 )
  {
    j = v14;
    v75 = v2;
    v76 = *(_QWORD *)v2 + 72LL;
    while ( 1 )
    {
      v77 = j - 24;
      v184 = v75;
      if ( !j )
        v77 = 0;
      v78 = sub_139F030(v77);
      v75 = v184;
      if ( !v78 )
        goto LABEL_174;
      if ( *(_BYTE *)(*(_QWORD *)v77 + 8LL) != 11 )
        break;
      v79 = *(_DWORD *)(v184 + 352);
      v80 = *(_DWORD *)(*(_QWORD *)v77 + 8LL) >> 8;
      if ( !v79 )
      {
        ++*(_QWORD *)(v184 + 328);
        goto LABEL_295;
      }
      v81 = *(_QWORD *)(v184 + 336);
      v185 = ((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4);
      v82 = (v79 - 1) & v185;
      v83 = (__int64 *)(v81 + 24LL * v82);
      v84 = *v83;
      if ( v77 != *v83 )
      {
        v180 = 1;
        v136 = 0;
        while ( v84 != -8 )
        {
          if ( !v136 && v84 == -16 )
            v136 = v83;
          v82 = (v79 - 1) & (v180 + v82);
          v83 = (__int64 *)(v81 + 24LL * v82);
          v84 = *v83;
          if ( v77 == *v83 )
            goto LABEL_174;
          ++v180;
        }
        v137 = *(_DWORD *)(v75 + 344);
        if ( v136 )
          v83 = v136;
        ++*(_QWORD *)(v75 + 328);
        v138 = v137 + 1;
        if ( 4 * (v137 + 1) < 3 * v79 )
        {
          if ( v79 - *(_DWORD *)(v75 + 348) - v138 > v79 >> 3 )
          {
LABEL_262:
            *(_DWORD *)(v75 + 344) = v138;
            if ( *v83 != -8 )
              --*(_DWORD *)(v75 + 348);
            *v83 = v77;
            *((_DWORD *)v83 + 4) = v80;
            if ( v80 > 0x40 )
            {
              v187 = v75;
              sub_16A4EF0(v83 + 1, 0, 0);
              v75 = v187;
            }
            else
            {
              v83[1] = 0;
            }
            v139 = (unsigned int)v210;
            if ( (unsigned int)v210 >= HIDWORD(v210) )
            {
              v190 = v75;
              sub_16CD150(&v209, v211, 0, 8);
              v139 = (unsigned int)v210;
              v75 = v190;
            }
            *(_QWORD *)&v209[8 * v139] = v77;
            LODWORD(v210) = v210 + 1;
            goto LABEL_174;
          }
          v181 = v75;
          sub_13A0F10(v174, v79);
          v75 = v181;
          v165 = *(_DWORD *)(v181 + 352);
          if ( v165 )
          {
            v166 = v165 - 1;
            v167 = 0;
            v168 = v166 & v185;
            v182 = *(_QWORD *)(v181 + 336);
            v83 = (__int64 *)(v182 + 24LL * (v166 & v185));
            v169 = *v83;
            v138 = *(_DWORD *)(v75 + 344) + 1;
            v170 = 1;
            if ( v77 != *v83 )
            {
              while ( v169 != -8 )
              {
                if ( v169 == -16 && !v167 )
                  v167 = v83;
                v168 = v166 & (v170 + v168);
                v83 = (__int64 *)(v182 + 24LL * v168);
                v169 = *v83;
                if ( v77 == *v83 )
                  goto LABEL_262;
                ++v170;
              }
              if ( v167 )
                v83 = v167;
            }
            goto LABEL_262;
          }
LABEL_374:
          ++*(_DWORD *)(v75 + 344);
          BUG();
        }
LABEL_295:
        v188 = v75;
        sub_13A0F10(v174, 2 * v79);
        v75 = v188;
        v152 = *(_DWORD *)(v188 + 352);
        if ( v152 )
        {
          v153 = *(_QWORD *)(v188 + 336);
          v189 = v152 - 1;
          v154 = (v152 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v138 = *(_DWORD *)(v75 + 344) + 1;
          v83 = (__int64 *)(v153 + 24LL * v154);
          v155 = *v83;
          if ( v77 != *v83 )
          {
            v156 = 1;
            v157 = 0;
            while ( v155 != -8 )
            {
              if ( !v157 && v155 == -16 )
                v157 = v83;
              v154 = v189 & (v156 + v154);
              v83 = (__int64 *)(v153 + 24LL * v154);
              v155 = *v83;
              if ( v77 == *v83 )
                goto LABEL_262;
              ++v156;
            }
            if ( v157 )
              v83 = v157;
          }
          goto LABEL_262;
        }
        goto LABEL_374;
      }
LABEL_174:
      for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v12 + 24) )
      {
        v85 = v12 - 24;
        if ( !v12 )
          v85 = 0;
        if ( j != v85 + 40 )
          break;
        v12 = *(_QWORD *)(v12 + 8);
        if ( v76 == v12 )
          goto LABEL_182;
        if ( !v12 )
          BUG();
      }
      if ( v76 == v12 )
      {
LABEL_182:
        v2 = v75;
        goto LABEL_27;
      }
    }
    v86 = 24LL * (*(_DWORD *)(v77 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v77 + 23) & 0x40) != 0 )
    {
      v87 = *(_QWORD *)(v77 - 8);
      v88 = v87 + v86;
    }
    else
    {
      v88 = v77;
      v87 = v77 - v86;
    }
    if ( v87 == v88 )
      goto LABEL_174;
    v186 = v76;
    v89 = v87;
    v90 = v88;
    v175 = j;
    v91 = v75;
    v178 = v12;
    while ( 2 )
    {
      v99 = *(_QWORD *)v89;
      if ( *(_BYTE *)(*(_QWORD *)v89 + 16LL) <= 0x17u )
        goto LABEL_196;
      if ( *(_BYTE *)(*(_QWORD *)v99 + 8LL) != 11 )
        goto LABEL_193;
      v100 = *(_DWORD *)(*(_QWORD *)v99 + 8LL) >> 8;
      LODWORD(v206) = v100;
      if ( v100 <= 0x40 )
      {
        v92 = *(_DWORD *)(v91 + 352);
        v205 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v100;
        if ( !v92 )
          goto LABEL_201;
      }
      else
      {
        sub_16A4EF0(&v205, -1, 1);
        v92 = *(_DWORD *)(v91 + 352);
        if ( !v92 )
        {
LABEL_201:
          ++*(_QWORD *)(v91 + 328);
          goto LABEL_202;
        }
      }
      v93 = *(_QWORD *)(v91 + 336);
      v94 = (v92 - 1) & (((unsigned int)v99 >> 4) ^ ((unsigned int)v99 >> 9));
      v95 = (__int64 *)(v93 + 24LL * v94);
      v96 = *v95;
      if ( v99 == *v95 )
      {
LABEL_189:
        if ( *((_DWORD *)v95 + 4) > 0x40u )
        {
          v97 = v95[1];
          if ( v97 )
            j_j___libc_free_0_0(v97);
        }
LABEL_192:
        v95[1] = v205;
        *((_DWORD *)v95 + 4) = v206;
LABEL_193:
        v98 = (unsigned int)v210;
        if ( (unsigned int)v210 >= HIDWORD(v210) )
        {
          sub_16CD150(&v209, v211, 0, 8);
          v98 = (unsigned int)v210;
        }
        *(_QWORD *)&v209[8 * v98] = v99;
        LODWORD(v210) = v210 + 1;
LABEL_196:
        v89 += 24;
        if ( v90 == v89 )
        {
          v75 = v91;
          v76 = v186;
          v12 = v178;
          j = v175;
          goto LABEL_174;
        }
        continue;
      }
      break;
    }
    v117 = 1;
    v118 = 0;
    while ( v96 != -8 )
    {
      if ( v96 == -16 && !v118 )
        v118 = v95;
      v94 = (v92 - 1) & (v117 + v94);
      v95 = (__int64 *)(v93 + 24LL * v94);
      v96 = *v95;
      if ( v99 == *v95 )
        goto LABEL_189;
      ++v117;
    }
    v119 = *(_DWORD *)(v91 + 344);
    if ( v118 )
      v95 = v118;
    ++*(_QWORD *)(v91 + 328);
    v105 = v119 + 1;
    if ( 4 * (v119 + 1) >= 3 * v92 )
    {
LABEL_202:
      sub_13A0F10(v174, 2 * v92);
      v101 = *(_DWORD *)(v91 + 352);
      if ( !v101 )
        goto LABEL_376;
      v102 = v101 - 1;
      v103 = *(_QWORD *)(v91 + 336);
      LODWORD(v104) = (v101 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
      v95 = (__int64 *)(v103 + 24LL * (unsigned int)v104);
      v105 = *(_DWORD *)(v91 + 344) + 1;
      v106 = *v95;
      if ( v99 != *v95 )
      {
        v107 = 1;
        v108 = 0;
        while ( v106 != -8 )
        {
          if ( v106 == -16 && !v108 )
            v108 = v95;
          v104 = v102 & (unsigned int)(v104 + v107);
          v95 = (__int64 *)(v103 + 24 * v104);
          v106 = *v95;
          if ( v99 == *v95 )
            goto LABEL_231;
          ++v107;
        }
        goto LABEL_206;
      }
    }
    else if ( v92 - *(_DWORD *)(v91 + 348) - v105 <= v92 >> 3 )
    {
      sub_13A0F10(v174, v92);
      v121 = *(_DWORD *)(v91 + 352);
      if ( !v121 )
      {
LABEL_376:
        ++*(_DWORD *)(v91 + 344);
        BUG();
      }
      v122 = v121 - 1;
      v108 = 0;
      v123 = *(_QWORD *)(v91 + 336);
      v124 = 1;
      LODWORD(v125) = (v121 - 1) & (((unsigned int)v99 >> 4) ^ ((unsigned int)v99 >> 9));
      v95 = (__int64 *)(v123 + 24LL * (unsigned int)v125);
      v105 = *(_DWORD *)(v91 + 344) + 1;
      v126 = *v95;
      if ( v99 != *v95 )
      {
        while ( v126 != -8 )
        {
          if ( !v108 && v126 == -16 )
            v108 = v95;
          v125 = v122 & (unsigned int)(v125 + v124);
          v95 = (__int64 *)(v123 + 24 * v125);
          v126 = *v95;
          if ( v99 == *v95 )
            goto LABEL_231;
          ++v124;
        }
LABEL_206:
        if ( v108 )
          v95 = v108;
      }
    }
LABEL_231:
    *(_DWORD *)(v91 + 344) = v105;
    if ( *v95 != -8 )
      --*(_DWORD *)(v91 + 348);
    *v95 = v99;
    *((_DWORD *)v95 + 4) = 1;
    v95[1] = 0;
    goto LABEL_192;
  }
LABEL_27:
  v15 = v210;
  if ( !(_DWORD)v210 )
    goto LABEL_84;
  while ( 2 )
  {
    v16 = *(_QWORD *)&v209[8 * v15 - 8];
    LODWORD(v210) = v15 - 1;
    v192 = 1;
    v191 = 0;
    if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) == 11 )
    {
      v56 = *(_DWORD *)(v2 + 352);
      if ( v56 )
      {
        v57 = *(_QWORD *)(v2 + 336);
        v58 = (v56 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v59 = (__int64 *)(v57 + 24LL * v58);
        v60 = *v59;
        if ( v16 == *v59 )
        {
LABEL_143:
          v61 = *((_DWORD *)v59 + 4);
          v62 = v59 + 1;
          if ( v61 > 0x40 )
          {
LABEL_144:
            sub_16A51C0(&v191, v62);
            goto LABEL_145;
          }
          v120 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v61) & v59[1];
LABEL_237:
          v192 = v61;
          v191 = v120;
LABEL_145:
          if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 11 )
            goto LABEL_29;
          goto LABEL_31;
        }
        v140 = 1;
        v141 = 0;
        while ( v60 != -8 )
        {
          if ( !v141 && v60 == -16 )
            v141 = v59;
          v58 = (v56 - 1) & (v140 + v58);
          v59 = (__int64 *)(v57 + 24LL * v58);
          v60 = *v59;
          if ( v16 == *v59 )
            goto LABEL_143;
          ++v140;
        }
        v142 = *(_DWORD *)(v2 + 344);
        if ( !v141 )
          v141 = v59;
        ++*(_QWORD *)(v2 + 328);
        v143 = v142 + 1;
        if ( 4 * (v142 + 1) < 3 * v56 )
        {
          if ( v56 - *(_DWORD *)(v2 + 348) - v143 <= v56 >> 3 )
          {
            sub_13A0F10(v174, v56);
            v158 = *(_DWORD *)(v2 + 352);
            if ( !v158 )
            {
LABEL_378:
              ++*(_DWORD *)(v2 + 344);
              BUG();
            }
            v159 = v158 - 1;
            v160 = *(_QWORD *)(v2 + 336);
            v161 = 1;
            v162 = v159 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v143 = *(_DWORD *)(v2 + 344) + 1;
            v163 = 0;
            v141 = (__int64 *)(v160 + 24LL * v162);
            v164 = *v141;
            if ( v16 != *v141 )
            {
              while ( v164 != -8 )
              {
                if ( v164 == -16 && !v163 )
                  v163 = v141;
                v162 = v159 & (v161 + v162);
                v141 = (__int64 *)(v160 + 24LL * v162);
                v164 = *v141;
                if ( v16 == *v141 )
                  goto LABEL_276;
                ++v161;
              }
              if ( v163 )
                v141 = v163;
            }
          }
          goto LABEL_276;
        }
      }
      else
      {
        ++*(_QWORD *)(v2 + 328);
      }
      sub_13A0F10(v174, 2 * v56);
      v145 = *(_DWORD *)(v2 + 352);
      if ( !v145 )
        goto LABEL_378;
      v146 = v145 - 1;
      v147 = *(_QWORD *)(v2 + 336);
      v148 = v146 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v143 = *(_DWORD *)(v2 + 344) + 1;
      v141 = (__int64 *)(v147 + 24LL * v148);
      v149 = *v141;
      if ( v16 != *v141 )
      {
        v150 = 1;
        v151 = 0;
        while ( v149 != -8 )
        {
          if ( !v151 && v149 == -16 )
            v151 = v141;
          v148 = v146 & (v150 + v148);
          v141 = (__int64 *)(v147 + 24LL * v148);
          v149 = *v141;
          if ( v16 == *v141 )
            goto LABEL_276;
          ++v150;
        }
        if ( v151 )
          v141 = v151;
      }
LABEL_276:
      *(_DWORD *)(v2 + 344) = v143;
      if ( *v141 != -8 )
        --*(_DWORD *)(v2 + 348);
      *v141 = v16;
      v62 = v141 + 1;
      v141[1] = 0;
      v144 = v192 <= 0x40;
      *((_DWORD *)v141 + 4) = 1;
      if ( !v144 )
        goto LABEL_144;
      v120 = 0;
      v61 = 1;
      goto LABEL_237;
    }
LABEL_29:
    v17 = *(__int64 **)(v2 + 40);
    if ( *(__int64 **)(v2 + 48) != v17 )
      goto LABEL_30;
    v53 = &v17[*(unsigned int *)(v2 + 60)];
    v54 = *(_DWORD *)(v2 + 60);
    if ( v17 == v53 )
    {
LABEL_234:
      if ( v54 >= *(_DWORD *)(v2 + 56) )
      {
LABEL_30:
        sub_16CCBA0(v176, v16);
      }
      else
      {
        *(_DWORD *)(v2 + 60) = v54 + 1;
        *v53 = v16;
        ++*(_QWORD *)(v2 + 32);
      }
    }
    else
    {
      v55 = 0;
      while ( v16 != *v17 )
      {
        if ( *v17 == -2 )
          v55 = v17;
        if ( v53 == ++v17 )
        {
          if ( !v55 )
            goto LABEL_234;
          *v55 = v16;
          --*(_DWORD *)(v2 + 64);
          ++*(_QWORD *)(v2 + 32);
          break;
        }
      }
    }
LABEL_31:
    v201 = 0;
    v202 = 1;
    v203 = 0;
    v204 = 1;
    v205 = 0;
    v206 = 1;
    v207 = 0;
    v208 = 1;
    if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
    {
      v18 = *(__int64 **)(v16 - 8);
      v183 = &v18[3 * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)];
    }
    else
    {
      v183 = (__int64 *)v16;
      v18 = (__int64 *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
    }
    if ( v183 == v18 )
      goto LABEL_71;
    while ( 2 )
    {
      while ( 2 )
      {
        v34 = *v18;
        if ( *(_BYTE *)(*v18 + 16) <= 0x17u )
          goto LABEL_57;
        if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) == 11 )
        {
          v19 = *(_DWORD *)(*(_QWORD *)v34 + 8LL) >> 8;
          v194 = v19;
          if ( v19 > 0x40 )
            sub_16A4EF0(&v193, -1, 1);
          else
            v193 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
          if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) == 11
            && (v192 <= 0x40 ? (v48 = v191 == 0) : (v177 = v192, v48 = v177 == (unsigned int)sub_16A57B0(&v191)),
                v48 && !(unsigned __int8)sub_139F030(v16)) )
          {
            v200 = v19;
            if ( v19 > 0x40 )
              sub_16A4EF0(&v199, 0, 0);
            else
              v199 = 0;
            if ( v194 > 0x40 && v193 )
              j_j___libc_free_0_0(v193);
            v196 = v19;
            v193 = v199;
            v194 = v200;
            if ( v19 > 0x40 )
              goto LABEL_115;
          }
          else
          {
            v20 = sub_1648720(v18);
            sub_139F940(v2, v16, v34, v20, (__int64)&v191, &v193, (__int64)&v201, (__int64)&v205);
            v196 = v19;
            if ( v19 > 0x40 )
            {
LABEL_115:
              sub_16A4EF0(&v195, 0, 0);
LABEL_40:
              v21 = *(unsigned int *)(v2 + 352);
              v22 = *(_QWORD *)(v2 + 336);
              if ( (_DWORD)v21 )
              {
                v23 = 1;
                v24 = (v21 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                v25 = (__int64 *)(v22 + 24LL * v24);
                v26 = *v25;
                if ( v34 == *v25 )
                {
LABEL_42:
                  if ( v25 != (__int64 *)(v22 + 24 * v21) )
                  {
                    if ( v196 <= 0x40 && (v27 = *((_DWORD *)v25 + 4), v27 <= 0x40) )
                    {
                      v52 = v25[1];
                      v196 = *((_DWORD *)v25 + 4);
                      v195 = v52 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v27);
                    }
                    else
                    {
                      sub_16A51C0(&v195, v25 + 1);
                    }
                  }
LABEL_46:
                  v28 = v194;
                  v200 = v194;
                  if ( v194 > 0x40 )
                  {
                    sub_16A4FD0(&v199, &v193);
                    v28 = v200;
                    if ( v200 > 0x40 )
                    {
                      sub_16A89F0(&v199, &v195);
                      v31 = v199;
                      v198 = v200;
                      v197 = v199;
                      if ( v200 > 0x40 )
                      {
                        v40 = sub_16A5220(&v197, &v195);
                        v33 = *(_DWORD *)(v2 + 352);
                        v41 = v40;
                        v32 = *(_QWORD *)(v2 + 336);
                        if ( v41 && v25 != (__int64 *)(v32 + 24LL * v33) )
                        {
LABEL_103:
                          if ( v197 )
                            j_j___libc_free_0_0(v197);
                          goto LABEL_51;
                        }
LABEL_95:
                        if ( v33 )
                        {
                          LODWORD(v42) = (v33 - 1) & (((unsigned int)v34 >> 4) ^ ((unsigned int)v34 >> 9));
                          v43 = (__int64 *)(v32 + 24LL * (unsigned int)v42);
                          v44 = *v43;
                          if ( v34 == *v43 )
                          {
LABEL_97:
                            if ( *((_DWORD *)v43 + 4) > 0x40u )
                            {
                              v45 = v43[1];
                              if ( v45 )
                                j_j___libc_free_0_0(v45);
                            }
                            goto LABEL_100;
                          }
                          v63 = 1;
                          v64 = 0;
                          while ( v44 != -8 )
                          {
                            if ( !v64 && v44 == -16 )
                              v64 = v43;
                            v42 = (v33 - 1) & ((_DWORD)v42 + v63);
                            v43 = (__int64 *)(v32 + 24 * v42);
                            v44 = *v43;
                            if ( v34 == *v43 )
                              goto LABEL_97;
                            ++v63;
                          }
                          v65 = *(_DWORD *)(v2 + 344);
                          if ( v64 )
                            v43 = v64;
                          ++*(_QWORD *)(v2 + 328);
                          v66 = v65 + 1;
                          if ( 4 * v66 < 3 * v33 )
                          {
                            if ( v33 - (v66 + *(_DWORD *)(v2 + 348)) <= v33 >> 3 )
                            {
                              v179 = ((unsigned int)v34 >> 4) ^ ((unsigned int)v34 >> 9);
                              sub_13A0F10(v174, v33);
                              v109 = *(_DWORD *)(v2 + 352);
                              if ( !v109 )
                              {
LABEL_377:
                                ++*(_DWORD *)(v2 + 344);
                                BUG();
                              }
                              v110 = v109 - 1;
                              v111 = 1;
                              v112 = 0;
                              v113 = *(_QWORD *)(v2 + 336);
                              v114 = v110 & v179;
                              v43 = (__int64 *)(v113 + 24LL * (v110 & v179));
                              v115 = *v43;
                              v66 = *(_DWORD *)(v2 + 344) + 1;
                              if ( v34 != *v43 )
                              {
                                while ( v115 != -8 )
                                {
                                  if ( v115 == -16 && !v112 )
                                    v112 = v43;
                                  v172 = v111 + 1;
                                  v173 = v110 & (unsigned int)(v114 + v111);
                                  v114 = v173;
                                  v43 = (__int64 *)(v113 + 24 * v173);
                                  v115 = *v43;
                                  if ( v34 == *v43 )
                                    goto LABEL_153;
                                  v111 = v172;
                                }
                                if ( v112 )
                                  v43 = v112;
                              }
                            }
                            goto LABEL_153;
                          }
                        }
                        else
                        {
                          ++*(_QWORD *)(v2 + 328);
                        }
                        sub_13A0F10(v174, 2 * v33);
                        v67 = *(_DWORD *)(v2 + 352);
                        if ( !v67 )
                          goto LABEL_377;
                        v68 = v67 - 1;
                        v69 = *(_QWORD *)(v2 + 336);
                        LODWORD(v70) = (v67 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                        v43 = (__int64 *)(v69 + 24LL * (unsigned int)v70);
                        v71 = *v43;
                        v66 = *(_DWORD *)(v2 + 344) + 1;
                        if ( v34 != *v43 )
                        {
                          v72 = 1;
                          v73 = 0;
                          while ( v71 != -8 )
                          {
                            if ( !v73 && v71 == -16 )
                              v73 = v43;
                            v70 = v68 & (unsigned int)(v70 + v72);
                            v43 = (__int64 *)(v69 + 24 * v70);
                            v71 = *v43;
                            if ( v34 == *v43 )
                              goto LABEL_153;
                            ++v72;
                          }
                          if ( v73 )
                            v43 = v73;
                        }
LABEL_153:
                        *(_DWORD *)(v2 + 344) = v66;
                        if ( *v43 != -8 )
                          --*(_DWORD *)(v2 + 348);
                        *v43 = v34;
                        *((_DWORD *)v43 + 4) = 1;
                        v43[1] = 0;
LABEL_100:
                        v43[1] = v197;
                        v46 = v198;
                        v198 = 0;
                        *((_DWORD *)v43 + 4) = v46;
                        v47 = (unsigned int)v210;
                        if ( (unsigned int)v210 >= HIDWORD(v210) )
                        {
                          sub_16CD150(&v209, v211, 0, 8);
                          v47 = (unsigned int)v210;
                        }
                        *(_QWORD *)&v209[8 * v47] = v34;
                        LODWORD(v210) = v210 + 1;
                        if ( v198 <= 0x40 )
                          goto LABEL_51;
                        goto LABEL_103;
                      }
                      v30 = v195;
LABEL_49:
                      v32 = *(_QWORD *)(v2 + 336);
                      v33 = *(_DWORD *)(v2 + 352);
                      if ( v30 == v31 && v25 != (__int64 *)(v32 + 24LL * v33) )
                      {
LABEL_51:
                        if ( v196 > 0x40 && v195 )
                          j_j___libc_free_0_0(v195);
                        if ( v194 > 0x40 && v193 )
                          j_j___libc_free_0_0(v193);
                        goto LABEL_57;
                      }
                      goto LABEL_95;
                    }
                    v29 = v199;
                  }
                  else
                  {
                    v29 = v193;
                  }
                  v30 = v195;
                  v198 = v28;
                  v31 = v195 | v29;
                  v197 = v31;
                  goto LABEL_49;
                }
                while ( v26 != -8 )
                {
                  v24 = (v21 - 1) & (v23 + v24);
                  v25 = (__int64 *)(v22 + 24LL * v24);
                  v26 = *v25;
                  if ( v34 == *v25 )
                    goto LABEL_42;
                  ++v23;
                }
              }
              v25 = (__int64 *)(v22 + 24 * v21);
              goto LABEL_46;
            }
          }
          v195 = 0;
          goto LABEL_40;
        }
        v35 = *(_QWORD **)(v2 + 48);
        v36 = *(_QWORD **)(v2 + 40);
        if ( v35 == v36 )
        {
          v49 = &v36[*(unsigned int *)(v2 + 60)];
          if ( v36 == v49 )
          {
            v37 = *(_QWORD **)(v2 + 40);
          }
          else
          {
            do
            {
              if ( v34 == *v36 )
                break;
              ++v36;
            }
            while ( v49 != v36 );
            v37 = v49;
          }
        }
        else
        {
          v37 = &v35[*(unsigned int *)(v2 + 56)];
          v36 = (_QWORD *)sub_16CC9F0(v176, *v18);
          if ( v34 == *v36 )
          {
            v50 = *(_QWORD *)(v2 + 48);
            if ( v50 == *(_QWORD *)(v2 + 40) )
              v51 = *(unsigned int *)(v2 + 60);
            else
              v51 = *(unsigned int *)(v2 + 56);
            v49 = (_QWORD *)(v50 + 8 * v51);
          }
          else
          {
            v38 = *(_QWORD *)(v2 + 48);
            if ( v38 != *(_QWORD *)(v2 + 40) )
            {
              v36 = (_QWORD *)(v38 + 8LL * *(unsigned int *)(v2 + 56));
              goto LABEL_64;
            }
            v49 = (_QWORD *)(v38 + 8LL * *(unsigned int *)(v2 + 60));
            v36 = v49;
          }
        }
        for ( ; v49 != v36; ++v36 )
        {
          if ( *v36 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
LABEL_64:
        if ( v36 != v37 )
        {
LABEL_57:
          v18 += 3;
          if ( v183 == v18 )
            goto LABEL_68;
          continue;
        }
        break;
      }
      v39 = (unsigned int)v210;
      if ( (unsigned int)v210 >= HIDWORD(v210) )
      {
        sub_16CD150(&v209, v211, 0, 8);
        v39 = (unsigned int)v210;
      }
      v18 += 3;
      *(_QWORD *)&v209[8 * v39] = v34;
      LODWORD(v210) = v210 + 1;
      if ( v183 != v18 )
        continue;
      break;
    }
LABEL_68:
    if ( (unsigned int)v208 > 0x40 && v207 )
      j_j___libc_free_0_0(v207);
LABEL_71:
    if ( (unsigned int)v206 > 0x40 && v205 )
      j_j___libc_free_0_0(v205);
    if ( (unsigned int)v204 > 0x40 && v203 )
      j_j___libc_free_0_0(v203);
    if ( (unsigned int)v202 > 0x40 && v201 )
      j_j___libc_free_0_0(v201);
    if ( v192 > 0x40 && v191 )
      j_j___libc_free_0_0(v191);
    v15 = v210;
    if ( (_DWORD)v210 )
      continue;
    break;
  }
LABEL_84:
  if ( v209 != v211 )
    _libc_free((unsigned __int64)v209);
}
