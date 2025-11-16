// Function: sub_322C940
// Address: 0x322c940
//
__int64 __fastcall sub_322C940(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // eax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  __int64 (*v9)(void); // rax
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 *v15; // r13
  __int64 *v16; // r15
  int v17; // edx
  __int64 *v18; // rsi
  unsigned int v19; // eax
  __int64 *v20; // rdi
  __int64 v21; // r8
  __int64 v22; // r14
  unsigned int v23; // edx
  unsigned int v24; // eax
  __int64 *v25; // r9
  unsigned int v26; // esi
  __int64 *v27; // rdi
  int v28; // esi
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // r8
  int v32; // r10d
  __int64 *v33; // r9
  unsigned int v34; // edx
  unsigned int v35; // edi
  unsigned int v36; // ecx
  unsigned int v37; // esi
  char v38; // dl
  __int64 *v39; // rcx
  __int64 v40; // rax
  __int64 *v41; // r12
  __int64 *v42; // rcx
  __int64 v43; // rax
  __int64 result; // rax
  __int64 v45; // r15
  __int64 v46; // r10
  __int64 v47; // rcx
  __int64 *v48; // rax
  __int64 v49; // r10
  unsigned __int64 v50; // rbx
  __int64 v51; // rdx
  __int64 v52; // r10
  int v53; // ecx
  __int64 *v54; // rdi
  char v55; // si
  int v56; // ecx
  __int64 *v57; // r11
  __int64 v58; // r8
  unsigned int v59; // edx
  __int64 v60; // r10
  int v61; // r10d
  __int64 *v62; // r8
  int v63; // ecx
  unsigned int v64; // edx
  __int64 v65; // rdi
  int v66; // esi
  __int64 *v67; // rax
  __int64 *v68; // r13
  unsigned int v69; // edx
  __int64 v70; // r10
  unsigned int v71; // edx
  __int64 v72; // r10
  __int64 v73; // rdi
  __int64 *v74; // r9
  int v75; // r10d
  int v76; // ecx
  __int64 v77; // rdx
  int v78; // r13d
  unsigned int v79; // edx
  __int64 v80; // r10
  __int64 k; // rbx
  unsigned __int64 v82; // rdx
  __int64 (*v83)(); // rax
  const void *v84; // r13
  unsigned __int64 v85; // r15
  __int64 v86; // rdi
  int v87; // eax
  __int64 v88; // r9
  __int64 v89; // rsi
  _QWORD *v90; // rax
  _QWORD *v91; // rdx
  __int64 v92; // rax
  __int64 m; // rbx
  int v94; // ebx
  _BYTE *v95; // r15
  __int64 *v96; // r12
  __int64 *v97; // r13
  __int64 v98; // rsi
  __int64 *v99; // r15
  __int64 *v100; // rbx
  __int64 v101; // rsi
  unsigned __int64 v102; // rcx
  unsigned __int64 v103; // rdx
  unsigned __int64 v104; // rdx
  __int64 v105; // rcx
  __int64 *v106; // rax
  __int64 v107; // rax
  __int64 v108; // r8
  __int64 v109; // r9
  unsigned __int64 v110; // rax
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // r15
  unsigned __int64 v114; // rdx
  __int64 v115; // rdi
  __int64 *v116; // r10
  int v117; // r11d
  __int64 v118; // r9
  __int64 v119; // rcx
  int v120; // edx
  __int64 v121; // rdi
  __int64 *v122; // r10
  int v123; // r11d
  __int64 v124; // r9
  __int64 v125; // rdx
  int v126; // ecx
  __int64 *v127; // rcx
  __int64 v128; // rax
  __int64 *v129; // r8
  int v130; // ecx
  unsigned int v131; // edx
  __int64 v132; // rdi
  int v133; // esi
  __int64 v134; // rax
  unsigned __int64 v135; // rax
  __int64 v136; // rdx
  _QWORD *v137; // rax
  __int64 v138; // rdx
  _QWORD *j; // rdx
  unsigned int v140; // eax
  unsigned int v141; // r12d
  char v142; // al
  __int64 v143; // rdi
  __int64 v144; // rax
  __int64 *v145; // r8
  int v146; // esi
  unsigned int v147; // ecx
  __int64 v148; // rdi
  int v149; // edx
  __int64 *v150; // r9
  __int64 *v151; // r9
  int v152; // r8d
  unsigned int v153; // ecx
  __int64 v154; // rdi
  int v155; // esi
  __int64 *v156; // rdx
  _QWORD *v157; // rax
  __int64 v158; // rdx
  _QWORD *v159; // rdx
  __int64 v160; // [rsp+8h] [rbp-1F8h]
  unsigned int v161; // [rsp+18h] [rbp-1E8h]
  unsigned int v162; // [rsp+18h] [rbp-1E8h]
  unsigned int v163; // [rsp+18h] [rbp-1E8h]
  __int64 *v164; // [rsp+20h] [rbp-1E0h]
  __int64 *v165; // [rsp+28h] [rbp-1D8h]
  unsigned __int64 v166; // [rsp+40h] [rbp-1C0h]
  __int64 v167; // [rsp+48h] [rbp-1B8h]
  __int64 *v168; // [rsp+48h] [rbp-1B8h]
  __int64 v170; // [rsp+58h] [rbp-1A8h]
  __int64 *v171; // [rsp+58h] [rbp-1A8h]
  __int64 v172; // [rsp+58h] [rbp-1A8h]
  __int64 v173; // [rsp+58h] [rbp-1A8h]
  __int64 v174; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v175; // [rsp+68h] [rbp-198h] BYREF
  _QWORD v176[2]; // [rsp+70h] [rbp-190h] BYREF
  _BYTE *v177; // [rsp+80h] [rbp-180h] BYREF
  __int64 v178; // [rsp+88h] [rbp-178h]
  _BYTE v179[16]; // [rsp+90h] [rbp-170h] BYREF
  __int64 v180; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v181; // [rsp+A8h] [rbp-158h]
  __int64 *v182; // [rsp+B0h] [rbp-150h] BYREF
  unsigned int v183; // [rsp+B8h] [rbp-148h]
  __int64 v184; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v185; // [rsp+D8h] [rbp-128h]
  __int64 *v186; // [rsp+E0h] [rbp-120h] BYREF
  unsigned int v187; // [rsp+E8h] [rbp-118h]
  _BYTE *v188; // [rsp+120h] [rbp-E0h] BYREF
  __int64 v189; // [rsp+128h] [rbp-D8h]
  _BYTE v190[208]; // [rsp+130h] [rbp-D0h] BYREF

  v2 = *(_DWORD *)(a1 + 3008);
  ++*(_QWORD *)(a1 + 3000);
  v3 = v2 >> 1;
  if ( v3 )
  {
    if ( (*(_BYTE *)(a1 + 3008) & 1) == 0 )
    {
      v4 = 4 * v3;
      goto LABEL_4;
    }
LABEL_112:
    v6 = 4;
    v7 = (_QWORD *)(a1 + 3016);
    goto LABEL_7;
  }
  if ( !*(_DWORD *)(a1 + 3012) )
    goto LABEL_10;
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 3008) & 1) != 0 )
    goto LABEL_112;
LABEL_4:
  v5 = *(unsigned int *)(a1 + 3024);
  if ( (unsigned int)v5 <= v4 || (unsigned int)v5 <= 0x40 )
  {
    v6 = v5;
    v7 = *(_QWORD **)(a1 + 3016);
LABEL_7:
    for ( i = &v7[v6]; i != v7; ++v7 )
      *v7 = -4096;
    *(_QWORD *)(a1 + 3008) &= 1uLL;
    goto LABEL_10;
  }
  if ( !v3 || (v140 = v3 - 1) == 0 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 3016), 8 * v5, 8);
    *(_BYTE *)(a1 + 3008) |= 1u;
    goto LABEL_269;
  }
  _BitScanReverse(&v140, v140);
  v141 = 1 << (33 - (v140 ^ 0x1F));
  if ( v141 - 5 <= 0x3A )
  {
    v141 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 3016), 8 * v5, 8);
    v142 = *(_BYTE *)(a1 + 3008);
    v143 = 512;
    goto LABEL_279;
  }
  if ( (_DWORD)v5 != v141 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 3016), 8 * v5, 8);
    v142 = *(_BYTE *)(a1 + 3008) | 1;
    *(_BYTE *)(a1 + 3008) = v142;
    if ( v141 <= 4 )
    {
LABEL_269:
      v172 = *(_QWORD *)(a1 + 3008);
      *(_QWORD *)(a1 + 3008) = v172 & 1;
      if ( (v172 & 1) != 0 )
      {
        v137 = (_QWORD *)(a1 + 3016);
        v138 = 4;
      }
      else
      {
        v137 = *(_QWORD **)(a1 + 3016);
        v138 = *(unsigned int *)(a1 + 3024);
      }
      for ( j = &v137[v138]; j != v137; ++v137 )
      {
        if ( v137 )
          *v137 = -4096;
      }
      goto LABEL_10;
    }
    v143 = 8LL * v141;
LABEL_279:
    *(_BYTE *)(a1 + 3008) = v142 & 0xFE;
    v144 = sub_C7D670(v143, 8);
    *(_DWORD *)(a1 + 3024) = v141;
    *(_QWORD *)(a1 + 3016) = v144;
    goto LABEL_269;
  }
  v173 = *(_QWORD *)(a1 + 3008);
  *(_QWORD *)(a1 + 3008) = v173 & 1;
  if ( (v173 & 1) != 0 )
  {
    v157 = (_QWORD *)(a1 + 3016);
    v158 = 4;
  }
  else
  {
    v158 = v5;
    v157 = *(_QWORD **)(a1 + 3016);
  }
  v159 = &v157[v158];
  do
  {
    if ( v157 )
      *v157 = -4096;
    ++v157;
  }
  while ( v159 != v157 );
LABEL_10:
  v160 = 0;
  v9 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v9 != sub_2DAC790 )
    v160 = v9();
  v10 = (__int64 *)&v182;
  v180 = 0;
  v181 = 1;
  do
    *v10++ = -4096;
  while ( v10 != &v184 );
  v11 = (__int64 *)&v186;
  v184 = 0;
  v185 = 1;
  do
  {
    *v11 = -4096;
    v11 += 2;
  }
  while ( v11 != (__int64 *)&v188 );
  v12 = *(_QWORD *)(a2 + 328);
  v170 = a2 + 320;
  if ( v12 != a2 + 320 )
  {
    while ( 1 )
    {
      v13 = v12 + 48;
      if ( v12 + 48 == (*(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_18;
      if ( !*(_DWORD *)(v12 + 72) )
        goto LABEL_18;
      v14 = *(_QWORD *)(v12 + 56);
      if ( v13 == v14 )
        goto LABEL_18;
      while ( !*(_QWORD *)(v14 + 56) || !*(_DWORD *)(sub_B10CD0(v14 + 56) + 4) )
      {
        if ( (*(_BYTE *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 44) & 8) != 0 )
        {
          do
            v14 = *(_QWORD *)(v14 + 8);
          while ( (*(_BYTE *)(v14 + 44) & 8) != 0 );
        }
        v14 = *(_QWORD *)(v14 + 8);
        if ( v13 == v14 )
          goto LABEL_18;
      }
      v15 = *(__int64 **)(v12 + 64);
      v16 = &v15[*(unsigned int *)(v12 + 72)];
      if ( v15 != v16 )
      {
        while ( 1 )
        {
          v22 = *v15;
          if ( (v181 & 1) != 0 )
          {
            v17 = 3;
            v18 = (__int64 *)&v182;
          }
          else
          {
            v23 = v183;
            v18 = v182;
            if ( !v183 )
            {
              v24 = v181;
              ++v180;
              v25 = 0;
              v26 = ((unsigned int)v181 >> 1) + 1;
              goto LABEL_39;
            }
            v17 = v183 - 1;
          }
          v19 = v17 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v20 = &v18[v19];
          v21 = *v20;
          if ( v22 == *v20 )
          {
LABEL_34:
            if ( v16 == ++v15 )
              break;
          }
          else
          {
            v61 = 1;
            v25 = 0;
            while ( v21 != -4096 )
            {
              if ( v25 || v21 != -8192 )
                v20 = v25;
              v19 = v17 & (v61 + v19);
              v21 = v18[v19];
              if ( v22 == v21 )
                goto LABEL_34;
              ++v61;
              v25 = v20;
              v20 = &v18[v19];
            }
            v24 = v181;
            if ( !v25 )
              v25 = v20;
            ++v180;
            v26 = ((unsigned int)v181 >> 1) + 1;
            if ( (v181 & 1) == 0 )
            {
              v23 = v183;
LABEL_39:
              if ( 4 * v26 < 3 * v23 )
                goto LABEL_40;
              goto LABEL_91;
            }
            v23 = 4;
            if ( 4 * v26 < 0xC )
            {
LABEL_40:
              if ( v23 - HIDWORD(v181) - v26 > v23 >> 3 )
                goto LABEL_41;
              sub_322B900((__int64)&v180, v23);
              if ( (v181 & 1) != 0 )
              {
                v130 = 3;
                v129 = (__int64 *)&v182;
              }
              else
              {
                v129 = v182;
                if ( !v183 )
                {
LABEL_339:
                  LODWORD(v181) = (2 * ((unsigned int)v181 >> 1) + 2) | v181 & 1;
                  BUG();
                }
                v130 = v183 - 1;
              }
              v131 = v130 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
              v25 = &v129[v131];
              v24 = v181;
              v132 = *v25;
              if ( v22 == *v25 )
                goto LABEL_41;
              v133 = 1;
              v67 = 0;
              while ( v132 != -4096 )
              {
                if ( !v67 && v132 == -8192 )
                  v67 = v25;
                v131 = v130 & (v133 + v131);
                v25 = &v129[v131];
                v132 = *v25;
                if ( v22 == *v25 )
                  goto LABEL_248;
                ++v133;
              }
              goto LABEL_246;
            }
LABEL_91:
            sub_322B900((__int64)&v180, 2 * v23);
            if ( (v181 & 1) != 0 )
            {
              v63 = 3;
              v62 = (__int64 *)&v182;
            }
            else
            {
              v62 = v182;
              if ( !v183 )
                goto LABEL_339;
              v63 = v183 - 1;
            }
            v64 = v63 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v25 = &v62[v64];
            v24 = v181;
            v65 = *v25;
            if ( v22 == *v25 )
              goto LABEL_41;
            v66 = 1;
            v67 = 0;
            while ( v65 != -4096 )
            {
              if ( !v67 && v65 == -8192 )
                v67 = v25;
              v64 = v63 & (v66 + v64);
              v25 = &v62[v64];
              v65 = *v25;
              if ( v22 == *v25 )
                goto LABEL_248;
              ++v66;
            }
LABEL_246:
            if ( v67 )
              v25 = v67;
LABEL_248:
            v24 = v181;
LABEL_41:
            LODWORD(v181) = (2 * (v24 >> 1) + 2) | v24 & 1;
            if ( *v25 != -4096 )
              --HIDWORD(v181);
            ++v15;
            *v25 = v22;
            if ( v16 == v15 )
              break;
          }
        }
      }
      if ( (v185 & 1) != 0 )
      {
        v27 = (__int64 *)&v186;
        v28 = 3;
      }
      else
      {
        v37 = v187;
        v27 = v186;
        if ( !v187 )
        {
          v34 = v185;
          ++v184;
          v30 = 0;
          v35 = ((unsigned int)v185 >> 1) + 1;
          goto LABEL_106;
        }
        v28 = v187 - 1;
      }
      v29 = v28 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v30 = &v27[2 * v29];
      v31 = *v30;
      if ( v12 != *v30 )
        break;
LABEL_18:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v170 == v12 )
        goto LABEL_53;
    }
    v32 = 1;
    v33 = 0;
    while ( v31 != -4096 )
    {
      if ( !v33 && v31 == -8192 )
        v33 = v30;
      v29 = v28 & (v32 + v29);
      v30 = &v27[2 * v29];
      v31 = *v30;
      if ( v12 == *v30 )
        goto LABEL_18;
      ++v32;
    }
    v34 = v185;
    if ( v33 )
      v30 = v33;
    ++v184;
    v35 = ((unsigned int)v185 >> 1) + 1;
    if ( (v185 & 1) != 0 )
    {
      v36 = 12;
      v37 = 4;
    }
    else
    {
      v37 = v187;
LABEL_106:
      v36 = 3 * v37;
    }
    if ( 4 * v35 >= v36 )
    {
      sub_322BD10((__int64)&v184, 2 * v37);
      if ( (v185 & 1) != 0 )
      {
        v151 = (__int64 *)&v186;
        v152 = 3;
      }
      else
      {
        v151 = v186;
        if ( !v187 )
        {
LABEL_340:
          LODWORD(v185) = (2 * ((unsigned int)v185 >> 1) + 2) | v185 & 1;
          BUG();
        }
        v152 = v187 - 1;
      }
      v34 = v185;
      v153 = v152 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v30 = &v151[2 * v153];
      v154 = *v30;
      if ( v12 == *v30 )
        goto LABEL_109;
      v155 = 1;
      v156 = 0;
      while ( v154 != -4096 )
      {
        if ( !v156 && v154 == -8192 )
          v156 = v30;
        v153 = v152 & (v155 + v153);
        v30 = &v151[2 * v153];
        v154 = *v30;
        if ( v12 == *v30 )
          goto LABEL_287;
        ++v155;
      }
      if ( v156 )
      {
        v30 = v156;
        v34 = v185;
        goto LABEL_109;
      }
    }
    else
    {
      if ( v37 - HIDWORD(v185) - v35 > v37 >> 3 )
        goto LABEL_109;
      sub_322BD10((__int64)&v184, v37);
      if ( (v185 & 1) != 0 )
      {
        v145 = (__int64 *)&v186;
        v146 = 3;
      }
      else
      {
        v145 = v186;
        if ( !v187 )
          goto LABEL_340;
        v146 = v187 - 1;
      }
      v34 = v185;
      v147 = v146 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v30 = &v145[2 * v147];
      v148 = *v30;
      if ( v12 == *v30 )
        goto LABEL_109;
      v149 = 1;
      v150 = 0;
      while ( v148 != -4096 )
      {
        if ( !v150 && v148 == -8192 )
          v150 = v30;
        v147 = v146 & (v149 + v147);
        v30 = &v145[2 * v147];
        v148 = *v30;
        if ( v12 == *v30 )
          goto LABEL_287;
        ++v149;
      }
      if ( v150 )
        v30 = v150;
    }
LABEL_287:
    v34 = v185;
LABEL_109:
    LODWORD(v185) = (2 * (v34 >> 1) + 2) | v34 & 1;
    if ( *v30 != -4096 )
      --HIDWORD(v185);
    *v30 = v12;
    v30[1] = v14;
    goto LABEL_18;
  }
LABEL_53:
  v38 = v181 & 1;
  if ( (unsigned int)v181 >> 1 )
  {
    if ( v38 )
    {
      v41 = (__int64 *)&v182;
      v165 = &v184;
      v42 = &v184;
    }
    else
    {
      v39 = v182;
      v40 = v183;
      v41 = v182;
      v165 = &v182[v183];
      if ( v165 == v182 )
      {
LABEL_62:
        v43 = v40;
        goto LABEL_63;
      }
      v42 = &v182[v183];
    }
    do
    {
      if ( *v41 != -4096 && *v41 != -8192 )
        break;
      ++v41;
    }
    while ( v42 != v41 );
  }
  else
  {
    if ( v38 )
    {
      v127 = (__int64 *)&v182;
      v128 = 4;
    }
    else
    {
      v127 = v182;
      v128 = v183;
    }
    v41 = &v127[v128];
    v165 = &v127[v128];
  }
  if ( !v38 )
  {
    v39 = v182;
    v40 = v183;
    goto LABEL_62;
  }
  v39 = (__int64 *)&v182;
  v43 = 4;
LABEL_63:
  result = (__int64)&v39[v43];
  v164 = (__int64 *)result;
  if ( (__int64 *)result == v41 )
    goto LABEL_78;
  do
  {
    v45 = *v41;
    v176[0] = &v184;
    v46 = *(unsigned int *)(v45 + 120);
    v47 = *(_QWORD *)(v45 + 48);
    v167 = v45 + 48;
    v176[1] = a1;
    v48 = *(__int64 **)(v45 + 112);
    v49 = v46;
    v166 = v47 & 0xFFFFFFFFFFFFFFF8LL;
    v50 = v47 & 0xFFFFFFFFFFFFFFF8LL;
    v171 = &v48[v49];
    if ( v45 + 48 == (v47 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v99 = &v48[v49];
      if ( v48 != &v48[v49] )
      {
        v100 = v48;
        do
        {
          v101 = *v100++;
          sub_322C540(v176, v101, 0);
        }
        while ( v99 != v100 );
      }
      goto LABEL_73;
    }
    v51 = (v49 * 8) >> 3;
    v52 = (v49 * 8) >> 5;
    if ( !v52 )
    {
LABEL_125:
      if ( v51 == 2 )
      {
        v58 = v185 & 1;
LABEL_229:
        v121 = *v48;
        if ( (_BYTE)v58 )
        {
          v122 = (__int64 *)&v186;
          v123 = 3;
        }
        else
        {
          v122 = v186;
          if ( !v187 )
          {
LABEL_234:
            ++v48;
LABEL_129:
            v73 = *v48;
            if ( (_BYTE)v58 )
            {
              v74 = (__int64 *)&v186;
              v75 = 3;
            }
            else
            {
              v74 = v186;
              if ( !v187 )
                goto LABEL_73;
              v75 = v187 - 1;
            }
            v76 = 1;
            v58 = v75 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
            v77 = v74[2 * v58];
            if ( v73 != v77 )
            {
              while ( 1 )
              {
                if ( v77 == -4096 )
                  goto LABEL_73;
                v58 = v75 & (unsigned int)(v76 + v58);
                v77 = v74[2 * (unsigned int)v58];
                if ( v73 == v77 )
                  break;
                ++v76;
              }
            }
            goto LABEL_72;
          }
          v123 = v187 - 1;
        }
        v124 = v123 & (((unsigned int)v121 >> 4) ^ ((unsigned int)v121 >> 9));
        v125 = v122[2 * v124];
        if ( v121 != v125 )
        {
          v126 = 1;
          while ( v125 != -4096 )
          {
            v124 = v123 & (unsigned int)(v124 + v126);
            v125 = v122[2 * v124];
            if ( v121 == v125 )
              goto LABEL_72;
            ++v126;
          }
          goto LABEL_234;
        }
LABEL_72:
        if ( v171 == v48 )
          goto LABEL_73;
        goto LABEL_140;
      }
      if ( v51 != 3 )
      {
        if ( v51 != 1 )
          goto LABEL_73;
        LOBYTE(v58) = v185 & 1;
        goto LABEL_129;
      }
      v115 = *v48;
      v58 = v185 & 1;
      if ( (v185 & 1) != 0 )
      {
        v116 = (__int64 *)&v186;
        v117 = 3;
      }
      else
      {
        v116 = v186;
        if ( !v187 )
        {
LABEL_262:
          ++v48;
          goto LABEL_229;
        }
        v117 = v187 - 1;
      }
      LODWORD(v118) = v117 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
      v119 = v116[2 * (unsigned int)v118];
      if ( v115 == v119 )
        goto LABEL_72;
      v120 = 1;
      while ( v119 != -4096 )
      {
        v118 = v117 & (unsigned int)(v118 + v120);
        v119 = v116[2 * v118];
        if ( v115 == v119 )
          goto LABEL_72;
        ++v120;
      }
      goto LABEL_262;
    }
    v53 = 4;
    v54 = (__int64 *)&v186;
    v55 = v185 & 1;
    if ( (v185 & 1) == 0 )
    {
      v54 = v186;
      v53 = v187;
    }
    v56 = v53 - 1;
    v57 = &v48[4 * v52];
    while ( 1 )
    {
      v58 = *v48;
      if ( !v55 && !v187 )
        goto LABEL_123;
      v59 = v56 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v60 = v54[2 * v59];
      if ( v58 == v60 )
        goto LABEL_72;
      v78 = 1;
      while ( v60 != -4096 )
      {
        v59 = v56 & (v78 + v59);
        v60 = v54[2 * v59];
        if ( v58 == v60 )
          goto LABEL_72;
        ++v78;
      }
      v58 = v48[1];
      v68 = v48 + 1;
      if ( !v55 && !v187 )
        goto LABEL_123;
      v79 = v56 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v80 = v54[2 * v79];
      if ( v58 == v80 )
        break;
      v161 = 1;
      while ( v80 != -4096 )
      {
        v58 = v161;
        v79 = v56 & (v161 + v79);
        ++v161;
        v80 = v54[2 * v79];
        if ( v48[1] == v80 )
          goto LABEL_139;
      }
      v58 = v48[2];
      v68 = v48 + 2;
      if ( v55 || v187 )
      {
        v69 = v56 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v70 = v54[2 * v69];
        if ( v70 == v58 )
          break;
        v162 = 1;
        while ( v70 != -4096 )
        {
          v58 = v162;
          v69 = v56 & (v162 + v69);
          ++v162;
          v70 = v54[2 * v69];
          if ( v48[2] == v70 )
            goto LABEL_139;
        }
        v58 = v48[3];
        v68 = v48 + 3;
        if ( v55 || v187 )
        {
          v71 = v56 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
          v72 = v54[2 * v71];
          if ( v58 == v72 )
            break;
          v163 = 1;
          while ( v72 != -4096 )
          {
            v58 = v163;
            v71 = v56 & (v163 + v71);
            ++v163;
            v72 = v54[2 * v71];
            if ( v48[3] == v72 )
              goto LABEL_139;
          }
        }
      }
LABEL_123:
      v48 += 4;
      if ( v57 == v48 )
      {
        v51 = v171 - v48;
        goto LABEL_125;
      }
    }
LABEL_139:
    if ( v171 == v68 )
      goto LABEL_73;
LABEL_140:
    v177 = v179;
    v178 = 0x200000000LL;
    if ( !v166 )
      BUG();
    if ( (*(_QWORD *)v166 & 4) == 0 && (*(_BYTE *)(v166 + 44) & 4) != 0 )
    {
      for ( k = *(_QWORD *)v166; ; k = *(_QWORD *)v50 )
      {
        v50 = k & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v50 + 44) & 4) == 0 )
          break;
      }
    }
    v82 = 2;
    v174 = 0;
    v188 = v190;
    v189 = 0x400000000LL;
    v175 = 0;
    v83 = *(__int64 (**)())(*(_QWORD *)v160 + 344LL);
    if ( v83 != sub_2DB1AE0 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v83)(
              v160,
              v45,
              &v174,
              &v175,
              &v188,
              0)
        && (_DWORD)v189
        && v175 )
      {
        v102 = *(_QWORD *)(v45 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v102 )
          BUG();
        v103 = *(_QWORD *)(v45 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v102 & 4) == 0 && (v134 = *(_QWORD *)v102, (*(_BYTE *)(v102 + 44) & 4) != 0) )
        {
          while ( 1 )
          {
            v135 = v134 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v135 + 44) & 4) == 0 )
              break;
            v134 = *(_QWORD *)v135;
          }
          v136 = *(_QWORD *)v102;
          if ( *(_QWORD *)(v135 + 56) )
          {
            while ( 1 )
            {
              v103 = v136 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v103 + 44) & 4) == 0 )
                break;
              v136 = *(_QWORD *)v103;
            }
            goto LABEL_201;
          }
        }
        else if ( *(_QWORD *)(v102 + 56) )
        {
LABEL_201:
          if ( *(_DWORD *)(sub_B10CD0(v103 + 56) + 4) )
          {
            v104 = *(_QWORD *)(v45 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v104 )
              BUG();
            v105 = *(_QWORD *)v104;
            v106 = (__int64 *)(*(_QWORD *)(v45 + 48) & 0xFFFFFFFFFFFFFFF8LL);
            if ( (*(_QWORD *)v104 & 4) == 0 && (*(_BYTE *)(v104 + 44) & 4) != 0 )
            {
              while ( 1 )
              {
                v106 = (__int64 *)(v105 & 0xFFFFFFFFFFFFFFF8LL);
                if ( (*(_BYTE *)((v105 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
                  break;
                v105 = *v106;
              }
            }
            v107 = sub_B10CD0((__int64)(v106 + 7));
            sub_322C540(v176, v175, *(_DWORD *)(v107 + 4));
            v110 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v110 )
              BUG();
            v111 = *(_QWORD *)v110;
            v50 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v110 & 4) == 0 && (*(_BYTE *)(v110 + 44) & 4) != 0 )
            {
              while ( 1 )
              {
                v50 = v111 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)((v111 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
                  break;
                v111 = *(_QWORD *)v50;
              }
            }
            v112 = (unsigned int)v178;
            v113 = v174;
            v114 = (unsigned int)v178 + 1LL;
            if ( v114 > HIDWORD(v178) )
            {
              sub_C8D5F0((__int64)&v177, v179, v114, 8u, v108, v109);
              v112 = (unsigned int)v178;
            }
            *(_QWORD *)&v177[8 * v112] = v113;
            LODWORD(v178) = v178 + 1;
            goto LABEL_152;
          }
        }
      }
      v82 = HIDWORD(v178);
    }
    v84 = *(const void **)(v45 + 112);
    v85 = *(unsigned int *)(v45 + 120);
    v86 = 0;
    v87 = 0;
    LODWORD(v178) = 0;
    v88 = 8 * v85;
    if ( v82 < v85 )
    {
      sub_C8D5F0((__int64)&v177, v179, v85, 8u, v58, v88);
      v87 = v178;
      v88 = 8 * v85;
      v86 = 8LL * (unsigned int)v178;
    }
    if ( v88 )
    {
      memcpy(&v177[v86], v84, v88);
      v87 = v178;
    }
    LODWORD(v178) = v85 + v87;
LABEL_152:
    if ( v188 != v190 )
      _libc_free((unsigned __int64)v188);
    if ( v167 == v50 )
    {
LABEL_167:
      v94 = 0;
      goto LABEL_168;
    }
    while ( 2 )
    {
      v89 = *(_QWORD *)(v50 + 56);
      v188 = (_BYTE *)v89;
      if ( !v89 || (sub_B96E90((__int64)&v188, v89, 1), !v188) )
      {
LABEL_160:
        v90 = (_QWORD *)(*(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL);
        v91 = v90;
        if ( !v90 )
          BUG();
        v50 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
        v92 = *v90;
        if ( (v92 & 4) == 0 && (*((_BYTE *)v91 + 44) & 4) != 0 )
        {
          for ( m = v92; ; m = *(_QWORD *)v50 )
          {
            v50 = m & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v50 + 44) & 4) == 0 )
              break;
          }
        }
        if ( v167 == v50 )
          goto LABEL_167;
        continue;
      }
      break;
    }
    if ( !*(_DWORD *)(sub_B10CD0((__int64)&v188) + 4) )
    {
      if ( v188 )
        sub_B91220((__int64)&v188, (__int64)v188);
      goto LABEL_160;
    }
    v94 = *(_DWORD *)(sub_B10CD0((__int64)&v188) + 4);
    if ( v188 )
      sub_B91220((__int64)&v188, (__int64)v188);
LABEL_168:
    v95 = v177;
    if ( &v177[8 * (unsigned int)v178] != v177 )
    {
      v168 = v41;
      v96 = (__int64 *)v177;
      v97 = (__int64 *)&v177[8 * (unsigned int)v178];
      do
      {
        v98 = *v96++;
        sub_322C540(v176, v98, v94);
      }
      while ( v97 != v96 );
      v41 = v168;
      v95 = v177;
    }
    if ( v95 != v179 )
      _libc_free((unsigned __int64)v95);
LABEL_73:
    result = (__int64)v165;
    for ( ++v41; v165 != v41; ++v41 )
    {
      result = *v41;
      if ( *v41 != -4096 && result != -8192 )
        break;
    }
  }
  while ( v41 != v164 );
LABEL_78:
  if ( (v185 & 1) == 0 )
    result = sub_C7D6A0((__int64)v186, 16LL * v187, 8);
  if ( (v181 & 1) == 0 )
    return sub_C7D6A0((__int64)v182, 8LL * v183, 8);
  return result;
}
