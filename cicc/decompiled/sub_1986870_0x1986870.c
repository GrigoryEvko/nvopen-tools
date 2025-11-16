// Function: sub_1986870
// Address: 0x1986870
//
__int64 __fastcall sub_1986870(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r15
  unsigned int v5; // ecx
  __int64 v6; // rax
  unsigned __int64 *v7; // rax
  int v8; // edx
  int v9; // ecx
  int v10; // r8d
  int v11; // r9d
  unsigned __int64 v12; // r14
  unsigned __int64 *v13; // rbx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  int v17; // ecx
  unsigned int v18; // r9d
  int v19; // ecx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 *v23; // r8
  __int64 *v24; // rdi
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 *v27; // r14
  __int64 *i; // rbx
  __int64 v29; // rsi
  __int64 *v30; // r9
  __int64 *v31; // rax
  __int64 *v32; // rcx
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // r14
  __int64 *v36; // r11
  __int64 *v37; // r9
  __int64 v38; // rsi
  __int64 *v39; // rbx
  __int64 v40; // rsi
  __int64 *v41; // rbx
  __int64 *j; // r13
  __int64 v43; // rsi
  __int64 *v44; // rdi
  __int64 *v45; // rax
  __int64 *v46; // rcx
  __int64 v47; // r12
  __int64 *v48; // rax
  __int64 v49; // rsi
  __int64 *v50; // rdi
  __int64 *v51; // r15
  __int64 *v52; // r14
  __int64 *v53; // r12
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  int v58; // ecx
  unsigned __int64 v59; // rax
  __int64 *v60; // rax
  __int64 *v61; // r10
  __int64 *v62; // rax
  __int64 *v63; // rcx
  __int64 *v64; // r10
  __int64 *v65; // rax
  __int64 *v66; // rcx
  unsigned int v67; // r12d
  __int64 *v69; // rax
  __int64 v70; // rdx
  __int64 *v71; // r13
  __int64 v72; // rdx
  __int64 *v73; // r12
  unsigned int v74; // eax
  __int64 v75; // r14
  __int64 v76; // r15
  __int64 *v77; // rbx
  __int64 *v78; // rdi
  __int64 *v79; // rcx
  __int64 *v80; // r13
  __int64 v81; // r8
  __int64 *v82; // r9
  __int64 *v83; // rax
  __int64 *v84; // rsi
  __int64 *v85; // rax
  __int64 v86; // rdx
  __int64 *v87; // r13
  __int64 *v88; // r9
  __int64 v89; // rsi
  __int64 *v90; // rbx
  __int64 v91; // rsi
  __int64 *v92; // rbx
  __int64 v93; // rax
  __int64 *v94; // r12
  __int64 v95; // rsi
  __int64 *v96; // rdi
  __int64 *v97; // r12
  _QWORD **v98; // rax
  __int64 *v99; // rax
  __int64 *v100; // r12
  __int64 *v101; // r13
  __int64 v102; // rax
  _QWORD **v103; // rax
  __int64 *v104; // rax
  __int64 *v105; // r8
  __int64 *v106; // rax
  __int64 *v107; // r10
  __int64 *v108; // rax
  __int64 *v109; // r8
  __int64 *v110; // rbx
  _QWORD **v111; // rax
  __int64 v112; // [rsp+10h] [rbp-180h]
  __int64 *v113; // [rsp+18h] [rbp-178h]
  __int64 v115; // [rsp+30h] [rbp-160h]
  unsigned int v116; // [rsp+3Ch] [rbp-154h]
  __int64 *v118; // [rsp+50h] [rbp-140h]
  unsigned int v119; // [rsp+50h] [rbp-140h]
  unsigned int v120; // [rsp+50h] [rbp-140h]
  __int64 v121; // [rsp+58h] [rbp-138h]
  __int64 v122; // [rsp+58h] [rbp-138h]
  __int64 v123; // [rsp+68h] [rbp-128h] BYREF
  __int64 v124; // [rsp+70h] [rbp-120h] BYREF
  __int64 *v125; // [rsp+78h] [rbp-118h]
  __int64 v126; // [rsp+80h] [rbp-110h]
  __int64 v127; // [rsp+88h] [rbp-108h]
  __int64 v128; // [rsp+90h] [rbp-100h] BYREF
  __int64 *v129; // [rsp+98h] [rbp-F8h]
  __int64 v130; // [rsp+A0h] [rbp-F0h]
  __int64 v131; // [rsp+A8h] [rbp-E8h]
  __int64 v132; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 *v133; // [rsp+B8h] [rbp-D8h]
  void *s; // [rsp+C0h] [rbp-D0h]
  _BYTE v135[12]; // [rsp+C8h] [rbp-C8h]
  _BYTE v136[184]; // [rsp+D8h] [rbp-B8h] BYREF

  v2 = a1 + 5368;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v4 = *(_QWORD *)(*(_QWORD *)v3 + 48LL);
  v121 = *(_QWORD *)v3 + 40LL;
  if ( v121 != v4 )
  {
    while ( 1 )
    {
      v6 = v4 - 24;
      if ( !v4 )
        v6 = 0;
      v132 = v6;
      v7 = (unsigned __int64 *)sub_19865A0(v2, &v132);
      v12 = v7[1];
      v13 = v7;
      if ( v12 << 6 > 0x21 )
        goto LABEL_3;
      v14 = 2 * v12;
      v15 = *v7;
      if ( 2 * v12 > 1 )
      {
        v16 = (__int64)realloc(v15, 16 * v12, v8, 16 * (int)v12, v10, v11);
        if ( !v16 && (16 * v12 || (v16 = malloc(1u)) == 0) )
        {
LABEL_86:
          sub_16BD1C0("Allocation failed", 1u);
          v16 = 0;
        }
      }
      else
      {
        v14 = 1;
        v16 = (__int64)realloc(v15, 8u, v8, v9, v10, v11);
        if ( !v16 )
          goto LABEL_86;
      }
      v17 = *((_DWORD *)v13 + 4);
      *v13 = v16;
      v13[1] = v14;
      v18 = (unsigned int)(v17 + 63) >> 6;
      if ( v18 < v14 && v14 != v18 )
      {
        v119 = (unsigned int)(v17 + 63) >> 6;
        memset((void *)(v16 + 8LL * v18), 0, 8 * (v14 - v18));
        v17 = *((_DWORD *)v13 + 4);
        v16 = *v13;
        v18 = v119;
      }
      v19 = v17 & 0x3F;
      if ( v19 )
      {
        *(_QWORD *)(v16 + 8LL * (v18 - 1)) &= ~(-1LL << v19);
        v16 = *v13;
      }
      v20 = v13[1] - (unsigned int)v12;
      if ( v20 )
      {
        memset((void *)(v16 + 8LL * (unsigned int)v12), 0, 8 * v20);
        v5 = *((_DWORD *)v13 + 4);
        if ( v5 > 0x21 )
          goto LABEL_4;
        goto LABEL_16;
      }
LABEL_3:
      v5 = *((_DWORD *)v13 + 4);
      if ( v5 > 0x21 )
        goto LABEL_4;
LABEL_16:
      v21 = v13[1];
      v22 = (v5 + 63) >> 6;
      if ( v21 > v22 )
      {
        v59 = v21 - v22;
        if ( v59 )
        {
          v120 = (v5 + 63) >> 6;
          memset((void *)(*v13 + 8LL * v22), 0, 8 * v59);
          v5 = *((_DWORD *)v13 + 4);
          v22 = v120;
        }
        if ( (v5 & 0x3F) != 0 )
        {
          LOBYTE(v5) = v5 & 0x3F;
LABEL_82:
          *(_QWORD *)(*v13 + 8LL * (v22 - 1)) &= ~(-1LL << v5);
          v5 = *((_DWORD *)v13 + 4);
        }
LABEL_4:
        *((_DWORD *)v13 + 4) = 34;
        if ( v5 > 0x22 )
        {
          v56 = 0x3FFFFFFFFLL;
          v57 = v13[1];
          if ( v57 > 1 )
          {
            memset((void *)(*v13 + 8), 0, 8 * v57 - 8);
            v58 = v13[2] & 0x3F;
            if ( !v58 )
              goto LABEL_5;
            v56 = ~(-1LL << v58);
          }
          *(_QWORD *)*v13 &= v56;
        }
LABEL_5:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v121 == v4 )
          break;
      }
      else
      {
        if ( v5 )
          goto LABEL_82;
        *((_DWORD *)v13 + 4) = 34;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v121 == v4 )
          break;
      }
    }
  }
  v23 = (__int64 *)v136;
  v132 = 0;
  v133 = (__int64 *)v136;
  v24 = (__int64 *)v136;
  v25 = *(_QWORD *)(a1 + 88);
  v26 = *(unsigned int *)(a1 + 96);
  s = v136;
  *(_QWORD *)v135 = 16;
  *(_DWORD *)&v135[8] = 0;
  v122 = v25 + 320 * v26;
  if ( v25 == v122 )
  {
    v41 = *(__int64 **)(a1 + 5224);
    j = &v41[*(unsigned int *)(a1 + 5232)];
    if ( j != v41 )
    {
      v37 = (__int64 *)v136;
      v36 = (__int64 *)v136;
      goto LABEL_45;
    }
    v132 = 1;
    goto LABEL_133;
  }
  do
  {
    v27 = *(__int64 **)(v25 + 8);
    for ( i = &v27[*(unsigned int *)(v25 + 16)]; i != v27; ++v27 )
    {
LABEL_24:
      v29 = *v27;
      if ( v24 != v23 )
        goto LABEL_22;
      v30 = &v24[*(unsigned int *)&v135[4]];
      if ( v30 != v24 )
      {
        v31 = v24;
        v32 = 0;
        while ( v29 != *v31 )
        {
          if ( *v31 == -2 )
            v32 = v31;
          if ( v30 == ++v31 )
          {
            if ( !v32 )
              goto LABEL_112;
            ++v27;
            *v32 = v29;
            v23 = (__int64 *)s;
            --*(_DWORD *)&v135[8];
            v24 = v133;
            ++v132;
            if ( i != v27 )
              goto LABEL_24;
            goto LABEL_33;
          }
        }
        continue;
      }
LABEL_112:
      if ( *(_DWORD *)&v135[4] < *(_DWORD *)v135 )
      {
        ++*(_DWORD *)&v135[4];
        *v30 = v29;
        v24 = v133;
        ++v132;
        v23 = (__int64 *)s;
      }
      else
      {
LABEL_22:
        sub_16CCBA0((__int64)&v132, v29);
        v23 = (__int64 *)s;
        v24 = v133;
      }
    }
LABEL_33:
    v33 = *(__int64 **)(v25 + 168);
    if ( v33 == *(__int64 **)(v25 + 160) )
      v34 = *(unsigned int *)(v25 + 180);
    else
      v34 = *(unsigned int *)(v25 + 176);
    v35 = &v33[v34];
    v36 = v23;
    v37 = v24;
    if ( v33 != v35 )
    {
      while ( 1 )
      {
        v38 = *v33;
        v39 = v33;
        if ( (unsigned __int64)*v33 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v35 == ++v33 )
          goto LABEL_38;
      }
      while ( 1 )
      {
        if ( v39 == v35 )
          goto LABEL_38;
        if ( v24 != v23 )
          break;
        v64 = &v24[*(unsigned int *)&v135[4]];
        if ( v24 == v64 )
        {
LABEL_110:
          if ( *(_DWORD *)&v135[4] >= *(_DWORD *)v135 )
            break;
          ++*(_DWORD *)&v135[4];
          *v64 = v38;
          v24 = v133;
          v23 = (__int64 *)s;
          ++v132;
          v37 = v133;
          v36 = (__int64 *)s;
        }
        else
        {
          v65 = v24;
          v66 = 0;
          while ( v38 != *v65 )
          {
            if ( *v65 == -2 )
              v66 = v65;
            if ( v64 == ++v65 )
            {
              if ( !v66 )
                goto LABEL_110;
              *v66 = v38;
              v23 = (__int64 *)s;
              v24 = v133;
              --*(_DWORD *)&v135[8];
              ++v132;
              v36 = (__int64 *)s;
              v37 = v133;
              break;
            }
          }
        }
LABEL_90:
        v60 = v39 + 1;
        if ( v39 + 1 == v35 )
          goto LABEL_38;
        while ( 1 )
        {
          v38 = *v60;
          v39 = v60;
          if ( (unsigned __int64)*v60 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v35 == ++v60 )
          {
            v40 = *(_QWORD *)v25;
            if ( v23 != v24 )
              goto LABEL_39;
            goto LABEL_94;
          }
        }
      }
      sub_16CCBA0((__int64)&v132, v38);
      v23 = (__int64 *)s;
      v24 = v133;
      v36 = (__int64 *)s;
      v37 = v133;
      goto LABEL_90;
    }
LABEL_38:
    v40 = *(_QWORD *)v25;
    if ( v23 != v24 )
    {
LABEL_39:
      sub_16CCBA0((__int64)&v132, v40);
      v23 = (__int64 *)s;
      v24 = v133;
      v36 = (__int64 *)s;
      v37 = v133;
      goto LABEL_40;
    }
LABEL_94:
    v61 = &v23[*(unsigned int *)&v135[4]];
    if ( v61 == v23 )
    {
LABEL_182:
      if ( *(_DWORD *)&v135[4] >= *(_DWORD *)v135 )
        goto LABEL_39;
      ++*(_DWORD *)&v135[4];
      *v61 = v40;
      v24 = v133;
      v23 = (__int64 *)s;
      ++v132;
      v37 = v133;
      v36 = (__int64 *)s;
    }
    else
    {
      v62 = v23;
      v63 = 0;
      while ( v40 != *v62 )
      {
        if ( *v62 == -2 )
          v63 = v62;
        if ( v61 == ++v62 )
        {
          if ( !v63 )
            goto LABEL_182;
          *v63 = v40;
          v23 = (__int64 *)s;
          v24 = v133;
          --*(_DWORD *)&v135[8];
          ++v132;
          v36 = (__int64 *)s;
          v37 = v133;
          break;
        }
      }
    }
LABEL_40:
    v25 += 320;
  }
  while ( v122 != v25 );
  v41 = *(__int64 **)(a1 + 5224);
  for ( j = &v41[*(unsigned int *)(a1 + 5232)]; v41 != j; ++v41 )
  {
LABEL_45:
    v43 = *v41;
    if ( v36 != v37 )
      goto LABEL_43;
    v44 = &v36[*(unsigned int *)&v135[4]];
    if ( v44 != v36 )
    {
      v45 = v36;
      v46 = 0;
      while ( v43 != *v45 )
      {
        if ( *v45 == -2 )
          v46 = v45;
        if ( v44 == ++v45 )
        {
          if ( !v46 )
            goto LABEL_184;
          ++v41;
          *v46 = v43;
          v36 = (__int64 *)s;
          --*(_DWORD *)&v135[8];
          v37 = v133;
          ++v132;
          if ( v41 != j )
            goto LABEL_45;
          goto LABEL_54;
        }
      }
      continue;
    }
LABEL_184:
    if ( *(_DWORD *)&v135[4] < *(_DWORD *)v135 )
    {
      ++*(_DWORD *)&v135[4];
      *v44 = v43;
      v37 = v133;
      ++v132;
      v36 = (__int64 *)s;
    }
    else
    {
LABEL_43:
      sub_16CCBA0((__int64)&v132, v43);
      v36 = (__int64 *)s;
      v37 = v133;
    }
  }
LABEL_54:
  v47 = *(_QWORD *)(a1 + 88);
  v112 = v47 + 320LL * *(unsigned int *)(a1 + 96);
  if ( v112 == v47 )
  {
LABEL_127:
    ++v132;
    if ( v133 != s )
    {
      v74 = 4 * (*(_DWORD *)&v135[4] - *(_DWORD *)&v135[8]);
      if ( v74 < 0x20 )
        v74 = 32;
      if ( *(_DWORD *)v135 > v74 )
      {
        sub_16CC920((__int64)&v132);
        v25 = *(_QWORD *)(a1 + 88);
        v26 = *(unsigned int *)(a1 + 96);
        goto LABEL_134;
      }
      memset(s, -1, 8LL * *(unsigned int *)v135);
    }
    v25 = *(_QWORD *)(a1 + 88);
    v26 = *(unsigned int *)(a1 + 96);
LABEL_133:
    *(_QWORD *)&v135[4] = 0;
LABEL_134:
    v75 = v25;
    v76 = v25 + 320 * v26;
    if ( v76 == v25 )
    {
LABEL_156:
      v128 = 0;
      v129 = 0;
      v92 = *(__int64 **)(a1 + 5224);
      v93 = *(unsigned int *)(a1 + 5232);
      v130 = 0;
      v131 = 0;
      v94 = &v92[v93];
      if ( v92 == v94 )
      {
        v96 = 0;
      }
      else
      {
        do
        {
          v95 = *v92++;
          sub_1986060(a1, v95, (__int64)&v132, a2, (__int64)&v128);
        }
        while ( v94 != v92 );
        v96 = v129;
        v97 = &v129[(unsigned int)v131];
        if ( (_DWORD)v130 && v129 != v97 )
        {
          v110 = v129;
          while ( *v110 == -16 || *v110 == -8 )
          {
            if ( ++v110 == v97 )
              goto LABEL_159;
          }
          if ( v110 != v97 )
          {
LABEL_228:
            v124 = *v110;
            v111 = (_QWORD **)sub_19865A0(a1 + 5368, &v124);
            **v111 |= 0x200000000uLL;
            while ( ++v110 != v97 )
            {
              if ( *v110 != -8 && *v110 != -16 )
              {
                if ( v110 != v97 )
                  goto LABEL_228;
                break;
              }
            }
            v96 = v129;
          }
        }
      }
LABEL_159:
      j___libc_free_0(v96);
      v67 = 1;
      goto LABEL_118;
    }
    while ( 1 )
    {
      v77 = *(__int64 **)(v75 + 8);
      v78 = (__int64 *)s;
      v79 = v133;
      v80 = &v77[*(unsigned int *)(v75 + 16)];
      if ( v77 != v80 )
        break;
LABEL_148:
      v85 = *(__int64 **)(v75 + 168);
      if ( v85 == *(__int64 **)(v75 + 160) )
        v86 = *(unsigned int *)(v75 + 180);
      else
        v86 = *(unsigned int *)(v75 + 176);
      v87 = &v85[v86];
      v88 = v79;
      if ( v85 != v87 )
      {
        while ( 1 )
        {
          v89 = *v85;
          v90 = v85;
          if ( (unsigned __int64)*v85 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v87 == ++v85 )
            goto LABEL_153;
        }
        while ( 1 )
        {
          if ( v90 == v87 )
            goto LABEL_153;
          if ( v79 != v78 )
            break;
          v107 = &v79[*(unsigned int *)&v135[4]];
          if ( v107 == v79 )
          {
LABEL_210:
            if ( *(_DWORD *)&v135[4] >= *(_DWORD *)v135 )
              break;
            ++*(_DWORD *)&v135[4];
            *v107 = v89;
            v79 = v133;
            ++v132;
            v78 = (__int64 *)s;
            v88 = v133;
          }
          else
          {
            v108 = v79;
            v109 = 0;
            while ( v89 != *v108 )
            {
              if ( *v108 == -2 )
                v109 = v108;
              if ( v107 == ++v108 )
              {
                if ( !v109 )
                  goto LABEL_210;
                *v109 = v89;
                v79 = v133;
                --*(_DWORD *)&v135[8];
                v78 = (__int64 *)s;
                ++v132;
                v88 = v133;
                break;
              }
            }
          }
LABEL_190:
          v104 = v90 + 1;
          if ( v90 + 1 == v87 )
            goto LABEL_153;
          while ( 1 )
          {
            v89 = *v104;
            v90 = v104;
            if ( (unsigned __int64)*v104 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v87 == ++v104 )
            {
              v91 = *(_QWORD *)v75;
              if ( v78 != v79 )
                goto LABEL_154;
              goto LABEL_194;
            }
          }
        }
        sub_16CCBA0((__int64)&v132, v89);
        v79 = v133;
        v78 = (__int64 *)s;
        v88 = v133;
        goto LABEL_190;
      }
LABEL_153:
      v91 = *(_QWORD *)v75;
      if ( v78 == v79 )
      {
LABEL_194:
        v105 = &v78[*(unsigned int *)&v135[4]];
        if ( v105 != v78 )
        {
          v106 = 0;
          while ( v91 != *v88 )
          {
            if ( *v88 == -2 )
              v106 = v88;
            if ( v105 == ++v88 )
            {
              if ( !v106 )
                goto LABEL_217;
              *v106 = v91;
              --*(_DWORD *)&v135[8];
              ++v132;
              goto LABEL_155;
            }
          }
          goto LABEL_155;
        }
LABEL_217:
        if ( *(_DWORD *)&v135[4] < *(_DWORD *)v135 )
        {
          ++*(_DWORD *)&v135[4];
          *v105 = v91;
          ++v132;
          goto LABEL_155;
        }
      }
LABEL_154:
      sub_16CCBA0((__int64)&v132, v91);
LABEL_155:
      v75 += 320;
      if ( v76 == v75 )
        goto LABEL_156;
    }
    while ( 1 )
    {
LABEL_139:
      v81 = *v77;
      if ( v78 != v79 )
        goto LABEL_137;
      v82 = &v78[*(unsigned int *)&v135[4]];
      if ( v82 != v78 )
      {
        v83 = v78;
        v84 = 0;
        while ( v81 != *v83 )
        {
          if ( *v83 == -2 )
            v84 = v83;
          if ( v82 == ++v83 )
          {
            if ( !v84 )
              goto LABEL_212;
            ++v77;
            *v84 = v81;
            v78 = (__int64 *)s;
            --*(_DWORD *)&v135[8];
            v79 = v133;
            ++v132;
            if ( v80 != v77 )
              goto LABEL_139;
            goto LABEL_148;
          }
        }
        goto LABEL_138;
      }
LABEL_212:
      if ( *(_DWORD *)&v135[4] < *(_DWORD *)v135 )
      {
        ++*(_DWORD *)&v135[4];
        *v82 = v81;
        v79 = v133;
        ++v132;
        v78 = (__int64 *)s;
      }
      else
      {
LABEL_137:
        sub_16CCBA0((__int64)&v132, *v77);
        v78 = (__int64 *)s;
        v79 = v133;
      }
LABEL_138:
      if ( v80 == ++v77 )
        goto LABEL_148;
    }
  }
  v115 = *(_QWORD *)(a1 + 88);
  while ( 1 )
  {
    v124 = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    sub_1986060(a1, *(_QWORD *)v115, (__int64)&v132, a2, (__int64)&v124);
    if ( (_DWORD)v126 )
    {
      v100 = v125;
      v101 = &v125[(unsigned int)v127];
      if ( v125 != v101 )
      {
        while ( *v100 == -8 || *v100 == -16 )
        {
          if ( v101 == ++v100 )
            goto LABEL_57;
        }
        while ( v101 != v100 )
        {
          v102 = *v100++;
          v128 = v102;
          v103 = (_QWORD **)sub_19865A0(a1 + 5368, &v128);
          **v103 |= 1uLL;
          if ( v100 == v101 )
            break;
          while ( *v100 == -16 || *v100 == -8 )
          {
            if ( v101 == ++v100 )
              goto LABEL_57;
          }
        }
      }
    }
LABEL_57:
    v116 = 1;
    v48 = *(__int64 **)(v115 + 8);
    v118 = v48;
    v113 = &v48[*(unsigned int *)(v115 + 16)];
    if ( v113 != v48 )
      break;
LABEL_121:
    v69 = *(__int64 **)(v115 + 168);
    if ( v69 == *(__int64 **)(v115 + 160) )
      v70 = *(unsigned int *)(v115 + 180);
    else
      v70 = *(unsigned int *)(v115 + 176);
    v71 = &v69[v70];
    if ( v69 != v71 )
    {
      while ( 1 )
      {
        v72 = *v69;
        v73 = v69;
        if ( (unsigned __int64)*v69 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v71 == ++v69 )
          goto LABEL_126;
      }
      if ( v71 != v69 )
      {
        do
        {
          v128 = v72;
          v98 = (_QWORD **)sub_19865A0(a1 + 5368, &v128);
          **v98 |= 0x200000000uLL;
          v99 = v73 + 1;
          if ( v73 + 1 == v71 )
            break;
          v72 = *v99;
          for ( ++v73; (unsigned __int64)*v99 >= 0xFFFFFFFFFFFFFFFELL; v73 = v99 )
          {
            if ( v71 == ++v99 )
              goto LABEL_126;
            v72 = *v99;
          }
        }
        while ( v71 != v73 );
      }
    }
LABEL_126:
    j___libc_free_0(v125);
    v115 += 320;
    if ( v112 == v115 )
      goto LABEL_127;
  }
  while ( 1 )
  {
    v49 = *v118;
    v128 = 0;
    v129 = 0;
    v130 = 0;
    v131 = 0;
    sub_1986060(a1, v49, (__int64)&v132, a2, (__int64)&v128);
    v50 = v129;
    if ( (_DWORD)v126 != (_DWORD)v130 )
      break;
    v51 = &v129[(unsigned int)v131];
    if ( (_DWORD)v130 && v51 != v129 )
    {
      v52 = v129;
      while ( *v52 == -16 || *v52 == -8 )
      {
        if ( v51 == ++v52 )
          goto LABEL_59;
      }
      if ( v51 != v52 )
      {
        v53 = &v129[(unsigned int)v131];
        do
        {
          v54 = *v52++;
          v123 = v54;
          v55 = (_QWORD *)sub_19865A0(a1 + 5368, &v123);
          *(_QWORD *)(8LL * (v116 >> 6) + *v55) |= 1LL << v116;
          if ( v52 == v53 )
            break;
          while ( *v52 == -16 || *v52 == -8 )
          {
            if ( v53 == ++v52 )
              goto LABEL_73;
          }
        }
        while ( v53 != v52 );
LABEL_73:
        v50 = v129;
      }
    }
LABEL_59:
    ++v116;
    j___libc_free_0(v50);
    if ( v113 == ++v118 )
      goto LABEL_121;
  }
  j___libc_free_0(v129);
  v67 = 0;
  j___libc_free_0(v125);
LABEL_118:
  if ( s != v133 )
    _libc_free((unsigned __int64)s);
  return v67;
}
