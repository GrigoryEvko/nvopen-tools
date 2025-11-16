// Function: sub_2C97D20
// Address: 0x2c97d20
//
__int64 __fastcall sub_2C97D20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9,
        __int64 a10)
{
  __int64 v10; // r11
  __int64 i; // rbx
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r11
  unsigned int v19; // r11d
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r10
  const __m128i *v23; // r12
  __int64 v24; // r15
  __m128i *v25; // rbx
  unsigned int v26; // esi
  __int64 v27; // r8
  _QWORD *v28; // r9
  int v29; // eax
  int v30; // ecx
  __int64 v31; // rdi
  __int64 v32; // rsi
  int v33; // edx
  __int64 v34; // r11
  __int64 *v35; // r10
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // edx
  int v39; // edx
  __int64 v40; // rdi
  unsigned int v41; // eax
  __int64 *v42; // r9
  __int64 v43; // rsi
  int v44; // eax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // r12
  unsigned int v48; // r9d
  __int64 v49; // r8
  unsigned int v50; // edx
  __int64 *v51; // rax
  __int64 v52; // r10
  unsigned int v53; // edi
  unsigned int v54; // edx
  __int64 *v55; // rax
  __int64 v56; // r10
  __m128i *v57; // r15
  unsigned int v58; // esi
  __m128i *v59; // rbx
  __int64 *v60; // r13
  int v61; // r10d
  int v62; // r10d
  __int64 v63; // r8
  unsigned int v64; // edx
  int v65; // eax
  __int64 *v66; // rdi
  __int64 v67; // rsi
  __int64 *v68; // r9
  __int64 v69; // rax
  __m128i *v70; // rax
  const __m128i *v71; // rdx
  __int64 *v72; // r13
  __int64 *v73; // r9
  int v74; // eax
  int v75; // eax
  __int64 v76; // rax
  int v78; // eax
  __int64 v79; // rax
  int v80; // ecx
  int v81; // ecx
  __int64 v82; // r8
  unsigned int v83; // edx
  __int64 v84; // rdi
  int v85; // ebx
  __int64 *v86; // r12
  int v87; // eax
  int v88; // eax
  int v89; // eax
  int v90; // ecx
  __int64 v91; // rdi
  __int64 v92; // rsi
  __int64 v93; // r11
  __int64 v94; // rax
  int v95; // edx
  int v96; // edx
  __int64 v97; // r8
  unsigned int v98; // r10d
  __int64 v99; // rsi
  int v100; // ecx
  int v101; // ecx
  __int64 v102; // r8
  int v103; // ebx
  unsigned int v104; // edx
  __int64 v105; // rdi
  int v106; // eax
  int v107; // r11d
  __int64 *v108; // r10
  _QWORD *v109; // [rsp+18h] [rbp-78h]
  int v110; // [rsp+18h] [rbp-78h]
  int v111; // [rsp+18h] [rbp-78h]
  _QWORD *v112; // [rsp+18h] [rbp-78h]
  int v113; // [rsp+18h] [rbp-78h]
  int v114; // [rsp+18h] [rbp-78h]
  __int64 v116; // [rsp+20h] [rbp-70h]
  int v117; // [rsp+20h] [rbp-70h]
  int v118; // [rsp+20h] [rbp-70h]
  int v119; // [rsp+20h] [rbp-70h]
  __int64 v120; // [rsp+20h] [rbp-70h]
  int v121; // [rsp+20h] [rbp-70h]
  __int64 v122; // [rsp+28h] [rbp-68h]
  __int64 v123; // [rsp+28h] [rbp-68h]
  __int64 v124; // [rsp+30h] [rbp-60h]
  __int64 v125; // [rsp+30h] [rbp-60h]
  __int64 v126; // [rsp+30h] [rbp-60h]
  unsigned int v127; // [rsp+38h] [rbp-58h]
  __int64 v128; // [rsp+38h] [rbp-58h]
  __int64 v129; // [rsp+38h] [rbp-58h]
  __int64 v130; // [rsp+38h] [rbp-58h]
  __int64 *v131; // [rsp+38h] [rbp-58h]
  __int64 v132; // [rsp+40h] [rbp-50h]
  __int64 *v133; // [rsp+48h] [rbp-48h]
  _QWORD *v134; // [rsp+48h] [rbp-48h]
  _QWORD v135[7]; // [rsp+58h] [rbp-38h] BYREF

  v10 = a4;
  v124 = a2;
  v122 = (a3 - 1) / 2;
  if ( a2 < v122 )
  {
    for ( i = a2; ; i = v24 )
    {
      v26 = *(_DWORD *)(a4 + 24);
      v27 = 2 * (i + 1);
      v24 = v27 - 1;
      v28 = *(_QWORD **)(a1 + ((i + 1) << 6) + 16);
      v23 = (const __m128i *)(a1 + 32 * (v27 - 1));
      v133 = (__int64 *)v23[1].m128i_i64[0];
      if ( v26 )
      {
        v14 = *v28;
        v15 = *(_QWORD *)(a4 + 8);
        v127 = v26 - 1;
        v16 = (v26 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( *v28 == *v17 )
        {
LABEL_4:
          v19 = *((_DWORD *)v17 + 2);
          goto LABEL_5;
        }
        v111 = 1;
        v35 = 0;
        while ( v18 != -4096 )
        {
          if ( !v35 && v18 == -8192 )
            v35 = v17;
          v16 = v127 & (v111 + v16);
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( v14 == *v17 )
            goto LABEL_4;
          ++v111;
        }
        if ( !v35 )
          v35 = v17;
        v88 = *(_DWORD *)(a4 + 16);
        ++*(_QWORD *)a4;
        v33 = v88 + 1;
        if ( 4 * (v88 + 1) < 3 * v26 )
        {
          if ( v26 - *(_DWORD *)(a4 + 20) - v33 > v26 >> 3 )
            goto LABEL_14;
          v112 = v28;
          sub_2C96F50(a4, v26);
          v89 = *(_DWORD *)(a4 + 24);
          if ( !v89 )
          {
LABEL_171:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          v28 = v112;
          v90 = v89 - 1;
          v91 = *(_QWORD *)(a4 + 8);
          v27 = 2 * (i + 1);
          v92 = *v112;
          v33 = *(_DWORD *)(a4 + 16) + 1;
          v93 = (v89 - 1) & (((unsigned int)*v112 >> 9) ^ ((unsigned int)*v112 >> 4));
          v35 = (__int64 *)(v91 + 16 * v93);
          v94 = *v35;
          if ( *v112 == *v35 )
            goto LABEL_14;
          v113 = 1;
          v131 = 0;
          while ( v94 != -4096 )
          {
            if ( v131 || v94 != -8192 )
              v35 = v131;
            LODWORD(v93) = v90 & (v113 + v93);
            v94 = *(_QWORD *)(v91 + 16LL * (unsigned int)v93);
            if ( v92 == v94 )
            {
              v35 = (__int64 *)(v91 + 16LL * (unsigned int)v93);
              goto LABEL_14;
            }
            v131 = v35;
            v35 = (__int64 *)(v91 + 16LL * (unsigned int)v93);
            ++v113;
          }
          goto LABEL_91;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v109 = v28;
      sub_2C96F50(a4, 2 * v26);
      v29 = *(_DWORD *)(a4 + 24);
      if ( !v29 )
        goto LABEL_171;
      v28 = v109;
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a4 + 8);
      v27 = 2 * (i + 1);
      v32 = *v109;
      v33 = *(_DWORD *)(a4 + 16) + 1;
      v34 = (v29 - 1) & (((unsigned int)*v109 >> 9) ^ ((unsigned int)*v109 >> 4));
      v35 = (__int64 *)(v31 + 16 * v34);
      v36 = *v35;
      if ( *v109 == *v35 )
        goto LABEL_14;
      v114 = 1;
      v131 = 0;
      while ( v36 != -4096 )
      {
        if ( !v131 )
        {
          if ( v36 != -8192 )
            v35 = 0;
          v131 = v35;
        }
        LODWORD(v34) = v30 & (v114 + v34);
        v35 = (__int64 *)(v31 + 16LL * (unsigned int)v34);
        v36 = *v35;
        if ( v32 == *v35 )
          goto LABEL_14;
        ++v114;
      }
LABEL_91:
      if ( v131 )
        v35 = v131;
LABEL_14:
      *(_DWORD *)(a4 + 16) = v33;
      if ( *v35 != -4096 )
        --*(_DWORD *)(a4 + 20);
      v37 = *v28;
      *((_DWORD *)v35 + 2) = 0;
      *v35 = v37;
      v26 = *(_DWORD *)(a4 + 24);
      if ( !v26 )
      {
        ++*(_QWORD *)a4;
        v135[0] = 0;
        goto LABEL_18;
      }
      v15 = *(_QWORD *)(a4 + 8);
      v19 = 0;
      v127 = v26 - 1;
LABEL_5:
      v20 = v127 & (((unsigned int)*v133 >> 9) ^ ((unsigned int)*v133 >> 4));
      v21 = (__int64 *)(v15 + 16LL * v20);
      v22 = *v21;
      if ( *v133 != *v21 )
      {
        v110 = 1;
        v42 = 0;
        while ( v22 != -4096 )
        {
          if ( v22 == -8192 && !v42 )
            v42 = v21;
          v20 = v127 & (v110 + v20);
          v21 = (__int64 *)(v15 + 16LL * v20);
          v22 = *v21;
          if ( *v133 == *v21 )
            goto LABEL_6;
          ++v110;
        }
        if ( !v42 )
          v42 = v21;
        v87 = *(_DWORD *)(a4 + 16);
        ++*(_QWORD *)a4;
        v45 = v87 + 1;
        v135[0] = v42;
        if ( 4 * v45 >= 3 * v26 )
        {
LABEL_18:
          v128 = v27;
          sub_2C96F50(a4, 2 * v26);
          v38 = *(_DWORD *)(a4 + 24);
          v27 = v128;
          if ( v38 )
          {
            v39 = v38 - 1;
            v40 = *(_QWORD *)(a4 + 8);
            v41 = v39 & (((unsigned int)*v133 >> 9) ^ ((unsigned int)*v133 >> 4));
            v42 = (__int64 *)(v40 + 16LL * v41);
            v43 = *v42;
            if ( *v42 == *v133 )
            {
LABEL_20:
              v44 = *(_DWORD *)(a4 + 16);
              v135[0] = v42;
              v45 = v44 + 1;
            }
            else
            {
              v107 = 1;
              v108 = 0;
              while ( v43 != -4096 )
              {
                if ( v43 == -8192 && !v108 )
                  v108 = v42;
                v41 = v39 & (v107 + v41);
                v42 = (__int64 *)(v40 + 16LL * v41);
                v43 = *v42;
                if ( *v133 == *v42 )
                  goto LABEL_20;
                ++v107;
              }
              if ( !v108 )
                v108 = v42;
              v45 = *(_DWORD *)(a4 + 16) + 1;
              v135[0] = v108;
              v42 = v108;
            }
          }
          else
          {
            v106 = *(_DWORD *)(a4 + 16);
            v135[0] = 0;
            v42 = 0;
            v45 = v106 + 1;
          }
        }
        else if ( v26 - (v45 + *(_DWORD *)(a4 + 20)) <= v26 >> 3 )
        {
          v130 = v27;
          sub_2C96F50(a4, v26);
          sub_2C95990(a4, v133, v135);
          v42 = (__int64 *)v135[0];
          v27 = v130;
          v45 = *(_DWORD *)(a4 + 16) + 1;
        }
        *(_DWORD *)(a4 + 16) = v45;
        if ( *v42 != -4096 )
          --*(_DWORD *)(a4 + 20);
        v23 = (const __m128i *)(a1 + ((i + 1) << 6));
        v24 = v27;
        v46 = *v133;
        *((_DWORD *)v42 + 2) = 0;
        *v42 = v46;
        goto LABEL_8;
      }
LABEL_6:
      if ( v19 >= *((_DWORD *)v21 + 2) )
      {
        v23 = (const __m128i *)(a1 + ((i + 1) << 6));
        v24 = v27;
      }
LABEL_8:
      v25 = (__m128i *)(a1 + 32 * i);
      *v25 = _mm_loadu_si128(v23);
      v25[1] = _mm_loadu_si128(v23 + 1);
      if ( v24 >= v122 )
      {
        v10 = a4;
        if ( (a3 & 1) != 0 )
          goto LABEL_25;
        goto LABEL_44;
      }
    }
  }
  if ( (a3 & 1) != 0 )
  {
    v123 = a7;
    v129 = a8;
    v134 = a9;
    v132 = a10;
    goto LABEL_134;
  }
  v24 = a2;
LABEL_44:
  if ( (a3 - 2) / 2 == v24 )
  {
    v69 = 32 * v24;
    v24 = 2 * v24 + 1;
    v70 = (__m128i *)(a1 + v69);
    v71 = (const __m128i *)(a1 + 32 * v24);
    *v70 = _mm_loadu_si128(v71);
    v70[1] = _mm_loadu_si128(v71 + 1);
  }
LABEL_25:
  v123 = a7;
  v129 = a8;
  v134 = a9;
  v132 = a10;
  v47 = (v24 - 1) / 2;
  if ( v24 <= v124 )
  {
    v124 = v24;
LABEL_134:
    v59 = (__m128i *)(a1 + 32 * v124);
    goto LABEL_55;
  }
  while ( 1 )
  {
    v58 = *(_DWORD *)(v10 + 24);
    v59 = (__m128i *)(a1 + 32 * v47);
    v60 = (__int64 *)v59[1].m128i_i64[0];
    if ( v58 )
    {
      v48 = v58 - 1;
      v49 = *(_QWORD *)(v10 + 8);
      v50 = (v58 - 1) & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
      v51 = (__int64 *)(v49 + 16LL * v50);
      v52 = *v51;
      if ( *v60 == *v51 )
      {
LABEL_28:
        v53 = *((_DWORD *)v51 + 2);
        goto LABEL_29;
      }
      v119 = 1;
      v66 = 0;
      while ( v52 != -4096 )
      {
        if ( v52 == -8192 && !v66 )
          v66 = v51;
        v50 = v48 & (v119 + v50);
        v51 = (__int64 *)(v49 + 16LL * v50);
        v52 = *v51;
        if ( *v60 == *v51 )
          goto LABEL_28;
        ++v119;
      }
      if ( !v66 )
        v66 = v51;
      v78 = *(_DWORD *)(v10 + 16);
      ++*(_QWORD *)v10;
      v65 = v78 + 1;
      if ( 4 * v65 < 3 * v58 )
      {
        if ( v58 - *(_DWORD *)(v10 + 20) - v65 > v58 >> 3 )
          goto LABEL_62;
        v120 = v10;
        sub_2C96F50(v10, v58);
        v10 = v120;
        v95 = *(_DWORD *)(v120 + 24);
        if ( !v95 )
        {
LABEL_172:
          ++*(_DWORD *)(v10 + 16);
          BUG();
        }
        v96 = v95 - 1;
        v97 = *(_QWORD *)(v120 + 8);
        v98 = v96 & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
        v65 = *(_DWORD *)(v120 + 16) + 1;
        v66 = (__int64 *)(v97 + 16LL * v98);
        v99 = *v66;
        if ( *v66 == *v60 )
          goto LABEL_62;
        v121 = 1;
        v68 = 0;
        while ( v99 != -4096 )
        {
          if ( v99 == -8192 && !v68 )
            v68 = v66;
          v98 = v96 & (v121 + v98);
          v66 = (__int64 *)(v97 + 16LL * v98);
          v99 = *v66;
          if ( *v60 == *v66 )
            goto LABEL_62;
          ++v121;
        }
        goto LABEL_39;
      }
    }
    else
    {
      ++*(_QWORD *)v10;
    }
    v116 = v10;
    sub_2C96F50(v10, 2 * v58);
    v10 = v116;
    v61 = *(_DWORD *)(v116 + 24);
    if ( !v61 )
      goto LABEL_172;
    v62 = v61 - 1;
    v63 = *(_QWORD *)(v116 + 8);
    v64 = v62 & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
    v65 = *(_DWORD *)(v116 + 16) + 1;
    v66 = (__int64 *)(v63 + 16LL * v64);
    v67 = *v66;
    if ( *v60 == *v66 )
      goto LABEL_62;
    v117 = 1;
    v68 = 0;
    while ( v67 != -4096 )
    {
      if ( !v68 && v67 == -8192 )
        v68 = v66;
      v64 = v62 & (v117 + v64);
      v66 = (__int64 *)(v63 + 16LL * v64);
      v67 = *v66;
      if ( *v60 == *v66 )
        goto LABEL_62;
      ++v117;
    }
LABEL_39:
    if ( v68 )
      v66 = v68;
LABEL_62:
    *(_DWORD *)(v10 + 16) = v65;
    if ( *v66 != -4096 )
      --*(_DWORD *)(v10 + 20);
    v79 = *v60;
    *((_DWORD *)v66 + 2) = 0;
    *v66 = v79;
    v58 = *(_DWORD *)(v10 + 24);
    if ( !v58 )
    {
      ++*(_QWORD *)v10;
      goto LABEL_66;
    }
    v49 = *(_QWORD *)(v10 + 8);
    v48 = v58 - 1;
    v53 = 0;
LABEL_29:
    v54 = v48 & (((unsigned int)*a9 >> 9) ^ ((unsigned int)*a9 >> 4));
    v55 = (__int64 *)(v49 + 16LL * v54);
    v56 = *v55;
    if ( *v55 != *a9 )
      break;
LABEL_30:
    v57 = (__m128i *)(a1 + 32 * v24);
    if ( v53 >= *((_DWORD *)v55 + 2) )
    {
      v59 = v57;
      goto LABEL_55;
    }
    *v57 = _mm_loadu_si128(v59);
    v57[1] = _mm_loadu_si128(v59 + 1);
    v24 = v47;
    if ( v124 >= v47 )
      goto LABEL_55;
    v47 = (v47 - 1) / 2;
  }
  v118 = 1;
  v72 = 0;
  while ( v56 != -4096 )
  {
    if ( v56 == -8192 && !v72 )
      v72 = v55;
    v54 = v48 & (v118 + v54);
    v55 = (__int64 *)(v49 + 16LL * v54);
    v56 = *v55;
    if ( *a9 == *v55 )
      goto LABEL_30;
    ++v118;
  }
  v73 = v72;
  if ( !v72 )
    v73 = v55;
  v74 = *(_DWORD *)(v10 + 16);
  ++*(_QWORD *)v10;
  v75 = v74 + 1;
  if ( 4 * v75 < 3 * v58 )
  {
    if ( v58 - (v75 + *(_DWORD *)(v10 + 20)) > v58 >> 3 )
      goto LABEL_52;
    v126 = v10;
    sub_2C96F50(v10, v58);
    v10 = v126;
    v100 = *(_DWORD *)(v126 + 24);
    if ( v100 )
    {
      v101 = v100 - 1;
      v102 = *(_QWORD *)(v126 + 8);
      v103 = 1;
      v86 = 0;
      v104 = v101 & (((unsigned int)*a9 >> 9) ^ ((unsigned int)*a9 >> 4));
      v75 = *(_DWORD *)(v126 + 16) + 1;
      v73 = (__int64 *)(v102 + 16LL * v104);
      v105 = *v73;
      if ( *v73 != *a9 )
      {
        while ( v105 != -4096 )
        {
          if ( !v86 && v105 == -8192 )
            v86 = v73;
          v104 = v101 & (v103 + v104);
          v73 = (__int64 *)(v102 + 16LL * v104);
          v105 = *v73;
          if ( *a9 == *v73 )
            goto LABEL_52;
          ++v103;
        }
LABEL_70:
        if ( v86 )
          v73 = v86;
      }
      goto LABEL_52;
    }
LABEL_173:
    ++*(_DWORD *)(v10 + 16);
    BUG();
  }
LABEL_66:
  v125 = v10;
  sub_2C96F50(v10, 2 * v58);
  v10 = v125;
  v80 = *(_DWORD *)(v125 + 24);
  if ( !v80 )
    goto LABEL_173;
  v81 = v80 - 1;
  v82 = *(_QWORD *)(v125 + 8);
  v83 = v81 & (((unsigned int)*a9 >> 9) ^ ((unsigned int)*a9 >> 4));
  v75 = *(_DWORD *)(v125 + 16) + 1;
  v73 = (__int64 *)(v82 + 16LL * v83);
  v84 = *v73;
  if ( *v73 != *a9 )
  {
    v85 = 1;
    v86 = 0;
    while ( v84 != -4096 )
    {
      if ( !v86 && v84 == -8192 )
        v86 = v73;
      v83 = v81 & (v85 + v83);
      v73 = (__int64 *)(v82 + 16LL * v83);
      v84 = *v73;
      if ( *a9 == *v73 )
        goto LABEL_52;
      ++v85;
    }
    goto LABEL_70;
  }
LABEL_52:
  *(_DWORD *)(v10 + 16) = v75;
  if ( *v73 != -4096 )
    --*(_DWORD *)(v10 + 20);
  v76 = *a9;
  *((_DWORD *)v73 + 2) = 0;
  v59 = (__m128i *)(a1 + 32 * v24);
  *v73 = v76;
LABEL_55:
  v59->m128i_i64[0] = v123;
  v59->m128i_i64[1] = v129;
  v59[1].m128i_i64[0] = (__int64)v134;
  v59[1].m128i_i64[1] = v132;
  return v132;
}
