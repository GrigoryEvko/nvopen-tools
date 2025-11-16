// Function: sub_22D2D60
// Address: 0x22d2d60
//
__int64 __fastcall sub_22D2D60(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8,
        __int64 a9)
{
  __int64 *v9; // rbx
  __int64 result; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // rdx
  __int64 *v13; // r8
  __int64 v14; // r9
  __int64 v15; // r12
  __int64 v16; // r10
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rdi
  int v21; // edx
  unsigned int v22; // ecx
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // r13
  char v28; // si
  __int64 v29; // rdi
  int v30; // edx
  unsigned int v31; // ecx
  __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 v36; // r13
  char v37; // si
  __int64 v38; // rdi
  int v39; // edx
  unsigned int v40; // ecx
  __int64 v41; // rax
  __int64 v42; // r11
  __int64 v43; // rdx
  __int64 v44; // r14
  __int64 v45; // r13
  char v46; // si
  __int64 v47; // rdi
  int v48; // edx
  unsigned int v49; // ecx
  __int64 v50; // rax
  __int64 v51; // r11
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 v54; // r14
  char v55; // si
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  int v63; // r11d
  unsigned int i; // eax
  _QWORD *v65; // rsi
  unsigned int v66; // eax
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  int v70; // r11d
  unsigned int j; // eax
  _QWORD *v72; // rsi
  unsigned int v73; // eax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rcx
  int v77; // r11d
  unsigned int k; // eax
  _QWORD *v79; // rsi
  unsigned int v80; // eax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rcx
  int v84; // r11d
  unsigned int m; // eax
  _QWORD *v86; // rsi
  unsigned int v87; // eax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 **v92; // r14
  __int64 **v93; // r9
  __int64 v94; // rdi
  int v95; // esi
  unsigned int v96; // edx
  __int64 v97; // rax
  __int64 *v98; // r10
  __int64 v99; // rdx
  char v100; // al
  __int64 v101; // r13
  __int64 *v102; // r15
  char v103; // cl
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 v107; // rsi
  int v108; // r11d
  unsigned int n; // eax
  __int64 **v110; // rcx
  unsigned int v111; // eax
  int v112; // eax
  int v113; // eax
  int v114; // eax
  int v115; // eax
  __int64 v116; // rsi
  int v117; // eax
  __int64 v118; // r12
  __int64 v119; // r13
  int v120; // r15d
  int v121; // r15d
  int v122; // r15d
  int v123; // r15d
  int v124; // r8d
  __int64 **v125; // [rsp+8h] [rbp-B8h]
  __int64 *dest; // [rsp+28h] [rbp-98h]
  __int64 v128; // [rsp+30h] [rbp-90h] BYREF
  char v129[8]; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v130; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v131; // [rsp+48h] [rbp-78h] BYREF
  __int64 v132; // [rsp+50h] [rbp-70h]
  _BYTE v133[16]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v134; // [rsp+70h] [rbp-50h]

  v9 = a1;
  result = *a1;
  v11 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*a1 & 4) == 0 )
  {
    if ( !v11 )
    {
      dest = a1;
LABEL_4:
      v12 = v9;
      v9 = dest;
      dest = v12;
      goto LABEL_101;
    }
    dest = a1 + 1;
    v130 = a7;
    v131 = a8;
    v132 = a9;
    goto LABEL_136;
  }
  v9 = *(__int64 **)v11;
  v13 = a7;
  v14 = (__int64)a8;
  v15 = 8LL * *(unsigned int *)(v11 + 8);
  v16 = a9;
  v17 = (__int64 *)(*(_QWORD *)v11 + v15);
  v130 = a7;
  dest = v17;
  v18 = v15 >> 3;
  v19 = v15 >> 5;
  v131 = a8;
  v132 = a9;
  if ( !v19 )
    goto LABEL_6;
  while ( 1 )
  {
    v53 = *v13;
    v54 = *v9;
    v55 = *(_BYTE *)(*v13 + 8) & 1;
    if ( v55 )
    {
      v20 = v53 + 16;
      v21 = 7;
    }
    else
    {
      v56 = *(unsigned int *)(v53 + 24);
      v20 = *(_QWORD *)(v53 + 16);
      if ( !(_DWORD)v56 )
        goto LABEL_68;
      v21 = v56 - 1;
    }
    v22 = v21 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
    v23 = v20 + 16LL * v22;
    v24 = *(_QWORD *)v23;
    if ( v54 == *(_QWORD *)v23 )
      goto LABEL_11;
    v112 = 1;
    while ( v24 != -4096 )
    {
      v120 = v112 + 1;
      v22 = v21 & (v112 + v22);
      v23 = v20 + 16LL * v22;
      v24 = *(_QWORD *)v23;
      if ( v54 == *(_QWORD *)v23 )
        goto LABEL_11;
      v112 = v120;
    }
    if ( v55 )
    {
      v88 = 128;
      goto LABEL_69;
    }
    v56 = *(unsigned int *)(v53 + 24);
LABEL_68:
    v88 = 16 * v56;
LABEL_69:
    v23 = v20 + v88;
LABEL_11:
    v25 = 128;
    if ( !v55 )
      v25 = 16LL * *(unsigned int *)(v53 + 24);
    if ( v23 == v20 + v25 )
    {
      v60 = v13[1];
      v61 = *(unsigned int *)(v60 + 24);
      v62 = *(_QWORD *)(v60 + 8);
      if ( (_DWORD)v61 )
      {
        v63 = 1;
        for ( i = (v61 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)
                    | ((unsigned __int64)(((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; i = (v61 - 1) & v66 )
        {
          v65 = (_QWORD *)(v62 + 24LL * i);
          if ( v54 == *v65 && v65[1] == v14 )
            break;
          if ( *v65 == -4096 && v65[1] == -4096 )
            goto LABEL_139;
          v66 = v63 + i;
          ++v63;
        }
      }
      else
      {
LABEL_139:
        v65 = (_QWORD *)(v62 + 24 * v61);
      }
      v129[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v65[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v65[2] + 24LL),
                  v14,
                  v16,
                  v13);
      v128 = v54;
      sub_BBCF50((__int64)v133, v53, &v128, v129);
      v23 = v134;
    }
    if ( *(_BYTE *)(v23 + 8) )
      goto LABEL_80;
    v26 = v9[1];
    v27 = *v130;
    v28 = *(_BYTE *)(*v130 + 8) & 1;
    if ( v28 )
    {
      v29 = v27 + 16;
      v30 = 7;
    }
    else
    {
      v57 = *(unsigned int *)(v27 + 24);
      v29 = *(_QWORD *)(v27 + 16);
      if ( !(_DWORD)v57 )
        goto LABEL_71;
      v30 = v57 - 1;
    }
    v31 = v30 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v32 = v29 + 16LL * v31;
    v33 = *(_QWORD *)v32;
    if ( v26 == *(_QWORD *)v32 )
      goto LABEL_18;
    v113 = 1;
    while ( v33 != -4096 )
    {
      v121 = v113 + 1;
      v31 = v30 & (v113 + v31);
      v32 = v29 + 16LL * v31;
      v33 = *(_QWORD *)v32;
      if ( v26 == *(_QWORD *)v32 )
        goto LABEL_18;
      v113 = v121;
    }
    if ( v28 )
    {
      v89 = 128;
      goto LABEL_72;
    }
    v57 = *(unsigned int *)(v27 + 24);
LABEL_71:
    v89 = 16 * v57;
LABEL_72:
    v32 = v29 + v89;
LABEL_18:
    v34 = 128;
    if ( !v28 )
      v34 = 16LL * *(unsigned int *)(v27 + 24);
    if ( v32 == v29 + v34 )
    {
      v67 = v130[1];
      v68 = *(unsigned int *)(v67 + 24);
      v69 = *(_QWORD *)(v67 + 8);
      if ( (_DWORD)v68 )
      {
        v70 = 1;
        for ( j = (v68 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4)
                    | ((unsigned __int64)(((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4)))); ; j = (v68 - 1) & v73 )
        {
          v72 = (_QWORD *)(v69 + 24LL * j);
          if ( v26 == *v72 && v131 == (__int64 *)v72[1] )
            break;
          if ( *v72 == -4096 && v72[1] == -4096 )
            goto LABEL_142;
          v73 = v70 + j;
          ++v70;
        }
      }
      else
      {
LABEL_142:
        v72 = (_QWORD *)(v69 + 24 * v68);
      }
      v129[0] = (*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64, __int64 *))(**(_QWORD **)(v72[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v72[2] + 24LL),
                  v131,
                  v132,
                  v130);
      v128 = v26;
      sub_BBCF50((__int64)v133, v27, &v128, v129);
      v32 = v134;
    }
    if ( *(_BYTE *)(v32 + 8) )
    {
      ++v9;
      goto LABEL_80;
    }
    v35 = v9[2];
    v36 = *v130;
    v37 = *(_BYTE *)(*v130 + 8) & 1;
    if ( v37 )
    {
      v38 = v36 + 16;
      v39 = 7;
    }
    else
    {
      v58 = *(unsigned int *)(v36 + 24);
      v38 = *(_QWORD *)(v36 + 16);
      if ( !(_DWORD)v58 )
        goto LABEL_74;
      v39 = v58 - 1;
    }
    v40 = v39 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v41 = v38 + 16LL * v40;
    v42 = *(_QWORD *)v41;
    if ( v35 == *(_QWORD *)v41 )
      goto LABEL_25;
    v114 = 1;
    while ( v42 != -4096 )
    {
      v122 = v114 + 1;
      v40 = v39 & (v114 + v40);
      v41 = v38 + 16LL * v40;
      v42 = *(_QWORD *)v41;
      if ( v35 == *(_QWORD *)v41 )
        goto LABEL_25;
      v114 = v122;
    }
    if ( v37 )
    {
      v90 = 128;
      goto LABEL_75;
    }
    v58 = *(unsigned int *)(v36 + 24);
LABEL_74:
    v90 = 16 * v58;
LABEL_75:
    v41 = v38 + v90;
LABEL_25:
    v43 = 128;
    if ( !v37 )
      v43 = 16LL * *(unsigned int *)(v36 + 24);
    if ( v41 == v38 + v43 )
    {
      v74 = v130[1];
      v75 = *(unsigned int *)(v74 + 24);
      v76 = *(_QWORD *)(v74 + 8);
      if ( (_DWORD)v75 )
      {
        v77 = 1;
        for ( k = (v75 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4)
                    | ((unsigned __int64)(((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4)))); ; k = (v75 - 1) & v80 )
        {
          v79 = (_QWORD *)(v76 + 24LL * k);
          if ( v35 == *v79 && v131 == (__int64 *)v79[1] )
            break;
          if ( *v79 == -4096 && v79[1] == -4096 )
            goto LABEL_145;
          v80 = v77 + k;
          ++v77;
        }
      }
      else
      {
LABEL_145:
        v79 = (_QWORD *)(v76 + 24 * v75);
      }
      v129[0] = (*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64, __int64 *))(**(_QWORD **)(v79[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v79[2] + 24LL),
                  v131,
                  v132,
                  v130);
      v128 = v35;
      sub_BBCF50((__int64)v133, v36, &v128, v129);
      if ( *(_BYTE *)(v134 + 8) )
      {
LABEL_147:
        v9 += 2;
        goto LABEL_80;
      }
    }
    else if ( *(_BYTE *)(v41 + 8) )
    {
      goto LABEL_147;
    }
    v44 = v9[3];
    v45 = *v130;
    v46 = *(_BYTE *)(*v130 + 8) & 1;
    if ( v46 )
    {
      v47 = v45 + 16;
      v48 = 7;
    }
    else
    {
      v59 = *(unsigned int *)(v45 + 24);
      v47 = *(_QWORD *)(v45 + 16);
      if ( !(_DWORD)v59 )
        goto LABEL_77;
      v48 = v59 - 1;
    }
    v49 = v48 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
    v50 = v47 + 16LL * v49;
    v51 = *(_QWORD *)v50;
    if ( v44 == *(_QWORD *)v50 )
      goto LABEL_32;
    v115 = 1;
    while ( v51 != -4096 )
    {
      v123 = v115 + 1;
      v49 = v48 & (v115 + v49);
      v50 = v47 + 16LL * v49;
      v51 = *(_QWORD *)v50;
      if ( v44 == *(_QWORD *)v50 )
        goto LABEL_32;
      v115 = v123;
    }
    if ( v46 )
    {
      v91 = 128;
      goto LABEL_78;
    }
    v59 = *(unsigned int *)(v45 + 24);
LABEL_77:
    v91 = 16 * v59;
LABEL_78:
    v50 = v47 + v91;
LABEL_32:
    v52 = 128;
    if ( !v46 )
      v52 = 16LL * *(unsigned int *)(v45 + 24);
    if ( v50 == v47 + v52 )
    {
      v81 = v130[1];
      v82 = *(unsigned int *)(v81 + 24);
      v83 = *(_QWORD *)(v81 + 8);
      if ( (_DWORD)v82 )
      {
        v84 = 1;
        for ( m = (v82 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4)
                    | ((unsigned __int64)(((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4)))); ; m = (v82 - 1) & v87 )
        {
          v86 = (_QWORD *)(v83 + 24LL * m);
          if ( v44 == *v86 && v131 == (__int64 *)v86[1] )
            break;
          if ( *v86 == -4096 && v86[1] == -4096 )
            goto LABEL_149;
          v87 = v84 + m;
          ++v84;
        }
      }
      else
      {
LABEL_149:
        v86 = (_QWORD *)(v83 + 24 * v82);
      }
      v129[0] = (*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64, __int64 *))(**(_QWORD **)(v86[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v86[2] + 24LL),
                  v131,
                  v132,
                  v130);
      v128 = v44;
      sub_BBCF50((__int64)v133, v45, &v128, v129);
      if ( *(_BYTE *)(v134 + 8) )
      {
LABEL_151:
        v9 += 3;
        goto LABEL_80;
      }
    }
    else if ( *(_BYTE *)(v50 + 8) )
    {
      goto LABEL_151;
    }
    v9 += 4;
    if ( !--v19 )
      break;
    v13 = v130;
    v16 = v132;
    v14 = (__int64)v131;
  }
  v18 = dest - v9;
LABEL_6:
  if ( v18 == 2 )
  {
LABEL_7:
    if ( !(unsigned __int8)sub_22D06C0(v130, *v9, (__int64)v131, v132) )
    {
      ++v9;
      goto LABEL_136;
    }
LABEL_80:
    if ( v9 == dest || (v92 = (__int64 **)(v9 + 1), v9 + 1 == dest) )
    {
LABEL_100:
      result = *a1;
      goto LABEL_101;
    }
    v93 = (__int64 **)dest;
    while ( 2 )
    {
      v101 = *a7;
      v102 = *v92;
      v103 = *(_BYTE *)(*a7 + 8) & 1;
      if ( v103 )
      {
        v94 = v101 + 16;
        v95 = 7;
        goto LABEL_84;
      }
      v104 = *(unsigned int *)(v101 + 24);
      v94 = *(_QWORD *)(v101 + 16);
      if ( (_DWORD)v104 )
      {
        v95 = v104 - 1;
LABEL_84:
        v96 = v95 & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
        v97 = v94 + 16LL * v96;
        v98 = *(__int64 **)v97;
        if ( v102 == *(__int64 **)v97 )
        {
LABEL_85:
          v99 = 128;
          if ( !v103 )
            v99 = 16LL * *(unsigned int *)(v101 + 24);
          if ( v97 == v94 + v99 )
          {
            v105 = a7[1];
            v106 = *(unsigned int *)(v105 + 24);
            v107 = *(_QWORD *)(v105 + 8);
            if ( (_DWORD)v106 )
            {
              v108 = 1;
              for ( n = (v106 - 1)
                      & (((0xBF58476D1CE4E5B9LL
                         * (((unsigned int)a8 >> 9) ^ ((unsigned int)a8 >> 4)
                          | ((unsigned __int64)(((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4)) << 32))) >> 31)
                       ^ (484763065 * (((unsigned int)a8 >> 9) ^ ((unsigned int)a8 >> 4)))); ; n = (v106 - 1) & v111 )
              {
                v110 = (__int64 **)(v107 + 24LL * n);
                if ( v102 == *v110 && v110[1] == a8 )
                  break;
                if ( *v110 == (__int64 *)-4096LL && v110[1] == (__int64 *)-4096LL )
                  goto LABEL_161;
                v111 = v108 + n;
                ++v108;
              }
            }
            else
            {
LABEL_161:
              v110 = (__int64 **)(v107 + 24 * v106);
            }
            v125 = v93;
            LOBYTE(v131) = (*(__int64 (__fastcall **)(__int64, __int64 *, __int64, __int64 *))(*(_QWORD *)v110[2][3]
                                                                                             + 16LL))(
                             v110[2][3],
                             a8,
                             a9,
                             a7);
            v130 = v102;
            sub_BBCF50((__int64)v133, v101, (__int64 *)&v130, &v131);
            v93 = v125;
            v100 = *(_BYTE *)(v134 + 8);
          }
          else
          {
            v100 = *(_BYTE *)(v97 + 8);
          }
          if ( !v100 )
            *v9++ = (__int64)*v92;
          if ( ++v92 == v93 )
            goto LABEL_100;
          continue;
        }
        v117 = 1;
        while ( v98 != (__int64 *)-4096LL )
        {
          v124 = v117 + 1;
          v96 = v95 & (v117 + v96);
          v97 = v94 + 16LL * v96;
          v98 = *(__int64 **)v97;
          if ( v102 == *(__int64 **)v97 )
            goto LABEL_85;
          v117 = v124;
        }
        if ( v103 )
        {
          v116 = 128;
          goto LABEL_124;
        }
        v104 = *(unsigned int *)(v101 + 24);
      }
      break;
    }
    v116 = 16 * v104;
LABEL_124:
    v97 = v94 + v116;
    goto LABEL_85;
  }
  if ( v18 == 3 )
  {
    if ( (unsigned __int8)sub_22D2B70((__int64)&v130, *v9) )
      goto LABEL_80;
    ++v9;
    goto LABEL_7;
  }
  if ( v18 != 1 )
  {
    v9 = dest;
    result = *a1;
    goto LABEL_4;
  }
LABEL_136:
  if ( (unsigned __int8)sub_22D06C0(v130, *v9, (__int64)v131, v132) )
    goto LABEL_80;
  v9 = dest;
  result = *a1;
LABEL_101:
  if ( ((result >> 2) & 1) != 0 )
  {
    if ( result )
    {
      if ( ((result >> 2) & 1) != 0 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v118 = result;
        if ( result )
        {
          result = *(_QWORD *)result;
          v119 = result + 8LL * *(unsigned int *)(v118 + 8) - (_QWORD)dest;
          if ( dest != (__int64 *)(result + 8LL * *(unsigned int *)(v118 + 8)) )
          {
            memmove(v9, dest, result + 8LL * *(unsigned int *)(v118 + 8) - (_QWORD)dest);
            result = *(_QWORD *)v118;
          }
          *(_DWORD *)(v118 + 8) = ((__int64)v9 + v119 - result) >> 3;
        }
      }
    }
  }
  else if ( a1 == v9 && v9 != dest )
  {
    result = (__int64)a1;
    *a1 = 0;
  }
  return result;
}
