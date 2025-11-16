// Function: sub_1386390
// Address: 0x1386390
//
void __fastcall sub_1386390(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 **v5; // r15
  const __m128i *v8; // r12
  __m128i v10; // xmm1
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // edi
  __int64 *v14; // rcx
  __int64 v15; // r11
  __int64 v16; // rsi
  __int64 *v17; // r13
  __int64 *i5; // r12
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // ecx
  __int64 v22; // r9
  int v23; // ecx
  unsigned int v24; // esi
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // r8
  unsigned int j; // eax
  __int64 v28; // r13
  __int64 v29; // r8
  unsigned int v30; // eax
  __int64 *v31; // r12
  __int64 *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // r13
  __int64 *i2; // r12
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 *v39; // r13
  __int64 *i3; // r12
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 *v43; // r13
  __int64 *i4; // r12
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 *v47; // r12
  __int64 *v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 *v51; // r13
  __int64 *kk; // r12
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 *v55; // r13
  __int64 *mm; // r12
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 *v59; // r13
  __int64 *nn; // r12
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 *v63; // r13
  __int64 *i1; // r12
  __int64 v65; // rdx
  __int64 v66; // rax
  int v67; // r9d
  unsigned int v68; // esi
  __int64 v69; // r10
  unsigned __int64 v70; // rdx
  unsigned __int64 v71; // rdx
  unsigned int k; // edi
  __int64 v73; // rax
  __int64 v74; // rdx
  unsigned int v75; // edi
  int v76; // edx
  int v77; // edx
  __int64 v78; // r10
  __int64 v79; // r11
  unsigned __int64 v80; // r8
  unsigned __int64 v81; // r8
  unsigned int i; // eax
  __int64 v83; // r8
  unsigned int v84; // eax
  int v85; // edx
  __m128i v86; // xmm3
  __int64 v87; // rax
  int v88; // esi
  int v89; // esi
  int v90; // esi
  __int64 v91; // r10
  __int64 v92; // r8
  int v93; // r11d
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // rdi
  unsigned int n; // edx
  __int64 v97; // rdi
  unsigned int v98; // edx
  int v99; // edi
  int v100; // edi
  __int64 v101; // rdx
  int v102; // r11d
  __int64 v103; // rcx
  unsigned __int64 v104; // r8
  unsigned __int64 v105; // r8
  unsigned int ii; // eax
  __int64 v107; // r8
  unsigned int v108; // eax
  __int64 v109; // r13
  __int64 *v110; // r10
  __int64 *v111; // r12
  __int64 *v112; // r13
  __int64 v113; // rax
  __int64 v114; // r15
  __int64 v115; // rbx
  __int64 *jj; // rax
  int v117; // edi
  int v118; // edi
  __int64 v119; // r11
  int v120; // r10d
  unsigned int m; // edx
  __int64 *v122; // rsi
  __int64 v123; // r8
  unsigned int v124; // edx
  unsigned __int64 v125; // [rsp+0h] [rbp-F0h]
  int v126; // [rsp+0h] [rbp-F0h]
  __int64 v127; // [rsp+0h] [rbp-F0h]
  const __m128i *v128; // [rsp+0h] [rbp-F0h]
  int v129; // [rsp+8h] [rbp-E8h]
  int v130; // [rsp+8h] [rbp-E8h]
  __int64 v131; // [rsp+8h] [rbp-E8h]
  __int64 v132; // [rsp+8h] [rbp-E8h]
  int v133; // [rsp+8h] [rbp-E8h]
  __int64 **v134; // [rsp+8h] [rbp-E8h]
  __int64 v135; // [rsp+8h] [rbp-E8h]
  __int64 v136; // [rsp+10h] [rbp-E0h]
  __int64 v137; // [rsp+10h] [rbp-E0h]
  int v138; // [rsp+10h] [rbp-E0h]
  int v139; // [rsp+10h] [rbp-E0h]
  __int64 v140; // [rsp+10h] [rbp-E0h]
  int v141; // [rsp+10h] [rbp-E0h]
  __int64 v143; // [rsp+28h] [rbp-C8h] BYREF
  __m128i v144; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v145; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v146; // [rsp+50h] [rbp-A0h] BYREF
  char v147; // [rsp+60h] [rbp-90h]
  __int64 v148; // [rsp+70h] [rbp-80h] BYREF
  __int64 v149; // [rsp+78h] [rbp-78h]
  char v150; // [rsp+80h] [rbp-70h]
  __m128i v151; // [rsp+90h] [rbp-60h] BYREF
  __m128i *v152; // [rsp+A0h] [rbp-50h]
  __int64 v153; // [rsp+A8h] [rbp-48h]
  __int64 v154; // [rsp+B0h] [rbp-40h]

  v5 = 0;
  v8 = a1;
  v10 = _mm_loadu_si128(a1 + 1);
  v11 = *(unsigned int *)(a2 + 24);
  v145 = v10;
  v144 = _mm_loadu_si128(a1);
  if ( !(_DWORD)v11 )
    goto LABEL_6;
  v12 = *(_QWORD *)(a2 + 8);
  v13 = (v11 - 1) & (((unsigned __int32)v145.m128i_i32[0] >> 9) ^ ((unsigned __int32)v145.m128i_i32[0] >> 4));
  v14 = (__int64 *)(v12 + 32LL * v13);
  v15 = *v14;
  if ( v145.m128i_i64[0] == *v14 )
  {
LABEL_3:
    if ( v14 != (__int64 *)(v12 + 32 * v11) )
    {
      v16 = v14[1];
      if ( -1227133513 * (unsigned int)((v14[2] - v16) >> 3) > v10.m128i_i32[2] )
      {
        v5 = (__int64 **)(v16 + 56LL * v10.m128i_u32[2]);
        goto LABEL_6;
      }
    }
  }
  else
  {
    v21 = 1;
    while ( v15 != -8 )
    {
      v67 = v21 + 1;
      v13 = (v11 - 1) & (v21 + v13);
      v14 = (__int64 *)(v12 + 32LL * v13);
      v15 = *v14;
      if ( v145.m128i_i64[0] == *v14 )
        goto LABEL_3;
      v21 = v67;
    }
  }
  v5 = 0;
LABEL_6:
  sub_1381F10((__int64)&v146, a2, v144.m128i_i64[0], v144.m128i_i32[2]);
  sub_1381F10((__int64)&v148, a2, v145.m128i_i64[0], v145.m128i_i32[2]);
  if ( !v147 || !v150 )
    goto LABEL_8;
  v22 = v148;
  v23 = v149;
  v24 = *(_DWORD *)(a4 + 24);
  v151 = _mm_loadu_si128(&v146);
  if ( !v24 )
  {
    ++*(_QWORD *)a4;
LABEL_62:
    v131 = v22;
    v138 = v23;
    sub_1385DE0(a4, 2 * v24);
    v76 = *(_DWORD *)(a4 + 24);
    if ( !v76 )
    {
      ++*(_DWORD *)(a4 + 16);
      BUG();
    }
    v77 = v76 - 1;
    v78 = 0;
    v126 = 1;
    v79 = *(_QWORD *)(a4 + 8);
    v23 = v138;
    v22 = v131;
    v80 = ((((unsigned int)(37 * v151.m128i_i32[2])
           | ((unsigned __int64)(((unsigned __int32)v151.m128i_i32[0] >> 9) ^ ((unsigned __int32)v151.m128i_i32[0] >> 4)) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * v151.m128i_i32[2]) << 32)) >> 22)
        ^ (((unsigned int)(37 * v151.m128i_i32[2])
          | ((unsigned __int64)(((unsigned __int32)v151.m128i_i32[0] >> 9) ^ ((unsigned __int32)v151.m128i_i32[0] >> 4)) << 32))
         - 1
         - ((unsigned __int64)(unsigned int)(37 * v151.m128i_i32[2]) << 32));
    v81 = ((9 * (((v80 - 1 - (v80 << 13)) >> 8) ^ (v80 - 1 - (v80 << 13)))) >> 15)
        ^ (9 * (((v80 - 1 - (v80 << 13)) >> 8) ^ (v80 - 1 - (v80 << 13))));
    for ( i = v77 & (((v81 - 1 - (v81 << 27)) >> 31) ^ (v81 - 1 - ((_DWORD)v81 << 27))); ; i = v77 & v84 )
    {
      v28 = v79 + 48LL * i;
      v83 = *(_QWORD *)v28;
      if ( v151.m128i_i64[0] == *(_QWORD *)v28 && v151.m128i_i32[2] == *(_DWORD *)(v28 + 8) )
      {
        v85 = *(_DWORD *)(a4 + 16) + 1;
        goto LABEL_74;
      }
      if ( v83 == -8 )
      {
        if ( *(_DWORD *)(v28 + 8) == -1 )
        {
          v85 = *(_DWORD *)(a4 + 16) + 1;
          if ( v78 )
            v28 = v78;
LABEL_74:
          *(_DWORD *)(a4 + 16) = v85;
          if ( *(_QWORD *)v28 != -8 || *(_DWORD *)(v28 + 8) != -1 )
            --*(_DWORD *)(a4 + 20);
          v86 = _mm_loadu_si128(&v151);
          *(_QWORD *)(v28 + 16) = 0;
          v69 = v28 + 16;
          v87 = 1;
          *(_QWORD *)(v28 + 24) = 0;
          *(_QWORD *)(v28 + 32) = 0;
          *(_DWORD *)(v28 + 40) = 0;
          *(__m128i *)v28 = v86;
LABEL_77:
          *(_QWORD *)(v28 + 16) = v87;
          v88 = 0;
          goto LABEL_78;
        }
      }
      else if ( v83 == -16 && *(_DWORD *)(v28 + 8) == -2 && !v78 )
      {
        v78 = v79 + 48LL * i;
      }
      v84 = v126 + i;
      ++v126;
    }
  }
  v129 = 1;
  v136 = 0;
  v25 = ((((unsigned int)(37 * v151.m128i_i32[2])
         | ((unsigned __int64)(((unsigned __int32)v151.m128i_i32[0] >> 9) ^ ((unsigned __int32)v151.m128i_i32[0] >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v151.m128i_i32[2]) << 32)) >> 22)
      ^ (((unsigned int)(37 * v151.m128i_i32[2])
        | ((unsigned __int64)(((unsigned __int32)v151.m128i_i32[0] >> 9) ^ ((unsigned __int32)v151.m128i_i32[0] >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v151.m128i_i32[2]) << 32));
  v26 = ((9 * (((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13)))) >> 15)
      ^ (9 * (((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13))));
  for ( j = (v24 - 1) & (((v26 - 1 - (v26 << 27)) >> 31) ^ (v26 - 1 - ((_DWORD)v26 << 27))); ; j = (v24 - 1) & v30 )
  {
    v28 = *(_QWORD *)(a4 + 8) + 48LL * j;
    v29 = *(_QWORD *)v28;
    if ( v151.m128i_i64[0] == *(_QWORD *)v28 && v151.m128i_i32[2] == *(_DWORD *)(v28 + 8) )
      break;
    if ( v29 == -8 )
    {
      if ( *(_DWORD *)(v28 + 8) == -1 )
      {
        if ( v136 )
          v28 = v136;
        ++*(_QWORD *)a4;
        v85 = *(_DWORD *)(a4 + 16) + 1;
        if ( 4 * v85 < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(a4 + 20) - v85 <= v24 >> 3 )
          {
            v127 = v22;
            v133 = v23;
            sub_1385DE0(a4, v24);
            sub_1383EC0(a4, v151.m128i_i64, &v143);
            v28 = v143;
            v22 = v127;
            v23 = v133;
            v85 = *(_DWORD *)(a4 + 16) + 1;
          }
          goto LABEL_74;
        }
        goto LABEL_62;
      }
    }
    else if ( v29 == -16 && *(_DWORD *)(v28 + 8) == -2 )
    {
      if ( v136 )
        v28 = v136;
      v136 = v28;
    }
    v30 = v129 + j;
    ++v129;
  }
  v68 = *(_DWORD *)(v28 + 40);
  v69 = v28 + 16;
  if ( !v68 )
  {
    v87 = *(_QWORD *)(v28 + 16) + 1LL;
    goto LABEL_77;
  }
  v130 = 1;
  v137 = 0;
  v70 = ((((unsigned int)(37 * v149) | ((unsigned __int64)(((unsigned int)v148 >> 9) ^ ((unsigned int)v148 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v149) << 32)) >> 22)
      ^ (((unsigned int)(37 * v149) | ((unsigned __int64)(((unsigned int)v148 >> 9) ^ ((unsigned int)v148 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v149) << 32));
  v71 = ((9 * (((v70 - 1 - (v70 << 13)) >> 8) ^ (v70 - 1 - (v70 << 13)))) >> 15)
      ^ (9 * (((v70 - 1 - (v70 << 13)) >> 8) ^ (v70 - 1 - (v70 << 13))));
  v125 = ((v71 - 1 - (v71 << 27)) >> 31) ^ (v71 - 1 - (v71 << 27));
  for ( k = (v68 - 1) & v125; ; k = (v68 - 1) & v75 )
  {
    v73 = *(_QWORD *)(v28 + 24) + 16LL * k;
    v74 = *(_QWORD *)v73;
    if ( *(_QWORD *)v73 == v148 && (_DWORD)v149 == *(_DWORD *)(v73 + 8) )
      goto LABEL_8;
    if ( v74 == -8 )
      break;
    if ( v74 == -16 && *(_DWORD *)(v73 + 8) == -2 )
    {
      if ( v137 )
        v73 = v137;
      v137 = v73;
    }
LABEL_58:
    v75 = v130 + k;
    ++v130;
  }
  if ( *(_DWORD *)(v73 + 8) != -1 )
    goto LABEL_58;
  if ( v137 )
    v73 = v137;
  v99 = *(_DWORD *)(v28 + 32);
  ++*(_QWORD *)(v28 + 16);
  v100 = v99 + 1;
  if ( 4 * v100 < 3 * v68 )
  {
    if ( v68 - *(_DWORD *)(v28 + 36) - v100 > v68 >> 3 )
      goto LABEL_91;
    v135 = v22;
    v141 = v23;
    sub_1386100(v28 + 16, v68);
    v117 = *(_DWORD *)(v28 + 40);
    if ( v117 )
    {
      v118 = v117 - 1;
      v119 = *(_QWORD *)(v28 + 24);
      v73 = 0;
      v23 = v141;
      v22 = v135;
      v120 = 1;
      for ( m = v118 & v125; ; m = v118 & v124 )
      {
        v122 = (__int64 *)(v119 + 16LL * m);
        v123 = *v122;
        if ( *v122 == v135 && v141 == *((_DWORD *)v122 + 2) )
        {
          v100 = *(_DWORD *)(v28 + 32) + 1;
          v73 = v119 + 16LL * m;
          goto LABEL_91;
        }
        if ( v123 == -8 )
        {
          if ( *((_DWORD *)v122 + 2) == -1 )
          {
            v100 = *(_DWORD *)(v28 + 32) + 1;
            if ( !v73 )
              v73 = v119 + 16LL * m;
            goto LABEL_91;
          }
        }
        else if ( v123 == -16 && *((_DWORD *)v122 + 2) == -2 && !v73 )
        {
          v73 = v119 + 16LL * m;
        }
        v124 = v120 + m;
        ++v120;
      }
    }
LABEL_172:
    ++*(_DWORD *)(v28 + 32);
    BUG();
  }
  v88 = 2 * v68;
LABEL_78:
  v132 = v22;
  v139 = v23;
  sub_1386100(v69, v88);
  v89 = *(_DWORD *)(v28 + 40);
  if ( !v89 )
    goto LABEL_172;
  v23 = v139;
  v90 = v89 - 1;
  v91 = *(_QWORD *)(v28 + 24);
  v92 = 0;
  v22 = v132;
  v93 = 1;
  v94 = ((((unsigned int)(37 * v139) | ((unsigned __int64)(((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v139) << 32)) >> 22)
      ^ (((unsigned int)(37 * v139) | ((unsigned __int64)(((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v139) << 32));
  v95 = ((9 * (((v94 - 1 - (v94 << 13)) >> 8) ^ (v94 - 1 - (v94 << 13)))) >> 15)
      ^ (9 * (((v94 - 1 - (v94 << 13)) >> 8) ^ (v94 - 1 - (v94 << 13))));
  for ( n = v90 & (((v95 - 1 - (v95 << 27)) >> 31) ^ (v95 - 1 - ((_DWORD)v95 << 27))); ; n = v90 & v98 )
  {
    v73 = v91 + 16LL * n;
    v97 = *(_QWORD *)v73;
    if ( *(_QWORD *)v73 == v132 && v139 == *(_DWORD *)(v73 + 8) )
      break;
    if ( v97 == -8 )
    {
      if ( *(_DWORD *)(v73 + 8) == -1 )
      {
        if ( v92 )
          v73 = v92;
        v100 = *(_DWORD *)(v28 + 32) + 1;
        goto LABEL_91;
      }
    }
    else if ( v97 == -16 && *(_DWORD *)(v73 + 8) == -2 && !v92 )
    {
      v92 = v91 + 16LL * n;
    }
    v98 = v93 + n;
    ++v93;
  }
  v100 = *(_DWORD *)(v28 + 32) + 1;
LABEL_91:
  *(_DWORD *)(v28 + 32) = v100;
  if ( *(_QWORD *)v73 != -8 || *(_DWORD *)(v73 + 8) != -1 )
    --*(_DWORD *)(v28 + 36);
  *(_QWORD *)v73 = v22;
  *(_DWORD *)(v73 + 8) = v23;
  sub_1385450(v146.m128i_i64[0], v146.m128i_i32[2], v148, v149, 1u, a3, a5);
  v101 = *(unsigned int *)(a3 + 24);
  if ( (_DWORD)v101 )
  {
    v102 = 1;
    v103 = *(_QWORD *)(a3 + 8);
    v104 = ((((unsigned int)(37 * v146.m128i_i32[2])
            | ((unsigned __int64)(((unsigned __int32)v146.m128i_i32[0] >> 9) ^ ((unsigned __int32)v146.m128i_i32[0] >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v146.m128i_i32[2]) << 32)) >> 22)
         ^ (((unsigned int)(37 * v146.m128i_i32[2])
           | ((unsigned __int64)(((unsigned __int32)v146.m128i_i32[0] >> 9) ^ ((unsigned __int32)v146.m128i_i32[0] >> 4)) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * v146.m128i_i32[2]) << 32));
    v105 = ((9 * (((v104 - 1 - (v104 << 13)) >> 8) ^ (v104 - 1 - (v104 << 13)))) >> 15)
         ^ (9 * (((v104 - 1 - (v104 << 13)) >> 8) ^ (v104 - 1 - (v104 << 13))));
    for ( ii = (v101 - 1) & (((v105 - 1 - (v105 << 27)) >> 31) ^ (v105 - 1 - ((_DWORD)v105 << 27)));
          ;
          ii = (v101 - 1) & v108 )
    {
      v107 = v103 + 48LL * ii;
      if ( *(_QWORD *)v107 == v146.m128i_i64[0] && *(_DWORD *)(v107 + 8) == v146.m128i_i32[2] )
        break;
      if ( *(_QWORD *)v107 == -8 && *(_DWORD *)(v107 + 8) == -1 )
        goto LABEL_8;
      v108 = v102 + ii;
      ++v102;
    }
    if ( v107 != 48 * v101 + v103 )
    {
      v109 = *(_QWORD *)(v107 + 24);
      v110 = (__int64 *)(v109 + 24LL * *(unsigned int *)(v107 + 40));
      if ( *(_DWORD *)(v107 + 32) )
      {
        if ( (__int64 *)v109 != v110 )
        {
          do
          {
            if ( *(_QWORD *)v109 == -8 )
            {
              if ( *(_DWORD *)(v109 + 8) != -1 )
                goto LABEL_106;
            }
            else if ( *(_QWORD *)v109 != -16 || *(_DWORD *)(v109 + 8) != -2 )
            {
LABEL_106:
              if ( v110 == (__int64 *)v109 )
                break;
              v128 = v8;
              v111 = (__int64 *)v109;
              v112 = v110;
              v134 = v5;
              v140 = a5;
LABEL_110:
              v113 = v111[2];
              v114 = *v111;
              v115 = v111[1];
              if ( (v113 & 1) != 0 )
              {
                sub_1385450(v114, v115, v148, v149, 2u, a3, v140);
                v113 = v111[2];
              }
              if ( (v113 & 8) != 0 )
              {
                sub_1385450(v114, v115, v148, v149, 5u, a3, v140);
                if ( (v111[2] & 0x10) != 0 )
                  goto LABEL_120;
              }
              else
              {
                if ( (v113 & 0x10) == 0 )
                  goto LABEL_114;
LABEL_120:
                sub_1385450(v114, v115, v148, v149, 6u, a3, v140);
              }
LABEL_114:
              for ( jj = v111 + 3; v112 != jj; jj += 3 )
              {
                v111 = jj;
                if ( *jj == -8 )
                {
                  if ( *((_DWORD *)jj + 2) != -1 )
                    goto LABEL_109;
                }
                else if ( *jj != -16 || *((_DWORD *)jj + 2) != -2 )
                {
LABEL_109:
                  if ( jj == v112 )
                    break;
                  goto LABEL_110;
                }
              }
              v5 = v134;
              v8 = v128;
              a5 = v140;
              break;
            }
            v109 += 24;
          }
          while ( v110 != (__int64 *)v109 );
        }
      }
    }
  }
LABEL_8:
  v153 = a3;
  v154 = a5;
  v151.m128i_i64[0] = a4;
  v151.m128i_i64[1] = (__int64)&v145;
  v152 = &v144;
  switch ( v8[2].m128i_i8[0] )
  {
    case 0:
      v51 = v5[4];
      for ( kk = v5[3]; kk != v51; kk += 3 )
      {
        v53 = *kk;
        v54 = kk[1];
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v53, v54, 0, a3, a5);
      }
      v55 = v5[1];
      for ( mm = *v5; v55 != mm; mm += 3 )
      {
        v57 = mm[1];
        v58 = *mm;
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v58, v57, 4u, a3, a5);
      }
      sub_1385BF0(v151.m128i_i64, 2u);
      break;
    case 1:
      v59 = v5[4];
      for ( nn = v5[3]; v59 != nn; nn += 3 )
      {
        v61 = nn[1];
        v62 = *nn;
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v62, v61, 0, a3, a5);
      }
      v63 = v5[1];
      for ( i1 = *v5; i1 != v63; i1 += 3 )
      {
        v65 = i1[1];
        v66 = *i1;
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v66, v65, 3u, a3, a5);
      }
      break;
    case 2:
      v35 = v5[4];
      for ( i2 = v5[3]; v35 != i2; i2 += 3 )
      {
        v37 = *i2;
        v38 = i2[1];
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v37, v38, 0, a3, a5);
      }
      v39 = v5[1];
      for ( i3 = *v5; v39 != i3; i3 += 3 )
      {
        v41 = i3[1];
        v42 = *i3;
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v42, v41, 4u, a3, a5);
      }
      break;
    case 3:
      v43 = v5[1];
      for ( i4 = *v5; v43 != i4; i4 += 3 )
      {
        v45 = i4[1];
        v46 = *i4;
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v46, v45, 3u, a3, a5);
      }
      sub_1385BF0(v151.m128i_i64, 5u);
      break;
    case 4:
      v17 = v5[1];
      for ( i5 = *v5; v17 != i5; i5 += 3 )
      {
        v19 = i5[1];
        v20 = *i5;
        sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v20, v19, 4u, a3, a5);
      }
      sub_1385BF0(v151.m128i_i64, 6u);
      break;
    case 5:
      v47 = *v5;
      v48 = v5[1];
      if ( *v5 != v48 )
      {
        do
        {
          v49 = v47[1];
          v50 = *v47;
          v47 += 3;
          sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v50, v49, 3u, a3, a5);
        }
        while ( v47 != v48 );
      }
      break;
    case 6:
      v31 = *v5;
      v32 = v5[1];
      if ( *v5 != v32 )
      {
        do
        {
          v33 = *v31;
          v34 = v31[1];
          v31 += 3;
          sub_1385450(v144.m128i_i64[0], v144.m128i_i32[2], v33, v34, 4u, a3, a5);
        }
        while ( v31 != v32 );
      }
      break;
    default:
      return;
  }
}
