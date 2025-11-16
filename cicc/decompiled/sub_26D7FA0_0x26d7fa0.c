// Function: sub_26D7FA0
// Address: 0x26d7fa0
//
unsigned __int64 *__fastcall sub_26D7FA0(unsigned __int64 *a1, __int64 a2, __int64 **a3, __int64 a4)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 v7; // rsi
  __int64 *v8; // r13
  __int64 *v9; // rbx
  bool v10; // zf
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r14
  char *v13; // rdi
  signed __int64 v14; // r14
  char *v15; // rdi
  unsigned __int64 v16; // r14
  char *v17; // rdi
  void *v18; // r14
  size_t v19; // rdx
  char *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  char **v25; // rdx
  char *v26; // r10
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rax
  __int64 v29; // r10
  unsigned int v30; // edx
  __int64 v31; // rcx
  __int64 v32; // rdi
  int v33; // r13d
  int v34; // eax
  int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 *v38; // r15
  __int64 *v39; // r13
  __int64 *v40; // rbx
  _BYTE *v41; // rsi
  __int64 v42; // rdi
  _BYTE *v43; // rsi
  __int64 v44; // rax
  __int64 v45; // r15
  __int64 v46; // rdi
  __int64 v47; // r15
  __int64 v48; // r14
  unsigned __int64 v49; // r15
  __int64 v50; // r9
  int v51; // r11d
  __int64 v52; // rcx
  unsigned int v53; // edx
  __int64 v54; // rdi
  __int64 v55; // r8
  __int64 *v56; // rax
  unsigned __int64 v57; // rax
  int v58; // edx
  _BYTE *v59; // rbx
  _BYTE *v60; // rax
  __int64 *v61; // rdx
  __int64 v62; // rax
  __int64 v63; // r12
  unsigned __int64 v64; // rbx
  int v65; // edx
  _BYTE *v66; // rbx
  unsigned int v67; // esi
  int v68; // edx
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rcx
  __int64 v71; // rax
  int v73; // eax
  __int64 *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rbx
  unsigned int v78; // esi
  __int64 v79; // r8
  __int64 *v80; // rbx
  __int64 *v81; // r13
  __m128i *v82; // rsi
  __int64 v83; // rcx
  unsigned int v84; // esi
  __int64 v85; // rdi
  unsigned int v86; // r8d
  unsigned int v87; // eax
  __int64 v88; // rdx
  __int64 v89; // rcx
  int v90; // r11d
  __int64 *v91; // r10
  unsigned int v92; // edx
  __int64 *v93; // rax
  __int64 v94; // r9
  __int64 v95; // rax
  __int64 *v96; // rax
  int v97; // eax
  int v98; // eax
  int v99; // r9d
  int v100; // esi
  int v101; // esi
  __int64 v102; // r8
  unsigned int v103; // edx
  __int64 v104; // rdi
  int v105; // r9d
  __int64 *v106; // r11
  int v107; // esi
  int v108; // esi
  __int64 v109; // r8
  int v110; // r11d
  __int64 *v111; // r9
  unsigned int v112; // edx
  int v113; // edx
  int v114; // ecx
  __int64 v115; // [rsp+0h] [rbp-C0h]
  size_t v117; // [rsp+18h] [rbp-A8h]
  __int64 *v118; // [rsp+18h] [rbp-A8h]
  __int64 *v119; // [rsp+20h] [rbp-A0h]
  unsigned __int64 *v120; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v122; // [rsp+28h] [rbp-98h]
  __int64 v123; // [rsp+30h] [rbp-90h] BYREF
  __int64 v124; // [rsp+38h] [rbp-88h] BYREF
  __m128i v125; // [rsp+40h] [rbp-80h] BYREF
  __m128i v126; // [rsp+50h] [rbp-70h] BYREF
  void *src; // [rsp+60h] [rbp-60h]
  _BYTE *v128; // [rsp+68h] [rbp-58h]
  __int64 v129; // [rsp+70h] [rbp-50h]
  void *v130; // [rsp+78h] [rbp-48h]
  _BYTE *v131; // [rsp+80h] [rbp-40h]
  __int64 v132; // [rsp+88h] [rbp-38h]

  v5 = a1;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  v7 = a3[1] - *a3;
  sub_26D0110(a1, v7);
  v8 = a3[1];
  v9 = *a3;
  if ( v9 == v8 )
    goto LABEL_43;
  v119 = v8;
  v115 = a4;
  do
  {
    v22 = *(_QWORD *)(a2 + 16);
    v15 = (char *)*v9;
    v125.m128i_i64[1] = 0;
    v126.m128i_i16[0] = 1;
    v126.m128i_i64[1] = 0;
    src = 0;
    v128 = 0;
    v129 = 0;
    v130 = 0;
    v131 = 0;
    v132 = 0;
    v23 = *(_QWORD *)(v22 + 8);
    v24 = *(unsigned int *)(v22 + 24);
    if ( (_DWORD)v24 )
    {
      v7 = ((_DWORD)v24 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v25 = (char **)(v23 + 16 * v7);
      v26 = *v25;
      if ( v15 == *v25 )
      {
LABEL_21:
        if ( v25 != (char **)(v23 + 16 * v24) )
        {
          v126.m128i_i8[0] = 0;
          v125.m128i_i64[1] = (__int64)v25[1];
        }
      }
      else
      {
        v113 = 1;
        while ( v26 != (char *)-4096LL )
        {
          v114 = v113 + 1;
          v7 = ((_DWORD)v24 - 1) & (unsigned int)(v113 + v7);
          v25 = (char **)(v23 + 16LL * (unsigned int)v7);
          v26 = *v25;
          if ( v15 == *v25 )
            goto LABEL_21;
          v113 = v114;
        }
      }
    }
    v27 = v5[1];
    v28 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v27 - *v5) >> 4);
    v125.m128i_i64[0] = v28;
    if ( v27 == v5[2] )
    {
      sub_26D33A0(v5, (_QWORD *)v27, (__int64)&v125);
      v18 = v130;
      v7 = v132 - (_QWORD)v130;
    }
    else
    {
      if ( !v27 )
      {
        v5[1] = 80;
        goto LABEL_16;
      }
      *(_QWORD *)v27 = v28;
      *(_QWORD *)(v27 + 8) = v125.m128i_i64[1];
      *(_WORD *)(v27 + 16) = v126.m128i_i16[0];
      *(_QWORD *)(v27 + 24) = v126.m128i_i64[1];
      v11 = v128 - (_BYTE *)src;
      v10 = v128 == src;
      *(_QWORD *)(v27 + 32) = 0;
      *(_QWORD *)(v27 + 40) = 0;
      v12 = v11;
      *(_QWORD *)(v27 + 48) = 0;
      if ( v10 )
      {
        v13 = 0;
      }
      else
      {
        if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_167;
        v13 = (char *)sub_22077B0(v11);
      }
      *(_QWORD *)(v27 + 32) = v13;
      *(_QWORD *)(v27 + 40) = v13;
      *(_QWORD *)(v27 + 48) = &v13[v12];
      v7 = (unsigned __int64)src;
      v14 = v128 - (_BYTE *)src;
      if ( v128 != src )
        v13 = (char *)memmove(v13, src, v128 - (_BYTE *)src);
      v15 = &v13[v14];
      *(_QWORD *)(v27 + 40) = v15;
      v11 = v131 - (_BYTE *)v130;
      v10 = v131 == v130;
      *(_QWORD *)(v27 + 56) = 0;
      *(_QWORD *)(v27 + 64) = 0;
      v16 = v11;
      *(_QWORD *)(v27 + 72) = 0;
      if ( v10 )
      {
        v17 = 0;
      }
      else
      {
        if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_167:
          sub_4261EA(v15, v7, v11);
        v17 = (char *)sub_22077B0(v11);
      }
      *(_QWORD *)(v27 + 56) = v17;
      *(_QWORD *)(v27 + 72) = &v17[v16];
      *(_QWORD *)(v27 + 64) = v17;
      v18 = v130;
      v19 = v131 - (_BYTE *)v130;
      if ( v131 != v130 )
      {
        v117 = v131 - (_BYTE *)v130;
        v20 = (char *)memmove(v17, v130, v19);
        v19 = v117;
        v17 = v20;
      }
      *(_QWORD *)(v27 + 64) = &v17[v19];
      v21 = v132;
      v5[1] += 80LL;
      v7 = v21 - (_QWORD)v18;
    }
    if ( v18 )
      j_j___libc_free_0((unsigned __int64)v18);
LABEL_16:
    if ( src )
    {
      v7 = v129 - (_QWORD)src;
      j_j___libc_free_0((unsigned __int64)src);
    }
    ++v9;
  }
  while ( v119 != v9 );
  v118 = a3[1];
  if ( v118 != *a3 )
  {
    v38 = *a3;
    while ( 1 )
    {
      v76 = *v38;
      v77 = *(_QWORD *)(a2 + 8);
      v123 = *v38;
      v78 = *(_DWORD *)(v77 + 24);
      if ( !v78 )
        break;
      v79 = *(_QWORD *)(v77 + 8);
      v33 = 1;
      v29 = 0;
      v30 = (v78 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v31 = v79 + 88LL * v30;
      v32 = *(_QWORD *)v31;
      if ( v76 == *(_QWORD *)v31 )
      {
LABEL_102:
        v80 = *(__int64 **)(v31 + 8);
        v81 = &v80[*(unsigned int *)(v31 + 16)];
        if ( v80 != v81 )
        {
          while ( 1 )
          {
            v83 = *v80;
            v84 = *(_DWORD *)(v115 + 24);
            v85 = *(_QWORD *)(v115 + 8);
            v124 = *v80;
            if ( v84 )
            {
              v86 = v84 - 1;
              v87 = (v84 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
              v88 = *(_QWORD *)(v85 + 16LL * v87);
              if ( v83 == v88 )
              {
LABEL_110:
                v89 = v123;
                v126.m128i_i64[0] = 0;
                v90 = 1;
                v126.m128i_i16[4] = 1;
                v91 = 0;
                src = 0;
                v92 = v86 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
                v93 = (__int64 *)(v85 + 16LL * v92);
                v94 = *v93;
                if ( v123 == *v93 )
                {
LABEL_111:
                  v95 = v93[1];
                }
                else
                {
                  while ( v94 != -4096 )
                  {
                    if ( !v91 && v94 == -8192 )
                      v91 = v93;
                    v92 = v86 & (v90 + v92);
                    v93 = (__int64 *)(v85 + 16LL * v92);
                    v94 = *v93;
                    if ( v123 == *v93 )
                      goto LABEL_111;
                    ++v90;
                  }
                  if ( !v91 )
                    v91 = v93;
                  v97 = *(_DWORD *)(v115 + 16);
                  ++*(_QWORD *)v115;
                  v98 = v97 + 1;
                  if ( 4 * v98 >= 3 * v84 )
                  {
                    sub_FE19E0(v115, 2 * v84);
                    v100 = *(_DWORD *)(v115 + 24);
                    if ( !v100 )
                      goto LABEL_168;
                    v89 = v123;
                    v101 = v100 - 1;
                    v102 = *(_QWORD *)(v115 + 8);
                    v103 = v101 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
                    v98 = *(_DWORD *)(v115 + 16) + 1;
                    v91 = (__int64 *)(v102 + 16LL * v103);
                    v104 = *v91;
                    if ( *v91 != v123 )
                    {
                      v105 = 1;
                      v106 = 0;
                      while ( v104 != -4096 )
                      {
                        if ( !v106 && v104 == -8192 )
                          v106 = v91;
                        v103 = v101 & (v105 + v103);
                        v91 = (__int64 *)(v102 + 16LL * v103);
                        v104 = *v91;
                        if ( v123 == *v91 )
                          goto LABEL_124;
                        ++v105;
                      }
                      if ( v106 )
                        v91 = v106;
                    }
                  }
                  else if ( v84 - *(_DWORD *)(v115 + 20) - v98 <= v84 >> 3 )
                  {
                    sub_FE19E0(v115, v84);
                    v107 = *(_DWORD *)(v115 + 24);
                    if ( !v107 )
                    {
LABEL_168:
                      ++*(_DWORD *)(v115 + 16);
                      BUG();
                    }
                    v108 = v107 - 1;
                    v109 = *(_QWORD *)(v115 + 8);
                    v110 = 1;
                    v111 = 0;
                    v112 = v108 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
                    v98 = *(_DWORD *)(v115 + 16) + 1;
                    v91 = (__int64 *)(v109 + 16LL * v112);
                    v89 = *v91;
                    if ( v123 != *v91 )
                    {
                      while ( v89 != -4096 )
                      {
                        if ( v89 == -8192 && !v111 )
                          v111 = v91;
                        v112 = v108 & (v110 + v112);
                        v91 = (__int64 *)(v109 + 16LL * v112);
                        v89 = *v91;
                        if ( v123 == *v91 )
                          goto LABEL_124;
                        ++v110;
                      }
                      v89 = v123;
                      if ( v111 )
                        v91 = v111;
                    }
                  }
LABEL_124:
                  *(_DWORD *)(v115 + 16) = v98;
                  if ( *v91 != -4096 )
                    --*(_DWORD *)(v115 + 20);
                  *v91 = v89;
                  v95 = 0;
                  v91[1] = 0;
                }
                v125.m128i_i64[0] = v95;
                v96 = sub_26CC460(v115, &v124);
                v82 = (__m128i *)v5[4];
                v125.m128i_i64[1] = *v96;
                if ( v82 == (__m128i *)v5[5] )
                {
                  sub_26D37E0(v5 + 3, v82, &v125);
                }
                else
                {
                  if ( v82 )
                  {
                    *v82 = _mm_loadu_si128(&v125);
                    v82[1] = _mm_loadu_si128(&v126);
                    v82[2].m128i_i64[0] = (__int64)src;
                    v82 = (__m128i *)v5[4];
                  }
                  v5[4] = (unsigned __int64)&v82[2].m128i_u64[1];
                }
              }
              else
              {
                v99 = 1;
                while ( v88 != -4096 )
                {
                  v87 = v86 & (v99 + v87);
                  v88 = *(_QWORD *)(v85 + 16LL * v87);
                  if ( v83 == v88 )
                    goto LABEL_110;
                  ++v99;
                }
              }
            }
            if ( v81 == ++v80 )
              goto LABEL_42;
          }
        }
        goto LABEL_42;
      }
      while ( v32 != -4096 )
      {
        if ( !v29 && v32 == -8192 )
          v29 = v31;
        v30 = (v78 - 1) & (v33 + v30);
        v31 = v79 + 88LL * v30;
        v32 = *(_QWORD *)v31;
        if ( v76 == *(_QWORD *)v31 )
          goto LABEL_102;
        ++v33;
      }
      if ( !v29 )
        v29 = v31;
      v125.m128i_i64[0] = v29;
      v34 = *(_DWORD *)(v77 + 16);
      ++*(_QWORD *)v77;
      v35 = v34 + 1;
      if ( 4 * (v34 + 1) >= 3 * v78 )
        goto LABEL_153;
      if ( v78 - *(_DWORD *)(v77 + 20) - v35 <= v78 >> 3 )
        goto LABEL_154;
LABEL_39:
      *(_DWORD *)(v77 + 16) = v35;
      v36 = (__int64 *)v125.m128i_i64[0];
      if ( *(_QWORD *)v125.m128i_i64[0] != -4096 )
        --*(_DWORD *)(v77 + 20);
      v37 = v123;
      v36[2] = 0x800000000LL;
      *v36 = v37;
      v36[1] = (__int64)(v36 + 3);
LABEL_42:
      if ( v118 == ++v38 )
        goto LABEL_43;
    }
    v125.m128i_i64[0] = 0;
    ++*(_QWORD *)v77;
LABEL_153:
    v78 *= 2;
LABEL_154:
    sub_26C9E50(v77, v78);
    sub_26C3200(v77, &v123, &v125);
    v35 = *(_DWORD *)(v77 + 16) + 1;
    goto LABEL_39;
  }
LABEL_43:
  v39 = (__int64 *)v5[4];
  v40 = (__int64 *)v5[3];
  if ( v39 != v40 )
  {
    do
    {
      while ( 1 )
      {
        v44 = *v40;
        v45 = v40[1];
        v125.m128i_i64[0] = (__int64)v40;
        v46 = *v5 + 80 * v44;
        v41 = *(_BYTE **)(v46 + 40);
        if ( v41 == *(_BYTE **)(v46 + 48) )
        {
          sub_26D7E10(v46 + 32, v41, &v125);
        }
        else
        {
          if ( v41 )
          {
            *(_QWORD *)v41 = v40;
            v41 = *(_BYTE **)(v46 + 40);
          }
          *(_QWORD *)(v46 + 40) = v41 + 8;
        }
        v125.m128i_i64[0] = (__int64)v40;
        v42 = *v5 + 80 * v45;
        v43 = *(_BYTE **)(v42 + 64);
        if ( v43 != *(_BYTE **)(v42 + 72) )
          break;
        v40 += 5;
        sub_26D7E10(v42 + 56, v43, &v125);
        if ( v39 == v40 )
          goto LABEL_55;
      }
      if ( v43 )
      {
        *(_QWORD *)v43 = v40;
        v43 = *(_BYTE **)(v42 + 64);
      }
      v40 += 5;
      *(_QWORD *)(v42 + 64) = v43 + 8;
    }
    while ( v39 != v40 );
LABEL_55:
    v47 = *(_QWORD *)(a2 + 8);
    v122 = v5[4];
    if ( v5[3] == v122 )
      goto LABEL_77;
    v48 = v47;
    v120 = v5;
    v49 = v5[3];
    while ( 1 )
    {
      v61 = *a3;
      v62 = (*a3)[*(_QWORD *)v49];
      v124 = v62;
      v63 = v61[*(_QWORD *)(v49 + 8)];
      v64 = *(_QWORD *)(v62 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v64 == v62 + 48 )
      {
        v66 = 0;
      }
      else
      {
        if ( !v64 )
LABEL_169:
          BUG();
        v65 = *(unsigned __int8 *)(v64 - 24);
        v66 = (_BYTE *)(v64 - 24);
        if ( (unsigned int)(v65 - 30) >= 0xB )
          v66 = 0;
      }
      v67 = *(_DWORD *)(v48 + 24);
      if ( v67 )
      {
        v50 = *(_QWORD *)(v48 + 8);
        v51 = 1;
        v52 = 0;
        v53 = (v67 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v54 = v50 + 88LL * v53;
        v55 = *(_QWORD *)v54;
        if ( *(_QWORD *)v54 == v62 )
        {
LABEL_58:
          if ( *(_DWORD *)(v54 + 16) == 2 )
          {
            v56 = sub_26CA180(v48, &v124);
            if ( v63 == *(_QWORD *)(*v56 + 8LL * *((unsigned int *)v56 + 2) - 8) && *v66 == 34 )
              *(_BYTE *)(v49 + 25) = 1;
          }
          goto LABEL_60;
        }
        while ( v55 != -4096 )
        {
          if ( v55 == -8192 && !v52 )
            v52 = v54;
          v53 = (v67 - 1) & (v51 + v53);
          v54 = v50 + 88LL * v53;
          v55 = *(_QWORD *)v54;
          if ( v62 == *(_QWORD *)v54 )
            goto LABEL_58;
          ++v51;
        }
        if ( !v52 )
          v52 = v54;
        v125.m128i_i64[0] = v52;
        v73 = *(_DWORD *)(v48 + 16);
        ++*(_QWORD *)v48;
        v68 = v73 + 1;
        if ( 4 * (v73 + 1) < 3 * v67 )
        {
          if ( v67 - *(_DWORD *)(v48 + 20) - v68 > v67 >> 3 )
            goto LABEL_95;
          goto LABEL_75;
        }
      }
      else
      {
        v125.m128i_i64[0] = 0;
        ++*(_QWORD *)v48;
      }
      v67 *= 2;
LABEL_75:
      sub_26C9E50(v48, v67);
      sub_26C3200(v48, &v124, &v125);
      v68 = *(_DWORD *)(v48 + 16) + 1;
LABEL_95:
      *(_DWORD *)(v48 + 16) = v68;
      v74 = (__int64 *)v125.m128i_i64[0];
      if ( *(_QWORD *)v125.m128i_i64[0] != -4096 )
        --*(_DWORD *)(v48 + 20);
      v75 = v124;
      v74[2] = 0x800000000LL;
      *v74 = v75;
      v74[1] = (__int64)(v74 + 3);
LABEL_60:
      v57 = *(_QWORD *)(v63 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v57 == v63 + 48 )
      {
        v59 = 0;
      }
      else
      {
        if ( !v57 )
          goto LABEL_169;
        v58 = *(unsigned __int8 *)(v57 - 24);
        v59 = 0;
        v60 = (_BYTE *)(v57 - 24);
        if ( (unsigned int)(v58 - 30) < 0xB )
          v59 = v60;
      }
      if ( !(unsigned int)sub_B46E30((__int64)v59) && *v59 == 36 )
        *(_BYTE *)(v49 + 25) = 1;
      v49 += 40LL;
      if ( v122 == v49 )
      {
        v5 = v120;
        break;
      }
    }
  }
LABEL_77:
  v69 = 0;
  v70 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v5[1] - *v5) >> 4);
  v71 = *v5;
  if ( v70 )
  {
    while ( *(_QWORD *)(v71 + 64) != *(_QWORD *)(v71 + 56) )
    {
      ++v69;
      v71 += 80;
      if ( v69 == v70 )
        goto LABEL_150;
    }
    v5[6] = v69;
    if ( !*(_QWORD *)(v71 + 8) )
    {
LABEL_82:
      if ( !*(_BYTE *)(v71 + 16) )
        *(_QWORD *)(v71 + 8) = 1;
    }
  }
  else
  {
LABEL_150:
    v71 = *v5 + 80 * v5[6];
    if ( !*(_QWORD *)(v71 + 8) )
      goto LABEL_82;
  }
  return v5;
}
