// Function: sub_BAAF70
// Address: 0xbaaf70
//
__m128i *__fastcall sub_BAAF70(__m128i *a1, __int64 a2, const void *a3, size_t a4, int a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned int v8; // esi
  unsigned int v9; // ecx
  __int64 v10; // r9
  int v11; // r11d
  unsigned __int64 v12; // rbx
  int *v13; // rax
  unsigned int i; // edi
  int *v15; // rdx
  int v16; // ecx
  unsigned int v17; // edi
  int v18; // eax
  int v20; // ecx
  int v21; // ecx
  unsigned int v22; // eax
  unsigned int v23; // r13d
  __int64 v24; // rdx
  int v25; // r13d
  __m128i *v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rcx
  unsigned int v30; // esi
  __int64 v31; // r8
  int *v32; // r11
  int v33; // r10d
  unsigned __int64 v34; // rbx
  unsigned int ii; // edi
  int *v36; // rax
  int v37; // edx
  unsigned int v38; // edi
  size_t v39; // rdx
  int v40; // edx
  __int64 v41; // rbx
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // rcx
  unsigned int v45; // esi
  int *v46; // rbx
  __int64 v47; // r9
  int v48; // r11d
  unsigned __int64 v49; // r12
  int *v50; // rax
  unsigned int v51; // ecx
  int *v52; // rdx
  int v53; // edi
  unsigned int v54; // ecx
  int v55; // edi
  int v56; // edi
  int v57; // r10d
  int *v58; // r9
  __int64 v59; // rsi
  unsigned int m; // edx
  int v61; // r8d
  unsigned int v62; // edx
  int v63; // edx
  int v64; // edi
  int v65; // eax
  int v66; // edx
  __int64 v67; // rdi
  int *v68; // r8
  unsigned int v69; // ebx
  int jj; // r9d
  int v71; // esi
  unsigned int v72; // ebx
  _DWORD *v73; // rax
  int v74; // esi
  int v75; // esi
  int *v76; // r8
  __int64 v77; // rcx
  int v78; // r9d
  unsigned int j; // edx
  int v80; // edi
  unsigned int v81; // edx
  int v82; // eax
  int v83; // edx
  __int64 v84; // rsi
  int v85; // r8d
  int *v86; // rdi
  unsigned int k; // ebx
  int v88; // ecx
  unsigned int v89; // ebx
  int v90; // edx
  __int64 v91; // rcx
  int v92; // esi
  int v93; // edi
  unsigned int kk; // edx
  int v95; // r8d
  unsigned int v96; // edx
  int v97; // ecx
  int v98; // ecx
  int v99; // eax
  int v100; // edx
  __int64 v101; // rsi
  int v102; // ecx
  unsigned int mm; // r12d
  int v104; // edi
  unsigned int v105; // r12d
  __int64 v106; // [rsp+0h] [rbp-140h]
  __int64 v107; // [rsp+0h] [rbp-140h]
  __int64 v108; // [rsp+8h] [rbp-138h]
  __int64 v109; // [rsp+20h] [rbp-120h]
  unsigned __int64 v110; // [rsp+28h] [rbp-118h]
  __m128i *dest; // [rsp+70h] [rbp-D0h]
  size_t v117; // [rsp+78h] [rbp-C8h]
  __m128i v118; // [rsp+80h] [rbp-C0h] BYREF
  __m128i *v119; // [rsp+90h] [rbp-B0h] BYREF
  size_t n; // [rsp+98h] [rbp-A8h]
  _QWORD src[2]; // [rsp+A0h] [rbp-A0h] BYREF
  _QWORD v122[2]; // [rsp+B0h] [rbp-90h] BYREF
  int v123; // [rsp+C0h] [rbp-80h]
  __int16 v124; // [rsp+D0h] [rbp-70h]
  const void *v125; // [rsp+E0h] [rbp-60h] BYREF
  size_t v126; // [rsp+E8h] [rbp-58h]
  char *v127; // [rsp+F0h] [rbp-50h]
  __int16 v128; // [rsp+100h] [rbp-40h]

  v6 = a2 + 832;
  v8 = *(_DWORD *)(a2 + 856);
  v108 = v6;
  if ( !v8 )
  {
    ++*(_QWORD *)(a2 + 832);
    goto LABEL_112;
  }
  v9 = a6;
  v10 = *(_QWORD *)(a2 + 840);
  v11 = 1;
  v110 = (unsigned __int64)(unsigned int)(37 * a5) << 32;
  v12 = ((0xBF58476D1CE4E5B9LL * (v110 | (v9 >> 9) ^ (v9 >> 4))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL * (v110 | (v9 >> 9) ^ (v9 >> 4)));
  v13 = 0;
  for ( i = v12 & (v8 - 1); ; i = (v8 - 1) & v17 )
  {
    v15 = (int *)(v10 + 24LL * i);
    v16 = *v15;
    if ( a5 == *v15 && a6 == *((_QWORD *)v15 + 1) )
    {
      v18 = v15[4];
      v124 = 2306;
      v125 = a3;
      v122[0] = &v125;
      v126 = a4;
      v127 = ".";
      v128 = 773;
      v123 = v18;
      sub_CA0F50(a1, v122);
      return a1;
    }
    if ( v16 == -1 )
      break;
    if ( v16 == -2 && *((_QWORD *)v15 + 1) == -8192 && !v13 )
      v13 = (int *)(v10 + 24LL * i);
LABEL_9:
    v17 = v11 + i;
    ++v11;
  }
  if ( *((_QWORD *)v15 + 1) != -4096 )
    goto LABEL_9;
  v20 = *(_DWORD *)(a2 + 848);
  if ( !v13 )
    v13 = (int *)(v10 + 24LL * i);
  ++*(_QWORD *)(a2 + 832);
  v21 = v20 + 1;
  if ( 4 * v21 >= 3 * v8 )
  {
LABEL_112:
    sub_BAACB0(v108, 2 * v8);
    v74 = *(_DWORD *)(a2 + 856);
    if ( v74 )
    {
      v75 = v74 - 1;
      v76 = 0;
      v77 = *(_QWORD *)(a2 + 840);
      v78 = 1;
      v110 = (unsigned __int64)(unsigned int)(37 * a5) << 32;
      for ( j = v75
              & (((0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))) >> 31)
               ^ (484763065 * (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4)))); ; j = v75 & v81 )
      {
        v13 = (int *)(v77 + 24LL * j);
        v80 = *v13;
        if ( a5 == *v13 && a6 == *((_QWORD *)v13 + 1) )
          break;
        if ( v80 == -1 )
        {
          if ( *((_QWORD *)v13 + 1) == -4096 )
          {
            v21 = *(_DWORD *)(a2 + 848) + 1;
            if ( v76 )
              v13 = v76;
            goto LABEL_18;
          }
        }
        else if ( v80 == -2 && *((_QWORD *)v13 + 1) == -8192 && !v76 )
        {
          v76 = (int *)(v77 + 24LL * j);
        }
        v81 = v78 + j;
        ++v78;
      }
      goto LABEL_143;
    }
LABEL_191:
    ++*(_DWORD *)(a2 + 848);
    BUG();
  }
  if ( v8 - *(_DWORD *)(a2 + 852) - v21 <= v8 >> 3 )
  {
    sub_BAACB0(v108, v8);
    v82 = *(_DWORD *)(a2 + 856);
    if ( v82 )
    {
      v83 = v82 - 1;
      v85 = 1;
      v86 = 0;
      for ( k = (v82 - 1) & v12; ; k = v83 & v89 )
      {
        v84 = *(_QWORD *)(a2 + 840);
        v13 = (int *)(v84 + 24LL * k);
        v88 = *v13;
        if ( a5 == *v13 && a6 == *((_QWORD *)v13 + 1) )
          break;
        if ( v88 == -1 )
        {
          if ( *((_QWORD *)v13 + 1) == -4096 )
          {
            v21 = *(_DWORD *)(a2 + 848) + 1;
            if ( v86 )
              v13 = v86;
            goto LABEL_18;
          }
        }
        else if ( v88 == -2 && *((_QWORD *)v13 + 1) == -8192 && !v86 )
        {
          v86 = (int *)(v84 + 24LL * k);
        }
        v89 = v85 + k;
        ++v85;
      }
LABEL_143:
      v21 = *(_DWORD *)(a2 + 848) + 1;
      goto LABEL_18;
    }
    goto LABEL_191;
  }
LABEL_18:
  *(_DWORD *)(a2 + 848) = v21;
  if ( *v13 != -1 || *((_QWORD *)v13 + 1) != -4096 )
    --*(_DWORD *)(a2 + 852);
  v13[4] = 0;
  *v13 = a5;
  *((_QWORD *)v13 + 1) = a6;
  v125 = a3;
  v126 = a4;
  v22 = sub_C92610(a3, a4);
  v23 = sub_C92740(a2 + 808, a3, a4, v22);
  v24 = *(_QWORD *)(*(_QWORD *)(a2 + 808) + 8LL * v23);
  v109 = *(_QWORD *)(a2 + 808) + 8LL * v23;
  if ( v24 )
  {
    if ( v24 != -8 )
      goto LABEL_22;
    --*(_DWORD *)(a2 + 824);
  }
  v41 = sub_C7D670(a4 + 17, 8);
  if ( a4 )
    memcpy((void *)(v41 + 16), a3, a4);
  *(_BYTE *)(v41 + a4 + 16) = 0;
  *(_QWORD *)v41 = a4;
  *(_DWORD *)(v41 + 8) = 0;
  *(_QWORD *)v109 = v41;
  ++*(_DWORD *)(a2 + 820);
  v42 = (__int64 *)(*(_QWORD *)(a2 + 808) + 8LL * (unsigned int)sub_C929D0(a2 + 808, v23));
  v24 = *v42;
  v109 = (__int64)v42;
  if ( !*v42 || v24 == -8 )
  {
    v43 = v42 + 1;
    do
    {
      do
      {
        v24 = *v43;
        v44 = v43++;
      }
      while ( !v24 );
    }
    while ( v24 == -8 );
    v109 = (__int64)v44;
  }
LABEL_22:
  v25 = *(_DWORD *)(v24 + 8);
  dest = &v118;
  v118.m128i_i8[0] = 0;
LABEL_23:
  v123 = v25;
  v124 = 2306;
  v125 = a3;
  v126 = a4;
  v127 = ".";
  v128 = 773;
  v122[0] = &v125;
  sub_CA0F50(&v119, v122);
  v26 = dest;
  if ( v119 == (__m128i *)src )
  {
    v39 = n;
    if ( n )
    {
      if ( n == 1 )
        dest->m128i_i8[0] = src[0];
      else
        memcpy(dest, src, n);
      v39 = n;
      v26 = dest;
    }
    v117 = v39;
    v26->m128i_i8[v39] = 0;
    v26 = v119;
  }
  else
  {
    if ( dest == &v118 )
    {
      dest = v119;
      v117 = n;
      v118.m128i_i64[0] = src[0];
    }
    else
    {
      v27 = v118.m128i_i64[0];
      dest = v119;
      v117 = n;
      v118.m128i_i64[0] = src[0];
      if ( v26 )
      {
        v119 = v26;
        src[0] = v27;
        goto LABEL_27;
      }
    }
    v119 = (__m128i *)src;
    v26 = (__m128i *)src;
  }
LABEL_27:
  n = 0;
  v26->m128i_i8[0] = 0;
  if ( v119 != (__m128i *)src )
    j_j___libc_free_0(v119, src[0] + 1LL);
  v28 = sub_BA8B30(a2, (__int64)dest, v117);
  if ( v28 )
  {
    v29 = *(_QWORD *)(v28 + 24);
    v30 = *(_DWORD *)(a2 + 856);
    if ( *(_BYTE *)(v29 + 8) != 13 )
      v29 = 0;
    if ( !v30 )
    {
      ++*(_QWORD *)(a2 + 832);
LABEL_76:
      v106 = v29;
      sub_BAACB0(v108, 2 * v30);
      v55 = *(_DWORD *)(a2 + 856);
      if ( v55 )
      {
        v29 = v106;
        v56 = v55 - 1;
        v57 = 1;
        v58 = 0;
        for ( m = v56
                & (((0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4))) >> 31)
                 ^ (484763065 * (v110 | ((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4)))); ; m = v56 & v62 )
        {
          v59 = *(_QWORD *)(a2 + 840);
          v36 = (int *)(v59 + 24LL * m);
          v61 = *v36;
          if ( a5 == *v36 && v106 == *((_QWORD *)v36 + 1) )
            break;
          if ( v61 == -1 )
          {
            if ( *((_QWORD *)v36 + 1) == -4096 )
            {
              if ( v58 )
                v36 = v58;
              v64 = *(_DWORD *)(a2 + 848) + 1;
              goto LABEL_92;
            }
          }
          else if ( v61 == -2 && *((_QWORD *)v36 + 1) == -8192 && !v58 )
          {
            v58 = (int *)(v59 + 24LL * m);
          }
          v62 = v57 + m;
          ++v57;
        }
        goto LABEL_110;
      }
LABEL_193:
      ++*(_DWORD *)(a2 + 848);
      BUG();
    }
    v31 = *(_QWORD *)(a2 + 840);
    v32 = 0;
    v33 = 1;
    v34 = ((0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4))) >> 31)
        ^ (0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4)));
    for ( ii = v34 & (v30 - 1); ; ii = (v30 - 1) & v38 )
    {
      v36 = (int *)(v31 + 24LL * ii);
      v37 = *v36;
      if ( a5 == *v36 && v29 == *((_QWORD *)v36 + 1) )
        goto LABEL_47;
      if ( v37 == -1 )
      {
        if ( *((_QWORD *)v36 + 1) == -4096 )
        {
          v63 = *(_DWORD *)(a2 + 848);
          if ( v32 )
            v36 = v32;
          ++*(_QWORD *)(a2 + 832);
          v64 = v63 + 1;
          if ( 4 * (v63 + 1) >= 3 * v30 )
            goto LABEL_76;
          if ( v30 - *(_DWORD *)(a2 + 852) - v64 > v30 >> 3 )
            goto LABEL_92;
          v107 = v29;
          sub_BAACB0(v108, v30);
          v65 = *(_DWORD *)(a2 + 856);
          if ( !v65 )
            goto LABEL_193;
          v66 = v65 - 1;
          v29 = v107;
          v68 = 0;
          v69 = (v65 - 1) & v34;
          for ( jj = 1; ; ++jj )
          {
            v67 = *(_QWORD *)(a2 + 840);
            v36 = (int *)(v67 + 24LL * v69);
            v71 = *v36;
            if ( a5 == *v36 && v107 == *((_QWORD *)v36 + 1) )
              break;
            if ( v71 == -1 )
            {
              if ( *((_QWORD *)v36 + 1) == -4096 )
              {
                if ( v68 )
                  v36 = v68;
                v64 = *(_DWORD *)(a2 + 848) + 1;
                goto LABEL_92;
              }
            }
            else if ( v71 == -2 && *((_QWORD *)v36 + 1) == -8192 && !v68 )
            {
              v68 = (int *)(v67 + 24LL * v69);
            }
            v72 = jj + v69;
            v69 = v66 & v72;
          }
LABEL_110:
          v64 = *(_DWORD *)(a2 + 848) + 1;
LABEL_92:
          *(_DWORD *)(a2 + 848) = v64;
          if ( *v36 != -1 || *((_QWORD *)v36 + 1) != -4096 )
            --*(_DWORD *)(a2 + 852);
          *((_QWORD *)v36 + 1) = v29;
          v36[4] = v25;
          *v36 = a5;
LABEL_47:
          v40 = v25 + 1;
          if ( v29 == a6 )
          {
            v36[4] = v25;
            goto LABEL_70;
          }
          ++v25;
          goto LABEL_23;
        }
      }
      else if ( v37 == -2 && *((_QWORD *)v36 + 1) == -8192 && !v32 )
      {
        v32 = (int *)(v31 + 24LL * ii);
      }
      v38 = v33 + ii;
      ++v33;
    }
  }
  v45 = *(_DWORD *)(a2 + 856);
  v46 = 0;
  if ( !v45 )
  {
    ++*(_QWORD *)(a2 + 832);
    goto LABEL_133;
  }
  v47 = *(_QWORD *)(a2 + 840);
  v48 = 1;
  v49 = ((0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4)));
  v50 = 0;
  v51 = v49 & (v45 - 1);
  while ( 2 )
  {
    v52 = (int *)(v47 + 24LL * v51);
    v53 = *v52;
    if ( a5 == *v52 && a6 == *((_QWORD *)v52 + 1) )
    {
      v73 = v52 + 4;
      goto LABEL_108;
    }
    if ( v53 != -1 )
    {
      if ( v53 == -2 && *((_QWORD *)v52 + 1) == -8192 && !v50 )
        v50 = (int *)(v47 + 24LL * v51);
      goto LABEL_68;
    }
    if ( *((_QWORD *)v52 + 1) != -4096 )
    {
LABEL_68:
      v54 = v48 + v51;
      ++v48;
      v51 = (v45 - 1) & v54;
      continue;
    }
    break;
  }
  v97 = *(_DWORD *)(a2 + 848);
  if ( !v50 )
    v50 = v52;
  ++*(_QWORD *)(a2 + 832);
  v98 = v97 + 1;
  if ( 4 * v98 >= 3 * v45 )
  {
LABEL_133:
    sub_BAACB0(v108, 2 * v45);
    v90 = *(_DWORD *)(a2 + 856);
    if ( v90 )
    {
      v92 = v90 - 1;
      v93 = 1;
      for ( kk = (v90 - 1)
               & (((0xBF58476D1CE4E5B9LL * (v110 | ((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))) >> 31)
                ^ (484763065 * (v110 | ((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4)))); ; kk = v92 & v96 )
      {
        v91 = *(_QWORD *)(a2 + 840);
        v50 = (int *)(v91 + 24LL * kk);
        v95 = *v50;
        if ( a5 == *v50 && a6 == *((_QWORD *)v50 + 1) )
          break;
        if ( v95 == -1 )
        {
          if ( *((_QWORD *)v50 + 1) == -4096 )
          {
LABEL_182:
            v98 = *(_DWORD *)(a2 + 848) + 1;
            if ( v46 )
              v50 = v46;
            goto LABEL_151;
          }
        }
        else if ( v95 == -2 && *((_QWORD *)v50 + 1) == -8192 && !v46 )
        {
          v46 = (int *)(v91 + 24LL * kk);
        }
        v96 = v93 + kk;
        ++v93;
      }
      goto LABEL_174;
    }
LABEL_192:
    ++*(_DWORD *)(a2 + 848);
    BUG();
  }
  if ( v45 - *(_DWORD *)(a2 + 852) - v98 <= v45 >> 3 )
  {
    sub_BAACB0(v108, v45);
    v99 = *(_DWORD *)(a2 + 856);
    if ( v99 )
    {
      v100 = v99 - 1;
      v102 = 1;
      for ( mm = (v99 - 1) & v49; ; mm = v100 & v105 )
      {
        v101 = *(_QWORD *)(a2 + 840);
        v50 = (int *)(v101 + 24LL * mm);
        v104 = *v50;
        if ( a5 == *v50 && a6 == *((_QWORD *)v50 + 1) )
          break;
        if ( v104 == -1 )
        {
          if ( *((_QWORD *)v50 + 1) == -4096 )
            goto LABEL_182;
        }
        else if ( v104 == -2 && *((_QWORD *)v50 + 1) == -8192 && !v46 )
        {
          v46 = (int *)(v101 + 24LL * mm);
        }
        v105 = v102 + mm;
        ++v102;
      }
LABEL_174:
      v98 = *(_DWORD *)(a2 + 848) + 1;
      goto LABEL_151;
    }
    goto LABEL_192;
  }
LABEL_151:
  *(_DWORD *)(a2 + 848) = v98;
  if ( *v50 != -1 || *((_QWORD *)v50 + 1) != -4096 )
    --*(_DWORD *)(a2 + 852);
  v50[4] = 0;
  v73 = v50 + 4;
  *(v73 - 4) = a5;
  *((_QWORD *)v73 - 1) = a6;
LABEL_108:
  *v73 = v25;
  v40 = v25 + 1;
LABEL_70:
  *(_DWORD *)(*(_QWORD *)v109 + 8LL) = v40;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( dest == &v118 )
  {
    a1[1] = _mm_load_si128(&v118);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)dest;
    a1[1].m128i_i64[0] = v118.m128i_i64[0];
  }
  a1->m128i_i64[1] = v117;
  return a1;
}
