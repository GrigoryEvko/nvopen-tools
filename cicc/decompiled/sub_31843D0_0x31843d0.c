// Function: sub_31843D0
// Address: 0x31843d0
//
__int64 __fastcall sub_31843D0(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  unsigned __int8 *v5; // r12
  unsigned __int8 *v6; // rax
  unsigned int v7; // esi
  unsigned __int8 *v8; // r13
  __int64 v9; // rcx
  int v10; // r11d
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  int v13; // eax
  unsigned int i; // r9d
  __int64 v15; // rdx
  unsigned __int8 *v16; // r8
  unsigned int v17; // r9d
  __int64 result; // rax
  int v19; // ecx
  int v20; // ecx
  __int64 v21; // rdx
  int v22; // r9d
  __int64 v23; // r8
  unsigned int j; // eax
  unsigned __int8 *v25; // rsi
  unsigned int v26; // eax
  int v27; // ecx
  int v28; // ecx
  unsigned int v29; // esi
  __int64 v30; // rdi
  int v31; // r11d
  __int64 v32; // r10
  unsigned int v33; // r8d
  __int64 v34; // rcx
  unsigned __int8 *v35; // r15
  unsigned int v36; // r8d
  int v37; // edx
  int v38; // edx
  __int64 v39; // rsi
  int v40; // r9d
  unsigned int k; // eax
  unsigned __int8 *v42; // rcx
  unsigned int v43; // eax
  int v44; // ecx
  __int64 v45; // rsi
  int v46; // r10d
  __int64 v47; // r9
  int v48; // r8d
  unsigned int v49; // edx
  unsigned __int8 *v50; // rdi
  unsigned int v51; // edx
  int v52; // edi
  int v53; // r8d
  int v54; // ecx
  int v55; // esi
  __int64 v56; // r9
  int v57; // r10d
  unsigned int m; // edx
  unsigned __int8 **v59; // rdi
  unsigned __int8 *v60; // r8
  unsigned int v61; // edx
  int v62; // [rsp+8h] [rbp-38h]
  unsigned __int8 v63; // [rsp+8h] [rbp-38h]
  unsigned __int8 v64; // [rsp+8h] [rbp-38h]

  v5 = sub_3183B70(a2, a1 + 40);
  v6 = sub_3183B70(a3, a1 + 40);
  if ( v5 == v6 )
    return 1;
  v7 = *(_DWORD *)(a1 + 32);
  v8 = v6;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_15;
  }
  v9 = *(_QWORD *)(a1 + 16);
  v10 = 1;
  v11 = 0;
  v12 = (0xBF58476D1CE4E5B9LL
       * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)
        | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31;
  v13 = v12 ^ (484763065 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)));
  for ( i = (v12 ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))) & (v7 - 1); ; i = (v7 - 1) & v17 )
  {
    v15 = v9 + 24LL * i;
    v16 = *(unsigned __int8 **)v15;
    if ( v5 == *(unsigned __int8 **)v15 && v8 == *(unsigned __int8 **)(v15 + 8) )
      return *(unsigned __int8 *)(v15 + 16);
    if ( v16 == (unsigned __int8 *)-4096LL )
      break;
    if ( v16 == (unsigned __int8 *)-8192LL && *(_QWORD *)(v15 + 8) == -8192 && !v11 )
      v11 = v9 + 24LL * i;
LABEL_10:
    v17 = v10 + i;
    ++v10;
  }
  if ( *(_QWORD *)(v15 + 8) != -4096 )
    goto LABEL_10;
  v27 = *(_DWORD *)(a1 + 24);
  if ( !v11 )
    v11 = v15;
  ++*(_QWORD *)(a1 + 8);
  v28 = v27 + 1;
  if ( 4 * v28 >= 3 * v7 )
  {
LABEL_15:
    sub_3184100(a1 + 8, 2 * v7);
    v19 = *(_DWORD *)(a1 + 32);
    if ( v19 )
    {
      v20 = v19 - 1;
      v22 = 1;
      v23 = 0;
      for ( j = v20
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
                  | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; j = v20 & v26 )
      {
        v21 = *(_QWORD *)(a1 + 16);
        v11 = v21 + 24LL * j;
        v25 = *(unsigned __int8 **)v11;
        if ( v5 == *(unsigned __int8 **)v11 && v8 == *(unsigned __int8 **)(v11 + 8) )
          break;
        if ( v25 == (unsigned __int8 *)-4096LL )
        {
          if ( *(_QWORD *)(v11 + 8) == -4096 )
          {
LABEL_91:
            if ( v23 )
              v11 = v23;
            v28 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_29;
          }
        }
        else if ( v25 == (unsigned __int8 *)-8192LL && *(_QWORD *)(v11 + 8) == -8192 && !v23 )
        {
          v23 = v21 + 24LL * j;
        }
        v26 = v22 + j;
        ++v22;
      }
      goto LABEL_83;
    }
LABEL_104:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 28) - v28 <= v7 >> 3 )
  {
    v62 = v13;
    sub_3184100(a1 + 8, v7);
    v37 = *(_DWORD *)(a1 + 32);
    if ( v37 )
    {
      v38 = v37 - 1;
      v23 = 0;
      v40 = 1;
      for ( k = v38 & v62; ; k = v38 & v43 )
      {
        v39 = *(_QWORD *)(a1 + 16);
        v11 = v39 + 24LL * k;
        v42 = *(unsigned __int8 **)v11;
        if ( v5 == *(unsigned __int8 **)v11 && v8 == *(unsigned __int8 **)(v11 + 8) )
          break;
        if ( v42 == (unsigned __int8 *)-4096LL )
        {
          if ( *(_QWORD *)(v11 + 8) == -4096 )
            goto LABEL_91;
        }
        else if ( v42 == (unsigned __int8 *)-8192LL && *(_QWORD *)(v11 + 8) == -8192 && !v23 )
        {
          v23 = v39 + 24LL * k;
        }
        v43 = v40 + k;
        ++v40;
      }
LABEL_83:
      v28 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_29;
    }
    goto LABEL_104;
  }
LABEL_29:
  *(_DWORD *)(a1 + 24) = v28;
  if ( *(_QWORD *)v11 != -4096 || *(_QWORD *)(v11 + 8) != -4096 )
    --*(_DWORD *)(a1 + 28);
  *(_QWORD *)v11 = v5;
  *(_QWORD *)(v11 + 8) = v8;
  *(_BYTE *)(v11 + 16) = 1;
  result = sub_3184CA0(a1, v5, v8);
  v29 = *(_DWORD *)(a1 + 32);
  if ( !v29 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_54;
  }
  v30 = *(_QWORD *)(a1 + 16);
  v31 = 1;
  v32 = 0;
  v33 = (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
          | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))))
      & (v29 - 1);
  while ( 2 )
  {
    v34 = v30 + 24LL * v33;
    v35 = *(unsigned __int8 **)v34;
    if ( v5 == *(unsigned __int8 **)v34 && v8 == *(unsigned __int8 **)(v34 + 8) )
      goto LABEL_43;
    if ( v35 != (unsigned __int8 *)-4096LL )
    {
      if ( v35 == (unsigned __int8 *)-8192LL && *(_QWORD *)(v34 + 8) == -8192 && !v32 )
        v32 = v30 + 24LL * v33;
      goto LABEL_39;
    }
    if ( *(_QWORD *)(v34 + 8) != -4096 )
    {
LABEL_39:
      v36 = v31 + v33;
      ++v31;
      v33 = (v29 - 1) & v36;
      continue;
    }
    break;
  }
  v52 = *(_DWORD *)(a1 + 24);
  if ( v32 )
    v34 = v32;
  ++*(_QWORD *)(a1 + 8);
  v53 = v52 + 1;
  if ( 4 * (v52 + 1) < 3 * v29 )
  {
    if ( v29 - *(_DWORD *)(a1 + 28) - v53 > v29 >> 3 )
      goto LABEL_68;
    v64 = result;
    sub_3184100(a1 + 8, v29);
    v54 = *(_DWORD *)(a1 + 32);
    if ( v54 )
    {
      v55 = v54 - 1;
      v34 = 0;
      result = v64;
      v57 = 1;
      for ( m = v55
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
                  | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; m = v55 & v61 )
      {
        v56 = *(_QWORD *)(a1 + 16);
        v59 = (unsigned __int8 **)(v56 + 24LL * m);
        v60 = *v59;
        if ( v5 == *v59 && v8 == v59[1] )
        {
          v53 = *(_DWORD *)(a1 + 24) + 1;
          v34 = v56 + 24LL * m;
          goto LABEL_68;
        }
        if ( v60 == (unsigned __int8 *)-4096LL )
        {
          if ( v59[1] == (unsigned __int8 *)-4096LL )
          {
            if ( !v34 )
              v34 = v56 + 24LL * m;
            v53 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_68;
          }
        }
        else if ( v60 == (unsigned __int8 *)-8192LL && v59[1] == (unsigned __int8 *)-8192LL && !v34 )
        {
          v34 = v56 + 24LL * m;
        }
        v61 = v57 + m;
        ++v57;
      }
    }
LABEL_105:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_54:
  v63 = result;
  sub_3184100(a1 + 8, 2 * v29);
  v44 = *(_DWORD *)(a1 + 32);
  if ( !v44 )
    goto LABEL_105;
  result = v63;
  v46 = 1;
  v47 = 0;
  v48 = v44 - 1;
  v49 = (v44 - 1)
      & (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
          | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))));
  while ( 2 )
  {
    v45 = *(_QWORD *)(a1 + 16);
    v34 = v45 + 24LL * v49;
    v50 = *(unsigned __int8 **)v34;
    if ( v5 == *(unsigned __int8 **)v34 && v8 == *(unsigned __int8 **)(v34 + 8) )
    {
      v53 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_68;
    }
    if ( v50 != (unsigned __int8 *)-4096LL )
    {
      if ( v50 == (unsigned __int8 *)-8192LL && *(_QWORD *)(v34 + 8) == -8192 && !v47 )
        v47 = v45 + 24LL * v49;
      goto LABEL_62;
    }
    if ( *(_QWORD *)(v34 + 8) != -4096 )
    {
LABEL_62:
      v51 = v46 + v49;
      ++v46;
      v49 = v48 & v51;
      continue;
    }
    break;
  }
  if ( v47 )
    v34 = v47;
  v53 = *(_DWORD *)(a1 + 24) + 1;
LABEL_68:
  *(_DWORD *)(a1 + 24) = v53;
  if ( *(_QWORD *)v34 != -4096 || *(_QWORD *)(v34 + 8) != -4096 )
    --*(_DWORD *)(a1 + 28);
  *(_QWORD *)v34 = v5;
  *(_QWORD *)(v34 + 8) = v8;
  *(_BYTE *)(v34 + 16) = 0;
LABEL_43:
  *(_BYTE *)(v34 + 16) = result;
  return result;
}
