// Function: sub_F25920
// Address: 0xf25920
//
__int64 __fastcall sub_F25920(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // cl
  __int64 v7; // rsi
  int v8; // edx
  int v9; // r10d
  __int64 *v10; // r9
  __int64 result; // rax
  __int64 *v12; // rdi
  __int64 v13; // r11
  unsigned int v14; // eax
  unsigned int v15; // edx
  unsigned int v16; // eax
  int v17; // esi
  unsigned int v18; // r9d
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r14
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 k; // r15
  __int64 v26; // rax
  _BYTE *v27; // r10
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  int v35; // edx
  int v36; // r9d
  __int64 *v37; // rsi
  unsigned int i; // eax
  __int64 v39; // r10
  unsigned int v40; // eax
  __int64 v41; // rcx
  int v42; // edx
  int v43; // r9d
  unsigned int j; // eax
  __int64 v45; // r10
  unsigned int v46; // eax
  int v47; // edx
  int v48; // edx
  __int64 v51; // [rsp+18h] [rbp-58h]
  _BYTE *v52; // [rsp+20h] [rbp-50h]
  __int64 v53[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(_BYTE *)(a1 + 256) & 1;
  if ( v6 )
  {
    v7 = a1 + 264;
    v8 = 7;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 272);
    v7 = *(_QWORD *)(a1 + 264);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(a1 + 256);
      ++*(_QWORD *)(a1 + 248);
      v12 = 0;
      v17 = (v16 >> 1) + 1;
LABEL_14:
      v18 = 3 * v15;
      goto LABEL_15;
    }
    v8 = v15 - 1;
  }
  v9 = 1;
  v10 = 0;
  for ( result = v8
               & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                                 | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
                ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; result = v8 & v14 )
  {
    v12 = (__int64 *)(v7 + 16LL * (unsigned int)result);
    v13 = *v12;
    if ( a2 == *v12 && a3 == v12[1] )
      return result;
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v10 )
      v10 = (__int64 *)(v7 + 16LL * (unsigned int)result);
LABEL_10:
    v14 = v9 + result;
    ++v9;
  }
  if ( v12[1] != -4096 )
    goto LABEL_10;
  v16 = *(_DWORD *)(a1 + 256);
  if ( v10 )
    v12 = v10;
  ++*(_QWORD *)(a1 + 248);
  v17 = (v16 >> 1) + 1;
  if ( !v6 )
  {
    v15 = *(_DWORD *)(a1 + 272);
    goto LABEL_14;
  }
  v18 = 24;
  v15 = 8;
LABEL_15:
  if ( v18 <= 4 * v17 )
  {
    sub_F19620((const __m128i *)(a1 + 248), 2 * v15);
    if ( (*(_BYTE *)(a1 + 256) & 1) != 0 )
    {
      v34 = a1 + 264;
      v35 = 7;
    }
    else
    {
      v47 = *(_DWORD *)(a1 + 272);
      v34 = *(_QWORD *)(a1 + 264);
      if ( !v47 )
        goto LABEL_92;
      v35 = v47 - 1;
    }
    v36 = 1;
    v37 = 0;
    for ( i = v35
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v35 & v40 )
    {
      v12 = (__int64 *)(v34 + 16LL * i);
      v39 = *v12;
      if ( a2 == *v12 && a3 == v12[1] )
        break;
      if ( v39 == -4096 )
      {
        if ( v12[1] == -4096 )
        {
LABEL_86:
          if ( v37 )
            v12 = v37;
          goto LABEL_82;
        }
      }
      else if ( v39 == -8192 && v12[1] == -8192 && !v37 )
      {
        v37 = (__int64 *)(v34 + 16LL * i);
      }
      v40 = v36 + i;
      ++v36;
    }
    goto LABEL_82;
  }
  if ( v15 - *(_DWORD *)(a1 + 260) - v17 > v15 >> 3 )
    goto LABEL_17;
  sub_F19620((const __m128i *)(a1 + 248), v15);
  if ( (*(_BYTE *)(a1 + 256) & 1) == 0 )
  {
    v48 = *(_DWORD *)(a1 + 272);
    v41 = *(_QWORD *)(a1 + 264);
    if ( v48 )
    {
      v42 = v48 - 1;
      goto LABEL_69;
    }
LABEL_92:
    *(_DWORD *)(a1 + 256) = (2 * (*(_DWORD *)(a1 + 256) >> 1) + 2) | *(_DWORD *)(a1 + 256) & 1;
    BUG();
  }
  v41 = a1 + 264;
  v42 = 7;
LABEL_69:
  v43 = 1;
  v37 = 0;
  for ( j = v42
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v42 & v46 )
  {
    v12 = (__int64 *)(v41 + 16LL * j);
    v45 = *v12;
    if ( a2 == *v12 && a3 == v12[1] )
      break;
    if ( v45 == -4096 )
    {
      if ( v12[1] == -4096 )
        goto LABEL_86;
    }
    else if ( v45 == -8192 && v12[1] == -8192 && !v37 )
    {
      v37 = (__int64 *)(v41 + 16LL * j);
    }
    v46 = v43 + j;
    ++v43;
  }
LABEL_82:
  v16 = *(_DWORD *)(a1 + 256);
LABEL_17:
  *(_DWORD *)(a1 + 256) = (2 * (v16 >> 1) + 2) | v16 & 1;
  if ( *v12 != -4096 || v12[1] != -4096 )
    --*(_DWORD *)(a1 + 260);
  *v12 = a2;
  v12[1] = a3;
  v21 = sub_AA5930(a3);
  v51 = v22;
  while ( v51 != v21 )
  {
    v20 = 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v21 + 7) & 0x40) != 0 )
    {
      v23 = *(_QWORD *)(v21 - 8);
      v24 = v23 + v20;
    }
    else
    {
      v24 = v21;
      v23 = v21 - v20;
    }
    for ( k = v23; v24 != k; *(_BYTE *)(a1 + 240) = 1 )
    {
      while ( a2 != *(_QWORD *)(*(_QWORD *)(v21 - 8)
                              + 32LL * *(unsigned int *)(v21 + 72)
                              + 8LL * (unsigned int)((k - *(_QWORD *)(v21 - 8)) >> 5))
           || **(_BYTE **)k == 13 )
      {
        k += 32;
        if ( v24 == k )
          goto LABEL_38;
      }
      v26 = sub_ACADE0(*(__int64 ***)(v21 + 8));
      v27 = *(_BYTE **)k;
      if ( *(_QWORD *)k )
      {
        v28 = *(_QWORD *)(k + 8);
        **(_QWORD **)(k + 16) = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = *(_QWORD *)(k + 16);
      }
      *(_QWORD *)k = v26;
      if ( v26 )
      {
        v29 = *(_QWORD *)(v26 + 16);
        *(_QWORD *)(k + 8) = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 16) = k + 8;
        *(_QWORD *)(k + 16) = v26 + 16;
        *(_QWORD *)(v26 + 16) = k;
      }
      v30 = *(_QWORD *)(a1 + 40);
      if ( *v27 > 0x1Cu )
      {
        v53[0] = (__int64)v27;
        v52 = v27;
        sub_F200C0(v30 + 2096, v53);
        v31 = v30 + 2096;
        v32 = *((_QWORD *)v52 + 2);
        if ( v32 && !*(_QWORD *)(v32 + 8) )
        {
          v53[0] = *(_QWORD *)(v32 + 24);
          sub_F200C0(v31, v53);
        }
        v30 = *(_QWORD *)(a1 + 40);
      }
      k += 32;
      sub_F15FC0(v30, v21);
    }
LABEL_38:
    v33 = *(_QWORD *)(v21 + 32);
    if ( !v33 )
      BUG();
    v21 = 0;
    if ( *(_BYTE *)(v33 - 24) == 84 )
      v21 = v33 - 24;
  }
  result = *(unsigned int *)(a4 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), result + 1, 8u, v19, v20);
    result = *(unsigned int *)(a4 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = a3;
  ++*(_DWORD *)(a4 + 8);
  return result;
}
