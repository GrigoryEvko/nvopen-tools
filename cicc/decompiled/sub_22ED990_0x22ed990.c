// Function: sub_22ED990
// Address: 0x22ed990
//
char __fastcall sub_22ED990(unsigned __int8 *a1, _BYTE *a2, __int64 a3)
{
  int v5; // eax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int8 **v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  unsigned __int8 **v13; // rdx
  int v14; // r10d
  unsigned __int8 **v15; // r9
  int v16; // eax
  int v17; // edx
  int v18; // eax
  __int64 v19; // rdx
  unsigned __int8 **i; // rdx
  unsigned int v21; // ecx
  unsigned int v22; // eax
  unsigned __int8 **v23; // rdi
  int v24; // r12d
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int8 **j; // rdx
  int v29; // r8d
  int v30; // r8d
  __int64 v31; // r10
  unsigned __int8 *v32; // rsi
  int v33; // edi
  unsigned __int8 **v34; // rcx
  int v35; // ecx
  int v36; // ecx
  __int64 v37; // r8
  int v38; // edi
  unsigned int v39; // r13d
  unsigned __int8 *v40; // rsi

  v5 = *a1;
  if ( (unsigned __int8)v5 <= 0x1Cu
    || (v6 = (unsigned int)(v5 - 34), (unsigned __int8)v6 > 0x33u)
    || (v7 = 0x8000000000041LL, !_bittest64(&v7, v6))
    || (v8 = *((_QWORD *)a1 - 4)) == 0
    || *(_BYTE *)v8
    || *(_QWORD *)(v8 + 24) != *((_QWORD *)a1 + 10)
    || *(_DWORD *)(v8 + 36) != 151 )
  {
    LOBYTE(v9) = sub_22ECB00(*((_QWORD *)a1 + 1));
    if ( !(_BYTE)v9 )
      return (char)v9;
    v10 = *(_DWORD *)(a3 + 24);
    if ( v10 )
    {
      v11 = *(_QWORD *)(a3 + 8);
      v12 = (v10 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = (unsigned __int8 **)(v11 + 8LL * v12);
      v9 = (unsigned __int8 **)*v13;
      if ( a1 == *v13 )
        return (char)v9;
      v14 = 1;
      v15 = 0;
      while ( v9 != (unsigned __int8 **)-4096LL )
      {
        if ( !v15 && v9 == (unsigned __int8 **)-8192LL )
          v15 = v13;
        v12 = (v10 - 1) & (v14 + v12);
        v13 = (unsigned __int8 **)(v11 + 8LL * v12);
        v9 = (unsigned __int8 **)*v13;
        if ( a1 == *v13 )
          return (char)v9;
        ++v14;
      }
      v16 = *(_DWORD *)(a3 + 16);
      if ( !v15 )
        v15 = v13;
      ++*(_QWORD *)a3;
      v17 = v16 + 1;
      if ( 4 * (v16 + 1) < 3 * v10 )
      {
        LODWORD(v9) = v10 - *(_DWORD *)(a3 + 20) - v17;
        if ( (unsigned int)v9 > v10 >> 3 )
        {
LABEL_17:
          *(_DWORD *)(a3 + 16) = v17;
          if ( *v15 != (unsigned __int8 *)-4096LL )
            --*(_DWORD *)(a3 + 20);
          *v15 = a1;
          return (char)v9;
        }
        sub_BD14B0(a3, v10);
        v35 = *(_DWORD *)(a3 + 24);
        if ( v35 )
        {
          v36 = v35 - 1;
          v37 = *(_QWORD *)(a3 + 8);
          v38 = 1;
          v39 = v36 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v15 = (unsigned __int8 **)(v37 + 8LL * v39);
          v40 = *v15;
          v17 = *(_DWORD *)(a3 + 16) + 1;
          v9 = 0;
          if ( a1 != *v15 )
          {
            while ( v40 != (unsigned __int8 *)-4096LL )
            {
              if ( !v9 && v40 == (unsigned __int8 *)-8192LL )
                v9 = v15;
              v39 = v36 & (v38 + v39);
              v15 = (unsigned __int8 **)(v37 + 8LL * v39);
              v40 = *v15;
              if ( a1 == *v15 )
                goto LABEL_17;
              ++v38;
            }
            if ( v9 )
              v15 = v9;
          }
          goto LABEL_17;
        }
LABEL_75:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    sub_BD14B0(a3, 2 * v10);
    v29 = *(_DWORD *)(a3 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a3 + 8);
      LODWORD(v9) = v30 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v15 = (unsigned __int8 **)(v31 + 8LL * (unsigned int)v9);
      v17 = *(_DWORD *)(a3 + 16) + 1;
      v32 = *v15;
      if ( a1 != *v15 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != (unsigned __int8 *)-4096LL )
        {
          if ( !v34 && v32 == (unsigned __int8 *)-8192LL )
            v34 = v15;
          LODWORD(v9) = v30 & (v33 + (_DWORD)v9);
          v15 = (unsigned __int8 **)(v31 + 8LL * (unsigned int)v9);
          v32 = *v15;
          if ( a1 == *v15 )
            goto LABEL_17;
          ++v33;
        }
        if ( v34 )
          v15 = v34;
      }
      goto LABEL_17;
    }
    goto LABEL_75;
  }
  *a2 = 1;
  v18 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  if ( !v18 )
  {
    LODWORD(v9) = *(_DWORD *)(a3 + 20);
    if ( !(_DWORD)v9 )
      return (char)v9;
    v19 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)v19 > 0x40 )
    {
      LOBYTE(v9) = sub_C7D6A0(*(_QWORD *)(a3 + 8), 8 * v19, 8);
      *(_QWORD *)(a3 + 8) = 0;
      *(_QWORD *)(a3 + 16) = 0;
      *(_DWORD *)(a3 + 24) = 0;
      return (char)v9;
    }
    goto LABEL_24;
  }
  v21 = 4 * v18;
  v19 = *(unsigned int *)(a3 + 24);
  if ( (unsigned int)(4 * v18) < 0x40 )
    v21 = 64;
  if ( v21 >= (unsigned int)v19 )
  {
LABEL_24:
    v9 = *(unsigned __int8 ***)(a3 + 8);
    for ( i = &v9[v19]; i != v9; ++v9 )
      *v9 = (unsigned __int8 *)-4096LL;
    *(_QWORD *)(a3 + 16) = 0;
    return (char)v9;
  }
  v22 = v18 - 1;
  if ( v22 )
  {
    _BitScanReverse(&v22, v22);
    v23 = *(unsigned __int8 ***)(a3 + 8);
    v24 = 1 << (33 - (v22 ^ 0x1F));
    if ( v24 < 64 )
      v24 = 64;
    if ( (_DWORD)v19 == v24 )
    {
      *(_QWORD *)(a3 + 16) = 0;
      v9 = &v23[v19];
      do
      {
        if ( v23 )
          *v23 = (unsigned __int8 *)-4096LL;
        ++v23;
      }
      while ( v9 != v23 );
      return (char)v9;
    }
  }
  else
  {
    v23 = *(unsigned __int8 ***)(a3 + 8);
    v24 = 64;
  }
  sub_C7D6A0((__int64)v23, 8 * v19, 8);
  v25 = ((((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
       | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
       | (4 * v24 / 3u + 1)
       | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 16;
  v26 = (v25
       | (((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
       | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
       | (4 * v24 / 3u + 1)
       | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a3 + 24) = v26;
  v9 = (unsigned __int8 **)sub_C7D670(8 * v26, 8);
  v27 = *(unsigned int *)(a3 + 24);
  *(_QWORD *)(a3 + 16) = 0;
  *(_QWORD *)(a3 + 8) = v9;
  for ( j = &v9[v27]; j != v9; ++v9 )
  {
    if ( v9 )
      *v9 = (unsigned __int8 *)-4096LL;
  }
  return (char)v9;
}
