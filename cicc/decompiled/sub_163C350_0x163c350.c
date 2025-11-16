// Function: sub_163C350
// Address: 0x163c350
//
char __fastcall sub_163C350(__int64 *a1, _BYTE *a2, __int64 a3)
{
  int v5; // eax
  unsigned int v6; // ecx
  __int64 v7; // rdx
  __int64 **v8; // rax
  __int64 **i; // rdx
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 **v13; // rdx
  int v14; // r10d
  __int64 **v15; // r9
  int v16; // eax
  int v17; // edx
  __int64 **v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // r13d
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 **j; // rdx
  int v27; // r8d
  int v28; // r8d
  __int64 v29; // r10
  __int64 *v30; // rsi
  int v31; // edi
  __int64 **v32; // rcx
  int v33; // esi
  int v34; // esi
  __int64 v35; // r8
  int v36; // edi
  unsigned int v37; // r13d
  __int64 *v38; // rcx

  if ( !(unsigned __int8)sub_1642D70() )
  {
    LOBYTE(v8) = sub_163B8C0(*a1);
    if ( !(_BYTE)v8 )
      return (char)v8;
    v10 = *(_DWORD *)(a3 + 24);
    if ( v10 )
    {
      v11 = *(_QWORD *)(a3 + 8);
      v12 = (v10 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = (__int64 **)(v11 + 8LL * v12);
      v8 = (__int64 **)*v13;
      if ( a1 == *v13 )
        return (char)v8;
      v14 = 1;
      v15 = 0;
      while ( v8 != (__int64 **)-8LL )
      {
        if ( v8 == (__int64 **)-16LL && !v15 )
          v15 = v13;
        v12 = (v10 - 1) & (v14 + v12);
        v13 = (__int64 **)(v11 + 8LL * v12);
        v8 = (__int64 **)*v13;
        if ( a1 == *v13 )
          return (char)v8;
        ++v14;
      }
      v16 = *(_DWORD *)(a3 + 16);
      if ( !v15 )
        v15 = v13;
      ++*(_QWORD *)a3;
      v17 = v16 + 1;
      if ( 4 * (v16 + 1) < 3 * v10 )
      {
        LODWORD(v8) = v10 - *(_DWORD *)(a3 + 20) - v17;
        if ( (unsigned int)v8 > v10 >> 3 )
        {
LABEL_19:
          *(_DWORD *)(a3 + 16) = v17;
          if ( *v15 != (__int64 *)-8LL )
            --*(_DWORD *)(a3 + 20);
          *v15 = a1;
          return (char)v8;
        }
        sub_13B3B90(a3, v10);
        v33 = *(_DWORD *)(a3 + 24);
        if ( v33 )
        {
          v34 = v33 - 1;
          v35 = *(_QWORD *)(a3 + 8);
          v36 = 1;
          v37 = v34 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v15 = (__int64 **)(v35 + 8LL * v37);
          v38 = *v15;
          v17 = *(_DWORD *)(a3 + 16) + 1;
          v8 = 0;
          if ( a1 != *v15 )
          {
            while ( v38 != (__int64 *)-8LL )
            {
              if ( !v8 && v38 == (__int64 *)-16LL )
                v8 = v15;
              v37 = v34 & (v36 + v37);
              v15 = (__int64 **)(v35 + 8LL * v37);
              v38 = *v15;
              if ( a1 == *v15 )
                goto LABEL_19;
              ++v36;
            }
            if ( v8 )
              v15 = v8;
          }
          goto LABEL_19;
        }
LABEL_70:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    sub_13B3B90(a3, 2 * v10);
    v27 = *(_DWORD *)(a3 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a3 + 8);
      LODWORD(v8) = v28 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v15 = (__int64 **)(v29 + 8LL * (unsigned int)v8);
      v17 = *(_DWORD *)(a3 + 16) + 1;
      v30 = *v15;
      if ( a1 != *v15 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != (__int64 *)-8LL )
        {
          if ( !v32 && v30 == (__int64 *)-16LL )
            v32 = v15;
          LODWORD(v8) = v28 & (v31 + (_DWORD)v8);
          v15 = (__int64 **)(v29 + 8LL * (unsigned int)v8);
          v30 = *v15;
          if ( a1 == *v15 )
            goto LABEL_19;
          ++v31;
        }
        if ( v32 )
          v15 = v32;
      }
      goto LABEL_19;
    }
    goto LABEL_70;
  }
  *a2 = 1;
  v5 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  if ( !v5 )
  {
    LODWORD(v8) = *(_DWORD *)(a3 + 20);
    if ( !(_DWORD)v8 )
      return (char)v8;
    v7 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)v7 > 0x40 )
    {
      LOBYTE(v8) = j___libc_free_0(*(_QWORD *)(a3 + 8));
      *(_QWORD *)(a3 + 8) = 0;
      *(_QWORD *)(a3 + 16) = 0;
      *(_DWORD *)(a3 + 24) = 0;
      return (char)v8;
    }
    goto LABEL_6;
  }
  v6 = 4 * v5;
  v7 = *(unsigned int *)(a3 + 24);
  if ( (unsigned int)(4 * v5) < 0x40 )
    v6 = 64;
  if ( (unsigned int)v7 <= v6 )
  {
LABEL_6:
    v8 = *(__int64 ***)(a3 + 8);
    for ( i = &v8[v7]; i != v8; ++v8 )
      *v8 = (__int64 *)-8LL;
    *(_QWORD *)(a3 + 16) = 0;
    return (char)v8;
  }
  v18 = *(__int64 ***)(a3 + 8);
  v19 = v5 - 1;
  if ( !v19 )
  {
    v24 = 1024;
    v23 = 128;
LABEL_30:
    j___libc_free_0(v18);
    *(_DWORD *)(a3 + 24) = v23;
    v8 = (__int64 **)sub_22077B0(v24);
    v25 = *(unsigned int *)(a3 + 24);
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 8) = v8;
    for ( j = &v8[v25]; j != v8; ++v8 )
    {
      if ( v8 )
        *v8 = (__int64 *)-8LL;
    }
    return (char)v8;
  }
  _BitScanReverse(&v19, v19);
  v20 = (unsigned int)(1 << (33 - (v19 ^ 0x1F)));
  if ( (int)v20 < 64 )
    v20 = 64;
  if ( (_DWORD)v20 != (_DWORD)v7 )
  {
    v21 = (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v20 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)
        | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v20 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4);
    v22 = (v21 >> 8) | v21;
    v23 = (v22 | (v22 >> 16)) + 1;
    v24 = 8 * ((v22 | (v22 >> 16)) + 1);
    goto LABEL_30;
  }
  *(_QWORD *)(a3 + 16) = 0;
  v8 = &v18[v20];
  do
  {
    if ( v18 )
      *v18 = (__int64 *)-8LL;
    ++v18;
  }
  while ( v8 != v18 );
  return (char)v8;
}
