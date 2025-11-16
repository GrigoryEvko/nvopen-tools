// Function: sub_1644900
// Address: 0x1644900
//
__int64 __fastcall sub_1644900(_QWORD *a1, unsigned int a2)
{
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // ecx
  int *v10; // rax
  unsigned int v11; // edx
  __int64 v12; // r8
  int v13; // r10d
  unsigned int *v14; // r14
  int v15; // eax
  int v16; // edx
  __int64 v17; // rax
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned int v22; // esi
  int v23; // r9d
  unsigned int *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  unsigned int *v29; // rdi
  unsigned int v30; // r15d
  unsigned int v31; // ecx

  if ( a2 == 32 )
    return sub_1643350(a1);
  if ( a2 > 0x20 )
  {
    if ( a2 == 64 )
      return sub_1643360(a1);
    if ( a2 == 128 )
      return sub_1643370(a1);
  }
  else
  {
    switch ( a2 )
    {
      case 8u:
        return sub_1643330(a1);
      case 0x10u:
        return sub_1643340(a1);
      case 1u:
        return sub_1643320(a1);
    }
  }
  v5 = *a1;
  v6 = *(_DWORD *)(*a1 + 2400LL);
  v7 = *a1 + 2376LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 2376);
    goto LABEL_29;
  }
  v8 = *(_QWORD *)(v5 + 2384);
  v9 = (v6 - 1) & (37 * a2);
  v10 = (int *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    v13 = 1;
    v14 = 0;
    while ( v11 != -1 )
    {
      if ( !v14 && v11 == -2 )
        v14 = (unsigned int *)v10;
      v9 = (v6 - 1) & (v13 + v9);
      v10 = (int *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_16;
      ++v13;
    }
    if ( !v14 )
      v14 = (unsigned int *)v10;
    v15 = *(_DWORD *)(v5 + 2392);
    ++*(_QWORD *)(v5 + 2376);
    v16 = v15 + 1;
    if ( 4 * (v15 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v5 + 2396) - v16 > v6 >> 3 )
      {
LABEL_24:
        *(_DWORD *)(v5 + 2392) = v16;
        if ( *v14 != -1 )
          --*(_DWORD *)(v5 + 2396);
        *v14 = a2;
        *((_QWORD *)v14 + 1) = 0;
        v5 = *a1;
        goto LABEL_27;
      }
      sub_1644740(v7, v6);
      v25 = *(_DWORD *)(v5 + 2400);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(v5 + 2384);
        v28 = 1;
        v29 = 0;
        v30 = v26 & (37 * a2);
        v16 = *(_DWORD *)(v5 + 2392) + 1;
        v14 = (unsigned int *)(v27 + 16LL * v30);
        v31 = *v14;
        if ( *v14 != a2 )
        {
          while ( v31 != -1 )
          {
            if ( !v29 && v31 == -2 )
              v29 = v14;
            v30 = v26 & (v28 + v30);
            v14 = (unsigned int *)(v27 + 16LL * v30);
            v31 = *v14;
            if ( *v14 == a2 )
              goto LABEL_24;
            ++v28;
          }
          if ( v29 )
            v14 = v29;
        }
        goto LABEL_24;
      }
LABEL_58:
      ++*(_DWORD *)(v5 + 2392);
      BUG();
    }
LABEL_29:
    sub_1644740(v7, 2 * v6);
    v18 = *(_DWORD *)(v5 + 2400);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v5 + 2384);
      v21 = (v18 - 1) & (37 * a2);
      v16 = *(_DWORD *)(v5 + 2392) + 1;
      v14 = (unsigned int *)(v20 + 16LL * v21);
      v22 = *v14;
      if ( *v14 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -1 )
        {
          if ( !v24 && v22 == -2 )
            v24 = v14;
          v21 = v19 & (v23 + v21);
          v14 = (unsigned int *)(v20 + 16LL * v21);
          v22 = *v14;
          if ( *v14 == a2 )
            goto LABEL_24;
          ++v23;
        }
        if ( v24 )
          v14 = v24;
      }
      goto LABEL_24;
    }
    goto LABEL_58;
  }
LABEL_16:
  v12 = *((_QWORD *)v10 + 1);
  if ( !v12 )
  {
    v14 = (unsigned int *)v10;
LABEL_27:
    v17 = sub_145CBF0((__int64 *)(v5 + 2272), 24, 16);
    *(_QWORD *)v17 = a1;
    v12 = v17;
    *(_BYTE *)(v17 + 8) = 11;
    *(_DWORD *)(v17 + 12) = 0;
    *(_QWORD *)(v17 + 16) = 0;
    *(_DWORD *)(v17 + 8) = *(unsigned __int8 *)(v17 + 8) | (a2 << 8);
    *((_QWORD *)v14 + 1) = v17;
  }
  return v12;
}
