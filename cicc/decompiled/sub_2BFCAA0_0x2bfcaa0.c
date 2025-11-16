// Function: sub_2BFCAA0
// Address: 0x2bfcaa0
//
__int64 __fastcall sub_2BFCAA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  unsigned int v5; // r11d
  __int64 v6; // rsi
  unsigned int v7; // r8d
  unsigned int v8; // r9d
  __int64 *v9; // rdx
  __int64 v10; // r10
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned int v14; // r9d
  __int64 *v15; // rdx
  __int64 v16; // r11
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  int v24; // edx
  int v25; // ebx
  int v26; // edx
  int v27; // r12d

  if ( a2 == a3 )
    return 0;
  v3 = *(_QWORD *)(a2 + 80);
  v4 = *(_QWORD *)(a3 + 80);
  if ( v3 == v4 )
  {
    v22 = *(_QWORD *)(v3 + 120);
    v23 = v3 + 112;
    if ( v22 == v23 )
LABEL_48:
      BUG();
    while ( 1 )
    {
      if ( v22 )
      {
        if ( a2 == v22 - 24 )
          return 1;
        if ( a3 == v22 - 24 )
          return 0;
      }
      v22 = *(_QWORD *)(v22 + 8);
      if ( v23 == v22 )
        goto LABEL_48;
    }
  }
  v5 = *(_DWORD *)(a1 + 112);
  v6 = *(_QWORD *)(a1 + 96);
  if ( !v5 )
    goto LABEL_35;
  v7 = v5 - 1;
  v8 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( v4 != *v9 )
  {
    v24 = 1;
    while ( v10 != -4096 )
    {
      v25 = v24 + 1;
      v8 = v7 & (v24 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v4 == *v9 )
        goto LABEL_5;
      v24 = v25;
    }
LABEL_35:
    v11 = (__int64 *)(v6 + 16LL * v5);
LABEL_36:
    LODWORD(v10) = 1;
    v7 = v5 - 1;
    if ( !v5 )
      return (unsigned int)v10;
    goto LABEL_37;
  }
LABEL_5:
  v11 = (__int64 *)(v6 + 16LL * v5);
  if ( v11 == v9 )
    goto LABEL_36;
  v12 = *((unsigned int *)v9 + 2);
  if ( *(_DWORD *)(a1 + 32) <= (unsigned int)v12 )
  {
LABEL_37:
    LODWORD(v10) = 1;
    v13 = 0;
    goto LABEL_8;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v12);
  LOBYTE(v10) = v13 == 0;
LABEL_8:
  v14 = v7 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v15 = (__int64 *)(v6 + 16LL * v14);
  v16 = *v15;
  if ( v3 != *v15 )
  {
    v26 = 1;
    while ( v16 != -4096 )
    {
      v27 = v26 + 1;
      v14 = v7 & (v26 + v14);
      v15 = (__int64 *)(v6 + 16LL * v14);
      v16 = *v15;
      if ( v3 == *v15 )
        goto LABEL_9;
      v26 = v27;
    }
    return (unsigned int)v10;
  }
LABEL_9:
  if ( v11 == v15 )
    return (unsigned int)v10;
  v17 = *((unsigned int *)v15 + 2);
  if ( *(_DWORD *)(a1 + 32) <= (unsigned int)v17 )
    return (unsigned int)v10;
  v18 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v17);
  if ( v18 == v13 || (_BYTE)v10 )
    goto LABEL_26;
  if ( v18 )
  {
    if ( *(_QWORD *)(v13 + 8) == v18 )
      goto LABEL_26;
    if ( *(_QWORD *)(v18 + 8) != v13 && *(_DWORD *)(v18 + 16) < *(_DWORD *)(v13 + 16) )
    {
      if ( !*(_BYTE *)(a1 + 136) )
      {
        v19 = *(_DWORD *)(a1 + 140) + 1;
        *(_DWORD *)(a1 + 140) = v19;
        if ( v19 <= 0x20 )
        {
          do
          {
            v20 = v13;
            v13 = *(_QWORD *)(v13 + 8);
          }
          while ( v13 && *(_DWORD *)(v18 + 16) <= *(_DWORD *)(v13 + 16) );
          LOBYTE(v10) = v20 == v18;
          return (unsigned int)v10;
        }
        sub_2BF23E0(a1);
      }
      if ( *(_DWORD *)(v13 + 72) >= *(_DWORD *)(v18 + 72) && *(_DWORD *)(v13 + 76) <= *(_DWORD *)(v18 + 76) )
      {
LABEL_26:
        LODWORD(v10) = 1;
        return (unsigned int)v10;
      }
    }
  }
  return 0;
}
