// Function: sub_2AC2F10
// Address: 0x2ac2f10
//
void __fastcall sub_2AC2F10(__int64 a1)
{
  int v1; // r15d
  __int64 v2; // rbx
  unsigned int v3; // eax
  __int64 v4; // r14
  __int64 v5; // r13
  int v6; // edx
  int v7; // r13d
  unsigned int v8; // r15d
  unsigned int v9; // eax
  unsigned int v10; // eax

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v1 && !*(_DWORD *)(a1 + 20) )
    return;
  v2 = *(_QWORD *)(a1 + 8);
  v3 = 4 * v1;
  v4 = 72LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v3 = 64;
  v5 = v2 + v4;
  if ( *(_DWORD *)(a1 + 24) <= v3 )
  {
    while ( 1 )
    {
      if ( v5 == v2 )
        goto LABEL_24;
      if ( *(_DWORD *)v2 != -1 )
        break;
      if ( !*(_BYTE *)(v2 + 4) )
      {
        if ( !*(_BYTE *)(v2 + 36) )
          goto LABEL_15;
        goto LABEL_9;
      }
LABEL_10:
      v2 += 72;
    }
    if ( (*(_DWORD *)v2 != -2 || *(_BYTE *)(v2 + 4)) && !*(_BYTE *)(v2 + 36) )
LABEL_15:
      _libc_free(*(_QWORD *)(v2 + 16));
LABEL_9:
    *(_DWORD *)v2 = -1;
    *(_BYTE *)(v2 + 4) = 1;
    goto LABEL_10;
  }
  do
  {
    if ( *(_DWORD *)v2 == -1 )
    {
      if ( !*(_BYTE *)(v2 + 4) && !*(_BYTE *)(v2 + 36) )
        goto LABEL_32;
    }
    else if ( (*(_DWORD *)v2 != -2 || *(_BYTE *)(v2 + 4)) && !*(_BYTE *)(v2 + 36) )
    {
LABEL_32:
      _libc_free(*(_QWORD *)(v2 + 16));
    }
    v2 += 72;
  }
  while ( v5 != v2 );
  v6 = *(_DWORD *)(a1 + 24);
  if ( v1 )
  {
    v7 = 64;
    v8 = v1 - 1;
    if ( v8 )
    {
      _BitScanReverse(&v9, v8);
      v7 = 1 << (33 - (v9 ^ 0x1F));
      if ( v7 < 64 )
        v7 = 64;
    }
    if ( v6 == v7 )
      goto LABEL_39;
    sub_C7D6A0(*(_QWORD *)(a1 + 8), v4, 8);
    v10 = sub_2AAAC60(v7);
    *(_DWORD *)(a1 + 24) = v10;
    if ( v10 )
    {
      *(_QWORD *)(a1 + 8) = sub_C7D670(72LL * v10, 8);
      goto LABEL_39;
    }
LABEL_23:
    *(_QWORD *)(a1 + 8) = 0;
LABEL_24:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  if ( v6 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), v4, 8);
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_23;
  }
LABEL_39:
  sub_2AC2ED0(a1);
}
