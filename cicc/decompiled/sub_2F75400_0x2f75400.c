// Function: sub_2F75400
// Address: 0x2f75400
//
unsigned __int64 __fastcall sub_2F75400(_QWORD *a1)
{
  __int64 v1; // r8
  unsigned __int64 v2; // rax
  __int64 v3; // rsi
  __int16 v4; // dx
  __int64 v6; // r8
  unsigned __int64 i; // rdx
  __int64 j; // rsi
  __int16 v9; // ax
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r8
  int v15; // eax
  int v16; // r10d

  v1 = a1[5];
  v2 = a1[8];
  v3 = v1 + 48;
  if ( v1 + 48 == v2 )
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[4] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v1 + 24) + 8);
  while ( 1 )
  {
    v4 = *(_WORD *)(v2 + 68);
    if ( (unsigned __int16)(v4 - 14) > 4u && v4 != 24 )
      break;
    if ( (*(_BYTE *)v2 & 4) != 0 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[4] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v1 + 24) + 8);
    }
    else
    {
      while ( (*(_BYTE *)(v2 + 44) & 8) != 0 )
        v2 = *(_QWORD *)(v2 + 8);
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[4] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v1 + 24) + 8);
    }
  }
  v6 = *(_QWORD *)(a1[4] + 32LL);
  for ( i = v2; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(v2 + 44) & 8) != 0; v2 = *(_QWORD *)(v2 + 8) )
    ;
  for ( j = *(_QWORD *)(v2 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v9 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v9 - 14) > 4u && v9 != 24 )
      break;
  }
  v10 = *(_DWORD *)(v6 + 144);
  v11 = *(_QWORD *)(v6 + 128);
  if ( v10 )
  {
    v12 = (v10 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v13 = (__int64 *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( i == *v13 )
      return v13[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
    v15 = 1;
    while ( v14 != -4096 )
    {
      v16 = v15 + 1;
      v12 = (v10 - 1) & (v15 + v12);
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( *v13 == i )
        return v13[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
      v15 = v16;
    }
  }
  return *(_QWORD *)(v11 + 16LL * v10 + 8) & 0xFFFFFFFFFFFFFFF8LL | 4;
}
