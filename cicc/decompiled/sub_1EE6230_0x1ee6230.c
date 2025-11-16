// Function: sub_1EE6230
// Address: 0x1ee6230
//
unsigned __int64 __fastcall sub_1EE6230(_QWORD *a1)
{
  __int64 v1; // rsi
  unsigned __int64 v2; // rax
  __int64 v3; // rcx
  __int64 i; // rcx
  __int64 v6; // rdi
  __int64 v7; // rcx
  unsigned int v8; // esi
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v11; // edx
  int v12; // r10d

  v1 = a1[5];
  v2 = a1[8];
  v3 = v1 + 24;
  if ( v1 + 24 == v2 )
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[4] + 272LL) + 392LL) + 16LL * *(unsigned int *)(v1 + 48) + 8);
  while ( (unsigned __int16)(**(_WORD **)(v2 + 16) - 12) <= 1u )
  {
    if ( (*(_BYTE *)v2 & 4) != 0 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[4] + 272LL) + 392LL) + 16LL * *(unsigned int *)(v1 + 48) + 8);
    }
    else
    {
      while ( (*(_BYTE *)(v2 + 46) & 8) != 0 )
        v2 = *(_QWORD *)(v2 + 8);
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[4] + 272LL) + 392LL) + 16LL * *(unsigned int *)(v1 + 48) + 8);
    }
  }
  for ( i = *(_QWORD *)(a1[4] + 272LL); (*(_BYTE *)(v2 + 46) & 4) != 0; v2 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v6 = *(_QWORD *)(i + 368);
  v7 = *(unsigned int *)(i + 384);
  if ( (_DWORD)v7 )
  {
    v8 = (v7 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == v2 )
      return v9[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
    v11 = 1;
    while ( v10 != -8 )
    {
      v12 = v11 + 1;
      v8 = (v7 - 1) & (v11 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( *v9 == v2 )
        return v9[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
      v11 = v12;
    }
  }
  return *(_QWORD *)(v6 + 16 * v7 + 8) & 0xFFFFFFFFFFFFFFF8LL | 4;
}
