// Function: sub_265E6F0
// Address: 0x265e6f0
//
__int64 __fastcall sub_265E6F0(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rax
  _DWORD *v3; // rdx
  int v5; // edi
  __int64 v6; // r10
  int v7; // r9d
  unsigned int v8; // esi
  int v9; // r8d
  int v10; // r11d

  v2 = *(_DWORD **)(a1 + 8);
  v3 = &v2[*(unsigned int *)(a1 + 24)];
  if ( !*(_DWORD *)(a1 + 16) || v2 == v3 )
    return 0;
  while ( *v2 > 0xFFFFFFFD )
  {
    if ( ++v2 == v3 )
      return 0;
  }
  if ( v2 == v3 )
    return 0;
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = v5 - 1;
LABEL_9:
  if ( !v5 )
  {
    while ( 1 )
    {
      if ( ++v2 == v3 )
        return 0;
      if ( *v2 <= 0xFFFFFFFD )
      {
        if ( v2 != v3 )
          goto LABEL_9;
        return 0;
      }
    }
  }
  while ( 1 )
  {
    v8 = v7 & (37 * *v2);
    v9 = *(_DWORD *)(v6 + 4LL * v8);
    if ( *v2 == v9 )
      break;
    v10 = 1;
    while ( v9 != -1 )
    {
      v8 = v7 & (v10 + v8);
      v9 = *(_DWORD *)(v6 + 4LL * v8);
      if ( *v2 == v9 )
        return 1;
      ++v10;
    }
    do
    {
      if ( ++v2 == v3 )
        return 0;
    }
    while ( *v2 > 0xFFFFFFFD );
    if ( v2 == v3 )
      return 0;
  }
  return 1;
}
