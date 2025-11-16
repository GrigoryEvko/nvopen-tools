// Function: sub_104A0A0
// Address: 0x104a0a0
//
__int64 __fastcall sub_104A0A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  bool v8; // zf

  v2 = *(_QWORD *)(a2 - 32);
  v3 = 0;
  if ( *(_BYTE *)v2 == 85 )
  {
    v3 = *(_QWORD *)(v2 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(v2 + 80) )
      {
        v8 = (*(_BYTE *)(v3 + 33) & 0x20) == 0;
        v3 = 0;
        if ( !v8 )
          v3 = *(_QWORD *)(a2 - 32);
      }
      else
      {
        v3 = 0;
      }
    }
  }
  **(_QWORD **)(a1 + 8) = v3;
  v4 = **(_QWORD **)(a1 + 8);
  if ( !v4 )
    return 0;
  v5 = *(_QWORD *)(v4 - 32);
  if ( !v5 || *(_BYTE *)v5 )
    return 0;
  v6 = *(_QWORD *)(v5 + 24);
  if ( (v6 != *(_QWORD *)(v4 + 80) || *(_DWORD *)(v5 + 36) != 369)
    && (v6 != *(_QWORD *)(v4 + 80) || *(_DWORD *)(v5 + 36) != 333) )
  {
    return 0;
  }
  if ( *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)) != *(_QWORD *)a1 )
  {
    if ( *(_QWORD *)a1 == *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) )
    {
      **(_DWORD **)(a1 + 16) = 1;
      return 1;
    }
    return 0;
  }
  **(_DWORD **)(a1 + 16) = 0;
  return 1;
}
