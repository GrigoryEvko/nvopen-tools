// Function: sub_10AC150
// Address: 0x10ac150
//
bool __fastcall sub_10AC150(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8

  result = 0;
  if ( *(_BYTE *)a2 == 85 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( *(_BYTE *)v3
        || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80)
        || *(_DWORD *)(v3 + 36) != *(_DWORD *)a1
        || (v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF,
            *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - v5)) != **(_QWORD **)(a1 + 16))
        || (result = 1, *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - v5)) != **(_QWORD **)(a1 + 32)) )
      {
        result = 0;
        if ( !*(_BYTE *)v3
          && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80)
          && *(_DWORD *)(v3 + 36) == *(_DWORD *)(a1 + 40) )
        {
          v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
          if ( *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 48) - v4)) == **(_QWORD **)(a1 + 56) )
            return *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 64) - v4)) == **(_QWORD **)(a1 + 72);
        }
      }
    }
  }
  return result;
}
