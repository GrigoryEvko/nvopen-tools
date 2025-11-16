// Function: sub_10C4C90
// Address: 0x10c4c90
//
__int64 __fastcall sub_10C4C90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 )
    return 0;
  if ( *(_BYTE *)v3 )
    return 0;
  if ( *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( *(_DWORD *)(v3 + 36) != *(_DWORD *)a1 )
    return 0;
  v4 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v4 )
    return 0;
  **(_QWORD **)(a1 + 16) = v4;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v5 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 32) = v5;
  return 1;
}
