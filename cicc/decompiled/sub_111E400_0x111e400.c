// Function: sub_111E400
// Address: 0x111e400
//
__int64 __fastcall sub_111E400(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax

  if ( *(_BYTE *)a2 != 93 )
    return 0;
  if ( *(_DWORD *)(a2 + 80) != 1 )
    return 0;
  if ( **(_DWORD **)(a2 + 72) )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v3 != 85 )
    return 0;
  v4 = *(_QWORD *)(v3 - 32);
  if ( !v4 )
    return 0;
  if ( *(_BYTE *)v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(v3 + 80) )
    return 0;
  if ( *(_DWORD *)(v4 + 36) != *(_DWORD *)a1 )
    return 0;
  v5 = *(_QWORD *)(v3 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 16) = v5;
  if ( *(_BYTE *)v3 != 85 )
    return 0;
  v6 = *(_QWORD *)(v3 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
  if ( !v6 )
    return 0;
  **(_QWORD **)(a1 + 32) = v6;
  return 1;
}
