// Function: sub_F5A4F0
// Address: 0xf5a4f0
//
__int64 __fastcall sub_F5A4F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdx
  _BYTE *v9; // rax

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( *(_DWORD *)(v3 + 36) != *(_DWORD *)a1 )
    return 0;
  v5 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 16) = v5;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v6 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v6 )
    return 0;
  **(_QWORD **)(a1 + 32) = v6;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v7 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 40) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v7 == 17 )
  {
    **(_QWORD **)(a1 + 48) = v7 + 24;
    return 1;
  }
  v8 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
  if ( (unsigned int)v8 > 1 )
    return 0;
  if ( *(_BYTE *)v7 > 0x15u )
    return 0;
  v9 = sub_AD7630(v7, *(unsigned __int8 *)(a1 + 56), v8);
  if ( !v9 || *v9 != 17 )
    return 0;
  **(_QWORD **)(a1 + 48) = v9 + 24;
  return 1;
}
