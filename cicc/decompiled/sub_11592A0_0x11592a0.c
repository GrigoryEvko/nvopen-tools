// Function: sub_11592A0
// Address: 0x11592a0
//
__int64 __fastcall sub_11592A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 )
    return 0;
  if ( *(_BYTE *)v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( *(_DWORD *)(v4 + 36) != *(_DWORD *)a1 )
    return 0;
  v5 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 8) )
    return 0;
  if ( *(_BYTE *)v5 != 47 )
    return 0;
  v7 = *(_QWORD *)(v5 - 64);
  if ( !v7 )
    return 0;
  **(_QWORD **)(a1 + 16) = v7;
  return sub_1009690((double *)(a1 + 24), *(_QWORD *)(v5 - 32));
}
