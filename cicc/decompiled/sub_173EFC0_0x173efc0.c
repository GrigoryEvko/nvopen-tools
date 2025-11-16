// Function: sub_173EFC0
// Address: 0x173efc0
//
__int64 __fastcall sub_173EFC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  if ( *(_BYTE *)(a2 + 16) != 78 )
    return 0;
  v3 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v3 + 16) )
    return 0;
  if ( *(_DWORD *)(v3 + 36) != *(_DWORD *)a1 )
    return 0;
  v4 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL)
                 + 24
                 * (*(unsigned int *)(a1 + 8)
                  - (unsigned __int64)(*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
  if ( !v4 )
    return 0;
  **(_QWORD **)(a1 + 16) = v4;
  if ( *(_BYTE *)(a2 + 16) != 78 )
    return 0;
  v5 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL)
                 + 24
                 * (*(unsigned int *)(a1 + 24)
                  - (unsigned __int64)(*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 32) = v5;
  return 1;
}
