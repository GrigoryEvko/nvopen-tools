// Function: sub_1456530
// Address: 0x1456530
//
__int64 __fastcall sub_1456530(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // rdi

  v3 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v3 + 16) != 5 || *(_WORD *)(v3 + 18) != 45 )
    return 0;
  v5 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v5 + 16) != 5 )
    return 0;
  if ( *(_WORD *)(v5 + 18) != 32 )
    return 0;
  if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(v5 - 72)) )
    return 0;
  result = sub_1593BB0(*(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))));
  if ( !(_BYTE)result )
    return 0;
  v6 = *(_QWORD *)(**(_QWORD **)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)) + 24LL);
  if ( (unsigned __int8)(*(_BYTE *)(v6 + 8) - 13) > 1u )
    return 0;
  *a2 = v6;
  *a3 = *(_QWORD *)(v5 + 24 * (2LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
  return result;
}
