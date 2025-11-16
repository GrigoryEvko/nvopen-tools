// Function: sub_14563F0
// Address: 0x14563f0
//
__int64 __fastcall sub_14563F0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rax

  v2 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v2 + 16) != 5 || *(_WORD *)(v2 + 18) != 45 )
    return 0;
  v4 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v4 + 16) != 5 )
    return 0;
  if ( *(_WORD *)(v4 + 18) != 32 )
    return 0;
  if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF))) )
    return 0;
  v5 = *(_QWORD *)(**(_QWORD **)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)) + 24LL);
  if ( *(_BYTE *)(v5 + 8) != 13 )
    return 0;
  if ( (*(_BYTE *)(v5 + 9) & 2) != 0 )
    return 0;
  if ( (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(v4 - 48)) )
    return 0;
  v6 = *(_QWORD *)(v4 + 24 * (2LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v6 + 16) != 13 )
    return 0;
  if ( !sub_1455000(v6 + 24) )
    return 0;
  if ( *(_DWORD *)(v5 + 12) != 2 )
    return 0;
  result = sub_1642F90(**(_QWORD **)(v5 + 16), 1);
  if ( !(_BYTE)result )
    return 0;
  *a2 = *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL);
  return result;
}
