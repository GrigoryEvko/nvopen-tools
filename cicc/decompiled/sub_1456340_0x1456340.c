// Function: sub_1456340
// Address: 0x1456340
//
bool __fastcall sub_1456340(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v4; // rbx
  __int64 v5; // rax

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
  if ( (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) != 2 )
    return 0;
  v5 = *(_QWORD *)(v4 - 24);
  if ( *(_BYTE *)(v5 + 16) != 13 )
    return 0;
  result = sub_1455000(v5 + 24);
  if ( !result )
    return 0;
  *a2 = *(_QWORD *)(**(_QWORD **)(v4 - 48) + 24LL);
  return result;
}
