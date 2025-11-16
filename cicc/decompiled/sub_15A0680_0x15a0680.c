// Function: sub_15A0680
// Address: 0x15a0680
//
__int64 __fastcall sub_15A0680(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax

  v3 = a1;
  if ( *(_BYTE *)(a1 + 8) == 16 )
    a1 = **(_QWORD **)(a1 + 16);
  result = sub_159C470(a1, a2, a3);
  if ( *(_BYTE *)(v3 + 8) == 16 )
    return sub_15A0390(*(_QWORD *)(v3 + 32), result);
  return result;
}
