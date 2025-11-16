// Function: sub_15FB730
// Address: 0x15fb730
//
bool __fastcall sub_15FB730(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  bool result; // al
  __int64 v7; // rdi

  if ( *(_BYTE *)(a1 + 16) != 52 )
    return 0;
  v5 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v5 + 16) > 0x10u || !(result = sub_1596070(v5, a2, a3, a4)) )
  {
    v7 = *(_QWORD *)(a1 - 48);
    return *(_BYTE *)(v7 + 16) <= 0x10u && sub_1596070(v7, a2, a3, a4);
  }
  return result;
}
