// Function: sub_39C7D40
// Address: 0x39c7d40
//
bool __fastcall sub_39C7D40(__int64 a1, __int64 a2)
{
  bool result; // al
  int v3; // edx

  if ( *(_BYTE *)(a2 + 24) )
    return 0;
  if ( !sub_39A24D0(a1, *(_QWORD *)(a2 + 8)) )
  {
    v3 = *(_DWORD *)(a2 + 88);
    result = 1;
    if ( !v3 )
      return result;
    if ( v3 == 1 )
      return sub_397FB50(*(_QWORD *)(a1 + 200), *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8LL)) == 0;
  }
  return 0;
}
