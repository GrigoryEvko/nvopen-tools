// Function: sub_1628D40
// Address: 0x1628d40
//
__int64 __fastcall sub_1628D40(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  _BYTE *v3; // rdx

  result = a2;
  if ( !a2 )
    return sub_1627350(a1, 0, 0, 0, 1);
  if ( (unsigned __int8)(*(_BYTE *)a2 - 4) <= 0x1Eu && *(_DWORD *)(a2 + 8) == 1 )
  {
    v3 = *(_BYTE **)(a2 - 8);
    if ( v3 )
    {
      if ( *v3 == 1 )
        return *(_QWORD *)(a2 - 8);
      return result;
    }
    return sub_1627350(a1, 0, 0, 0, 1);
  }
  return result;
}
