// Function: sub_2957DE0
// Address: 0x2957de0
//
bool __fastcall sub_2957DE0(__int64 a1)
{
  __int64 v2; // rdi
  bool result; // al
  _BYTE *v4; // rdi

  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 0;
  v2 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  result = sub_BCAC40(v2, 1);
  if ( result )
  {
    if ( *(_BYTE *)a1 == 57 )
      return result;
    if ( *(_BYTE *)a1 == 86 && *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) == *(_QWORD *)(a1 + 8) )
    {
      v4 = *(_BYTE **)(a1 - 32);
      if ( *v4 <= 0x15u )
        return sub_AC30F0((__int64)v4);
    }
  }
  return 0;
}
