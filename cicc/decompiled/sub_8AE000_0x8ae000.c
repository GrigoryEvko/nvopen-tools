// Function: sub_8AE000
// Address: 0x8ae000
//
__int64 __fastcall sub_8AE000(__int64 a1)
{
  char v1; // al
  __int64 result; // rax

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( v1 != 8 )
  {
    result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
      return result;
    return sub_8AD220(a1, 0);
  }
  result = sub_8D40F0(a1);
  a1 = result;
  if ( result )
  {
    while ( *(_BYTE *)(a1 + 140) == 12 )
      a1 = *(_QWORD *)(a1 + 160);
    if ( (*(_BYTE *)(a1 + 141) & 0x20) != 0 )
    {
      result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
      if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
        return sub_8AD220(a1, 0);
    }
  }
  return result;
}
