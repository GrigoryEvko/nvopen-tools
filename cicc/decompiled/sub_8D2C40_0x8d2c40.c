// Function: sub_8D2C40
// Address: 0x8d2c40
//
_BOOL8 __fastcall sub_8D2C40(__int64 a1)
{
  __int64 i; // rbx
  _BOOL8 result; // rax
  char v3; // dl

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = sub_8D2820(i);
  if ( result )
    return 1;
  v3 = *(_BYTE *)(i + 140);
  if ( v3 == 18 )
    return 1;
  if ( v3 == 3 )
  {
    result = 1;
    if ( (*(_BYTE *)(i + 160) & 0xF7) != 1 )
      return ((*(_BYTE *)(i + 160) - 2) & 0xFD) == 0;
  }
  return result;
}
