// Function: sub_697870
// Address: 0x697870
//
_BOOL8 __fastcall sub_697870(__int64 a1)
{
  char v1; // cl
  _BOOL8 result; // rax
  __int64 v3; // rax
  char i; // dl

  v1 = *(_BYTE *)(a1 + 24);
  if ( !v1 )
    return 1;
  v3 = *(_QWORD *)(a1 + 8);
  for ( i = *(_BYTE *)(v3 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  if ( !i )
    return 1;
  if ( v1 == 1 )
  {
    if ( (unsigned int)sub_732350(*(_QWORD *)(a1 + 152)) )
      return 1;
    v1 = *(_BYTE *)(a1 + 24);
  }
  if ( v1 == 2 && (unsigned int)sub_7323F0(a1 + 152) )
    return 1;
  result = 0;
  if ( (*(_BYTE *)(a1 + 27) & 8) != 0 )
    return (unsigned int)sub_893F30(*(_QWORD *)(a1 + 112)) != 0;
  return result;
}
