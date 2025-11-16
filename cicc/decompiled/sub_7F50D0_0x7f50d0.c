// Function: sub_7F50D0
// Address: 0x7f50d0
//
_BOOL8 __fastcall sub_7F50D0(__int64 a1)
{
  char v1; // al
  _BOOL8 result; // rax
  __int64 v3; // rdx

  v1 = *(_BYTE *)(a1 + 173);
  if ( v1 == 10 )
    return *(_BYTE *)(a1 + 192) & 1;
  while ( v1 == 11 )
  {
    a1 = *(_QWORD *)(a1 + 176);
    v1 = *(_BYTE *)(a1 + 173);
    if ( v1 == 10 )
      return *(_BYTE *)(a1 + 192) & 1;
  }
  if ( v1 != 9 )
    return 0;
  v3 = *(_QWORD *)(a1 + 176);
  result = 0;
  if ( *(_BYTE *)(v3 + 48) == 6 )
    return (unsigned int)sub_7F50D0(*(_QWORD *)(v3 + 56)) != 0;
  return result;
}
