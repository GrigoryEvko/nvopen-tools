// Function: sub_8D4C80
// Address: 0x8d4c80
//
_BOOL8 __fastcall sub_8D4C80(__int64 a1)
{
  char v1; // al
  _BOOL8 result; // rax
  __int64 v3; // r12

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( v1 != 6 || (*(_BYTE *)(a1 + 168) & 1) != 0 )
    return 0;
  v3 = sub_8D46C0(a1);
  if ( !sub_8D2600(v3) )
    return 0;
  result = 1;
  if ( (*(_BYTE *)(v3 + 140) & 0xFB) == 8 )
    return (unsigned int)sub_8D4C10(v3, dword_4F077C4 != 2) == 0;
  return result;
}
