// Function: sub_8D4D20
// Address: 0x8d4d20
//
_BOOL8 __fastcall sub_8D4D20(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rax

  if ( (unsigned int)sub_8D3070(a1) )
  {
    v1 = sub_8D46C0(a1);
    if ( (*(_BYTE *)(v1 + 140) & 0xFB) == 8 && (sub_8D4C10(v1, dword_4F077C4 != 2) & 1) != 0 )
    {
      if ( (*(_BYTE *)(v1 + 140) & 0xFB) == 8 && (sub_8D4C10(v1, dword_4F077C4 != 2) & 2) != 0 )
        return qword_4D0495C != 0;
      return 1;
    }
    return 0;
  }
  if ( !(unsigned int)sub_8D3110(a1) )
    return 0;
  v3 = sub_8D46C0(a1);
  if ( !sub_8D2310(v3) )
    return 1;
  return (unsigned int)sub_8D3190() == 0;
}
