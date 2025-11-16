// Function: sub_7DB020
// Address: 0x7db020
//
__int64 __fastcall sub_7DB020(__int64 a1)
{
  __int64 result; // rax

  while ( (unsigned int)sub_8D2E30(a1) )
    a1 = sub_8D46C0(a1);
  if ( !(unsigned int)sub_8D3D10(a1) )
  {
    result = sub_8D3B80(a1);
    if ( !(_DWORD)result )
      return result;
    return (unsigned int)sub_8D23B0(a1) != 0;
  }
  a1 = sub_8D4890(a1);
  result = sub_8D3B80(a1);
  if ( (_DWORD)result )
    return (unsigned int)sub_8D23B0(a1) != 0;
  return result;
}
