// Function: sub_8D22E0
// Address: 0x8d22e0
//
__int64 __fastcall sub_8D22E0(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
  {
    if ( (*(_BYTE *)(result + 186) & 0x18) != 0 )
      break;
  }
  return result;
}
