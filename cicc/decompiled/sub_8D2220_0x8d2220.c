// Function: sub_8D2220
// Address: 0x8d2220
//
__int64 __fastcall sub_8D2220(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
  {
    if ( (*(_BYTE *)(result + 186) & 8) != 0 )
      break;
  }
  return result;
}
