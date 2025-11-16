// Function: sub_8D21F0
// Address: 0x8d21f0
//
__int64 __fastcall sub_8D21F0(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
  {
    if ( *(_QWORD *)(result + 8) )
      break;
  }
  return result;
}
