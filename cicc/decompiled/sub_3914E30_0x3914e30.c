// Function: sub_3914E30
// Address: 0x3914e30
//
__int64 __fastcall sub_3914E30(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  for ( result = a2; (*(_BYTE *)(result + 9) & 0xC) == 8; result = *(_QWORD *)(v3 + 24) )
  {
    v3 = *(_QWORD *)(result + 24);
    *(_BYTE *)(result + 8) |= 4u;
    if ( *(_DWORD *)v3 != 2 )
      break;
  }
  return result;
}
