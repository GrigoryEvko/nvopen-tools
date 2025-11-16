// Function: sub_1071DB0
// Address: 0x1071db0
//
__int64 __fastcall sub_1071DB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  for ( result = a2; (*(_BYTE *)(result + 9) & 0x70) == 0x20; result = *(_QWORD *)(v3 + 16) )
  {
    v3 = *(_QWORD *)(result + 24);
    *(_BYTE *)(result + 8) |= 8u;
    if ( *(_BYTE *)v3 != 2 )
      break;
  }
  return result;
}
