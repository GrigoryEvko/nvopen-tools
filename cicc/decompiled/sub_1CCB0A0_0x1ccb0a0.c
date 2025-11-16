// Function: sub_1CCB0A0
// Address: 0x1ccb0a0
//
unsigned __int64 __fastcall sub_1CCB0A0(unsigned __int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // rdx

  v2 = a1;
  v3 = a2;
  if ( !a2 )
    return a1;
  while ( 1 )
  {
    v4 = v2 % v3;
    v2 = v3;
    if ( !v4 )
      break;
    v3 = v4;
  }
  return v3;
}
