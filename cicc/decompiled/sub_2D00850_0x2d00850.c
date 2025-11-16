// Function: sub_2D00850
// Address: 0x2d00850
//
__int64 __fastcall sub_2D00850(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // eax
  unsigned int v4; // r8d
  int v5; // edx

  v3 = a2;
  v4 = a3;
  if ( !a3 )
    return a2;
  while ( 1 )
  {
    v5 = v3 % v4;
    v3 = v4;
    if ( !v5 )
      break;
    v4 = v5;
  }
  return v4;
}
