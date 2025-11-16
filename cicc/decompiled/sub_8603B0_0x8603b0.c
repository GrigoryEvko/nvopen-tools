// Function: sub_8603B0
// Address: 0x8603b0
//
__int64 *__fastcall sub_8603B0(__int64 a1, int a2, unsigned int a3, int a4)
{
  unsigned int v4; // eax

  if ( a3 )
    a3 = 0x800000;
  v4 = a3;
  if ( a4 )
  {
    BYTE1(v4) = BYTE1(a3) | 2;
    a3 = v4;
  }
  return sub_85C120(8u, a2, 0, 0, 0, 0, 0, 0, a1, 0, 0, 0, a3);
}
