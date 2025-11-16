// Function: sub_1683CB0
// Address: 0x1683cb0
//
__int64 __fastcall sub_1683CB0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rdx
  __int64 result; // rax

  if ( !a1 )
    return 0xFFFFFFFFLL;
  do
  {
    v1 = a1;
    a1 &= a1 - 1;
  }
  while ( a1 );
  _BitScanForward64((unsigned __int64 *)&result, v1);
  return result;
}
