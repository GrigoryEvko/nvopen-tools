// Function: sub_CC3EA0
// Address: 0xcc3ea0
//
__int64 __fastcall sub_CC3EA0(int a1, unsigned __int8 a2)
{
  __int64 v2; // rax
  unsigned int v3; // ecx

  v2 = sub_CC3E70(a2);
  if ( BYTE4(v2) )
    v3 = 8 / (unsigned int)v2;
  else
    v3 = 8 * v2;
  return 8 * a1 / v3;
}
