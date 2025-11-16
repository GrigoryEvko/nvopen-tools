// Function: sub_96DFC0
// Address: 0x96dfc0
//
__int64 __fastcall sub_96DFC0(_QWORD *a1, unsigned int a2)
{
  unsigned int v2; // r8d

  v2 = 0;
  if ( (a1[((unsigned __int64)a2 >> 6) + 1] & (1LL << a2)) == 0 )
    return ((int)*(unsigned __int8 *)(*a1 + (a2 >> 2)) >> (2 * (a2 & 3))) & 3;
  return v2;
}
