// Function: sub_117B160
// Address: 0x117b160
//
__int64 __fastcall sub_117B160(unsigned __int8 *a1)
{
  unsigned int v1; // eax
  unsigned int v2; // edx

  v1 = *a1;
  v2 = v1 - 48;
  LOBYTE(v2) = v1 - 48 <= 1;
  v1 -= 51;
  LOBYTE(v1) = (unsigned __int8)v1 <= 1u;
  return v2 | v1;
}
