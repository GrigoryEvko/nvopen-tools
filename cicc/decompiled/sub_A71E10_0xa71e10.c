// Function: sub_A71E10
// Address: 0xa71e10
//
__int64 __fastcall sub_A71E10(__int64 *a1)
{
  unsigned int v1; // eax
  unsigned int v2; // edx

  v1 = sub_A71B70(*a1);
  v2 = (unsigned __int8)(v1 >> 4);
  BYTE1(v2) = v1 & 0xF;
  return v2;
}
