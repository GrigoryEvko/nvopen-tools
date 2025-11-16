// Function: sub_B49E20
// Address: 0xb49e20
//
__int64 __fastcall sub_B49E20(__int64 a1)
{
  unsigned int v1; // eax

  v1 = sub_B49D00(a1);
  return (((unsigned __int8)((v1 >> 6) | (v1 >> 4) | v1 | (v1 >> 2)) >> 1) ^ 1) & 1;
}
