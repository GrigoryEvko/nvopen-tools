// Function: sub_16CB1E0
// Address: 0x16cb1e0
//
__int64 __fastcall sub_16CB1E0(__int64 a1)
{
  unsigned int *v1; // rax
  unsigned int v2; // edx

  sub_16CB120(a1);
  v1 = (unsigned int *)(a1 + 64);
  do
  {
    v2 = *v1++;
    v1[6] = _byteswap_ulong(v2);
  }
  while ( v1 != (unsigned int *)(a1 + 84) );
  return a1 + 92;
}
