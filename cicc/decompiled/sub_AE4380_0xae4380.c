// Function: sub_AE4380
// Address: 0xae4380
//
__int64 __fastcall sub_AE4380(__int64 a1, unsigned int a2)
{
  _DWORD *v2; // rax

  v2 = sub_AE2980(a1, a2);
  return (v2[1] != 0) + ((v2[1] - (unsigned int)(v2[1] != 0)) >> 3);
}
