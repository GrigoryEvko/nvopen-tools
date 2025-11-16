// Function: sub_AE43D0
// Address: 0xae43d0
//
__int64 __fastcall sub_AE43D0(__int64 a1, unsigned int a2)
{
  _DWORD *v2; // rax

  v2 = sub_AE2980(a1, a2);
  return (v2[3] != 0) + ((v2[3] - (unsigned int)(v2[3] != 0)) >> 3);
}
