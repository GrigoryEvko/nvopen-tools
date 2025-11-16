// Function: sub_B914E0
// Address: 0xb914e0
//
__int64 __fastcall sub_B914E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12

  v2 = a1 - (8LL * ((*(_BYTE *)(a1 - 16) >> 2) & 0xF) + 16);
  sub_B91430((_BYTE *)(a1 - 16), a2);
  return j___libc_free_0(v2);
}
