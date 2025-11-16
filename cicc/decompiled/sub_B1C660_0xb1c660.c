// Function: sub_B1C660
// Address: 0xb1c660
//
bool __fastcall sub_B1C660(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r13d
  __int64 *v6[4]; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v7[8]; // [rsp+20h] [rbp-40h] BYREF

  sub_B1C5B0(v6, *a1, a2);
  v4 = *((_DWORD *)v6[2] + 2);
  sub_B1C5B0(v7, *a1, a3);
  return v4 < *((_DWORD *)v7[2] + 2);
}
