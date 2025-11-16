// Function: sub_FF06D0
// Address: 0xff06d0
//
bool __fastcall sub_FF06D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  _DWORD v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_F02DB0(v6, 4u, 5u);
  v4 = sub_FF0430(a1, a2, a3);
  return v4 > v6[0];
}
