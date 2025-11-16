// Function: sub_39A3940
// Address: 0x39a3940
//
__int64 __fastcall sub_39A3940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5[6]; // [rsp+0h] [rbp-30h] BYREF

  sub_39A34D0(a1, a2, 60);
  v5[1] = a3;
  v5[0] = 0x20006900000001LL;
  return sub_39A31C0((__int64 *)(a2 + 8), (__int64 *)(a1 + 88), v5);
}
