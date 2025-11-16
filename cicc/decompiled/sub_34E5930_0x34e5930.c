// Function: sub_34E5930
// Address: 0x34e5930
//
__int64 __fastcall sub_34E5930(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  _BYTE v4[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v5; // [rsp+8h] [rbp-38h]
  int v6; // [rsp+10h] [rbp-30h]
  unsigned int v7; // [rsp+18h] [rbp-28h]

  v2 = 0;
  sub_34BA1B0((__int64)v4, a2);
  if ( v6 )
  {
    v2 = 1;
    sub_34E54F0((unsigned __int64 *)(a2 + 320), (__int64)v4);
  }
  sub_C7D6A0(v5, 16LL * v7, 8);
  return v2;
}
