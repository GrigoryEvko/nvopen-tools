// Function: sub_2D51390
// Address: 0x2d51390
//
__int64 __fastcall sub_2D51390(__int64 a1, const void *a2, size_t a3)
{
  unsigned int v3; // r12d
  _BYTE v5[8]; // [rsp+0h] [rbp-60h] BYREF
  char *v6; // [rsp+8h] [rbp-58h]
  char v7; // [rsp+18h] [rbp-48h] BYREF

  sub_2D51270((__int64)v5, a1, a2, a3);
  v3 = v5[0];
  if ( v6 != &v7 )
    _libc_free((unsigned __int64)v6);
  return v3;
}
