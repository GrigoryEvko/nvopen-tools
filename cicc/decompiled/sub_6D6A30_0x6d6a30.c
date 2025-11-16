// Function: sub_6D6A30
// Address: 0x6d6a30
//
__m128i *sub_6D6A30()
{
  __m128i *v0; // r12
  __int64 v2; // [rsp+8h] [rbp-218h] BYREF
  _BYTE v3[19]; // [rsp+10h] [rbp-210h] BYREF
  char v4; // [rsp+23h] [rbp-1FDh]
  _BYTE v5[360]; // [rsp+B0h] [rbp-170h] BYREF

  sub_6E1DD0(&v2);
  sub_6E1E00(5, v3, 0, 1);
  v4 |= 2u;
  sub_69ED20((__int64)v5, 0, 3, 4096);
  v0 = (__m128i *)sub_6F6F40(v5, 0);
  sub_68A310(v0);
  sub_6E2B30(v0, 0);
  sub_6E1DF0(v2);
  return v0;
}
