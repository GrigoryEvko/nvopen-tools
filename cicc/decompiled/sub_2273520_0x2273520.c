// Function: sub_2273520
// Address: 0x2273520
//
void __fastcall sub_2273520(__int64 a1)
{
  int v1; // r8d
  int v2; // r9d
  __m128i v3; // [rsp+20h] [rbp-9D0h] BYREF
  __int64 v4; // [rsp+30h] [rbp-9C0h]
  int v5; // [rsp+38h] [rbp-9B8h]
  _BYTE v6[152]; // [rsp+40h] [rbp-9B0h] BYREF
  char v7; // [rsp+D8h] [rbp-918h]
  _BYTE v8[2320]; // [rsp+E0h] [rbp-910h] BYREF

  v7 = 0;
  sub_23A0D00(&v3);
  sub_2356B40((unsigned int)v8, 0, (unsigned int)v6, 0, v1, v2, *(_OWORD *)&_mm_loadu_si128(&v3), v4, v5);
  if ( v7 )
  {
    v7 = 0;
    sub_23C66F0(v6);
  }
  sub_233C410(v8, a1);
  sub_2272BE0((__int64)v8);
}
