// Function: sub_6966B0
// Address: 0x6966b0
//
__int64 __fastcall sub_6966B0(int a1, __int64 a2, int a3)
{
  __int64 v4; // rdi
  char v6; // [rsp+4h] [rbp-CCh] BYREF
  __int64 v7; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE v8[192]; // [rsp+10h] [rbp-C0h] BYREF

  sub_6E2250(v8, &v7, 4, 1, 0, a2);
  sub_8326B0(a1, 0, 1, a3, 0, (unsigned int)&v6, a2 + 8, a2, a2, 0);
  v4 = *(_QWORD *)(a2 + 8);
  if ( v4 )
    sub_6E2920(v4);
  return sub_6E2C70(v7, 1, 0, a2);
}
