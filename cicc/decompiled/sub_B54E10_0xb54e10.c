// Function: sub_B54E10
// Address: 0xb54e10
//
__int64 __fastcall sub_B54E10(__int64 a1)
{
  unsigned __int16 v1; // bx
  __int64 v2; // r13
  __int64 v3; // r14
  char v4; // r15
  __int16 v5; // cx
  char v6; // r8
  __int16 v7; // bx
  unsigned __int64 v8; // rax
  char v9; // r12
  __int64 result; // rax
  char v11; // r8
  char v12; // [rsp+8h] [rbp-68h]
  __int64 v13; // [rsp+8h] [rbp-68h]
  char v14[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v1 = *(_WORD *)(a1 + 2);
  v2 = *(_QWORD *)(a1 + 8);
  v15 = 257;
  v3 = *(_QWORD *)(a1 - 32);
  v4 = *(_BYTE *)(a1 + 72);
  v5 = v1 >> 1;
  v6 = v1 & 1;
  v7 = (v1 >> 7) & 7;
  v12 = v6;
  _BitScanReverse64(&v8, 1LL << v5);
  v9 = 63 - (v8 ^ 0x3F);
  result = sub_BD2C40(80, unk_3F10A14);
  if ( result )
  {
    v11 = v12;
    v13 = result;
    sub_B4D0A0(result, v2, v3, (__int64)v14, v11, v9, v7, v4, 0, 0);
    return v13;
  }
  return result;
}
