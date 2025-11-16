// Function: sub_B54EC0
// Address: 0xb54ec0
//
__int64 __fastcall sub_B54EC0(__int64 a1)
{
  unsigned __int16 v1; // r9
  __int64 v2; // r12
  __int64 v3; // r13
  char v4; // r15
  char v5; // r14
  unsigned __int64 v6; // rax
  char v7; // bl
  __int64 result; // rax
  __int16 v9; // r9
  __int16 v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v1 = *(_WORD *)(a1 + 2);
  v2 = *(_QWORD *)(a1 - 64);
  v3 = *(_QWORD *)(a1 - 32);
  v4 = *(_BYTE *)(a1 + 72);
  v5 = v1 & 1;
  v10 = (v1 >> 7) & 7;
  _BitScanReverse64(&v6, 1LL << (v1 >> 1));
  v7 = 63 - (v6 ^ 0x3F);
  result = sub_BD2C40(80, unk_3F10A10);
  if ( result )
  {
    v9 = v10;
    v11 = result;
    sub_B4D260(result, v2, v3, v5, v7, v9, v4, 0, 0);
    return v11;
  }
  return result;
}
