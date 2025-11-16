// Function: sub_B55060
// Address: 0xb55060
//
__int64 __fastcall sub_B55060(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // r15
  unsigned __int16 v3; // r12
  __int16 v4; // r9
  __int16 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int8 v7; // r13
  __int64 result; // rax
  char v9; // [rsp-20h] [rbp-60h]
  __int16 v10; // [rsp+4h] [rbp-3Ch]
  char v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 - 64);
  v2 = *(_QWORD *)(a1 - 32);
  v3 = *(_WORD *)(a1 + 2);
  v4 = (v3 >> 1) & 7;
  v5 = (v3 >> 4) & 0x1F;
  v10 = v4;
  _BitScanReverse64(&v6, 1LL << (*(_WORD *)(a1 + 2) >> 9));
  v7 = 63 - (v6 ^ 0x3F);
  v11 = *(_BYTE *)(a1 + 72);
  result = sub_BD2C40(80, unk_3F148C0);
  if ( result )
  {
    v9 = v11;
    v12 = result;
    sub_B4D750(result, v5, v1, v2, v7, v10, v9, 0, 0);
    result = v12;
  }
  *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFE | *(_WORD *)(a1 + 2) & 1;
  return result;
}
