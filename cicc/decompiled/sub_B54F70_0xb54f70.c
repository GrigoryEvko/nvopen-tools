// Function: sub_B54F70
// Address: 0xb54f70
//
__int64 __fastcall sub_B54F70(__int64 a1)
{
  __int64 v1; // r15
  unsigned __int16 v2; // r9
  __int16 v3; // r13
  unsigned __int64 v4; // rax
  int v5; // r14d
  __int64 v6; // r12
  __int16 v7; // ax
  __int64 v9; // [rsp+8h] [rbp-48h]
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int16 v11; // [rsp+18h] [rbp-38h]
  char v12; // [rsp+1Ch] [rbp-34h]

  v1 = *(_QWORD *)(a1 - 96);
  v2 = *(_WORD *)(a1 + 2);
  v9 = *(_QWORD *)(a1 - 64);
  v10 = *(_QWORD *)(a1 - 32);
  v3 = (unsigned __int8)v2 >> 5;
  _BitScanReverse64(&v4, 1LL << SHIBYTE(v2));
  v11 = (v2 >> 2) & 7;
  v5 = 63 - (v4 ^ 0x3F);
  v12 = *(_BYTE *)(a1 + 72);
  v6 = sub_BD2C40(80, unk_3F148C4);
  if ( v6 )
    sub_B4D5A0(v6, v1, v9, v10, v5, v11, v3, v12, 0, 0);
  v7 = *(_WORD *)(v6 + 2) & 0xFFFE | *(_WORD *)(a1 + 2) & 1;
  *(_WORD *)(v6 + 2) = v7;
  *(_WORD *)(v6 + 2) = *(_WORD *)(a1 + 2) & 2 | v7 & 0xFFFD;
  return v6;
}
