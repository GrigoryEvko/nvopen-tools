// Function: sub_B54D40
// Address: 0xb54d40
//
__int64 __fastcall sub_B54D40(__int64 a1)
{
  __int64 *v1; // r15
  unsigned int v2; // r13d
  unsigned __int64 v3; // rax
  unsigned int v4; // r13d
  unsigned __int8 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int16 v8; // ax
  __int64 v10; // [rsp+8h] [rbp-68h]
  char v11[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v12; // [rsp+30h] [rbp-40h]

  v1 = *(__int64 **)(a1 + 72);
  v2 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL);
  v10 = *(_QWORD *)(a1 - 32);
  _BitScanReverse64(&v3, 1LL << *(_WORD *)(a1 + 2));
  v12 = 257;
  v4 = v2 >> 8;
  v5 = 63 - (v3 ^ 0x3F);
  v6 = sub_BD2C40(80, unk_3F10A14);
  v7 = v6;
  if ( v6 )
    sub_B4CCA0(v6, v1, v4, v10, v5, (__int64)v11, 0, 0);
  v8 = *(_WORD *)(a1 + 2) & 0x40 | *(_WORD *)(v7 + 2) & 0xFFBF;
  *(_WORD *)(v7 + 2) = v8;
  LOBYTE(v8) = v8 & 0x7F;
  *(_WORD *)(v7 + 2) = *(_WORD *)(a1 + 2) & 0x80 | v8;
  return v7;
}
