// Function: sub_156E6B0
// Address: 0x156e6b0
//
__int64 __fastcall sub_156E6B0(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  _QWORD v11[2]; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v12[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v13; // [rsp+20h] [rbp-30h]

  v5 = sub_15F2050(a2);
  v6 = sub_15E26F0(v5, a3, 0, 0);
  v13 = 257;
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v8 = *(_QWORD *)(v6 + 24);
  v11[0] = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v11[1] = *(_QWORD *)(a2 + 24 * (1 - v7));
  v9 = sub_1285290(a1, v8, v6, (int)v11, 2, (__int64)v12, 0);
  return sub_156BB10(
           a1,
           *(_BYTE **)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
           v9,
           *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
}
