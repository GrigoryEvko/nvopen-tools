// Function: sub_1E6D6E0
// Address: 0x1e6d6e0
//
__int64 *__fastcall sub_1E6D6E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // rcx
  char v8; // al
  __int64 v9; // rsi
  _QWORD v11[2]; // [rsp+0h] [rbp-20h] BYREF
  char v12; // [rsp+10h] [rbp-10h]

  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_BYTE *)(a1 + 32);
  v9 = *(_QWORD *)(a1 + 48);
  v11[1] = *(_QWORD *)(a1 + 24);
  v11[0] = v7;
  v12 = v8;
  return sub_1E6D620(v6, v9, (__int64)v11, v7, v6, a6);
}
