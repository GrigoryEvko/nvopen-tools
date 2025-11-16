// Function: sub_3880EA0
// Address: 0x3880ea0
//
__int64 __fastcall sub_3880EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 *v6; // r13
  __int64 result; // rax
  __int64 v8[8]; // [rsp+0h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  v6 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v8, 0.0);
  sub_169E320((_QWORD *)(a1 + 120), v8, v6);
  sub_1698460((__int64)v8);
  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 152) = 1;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)a1 = result;
  return result;
}
