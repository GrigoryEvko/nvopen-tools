// Function: sub_39A3CF0
// Address: 0x39a3cf0
//
__int64 __fastcall sub_39A3CF0(__int64 a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5)
{
  _QWORD *v8; // rax
  __int64 v10[8]; // [rsp+0h] [rbp-40h] BYREF

  v8 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 88), 16, 16);
  *v8 = a4;
  v8[1] = a5;
  HIWORD(v10[0]) = 6;
  WORD2(v10[0]) = a3;
  LODWORD(v10[0]) = 5;
  v10[1] = (__int64)v8;
  return sub_39A31C0((__int64 *)(a2 + 8), (__int64 *)(a1 + 88), v10);
}
