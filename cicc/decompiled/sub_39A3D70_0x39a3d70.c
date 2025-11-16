// Function: sub_39A3D70
// Address: 0x39a3d70
//
__int64 __fastcall sub_39A3D70(__int64 a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5)
{
  _QWORD *v7; // rax
  __int64 v8; // rdi
  __int64 *v11; // [rsp+8h] [rbp-48h]
  __int64 v12[8]; // [rsp+10h] [rbp-40h] BYREF

  v11 = (__int64 *)(a1 + 88);
  v7 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 88), 16, 16);
  v8 = *(_QWORD *)(a1 + 200);
  v7[1] = a5;
  *v7 = a4;
  WORD2(v12[0]) = a3;
  v12[1] = (__int64)v7;
  LODWORD(v12[0]) = 5;
  HIWORD(v12[0]) = (unsigned __int16)sub_398C0A0(v8) < 4u ? 6 : 23;
  return sub_39A31C0((__int64 *)(a2 + 8), v11, v12);
}
