// Function: sub_11FD2F0
// Address: 0x11fd2f0
//
__int64 __fastcall sub_11FD2F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // r13
  __int64 result; // rax
  __int64 v8[8]; // [rsp+0h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 40) = a4;
  *(_QWORD *)(a1 + 48) = a6;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  v6 = sub_C33320();
  sub_C3B1B0((__int64)v8, 0.0);
  sub_C407B0((_QWORD *)(a1 + 120), v8, v6);
  sub_C338F0((__int64)v8);
  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 156) = 1;
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)a1 = result;
  return result;
}
