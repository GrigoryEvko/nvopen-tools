// Function: sub_2A3DCE0
// Address: 0x2a3dce0
//
__int64 __fastcall sub_2A3DCE0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v6[0] = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  sub_2A3D0F0(a3, (__int64 (__fastcall *)(__int64, __int64))sub_2A3CDD0, (__int64)v6);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
