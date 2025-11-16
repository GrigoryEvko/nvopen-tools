// Function: sub_11E3750
// Address: 0x11e3750
//
__int64 __fastcall sub_11E3750(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r8
  unsigned __int8 *v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  _BYTE *v12; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v13[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v14; // [rsp+30h] [rbp-30h]

  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(_QWORD *)(a2 - 32 * v5);
  v7 = *(_QWORD *)(a2 + 32 * (1 - v5));
  v12 = *(_BYTE **)(a2 + 32 * (2 - v5));
  v8 = (unsigned __int8 *)sub_B343C0(a3, 0xEEu, v6, 0x100u, v7, 0x100u, (__int64)v12, 0, 0, 0, 0, 0);
  sub_11DAF00(v8, a2);
  v9 = *(_QWORD **)(a3 + 72);
  v14 = 257;
  v10 = sub_BCB2B0(v9);
  return sub_921130((unsigned int **)a3, v10, v6, &v12, 1, (__int64)v13, 3u);
}
