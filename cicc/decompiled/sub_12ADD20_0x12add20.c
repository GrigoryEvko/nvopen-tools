// Function: sub_12ADD20
// Address: 0x12add20
//
__int64 __fastcall sub_12ADD20(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // [rsp+8h] [rbp-A8h]
  __int64 v13; // [rsp+10h] [rbp-A0h]
  __int64 v14; // [rsp+18h] [rbp-98h]
  int v15; // [rsp+2Ch] [rbp-84h] BYREF
  _BYTE v16[16]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v17; // [rsp+40h] [rbp-70h]
  _QWORD v18[12]; // [rsp+50h] [rbp-60h] BYREF

  v12 = *(_QWORD *)(a4 + 16);
  v5 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 16LL);
  v13 = *(_QWORD *)(v12 + 16);
  v14 = *(_QWORD *)(v5 + 16);
  v6 = sub_1643350(*(_QWORD *)(a2 + 40));
  v7 = sub_159C470(v6, a3, 0);
  v8 = sub_126A190(*(_QWORD **)(a2 + 32), 4405, 0, 0);
  v18[1] = v7;
  v18[0] = sub_128F980(a2, v12);
  v18[2] = sub_128F980(a2, v13);
  v18[3] = sub_128F980(a2, v5);
  v18[4] = sub_128F980(a2, v14);
  v17 = 257;
  v9 = sub_1285290((__int64 *)(a2 + 48), *(_QWORD *)(v8 + 24), v8, (int)v18, 5, (__int64)v16, 0);
  v17 = 257;
  v15 = 0;
  v10 = sub_12A9E60((__int64 *)(a2 + 48), v9, (__int64)&v15, 1, (__int64)v16);
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
