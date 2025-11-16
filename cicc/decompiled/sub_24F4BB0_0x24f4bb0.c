// Function: sub_24F4BB0
// Address: 0x24f4bb0
//
__int64 __fastcall sub_24F4BB0(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  int v4; // eax
  unsigned __int64 v6; // r13
  __int64 **v7; // r14
  unsigned int v8; // ebx
  unsigned int v9; // eax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r12
  unsigned __int64 v16; // [rsp+18h] [rbp-78h] BYREF
  int v17; // [rsp+28h] [rbp-68h]
  _BYTE v18[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v19; // [rsp+50h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 280);
  v16 = a3;
  if ( (unsigned int)(v4 - 1) > 1 )
    BUG();
  v6 = *(_QWORD *)(a1 + 336);
  v19 = 257;
  v7 = *(__int64 ***)(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL) + 8LL);
  v8 = sub_BCB060(*(_QWORD *)(v16 + 8));
  v9 = sub_BCB060((__int64)v7);
  v10 = sub_24F2F10(a2, (unsigned int)(v8 <= v9) + 38, v16, v7, (__int64)v18, 0, v17, 0);
  v19 = 257;
  v11 = *(_QWORD *)(v6 + 24);
  v16 = v10;
  v12 = sub_921880((unsigned int **)a2, v11, v6, (int)&v16, 1, (__int64)v18, 0);
  v13 = v12;
  *(_WORD *)(v12 + 2) = *(_WORD *)(v12 + 2) & 0xF003 | (4 * ((*(_WORD *)(v6 + 2) >> 4) & 0x3FF));
  if ( a4 )
    sub_24F49B0(a4, v12, v6);
  return v13;
}
