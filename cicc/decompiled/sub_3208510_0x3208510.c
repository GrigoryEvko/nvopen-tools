// Function: sub_3208510
// Address: 0x3208510
//
__int64 __fastcall sub_3208510(__int64 a1, __int64 a2)
{
  int v2; // ebx
  _BOOL4 v3; // r14d
  __int64 v4; // rax
  __int16 v6; // [rsp+0h] [rbp-40h] BYREF
  int v7; // [rsp+2h] [rbp-3Eh]
  int v8; // [rsp+8h] [rbp-38h]
  char v9; // [rsp+12h] [rbp-2Eh]

  v2 = (unsigned __int8)(*(_QWORD *)(a2 + 24) >> 3);
  v3 = sub_AE2980(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0)[1] >> 3 == 8;
  v6 = 4098;
  v8 = (2 * v3 + 42) | (v2 << 13);
  v7 = sub_3206530(a1, (unsigned __int8 *)a2, 0);
  v9 = 0;
  v4 = sub_3708FB0(a1 + 648, &v6);
  return sub_3707F80(a1 + 632, v4);
}
