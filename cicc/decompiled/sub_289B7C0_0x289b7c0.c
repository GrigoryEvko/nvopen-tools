// Function: sub_289B7C0
// Address: 0x289b7c0
//
__int64 __fastcall sub_289B7C0(unsigned int ***a1, __int64 a2, unsigned int a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v14[10]; // [rsp+10h] [rbp-50h] BYREF

  v8 = sub_BCDA70(*(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL), a4 * a3);
  v14[0] = a2;
  v13 = v8;
  v9 = sub_BCB2D0((*a1)[9]);
  v14[1] = sub_ACD640(v9, a3, 0);
  v10 = sub_BCB2D0((*a1)[9]);
  v14[2] = sub_ACD640(v10, a4, 0);
  v11 = sub_B6E160(*(__int64 **)(*((_QWORD *)(*a1)[6] + 9) + 40LL), 0xEAu, (__int64)&v13, 1);
  return sub_921880(*a1, *(_QWORD *)(v11 + 24), v11, (int)v14, 3, a5, 0);
}
