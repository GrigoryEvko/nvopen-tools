// Function: sub_2437EA0
// Address: 0x2437ea0
//
__int64 __fastcall sub_2437EA0(__int64 *a1, unsigned int **a2, __int64 a3)
{
  _QWORD **v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  int v8; // r12d
  __int64 v10; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v11[4]; // [rsp+10h] [rbp-50h] BYREF
  char v12; // [rsp+30h] [rbp-30h]
  char v13; // [rsp+31h] [rbp-2Fh]

  v11[0] = *(_QWORD *)(a3 + 8);
  v5 = (_QWORD **)sub_BCF480(a1, v11, 1, 0);
  v6 = sub_B41A60(v5, (__int64)byte_3F871B3, 0, (__int64)"=r,0", 4, 0, 0, 0, 0);
  v13 = 1;
  v7 = 0;
  v8 = v6;
  v12 = 3;
  v11[0] = ".hwasan.shadow";
  v10 = a3;
  if ( v6 )
    v7 = sub_B3B7D0(v6);
  return sub_921880(a2, v7, v8, (int)&v10, 1, (__int64)v11, 0);
}
