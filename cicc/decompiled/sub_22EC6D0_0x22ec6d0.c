// Function: sub_22EC6D0
// Address: 0x22ec6d0
//
void __fastcall sub_22EC6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r10
  __int64 v8; // r11
  _QWORD *v9; // r8
  __int64 v10; // r9
  _QWORD *v12; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13; // [rsp+18h] [rbp-48h]
  _BYTE v14[64]; // [rsp+20h] [rbp-40h] BYREF

  sub_22EC110(a1, a2, a3, a4, 0);
  v5 = (char *)sub_BD5D20(a4);
  v7 = a3;
  v8 = a2;
  if ( v5 )
  {
    v12 = v14;
    sub_22EB340((__int64 *)&v12, v5, (__int64)&v5[v6]);
    v9 = v12;
    v10 = v13;
    v7 = a3;
    v8 = a2;
  }
  else
  {
    v13 = 0;
    v10 = 0;
    v12 = v14;
    v9 = v14;
    v14[0] = 0;
  }
  sub_22EBD50(a1, a4, v8, v7, (__int64)v9, v10, (__int64)"Function", 8);
  if ( v12 != (_QWORD *)v14 )
    j_j___libc_free_0((unsigned __int64)v12);
}
