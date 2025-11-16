// Function: sub_937020
// Address: 0x937020
//
__int64 __fastcall sub_937020(_QWORD *a1, __int64 *a2)
{
  unsigned __int64 v2; // r15
  __int64 v3; // r14
  _QWORD *v4; // r13
  __int64 v5; // rax
  unsigned __int64 v6; // rsi
  unsigned int v8; // r14d
  __int64 v9; // rax
  _QWORD *v10; // [rsp+8h] [rbp-48h]
  unsigned __int8 v11; // [rsp+10h] [rbp-40h]
  _QWORD *v12; // [rsp+18h] [rbp-38h]

  v2 = a2[9];
  v3 = a2[10];
  v12 = (_QWORD *)sub_945CA0(a1, "if.then", 0, 0);
  v4 = (_QWORD *)sub_945CA0(a1, "if.end", 0, 0);
  if ( v3 )
  {
    v10 = (_QWORD *)sub_945CA0(a1, "if.else", 0, 0);
    v11 = sub_92F9D0(v2, v3);
    v5 = sub_921E00((__int64)a1, a2[6]);
    sub_945D00(a1, v5, v12, v10, v11);
    sub_92FEA0((__int64)a1, v12, 0);
    sub_9363D0(a1, v2);
    sub_92FD90((__int64)a1, (__int64)v4);
    sub_92FEA0((__int64)a1, v10, 0);
    v6 = v3;
  }
  else
  {
    v8 = sub_92F9D0(v2, 0);
    v9 = sub_921E00((__int64)a1, a2[6]);
    sub_945D00(a1, v9, v12, v4, v8);
    sub_92FEA0((__int64)a1, v12, 0);
    v6 = v2;
  }
  sub_9363D0(a1, v6);
  sub_92FD90((__int64)a1, (__int64)v4);
  return sub_92FEA0((__int64)a1, v4, 1);
}
