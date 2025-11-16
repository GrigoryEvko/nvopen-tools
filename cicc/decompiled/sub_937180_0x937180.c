// Function: sub_937180
// Address: 0x937180
//
__int64 __fastcall sub_937180(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r14
  _QWORD *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int8 v9; // [rsp+Fh] [rbp-31h]

  v2 = (_QWORD *)sub_945CA0(a1, "while.cond", 0, 0);
  sub_92FEA0((__int64)a1, v2, 0);
  v3 = (_QWORD *)sub_945CA0(a1, "while.end", 0, 0);
  v4 = (_QWORD *)sub_945CA0(a1, "while.body", 0, 0);
  v9 = sub_92F9D0(a2[9], 0);
  v5 = sub_921E00((__int64)a1, a2[6]);
  sub_945D00(a1, v5, v4, v3, v9);
  sub_92FEA0((__int64)a1, v4, 0);
  sub_9363D0(a1, a2[9]);
  v6 = sub_92FD90((__int64)a1, (__int64)v2);
  if ( v6 )
  {
    v7 = v6;
    if ( a2[8] )
      sub_9305A0((__int64)a1, v6, (__int64)a2);
    sub_930810((__int64)a1, v7);
  }
  return sub_92FEA0((__int64)a1, v3, 1);
}
