// Function: sub_1297100
// Address: 0x1297100
//
__int64 __fastcall sub_1297100(__int64 *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r14
  _QWORD *v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // rax
  unsigned __int8 v8; // [rsp+Fh] [rbp-31h]

  v2 = (_QWORD *)sub_12A4D50(a1, "while.cond", 0, 0);
  sub_1290AF0(a1, v2, 0);
  v3 = (_QWORD *)sub_12A4D50(a1, "while.end", 0, 0);
  v4 = (_QWORD *)sub_12A4D50(a1, "while.body", 0, 0);
  v8 = sub_12905B0(a2[9], 0);
  v5 = sub_127FEC0((__int64)a1, a2[6]);
  sub_12A4DB0(a1, v5, v4, v3, v8);
  sub_1290AF0(a1, v4, 0);
  sub_1296350(a1, a2[9]);
  v6 = sub_12909B0(a1, (__int64)v2);
  if ( v6 && a2[8] )
    sub_1291160((__int64)a1, (__int64)v6, (__int64)a2);
  return sub_1290AF0(a1, v3, 1);
}
