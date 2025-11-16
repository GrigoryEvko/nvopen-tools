// Function: sub_2304560
// Address: 0x2304560
//
_QWORD *__fastcall sub_2304560(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 v7[6]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v8[10]; // [rsp+30h] [rbp-50h] BYREF

  sub_D89F40(v7, a2 + 8, a3, a4);
  sub_D89930((__int64)v8, v7);
  v4 = (_QWORD *)sub_22077B0(0x38u);
  v5 = v4;
  if ( v4 )
  {
    *v4 = &unk_4A0AD90;
    sub_D89930((__int64)(v4 + 1), v8);
  }
  sub_D89A50((__int64)v8);
  *a1 = v5;
  sub_D89A50((__int64)v7);
  return a1;
}
