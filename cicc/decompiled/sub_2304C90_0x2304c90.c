// Function: sub_2304C90
// Address: 0x2304c90
//
_QWORD *__fastcall sub_2304C90(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v8[8]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v9[12]; // [rsp+40h] [rbp-60h] BYREF

  sub_D902F0(v8, a2 + 8, a3, a4);
  v4 = v8;
  sub_D89C50((__int64)v9, v8);
  v5 = (_QWORD *)sub_22077B0(0x40u);
  v6 = v5;
  if ( v5 )
  {
    v4 = v9;
    *v5 = &unk_4A0B268;
    sub_D89C50((__int64)(v5 + 1), v9);
  }
  sub_D89DE0((__int64)v9, (__int64)v4);
  *a1 = v6;
  sub_D89DE0((__int64)v8, (__int64)v4);
  return a1;
}
