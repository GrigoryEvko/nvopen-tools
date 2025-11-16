// Function: sub_2305600
// Address: 0x2305600
//
_QWORD *__fastcall sub_2305600(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // rsi
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_105D2A0(&v9, a2 + 8, a3, a4);
  v4 = v9;
  v9 = 0;
  v10[0] = v4;
  v5 = (_QWORD *)sub_22077B0(0x10u);
  v6 = v5;
  if ( v5 )
  {
    *v5 = &unk_4A0AD40;
    v5[1] = v10[0];
  }
  else if ( v10[0] )
  {
    sub_10568E0((__int64)v10, v10[0]);
  }
  v7 = v9;
  *a1 = v6;
  if ( v7 )
    sub_10568E0((__int64)&v9, v7);
  return a1;
}
