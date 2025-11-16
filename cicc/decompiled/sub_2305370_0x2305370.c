// Function: sub_2305370
// Address: 0x2305370
//
_QWORD *__fastcall sub_2305370(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // r13
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_10431D0(v9, a2 + 8, a3, a4);
  v4 = v9[0];
  v9[0] = 0;
  v5 = (_QWORD *)sub_22077B0(0x10u);
  v6 = v5;
  if ( v5 )
  {
    v5[1] = v4;
    *v5 = &unk_4A0AF20;
  }
  else if ( v4 )
  {
    sub_103C970(v4);
    j_j___libc_free_0(v4);
  }
  v7 = v9[0];
  *a1 = v6;
  if ( v7 )
  {
    sub_103C970(v7);
    j_j___libc_free_0(v7);
  }
  return a1;
}
