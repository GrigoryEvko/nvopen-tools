// Function: sub_2305410
// Address: 0x2305410
//
_QWORD *__fastcall sub_2305410(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r13
  unsigned __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_2E82A00(v7, a2 + 8);
  v2 = v7[0];
  v7[0] = 0;
  v3 = (_QWORD *)sub_22077B0(0x10u);
  v4 = v3;
  if ( v3 )
  {
    v3[1] = v2;
    *v3 = &unk_4A0B010;
  }
  else if ( v2 )
  {
    sub_2E81F20(v2);
    j_j___libc_free_0(v2);
  }
  v5 = v7[0];
  *a1 = v4;
  if ( v5 )
  {
    sub_2E81F20(v5);
    j_j___libc_free_0(v5);
  }
  return a1;
}
