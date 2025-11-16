// Function: sub_23044D0
// Address: 0x23044d0
//
_QWORD *__fastcall sub_23044D0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_DFE9F0((__int64)&v6, a2 + 8, a3);
  sub_DF93A0(v7, &v6);
  v3 = (_QWORD *)sub_22077B0(0x10u);
  v4 = v3;
  if ( v3 )
  {
    *v3 = &unk_4A0AD68;
    sub_DF93A0(v3 + 1, v7);
  }
  sub_DFE7B0(v7);
  *a1 = v4;
  sub_DFE7B0(&v6);
  return a1;
}
