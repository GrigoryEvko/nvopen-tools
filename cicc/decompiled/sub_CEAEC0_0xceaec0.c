// Function: sub_CEAEC0
// Address: 0xceaec0
//
int sub_CEAEC0()
{
  _QWORD *v0; // rax
  _QWORD *v1; // r12

  if ( !qword_4F85270 )
    sub_C7D570(&qword_4F85270, (__int64 (*)(void))sub_CEAC80, (__int64)sub_CEAA20);
  v0 = sub_C94E20(qword_4F85270);
  v1 = v0;
  if ( v0 )
  {
    if ( (_QWORD *)*v0 != v0 + 2 )
      j_j___libc_free_0(*v0, v0[2] + 1LL);
    j_j___libc_free_0(v1, 32);
    if ( !qword_4F85270 )
      sub_C7D570(&qword_4F85270, (__int64 (*)(void))sub_CEAC80, (__int64)sub_CEAA20);
    LODWORD(v0) = sub_C94E10(qword_4F85270, 0);
  }
  return (int)v0;
}
