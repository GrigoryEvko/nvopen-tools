// Function: sub_CF4C80
// Address: 0xcf4c80
//
__int64 __fastcall sub_CF4C80(_QWORD *a1)
{
  _QWORD *v1; // r13

  v1 = (_QWORD *)a1[22];
  *a1 = &unk_49DD888;
  if ( v1 )
  {
    sub_CF4BF0(v1);
    j_j___libc_free_0(v1, 56);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
