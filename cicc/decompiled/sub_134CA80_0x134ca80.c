// Function: sub_134CA80
// Address: 0x134ca80
//
__int64 __fastcall sub_134CA80(_QWORD *a1)
{
  _QWORD *v1; // r13

  v1 = (_QWORD *)a1[20];
  *a1 = &unk_49E8128;
  if ( v1 )
  {
    sub_134CA00(v1);
    j_j___libc_free_0(v1, 96);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
