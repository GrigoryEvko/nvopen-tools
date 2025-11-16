// Function: sub_1353800
// Address: 0x1353800
//
__int64 __fastcall sub_1353800(_QWORD *a1, const char *a2, __int64 a3)
{
  _QWORD *v3; // r13

  v3 = (_QWORD *)a1[20];
  *a1 = &unk_49E8550;
  if ( v3 )
  {
    sub_13525A0(v3, a2, a3);
    j_j___libc_free_0(v3, 104);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
