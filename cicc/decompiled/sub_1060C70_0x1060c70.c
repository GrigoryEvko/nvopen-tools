// Function: sub_1060C70
// Address: 0x1060c70
//
__int64 __fastcall sub_1060C70(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49DB390;
  v2 = (_QWORD *)a1[1];
  if ( v2 != a1 + 3 )
    j_j___libc_free_0(v2, a1[3] + 1LL);
  return j_j___libc_free_0(a1, 48);
}
