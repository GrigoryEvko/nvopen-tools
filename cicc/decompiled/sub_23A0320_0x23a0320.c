// Function: sub_23A0320
// Address: 0x23a0320
//
void __fastcall sub_23A0320(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A15CD0;
  v2 = a1[1];
  if ( (_QWORD *)v2 != a1 + 3 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
