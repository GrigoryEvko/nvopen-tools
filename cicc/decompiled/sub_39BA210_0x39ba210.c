// Function: sub_39BA210
// Address: 0x39ba210
//
void __fastcall sub_39BA210(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_49EE580;
  v2 = a1[8];
  if ( (_QWORD *)v2 != a1 + 10 )
    j_j___libc_free_0(v2);
  v3 = a1[1];
  if ( (_QWORD *)v3 != a1 + 3 )
    j_j___libc_free_0(v3);
}
