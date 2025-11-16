// Function: sub_230A300
// Address: 0x230a300
//
void __fastcall sub_230A300(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A155B8;
  v2 = a1[1];
  if ( (_QWORD *)v2 != a1 + 3 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
