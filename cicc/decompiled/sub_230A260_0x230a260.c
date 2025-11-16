// Function: sub_230A260
// Address: 0x230a260
//
void __fastcall sub_230A260(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A10438;
  v2 = a1[2];
  if ( (_QWORD *)v2 != a1 + 4 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
