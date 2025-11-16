// Function: sub_3700D00
// Address: 0x3700d00
//
void __fastcall sub_3700D00(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_49DC7F0;
  v2 = a1[1];
  if ( (_QWORD *)v2 != a1 + 3 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
