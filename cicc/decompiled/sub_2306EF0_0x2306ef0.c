// Function: sub_2306EF0
// Address: 0x2306ef0
//
void __fastcall sub_2306EF0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A11A38;
  v2 = a1[1];
  if ( v2 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
