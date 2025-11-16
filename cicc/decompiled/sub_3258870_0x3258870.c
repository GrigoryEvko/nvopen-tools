// Function: sub_3258870
// Address: 0x3258870
//
void __fastcall sub_3258870(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A36020;
  v2 = a1[6];
  if ( v2 )
    j_j___libc_free_0(v2);
  sub_3252AD0(a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
