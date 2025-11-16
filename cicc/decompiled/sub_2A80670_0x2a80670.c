// Function: sub_2A80670
// Address: 0x2a80670
//
void __fastcall sub_2A80670(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = off_49D3E30;
  v2 = a1[6];
  if ( (_QWORD *)v2 != a1 + 8 )
    j_j___libc_free_0(v2);
  v3 = a1[2];
  if ( (_QWORD *)v3 != a1 + 4 )
    j_j___libc_free_0(v3);
  j_j___libc_free_0((unsigned __int64)a1);
}
