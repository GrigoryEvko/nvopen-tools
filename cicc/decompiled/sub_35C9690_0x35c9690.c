// Function: sub_35C9690
// Address: 0x35c9690
//
void __fastcall sub_35C9690(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_4A3A778;
  v2 = a1[9];
  if ( v2 )
    j_j___libc_free_0_0(v2);
  v3 = a1[6];
  if ( v3 )
    j_j___libc_free_0_0(v3);
  nullsub_1665();
  j_j___libc_free_0((unsigned __int64)a1);
}
