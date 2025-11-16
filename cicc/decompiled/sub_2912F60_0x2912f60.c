// Function: sub_2912F60
// Address: 0x2912f60
//
void __fastcall sub_2912F60(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_49D3D08;
  v2 = a1[1];
  if ( (_QWORD *)v2 != a1 + 3 )
    j_j___libc_free_0(v2);
  nullsub_61();
  j_j___libc_free_0((unsigned __int64)a1);
}
