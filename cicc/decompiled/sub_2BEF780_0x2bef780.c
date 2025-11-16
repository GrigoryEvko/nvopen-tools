// Function: sub_2BEF780
// Address: 0x2bef780
//
void __fastcall sub_2BEF780(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *a1 = &unk_4A23970;
  v2 = a1[10];
  if ( (_QWORD *)v2 != a1 + 12 )
    _libc_free(v2);
  v3 = a1[7];
  if ( (_QWORD *)v3 != a1 + 9 )
    _libc_free(v3);
  v4 = a1[2];
  if ( (_QWORD *)v4 != a1 + 4 )
    j_j___libc_free_0(v4);
  j_j___libc_free_0((unsigned __int64)a1);
}
