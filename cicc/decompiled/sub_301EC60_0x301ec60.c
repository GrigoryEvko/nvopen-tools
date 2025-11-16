// Function: sub_301EC60
// Address: 0x301ec60
//
void __fastcall sub_301EC60(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = &unk_49E41D0;
  v2 = a1[34];
  if ( (_QWORD *)v2 != a1 + 36 )
    j_j___libc_free_0(v2);
  v3 = a1[12];
  if ( (_QWORD *)v3 != a1 + 14 )
    j_j___libc_free_0(v3);
  v4 = a1[8];
  if ( (_QWORD *)v4 != a1 + 10 )
    j_j___libc_free_0(v4);
  v5 = a1[1];
  if ( (_QWORD *)v5 != a1 + 3 )
    j_j___libc_free_0(v5);
  j_j___libc_free_0((unsigned __int64)a1);
}
