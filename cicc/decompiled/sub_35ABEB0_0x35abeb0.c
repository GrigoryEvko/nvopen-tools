// Function: sub_35ABEB0
// Address: 0x35abeb0
//
void __fastcall sub_35ABEB0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 35;
  v3 = a1[33];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[27];
  if ( (_QWORD *)v4 != a1 + 29 )
    _libc_free(v4);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
