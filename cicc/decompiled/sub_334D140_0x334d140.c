// Function: sub_334D140
// Address: 0x334d140
//
void __fastcall sub_334D140(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = off_4A36128;
  v2 = a1[101];
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = a1[98];
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = a1[79];
  if ( (_QWORD *)v4 != a1 + 81 )
    _libc_free(v4);
  v5 = a1[76];
  *a1 = &unk_4A365B8;
  if ( v5 )
    j_j___libc_free_0(v5);
  sub_2F8EAD0(a1);
}
