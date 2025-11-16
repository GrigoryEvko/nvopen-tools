// Function: sub_38BBD10
// Address: 0x38bbd10
//
void __fastcall sub_38BBD10(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 28;
  v3 = a1[26];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
  j_j___libc_free_0((unsigned __int64)a1);
}
