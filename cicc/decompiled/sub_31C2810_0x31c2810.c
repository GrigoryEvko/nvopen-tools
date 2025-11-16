// Function: sub_31C2810
// Address: 0x31c2810
//
void __fastcall sub_31C2810(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 88;
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v4 != a1 + 24 )
    _libc_free(v4);
  j_j___libc_free_0(a1);
}
