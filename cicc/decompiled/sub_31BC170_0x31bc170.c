// Function: sub_31BC170
// Address: 0x31bc170
//
void __fastcall sub_31BC170(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 40;
  v3 = *(_QWORD *)(a1 + 24);
  if ( v3 != v2 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
