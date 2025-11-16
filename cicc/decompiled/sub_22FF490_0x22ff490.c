// Function: sub_22FF490
// Address: 0x22ff490
//
void __fastcall sub_22FF490(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 80;
  v3 = *(_QWORD *)(a1 + 64);
  if ( v3 != v2 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
