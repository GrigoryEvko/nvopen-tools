// Function: sub_2461EC0
// Address: 0x2461ec0
//
void __fastcall sub_2461EC0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != v2 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
