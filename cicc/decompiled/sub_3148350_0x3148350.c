// Function: sub_3148350
// Address: 0x3148350
//
void __fastcall sub_3148350(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 32;
  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 != v2 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
