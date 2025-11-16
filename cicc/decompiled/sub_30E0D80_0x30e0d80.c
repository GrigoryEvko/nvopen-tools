// Function: sub_30E0D80
// Address: 0x30e0d80
//
void __fastcall sub_30E0D80(unsigned __int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 32;
  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 != v2 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
