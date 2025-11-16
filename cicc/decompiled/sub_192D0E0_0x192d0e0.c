// Function: sub_192D0E0
// Address: 0x192d0e0
//
void __fastcall sub_192D0E0(_QWORD *a1)
{
  char *v2; // rax
  unsigned __int64 v3; // rdi

  v2 = (char *)(a1 + 8);
  v3 = a1[6];
  if ( (char *)v3 != v2 )
    _libc_free(v3);
  if ( (_QWORD *)*a1 != a1 + 2 )
    _libc_free(*a1);
}
