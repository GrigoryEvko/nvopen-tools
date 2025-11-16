// Function: sub_1F44C40
// Address: 0x1f44c40
//
void __fastcall sub_1F44C40(_QWORD *a1)
{
  char *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = (char *)(a1 + 27);
  v3 = a1[25];
  if ( (char *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
}
