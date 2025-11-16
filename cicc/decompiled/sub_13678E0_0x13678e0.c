// Function: sub_13678E0
// Address: 0x13678e0
//
void __fastcall sub_13678E0(_QWORD *a1)
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
