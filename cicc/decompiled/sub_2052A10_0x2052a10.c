// Function: sub_2052A10
// Address: 0x2052a10
//
void __fastcall sub_2052A10(unsigned __int64 *a1)
{
  unsigned __int64 *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1 + 19;
  v3 = a1[17];
  if ( (unsigned __int64 *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[13];
  if ( (unsigned __int64 *)v4 != a1 + 15 )
    _libc_free(v4);
  v5 = a1[10];
  if ( (unsigned __int64 *)v5 != a1 + 12 )
    _libc_free(v5);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    _libc_free(*a1);
}
