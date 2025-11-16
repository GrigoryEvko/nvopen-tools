// Function: sub_2D22970
// Address: 0x2d22970
//
void __fastcall sub_2D22970(unsigned __int64 *a1)
{
  unsigned __int64 *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1 + 27;
  v3 = a1[25];
  if ( (unsigned __int64 *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[17];
  if ( (unsigned __int64 *)v4 != a1 + 19 )
    _libc_free(v4);
  v5 = a1[9];
  if ( (unsigned __int64 *)v5 != a1 + 11 )
    _libc_free(v5);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    _libc_free(*a1);
}
