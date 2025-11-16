// Function: sub_1D7EF90
// Address: 0x1d7ef90
//
void *__fastcall sub_1D7EF90(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 *v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // rdi

  *a1 = off_49F9F28;
  _libc_free(a1[45]);
  _libc_free(a1[42]);
  v2 = a1[32];
  if ( v2 )
  {
    v3 = (__int64 *)a1[37];
    v4 = a1[41] + 8LL;
    if ( v4 > (unsigned __int64)v3 )
    {
      do
      {
        v5 = *v3++;
        j_j___libc_free_0(v5, 512);
      }
      while ( v4 > (unsigned __int64)v3 );
      v2 = a1[32];
    }
    j_j___libc_free_0(v2, 8LL * a1[33]);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
