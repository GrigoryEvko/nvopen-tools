// Function: sub_1A1B230
// Address: 0x1a1b230
//
unsigned __int64 __fastcall sub_1A1B230(__int64 *a1)
{
  __int64 *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 *v5; // rdi
  unsigned __int64 result; // rax

  v2 = a1 + 19;
  v3 = a1[17];
  if ( (__int64 *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[13];
  if ( (__int64 *)v4 != a1 + 15 )
    _libc_free(v4);
  v5 = (__int64 *)a1[8];
  result = (unsigned __int64)(a1 + 10);
  if ( v5 != a1 + 10 )
    result = j_j___libc_free_0(v5, a1[10] + 1);
  if ( *a1 )
    return sub_161E7C0((__int64)a1, *a1);
  return result;
}
