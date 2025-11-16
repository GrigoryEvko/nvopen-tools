// Function: sub_16B0D50
// Address: 0x16b0d50
//
__int64 __fastcall sub_16B0D50(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = a1[32];
    if ( v2 != a1[31] )
      _libc_free(v2);
    v3 = a1[11];
    if ( v3 != a1[10] )
      _libc_free(v3);
    v4 = a1[6];
    if ( v4 )
      j_j___libc_free_0(v4, a1[8] - v4);
    if ( (_QWORD *)*a1 != a1 + 2 )
      j_j___libc_free_0(*a1, a1[2] + 1LL);
    return j_j___libc_free_0(a1, 320);
  }
  return result;
}
