// Function: sub_217DAC0
// Address: 0x217dac0
//
__int64 __fastcall sub_217DAC0(unsigned __int64 **a1)
{
  unsigned __int64 *v1; // r12
  __int64 result; // rax
  unsigned __int64 *v3; // r12
  unsigned __int64 *v4; // r12
  unsigned __int64 *v5; // r12

  v1 = *a1;
  if ( *a1 )
  {
    _libc_free(*v1);
    result = j_j___libc_free_0(v1, 24);
  }
  v3 = a1[1];
  if ( v3 )
  {
    _libc_free(*v3);
    result = j_j___libc_free_0(v3, 24);
  }
  v4 = a1[2];
  if ( v4 )
  {
    _libc_free(*v4);
    result = j_j___libc_free_0(v4, 24);
  }
  v5 = a1[3];
  if ( v5 )
  {
    _libc_free(*v5);
    return j_j___libc_free_0(v5, 24);
  }
  return result;
}
