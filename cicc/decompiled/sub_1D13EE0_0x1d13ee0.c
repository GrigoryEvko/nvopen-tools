// Function: sub_1D13EE0
// Address: 0x1d13ee0
//
__int64 __fastcall sub_1D13EE0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 result; // rax

  if ( *((_DWORD *)a1 + 2) > 0x40u )
  {
    v3 = *a1;
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  *a1 = *a2;
  *((_DWORD *)a1 + 2) = *((_DWORD *)a2 + 2);
  *((_DWORD *)a2 + 2) = 0;
  if ( *((_DWORD *)a1 + 6) > 0x40u )
  {
    v4 = a1[2];
    if ( v4 )
      j_j___libc_free_0_0(v4);
  }
  a1[2] = a2[2];
  result = *((unsigned int *)a2 + 6);
  *((_DWORD *)a1 + 6) = result;
  *((_DWORD *)a2 + 6) = 0;
  return result;
}
