// Function: sub_14A9CA0
// Address: 0x14a9ca0
//
__int64 *__fastcall sub_14A9CA0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdi

  if ( *((_DWORD *)a1 + 2) > 0x40u )
  {
    v3 = *a1;
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  *a1 = *a2;
  *((_DWORD *)a1 + 2) = *((_DWORD *)a2 + 2);
  *((_DWORD *)a2 + 2) = 0;
  return a1;
}
