// Function: sub_AAD550
// Address: 0xaad550
//
__int64 __fastcall sub_AAD550(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax

  if ( *((_DWORD *)a1 + 2) > 0x40u )
  {
    v3 = *a1;
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  *a1 = *a2;
  result = *((unsigned int *)a2 + 2);
  *((_DWORD *)a1 + 2) = result;
  *((_DWORD *)a2 + 2) = 0;
  return result;
}
