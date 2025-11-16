// Function: sub_D00D90
// Address: 0xd00d90
//
__int64 __fastcall sub_D00D90(__int64 *a1, __int64 *a2)
{
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdi

  if ( *((_BYTE *)a1 + 16) )
  {
    if ( *((_DWORD *)a1 + 2) > 0x40u )
    {
      v5 = *a1;
      if ( v5 )
        j_j___libc_free_0_0(v5);
    }
    *a1 = *a2;
    result = *((unsigned int *)a2 + 2);
    *((_DWORD *)a1 + 2) = result;
    *((_DWORD *)a2 + 2) = 0;
  }
  else
  {
    v3 = *((_DWORD *)a2 + 2);
    *((_DWORD *)a2 + 2) = 0;
    *((_DWORD *)a1 + 2) = v3;
    result = *a2;
    *((_BYTE *)a1 + 16) = 1;
    *a1 = result;
  }
  return result;
}
