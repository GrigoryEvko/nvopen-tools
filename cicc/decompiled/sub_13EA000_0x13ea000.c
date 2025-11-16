// Function: sub_13EA000
// Address: 0x13ea000
//
__int64 __fastcall sub_13EA000(__int64 a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  __int64 v4; // rdi

  if ( *(_DWORD *)a1 == 3 )
  {
    if ( *(_DWORD *)(a1 + 32) > 0x40u )
    {
      v2 = *(_QWORD *)(a1 + 24);
      if ( v2 )
        result = j_j___libc_free_0_0(v2);
    }
    if ( *(_DWORD *)(a1 + 16) > 0x40u )
    {
      v4 = *(_QWORD *)(a1 + 8);
      if ( v4 )
        return j_j___libc_free_0_0(v4);
    }
  }
  return result;
}
