// Function: sub_325F550
// Address: 0x325f550
//
__int64 __fastcall sub_325F550(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    v3 = *(_QWORD *)a1;
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  *(_QWORD *)a1 = *(_QWORD *)a2;
  result = *(unsigned int *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = result;
  *(_DWORD *)(a2 + 8) = 0;
  return result;
}
