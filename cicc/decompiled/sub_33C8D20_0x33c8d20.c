// Function: sub_33C8D20
// Address: 0x33c8d20
//
__int64 __fastcall sub_33C8D20(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    v3 = *(_QWORD *)a1;
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v4 = *(_QWORD *)(a1 + 16);
    if ( v4 )
      j_j___libc_free_0_0(v4);
  }
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  result = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = result;
  *(_DWORD *)(a2 + 24) = 0;
  return result;
}
