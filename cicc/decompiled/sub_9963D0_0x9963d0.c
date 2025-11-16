// Function: sub_9963D0
// Address: 0x9963d0
//
__int64 __fastcall sub_9963D0(__int64 a1)
{
  bool v2; // cc
  __int64 v3; // rdi
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 24) <= 0x40u;
  *(_BYTE *)(a1 + 32) = 0;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
      result = j_j___libc_free_0_0(v3);
  }
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    if ( *(_QWORD *)a1 )
      return j_j___libc_free_0_0(*(_QWORD *)a1);
  }
  return result;
}
