// Function: sub_CB5980
// Address: 0xcb5980
//
__int64 __fastcall sub_CB5980(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v7; // rdi
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 44) == 1 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    if ( v7 )
      result = j_j___libc_free_0_0(v7);
  }
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a2 + a3;
  *(_QWORD *)(a1 + 32) = a2;
  *(_DWORD *)(a1 + 44) = a4;
  return result;
}
