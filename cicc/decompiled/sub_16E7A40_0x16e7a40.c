// Function: sub_16E7A40
// Address: 0x16e7a40
//
__int64 __fastcall sub_16E7A40(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v7; // rdi
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 32) == 1 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( v7 )
      result = j_j___libc_free_0_0(v7);
  }
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a2 + a3;
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 32) = a4;
  return result;
}
