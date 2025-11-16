// Function: sub_108C910
// Address: 0x108c910
//
__int64 __fastcall sub_108C910(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // r12
  _QWORD *v3; // rdi

  result = 4294967293LL;
  v2 = *(_QWORD **)(a1 + 64);
  *(_WORD *)(a1 + 56) = -3;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  if ( v2 )
  {
    v3 = (_QWORD *)v2[4];
    if ( v3 != v2 + 6 )
      j_j___libc_free_0(v3, v2[6] + 1LL);
    if ( (_QWORD *)*v2 != v2 + 2 )
      j_j___libc_free_0(*v2, v2[2] + 1LL);
    return j_j___libc_free_0(v2, 72);
  }
  return result;
}
