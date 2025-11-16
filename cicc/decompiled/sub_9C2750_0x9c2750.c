// Function: sub_9C2750
// Address: 0x9c2750
//
__int64 __fastcall sub_9C2750(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi

  v3 = a1 + 56;
  v4 = *(_QWORD *)(a1 + 40);
  if ( v4 != v3 )
    _libc_free(v4, a2);
  return j_j___libc_free_0(a1, 56);
}
