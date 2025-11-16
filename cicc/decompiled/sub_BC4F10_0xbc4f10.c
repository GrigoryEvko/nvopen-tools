// Function: sub_BC4F10
// Address: 0xbc4f10
//
__int64 __fastcall sub_BC4F10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi

  v3 = a1 + 32;
  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 != v3 )
    _libc_free(v4, a2);
  return j_j___libc_free_0(a1, 416);
}
