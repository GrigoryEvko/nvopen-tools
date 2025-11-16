// Function: sub_12F3D00
// Address: 0x12f3d00
//
__int64 __fastcall sub_12F3D00(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  v3 = *(_QWORD *)(a1 + 96);
  if ( v3 != *(_QWORD *)(a1 + 88) )
    _libc_free(v3, a2);
  return j_j___libc_free_0(a1, 200);
}
