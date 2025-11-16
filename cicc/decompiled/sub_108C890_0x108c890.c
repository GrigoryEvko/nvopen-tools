// Function: sub_108C890
// Address: 0x108c890
//
__int64 __fastcall sub_108C890(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v3; // rdi

  v1 = (_QWORD *)a1[8];
  *a1 = off_497C0E0;
  if ( v1 )
  {
    v3 = (_QWORD *)v1[4];
    if ( v3 != v1 + 6 )
      j_j___libc_free_0(v3, v1[6] + 1LL);
    if ( (_QWORD *)*v1 != v1 + 2 )
      j_j___libc_free_0(*v1, v1[2] + 1LL);
    j_j___libc_free_0(v1, 72);
  }
  return j_j___libc_free_0(a1, 72);
}
