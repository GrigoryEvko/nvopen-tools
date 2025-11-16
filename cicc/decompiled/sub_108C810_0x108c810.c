// Function: sub_108C810
// Address: 0x108c810
//
__int64 (__fastcall **__fastcall sub_108C810(_QWORD *a1))()
{
  __int64 (__fastcall **result)(); // rax
  _QWORD *v2; // r12
  _QWORD *v3; // rdi

  result = off_497C0E0;
  v2 = (_QWORD *)a1[8];
  *a1 = off_497C0E0;
  if ( v2 )
  {
    v3 = (_QWORD *)v2[4];
    if ( v3 != v2 + 6 )
      j_j___libc_free_0(v3, v2[6] + 1LL);
    if ( (_QWORD *)*v2 != v2 + 2 )
      j_j___libc_free_0(*v2, v2[2] + 1LL);
    return (__int64 (__fastcall **)())j_j___libc_free_0(v2, 72);
  }
  return result;
}
