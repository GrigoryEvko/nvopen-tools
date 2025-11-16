// Function: sub_108C040
// Address: 0x108c040
//
__int64 (__fastcall **__fastcall sub_108C040(_QWORD *a1, __int64 a2))()
{
  __int64 (__fastcall **result)(); // rax
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rdi

  result = off_497C080;
  v3 = a1[8];
  *a1 = off_497C080;
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 64);
    if ( v4 != v3 + 80 )
      _libc_free(v4, a2);
    v5 = *(_QWORD *)(v3 + 32);
    if ( v5 != v3 + 48 )
      _libc_free(v5, a2);
    return (__int64 (__fastcall **)())j_j___libc_free_0(v3, 96);
  }
  return result;
}
