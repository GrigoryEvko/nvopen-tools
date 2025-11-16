// Function: sub_1D02630
// Address: 0x1d02630
//
__int64 (__fastcall **__fastcall sub_1D02630(_QWORD *a1))()
{
  __int64 (__fastcall **result)(); // rax
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi

  result = &off_49F9520;
  *a1 = &off_49F9520;
  v3 = a1[18];
  if ( v3 )
    result = (__int64 (__fastcall **)())j_j___libc_free_0(v3, a1[20] - v3);
  v4 = a1[15];
  if ( v4 )
    result = (__int64 (__fastcall **)())j_j___libc_free_0(v4, a1[17] - v4);
  v5 = a1[12];
  if ( v5 )
    result = (__int64 (__fastcall **)())j_j___libc_free_0(v5, a1[14] - v5);
  v6 = a1[2];
  if ( v6 )
    return (__int64 (__fastcall **)())j_j___libc_free_0(v6, a1[4] - v6);
  return result;
}
