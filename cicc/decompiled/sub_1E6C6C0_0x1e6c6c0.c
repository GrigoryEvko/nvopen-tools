// Function: sub_1E6C6C0
// Address: 0x1e6c6c0
//
__int64 (__fastcall **__fastcall sub_1E6C6C0(_QWORD *a1))()
{
  __int64 v1; // r8
  __int64 (__fastcall **result)(); // rax

  v1 = a1[5];
  result = off_49FCB18;
  *a1 = off_49FCB18;
  if ( v1 )
    return (__int64 (__fastcall **)())j_j___libc_free_0(v1, a1[7] - v1);
  return result;
}
