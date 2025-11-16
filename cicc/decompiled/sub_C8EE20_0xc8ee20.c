// Function: sub_C8EE20
// Address: 0xc8ee20
//
__int64 __fastcall sub_C8EE20(__int64 *a1)
{
  _QWORD *v1; // r12
  __int64 result; // rax
  __int64 v4; // rdi

  v1 = (_QWORD *)a1[1];
  if ( v1 )
  {
    if ( *v1 )
      j_j___libc_free_0(*v1, v1[2] - *v1);
    result = j_j___libc_free_0(v1, 24);
    a1[1] = 0;
  }
  v4 = *a1;
  if ( *a1 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  return result;
}
