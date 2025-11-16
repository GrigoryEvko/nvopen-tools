// Function: sub_CA1BE0
// Address: 0xca1be0
//
__int64 __fastcall sub_CA1BE0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdi

  result = (__int64)(a1 + 22);
  v4 = a1[19];
  if ( v4 != result )
    result = _libc_free(v4, a2);
  if ( (_QWORD *)*a1 != a1 + 3 )
    return _libc_free(*a1, a2);
  return result;
}
