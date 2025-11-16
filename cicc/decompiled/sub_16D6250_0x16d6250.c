// Function: sub_16D6250
// Address: 0x16d6250
//
__int64 __fastcall sub_16D6250(_QWORD *a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = a1 + 2;
    v3 = (_QWORD *)*a1;
    if ( v3 != v2 )
      j_j___libc_free_0(v3, a1[2] + 1LL);
    return j_j___libc_free_0(a1, 32);
  }
  return result;
}
