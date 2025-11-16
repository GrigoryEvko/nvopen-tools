// Function: sub_12C7330
// Address: 0x12c7330
//
__int64 __fastcall sub_12C7330(_QWORD *a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  __int64 result; // rax

  v2 = a1 + 10;
  v3 = (_QWORD *)a1[8];
  if ( v3 != v2 )
    j_j___libc_free_0(v3, a1[10] + 1LL);
  v4 = (_QWORD *)a1[4];
  if ( v4 != a1 + 6 )
    j_j___libc_free_0(v4, a1[6] + 1LL);
  result = (__int64)(a1 + 2);
  if ( (_QWORD *)*a1 != a1 + 2 )
    return j_j___libc_free_0(*a1, a1[2] + 1LL);
  return result;
}
