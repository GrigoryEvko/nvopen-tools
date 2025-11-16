// Function: sub_1B681D0
// Address: 0x1b681d0
//
__int64 __fastcall sub_1B681D0(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  __int64 result; // rax

  *a1 = off_49853D0;
  v2 = (_QWORD *)a1[6];
  if ( v2 != a1 + 8 )
    j_j___libc_free_0(v2, a1[8] + 1LL);
  v3 = (_QWORD *)a1[2];
  result = (__int64)(a1 + 4);
  if ( v3 != a1 + 4 )
    return j_j___libc_free_0(v3, a1[4] + 1LL);
  return result;
}
