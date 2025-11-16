// Function: sub_C12480
// Address: 0xc12480
//
__int64 __fastcall sub_C12480(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi
  __int64 result; // rax

  *a1 = &unk_49E41D0;
  v2 = (_QWORD *)a1[34];
  if ( v2 != a1 + 36 )
    j_j___libc_free_0(v2, a1[36] + 1LL);
  v3 = (_QWORD *)a1[12];
  if ( v3 != a1 + 14 )
    j_j___libc_free_0(v3, a1[14] + 1LL);
  v4 = (_QWORD *)a1[8];
  if ( v4 != a1 + 10 )
    j_j___libc_free_0(v4, a1[10] + 1LL);
  v5 = (_QWORD *)a1[1];
  result = (__int64)(a1 + 3);
  if ( v5 != a1 + 3 )
    return j_j___libc_free_0(v5, a1[3] + 1LL);
  return result;
}
