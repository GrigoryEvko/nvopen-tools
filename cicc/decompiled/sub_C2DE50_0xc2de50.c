// Function: sub_C2DE50
// Address: 0xc2de50
//
__int64 __fastcall sub_C2DE50(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  __int64 result; // rax

  *a1 = &unk_49DBDA0;
  v2 = (_QWORD *)a1[6];
  if ( v2 != a1 + 8 )
    j_j___libc_free_0(v2, a1[8] + 1LL);
  v3 = (_QWORD *)a1[1];
  result = (__int64)(a1 + 3);
  if ( v3 != a1 + 3 )
    return j_j___libc_free_0(v3, a1[3] + 1LL);
  return result;
}
