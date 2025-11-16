// Function: sub_CA4290
// Address: 0xca4290
//
__int64 __fastcall sub_CA4290(_QWORD *a1)
{
  _QWORD *v1; // r8
  __int64 result; // rax

  v1 = (_QWORD *)a1[1];
  *a1 = &unk_49DCC30;
  result = (__int64)(a1 + 3);
  if ( v1 != a1 + 3 )
    return j_j___libc_free_0(v1, a1[3] + 1LL);
  return result;
}
