// Function: sub_1E72310
// Address: 0x1e72310
//
__int64 __fastcall sub_1E72310(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  _QWORD *v6; // rdi
  __int64 v7; // rdi
  _QWORD *v8; // rdi
  __int64 result; // rax

  v2 = a1[19];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[36];
  if ( (_QWORD *)v3 != a1 + 38 )
    _libc_free(v3);
  v4 = a1[24];
  if ( (_QWORD *)v4 != a1 + 26 )
    _libc_free(v4);
  v5 = a1[16];
  if ( v5 )
    j_j___libc_free_0(v5, a1[18] - v5);
  v6 = (_QWORD *)a1[12];
  if ( v6 != a1 + 14 )
    j_j___libc_free_0(v6, a1[14] + 1LL);
  v7 = a1[8];
  if ( v7 )
    j_j___libc_free_0(v7, a1[10] - v7);
  v8 = (_QWORD *)a1[4];
  result = (__int64)(a1 + 6);
  if ( v8 != a1 + 6 )
    return j_j___libc_free_0(v8, a1[6] + 1LL);
  return result;
}
