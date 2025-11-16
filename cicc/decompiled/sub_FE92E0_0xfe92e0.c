// Function: sub_FE92E0
// Address: 0xfe92e0
//
__int64 __fastcall sub_FE92E0(_QWORD **a1, __int64 a2)
{
  _QWORD *i; // rbx
  _QWORD *v4; // r12
  _QWORD *v5; // rdi
  _QWORD *v6; // rdi
  _QWORD *v7; // rdi
  __int64 result; // rax

  for ( i = *a1; i != a1; result = j_j___libc_free_0(v4, 192) )
  {
    v4 = i;
    i = (_QWORD *)*i;
    v5 = (_QWORD *)v4[18];
    if ( v5 != v4 + 20 )
      _libc_free(v5, a2);
    v6 = (_QWORD *)v4[14];
    if ( v6 != v4 + 16 )
      _libc_free(v6, a2);
    v7 = (_QWORD *)v4[4];
    if ( v7 != v4 + 6 )
      _libc_free(v7, a2);
    a2 = 192;
  }
  return result;
}
