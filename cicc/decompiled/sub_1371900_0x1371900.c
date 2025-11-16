// Function: sub_1371900
// Address: 0x1371900
//
__int64 __fastcall sub_1371900(_QWORD **a1)
{
  _QWORD *i; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 result; // rax

  for ( i = *a1; i != a1; result = j_j___libc_free_0(v3, 192) )
  {
    v3 = i;
    i = (_QWORD *)*i;
    v4 = v3[18];
    if ( (_QWORD *)v4 != v3 + 20 )
      _libc_free(v4);
    v5 = v3[14];
    if ( (_QWORD *)v5 != v3 + 16 )
      _libc_free(v5);
    v6 = v3[4];
    if ( (_QWORD *)v6 != v3 + 6 )
      _libc_free(v6);
  }
  return result;
}
