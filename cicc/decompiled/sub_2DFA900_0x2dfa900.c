// Function: sub_2DFA900
// Address: 0x2dfa900
//
void *__fastcall sub_2DFA900(__int64 a1)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = (unsigned __int64)v2;
    v2 = (_QWORD *)*v2;
    v4 = *(_QWORD *)(v3 + 96);
    if ( v4 != v3 + 112 )
      _libc_free(v4);
    v5 = *(_QWORD *)(v3 + 48);
    if ( v5 != v3 + 64 )
      _libc_free(v5);
    j_j___libc_free_0(v3);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
