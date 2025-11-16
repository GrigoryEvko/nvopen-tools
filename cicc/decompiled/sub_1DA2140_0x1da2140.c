// Function: sub_1DA2140
// Address: 0x1da2140
//
void *__fastcall sub_1DA2140(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = v3[12];
    if ( (_QWORD *)v4 != v3 + 14 )
      _libc_free(v4);
    v5 = v3[6];
    if ( (_QWORD *)v5 != v3 + 8 )
      _libc_free(v5);
    j_j___libc_free_0(v3, 200);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
