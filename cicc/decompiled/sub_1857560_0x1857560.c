// Function: sub_1857560
// Address: 0x1857560
//
void *__fastcall sub_1857560(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = v3[4];
    if ( v4 != v3[3] )
      _libc_free(v4);
    j_j___libc_free_0(v3, 120);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
