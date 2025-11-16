// Function: sub_E90070
// Address: 0xe90070
//
void *__fastcall sub_E90070(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  __int64 v5; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = (_QWORD *)v3[3];
    if ( v4 )
    {
      v5 = v4[7];
      if ( v5 )
        j_j___libc_free_0(v5, v4[9] - v5);
      sub_E90070(v4);
      if ( (_QWORD *)*v4 != v4 + 6 )
        j_j___libc_free_0(*v4, 8LL * v4[1]);
      j_j___libc_free_0(v4, 96);
    }
    j_j___libc_free_0(v3, 40);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
