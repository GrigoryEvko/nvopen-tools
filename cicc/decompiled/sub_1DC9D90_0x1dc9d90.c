// Function: sub_1DC9D90
// Address: 0x1dc9d90
//
void *__fastcall sub_1DC9D90(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    sub_1DB4CE0((__int64)(v3 + 2));
    v4 = v3[14];
    if ( v4 )
    {
      v5 = *(_QWORD *)(v4 + 16);
      while ( v5 )
      {
        sub_1DC95B0(*(_QWORD *)(v5 + 24));
        v6 = v5;
        v5 = *(_QWORD *)(v5 + 16);
        j_j___libc_free_0(v6, 56);
      }
      j_j___libc_free_0(v4, 48);
    }
    v7 = v3[10];
    if ( (_QWORD *)v7 != v3 + 12 )
      _libc_free(v7);
    v8 = v3[2];
    if ( (_QWORD *)v8 != v3 + 4 )
      _libc_free(v8);
    j_j___libc_free_0(v3, 136);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
