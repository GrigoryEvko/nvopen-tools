// Function: sub_3112140
// Address: 0x3112140
//
void *__fastcall sub_3112140(__int64 a1)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = (unsigned __int64)v2;
    v2 = (_QWORD *)*v2;
    v4 = *(_QWORD *)(v3 + 16);
    if ( v4 )
    {
      sub_3112140(v4 + 16);
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 != v4 + 64 )
        j_j___libc_free_0(v5);
      j_j___libc_free_0(v4);
    }
    j_j___libc_free_0(v3);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
