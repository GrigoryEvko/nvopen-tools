// Function: sub_1E6C000
// Address: 0x1e6c000
//
void *__fastcall sub_1E6C000(_QWORD *a1)
{
  __int64 *v1; // r12
  void *result; // rax
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  __int64 i; // rbx
  __int64 v7; // rdi

  v1 = (__int64 *)a1[7];
  result = &unk_49FC458;
  *a1 = &unk_49FC458;
  if ( v1 )
  {
    v3 = v1[11];
    if ( v3 )
      j_j___libc_free_0_0(v3);
    _libc_free(v1[8]);
    v4 = v1[5];
    if ( (__int64 *)v4 != v1 + 7 )
      _libc_free(v4);
    v5 = *v1;
    if ( *v1 )
    {
      for ( i = v5 + 24LL * *(_QWORD *)(v5 - 8); v5 != i; i -= 24 )
      {
        v7 = *(_QWORD *)(i - 8);
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
      j_j_j___libc_free_0_0(v5 - 8);
    }
    return (void *)j_j___libc_free_0(v1, 96);
  }
  return result;
}
