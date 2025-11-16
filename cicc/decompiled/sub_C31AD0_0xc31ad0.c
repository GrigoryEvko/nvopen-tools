// Function: sub_C31AD0
// Address: 0xc31ad0
//
void *__fastcall sub_C31AD0(__int64 a1, __int64 a2)
{
  void *result; // rax
  bool v4; // zf
  _QWORD *v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *i; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // r12
  __int64 *v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rdi

  *(_QWORD *)a1 = &unk_49DBE40;
  sub_CB0A00();
  result = &unk_49DBE10;
  v4 = *(_BYTE *)(a1 + 160) == 0;
  *(_QWORD *)a1 = &unk_49DBE10;
  if ( !v4 )
  {
    v5 = *(_QWORD **)(a1 + 72);
    v6 = *(unsigned int *)(a1 + 80);
    *(_BYTE *)(a1 + 160) = 0;
    v7 = &v5[v6];
    if ( v5 != v7 )
    {
      for ( i = v5; ; i = *(_QWORD **)(a1 + 72) )
      {
        v9 = *v5;
        v10 = (unsigned int)(v5 - i) >> 7;
        a2 = 4096LL << v10;
        if ( v10 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v5;
        sub_C7D6A0(v9, a2, 16);
        if ( v7 == v5 )
          break;
      }
    }
    v11 = *(__int64 **)(a1 + 120);
    v12 = &v11[2 * *(unsigned int *)(a1 + 128)];
    if ( v11 != v12 )
    {
      do
      {
        a2 = v11[1];
        v13 = *v11;
        v11 += 2;
        sub_C7D6A0(v13, a2, 16);
      }
      while ( v12 != v11 );
      v12 = *(__int64 **)(a1 + 120);
    }
    if ( v12 != (__int64 *)(a1 + 136) )
      _libc_free(v12, a2);
    v14 = *(_QWORD *)(a1 + 72);
    if ( v14 != a1 + 88 )
      _libc_free(v14, a2);
    return (void *)_libc_free(*(_QWORD *)(a1 + 32), a2);
  }
  return result;
}
