// Function: sub_EFE3D0
// Address: 0xefe3d0
//
void *__fastcall sub_EFE3D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  void *result; // rax
  bool v9; // zf
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 *i; // rax
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // r12
  __int64 *v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rdi

  v7 = a1 + 176;
  *(_QWORD *)(v7 - 176) = &unk_49E4EB8;
  sub_EFDF60(v7, (unsigned __int8 *)a2, a3, a4, a5, a6);
  result = &unk_49DBE10;
  v9 = *(_BYTE *)(a1 + 160) == 0;
  *(_QWORD *)a1 = &unk_49DBE10;
  if ( !v9 )
  {
    v10 = *(__int64 **)(a1 + 72);
    v11 = *(unsigned int *)(a1 + 80);
    *(_BYTE *)(a1 + 160) = 0;
    v12 = &v10[v11];
    if ( v10 != v12 )
    {
      for ( i = v10; ; i = *(__int64 **)(a1 + 72) )
      {
        v14 = *v10;
        v15 = (unsigned int)(v10 - i) >> 7;
        a2 = 4096LL << v15;
        if ( v15 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v10;
        sub_C7D6A0(v14, a2, 16);
        if ( v12 == v10 )
          break;
      }
    }
    v16 = *(__int64 **)(a1 + 120);
    v17 = &v16[2 * *(unsigned int *)(a1 + 128)];
    if ( v16 != v17 )
    {
      do
      {
        a2 = v16[1];
        v18 = *v16;
        v16 += 2;
        sub_C7D6A0(v18, a2, 16);
      }
      while ( v17 != v16 );
      v17 = *(__int64 **)(a1 + 120);
    }
    if ( v17 != (__int64 *)(a1 + 136) )
      _libc_free(v17, a2);
    v19 = *(_QWORD *)(a1 + 72);
    if ( v19 != a1 + 88 )
      _libc_free(v19, a2);
    return (void *)_libc_free(*(_QWORD *)(a1 + 32), a2);
  }
  return result;
}
