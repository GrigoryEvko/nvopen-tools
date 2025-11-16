// Function: sub_CCC0B0
// Address: 0xccc0b0
//
__int64 __fastcall sub_CCC0B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v5; // zf
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 v8; // r12
  __int64 *v9; // r15
  __int64 *v10; // r13
  __int64 i; // rax
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // r13
  __int64 *v15; // r14
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 result; // rax

  v5 = *(_BYTE *)(a1 + 137) == 0;
  *(_QWORD *)a1 = &unk_49DD408;
  if ( !v5 )
  {
    v6 = *(_QWORD *)(a1 + 16);
    if ( v6 )
    {
      sub_BA9C10(*(_QWORD ***)(a1 + 16), a2, a3, a4);
      a2 = 880;
      j_j___libc_free_0(v6, 880);
    }
  }
  if ( *(_BYTE *)(a1 + 136) )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( v7 )
    {
      sub_B6E710(*(_QWORD **)(a1 + 8));
      a2 = 8;
      j_j___libc_free_0(v7, 8);
    }
  }
  v8 = *(_QWORD *)(a1 + 144);
  if ( v8 )
  {
    v9 = *(__int64 **)(v8 + 16);
    v10 = &v9[*(unsigned int *)(v8 + 24)];
    if ( v9 != v10 )
    {
      for ( i = *(_QWORD *)(v8 + 16); ; i = *(_QWORD *)(v8 + 16) )
      {
        v12 = *v9;
        v13 = (unsigned int)(((__int64)v9 - i) >> 3) >> 7;
        a2 = 4096LL << v13;
        if ( v13 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v9;
        sub_C7D6A0(v12, a2, 16);
        if ( v10 == v9 )
          break;
      }
    }
    v14 = *(__int64 **)(v8 + 64);
    v15 = &v14[2 * *(unsigned int *)(v8 + 72)];
    if ( v14 != v15 )
    {
      do
      {
        a2 = v14[1];
        v16 = *v14;
        v14 += 2;
        sub_C7D6A0(v16, a2, 16);
      }
      while ( v15 != v14 );
      v15 = *(__int64 **)(v8 + 64);
    }
    if ( v15 != (__int64 *)(v8 + 80) )
      _libc_free(v15, a2);
    v17 = *(_QWORD *)(v8 + 16);
    if ( v17 != v8 + 32 )
      _libc_free(v17, a2);
    j_j___libc_free_0(v8, 96);
  }
  v18 = *(_QWORD *)(a1 + 80);
  result = a1 + 96;
  if ( v18 != a1 + 96 )
    return j_j___libc_free_0(v18, *(_QWORD *)(a1 + 96) + 1LL);
  return result;
}
