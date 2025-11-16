// Function: sub_26BB540
// Address: 0x26bb540
//
void __fastcall sub_26BB540(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 *v2; // r14
  __int64 *v3; // rbx
  __int64 i; // rax
  __int64 v5; // rdi
  unsigned int v6; // ecx
  __int64 v7; // rsi
  __int64 *v8; // rbx
  unsigned __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 )
  {
    v2 = *(__int64 **)(v1 + 56);
    v3 = &v2[*(unsigned int *)(v1 + 64)];
    if ( v2 != v3 )
    {
      for ( i = *(_QWORD *)(v1 + 56); ; i = *(_QWORD *)(v1 + 56) )
      {
        v5 = *v2;
        v6 = (unsigned int)(((__int64)v2 - i) >> 3) >> 7;
        v7 = 4096LL << v6;
        if ( v6 >= 0x1E )
          v7 = 0x40000000000LL;
        ++v2;
        sub_C7D6A0(v5, v7, 16);
        if ( v3 == v2 )
          break;
      }
    }
    v8 = *(__int64 **)(v1 + 104);
    v9 = (unsigned __int64)&v8[2 * *(unsigned int *)(v1 + 112)];
    if ( v8 != (__int64 *)v9 )
    {
      do
      {
        v10 = v8[1];
        v11 = *v8;
        v8 += 2;
        sub_C7D6A0(v11, v10, 16);
      }
      while ( (__int64 *)v9 != v8 );
      v9 = *(_QWORD *)(v1 + 104);
    }
    if ( v9 != v1 + 120 )
      _libc_free(v9);
    v12 = *(_QWORD *)(v1 + 56);
    if ( v12 != v1 + 72 )
      _libc_free(v12);
    sub_C7D6A0(*(_QWORD *)(v1 + 16), 16LL * *(unsigned int *)(v1 + 32), 8);
    j_j___libc_free_0(v1);
  }
}
