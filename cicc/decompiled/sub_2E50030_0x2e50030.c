// Function: sub_2E50030
// Address: 0x2e50030
//
__int64 __fastcall sub_2E50030(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 *v3; // r14
  __int64 v4; // rax
  __int64 *v5; // r12
  __int64 *i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rsi
  __int64 *v10; // r12
  unsigned __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int64 v14; // rdi

  v2 = *(_QWORD *)(a1 + 288);
  if ( v2 != a1 + 304 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), 16LL * *(unsigned int *)(a1 + 272), 8);
  v3 = *(__int64 **)(a1 + 168);
  v4 = *(unsigned int *)(a1 + 176);
  *(_QWORD *)(a1 + 144) = 0;
  v5 = &v3[v4];
  if ( v3 != v5 )
  {
    for ( i = v3; ; i = *(__int64 **)(a1 + 168) )
    {
      v7 = *v3;
      v8 = (unsigned int)(v3 - i) >> 7;
      v9 = 4096LL << v8;
      if ( v8 >= 0x1E )
        v9 = 0x40000000000LL;
      ++v3;
      sub_C7D6A0(v7, v9, 16);
      if ( v5 == v3 )
        break;
    }
  }
  v10 = *(__int64 **)(a1 + 216);
  v11 = (unsigned __int64)&v10[2 * *(unsigned int *)(a1 + 224)];
  if ( v10 != (__int64 *)v11 )
  {
    do
    {
      v12 = v10[1];
      v13 = *v10;
      v10 += 2;
      sub_C7D6A0(v13, v12, 16);
    }
    while ( (__int64 *)v11 != v10 );
    v11 = *(_QWORD *)(a1 + 216);
  }
  if ( v11 != a1 + 232 )
    _libc_free(v11);
  v14 = *(_QWORD *)(a1 + 168);
  if ( v14 != a1 + 184 )
    _libc_free(v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 16LL * *(unsigned int *)(a1 + 136), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 88), 16LL * *(unsigned int *)(a1 + 104), 8);
}
