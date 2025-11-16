// Function: sub_2E22D50
// Address: 0x2e22d50
//
void __fastcall sub_2E22D50(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi
  _QWORD *v4; // r12
  _QWORD *v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 *v7; // r14
  __int64 *v8; // r12
  __int64 i; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rsi
  __int64 *v13; // r12
  unsigned __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int64 v17; // rdi

  v1 = *(_QWORD *)(a1 + 176);
  while ( v1 )
  {
    sub_2E22B80(*(_QWORD *)(v1 + 24));
    v3 = v1;
    v1 = *(_QWORD *)(v1 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = *(_QWORD **)(a1 + 120);
  while ( v4 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    sub_2E22AE0(v5);
  }
  memset(*(void **)(a1 + 104), 0, 8LL * *(_QWORD *)(a1 + 112));
  v6 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  if ( v6 != a1 + 152 )
    j_j___libc_free_0(v6);
  v7 = *(__int64 **)(a1 + 24);
  v8 = &v7[*(unsigned int *)(a1 + 32)];
  if ( v7 != v8 )
  {
    for ( i = *(_QWORD *)(a1 + 24); ; i = *(_QWORD *)(a1 + 24) )
    {
      v10 = *v7;
      v11 = (unsigned int)(((__int64)v7 - i) >> 3) >> 7;
      v12 = 4096LL << v11;
      if ( v11 >= 0x1E )
        v12 = 0x40000000000LL;
      ++v7;
      sub_C7D6A0(v10, v12, 16);
      if ( v8 == v7 )
        break;
    }
  }
  v13 = *(__int64 **)(a1 + 72);
  v14 = (unsigned __int64)&v13[2 * *(unsigned int *)(a1 + 80)];
  if ( v13 != (__int64 *)v14 )
  {
    do
    {
      v15 = v13[1];
      v16 = *v13;
      v13 += 2;
      sub_C7D6A0(v16, v15, 16);
    }
    while ( (__int64 *)v14 != v13 );
    v14 = *(_QWORD *)(a1 + 72);
  }
  if ( v14 != a1 + 88 )
    _libc_free(v14);
  v17 = *(_QWORD *)(a1 + 24);
  if ( v17 != a1 + 40 )
    _libc_free(v17);
}
