// Function: sub_32478E0
// Address: 0x32478e0
//
void __fastcall sub_32478E0(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
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

  v1 = *(_QWORD *)(a1 + 376);
  *(_QWORD *)a1 = &unk_4A35DB8;
  v3 = v1 + 88LL * *(unsigned int *)(a1 + 384);
  if ( v1 != v3 )
  {
    do
    {
      v3 -= 88LL;
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 != v3 + 24 )
        _libc_free(v4);
    }
    while ( v1 != v3 );
    v3 = *(_QWORD *)(a1 + 376);
  }
  if ( v3 != a1 + 392 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 352), 16LL * *(unsigned int *)(a1 + 368), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 320), 16LL * *(unsigned int *)(a1 + 336), 8);
  v5 = *(_QWORD *)(a1 + 288);
  if ( v5 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 264);
  if ( v6 )
    j_j___libc_free_0(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16LL * *(unsigned int *)(a1 + 256), 8);
  v7 = *(__int64 **)(a1 + 104);
  v8 = &v7[*(unsigned int *)(a1 + 112)];
  if ( v7 != v8 )
  {
    for ( i = *(_QWORD *)(a1 + 104); ; i = *(_QWORD *)(a1 + 104) )
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
  v13 = *(__int64 **)(a1 + 152);
  v14 = (unsigned __int64)&v13[2 * *(unsigned int *)(a1 + 160)];
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
    v14 = *(_QWORD *)(a1 + 152);
  }
  if ( v14 != a1 + 168 )
    _libc_free(v14);
  v17 = *(_QWORD *)(a1 + 104);
  if ( v17 != a1 + 120 )
    _libc_free(v17);
}
