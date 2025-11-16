// Function: sub_321E250
// Address: 0x321e250
//
void __fastcall sub_321E250(__int64 a1)
{
  unsigned __int64 *v1; // r13
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi
  __int64 *v8; // r14
  __int64 *v9; // r12
  __int64 i; // rax
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rsi
  __int64 *v14; // r12
  unsigned __int64 v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned __int64 v18; // rdi

  v1 = *(unsigned __int64 **)(a1 + 192);
  v3 = *(unsigned __int64 **)(a1 + 184);
  if ( v1 != v3 )
  {
    do
    {
      if ( *v3 )
        j_j___libc_free_0(*v3);
      v3 += 3;
    }
    while ( v1 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 184);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v4 = *(_QWORD *)(a1 + 160);
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD *)(a1 + 128);
  v6 = v5 + ((unsigned __int64)*(unsigned int *)(a1 + 136) << 6);
  if ( v5 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v6 - 32);
      v6 -= 64LL;
      if ( v7 )
        j_j___libc_free_0(v7);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 128);
  }
  if ( v6 != a1 + 144 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 24LL * *(unsigned int *)(a1 + 120), 8);
  v8 = *(__int64 **)(a1 + 16);
  v9 = &v8[*(unsigned int *)(a1 + 24)];
  if ( v8 != v9 )
  {
    for ( i = *(_QWORD *)(a1 + 16); ; i = *(_QWORD *)(a1 + 16) )
    {
      v11 = *v8;
      v12 = (unsigned int)(((__int64)v8 - i) >> 3) >> 7;
      v13 = 4096LL << v12;
      if ( v12 >= 0x1E )
        v13 = 0x40000000000LL;
      ++v8;
      sub_C7D6A0(v11, v13, 16);
      if ( v9 == v8 )
        break;
    }
  }
  v14 = *(__int64 **)(a1 + 64);
  v15 = (unsigned __int64)&v14[2 * *(unsigned int *)(a1 + 72)];
  if ( v14 != (__int64 *)v15 )
  {
    do
    {
      v16 = v14[1];
      v17 = *v14;
      v14 += 2;
      sub_C7D6A0(v17, v16, 16);
    }
    while ( (__int64 *)v15 != v14 );
    v15 = *(_QWORD *)(a1 + 64);
  }
  if ( v15 != a1 + 80 )
    _libc_free(v15);
  v18 = *(_QWORD *)(a1 + 16);
  if ( v18 != a1 + 32 )
    _libc_free(v18);
}
