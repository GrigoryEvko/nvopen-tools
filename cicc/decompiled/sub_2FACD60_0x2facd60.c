// Function: sub_2FACD60
// Address: 0x2facd60
//
void __fastcall sub_2FACD60(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 *v5; // r14
  __int64 *v6; // r12
  __int64 i; // rax
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 v10; // rsi
  __int64 *v11; // r12
  unsigned __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  v1 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 104) = a1 + 96;
  *(_QWORD *)(a1 + 96) = (a1 + 96) | v1 & 7;
  v3 = *(_QWORD *)(a1 + 296);
  if ( v3 != a1 + 312 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 152);
  if ( v4 != a1 + 168 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * *(unsigned int *)(a1 + 144), 8);
  v5 = *(__int64 **)(a1 + 16);
  v6 = &v5[*(unsigned int *)(a1 + 24)];
  if ( v5 != v6 )
  {
    for ( i = *(_QWORD *)(a1 + 16); ; i = *(_QWORD *)(a1 + 16) )
    {
      v8 = *v5;
      v9 = (unsigned int)(((__int64)v5 - i) >> 3) >> 7;
      v10 = 4096LL << v9;
      if ( v9 >= 0x1E )
        v10 = 0x40000000000LL;
      ++v5;
      sub_C7D6A0(v8, v10, 16);
      if ( v6 == v5 )
        break;
    }
  }
  v11 = *(__int64 **)(a1 + 64);
  v12 = (unsigned __int64)&v11[2 * *(unsigned int *)(a1 + 72)];
  if ( v11 != (__int64 *)v12 )
  {
    do
    {
      v13 = v11[1];
      v14 = *v11;
      v11 += 2;
      sub_C7D6A0(v14, v13, 16);
    }
    while ( (__int64 *)v12 != v11 );
    v12 = *(_QWORD *)(a1 + 64);
  }
  if ( v12 != a1 + 80 )
    _libc_free(v12);
  v15 = *(_QWORD *)(a1 + 16);
  if ( v15 != a1 + 32 )
    _libc_free(v15);
}
