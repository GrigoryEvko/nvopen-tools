// Function: sub_EFE770
// Address: 0xefe770
//
__int64 __fastcall sub_EFE770(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rdi
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 *v8; // r14
  __int64 v9; // rdx
  __int64 *v10; // r12
  __int64 *i; // rdx
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // r12
  __int64 *v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 *v18; // [rsp+8h] [rbp-38h]

  v4 = a1 + 176;
  v5 = a1 + 88;
  *(_QWORD *)(v4 - 160) = a2;
  v6 = 1;
  *(_DWORD *)(v4 - 168) = 3;
  *(_DWORD *)(v4 - 152) = a3;
  *(_QWORD *)(v4 - 176) = &unk_49E4EB8;
  *(_BYTE *)(v4 - 16) = 0;
  *(_BYTE *)(v4 - 8) = 0;
  sub_EFDDC0(v4, 1);
  v18 = (__int64 *)(a1 + 136);
  if ( *(_BYTE *)(a1 + 160) )
  {
    v8 = *(__int64 **)(a1 + 72);
    v9 = *(unsigned int *)(a1 + 80);
    *(_BYTE *)(a1 + 160) = 0;
    v10 = &v8[v9];
    if ( v8 != v10 )
    {
      for ( i = v8; ; i = *(__int64 **)(a1 + 72) )
      {
        v12 = *v8;
        v13 = (unsigned int)(v8 - i) >> 7;
        v6 = 4096LL << v13;
        if ( v13 >= 0x1E )
          v6 = 0x40000000000LL;
        ++v8;
        sub_C7D6A0(v12, v6, 16);
        if ( v10 == v8 )
          break;
      }
    }
    v14 = *(__int64 **)(a1 + 120);
    v15 = &v14[2 * *(unsigned int *)(a1 + 128)];
    if ( v14 != v15 )
    {
      do
      {
        v6 = v14[1];
        v16 = *v14;
        v14 += 2;
        sub_C7D6A0(v16, v6, 16);
      }
      while ( v15 != v14 );
      v15 = *(__int64 **)(a1 + 120);
    }
    if ( v15 != v18 )
      _libc_free(v15, v6);
    v17 = *(_QWORD *)(a1 + 72);
    if ( v17 != v5 )
      _libc_free(v17, v6);
    _libc_free(*(_QWORD *)(a1 + 32), v6);
  }
  memset((void *)(a1 + 32), 0, 0x80u);
  *(_QWORD *)(a1 + 72) = v5;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  *(_BYTE *)(a1 + 52) = 16;
  *(_QWORD *)(a1 + 120) = v18;
  *(_QWORD *)(a1 + 144) = 1;
  *(_BYTE *)(a1 + 160) = 1;
  return a1 + 136;
}
