// Function: sub_234E0E0
// Address: 0x234e0e0
//
void __fastcall sub_234E0E0(__int64 a1)
{
  __int64 *v2; // r12
  __int64 v3; // rsi
  __int64 v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // rbx
  bool v7; // cf
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 i; // r15
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdi
  _QWORD *v14; // r13
  __int64 v15; // r15
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rdi
  __int64 *v20; // [rsp+8h] [rbp-38h]
  _QWORD *j; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v20 = &v2[v3];
  if ( v2 != v20 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v6 = 4096LL << v5;
      v7 = v5 < 0x1E;
      v8 = *v2;
      if ( !v7 )
        v6 = 0x40000000000LL;
      v9 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v8 + v6;
      if ( v8 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v10 = *(_QWORD *)a1;
      for ( i = v9 + 136; v10 >= i; i += 136LL )
      {
        v12 = i - 136;
        if ( (*(_BYTE *)(i - 72) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v12 + 72), 16LL * *(unsigned int *)(v12 + 80), 8);
        v13 = *(_QWORD *)(v12 + 8);
        if ( v13 != i - 112 )
          _libc_free(v13);
      }
      if ( v20 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v14 = *(_QWORD **)(a1 + 64);
  v15 = 2LL * *(unsigned int *)(a1 + 72);
  for ( j = &v14[v15]; j != v14; v14 += 2 )
  {
    v16 = *v14 + v14[1];
    v17 = (*v14 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v17 += 136LL;
      if ( v16 < v17 )
        break;
      while ( 1 )
      {
        v18 = v17 - 136;
        if ( (*(_BYTE *)(v17 - 72) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v18 + 72), 16LL * *(unsigned int *)(v18 + 80), 8);
        v19 = *(_QWORD *)(v18 + 8);
        if ( v19 == v17 - 112 )
          break;
        _libc_free(v19);
        v17 += 136LL;
        if ( v16 < v17 )
          goto LABEL_22;
      }
    }
LABEL_22:
    ;
  }
  sub_E66D20(a1);
}
