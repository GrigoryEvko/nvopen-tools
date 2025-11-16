// Function: sub_234DF60
// Address: 0x234df60
//
void __fastcall sub_234DF60(__int64 a1)
{
  __int64 *v2; // r12
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rdi
  unsigned int v6; // ecx
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  _QWORD *v13; // r12
  __int64 v14; // r14
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 *v20; // [rsp+8h] [rbp-38h]
  _QWORD *i; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v20 = &v2[v3];
  if ( v2 != v20 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = *v2;
      v6 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v7 = 4096LL << v6;
      if ( v6 >= 0x1E )
        v7 = 0x40000000000LL;
      v8 = (*v2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v9 = v5 + v7;
      if ( v5 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v9 = *(_QWORD *)a1;
      while ( 1 )
      {
        v8 += 112LL;
        if ( v9 < v8 )
          break;
        while ( *(_BYTE *)(v8 - 8) )
        {
          v10 = *(unsigned int *)(v8 - 16);
          v11 = *(_QWORD *)(v8 - 32);
          *(_BYTE *)(v8 - 8) = 0;
          sub_C7D6A0(v11, 16 * v10, 8);
          v12 = *(_QWORD *)(v8 - 88);
          if ( v12 == v8 - 72 )
            break;
          _libc_free(v12);
          v8 += 112LL;
          if ( v9 < v8 )
            goto LABEL_11;
        }
      }
LABEL_11:
      if ( v20 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v13 = *(_QWORD **)(a1 + 64);
  v14 = 2LL * *(unsigned int *)(a1 + 72);
  for ( i = &v13[v14]; i != v13; v13 += 2 )
  {
    v15 = *v13 + v13[1];
    v16 = (*v13 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v16 += 112LL;
      if ( v15 < v16 )
        break;
      while ( *(_BYTE *)(v16 - 8) )
      {
        v17 = *(unsigned int *)(v16 - 16);
        v18 = *(_QWORD *)(v16 - 32);
        *(_BYTE *)(v16 - 8) = 0;
        sub_C7D6A0(v18, 16 * v17, 8);
        v19 = *(_QWORD *)(v16 - 88);
        if ( v19 == v16 - 72 )
          break;
        _libc_free(v19);
        v16 += 112LL;
        if ( v15 < v16 )
          goto LABEL_19;
      }
    }
LABEL_19:
    ;
  }
  sub_E66D20(a1);
}
