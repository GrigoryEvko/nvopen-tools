// Function: sub_16104D0
// Address: 0x16104d0
//
void __fastcall sub_16104D0(__int64 a1)
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
  unsigned __int64 v12; // rdi
  _QWORD *v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r14
  __int64 v18; // r15
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  _QWORD *v22; // r15
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // rbx
  __int64 v31; // rdx
  unsigned __int64 *v32; // r12
  unsigned __int64 *v33; // rbx
  unsigned __int64 v34; // rdi
  __int64 *v35; // [rsp+8h] [rbp-38h]
  _QWORD *v36; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v35 = &v2[v3];
  if ( v2 != v35 )
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
      for ( i = v9 + 176; v10 >= i; i += 176LL )
      {
        v12 = *(_QWORD *)(i - 24);
        v13 = (_QWORD *)(i - 176);
        if ( v12 != i - 8 )
          _libc_free(v12);
        v14 = v13[15];
        if ( v14 != i - 40 )
          _libc_free(v14);
        v15 = v13[11];
        if ( v15 != i - 72 )
          _libc_free(v15);
        v16 = v13[1];
        if ( v16 != i - 152 )
          _libc_free(v16);
      }
      if ( v35 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v17 = *(_QWORD **)(a1 + 64);
  v18 = 2LL * *(unsigned int *)(a1 + 72);
  v36 = &v17[v18];
  if ( &v17[v18] != v17 )
  {
    do
    {
      v19 = *v17 + v17[1];
      v20 = (*v17 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v20 += 176LL;
        if ( v19 < v20 )
          break;
        while ( 1 )
        {
          v21 = *(_QWORD *)(v20 - 24);
          v22 = (_QWORD *)(v20 - 176);
          if ( v21 != v20 - 8 )
            _libc_free(v21);
          v23 = v22[15];
          if ( v23 != v20 - 40 )
            _libc_free(v23);
          v24 = v22[11];
          if ( v24 != v20 - 72 )
            _libc_free(v24);
          v25 = v22[1];
          if ( v25 == v20 - 152 )
            break;
          _libc_free(v25);
          v20 += 176LL;
          if ( v19 < v20 )
            goto LABEL_30;
        }
      }
LABEL_30:
      v17 += 2;
    }
    while ( v36 != v17 );
    v26 = *(unsigned __int64 **)(a1 + 64);
    v27 = &v26[2 * *(unsigned int *)(a1 + 72)];
    while ( v26 != v27 )
    {
      v28 = *v26;
      v26 += 2;
      _libc_free(v28);
    }
  }
  v29 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v29 )
  {
    v30 = *(_QWORD **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v31 = *v30;
    v32 = &v30[v29];
    v33 = v30 + 1;
    *(_QWORD *)a1 = v31;
    *(_QWORD *)(a1 + 8) = v31 + 4096;
    while ( v32 != v33 )
    {
      v34 = *v33++;
      _libc_free(v34);
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
