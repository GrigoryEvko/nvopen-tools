// Function: sub_22B0CF0
// Address: 0x22b0cf0
//
void __fastcall sub_22B0CF0(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rax
  unsigned int v5; // ecx
  __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rcx
  unsigned __int64 i; // r15
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdi
  _QWORD *v14; // r13
  __int64 v15; // r15
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  _QWORD *v22; // [rsp+8h] [rbp-38h]
  _QWORD *j; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v22 = &v2[v3];
  if ( v2 != v22 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v6 = 4096LL << v5;
      if ( v5 >= 0x1E )
        v6 = 0x40000000000LL;
      v7 = *v2 + v6;
      v8 = (*v2 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v2 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v7 = *(_QWORD *)a1;
      for ( i = v8 + 168; v7 >= i; i += 168LL )
      {
        v11 = *(_QWORD *)(i - 40);
        v12 = i - 168;
        if ( v11 != i - 24 )
          _libc_free(v11);
        if ( *(_BYTE *)(v12 + 120) )
        {
          v13 = *(_QWORD *)(v12 + 88);
          *(_BYTE *)(v12 + 120) = 0;
          if ( v13 != i - 64 )
            j_j___libc_free_0(v13);
        }
        v10 = *(_QWORD *)(v12 + 24);
        if ( v10 != i - 128 )
          _libc_free(v10);
      }
      if ( v22 == ++v2 )
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
      v17 += 168LL;
      if ( v16 < v17 )
        break;
      while ( 1 )
      {
        v18 = *(_QWORD *)(v17 - 40);
        v19 = v17 - 168;
        if ( v18 != v17 - 24 )
          _libc_free(v18);
        if ( *(_BYTE *)(v19 + 120) )
        {
          v21 = *(_QWORD *)(v19 + 88);
          *(_BYTE *)(v19 + 120) = 0;
          if ( v21 != v17 - 64 )
            j_j___libc_free_0(v21);
        }
        v20 = *(_QWORD *)(v19 + 24);
        if ( v20 == v17 - 128 )
          break;
        _libc_free(v20);
        v17 += 168LL;
        if ( v16 < v17 )
          goto LABEL_27;
      }
    }
LABEL_27:
    ;
  }
  sub_E66D20(a1);
}
