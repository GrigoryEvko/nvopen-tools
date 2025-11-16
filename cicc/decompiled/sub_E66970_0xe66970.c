// Function: sub_E66970
// Address: 0xe66970
//
__int64 __fastcall sub_E66970(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // rsi
  __int64 v3; // rdx
  unsigned int v4; // ecx
  __int64 v5; // r12
  bool v6; // cf
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  unsigned __int64 i; // r14
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // r15
  __int64 (__fastcall *v16)(_QWORD *); // rdx
  _QWORD *v17; // r13
  __int64 v18; // r15
  unsigned __int64 v19; // r12
  unsigned __int64 j; // r15
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  _QWORD *v25; // r14
  __int64 (__fastcall *v26)(_QWORD *); // rdx
  __int64 *v27; // rbx
  __int64 *v28; // r12
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 result; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // r13
  __int64 *v36; // rbx
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 v39; // rsi
  __int64 *v40; // [rsp+0h] [rbp-40h]
  _QWORD *v41; // [rsp+0h] [rbp-40h]

  v1 = *(__int64 **)(a1 + 16);
  v2 = *(unsigned int *)(a1 + 24);
  v40 = &v1[v2];
  if ( v1 != v40 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v4 = (unsigned int)(((__int64)v1 - v3) >> 3) >> 7;
      v5 = 4096LL << v4;
      v6 = v4 < 0x1E;
      v7 = *v1;
      if ( !v6 )
        v5 = 0x40000000000LL;
      v8 = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v9 = v7 + v5;
      if ( v7 == *(_QWORD *)(v3 + 8 * v2 - 8) )
        v9 = *(_QWORD *)a1;
      for ( i = v8 + 304; v9 >= i; i += 304LL )
      {
        while ( 1 )
        {
          v15 = (_QWORD *)(i - 304);
          v16 = **(__int64 (__fastcall ***)(_QWORD *))(i - 304);
          if ( v16 == sub_C12480 )
            break;
          i += 304LL;
          v16(v15);
          if ( v9 < i )
            goto LABEL_20;
        }
        *(_QWORD *)(i - 304) = &unk_49E41D0;
        v11 = v15[34];
        if ( v11 != i - 16 )
          j_j___libc_free_0(v11, v15[36] + 1LL);
        v12 = v15[12];
        if ( v12 != i - 192 )
          j_j___libc_free_0(v12, v15[14] + 1LL);
        v13 = v15[8];
        if ( v13 != i - 224 )
          j_j___libc_free_0(v13, v15[10] + 1LL);
        v14 = v15[1];
        if ( v14 != i - 280 )
          j_j___libc_free_0(v14, v15[3] + 1LL);
      }
LABEL_20:
      if ( v40 == ++v1 )
        break;
      v3 = *(_QWORD *)(a1 + 16);
      v2 = *(unsigned int *)(a1 + 24);
    }
  }
  v17 = *(_QWORD **)(a1 + 64);
  v18 = 2LL * *(unsigned int *)(a1 + 72);
  v41 = &v17[v18];
  if ( &v17[v18] != v17 )
  {
    do
    {
      v19 = *v17 + v17[1];
      for ( j = ((*v17 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 304; v19 >= j; j += 304LL )
      {
        while ( 1 )
        {
          v25 = (_QWORD *)(j - 304);
          v26 = **(__int64 (__fastcall ***)(_QWORD *))(j - 304);
          if ( v26 == sub_C12480 )
            break;
          j += 304LL;
          v26(v25);
          if ( v19 < j )
            goto LABEL_36;
        }
        *(_QWORD *)(j - 304) = &unk_49E41D0;
        v21 = v25[34];
        if ( v21 != j - 16 )
          j_j___libc_free_0(v21, v25[36] + 1LL);
        v22 = v25[12];
        if ( v22 != j - 192 )
          j_j___libc_free_0(v22, v25[14] + 1LL);
        v23 = v25[8];
        if ( v23 != j - 224 )
          j_j___libc_free_0(v23, v25[10] + 1LL);
        v24 = v25[1];
        if ( v24 != j - 280 )
          j_j___libc_free_0(v24, v25[3] + 1LL);
      }
LABEL_36:
      v17 += 2;
    }
    while ( v41 != v17 );
    v27 = *(__int64 **)(a1 + 64);
    v28 = &v27[2 * *(unsigned int *)(a1 + 72)];
    while ( v28 != v27 )
    {
      v29 = v27[1];
      v30 = *v27;
      v27 += 2;
      sub_C7D6A0(v30, v29, 16);
    }
  }
  result = a1;
  v32 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v32 )
  {
    v33 = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v34 = *v33;
    v35 = v33 + 1;
    *(_QWORD *)a1 = *v33;
    *(_QWORD *)(a1 + 8) = v34 + 4096;
    v36 = &v33[v32];
    if ( v36 != v33 + 1 )
    {
      while ( 1 )
      {
        v37 = *v35;
        v38 = (unsigned int)(v35 - v33) >> 7;
        v39 = 4096LL << v38;
        if ( v38 >= 0x1E )
          v39 = 0x40000000000LL;
        ++v35;
        sub_C7D6A0(v37, v39, 16);
        if ( v36 == v35 )
          break;
        v33 = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  return result;
}
