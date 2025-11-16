// Function: sub_38C0010
// Address: 0x38c0010
//
__int64 __fastcall sub_38C0010(__int64 a1)
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
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  _QWORD *v13; // r15
  __int64 (__fastcall *v14)(_QWORD *); // rdx
  _QWORD *v15; // r13
  __int64 v16; // r15
  unsigned __int64 v17; // r12
  unsigned __int64 j; // r15
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  _QWORD *v21; // r14
  __int64 (__fastcall *v22)(_QWORD *); // rdx
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 result; // rax
  _QWORD *v27; // rbx
  __int64 v28; // rdx
  unsigned __int64 *v29; // r12
  unsigned __int64 *v30; // rbx
  unsigned __int64 v31; // rdi
  __int64 *v32; // [rsp+0h] [rbp-40h]
  _QWORD *v33; // [rsp+0h] [rbp-40h]

  v1 = *(__int64 **)(a1 + 16);
  v2 = *(unsigned int *)(a1 + 24);
  v32 = &v1[v2];
  if ( v1 != v32 )
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
      for ( i = v8 + 216; v9 >= i; i += 216LL )
      {
        while ( 1 )
        {
          v13 = (_QWORD *)(i - 216);
          v14 = **(__int64 (__fastcall ***)(_QWORD *))(i - 216);
          if ( v14 == sub_168C470 )
            break;
          i += 216LL;
          v14(v13);
          if ( v9 < i )
            goto LABEL_16;
        }
        *(_QWORD *)(i - 216) = &unk_49EE580;
        v11 = v13[8];
        if ( v11 != i - 136 )
          j_j___libc_free_0(v11);
        v12 = v13[1];
        if ( v12 != i - 192 )
          j_j___libc_free_0(v12);
      }
LABEL_16:
      if ( v32 == ++v1 )
        break;
      v3 = *(_QWORD *)(a1 + 16);
      v2 = *(unsigned int *)(a1 + 24);
    }
  }
  v15 = *(_QWORD **)(a1 + 64);
  v16 = 2LL * *(unsigned int *)(a1 + 72);
  v33 = &v15[v16];
  if ( &v15[v16] != v15 )
  {
    do
    {
      v17 = *v15 + v15[1];
      for ( j = ((*v15 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 216; v17 >= j; j += 216LL )
      {
        while ( 1 )
        {
          v21 = (_QWORD *)(j - 216);
          v22 = **(__int64 (__fastcall ***)(_QWORD *))(j - 216);
          if ( v22 == sub_168C470 )
            break;
          j += 216LL;
          v22(v21);
          if ( v17 < j )
            goto LABEL_28;
        }
        *(_QWORD *)(j - 216) = &unk_49EE580;
        v19 = v21[8];
        if ( v19 != j - 136 )
          j_j___libc_free_0(v19);
        v20 = v21[1];
        if ( v20 != j - 192 )
          j_j___libc_free_0(v20);
      }
LABEL_28:
      v15 += 2;
    }
    while ( v33 != v15 );
    v23 = *(unsigned __int64 **)(a1 + 64);
    v24 = &v23[2 * *(unsigned int *)(a1 + 72)];
    while ( v23 != v24 )
    {
      v25 = *v23;
      v23 += 2;
      _libc_free(v25);
    }
  }
  *(_DWORD *)(a1 + 72) = 0;
  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v27 = *(_QWORD **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v28 = *v27;
    v29 = &v27[result];
    v30 = v27 + 1;
    *(_QWORD *)a1 = v28;
    *(_QWORD *)(a1 + 8) = v28 + 4096;
    while ( v29 != v30 )
    {
      v31 = *v30++;
      _libc_free(v31);
    }
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  return result;
}
