// Function: sub_E38EC0
// Address: 0xe38ec0
//
__int64 **__fastcall sub_E38EC0(__int64 **a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  bool v6; // zf
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // r12
  _QWORD *i; // r13
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 *v21; // rbx
  __int64 *v22; // r14
  __int64 v23; // rdi
  void *v24; // rdi
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 *v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+18h] [rbp-58h]
  __int64 *v31; // [rsp+28h] [rbp-48h]
  _QWORD *v32; // [rsp+30h] [rbp-40h]
  __int64 *src; // [rsp+38h] [rbp-38h]

  src = (__int64 *)a2;
  v28 = a1[1];
  v4 = v28 - *a1;
  v31 = *a1;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = v28 - *a1;
  v6 = v4 == 0;
  v7 = 1;
  if ( !v6 )
    v7 = v28 - *a1;
  v8 = __CFADD__(v5, v7);
  v9 = v5 + v7;
  v10 = a2 - (_QWORD)v31;
  if ( v8 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v29 = 0;
      v11 = 8;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      v9 = 0xFFFFFFFFFFFFFFFLL;
    v26 = 8 * v9;
  }
  v27 = sub_22077B0(v26);
  v10 = a2 - (_QWORD)v31;
  v32 = (_QWORD *)v27;
  v11 = v27 + 8;
  v29 = v27 + v26;
LABEL_7:
  v12 = (_QWORD *)((char *)v32 + v10);
  if ( v12 )
  {
    v13 = *a3;
    *a3 = 0;
    *v12 = v13;
  }
  v14 = v31;
  if ( (__int64 *)a2 != v31 )
  {
    for ( i = v32; ; i = (_QWORD *)v16 )
    {
      v17 = *v14;
      if ( i )
        break;
      if ( !v17 )
        goto LABEL_12;
      v18 = *(_QWORD *)(v17 + 176);
      if ( v18 != v17 + 192 )
        _libc_free(v18, a2);
      v19 = *(_QWORD *)(v17 + 88);
      if ( v19 != v17 + 104 )
        _libc_free(v19, a2);
      v20 = 8LL * *(unsigned int *)(v17 + 80);
      sub_C7D6A0(*(_QWORD *)(v17 + 64), v20, 8);
      v21 = *(__int64 **)(v17 + 40);
      v22 = *(__int64 **)(v17 + 32);
      if ( v21 != v22 )
      {
        do
        {
          if ( *v22 )
            sub_E38110(*v22, v20);
          ++v22;
        }
        while ( v21 != v22 );
        v22 = *(__int64 **)(v17 + 32);
      }
      if ( v22 )
      {
        v20 = *(_QWORD *)(v17 + 48) - (_QWORD)v22;
        j_j___libc_free_0(v22, v20);
      }
      v23 = *(_QWORD *)(v17 + 8);
      if ( v23 != v17 + 24 )
        _libc_free(v23, v20);
      a2 = 224;
      ++v14;
      j_j___libc_free_0(v17, 224);
      v16 = 8;
      if ( v14 == src )
      {
LABEL_30:
        v11 = (__int64)(i + 2);
        goto LABEL_31;
      }
LABEL_13:
      ;
    }
    *i = v17;
    *v14 = 0;
LABEL_12:
    ++v14;
    v16 = (__int64)(i + 1);
    if ( v14 == src )
      goto LABEL_30;
    goto LABEL_13;
  }
LABEL_31:
  if ( src != v28 )
  {
    v24 = (void *)v11;
    v11 += (char *)v28 - (char *)src;
    memcpy(v24, src, (char *)v28 - (char *)src);
  }
  if ( v31 )
    j_j___libc_free_0(v31, (char *)a1[2] - (char *)v31);
  *a1 = v32;
  a1[1] = (__int64 *)v11;
  a1[2] = (__int64 *)v29;
  return a1;
}
