// Function: sub_E9B0A0
// Address: 0xe9b0a0
//
__int64 *__fastcall sub_E9B0A0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  bool v5; // zf
  __int64 v6; // rcx
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 *v13; // r14
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdi
  __int64 v24; // rdi
  void *v25; // rdi
  __int64 v27; // rbx
  __int64 *v28; // [rsp+0h] [rbp-70h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 *v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  __int64 *src; // [rsp+30h] [rbp-40h]
  _QWORD *i; // [rsp+38h] [rbp-38h]

  src = (__int64 *)a2;
  v28 = (__int64 *)a1[1];
  v3 = ((__int64)v28 - *a1) >> 3;
  v31 = (__int64 *)*a1;
  if ( v3 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = v3 == 0;
  v6 = ((__int64)v28 - *a1) >> 3;
  v7 = 1;
  if ( !v5 )
    v7 = ((__int64)v28 - *a1) >> 3;
  v8 = __CFADD__(v6, v7);
  v9 = v6 + v7;
  if ( v8 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v29 = 0;
      v10 = 8;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      v9 = 0xFFFFFFFFFFFFFFFLL;
    v27 = 8 * v9;
  }
  v32 = sub_22077B0(v27);
  v29 = v32 + v27;
  v10 = v32 + 8;
LABEL_7:
  v11 = (_QWORD *)(v32 + a2 - (_QWORD)v31);
  if ( v11 )
  {
    v12 = *a3;
    *a3 = 0;
    *v11 = v12;
  }
  v13 = v31;
  if ( (__int64 *)a2 != v31 )
  {
    for ( i = (_QWORD *)v32; ; ++i )
    {
      v14 = *v13;
      if ( i )
      {
        *i = v14;
        *v13 = 0;
      }
      else if ( v14 )
      {
        v15 = *(_QWORD *)(v14 + 168);
        v16 = *(_QWORD *)(v14 + 160);
        if ( v15 != v16 )
        {
          do
          {
            v17 = *(_QWORD *)(v16 + 64);
            v18 = v16 + 80;
            if ( v17 != v16 + 80 )
              _libc_free(v17, a2);
            v19 = *(unsigned int *)(v16 + 56);
            v20 = *(_QWORD *)(v16 + 40);
            v16 += 80;
            a2 = 16 * v19;
            sub_C7D6A0(v20, a2, 8);
          }
          while ( v15 != v18 );
          v16 = *(_QWORD *)(v14 + 160);
        }
        if ( v16 )
        {
          a2 = *(_QWORD *)(v14 + 176) - v16;
          j_j___libc_free_0(v16, a2);
        }
        v21 = *(_QWORD *)(v14 + 144);
        v22 = v21 + 48LL * *(unsigned int *)(v14 + 152);
        if ( v21 != v22 )
        {
          do
          {
            v23 = *(_QWORD *)(v22 - 40);
            v22 -= 48;
            if ( v23 )
            {
              a2 = *(_QWORD *)(v22 + 24) - v23;
              j_j___libc_free_0(v23, a2);
            }
          }
          while ( v21 != v22 );
          v22 = *(_QWORD *)(v14 + 144);
        }
        if ( v14 + 160 != v22 )
          _libc_free(v22, a2);
        sub_C7D6A0(*(_QWORD *)(v14 + 120), 16LL * *(unsigned int *)(v14 + 136), 8);
        v24 = *(_QWORD *)(v14 + 88);
        if ( v24 )
          j_j___libc_free_0(v24, *(_QWORD *)(v14 + 104) - v24);
        a2 = 184;
        j_j___libc_free_0(v14, 184);
      }
      if ( ++v13 == src )
        break;
    }
    v10 = (__int64)(i + 2);
  }
  if ( src != v28 )
  {
    v25 = (void *)v10;
    v10 += (char *)v28 - (char *)src;
    memcpy(v25, src, (char *)v28 - (char *)src);
  }
  if ( v31 )
    j_j___libc_free_0(v31, a1[2] - (_QWORD)v31);
  *a1 = v32;
  a1[1] = v10;
  a1[2] = v29;
  return a1;
}
