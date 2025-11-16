// Function: sub_1084C60
// Address: 0x1084c60
//
__int64 __fastcall sub_1084C60(_QWORD *a1, __int64 a2, size_t a3)
{
  size_t v3; // r15
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // r12
  void *v9; // rdi
  __int64 *v10; // r14
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  bool v17; // cf
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r15
  _QWORD *i; // r13
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 *v25; // rdi
  __int64 v26; // r13
  __int64 v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 *v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  v3 = a3;
  v6 = sub_22077B0(136);
  v8 = v6;
  if ( v6 )
  {
    v9 = (void *)(v6 + 48);
    *(_DWORD *)(v6 + 16) = 0;
    *(_QWORD *)(v6 + 24) = v6 + 48;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = 8;
    *(_OWORD *)v6 = 0;
    if ( a3 > 8 )
    {
      sub_C8D290(v6 + 24, (const void *)(v6 + 48), a3, 1u, v6 + 24, v7);
      v9 = (void *)(*(_QWORD *)(v8 + 24) + *(_QWORD *)(v8 + 32));
    }
    else if ( !a3 )
    {
LABEL_4:
      *(_QWORD *)(v8 + 32) = v3;
      *(_QWORD *)(v8 + 64) = v8 + 80;
      *(_DWORD *)(v8 + 56) = 0;
      *(_QWORD *)(v8 + 72) = 0x100000000LL;
      *(_QWORD *)(v8 + 104) = 0;
      *(_QWORD *)(v8 + 112) = 0;
      *(_DWORD *)(v8 + 120) = 0;
      *(_QWORD *)(v8 + 128) = 0;
      goto LABEL_5;
    }
    memcpy(v9, (const void *)a2, a3);
    v3 = a3 + *(_QWORD *)(v8 + 32);
    goto LABEL_4;
  }
LABEL_5:
  v10 = (__int64 *)a1[10];
  if ( v10 == (__int64 *)a1[11] )
  {
    v14 = (__int64)v10 - a1[9];
    v30 = (__int64 *)a1[9];
    v15 = v14 >> 3;
    if ( v14 >> 3 == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v16 = 1;
    if ( v15 )
      v16 = v14 >> 3;
    v17 = __CFADD__(v16, v15);
    v18 = v16 + v15;
    if ( v17 )
    {
      v26 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v18 )
      {
        v27 = 0;
        v19 = 8;
        v32 = 0;
LABEL_24:
        a2 = v32 + v14;
        if ( a2 )
        {
          *(_QWORD *)a2 = v8;
          v8 = 0;
        }
        v20 = v30;
        if ( v10 == v30 )
        {
LABEL_39:
          v25 = v30;
          if ( v30 )
          {
            v31 = v19;
            a2 = a1[11] - (_QWORD)v25;
            j_j___libc_free_0(v25, a2);
            v19 = v31;
          }
          a1[10] = v19;
          a1[9] = v32;
          a1[11] = v27;
          goto LABEL_12;
        }
        for ( i = (_QWORD *)v32; ; i = (_QWORD *)a2 )
        {
          v22 = *v20;
          if ( i )
            break;
          if ( !v22 )
            goto LABEL_29;
          v23 = *(_QWORD *)(v22 + 64);
          if ( v23 != v22 + 80 )
          {
            v28 = *v20;
            _libc_free(v23, v22 + 80);
            v22 = v28;
          }
          v24 = *(_QWORD *)(v22 + 24);
          if ( v24 != v22 + 48 )
          {
            v29 = v22;
            _libc_free(v24, v22 + 48);
            v22 = v29;
          }
          ++v20;
          j_j___libc_free_0(v22, 136);
          a2 = 8;
          if ( v10 == v20 )
          {
LABEL_38:
            v19 = (__int64)(i + 2);
            goto LABEL_39;
          }
LABEL_30:
          ;
        }
        *i = v22;
        *v20 = 0;
LABEL_29:
        ++v20;
        a2 = (__int64)(i + 1);
        if ( v10 == v20 )
          goto LABEL_38;
        goto LABEL_30;
      }
      if ( v18 > 0xFFFFFFFFFFFFFFFLL )
        v18 = 0xFFFFFFFFFFFFFFFLL;
      v26 = 8 * v18;
    }
    v32 = sub_22077B0(v26);
    v19 = v32 + 8;
    v27 = v32 + v26;
    goto LABEL_24;
  }
  if ( v10 )
  {
    *v10 = v8;
    a1[10] += 8LL;
    return *(_QWORD *)(a1[10] - 8LL);
  }
  a1[10] = 8;
LABEL_12:
  if ( v8 )
  {
    v12 = *(_QWORD *)(v8 + 64);
    if ( v12 != v8 + 80 )
      _libc_free(v12, a2);
    v13 = *(_QWORD *)(v8 + 24);
    if ( v13 != v8 + 48 )
      _libc_free(v13, a2);
    j_j___libc_free_0(v8, 136);
  }
  return *(_QWORD *)(a1[10] - 8LL);
}
