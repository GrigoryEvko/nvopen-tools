// Function: sub_3925260
// Address: 0x3925260
//
__int64 __fastcall sub_3925260(_QWORD *a1, const void *a2, unsigned __int64 a3)
{
  int v3; // r15d
  __int64 v6; // rax
  int v7; // r9d
  unsigned __int64 v8; // r12
  void *v9; // rdi
  unsigned __int64 *v10; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  bool v17; // cf
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 *v20; // rsi
  unsigned __int64 *v21; // r15
  _QWORD *i; // r13
  __int64 v23; // rsi
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // [rsp+0h] [rbp-50h]
  unsigned __int64 v30; // [rsp+8h] [rbp-48h]
  unsigned __int64 v31; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]

  v3 = a3;
  v6 = sub_22077B0(0x80u);
  v8 = v6;
  if ( v6 )
  {
    v9 = (void *)(v6 + 40);
    *(_DWORD *)(v6 + 16) = 0;
    *(_QWORD *)(v6 + 24) = v6 + 40;
    *(_OWORD *)v6 = 0;
    *(_QWORD *)(v6 + 32) = 0x800000000LL;
    if ( a3 > 8 )
    {
      sub_16CD150(v6 + 24, v9, a3, 1, v6 + 24, v7);
      v9 = (void *)(*(_QWORD *)(v8 + 24) + *(unsigned int *)(v8 + 32));
    }
    else if ( !a3 )
    {
LABEL_4:
      *(_DWORD *)(v8 + 32) = v3;
      *(_QWORD *)(v8 + 56) = v8 + 72;
      *(_QWORD *)(v8 + 64) = 0x100000000LL;
      *(_QWORD *)(v8 + 96) = 0;
      *(_QWORD *)(v8 + 104) = 0;
      *(_DWORD *)(v8 + 112) = 0;
      *(_QWORD *)(v8 + 120) = 0;
      goto LABEL_5;
    }
    memcpy(v9, a2, a3);
    v3 = a3 + *(_DWORD *)(v8 + 32);
    goto LABEL_4;
  }
LABEL_5:
  v10 = (unsigned __int64 *)a1[11];
  if ( v10 == (unsigned __int64 *)a1[12] )
  {
    v14 = (__int64)v10 - a1[10];
    v32 = (unsigned __int64 *)a1[10];
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
      v28 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v18 )
      {
        v29 = 0;
        v19 = 8;
        v34 = 0;
LABEL_24:
        v20 = (unsigned __int64 *)(v34 + v14);
        if ( v20 )
        {
          *v20 = v8;
          v8 = 0;
        }
        v21 = v32;
        if ( v10 == v32 )
        {
LABEL_39:
          v27 = (unsigned __int64)v32;
          if ( v32 )
          {
            v33 = v19;
            j_j___libc_free_0(v27);
            v19 = v33;
          }
          a1[11] = v19;
          a1[10] = v34;
          a1[12] = v29;
          goto LABEL_12;
        }
        for ( i = (_QWORD *)v34; ; i = (_QWORD *)v23 )
        {
          v24 = *v21;
          if ( i )
            break;
          if ( !v24 )
            goto LABEL_29;
          v25 = *(_QWORD *)(v24 + 56);
          if ( v25 != v24 + 72 )
          {
            v30 = *v21;
            _libc_free(v25);
            v24 = v30;
          }
          v26 = *(_QWORD *)(v24 + 24);
          if ( v26 != v24 + 40 )
          {
            v31 = v24;
            _libc_free(v26);
            v24 = v31;
          }
          ++v21;
          j_j___libc_free_0(v24);
          v23 = 8;
          if ( v10 == v21 )
          {
LABEL_38:
            v19 = (__int64)(i + 2);
            goto LABEL_39;
          }
LABEL_30:
          ;
        }
        *i = v24;
        *v21 = 0;
LABEL_29:
        ++v21;
        v23 = (__int64)(i + 1);
        if ( v10 == v21 )
          goto LABEL_38;
        goto LABEL_30;
      }
      if ( v18 > 0xFFFFFFFFFFFFFFFLL )
        v18 = 0xFFFFFFFFFFFFFFFLL;
      v28 = 8 * v18;
    }
    v34 = sub_22077B0(v28);
    v19 = v34 + 8;
    v29 = v34 + v28;
    goto LABEL_24;
  }
  if ( v10 )
  {
    *v10 = v8;
    a1[11] += 8LL;
    return *(_QWORD *)(a1[11] - 8LL);
  }
  a1[11] = 8;
LABEL_12:
  if ( v8 )
  {
    v12 = *(_QWORD *)(v8 + 56);
    if ( v12 != v8 + 72 )
      _libc_free(v12);
    v13 = *(_QWORD *)(v8 + 24);
    if ( v13 != v8 + 40 )
      _libc_free(v13);
    j_j___libc_free_0(v8);
  }
  return *(_QWORD *)(a1[11] - 8LL);
}
