// Function: sub_1E0BDD0
// Address: 0x1e0bdd0
//
void __fastcall sub_1E0BDD0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  const void *v3; // r8
  __int64 v4; // r15
  unsigned __int64 v5; // r13
  bool v6; // cf
  unsigned __int64 v7; // rax
  signed __int64 v8; // r9
  char *v9; // r14
  const void *v10; // r8
  signed __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // r9
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // r8
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  const void **v25; // rdi
  signed __int64 v26; // [rsp-50h] [rbp-50h]
  const void *v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-40h] [rbp-40h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  v14 = a1 + 40;
  v15 = a1[12];
  if ( a1 + 40 != (_QWORD *)(a1[40] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( !a2 || a2 == a1[41] )
    {
      a2 = a1[41];
      LODWORD(v16) = 0;
      if ( v14 == (_QWORD *)a2 )
      {
        v20 = a1[13];
        v16 = 0;
        v22 = (v20 - v15) >> 3;
LABEL_13:
        if ( v16 < v22 )
        {
          v23 = v15 + 8 * v16;
          if ( v20 != v23 )
            a1[13] = v23;
        }
        return;
      }
    }
    else
    {
      v16 = (unsigned int)(*(_DWORD *)((*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) + 48) + 1);
      if ( (_QWORD *)a2 == v14 )
        goto LABEL_12;
    }
    do
    {
      v17 = *(int *)(a2 + 48);
      if ( (_DWORD)v16 != (_DWORD)v17 )
      {
        if ( (_DWORD)v17 != -1 )
        {
          *(_QWORD *)(v15 + 8 * v17) = 0;
          v15 = a1[12];
        }
        v18 = 8LL * (unsigned int)v16;
        v19 = v18 + v15;
        if ( *(_QWORD *)v19 )
        {
          *(_DWORD *)(*(_QWORD *)v19 + 48LL) = -1;
          v19 = v18 + a1[12];
        }
        *(_QWORD *)v19 = a2;
        *(_DWORD *)(a2 + 48) = v16;
        v15 = a1[12];
      }
      a2 = *(_QWORD *)(a2 + 8);
      v16 = (unsigned int)(v16 + 1);
    }
    while ( v14 != (_QWORD *)a2 );
LABEL_12:
    v20 = a1[13];
    v21 = (v20 - v15) >> 3;
    v22 = v21;
    if ( v16 <= v21 )
      goto LABEL_13;
    v24 = v16 - v21;
    v25 = (const void **)(a1 + 12);
    if ( !v24 )
    {
      nullsub_2019();
      return;
    }
    v2 = v24;
    v3 = *v25;
    v4 = (_BYTE *)v25[1] - (_BYTE *)*v25;
    v5 = v4 >> 3;
    if ( ((_BYTE *)v25[2] - (_BYTE *)v25[1]) >> 3 >= v24 )
    {
      v25[1] = (char *)memset((void *)v25[1], 0, 8 * v24) + 8 * v24;
      return;
    }
    if ( 0xFFFFFFFFFFFFFFFLL - v5 < v24 )
      sub_4262D8((__int64)"vector::_M_default_append");
    if ( v5 >= v24 )
      v24 = ((_BYTE *)v25[1] - (_BYTE *)*v25) >> 3;
    v6 = __CFADD__(v5, v24);
    v7 = v5 + v24;
    if ( v6 )
    {
      v12 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v7 )
      {
        v28 = 0;
        v8 = (_BYTE *)v25[1] - (_BYTE *)*v25;
        v9 = 0;
        goto LABEL_30;
      }
      if ( v7 > 0xFFFFFFFFFFFFFFFLL )
        v7 = 0xFFFFFFFFFFFFFFFLL;
      v12 = 8 * v7;
    }
    v29 = v12;
    v13 = sub_22077B0(v12);
    v3 = *v25;
    v9 = (char *)v13;
    v28 = v29 + v13;
    v8 = (_BYTE *)v25[1] - (_BYTE *)*v25;
LABEL_30:
    v26 = v8;
    v27 = v3;
    memset(&v9[v4], 0, 8 * v2);
    v10 = v27;
    if ( v26 > 0 )
    {
      memmove(v9, v27, v26);
      v10 = v27;
      v11 = (_BYTE *)v25[2] - (_BYTE *)v27;
    }
    else
    {
      if ( !v27 )
      {
LABEL_32:
        *v25 = v9;
        v25[1] = &v9[8 * v5 + 8 * v2];
        v25[2] = (const void *)v28;
        return;
      }
      v11 = (_BYTE *)v25[2] - (_BYTE *)v27;
    }
    j_j___libc_free_0(v10, v11);
    goto LABEL_32;
  }
  if ( a1[13] != v15 )
    a1[13] = v15;
}
