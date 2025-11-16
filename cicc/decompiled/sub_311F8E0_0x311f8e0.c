// Function: sub_311F8E0
// Address: 0x311f8e0
//
void __fastcall sub_311F8E0(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  _QWORD *v4; // r13
  _QWORD *v5; // r15
  __int64 v6; // r14
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // r9
  int v14; // eax
  void *v15; // rdi
  unsigned int v16; // r10d
  size_t v17; // rdx
  _QWORD *v18; // r14
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // [rsp-54h] [rbp-54h]
  unsigned int v25; // [rsp-54h] [rbp-54h]
  unsigned __int64 v26; // [rsp-50h] [rbp-50h]
  unsigned __int64 v27; // [rsp-50h] [rbp-50h]
  __int64 v28; // [rsp-48h] [rbp-48h]
  unsigned __int64 v29; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v3 = a2;
  v4 = (_QWORD *)a1[1];
  v5 = (_QWORD *)*a1;
  v6 = (__int64)v4 - *a1;
  v29 = 0x8E38E38E38E38E39LL * (v6 >> 4);
  if ( a2 <= 0x8E38E38E38E38E39LL * ((a1[2] - (__int64)v4) >> 4) )
  {
    v7 = a1[1];
    do
    {
      if ( v7 )
      {
        memset((void *)v7, 0, 0x90u);
        *(_DWORD *)(v7 + 92) = 3;
        *(_QWORD *)(v7 + 8) = v7 + 24;
        *(_QWORD *)(v7 + 40) = v7 + 56;
        *(_QWORD *)(v7 + 80) = v7 + 96;
      }
      v7 += 144;
      --a2;
    }
    while ( a2 );
    a1[1] = (__int64)&v4[18 * v3];
    return;
  }
  if ( 0xE38E38E38E38E3LL - v29 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = 0x8E38E38E38E38E39LL * ((a1[1] - *a1) >> 4);
  if ( a2 >= v29 )
    v8 = a2;
  v9 = __CFADD__(v29, v8);
  v10 = v29 + v8;
  if ( v9 )
  {
    v22 = 0x7FFFFFFFFFFFFFB0LL;
  }
  else
  {
    if ( !v10 )
    {
      v26 = 0;
      v28 = 0;
      goto LABEL_15;
    }
    if ( v10 > 0xE38E38E38E38E3LL )
      v10 = 0xE38E38E38E38E3LL;
    v22 = 144 * v10;
  }
  v27 = v22;
  v23 = sub_22077B0(v22);
  v4 = (_QWORD *)a1[1];
  v28 = v23;
  v5 = (_QWORD *)*a1;
  v26 = v23 + v27;
LABEL_15:
  v11 = v28 + v6;
  do
  {
    if ( v11 )
    {
      memset((void *)v11, 0, 0x90u);
      *(_DWORD *)(v11 + 92) = 3;
      *(_QWORD *)(v11 + 8) = v11 + 24;
      *(_QWORD *)(v11 + 40) = v11 + 56;
      *(_QWORD *)(v11 + 80) = v11 + 96;
    }
    v11 += 144;
    --a2;
  }
  while ( a2 );
  if ( v5 != v4 )
  {
    v12 = v28;
    do
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = *v5;
        *(_QWORD *)(v12 + 8) = v12 + 24;
        sub_311CD80((__int64 *)(v12 + 8), (_BYTE *)v5[1], v5[1] + v5[2]);
        *(_QWORD *)(v12 + 40) = v12 + 56;
        sub_311CD80((__int64 *)(v12 + 40), (_BYTE *)v5[5], v5[5] + v5[6]);
        v14 = *((_DWORD *)v5 + 18);
        v15 = (void *)(v12 + 96);
        *(_DWORD *)(v12 + 88) = 0;
        *(_QWORD *)(v12 + 80) = v12 + 96;
        *(_DWORD *)(v12 + 72) = v14;
        *(_DWORD *)(v12 + 92) = 3;
        v16 = *((_DWORD *)v5 + 22);
        if ( v16 )
        {
          if ( (_QWORD *)(v12 + 80) != v5 + 10 )
          {
            v17 = 16LL * v16;
            if ( v16 <= 3
              || (v25 = *((_DWORD *)v5 + 22),
                  sub_C8D5F0(v12 + 80, (const void *)(v12 + 96), v16, 0x10u, v16, v13),
                  v15 = *(void **)(v12 + 80),
                  v16 = v25,
                  (v17 = 16LL * *((unsigned int *)v5 + 22)) != 0) )
            {
              v24 = v16;
              memcpy(v15, (const void *)v5[10], v17);
              v16 = v24;
            }
            *(_DWORD *)(v12 + 88) = v16;
          }
        }
      }
      v5 += 18;
      v12 += 144;
    }
    while ( v4 != v5 );
    v18 = (_QWORD *)a1[1];
    v4 = (_QWORD *)*a1;
    if ( v18 != (_QWORD *)*a1 )
    {
      do
      {
        v19 = v4[10];
        if ( (_QWORD *)v19 != v4 + 12 )
          _libc_free(v19);
        v20 = v4[5];
        if ( (_QWORD *)v20 != v4 + 7 )
          j_j___libc_free_0(v20);
        v21 = v4[1];
        if ( (_QWORD *)v21 != v4 + 3 )
          j_j___libc_free_0(v21);
        v4 += 18;
      }
      while ( v18 != v4 );
      v4 = (_QWORD *)*a1;
    }
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  *a1 = v28;
  a1[1] = v28 + 144 * (v29 + v3);
  a1[2] = v26;
}
