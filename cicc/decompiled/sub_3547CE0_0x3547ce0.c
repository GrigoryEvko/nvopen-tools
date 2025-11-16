// Function: sub_3547CE0
// Address: 0x3547ce0
//
void __fastcall sub_3547CE0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned __int64 v8; // rbx
  unsigned __int64 *v9; // r12
  __int64 v10; // r15
  __int64 v11; // r13
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  _QWORD *v16; // r13
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned __int64 *v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // [rsp-50h] [rbp-50h]
  unsigned __int64 v24; // [rsp-50h] [rbp-50h]
  __int64 v25; // [rsp-48h] [rbp-48h]
  unsigned __int64 v26; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v6 = 0x8E38E38E38E38E39LL;
  v8 = a2;
  v9 = (unsigned __int64 *)a1[1];
  v10 = *a1;
  v11 = (__int64)v9 - *a1;
  v26 = 0x8E38E38E38E38E39LL * (v11 >> 5);
  if ( a2 <= 0x8E38E38E38E38E39LL * ((a1[2] - (__int64)v9) >> 5) )
  {
    v12 = (_QWORD *)a1[1];
    do
    {
      if ( v12 )
      {
        memset(v12, 0, 0x120u);
        *((_DWORD *)v12 + 3) = 4;
        *v12 = v12 + 2;
        v12[18] = v12 + 20;
        *((_DWORD *)v12 + 39) = 4;
      }
      v12 += 36;
      --a2;
    }
    while ( a2 );
    a1[1] = (__int64)&v9[36 * v8];
    return;
  }
  if ( 0x71C71C71C71C71LL - v26 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v13 = 0x8E38E38E38E38E39LL * ((a1[1] - *a1) >> 5);
  if ( a2 >= v26 )
    v13 = a2;
  v14 = __CFADD__(v26, v13);
  v15 = v26 + v13;
  if ( v14 )
  {
    v21 = 0x7FFFFFFFFFFFFF20LL;
  }
  else
  {
    if ( !v15 )
    {
      v23 = 0;
      v25 = 0;
      goto LABEL_15;
    }
    if ( v15 > 0x71C71C71C71C71LL )
      v15 = 0x71C71C71C71C71LL;
    v21 = 288 * v15;
  }
  v24 = v21;
  v22 = sub_22077B0(v21);
  v9 = (unsigned __int64 *)a1[1];
  v25 = v22;
  v10 = *a1;
  v23 = v22 + v24;
LABEL_15:
  v16 = (_QWORD *)(v25 + v11);
  do
  {
    if ( v16 )
    {
      memset(v16, 0, 0x120u);
      *((_DWORD *)v16 + 3) = 4;
      *v16 = v16 + 2;
      v6 = (__int64)(v16 + 20);
      v16[18] = v16 + 20;
      *((_DWORD *)v16 + 39) = 4;
    }
    v16 += 36;
    --a2;
  }
  while ( a2 );
  if ( (unsigned __int64 *)v10 != v9 )
  {
    v17 = v25;
    do
    {
      if ( v17 )
      {
        *(_DWORD *)(v17 + 8) = 0;
        *(_QWORD *)v17 = v17 + 16;
        *(_DWORD *)(v17 + 12) = 4;
        v18 = *(unsigned int *)(v10 + 8);
        if ( (_DWORD)v18 )
          sub_353D740(v17, v10, v18, v6, a5, a6);
        *(_DWORD *)(v17 + 152) = 0;
        *(_QWORD *)(v17 + 144) = v17 + 160;
        *(_DWORD *)(v17 + 156) = 4;
        if ( *(_DWORD *)(v10 + 152) )
          sub_353D740(v17 + 144, v10 + 144, v18, v6, a5, a6);
      }
      v10 += 288;
      v17 += 288;
    }
    while ( (unsigned __int64 *)v10 != v9 );
    v19 = (unsigned __int64 *)a1[1];
    v9 = (unsigned __int64 *)*a1;
    if ( v19 != (unsigned __int64 *)*a1 )
    {
      do
      {
        v20 = v9[18];
        if ( (unsigned __int64 *)v20 != v9 + 20 )
          _libc_free(v20);
        if ( (unsigned __int64 *)*v9 != v9 + 2 )
          _libc_free(*v9);
        v9 += 36;
      }
      while ( v19 != v9 );
      v9 = (unsigned __int64 *)*a1;
    }
  }
  if ( v9 )
    j_j___libc_free_0((unsigned __int64)v9);
  *a1 = v25;
  a1[1] = v25 + 288 * (v26 + v8);
  a1[2] = v23;
}
