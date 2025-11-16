// Function: sub_1378660
// Address: 0x1378660
//
void __fastcall sub_1378660(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r14
  void *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-50h] [rbp-50h]
  unsigned __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v3 = a1[1];
  v4 = *a1;
  v5 = v3 - *a1;
  v23 = v5 >> 5;
  if ( (a1[2] - v3) >> 5 >= a2 )
  {
    v6 = a1[1];
    v7 = a2;
    do
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = 0;
        *(_DWORD *)(v6 + 24) = 0;
        *(_QWORD *)(v6 + 8) = 0;
        *(_DWORD *)(v6 + 16) = 0;
        *(_DWORD *)(v6 + 20) = 0;
      }
      v6 += 32;
      --v7;
    }
    while ( v7 );
    a1[1] = v3 + 32 * a2;
    return;
  }
  if ( 0x3FFFFFFFFFFFFFFLL - v23 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = (a1[1] - *a1) >> 5;
  if ( a2 >= v23 )
    v8 = a2;
  v9 = __CFADD__(v23, v8);
  v10 = v23 + v8;
  if ( v9 )
  {
    v19 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_35:
    v22 = v19;
    v20 = sub_22077B0(v19);
    v3 = a1[1];
    v24 = v20;
    v4 = *a1;
    v21 = v20 + v22;
    goto LABEL_14;
  }
  if ( v10 )
  {
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v19 = 32 * v10;
    goto LABEL_35;
  }
  v21 = 0;
  v24 = 0;
LABEL_14:
  v11 = a2;
  v12 = v5 + v24;
  do
  {
    if ( v12 )
    {
      *(_QWORD *)v12 = 0;
      *(_DWORD *)(v12 + 24) = 0;
      *(_QWORD *)(v12 + 8) = 0;
      *(_DWORD *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 20) = 0;
    }
    v12 += 32;
    --v11;
  }
  while ( v11 );
  if ( v4 != v3 )
  {
    v13 = v24;
    while ( 1 )
    {
      if ( !v13 )
        goto LABEL_21;
      *(_QWORD *)v13 = 0;
      *(_DWORD *)(v13 + 24) = 0;
      *(_QWORD *)(v13 + 8) = 0;
      *(_DWORD *)(v13 + 16) = 0;
      *(_DWORD *)(v13 + 20) = 0;
      j___libc_free_0(0);
      v16 = *(unsigned int *)(v4 + 24);
      *(_DWORD *)(v13 + 24) = v16;
      if ( (_DWORD)v16 )
      {
        v14 = (void *)sub_22077B0(16 * v16);
        v15 = *(unsigned int *)(v13 + 24);
        *(_QWORD *)(v13 + 8) = v14;
        *(_DWORD *)(v13 + 16) = *(_DWORD *)(v4 + 16);
        *(_DWORD *)(v13 + 20) = *(_DWORD *)(v4 + 20);
        memcpy(v14, *(const void **)(v4 + 8), 16 * v15);
LABEL_21:
        v4 += 32;
        v13 += 32;
        if ( v4 == v3 )
          goto LABEL_25;
      }
      else
      {
        v4 += 32;
        *(_QWORD *)(v13 + 8) = 0;
        v13 += 32;
        *(_DWORD *)(v13 - 16) = 0;
        *(_DWORD *)(v13 - 12) = 0;
        if ( v4 == v3 )
        {
LABEL_25:
          v17 = a1[1];
          v3 = *a1;
          if ( v17 != *a1 )
          {
            do
            {
              v18 = *(_QWORD *)(v3 + 8);
              v3 += 32;
              j___libc_free_0(v18);
            }
            while ( v17 != v3 );
            v3 = *a1;
          }
          break;
        }
      }
    }
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[2] - v3);
  *a1 = v24;
  a1[1] = v24 + 32 * (a2 + v23);
  a1[2] = v21;
}
