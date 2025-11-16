// Function: sub_FF1200
// Address: 0xff1200
//
void __fastcall sub_FF1200(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r14
  void *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // [rsp-50h] [rbp-50h]
  __int64 v24; // [rsp-50h] [rbp-50h]
  unsigned __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = *a1;
  v6 = v4 - *a1;
  v25 = v6 >> 5;
  if ( a2 <= (a1[2] - v4) >> 5 )
  {
    v7 = a1[1];
    v8 = a2;
    do
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = 0;
        *(_DWORD *)(v7 + 24) = 0;
        *(_QWORD *)(v7 + 8) = 0;
        *(_DWORD *)(v7 + 16) = 0;
        *(_DWORD *)(v7 + 20) = 0;
      }
      v7 += 32;
      --v8;
    }
    while ( v8 );
    a1[1] = v4 + 32 * a2;
    return;
  }
  if ( 0x3FFFFFFFFFFFFFFLL - v25 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v9 = (a1[1] - *a1) >> 5;
  if ( a2 >= v25 )
    v9 = a2;
  v10 = __CFADD__(v25, v9);
  v11 = v25 + v9;
  if ( v10 )
  {
    v21 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_35:
    v24 = v21;
    v22 = sub_22077B0(v21);
    v4 = a1[1];
    v26 = v22;
    v5 = *a1;
    v23 = v22 + v24;
    goto LABEL_14;
  }
  if ( v11 )
  {
    if ( v11 > 0x3FFFFFFFFFFFFFFLL )
      v11 = 0x3FFFFFFFFFFFFFFLL;
    v21 = 32 * v11;
    goto LABEL_35;
  }
  v23 = 0;
  v26 = 0;
LABEL_14:
  v12 = a2;
  v13 = v6 + v26;
  do
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = 0;
      *(_DWORD *)(v13 + 24) = 0;
      *(_QWORD *)(v13 + 8) = 0;
      *(_DWORD *)(v13 + 16) = 0;
      *(_DWORD *)(v13 + 20) = 0;
    }
    v13 += 32;
    --v12;
  }
  while ( v12 );
  if ( v4 != v5 )
  {
    v14 = v26;
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_21;
      *(_QWORD *)v14 = 0;
      *(_DWORD *)(v14 + 24) = 0;
      *(_QWORD *)(v14 + 8) = 0;
      *(_DWORD *)(v14 + 16) = 0;
      *(_DWORD *)(v14 + 20) = 0;
      sub_C7D6A0(0, 0, 8);
      v17 = *(unsigned int *)(v5 + 24);
      *(_DWORD *)(v14 + 24) = v17;
      if ( (_DWORD)v17 )
      {
        v15 = (void *)sub_C7D670(16 * v17, 8);
        v16 = *(unsigned int *)(v14 + 24);
        *(_QWORD *)(v14 + 8) = v15;
        *(_DWORD *)(v14 + 16) = *(_DWORD *)(v5 + 16);
        *(_DWORD *)(v14 + 20) = *(_DWORD *)(v5 + 20);
        memcpy(v15, *(const void **)(v5 + 8), 16 * v16);
LABEL_21:
        v5 += 32;
        v14 += 32;
        if ( v5 == v4 )
          goto LABEL_25;
      }
      else
      {
        v5 += 32;
        *(_QWORD *)(v14 + 8) = 0;
        v14 += 32;
        *(_DWORD *)(v14 - 16) = 0;
        *(_DWORD *)(v14 - 12) = 0;
        if ( v5 == v4 )
        {
LABEL_25:
          v18 = a1[1];
          v5 = *a1;
          if ( v18 != *a1 )
          {
            do
            {
              v19 = *(unsigned int *)(v5 + 24);
              v20 = *(_QWORD *)(v5 + 8);
              v5 += 32;
              sub_C7D6A0(v20, 16 * v19, 8);
            }
            while ( v18 != v5 );
            v5 = *a1;
          }
          break;
        }
      }
    }
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  *a1 = v26;
  a1[1] = v26 + 32 * (a2 + v25);
  a1[2] = v23;
}
