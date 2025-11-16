// Function: sub_E60D70
// Address: 0xe60d70
//
void __fastcall sub_E60D70(__int64 *a1, unsigned __int64 a2)
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
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // [rsp-50h] [rbp-50h]
  __int64 v25; // [rsp-50h] [rbp-50h]
  unsigned __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = *a1;
  v6 = v4 - *a1;
  v26 = 0x6DB6DB6DB6DB6DB7LL * (v6 >> 3);
  if ( a2 <= 0x6DB6DB6DB6DB6DB7LL * ((a1[2] - v4) >> 3) )
  {
    v7 = a1[1];
    v8 = a2;
    do
    {
      if ( v7 )
      {
        *(_QWORD *)(v7 + 48) = 0;
        *(_OWORD *)(v7 + 16) = 0;
        *(_QWORD *)(v7 + 32) = 0;
        *(_QWORD *)(v7 + 24) = 0;
        *(_DWORD *)(v7 + 40) = 0;
        *(_DWORD *)(v7 + 44) = 0;
        *(_OWORD *)v7 = 0;
      }
      v7 += 56;
      --v8;
    }
    while ( v8 );
    a1[1] = v4 + 56 * a2;
    return;
  }
  if ( 0x249249249249249LL - v26 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v9 = 0x6DB6DB6DB6DB6DB7LL * ((v4 - *a1) >> 3);
  if ( a2 >= v26 )
    v9 = a2;
  v10 = __CFADD__(v26, v9);
  v11 = v26 + v9;
  if ( v10 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_35:
    v25 = v22;
    v23 = sub_22077B0(v22);
    v4 = a1[1];
    v27 = v23;
    v5 = *a1;
    v24 = v23 + v25;
    goto LABEL_14;
  }
  if ( v11 )
  {
    if ( v11 > 0x249249249249249LL )
      v11 = 0x249249249249249LL;
    v22 = 56 * v11;
    goto LABEL_35;
  }
  v24 = 0;
  v27 = 0;
LABEL_14:
  v12 = a2;
  v13 = v6 + v27;
  do
  {
    if ( v13 )
    {
      *(_QWORD *)(v13 + 48) = 0;
      *(_OWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 32) = 0;
      *(_QWORD *)(v13 + 24) = 0;
      *(_DWORD *)(v13 + 40) = 0;
      *(_DWORD *)(v13 + 44) = 0;
      *(_OWORD *)v13 = 0;
    }
    v13 += 56;
    --v12;
  }
  while ( v12 );
  if ( v4 != v5 )
  {
    v14 = v27;
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_21;
      *(_DWORD *)v14 = *(_DWORD *)v5;
      *(_QWORD *)(v14 + 4) = *(_QWORD *)(v5 + 4);
      *(_DWORD *)(v14 + 12) = *(_DWORD *)(v5 + 12);
      v17 = *(_QWORD *)(v5 + 16);
      *(_QWORD *)(v14 + 24) = 0;
      *(_QWORD *)(v14 + 16) = v17;
      *(_DWORD *)(v14 + 48) = 0;
      *(_QWORD *)(v14 + 32) = 0;
      *(_DWORD *)(v14 + 40) = 0;
      *(_DWORD *)(v14 + 44) = 0;
      sub_C7D6A0(0, 0, 4);
      v18 = *(unsigned int *)(v5 + 48);
      *(_DWORD *)(v14 + 48) = v18;
      if ( (_DWORD)v18 )
      {
        v15 = (void *)sub_C7D670(16 * v18, 4);
        v16 = *(unsigned int *)(v14 + 48);
        *(_QWORD *)(v14 + 32) = v15;
        *(_DWORD *)(v14 + 40) = *(_DWORD *)(v5 + 40);
        *(_DWORD *)(v14 + 44) = *(_DWORD *)(v5 + 44);
        memcpy(v15, *(const void **)(v5 + 32), 16 * v16);
LABEL_21:
        v5 += 56;
        v14 += 56;
        if ( v4 == v5 )
          goto LABEL_25;
      }
      else
      {
        v5 += 56;
        *(_QWORD *)(v14 + 32) = 0;
        v14 += 56;
        *(_DWORD *)(v14 - 16) = 0;
        *(_DWORD *)(v14 - 12) = 0;
        if ( v4 == v5 )
        {
LABEL_25:
          v19 = a1[1];
          v5 = *a1;
          if ( v19 != *a1 )
          {
            do
            {
              v20 = *(unsigned int *)(v5 + 48);
              v21 = *(_QWORD *)(v5 + 32);
              v5 += 56;
              sub_C7D6A0(v21, 16 * v20, 4);
            }
            while ( v19 != v5 );
            v5 = *a1;
          }
          break;
        }
      }
    }
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  *a1 = v27;
  a1[1] = v27 + 56 * (v26 + a2);
  a1[2] = v24;
}
