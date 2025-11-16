// Function: sub_3911920
// Address: 0x3911920
//
void __fastcall sub_3911920(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r14
  void *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // [rsp-50h] [rbp-50h]
  unsigned __int64 v23; // [rsp-50h] [rbp-50h]
  unsigned __int64 v24; // [rsp-48h] [rbp-48h]
  __int64 v25; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v3 = a1[1];
  v4 = *a1;
  v5 = v3 - *a1;
  v24 = 0x6DB6DB6DB6DB6DB7LL * (v5 >> 3);
  if ( 0x6DB6DB6DB6DB6DB7LL * ((__int64)(a1[2] - v3) >> 3) >= a2 )
  {
    v6 = a1[1];
    v7 = a2;
    do
    {
      if ( v6 )
      {
        *(_QWORD *)(v6 + 48) = 0;
        *(_OWORD *)(v6 + 16) = 0;
        *(_QWORD *)(v6 + 32) = 0;
        *(_QWORD *)(v6 + 24) = 0;
        *(_DWORD *)(v6 + 40) = 0;
        *(_DWORD *)(v6 + 44) = 0;
        *(_OWORD *)v6 = 0;
      }
      v6 += 56LL;
      --v7;
    }
    while ( v7 );
    a1[1] = v3 + 56 * a2;
    return;
  }
  if ( 0x249249249249249LL - v24 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v3 - *a1) >> 3);
  if ( a2 >= v24 )
    v8 = a2;
  v9 = __CFADD__(v24, v8);
  v10 = v24 + v8;
  if ( v9 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_35:
    v23 = v20;
    v21 = sub_22077B0(v20);
    v3 = a1[1];
    v25 = v21;
    v4 = *a1;
    v22 = v21 + v23;
    goto LABEL_14;
  }
  if ( v10 )
  {
    if ( v10 > 0x249249249249249LL )
      v10 = 0x249249249249249LL;
    v20 = 56 * v10;
    goto LABEL_35;
  }
  v22 = 0;
  v25 = 0;
LABEL_14:
  v11 = a2;
  v12 = v5 + v25;
  do
  {
    if ( v12 )
    {
      *(_QWORD *)(v12 + 48) = 0;
      *(_OWORD *)(v12 + 16) = 0;
      *(_QWORD *)(v12 + 32) = 0;
      *(_QWORD *)(v12 + 24) = 0;
      *(_DWORD *)(v12 + 40) = 0;
      *(_DWORD *)(v12 + 44) = 0;
      *(_OWORD *)v12 = 0;
    }
    v12 += 56;
    --v11;
  }
  while ( v11 );
  if ( v4 != v3 )
  {
    v13 = v25;
    while ( 1 )
    {
      if ( !v13 )
        goto LABEL_21;
      *(_DWORD *)v13 = *(_DWORD *)v4;
      *(_QWORD *)(v13 + 4) = *(_QWORD *)(v4 + 4);
      *(_DWORD *)(v13 + 12) = *(_DWORD *)(v4 + 12);
      v16 = *(_QWORD *)(v4 + 16);
      *(_QWORD *)(v13 + 24) = 0;
      *(_QWORD *)(v13 + 16) = v16;
      *(_DWORD *)(v13 + 48) = 0;
      *(_QWORD *)(v13 + 32) = 0;
      *(_DWORD *)(v13 + 40) = 0;
      *(_DWORD *)(v13 + 44) = 0;
      j___libc_free_0(0);
      v17 = *(unsigned int *)(v4 + 48);
      *(_DWORD *)(v13 + 48) = v17;
      if ( (_DWORD)v17 )
      {
        v14 = (void *)sub_22077B0(16 * v17);
        v15 = *(unsigned int *)(v13 + 48);
        *(_QWORD *)(v13 + 32) = v14;
        *(_DWORD *)(v13 + 40) = *(_DWORD *)(v4 + 40);
        *(_DWORD *)(v13 + 44) = *(_DWORD *)(v4 + 44);
        memcpy(v14, *(const void **)(v4 + 32), 16 * v15);
LABEL_21:
        v4 += 56LL;
        v13 += 56;
        if ( v4 == v3 )
          goto LABEL_25;
      }
      else
      {
        v4 += 56LL;
        *(_QWORD *)(v13 + 32) = 0;
        v13 += 56;
        *(_DWORD *)(v13 - 16) = 0;
        *(_DWORD *)(v13 - 12) = 0;
        if ( v4 == v3 )
        {
LABEL_25:
          v18 = a1[1];
          v3 = *a1;
          if ( v18 != *a1 )
          {
            do
            {
              v19 = *(_QWORD *)(v3 + 32);
              v3 += 56LL;
              j___libc_free_0(v19);
            }
            while ( v18 != v3 );
            v3 = *a1;
          }
          break;
        }
      }
    }
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  *a1 = v25;
  a1[1] = v25 + 56 * (v24 + a2);
  a1[2] = v22;
}
