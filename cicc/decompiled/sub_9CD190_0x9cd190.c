// Function: sub_9CD190
// Address: 0x9cd190
//
void __fastcall sub_9CD190(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // r13
  __int64 v7; // r13
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // r14
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // [rsp-50h] [rbp-50h]
  __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-48h] [rbp-48h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = v4 - *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 4);
  if ( a2 <= 0xAAAAAAAAAAAAAAABLL * ((a1[2] - v4) >> 4) )
  {
    v7 = a1[1];
    v8 = a2;
    do
    {
      if ( v7 )
      {
        *(_OWORD *)v7 = 0;
        *(_QWORD *)(v7 + 8) = 0;
        *(_OWORD *)(v7 + 16) = 0;
        *(_OWORD *)(v7 + 32) = 0;
        sub_AADB10(v7 + 16, 64, 1);
      }
      v7 += 48;
      --v8;
    }
    while ( v8 );
    a1[1] = 48 * a2 + v4;
    return;
  }
  if ( 0x2AAAAAAAAAAAAAALL - v6 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v9 = 0xAAAAAAAAAAAAAAABLL * ((a1[1] - *a1) >> 4);
  if ( a2 >= v6 )
    v9 = a2;
  v10 = __CFADD__(v6, v9);
  v11 = v6 + v9;
  if ( v10 )
  {
    v24 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_43:
    v28 = v24;
    v29 = sub_22077B0(v24);
    v25 = v29 + v28;
    goto LABEL_14;
  }
  if ( v11 )
  {
    if ( v11 > 0x2AAAAAAAAAAAAAALL )
      v11 = 0x2AAAAAAAAAAAAAALL;
    v24 = 48 * v11;
    goto LABEL_43;
  }
  v25 = 0;
  v29 = 0;
LABEL_14:
  v12 = v29 + v5;
  v13 = a2;
  do
  {
    if ( v12 )
    {
      *(_OWORD *)v12 = 0;
      *(_QWORD *)(v12 + 8) = 0;
      *(_OWORD *)(v12 + 16) = 0;
      *(_OWORD *)(v12 + 32) = 0;
      sub_AADB10(v12 + 16, 64, 1);
    }
    v12 += 48;
    --v13;
  }
  while ( v13 );
  v14 = a1[1];
  v15 = *a1;
  if ( v14 != *a1 )
  {
    v16 = v29;
    while ( 1 )
    {
      if ( !v16 )
        goto LABEL_22;
      *(_QWORD *)v16 = *(_QWORD *)v15;
      *(_QWORD *)(v16 + 8) = *(_QWORD *)(v15 + 8);
      v18 = *(_DWORD *)(v15 + 24);
      *(_DWORD *)(v16 + 24) = v18;
      if ( v18 > 0x40 )
        break;
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(v15 + 16);
      v17 = *(_DWORD *)(v15 + 40);
      *(_DWORD *)(v16 + 40) = v17;
      if ( v17 > 0x40 )
      {
LABEL_26:
        v20 = v15 + 32;
        v27 = v14;
        v15 += 48;
        sub_C43780(v16 + 32, v20);
        v14 = v27;
        v16 += 48;
        if ( v27 == v15 )
        {
LABEL_27:
          v21 = a1[1];
          v15 = *a1;
          if ( v21 != *a1 )
          {
            do
            {
              if ( *(_DWORD *)(v15 + 40) > 0x40u )
              {
                v22 = *(_QWORD *)(v15 + 32);
                if ( v22 )
                  j_j___libc_free_0_0(v22);
              }
              if ( *(_DWORD *)(v15 + 24) > 0x40u )
              {
                v23 = *(_QWORD *)(v15 + 16);
                if ( v23 )
                  j_j___libc_free_0_0(v23);
              }
              v15 += 48;
            }
            while ( v21 != v15 );
            v15 = *a1;
          }
          goto LABEL_36;
        }
      }
      else
      {
LABEL_21:
        *(_QWORD *)(v16 + 32) = *(_QWORD *)(v15 + 32);
LABEL_22:
        v15 += 48;
        v16 += 48;
        if ( v14 == v15 )
          goto LABEL_27;
      }
    }
    v26 = v14;
    sub_C43780(v16 + 16, v15 + 16);
    v19 = *(_DWORD *)(v15 + 40);
    v14 = v26;
    *(_DWORD *)(v16 + 40) = v19;
    if ( v19 > 0x40 )
      goto LABEL_26;
    goto LABEL_21;
  }
LABEL_36:
  if ( v15 )
    j_j___libc_free_0(v15, a1[2] - v15);
  *a1 = v29;
  a1[1] = v29 + 48 * (v6 + a2);
  a1[2] = v25;
}
