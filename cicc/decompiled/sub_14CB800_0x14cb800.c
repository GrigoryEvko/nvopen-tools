// Function: sub_14CB800
// Address: 0x14cb800
//
void __fastcall sub_14CB800(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r13
  unsigned __int64 v3; // r9
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // r10
  __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // r8
  __int64 *i; // r13
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // eax
  __int64 *v33; // [rsp-48h] [rbp-48h]
  unsigned __int64 v34; // [rsp-48h] [rbp-48h]
  unsigned __int64 v35; // [rsp-48h] [rbp-48h]
  unsigned __int64 v36; // [rsp-48h] [rbp-48h]
  unsigned __int64 v37; // [rsp-48h] [rbp-48h]
  unsigned __int64 v38; // [rsp-48h] [rbp-48h]
  unsigned __int64 v39; // [rsp-48h] [rbp-48h]
  int v40; // [rsp-3Ch] [rbp-3Ch]

  if ( (__int64 *)a1 == a2 )
    return;
  v2 = a2 + 2;
  v3 = *(_QWORD *)a1;
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(_QWORD *)a1;
  if ( (__int64 *)*a2 != a2 + 2 )
  {
    v6 = v3 + 32 * v4;
    if ( v6 != v3 )
    {
      do
      {
        v7 = *(_QWORD *)(v6 - 16);
        v6 -= 32LL;
        if ( v7 != 0 && v7 != -8 && v7 != -16 )
          sub_1649B30(v6);
      }
      while ( v6 != v5 );
      v3 = *(_QWORD *)a1;
    }
    if ( v3 != a1 + 16 )
      _libc_free(v3);
    *(_QWORD *)a1 = *a2;
    *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
    *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
    *a2 = (__int64)v2;
    a2[1] = 0;
    return;
  }
  v8 = *((unsigned int *)a2 + 2);
  v40 = *((_DWORD *)a2 + 2);
  if ( v8 > v4 )
  {
    if ( v8 > *(unsigned int *)(a1 + 12) )
    {
      v17 = v3 + 32 * v4;
      while ( v17 != v5 )
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(v17 - 16);
          v17 -= 32LL;
          if ( v18 == -8 || v18 == 0 || v18 == -16 )
            break;
          v34 = v8;
          sub_1649B30(v17);
          v8 = v34;
          if ( v17 == v5 )
            goto LABEL_34;
        }
      }
LABEL_34:
      *(_DWORD *)(a1 + 8) = 0;
      v4 = 0;
      sub_14CB640(a1, v8);
      v2 = (__int64 *)*a2;
      v8 = *((unsigned int *)a2 + 2);
      v5 = *(_QWORD *)a1;
      v9 = (__int64 *)*a2;
    }
    else
    {
      v9 = a2 + 2;
      if ( *(_DWORD *)(a1 + 8) )
      {
        v4 *= 32LL;
        v29 = v3 + v4;
        do
        {
          v30 = *(_QWORD *)(v3 + 16);
          v31 = v2[2];
          if ( v30 != v31 )
          {
            if ( v30 != 0 && v30 != -8 && v30 != -16 )
            {
              v38 = v3;
              sub_1649B30(v3);
              v31 = v2[2];
              v3 = v38;
            }
            *(_QWORD *)(v3 + 16) = v31;
            if ( v31 != -8 && v31 != 0 && v31 != -16 )
            {
              v39 = v3;
              sub_1649AC0(v3, *v2 & 0xFFFFFFFFFFFFFFF8LL);
              v3 = v39;
            }
          }
          v32 = *((_DWORD *)v2 + 6);
          v3 += 32LL;
          v2 += 4;
          *(_DWORD *)(v3 - 8) = v32;
        }
        while ( v3 != v29 );
        v2 = (__int64 *)*a2;
        v8 = *((unsigned int *)a2 + 2);
        v5 = *(_QWORD *)a1;
        v9 = (__int64 *)(*a2 + v4);
      }
    }
    v10 = v5 + v4;
    v11 = &v2[4 * v8];
    for ( i = v9; v11 != i; v10 += 32 )
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = 6;
        *(_QWORD *)(v10 + 8) = 0;
        v13 = i[2];
        *(_QWORD *)(v10 + 16) = v13;
        if ( v13 != -8 && v13 != 0 && v13 != -16 )
        {
          v33 = v11;
          sub_1649AC0(v10, *i & 0xFFFFFFFFFFFFFFF8LL);
          v11 = v33;
        }
        *(_DWORD *)(v10 + 24) = *((_DWORD *)i + 6);
      }
      i += 4;
    }
    *(_DWORD *)(a1 + 8) = v40;
    v14 = *a2;
    v15 = *a2 + 32LL * *((unsigned int *)a2 + 2);
    if ( *a2 != v15 )
    {
      do
      {
        v16 = *(_QWORD *)(v15 - 16);
        v15 -= 32;
        if ( v16 != -8 && v16 != 0 && v16 != -16 )
          sub_1649B30(v15);
      }
      while ( v14 != v15 );
    }
    goto LABEL_27;
  }
  v19 = *(_QWORD *)a1;
  if ( *((_DWORD *)a2 + 2) )
  {
    v25 = v3 + 32 * v8;
    do
    {
      v26 = *(_QWORD *)(v3 + 16);
      v27 = v2[2];
      if ( v26 != v27 )
      {
        if ( v26 != 0 && v26 != -8 && v26 != -16 )
        {
          v36 = v3;
          sub_1649B30(v3);
          v27 = v2[2];
          v3 = v36;
        }
        *(_QWORD *)(v3 + 16) = v27;
        if ( v27 != -8 && v27 != 0 && v27 != -16 )
        {
          v37 = v3;
          sub_1649AC0(v3, *v2 & 0xFFFFFFFFFFFFFFF8LL);
          v3 = v37;
        }
      }
      v28 = *((_DWORD *)v2 + 6);
      v3 += 32LL;
      v2 += 4;
      *(_DWORD *)(v3 - 8) = v28;
    }
    while ( v3 != v25 );
    v19 = *(_QWORD *)a1;
    v4 = *(unsigned int *)(a1 + 8);
  }
  v20 = v19 + 32 * v4;
  while ( v3 != v20 )
  {
    v21 = *(_QWORD *)(v20 - 16);
    v20 -= 32LL;
    if ( v21 != -8 && v21 != 0 && v21 != -16 )
    {
      v35 = v3;
      sub_1649B30(v20);
      v3 = v35;
    }
  }
  *(_DWORD *)(a1 + 8) = v40;
  v22 = *a2;
  v23 = *a2 + 32LL * *((unsigned int *)a2 + 2);
  if ( *a2 == v23 )
  {
LABEL_27:
    *((_DWORD *)a2 + 2) = 0;
    return;
  }
  do
  {
    v24 = *(_QWORD *)(v23 - 16);
    v23 -= 32;
    if ( v24 != 0 && v24 != -8 && v24 != -16 )
      sub_1649B30(v23);
  }
  while ( v22 != v23 );
  *((_DWORD *)a2 + 2) = 0;
}
