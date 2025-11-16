// Function: sub_1BC1780
// Address: 0x1bc1780
//
void __fastcall sub_1BC1780(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  unsigned __int64 *v3; // r9
  unsigned __int64 v4; // rbx
  unsigned __int64 *v5; // r15
  unsigned __int64 *v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // r10
  _QWORD *v9; // rcx
  unsigned __int64 *v10; // rbx
  _QWORD *v11; // rdx
  _QWORD *i; // r14
  unsigned __int64 v13; // rax
  _QWORD *v14; // r13
  _QWORD *v15; // rbx
  __int64 v16; // rax
  unsigned __int64 *v17; // r14
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  unsigned __int64 *v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // rbx
  __int64 v24; // rax
  unsigned __int64 *v25; // rbx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  unsigned __int64 *v28; // r15
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // [rsp-50h] [rbp-50h]
  _QWORD *v32; // [rsp-48h] [rbp-48h]
  unsigned __int64 v33; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v34; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v35; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v36; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v37; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v38; // [rsp-48h] [rbp-48h]
  int v39; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 == a2 )
    return;
  v2 = (_QWORD *)(a2 + 16);
  v3 = *(unsigned __int64 **)a1;
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(unsigned __int64 **)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v6 = &v3[3 * v4];
    if ( v6 != v3 )
    {
      do
      {
        v7 = *(v6 - 1);
        v6 -= 3;
        if ( v7 != 0 && v7 != -8 && v7 != -16 )
          sub_1649B30(v6);
      }
      while ( v6 != v5 );
      v3 = *(unsigned __int64 **)a1;
    }
    if ( v3 != (unsigned __int64 *)(a1 + 16) )
      _libc_free((unsigned __int64)v3);
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v2;
    *(_QWORD *)(a2 + 8) = 0;
    return;
  }
  v8 = *(unsigned int *)(a2 + 8);
  v39 = *(_DWORD *)(a2 + 8);
  if ( v8 > v4 )
  {
    if ( v8 > *(unsigned int *)(a1 + 12) )
    {
      v17 = &v3[3 * v4];
      while ( v17 != v5 )
      {
        while ( 1 )
        {
          v18 = *(v17 - 1);
          v17 -= 3;
          if ( v18 == -8 || v18 == 0 || v18 == -16 )
            break;
          v33 = v8;
          sub_1649B30(v17);
          v8 = v33;
          if ( v17 == v5 )
            goto LABEL_33;
        }
      }
LABEL_33:
      *(_DWORD *)(a1 + 8) = 0;
      v4 = 0;
      sub_170B450(a1, v8);
      v2 = *(_QWORD **)a2;
      v8 = *(unsigned int *)(a2 + 8);
      v5 = *(unsigned __int64 **)a1;
      v9 = *(_QWORD **)a2;
    }
    else
    {
      v9 = (_QWORD *)(a2 + 16);
      if ( *(_DWORD *)(a1 + 8) )
      {
        v31 = 24 * v4;
        v4 *= 24LL;
        v28 = (unsigned __int64 *)((char *)v3 + v4);
        do
        {
          v29 = v3[2];
          v30 = v2[2];
          if ( v29 != v30 )
          {
            if ( v29 != -8 && v29 != 0 && v29 != -16 )
            {
              v37 = v3;
              sub_1649B30(v3);
              v30 = v2[2];
              v3 = v37;
            }
            v3[2] = v30;
            if ( v30 != -8 && v30 != 0 && v30 != -16 )
            {
              v38 = v3;
              sub_1649AC0(v3, *v2 & 0xFFFFFFFFFFFFFFF8LL);
              v3 = v38;
            }
          }
          v3 += 3;
          v2 += 3;
        }
        while ( v3 != v28 );
        v2 = *(_QWORD **)a2;
        v8 = *(unsigned int *)(a2 + 8);
        v5 = *(unsigned __int64 **)a1;
        v9 = (_QWORD *)(*(_QWORD *)a2 + v31);
      }
    }
    v10 = (unsigned __int64 *)((char *)v5 + v4);
    v11 = &v2[3 * v8];
    for ( i = v9; v11 != i; v10 += 3 )
    {
      if ( v10 )
      {
        *v10 = 6;
        v10[1] = 0;
        v13 = i[2];
        v10[2] = v13;
        if ( v13 != -8 && v13 != 0 && v13 != -16 )
        {
          v32 = v11;
          sub_1649AC0(v10, *i & 0xFFFFFFFFFFFFFFF8LL);
          v11 = v32;
        }
      }
      i += 3;
    }
    *(_DWORD *)(a1 + 8) = v39;
    v14 = *(_QWORD **)a2;
    v15 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
    if ( *(_QWORD **)a2 != v15 )
    {
      do
      {
        v16 = *(v15 - 1);
        v15 -= 3;
        if ( v16 != -8 && v16 != 0 && v16 != -16 )
          sub_1649B30(v15);
      }
      while ( v14 != v15 );
    }
    goto LABEL_26;
  }
  v19 = *(unsigned __int64 **)a1;
  if ( *(_DWORD *)(a2 + 8) )
  {
    v25 = &v3[3 * v8];
    do
    {
      v26 = v3[2];
      v27 = v2[2];
      if ( v26 != v27 )
      {
        if ( v26 != 0 && v26 != -8 && v26 != -16 )
        {
          v35 = v3;
          sub_1649B30(v3);
          v27 = v2[2];
          v3 = v35;
        }
        v3[2] = v27;
        if ( v27 != -8 && v27 != 0 && v27 != -16 )
        {
          v36 = v3;
          sub_1649AC0(v3, *v2 & 0xFFFFFFFFFFFFFFF8LL);
          v3 = v36;
        }
      }
      v3 += 3;
      v2 += 3;
    }
    while ( v3 != v25 );
    v19 = *(unsigned __int64 **)a1;
    v4 = *(unsigned int *)(a1 + 8);
  }
  v20 = &v19[3 * v4];
  while ( v3 != v20 )
  {
    v21 = *(v20 - 1);
    v20 -= 3;
    if ( v21 != -8 && v21 != 0 && v21 != -16 )
    {
      v34 = v3;
      sub_1649B30(v20);
      v3 = v34;
    }
  }
  *(_DWORD *)(a1 + 8) = v39;
  v22 = *(_QWORD **)a2;
  v23 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
  if ( *(_QWORD **)a2 == v23 )
  {
LABEL_26:
    *(_DWORD *)(a2 + 8) = 0;
    return;
  }
  do
  {
    v24 = *(v23 - 1);
    v23 -= 3;
    if ( v24 != 0 && v24 != -8 && v24 != -16 )
      sub_1649B30(v23);
  }
  while ( v22 != v23 );
  *(_DWORD *)(a2 + 8) = 0;
}
