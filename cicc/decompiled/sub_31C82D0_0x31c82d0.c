// Function: sub_31C82D0
// Address: 0x31c82d0
//
signed __int64 __fastcall sub_31C82D0(char *a1, char *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 v4; // r14
  char *v6; // r12
  char *v7; // r11
  __int64 *v8; // rbx
  __int64 v9; // r15
  char *v10; // r8
  __int64 v11; // rsi
  unsigned int v12; // edx
  unsigned int v13; // eax
  bool v14; // r9
  __int64 v15; // rdi
  unsigned int v16; // ecx
  bool v17; // r9
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rsi
  __int64 *v21; // r9
  char *v22; // r8
  unsigned int v23; // ecx
  unsigned int v24; // eax
  bool v25; // dl
  char *i; // rax
  unsigned int v27; // edx
  bool v28; // r9
  bool v29; // si
  bool v30; // dl
  __int64 v31; // r14
  __int64 j; // rbx
  __int64 *v33; // r12
  __int64 v34; // rcx
  __int64 v35; // rbx

  result = a2 - a1;
  if ( a2 - a1 <= 128 )
    return result;
  v4 = a3;
  v6 = a2;
  if ( !a3 )
    goto LABEL_38;
  v7 = a2;
  v8 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v9 = *((_QWORD *)a1 + 1);
    --v4;
    v10 = &a1[8 * ((__int64)(((v7 - a1) >> 3) + ((unsigned __int64)(v7 - a1) >> 63)) >> 1)];
    v11 = *(_QWORD *)v10;
    v12 = *(_DWORD *)(*(_QWORD *)(v9 + 56) + 72LL);
    v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 56LL) + 72LL);
    v14 = v12 < v13;
    if ( v12 == v13 )
      v14 = *(_DWORD *)(v9 + 64) < *(_DWORD *)(v11 + 64);
    v15 = *((_QWORD *)v7 - 1);
    v16 = *(_DWORD *)(*(_QWORD *)(v15 + 56) + 72LL);
    if ( v14 )
    {
      v17 = v16 > v13;
      if ( v16 == v13 )
        v17 = *(_DWORD *)(v11 + 64) < *(_DWORD *)(v15 + 64);
      v18 = *(_QWORD *)a1;
      if ( v17 )
      {
        *(_QWORD *)a1 = v11;
        *(_QWORD *)v10 = v18;
        v9 = *(_QWORD *)a1;
        v19 = *((_QWORD *)a1 + 1);
        v20 = *((_QWORD *)v7 - 1);
      }
      else
      {
        v29 = v16 > v12;
        if ( v16 == v12 )
          v29 = *(_DWORD *)(v9 + 64) < *(_DWORD *)(v15 + 64);
        if ( v29 )
        {
          *(_QWORD *)a1 = v15;
          v20 = v18;
          *((_QWORD *)v7 - 1) = v18;
          v9 = *(_QWORD *)a1;
          v19 = *((_QWORD *)a1 + 1);
        }
        else
        {
          *(_QWORD *)a1 = v9;
          v19 = v18;
          *((_QWORD *)a1 + 1) = v18;
          v20 = *((_QWORD *)v7 - 1);
        }
      }
    }
    else
    {
      v28 = v16 > v12;
      if ( v16 == v12 )
        v28 = *(_DWORD *)(v9 + 64) < *(_DWORD *)(v15 + 64);
      v19 = *(_QWORD *)a1;
      if ( v28 )
      {
        *(_QWORD *)a1 = v9;
        *((_QWORD *)a1 + 1) = v19;
        v20 = *((_QWORD *)v7 - 1);
      }
      else
      {
        v30 = v16 > v13;
        if ( v16 == v13 )
          v30 = *(_DWORD *)(v11 + 64) < *(_DWORD *)(v15 + 64);
        if ( v30 )
        {
          *(_QWORD *)a1 = v15;
          v20 = v19;
          *((_QWORD *)v7 - 1) = v19;
          v9 = *(_QWORD *)a1;
          v19 = *((_QWORD *)a1 + 1);
        }
        else
        {
          *(_QWORD *)a1 = v11;
          *(_QWORD *)v10 = v19;
          v9 = *(_QWORD *)a1;
          v19 = *((_QWORD *)a1 + 1);
          v20 = *((_QWORD *)v7 - 1);
        }
      }
    }
    v21 = v8;
    v22 = v7;
    v23 = *(_DWORD *)(*(_QWORD *)(v9 + 56) + 72LL);
    while ( 1 )
    {
      v6 = (char *)(v21 - 1);
      v24 = *(_DWORD *)(*(_QWORD *)(v19 + 56) + 72LL);
      v25 = v23 > v24;
      if ( v23 == v24 )
        v25 = *(_DWORD *)(v19 + 64) < *(_DWORD *)(v9 + 64);
      if ( !v25 )
        break;
LABEL_22:
      v19 = *v21++;
    }
    for ( i = v22 - 8; ; v20 = *(_QWORD *)i )
    {
      v22 = i;
      v27 = *(_DWORD *)(*(_QWORD *)(v20 + 56) + 72LL);
      if ( v23 == v27 )
        break;
      i -= 8;
      if ( v23 >= v27 )
        goto LABEL_20;
LABEL_17:
      ;
    }
    i -= 8;
    if ( *(_DWORD *)(v9 + 64) < *(_DWORD *)(v20 + 64) )
      goto LABEL_17;
LABEL_20:
    if ( v22 > v6 )
    {
      *(v21 - 1) = v20;
      v20 = *((_QWORD *)v22 - 1);
      *(_QWORD *)v22 = v19;
      v9 = *(_QWORD *)a1;
      v23 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 72LL);
      goto LABEL_22;
    }
    sub_31C82D0(v21 - 1, v7, v4);
    result = v6 - a1;
    if ( v6 - a1 > 128 )
    {
      if ( v4 )
      {
        v7 = v6;
        continue;
      }
LABEL_38:
      v31 = result >> 3;
      for ( j = ((result >> 3) - 2) >> 1; ; --j )
      {
        sub_31C8130((__int64)a1, j, v31, *(_QWORD *)&a1[8 * j]);
        if ( !j )
          break;
      }
      v33 = (__int64 *)(v6 - 8);
      do
      {
        v34 = *v33;
        v35 = (char *)v33-- - a1;
        v33[1] = *(_QWORD *)a1;
        result = (signed __int64)sub_31C8130((__int64)a1, 0, v35 >> 3, v34);
      }
      while ( v35 > 8 );
    }
    return result;
  }
}
