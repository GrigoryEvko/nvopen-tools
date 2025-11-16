// Function: sub_2A4C350
// Address: 0x2a4c350
//
__int64 __fastcall sub_2A4C350(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r9
  __int64 v6; // rbx
  __int64 *v7; // r12
  __int64 v8; // r13
  __int64 v9; // rdi
  char *v10; // r10
  __int64 v11; // rax
  unsigned int v12; // esi
  unsigned int v13; // ecx
  unsigned int v14; // edx
  __int64 v15; // rdx
  __int64 *v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // r13
  __int64 *v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 *v24; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v20 = a2;
    goto LABEL_23;
  }
  v7 = (__int64 *)(a1 + 8);
  v24 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v8 = *(v4 - 1);
    v9 = *(_QWORD *)a1;
    --v6;
    v10 = &a1[8 * (result >> 4)];
    v11 = *((_QWORD *)a1 + 1);
    v12 = *(_DWORD *)(v8 + 44);
    v13 = *(_DWORD *)(v11 + 44);
    v14 = *(_DWORD *)(*(_QWORD *)v10 + 44LL);
    if ( v13 >= v14 )
    {
      if ( v13 < v12 )
        goto LABEL_7;
      if ( v14 < v12 )
      {
LABEL_17:
        *(_QWORD *)a1 = v8;
        v15 = v9;
        *(v4 - 1) = v9;
        v11 = *(_QWORD *)a1;
        v9 = *((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_21:
      *(_QWORD *)a1 = *(_QWORD *)v10;
      *(_QWORD *)v10 = v9;
      v15 = *(v4 - 1);
      v11 = *(_QWORD *)a1;
      v9 = *((_QWORD *)a1 + 1);
      goto LABEL_8;
    }
    if ( v14 < v12 )
      goto LABEL_21;
    if ( v13 < v12 )
      goto LABEL_17;
LABEL_7:
    *(_QWORD *)a1 = v11;
    *((_QWORD *)a1 + 1) = v9;
    v15 = *(v4 - 1);
LABEL_8:
    v16 = v24;
    v17 = *(_DWORD *)(v11 + 44);
    v18 = v7;
    v19 = v4;
    while ( 1 )
    {
      v20 = v18;
      if ( *(_DWORD *)(v9 + 44) < v17 )
        goto LABEL_14;
      for ( --v19; *(_DWORD *)(v15 + 44) > v17; --v19 )
        v15 = *(v19 - 1);
      if ( v19 <= v18 )
        break;
      *v18 = v15;
      v15 = *(v19 - 1);
      *v19 = v9;
      v17 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
LABEL_14:
      v9 = *v16;
      ++v18;
      ++v16;
    }
    sub_2A4C350(v18, v4, v6);
    result = (char *)v18 - a1;
    if ( (char *)v18 - a1 > 128 )
    {
      if ( v6 )
      {
        v4 = v18;
        continue;
      }
LABEL_23:
      v21 = result >> 3;
      v22 = ((result >> 3) - 2) >> 1;
      sub_2A4C210((__int64)a1, v22, result >> 3, *(_QWORD *)&a1[8 * v22]);
      do
      {
        --v22;
        sub_2A4C210((__int64)a1, v22, v21, *(_QWORD *)&a1[8 * v22]);
      }
      while ( v22 );
      do
      {
        v23 = *--v20;
        *v20 = *(_QWORD *)a1;
        result = sub_2A4C210((__int64)a1, 0, ((char *)v20 - a1) >> 3, v23);
      }
      while ( (char *)v20 - a1 > 8 );
    }
    return result;
  }
}
