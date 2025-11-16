// Function: sub_2B0DEF0
// Address: 0x2b0def0
//
__int64 __fastcall sub_2B0DEF0(_DWORD *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  _DWORD *v4; // r9
  __int64 v6; // rbx
  _DWORD *v7; // r12
  int v8; // ecx
  int v9; // esi
  int *v10; // rdx
  int v11; // edi
  int v12; // eax
  int v13; // eax
  int v14; // edx
  int *v15; // rsi
  unsigned __int64 v16; // r13
  _DWORD *v17; // rdx
  unsigned __int64 v18; // r14
  _DWORD *v19; // rax
  int i; // edx
  int v21; // ecx
  int v22; // edx
  int v23; // edx
  int v24; // eax
  int v25; // ecx
  int v26; // eax
  __int64 v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // rcx
  int *v30; // [rsp-40h] [rbp-40h]

  result = a2 - (char *)a1;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v18 = (unsigned __int64)a2;
    goto LABEL_23;
  }
  v7 = a1 + 2;
  v30 = a1 + 4;
  while ( 2 )
  {
    v8 = a1[2];
    v9 = *(v4 - 2);
    --v6;
    v10 = &a1[2 * (result >> 4)];
    v11 = *a1;
    v12 = *v10;
    if ( v8 >= *v10 )
    {
      if ( v8 < v9 )
        goto LABEL_7;
      if ( v12 < v9 )
      {
LABEL_17:
        *a1 = v9;
        v23 = *(v4 - 1);
        *(v4 - 2) = v11;
        v24 = a1[1];
        a1[1] = v23;
        *(v4 - 1) = v24;
        v11 = a1[2];
        v8 = *a1;
        goto LABEL_8;
      }
LABEL_21:
      *a1 = v12;
      v25 = v10[1];
      *v10 = v11;
      v26 = a1[1];
      a1[1] = v25;
      v10[1] = v26;
      v11 = a1[2];
      v8 = *a1;
      goto LABEL_8;
    }
    if ( v12 < v9 )
      goto LABEL_21;
    if ( v8 < v9 )
      goto LABEL_17;
LABEL_7:
    v13 = a1[1];
    v14 = a1[3];
    *a1 = v8;
    a1[2] = v11;
    a1[1] = v14;
    a1[3] = v13;
LABEL_8:
    v15 = v30;
    v16 = (unsigned __int64)v7;
    v17 = v4;
    while ( 1 )
    {
      v18 = v16;
      if ( v11 < v8 )
        goto LABEL_14;
      v19 = v17 - 2;
      for ( i = *(v17 - 2); i > v8; v19 -= 2 )
        i = *(v19 - 2);
      if ( v16 >= (unsigned __int64)v19 )
        break;
      *(v15 - 2) = i;
      v21 = v19[1];
      *v19 = v11;
      v22 = *(v15 - 1);
      *(v15 - 1) = v21;
      v19[1] = v22;
      v8 = *a1;
      v17 = v19;
LABEL_14:
      v11 = *v15;
      v16 += 8LL;
      v15 += 2;
    }
    sub_2B0DEF0(v16, v4, v6);
    result = v16 - (_QWORD)a1;
    if ( (__int64)(v16 - (_QWORD)a1) > 128 )
    {
      if ( v6 )
      {
        v4 = (_DWORD *)v16;
        continue;
      }
LABEL_23:
      v27 = result >> 3;
      v28 = ((result >> 3) - 2) >> 1;
      sub_2B09650((__int64)a1, v28, result >> 3, *(_QWORD *)&a1[2 * v28]);
      do
      {
        --v28;
        sub_2B09650((__int64)a1, v28, v27, *(_QWORD *)&a1[2 * v28]);
      }
      while ( v28 );
      do
      {
        v18 -= 8LL;
        v29 = *(_QWORD *)v18;
        *(_DWORD *)v18 = *a1;
        *(_DWORD *)(v18 + 4) = a1[1];
        result = sub_2B09650((__int64)a1, 0, (__int64)(v18 - (_QWORD)a1) >> 3, v29);
      }
      while ( (__int64)(v18 - (_QWORD)a1) > 8 );
    }
    return result;
  }
}
