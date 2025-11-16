// Function: sub_3873120
// Address: 0x3873120
//
char *__fastcall sub_3873120(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6, __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r10
  __int64 v10; // r8
  _QWORD *v11; // rcx
  char *v12; // rax
  __int64 v13; // r11
  __int64 v14; // r10
  __int64 v15; // rax
  char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // r10
  __int64 v24; // r8
  _QWORD *v25; // rcx
  char *v26; // rax
  __int64 v27; // r11
  __int64 v28; // r10
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rsi
  __int64 v33; // rdi

  if ( a4 > a5 && a5 <= a7 )
  {
    if ( !a5 )
      return a1;
    v7 = a3 - (_QWORD)a2;
    v8 = a2 - a1;
    v9 = (a3 - (__int64)a2) >> 4;
    v10 = (a2 - a1) >> 4;
    if ( a3 - (__int64)a2 <= 0 )
    {
      if ( v8 <= 0 )
        return a1;
      v14 = 0;
      v7 = 0;
    }
    else
    {
      v11 = a6;
      v12 = a2;
      do
      {
        v13 = *(_QWORD *)v12;
        v11 += 2;
        v12 += 16;
        *(v11 - 2) = v13;
        *(v11 - 1) = *((_QWORD *)v12 - 1);
        --v9;
      }
      while ( v9 );
      if ( v7 <= 0 )
        v7 = 16;
      v14 = v7 >> 4;
      if ( v8 <= 0 )
      {
LABEL_11:
        if ( v7 > 0 )
        {
          v16 = a1;
          v17 = v14;
          do
          {
            v18 = *a6;
            v16 += 16;
            a6 += 2;
            *((_QWORD *)v16 - 2) = v18;
            *((_QWORD *)v16 - 1) = *(a6 - 1);
            --v17;
          }
          while ( v17 );
          v19 = 16 * v14;
          if ( v14 <= 0 )
            v19 = 16;
          return &a1[v19];
        }
        return a1;
      }
    }
    do
    {
      v15 = *((_QWORD *)a2 - 2);
      a2 -= 16;
      a3 -= 16;
      *(_QWORD *)a3 = v15;
      *(_QWORD *)(a3 + 8) = *((_QWORD *)a2 + 1);
      --v10;
    }
    while ( v10 );
    goto LABEL_11;
  }
  if ( a4 > a7 )
    return sub_386EE20(a1, a2, (char *)a3);
  if ( !a4 )
    return (char *)a3;
  v21 = a2 - a1;
  v22 = a3 - (_QWORD)a2;
  v23 = (a2 - a1) >> 4;
  v24 = (a3 - (__int64)a2) >> 4;
  if ( a2 - a1 <= 0 )
  {
    if ( v22 <= 0 )
      return (char *)a3;
    v28 = 0;
    v21 = 0;
    goto LABEL_25;
  }
  v25 = a6;
  v26 = a1;
  do
  {
    v27 = *(_QWORD *)v26;
    v25 += 2;
    v26 += 16;
    *(v25 - 2) = v27;
    *(v25 - 1) = *((_QWORD *)v26 - 1);
    --v23;
  }
  while ( v23 );
  if ( v21 <= 0 )
    v21 = 16;
  a6 = (_QWORD *)((char *)a6 + v21);
  v28 = v21 >> 4;
  if ( v22 > 0 )
  {
    do
    {
LABEL_25:
      v29 = *(_QWORD *)a2;
      a1 += 16;
      a2 += 16;
      *((_QWORD *)a1 - 2) = v29;
      *((_QWORD *)a1 - 1) = *((_QWORD *)a2 - 1);
      --v24;
    }
    while ( v24 );
  }
  if ( v21 <= 0 )
    return (char *)a3;
  v30 = (_QWORD *)a3;
  v31 = v28;
  do
  {
    v32 = *(a6 - 2);
    a6 -= 2;
    v30 -= 2;
    *v30 = v32;
    v30[1] = a6[1];
    --v31;
  }
  while ( v31 );
  v33 = -16 * v28;
  if ( v28 <= 0 )
    v33 = -16;
  return (char *)(a3 + v33);
}
