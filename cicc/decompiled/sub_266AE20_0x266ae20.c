// Function: sub_266AE20
// Address: 0x266ae20
//
char *__fastcall sub_266AE20(char *a1, char *a2, char *a3, __int64 a4, __int64 a5, _QWORD *a6, __int64 a7)
{
  char *v7; // r11
  char *v8; // r10
  char *v9; // rax
  char *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // r11
  __int64 v14; // r8
  _QWORD *v15; // rsi
  char *v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rcx
  char *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r10
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // r8
  _QWORD *v29; // rcx
  __int64 v30; // r14
  __int64 v31; // rdi
  __int64 v32; // rcx
  char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx

  v7 = a3;
  v8 = a1;
  v9 = a2;
  if ( a4 > a5 && a5 <= a7 )
  {
    v10 = a1;
    if ( !a5 )
      return v10;
    v11 = a3 - a2;
    v12 = a2 - a1;
    v13 = (a3 - a2) >> 4;
    v14 = (a2 - a1) >> 4;
    if ( a3 - a2 <= 0 )
    {
      if ( v12 <= 0 )
        return v8;
      v18 = 0;
      v11 = 0;
    }
    else
    {
      v15 = a6;
      v16 = v9;
      do
      {
        v17 = *(_QWORD *)v16;
        v15 += 2;
        v16 += 16;
        *(v15 - 2) = v17;
        *(v15 - 1) = *((_QWORD *)v16 - 1);
        --v13;
      }
      while ( v13 );
      if ( v11 <= 0 )
        v11 = 16;
      v18 = v11 >> 4;
      if ( v12 <= 0 )
        goto LABEL_11;
    }
    do
    {
      v19 = *((_QWORD *)v9 - 2);
      v9 -= 16;
      a3 -= 16;
      *(_QWORD *)a3 = v19;
      *((_QWORD *)a3 + 1) = *((_QWORD *)v9 + 1);
      --v14;
    }
    while ( v14 );
LABEL_11:
    if ( v11 > 0 )
    {
      v20 = a1;
      v21 = v18;
      do
      {
        v22 = *a6;
        v20 += 16;
        a6 += 2;
        *((_QWORD *)v20 - 2) = v22;
        *((_QWORD *)v20 - 1) = *(a6 - 1);
        --v21;
      }
      while ( v21 );
      v23 = 16 * v18;
      if ( v18 <= 0 )
        v23 = 16;
      return &a1[v23];
    }
    return v8;
  }
  if ( a4 <= a7 )
  {
    v10 = a3;
    if ( !a4 )
      return v10;
    v25 = a3 - a2;
    v26 = a2 - a1;
    v27 = (a3 - a2) >> 4;
    v28 = v26 >> 4;
    if ( v26 <= 0 )
    {
      if ( v25 <= 0 )
        return a3;
      v31 = 0;
      v26 = 0;
    }
    else
    {
      v29 = a6;
      do
      {
        v30 = *(_QWORD *)a1;
        v29 += 2;
        a1 += 16;
        *(v29 - 2) = v30;
        *(v29 - 1) = *((_QWORD *)a1 - 1);
        --v28;
      }
      while ( v28 );
      a6 = (_QWORD *)((char *)a6 + v26);
      v31 = v26 >> 4;
      if ( v25 <= 0 )
      {
LABEL_26:
        if ( v26 > 0 )
        {
          v33 = a3;
          v34 = v31;
          do
          {
            v35 = *(a6 - 2);
            a6 -= 2;
            v33 -= 16;
            *(_QWORD *)v33 = v35;
            *((_QWORD *)v33 + 1) = a6[1];
            --v34;
          }
          while ( v34 );
          v36 = -16 * v31;
          if ( v31 <= 0 )
            v36 = -16;
          return &v7[v36];
        }
        return a3;
      }
    }
    do
    {
      v32 = *(_QWORD *)v9;
      v8 += 16;
      v9 += 16;
      *((_QWORD *)v8 - 2) = v32;
      *((_QWORD *)v8 - 1) = *((_QWORD *)v9 - 1);
      --v27;
    }
    while ( v27 );
    goto LABEL_26;
  }
  return sub_2664930(a1, a2, a3);
}
