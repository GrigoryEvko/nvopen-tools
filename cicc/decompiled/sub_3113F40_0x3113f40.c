// Function: sub_3113F40
// Address: 0x3113f40
//
signed __int64 __fastcall sub_3113F40(unsigned __int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  signed __int64 result; // rax
  unsigned __int64 *v4; // r9
  __int64 v6; // r12
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // rcx
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  unsigned __int64 *v15; // rdx
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // r12
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // rdx
  unsigned __int64 *v32; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v19 = a2;
    goto LABEL_32;
  }
  v7 = a1 + 2;
  v32 = a1 + 4;
  while ( 2 )
  {
    v8 = a1[2];
    --v6;
    v9 = &a1[2 * (result >> 5)];
    v10 = *v9;
    if ( v8 < *v9 || v8 == *v9 && a1[3] < v9[1] )
    {
      v11 = *(v4 - 2);
      if ( v10 < v11 || v10 == v11 && v9[1] < *(v4 - 1) )
        goto LABEL_26;
      if ( v8 >= v11 )
      {
        v23 = a1[3];
        if ( v8 != v11 || *(v4 - 1) <= v23 )
        {
          v13 = *a1;
          v31 = a1[1];
          *a1 = v8;
          a1[1] = v23;
          a1[2] = v13;
          a1[3] = v31;
          goto LABEL_10;
        }
      }
      goto LABEL_30;
    }
    v11 = *(v4 - 2);
    if ( v8 >= v11 )
    {
      if ( v8 == v11 )
      {
        v12 = a1[3];
        if ( v12 < *(v4 - 1) )
          goto LABEL_9;
      }
      if ( v10 >= v11 && (v10 != v11 || v9[1] >= *(v4 - 1)) )
      {
LABEL_26:
        v21 = *a1;
        *a1 = v10;
        *v9 = v21;
        v22 = a1[1];
        a1[1] = v9[1];
        v9[1] = v22;
        v13 = a1[2];
        v8 = *a1;
        goto LABEL_10;
      }
LABEL_30:
      v24 = *a1;
      *a1 = v11;
      v25 = *(v4 - 1);
      *(v4 - 2) = v24;
      v26 = a1[1];
      a1[1] = v25;
      *(v4 - 1) = v26;
      v13 = a1[2];
      v8 = *a1;
      goto LABEL_10;
    }
    v12 = a1[3];
LABEL_9:
    v13 = *a1;
    v14 = a1[1];
    *a1 = v8;
    a1[1] = v12;
    a1[2] = v13;
    a1[3] = v14;
LABEL_10:
    v15 = v32;
    v16 = v7;
    v17 = v4;
    while ( 1 )
    {
      v19 = v16;
      if ( v13 >= v8 && (v13 != v8 || *(v15 - 1) >= a1[1]) )
        break;
LABEL_12:
      v13 = *v15;
      v16 += 2;
      v15 += 2;
    }
    do
    {
      do
      {
        v20 = *(v17 - 2);
        v17 -= 2;
      }
      while ( v20 > v8 );
    }
    while ( v20 == v8 && a1[1] < v17[1] );
    if ( v16 < v17 )
    {
      *(v15 - 2) = v20;
      *v17 = v13;
      v18 = *(v15 - 1);
      *(v15 - 1) = v17[1];
      v17[1] = v18;
      v8 = *a1;
      goto LABEL_12;
    }
    sub_3113F40(v16, v4, v6);
    result = (char *)v16 - (char *)a1;
    if ( (char *)v16 - (char *)a1 > 256 )
    {
      if ( v6 )
      {
        v4 = v16;
        continue;
      }
LABEL_32:
      v27 = result >> 4;
      v28 = ((result >> 4) - 2) >> 1;
      sub_3113BF0((__int64)a1, v28, result >> 4, a1[2 * v28], a1[2 * v28 + 1]);
      do
      {
        --v28;
        sub_3113BF0((__int64)a1, v28, v27, a1[2 * v28], a1[2 * v28 + 1]);
      }
      while ( v28 );
      do
      {
        v19 -= 2;
        v29 = *v19;
        v30 = v19[1];
        *v19 = *a1;
        v19[1] = a1[1];
        result = (signed __int64)sub_3113BF0((__int64)a1, 0, ((char *)v19 - (char *)a1) >> 4, v29, v30);
      }
      while ( (char *)v19 - (char *)a1 > 16 );
    }
    return result;
  }
}
