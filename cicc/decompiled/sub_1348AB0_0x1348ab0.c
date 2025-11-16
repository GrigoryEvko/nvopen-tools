// Function: sub_1348AB0
// Address: 0x1348ab0
//
_QWORD *__fastcall sub_1348AB0(_QWORD *a1)
{
  __int64 v1; // r8
  _QWORD *v2; // rax
  _QWORD *v4; // r8
  _QWORD *v5; // rsi
  _QWORD *v6; // rbx
  __int64 v7; // rcx
  _QWORD *v8; // r11
  __int64 v9; // r9
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // rdi
  _QWORD *v13; // rcx
  __int64 v14; // r9
  _QWORD *v15; // rdx
  __int64 v16; // r9
  _QWORD *v17; // rsi
  __int64 v18; // r9
  _QWORD *result; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx

  v1 = *a1;
  if ( !*a1 )
    return (_QWORD *)v1;
  a1[1] = 0;
  v2 = *(_QWORD **)(v1 + 48);
  if ( !v2 )
    return (_QWORD *)v1;
  *(_QWORD *)(v1 + 40) = 0;
  *(_QWORD *)(*a1 + 48LL) = 0;
  v2[5] = 0;
  v4 = (_QWORD *)v2[6];
  if ( v4 )
  {
    v5 = (_QWORD *)v4[6];
    v6 = v4 + 5;
    if ( v5 )
      v5[5] = 0;
    v2[5] = 0;
    v2[6] = 0;
    *v6 = 0;
    v4[6] = 0;
    if ( (v2[1] > v4[1]) - (v2[1] < v4[1]) == -1 )
    {
      *v6 = v2;
      v22 = v2[7];
      v4[6] = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 40) = v4;
      v2[7] = v4;
      v6 = v2 + 5;
      v4 = v2;
    }
    else
    {
      v2[5] = v4;
      v7 = v4[7];
      v2[6] = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 40) = v2;
      v4[7] = v2;
    }
    if ( v5 )
    {
      v8 = v4;
      while ( 1 )
      {
        v10 = (_QWORD *)v5[6];
        v11 = v5 + 5;
        if ( !v10 )
          break;
        v12 = (_QWORD *)v10[6];
        v13 = v10 + 5;
        if ( v12 )
          v12[5] = 0;
        *v11 = 0;
        v5[6] = 0;
        *v13 = 0;
        v10[6] = 0;
        if ( (v5[1] > v10[1]) - (v5[1] < v10[1]) == -1 )
        {
          *v13 = v5;
          v14 = v5[7];
          v10[6] = v14;
          if ( v14 )
            *(_QWORD *)(v14 + 40) = v10;
          v5[7] = v10;
          v10 = v5;
          v8[6] = v5;
          if ( !v12 )
            goto LABEL_23;
        }
        else
        {
          *v11 = v10;
          v9 = v10[7];
          v5[6] = v9;
          if ( v9 )
            *(_QWORD *)(v9 + 40) = v5;
          v10[7] = v5;
          v8[6] = v10;
          if ( !v12 )
            goto LABEL_23;
        }
        v8 = v10;
        v5 = v12;
      }
      v8[6] = v5;
      v10 = v5;
    }
    else
    {
      v10 = v4;
    }
LABEL_23:
    v15 = (_QWORD *)v6[1];
    if ( !v15 )
      goto LABEL_35;
    while ( 1 )
    {
      v17 = (_QWORD *)v15[6];
      v4[6] = 0;
      v15[6] = 0;
      if ( !v15 )
        goto LABEL_28;
      if ( (v4[1] > v15[1]) - (v4[1] < v15[1]) != -1 )
        break;
      v15[5] = v4;
      v18 = v4[7];
      v15[6] = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 40) = v15;
      v4[7] = v15;
      if ( !v17 )
        goto LABEL_35;
LABEL_29:
      v10[6] = v4;
      v15 = (_QWORD *)v17[6];
      v10 = v4;
      v4 = v17;
    }
    v4[5] = v15;
    v16 = v15[7];
    v4[6] = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 40) = v4;
    v15[7] = v4;
    v4 = v15;
LABEL_28:
    if ( !v17 )
      goto LABEL_35;
    goto LABEL_29;
  }
  v4 = v2;
LABEL_35:
  result = (_QWORD *)*a1;
  if ( !*a1 )
  {
LABEL_40:
    *a1 = v4;
    return v4;
  }
  if ( (result[1] > v4[1]) - (result[1] < v4[1]) != -1 )
  {
    result[5] = v4;
    v20 = v4[7];
    result[6] = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 40) = result;
    v4[7] = result;
    goto LABEL_40;
  }
  v4[5] = result;
  v21 = result[7];
  v4[6] = v21;
  if ( v21 )
    *(_QWORD *)(v21 + 40) = v4;
  result[7] = v4;
  *a1 = result;
  return result;
}
