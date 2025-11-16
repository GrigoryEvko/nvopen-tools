// Function: sub_133F530
// Address: 0x133f530
//
_QWORD *__fastcall sub_133F530(_QWORD *a1)
{
  __int64 v1; // r8
  _QWORD *v2; // rax
  _QWORD *v4; // r8
  _QWORD *v5; // rcx
  _QWORD *v6; // r12
  unsigned __int64 v7; // rbx
  int v8; // edx
  __int64 v9; // rdx
  _QWORD *v10; // r11
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  _QWORD *v14; // r9
  _QWORD *v15; // rdi
  int v16; // edx
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  __int64 v19; // rdi
  _QWORD *v20; // rsi
  int v21; // edi
  __int64 v22; // rdi
  _QWORD *result; // rax
  unsigned __int64 v24; // rbx
  int v25; // edx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx

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
    v7 = v4[4];
    v8 = (v2[4] > v7) - (v2[4] < v7);
    if ( v2[4] > v7 == v2[4] < v7 )
      v8 = (v2[1] > v4[1]) - (v2[1] < v4[1]);
    if ( v8 == -1 )
    {
      *v6 = v2;
      v28 = v2[7];
      v4[6] = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 40) = v4;
      v2[7] = v4;
      v6 = v2 + 5;
      v4 = v2;
    }
    else
    {
      v2[5] = v4;
      v9 = v4[7];
      v2[6] = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 40) = v2;
      v4[7] = v2;
    }
    if ( v5 )
    {
      v10 = v4;
      while ( 1 )
      {
        v12 = (_QWORD *)v5[6];
        v13 = v5 + 5;
        if ( !v12 )
          break;
        v14 = (_QWORD *)v12[6];
        v15 = v12 + 5;
        if ( v14 )
          v14[5] = 0;
        *v13 = 0;
        v5[6] = 0;
        *v15 = 0;
        v12[6] = 0;
        v16 = (v5[4] > v12[4]) - (v5[4] < v12[4]);
        if ( v5[4] > v12[4] == v5[4] < v12[4] )
          v16 = (v5[1] > v12[1]) - (v5[1] < v12[1]);
        if ( v16 == -1 )
        {
          *v15 = v5;
          v17 = v5[7];
          v12[6] = v17;
          if ( v17 )
            *(_QWORD *)(v17 + 40) = v12;
          v5[7] = v12;
          v12 = v5;
          v10[6] = v5;
          if ( !v14 )
            goto LABEL_27;
        }
        else
        {
          *v13 = v12;
          v11 = v12[7];
          v5[6] = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 40) = v5;
          v12[7] = v5;
          v10[6] = v12;
          if ( !v14 )
            goto LABEL_27;
        }
        v10 = v12;
        v5 = v14;
      }
      v10[6] = v5;
      v12 = v5;
    }
    else
    {
      v12 = v4;
    }
LABEL_27:
    v18 = (_QWORD *)v6[1];
    if ( !v18 )
      goto LABEL_41;
    while ( 1 )
    {
      v20 = (_QWORD *)v18[6];
      v4[6] = 0;
      v18[6] = 0;
      if ( !v18 )
        goto LABEL_32;
      v21 = (v4[4] > v18[4]) - (v4[4] < v18[4]);
      if ( v4[4] > v18[4] == v4[4] < v18[4] )
        v21 = (v4[1] > v18[1]) - (v4[1] < v18[1]);
      if ( v21 != -1 )
        break;
      v18[5] = v4;
      v22 = v4[7];
      v18[6] = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 40) = v18;
      v4[7] = v18;
      if ( !v20 )
        goto LABEL_41;
LABEL_33:
      v12[6] = v4;
      v18 = (_QWORD *)v20[6];
      v12 = v4;
      v4 = v20;
    }
    v4[5] = v18;
    v19 = v18[7];
    v4[6] = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 40) = v4;
    v18[7] = v4;
    v4 = v18;
LABEL_32:
    if ( !v20 )
      goto LABEL_41;
    goto LABEL_33;
  }
  v4 = v2;
LABEL_41:
  result = (_QWORD *)*a1;
  if ( !*a1 )
  {
LABEL_48:
    *a1 = v4;
    return v4;
  }
  v24 = v4[4];
  v25 = (result[4] > v24) - (result[4] < v24);
  if ( result[4] > v24 == result[4] < v24 )
    v25 = (result[1] > v4[1]) - (result[1] < v4[1]);
  if ( v25 != -1 )
  {
    result[5] = v4;
    v26 = v4[7];
    result[6] = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 40) = result;
    v4[7] = result;
    goto LABEL_48;
  }
  v4[5] = result;
  v27 = result[7];
  v4[6] = v27;
  if ( v27 )
    *(_QWORD *)(v27 + 40) = v4;
  result[7] = v4;
  *a1 = result;
  return result;
}
