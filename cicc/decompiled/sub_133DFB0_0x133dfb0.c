// Function: sub_133DFB0
// Address: 0x133dfb0
//
_QWORD *__fastcall sub_133DFB0(_QWORD *a1)
{
  __int64 v1; // r8
  _QWORD *v2; // rax
  _QWORD *v4; // r8
  _QWORD *v5; // rsi
  _QWORD *v6; // r12
  unsigned __int64 v7; // rdi
  int v8; // edx
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  _QWORD *v14; // r9
  _QWORD *v15; // rdi
  unsigned __int64 v16; // r10
  int v17; // edx
  __int64 v18; // rdx
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  _QWORD *v21; // rsi
  unsigned __int64 v22; // r10
  int v23; // edi
  __int64 v24; // rdi
  _QWORD *result; // rax
  unsigned __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx

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
    v7 = v2[2] & 0xFFFLL;
    v8 = (v7 > (v4[2] & 0xFFFuLL)) - (v7 < (v4[2] & 0xFFFuLL));
    if ( v7 > (v4[2] & 0xFFFuLL) == v7 < (v4[2] & 0xFFFuLL) )
      v8 = (v2 > v4) - (v2 < v4);
    if ( v8 == -1 )
    {
      *v6 = v2;
      v30 = v2[7];
      v4[6] = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 40) = v4;
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
        v16 = v5[2] & 0xFFFLL;
        v17 = (v16 > (v12[2] & 0xFFFuLL)) - (v16 < (v12[2] & 0xFFFuLL));
        if ( v16 > (v12[2] & 0xFFFuLL) == v16 < (v12[2] & 0xFFFuLL) )
          v17 = (v5 > v12) - (v5 < v12);
        if ( v17 == -1 )
        {
          *v15 = v5;
          v18 = v5[7];
          v12[6] = v18;
          if ( v18 )
            *(_QWORD *)(v18 + 40) = v12;
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
    v19 = (_QWORD *)v6[1];
    if ( !v19 )
      goto LABEL_41;
    while ( 1 )
    {
      v21 = (_QWORD *)v19[6];
      v4[6] = 0;
      v19[6] = 0;
      if ( !v19 )
        goto LABEL_32;
      v22 = v4[2] & 0xFFFLL;
      v23 = (v22 > (v19[2] & 0xFFFuLL)) - (v22 < (v19[2] & 0xFFFuLL));
      if ( v22 > (v19[2] & 0xFFFuLL) == v22 < (v19[2] & 0xFFFuLL) )
        v23 = (v19 < v4) - (v19 > v4);
      if ( v23 != -1 )
        break;
      v19[5] = v4;
      v24 = v4[7];
      v19[6] = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 40) = v19;
      v4[7] = v19;
      if ( !v21 )
        goto LABEL_41;
LABEL_33:
      v12[6] = v4;
      v19 = (_QWORD *)v21[6];
      v12 = v4;
      v4 = v21;
    }
    v4[5] = v19;
    v20 = v19[7];
    v4[6] = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 40) = v4;
    v19[7] = v4;
    v4 = v19;
LABEL_32:
    if ( !v21 )
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
  v26 = result[2] & 0xFFFLL;
  v27 = (v26 > (v4[2] & 0xFFFuLL)) - (v26 < (v4[2] & 0xFFFuLL));
  if ( v26 > (v4[2] & 0xFFFuLL) == v26 < (v4[2] & 0xFFFuLL) )
    v27 = (result > v4) - (result < v4);
  if ( v27 != -1 )
  {
    result[5] = v4;
    v28 = v4[7];
    result[6] = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 40) = result;
    v4[7] = result;
    goto LABEL_48;
  }
  v4[5] = result;
  v29 = result[7];
  v4[6] = v29;
  if ( v29 )
    *(_QWORD *)(v29 + 40) = v4;
  result[7] = v4;
  *a1 = result;
  return result;
}
