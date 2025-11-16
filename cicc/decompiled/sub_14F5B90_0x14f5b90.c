// Function: sub_14F5B90
// Address: 0x14f5b90
//
_QWORD *__fastcall sub_14F5B90(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 *v4; // rbx
  _QWORD *result; // rax
  __int64 v6; // r15
  unsigned __int64 v7; // rcx
  _QWORD *v9; // r14
  _QWORD *v10; // r8
  _QWORD *v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rsi
  _QWORD *v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdx
  _QWORD *v17; // rdi
  char *v18; // rsi
  _QWORD *v19; // r8
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rsi
  _QWORD *v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r8
  unsigned __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = (__int64 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  result = (_QWORD *)*a3;
  v6 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v7 = *(_QWORD *)(*a3 + 16LL);
  v27 = v7;
  if ( v6 != v7 && v7 )
  {
    result = (_QWORD *)a1[18];
    v9 = a1 + 17;
    if ( !result )
    {
      v14 = a1 + 17;
LABEL_16:
      v17 = a1 + 16;
LABEL_17:
      v28[0] = &v27;
      result = (_QWORD *)sub_14F5AE0(v17, v14, v28);
      v14 = result;
LABEL_18:
      v14[5] = v6;
      goto LABEL_19;
    }
    v10 = a1 + 17;
    v11 = (_QWORD *)a1[18];
    do
    {
      while ( 1 )
      {
        v12 = v11[2];
        v13 = v11[3];
        if ( v7 <= v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v13 )
          goto LABEL_8;
      }
      v10 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v12 );
LABEL_8:
    if ( v9 == v10 || v7 < v10[4] )
    {
LABEL_10:
      v14 = v9;
      do
      {
        while ( 1 )
        {
          v15 = result[2];
          v16 = result[3];
          if ( result[4] >= v27 )
            break;
          result = (_QWORD *)result[3];
          if ( !v16 )
            goto LABEL_14;
        }
        v14 = result;
        result = (_QWORD *)result[2];
      }
      while ( v15 );
LABEL_14:
      if ( v9 != v14 && v27 >= v14[4] )
        goto LABEL_18;
      goto LABEL_16;
    }
    v19 = v9;
    v20 = result;
    do
    {
      while ( 1 )
      {
        v21 = v20[2];
        v22 = v20[3];
        if ( v7 <= v20[4] )
          break;
        v20 = (_QWORD *)v20[3];
        if ( !v22 )
          goto LABEL_27;
      }
      v19 = v20;
      v20 = (_QWORD *)v20[2];
    }
    while ( v21 );
LABEL_27:
    if ( v9 == v19 || v7 < v19[4] )
    {
      v28[0] = &v27;
      v17 = a1 + 16;
      v26 = sub_14F5AE0(a1 + 16, v19, v28);
      result = (_QWORD *)a1[18];
      if ( v6 == *(_QWORD *)(v26 + 40) )
      {
        if ( !result )
        {
          v14 = v9;
          goto LABEL_17;
        }
        goto LABEL_10;
      }
      if ( !result )
      {
        v23 = v9;
        goto LABEL_37;
      }
    }
    else if ( v6 == v19[5] )
    {
      goto LABEL_10;
    }
    v23 = v9;
    do
    {
      while ( 1 )
      {
        v24 = result[2];
        v25 = result[3];
        if ( result[4] >= v27 )
          break;
        result = (_QWORD *)result[3];
        if ( !v25 )
          goto LABEL_34;
      }
      v23 = result;
      result = (_QWORD *)result[2];
    }
    while ( v24 );
LABEL_34:
    if ( v9 != v23 && v27 >= v23[4] )
    {
LABEL_38:
      v23[5] = 0;
      v18 = (char *)v4[4];
      if ( v18 != (char *)v4[5] )
        goto LABEL_20;
      return (_QWORD *)sub_142DF10(v4 + 3, v18, a3);
    }
    v17 = a1 + 16;
LABEL_37:
    v28[0] = &v27;
    result = (_QWORD *)sub_14F5AE0(v17, v23, v28);
    v23 = result;
    goto LABEL_38;
  }
LABEL_19:
  v18 = (char *)v4[4];
  if ( v18 == (char *)v4[5] )
    return (_QWORD *)sub_142DF10(v4 + 3, v18, a3);
LABEL_20:
  if ( v18 )
  {
    result = (_QWORD *)*a3;
    *(_QWORD *)v18 = *a3;
    *a3 = 0;
    v18 = (char *)v4[4];
  }
  v4[4] = (__int64)(v18 + 8);
  return result;
}
