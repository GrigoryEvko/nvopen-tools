// Function: sub_18349D0
// Address: 0x18349d0
//
_QWORD *__fastcall sub_18349D0(_QWORD *a1, _QWORD *a2, unsigned __int64 *a3)
{
  unsigned __int64 v5; // r14
  _QWORD *result; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  char *v10; // rsi
  char *v11; // rax
  _QWORD *v12; // rcx
  signed __int64 v13; // rdi
  char *v14; // r10
  char *v15; // r8
  char *v16; // rcx
  char *v17; // rsi
  signed __int64 v18; // r9
  signed __int64 v19; // r11
  char *v20; // rdx
  char *v21; // rdi
  char *v22; // rax
  _QWORD *v23; // rbx
  char *v24; // rdi
  char *v25; // rcx
  char *v26; // rax
  char *v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  char *v30; // r8
  char *v31; // rsi
  char *v32; // rcx
  char *v33; // rax

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_1834870((__int64)a1, a3);
    v23 = (_QWORD *)a1[4];
    if ( v23[4] >= *a3 )
    {
      if ( v23[4] != *a3 )
        return sub_1834870((__int64)a1, a3);
      v24 = (char *)a3[2];
      v25 = (char *)v23[6];
      v26 = (char *)v23[5];
      v27 = (char *)a3[1];
      if ( v25 - v26 > v24 - v27 )
        v25 = &v26[v24 - v27];
      if ( v26 == v25 )
      {
LABEL_64:
        if ( v24 == v27 )
          return sub_1834870((__int64)a1, a3);
      }
      else
      {
        while ( *(_QWORD *)v26 >= *(_QWORD *)v27 )
        {
          if ( *(_QWORD *)v26 > *(_QWORD *)v27 )
            return sub_1834870((__int64)a1, a3);
          v26 += 8;
          v27 += 8;
          if ( v25 == v26 )
            goto LABEL_64;
        }
      }
    }
    return 0;
  }
  v5 = *a3;
  if ( *a3 < a2[4] )
    goto LABEL_3;
  if ( *a3 == a2[4] )
  {
    v14 = (char *)a3[2];
    v15 = (char *)a2[6];
    v16 = (char *)a2[5];
    v17 = (char *)a3[1];
    v18 = v15 - v16;
    v19 = v14 - v17;
    v20 = v16;
    v21 = &v17[v15 - v16];
    if ( v14 - v17 <= v15 - v16 )
      v21 = v14;
    if ( v17 != v21 )
    {
      v22 = v17;
      while ( *(_QWORD *)v22 >= *(_QWORD *)v20 )
      {
        if ( *(_QWORD *)v22 > *(_QWORD *)v20 )
          goto LABEL_42;
        v22 += 8;
        v20 += 8;
        if ( v21 == v22 )
          goto LABEL_41;
      }
LABEL_3:
      result = a2;
      if ( (_QWORD *)a1[3] == a2 )
        return result;
      v8 = (_QWORD *)sub_220EF80(a2);
      v9 = v8;
      if ( v5 > v8[4] )
        goto LABEL_13;
      if ( v5 == v8[4] )
      {
        v10 = (char *)v8[6];
        v11 = (char *)v8[5];
        v12 = (_QWORD *)a3[1];
        v13 = a3[2] - (_QWORD)v12;
        if ( v10 - v11 > v13 )
          v10 = &v11[v13];
        if ( v11 != v10 )
        {
          while ( *(_QWORD *)v11 >= *v12 )
          {
            if ( *(_QWORD *)v11 > *v12 )
              return sub_1834870((__int64)a1, a3);
            v11 += 8;
            ++v12;
            if ( v10 == v11 )
              goto LABEL_62;
          }
          goto LABEL_13;
        }
LABEL_62:
        if ( (_QWORD *)a3[2] != v12 )
        {
LABEL_13:
          result = 0;
          if ( v9[3] )
            return a2;
          return result;
        }
      }
      return sub_1834870((__int64)a1, a3);
    }
LABEL_41:
    if ( v15 != v20 )
      goto LABEL_3;
  }
  else
  {
    if ( *a3 > a2[4] )
      goto LABEL_37;
    v15 = (char *)a2[6];
    v14 = (char *)a3[2];
    v16 = (char *)a2[5];
    v17 = (char *)a3[1];
    v19 = v14 - v17;
    v18 = v15 - v16;
  }
LABEL_42:
  if ( v19 < v18 )
    v15 = &v16[v19];
  if ( v15 == v16 )
  {
LABEL_60:
    if ( v17 == v14 )
      return a2;
  }
  else
  {
    while ( *(_QWORD *)v16 >= *(_QWORD *)v17 )
    {
      if ( *(_QWORD *)v16 > *(_QWORD *)v17 )
        return a2;
      v16 += 8;
      v17 += 8;
      if ( v15 == v16 )
        goto LABEL_60;
    }
  }
LABEL_37:
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  v28 = (_QWORD *)sub_220EEE0(a2);
  v29 = v28;
  if ( v5 >= v28[4] )
  {
    if ( v5 != v28[4] )
      return sub_1834870((__int64)a1, a3);
    v30 = (char *)v28[6];
    v31 = (char *)a3[2];
    v32 = (char *)v28[5];
    v33 = (char *)a3[1];
    if ( v31 - v33 > v30 - v32 )
      v31 = &v33[v30 - v32];
    if ( v33 == v31 )
    {
LABEL_66:
      if ( v32 == v30 )
        return sub_1834870((__int64)a1, a3);
    }
    else
    {
      while ( *(_QWORD *)v33 >= *(_QWORD *)v32 )
      {
        if ( *(_QWORD *)v33 > *(_QWORD *)v32 )
          return sub_1834870((__int64)a1, a3);
        v33 += 8;
        v32 += 8;
        if ( v31 == v33 )
          goto LABEL_66;
      }
    }
  }
  result = 0;
  if ( a2[3] )
    return v29;
  return result;
}
