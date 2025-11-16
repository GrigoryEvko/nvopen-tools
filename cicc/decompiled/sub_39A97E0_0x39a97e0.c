// Function: sub_39A97E0
// Address: 0x39a97e0
//
__int64 __fastcall sub_39A97E0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v5; // r14
  __int64 v6; // rdi
  __int64 *v7; // rbx
  char *v8; // rsi
  char *v9; // rax
  char *v10; // rcx
  char *v11; // rdx
  signed __int64 v12; // r10
  char *v13; // r12
  char *v14; // r11
  char *v15; // r8
  __int64 v16; // r12
  char *v17; // r8
  signed __int64 v18; // r11
  bool v19; // cc
  char *v20; // r10
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 v23; // r8
  char *v24; // r9
  char *v25; // rcx
  __int64 *v26; // rsi
  __int64 *i; // r11
  char *v28; // r12
  char *v29; // rax
  __int64 *v30; // rbx
  char *v31; // rdx
  char *v32; // r12
  char *v33; // rax
  char *v34; // rdx
  char *v35; // r13
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rcx
  __int64 v39; // r13
  char *v40; // r11
  signed __int64 v41; // r12
  char *v42; // r8
  __int64 *v43; // [rsp+8h] [rbp-58h]
  signed __int64 v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  char *v46; // [rsp+28h] [rbp-38h]
  char *v47; // [rsp+28h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v45 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  v5 = a2;
  if ( !a3 )
  {
LABEL_44:
    v36 = result >> 3;
    v37 = ((result >> 3) - 2) >> 1;
    sub_39A95F0((__int64)a1, v37, result >> 3, a1[v37]);
    do
    {
      --v37;
      sub_39A95F0((__int64)a1, v37, v36, a1[v37]);
    }
    while ( v37 );
    do
    {
      v38 = *--v5;
      *v5 = *a1;
      result = sub_39A95F0((__int64)a1, 0, v5 - a1, v38);
    }
    while ( (char *)v5 - (char *)a1 > 8 );
    return result;
  }
  v43 = a1 + 2;
  while ( 2 )
  {
    v6 = a1[1];
    --v45;
    v7 = &a1[result >> 4];
    v8 = *(char **)(v6 + 104);
    v9 = *(char **)(v6 + 96);
    v10 = *(char **)(*v7 + 104);
    v11 = *(char **)(*v7 + 96);
    v44 = v8 - v9;
    v12 = v10 - v11;
    v13 = &v9[v10 - v11];
    v14 = v11;
    if ( v8 - v9 <= v10 - v11 )
      v13 = *(char **)(v6 + 104);
    if ( v9 == v13 )
    {
LABEL_48:
      if ( v14 == v10 )
      {
LABEL_49:
        v39 = *(v5 - 1);
        v47 = *(char **)(v39 + 104);
        v40 = *(char **)(v39 + 96);
        v41 = v47 - v40;
        if ( v44 > v47 - v40 )
          v8 = &v9[v47 - v40];
        v42 = *(char **)(v39 + 96);
        if ( v9 == v8 )
        {
LABEL_67:
          if ( v47 == v42 )
          {
LABEL_68:
            if ( v12 > v41 )
              v10 = &v11[v41];
            if ( v11 == v10 )
            {
LABEL_79:
              if ( v40 == v47 )
                goto LABEL_18;
            }
            else
            {
              while ( *(_DWORD *)v11 >= *(_DWORD *)v40 )
              {
                if ( *(_DWORD *)v11 > *(_DWORD *)v40 )
                  goto LABEL_18;
                v11 += 4;
                v40 += 4;
                if ( v10 == v11 )
                  goto LABEL_79;
              }
            }
            v23 = *a1;
            *a1 = v39;
            *(v5 - 1) = v23;
            v6 = *a1;
            v22 = a1[1];
            goto LABEL_19;
          }
        }
        else
        {
          while ( *(_DWORD *)v9 >= *(_DWORD *)v42 )
          {
            if ( *(_DWORD *)v9 > *(_DWORD *)v42 )
              goto LABEL_68;
            v9 += 4;
            v42 += 4;
            if ( v8 == v9 )
              goto LABEL_67;
          }
        }
LABEL_56:
        v22 = *a1;
        *a1 = v6;
        a1[1] = v22;
        v23 = *(v5 - 1);
        goto LABEL_19;
      }
    }
    else
    {
      v15 = *(char **)(v6 + 96);
      while ( *(_DWORD *)v15 >= *(_DWORD *)v14 )
      {
        if ( *(_DWORD *)v15 > *(_DWORD *)v14 )
          goto LABEL_49;
        v15 += 4;
        v14 += 4;
        if ( v13 == v15 )
          goto LABEL_48;
      }
    }
    v16 = *(v5 - 1);
    v17 = *(char **)(v16 + 96);
    v46 = *(char **)(v16 + 104);
    v18 = v46 - v17;
    v19 = v12 <= v46 - v17;
    v20 = v17;
    if ( !v19 )
      v10 = &v11[v46 - v17];
    if ( v11 != v10 )
    {
      while ( *(_DWORD *)v11 >= *(_DWORD *)v20 )
      {
        if ( *(_DWORD *)v11 > *(_DWORD *)v20 )
          goto LABEL_58;
        v11 += 4;
        v20 += 4;
        if ( v10 == v11 )
          goto LABEL_57;
      }
      goto LABEL_18;
    }
LABEL_57:
    if ( v46 != v20 )
    {
LABEL_18:
      v21 = *a1;
      *a1 = *v7;
      *v7 = v21;
      v6 = *a1;
      v22 = a1[1];
      v23 = *(v5 - 1);
      goto LABEL_19;
    }
LABEL_58:
    if ( v44 > v18 )
      v8 = &v9[v18];
    if ( v9 == v8 )
    {
LABEL_77:
      if ( v46 == v17 )
      {
        v22 = *a1;
        *a1 = v6;
        a1[1] = v22;
        v23 = *(v5 - 1);
        goto LABEL_19;
      }
    }
    else
    {
      while ( *(_DWORD *)v9 >= *(_DWORD *)v17 )
      {
        if ( *(_DWORD *)v9 > *(_DWORD *)v17 )
          goto LABEL_56;
        v9 += 4;
        v17 += 4;
        if ( v8 == v9 )
          goto LABEL_77;
      }
    }
    v23 = *a1;
    *a1 = v16;
    *(v5 - 1) = v23;
    v6 = *a1;
    v22 = a1[1];
LABEL_19:
    v24 = *(char **)(v6 + 104);
    v25 = *(char **)(v6 + 96);
    v26 = v5;
    for ( i = v43; ; ++i )
    {
LABEL_20:
      v28 = *(char **)(v22 + 104);
      v29 = *(char **)(v22 + 96);
      v30 = i - 1;
      if ( v28 - v29 > v24 - v25 )
        v28 = &v29[v24 - v25];
      v31 = v25;
      if ( v29 == v28 )
        break;
      while ( *(_DWORD *)v29 >= *(_DWORD *)v31 )
      {
        if ( *(_DWORD *)v29 > *(_DWORD *)v31 )
          goto LABEL_29;
        v29 += 4;
        v31 += 4;
        if ( v28 == v29 )
          goto LABEL_28;
      }
LABEL_27:
      v22 = *i;
    }
LABEL_28:
    if ( v31 != v24 )
      goto LABEL_27;
LABEL_29:
    for ( --v26; ; v23 = *v26 )
    {
      v32 = *(char **)(v23 + 104);
      v33 = *(char **)(v23 + 96);
      v34 = &v25[v32 - v33];
      if ( v24 - v25 <= v32 - v33 )
        v34 = v24;
      if ( v34 == v25 )
        break;
      v35 = v25;
      while ( *(_DWORD *)v35 >= *(_DWORD *)v33 )
      {
        if ( *(_DWORD *)v35 > *(_DWORD *)v33 )
          goto LABEL_40;
        v35 += 4;
        v33 += 4;
        if ( v34 == v35 )
        {
          v32 = *(char **)(v23 + 104);
          goto LABEL_39;
        }
      }
LABEL_37:
      --v26;
    }
LABEL_39:
    if ( v33 != v32 )
      goto LABEL_37;
LABEL_40:
    if ( v26 > v30 )
    {
      *(i++ - 1) = v23;
      v23 = *(v26 - 1);
      *v26 = v22;
      v22 = *(i - 1);
      v24 = *(char **)(*a1 + 104);
      v25 = *(char **)(*a1 + 96);
      goto LABEL_20;
    }
    sub_39A97E0(i - 1, v5, v45);
    result = (char *)v30 - (char *)a1;
    if ( (char *)v30 - (char *)a1 > 128 )
    {
      v5 = v30;
      if ( !v45 )
        goto LABEL_44;
      continue;
    }
    return result;
  }
}
