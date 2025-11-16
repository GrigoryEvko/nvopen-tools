// Function: sub_1601B20
// Address: 0x1601b20
//
__int64 __fastcall sub_1601B20(const char **a1, __int64 a2, _BYTE *a3, size_t a4)
{
  __int64 v4; // rbx
  __int64 v5; // rdi
  size_t v6; // rdx
  _BYTE *v7; // rax
  size_t v8; // rax
  size_t v9; // r15
  __int64 v10; // r14
  const char *v11; // rbx
  __int64 v12; // r13
  char *v13; // r12
  const char *v14; // r12
  size_t v15; // rax
  size_t v16; // rbx
  unsigned int v17; // r13d
  __int64 v19; // r12
  char *v20; // r14
  char *v21; // rbx
  __int64 i; // rbx
  char *v23; // r13
  const char **v24; // rax
  const char **v27; // [rsp+10h] [rbp-80h]
  char *v28; // [rsp+18h] [rbp-78h]
  const char **v29; // [rsp+20h] [rbp-70h]
  char *v30; // [rsp+20h] [rbp-70h]
  unsigned __int64 v33; // [rsp+38h] [rbp-58h]
  __int64 v34; // [rsp+40h] [rbp-50h]
  char *v35; // [rsp+40h] [rbp-50h]
  char *v36; // [rsp+48h] [rbp-48h]
  __int64 v37; // [rsp+50h] [rbp-40h]
  char *s2; // [rsp+58h] [rbp-38h]
  char *s2a; // [rsp+58h] [rbp-38h]

  v4 = 8 * a2;
  v27 = &a1[a2];
  if ( a4 <= 4 )
  {
    v24 = a1;
    v30 = (char *)a1;
  }
  else
  {
    v29 = a1;
    v37 = 4;
    if ( v4 <= 0 )
      goto LABEL_15;
    while ( 1 )
    {
      v5 = v37 + 1;
      if ( v37 + 1 >= a4 )
      {
        v33 = a4;
      }
      else
      {
        v6 = a4 - v5;
        if ( (__int64)(a4 - v5) < 0 )
          v6 = 0x7FFFFFFFFFFFFFFFLL;
        v7 = memchr(&a3[v5], 46, v6);
        v33 = a4;
        if ( v7 )
        {
          v8 = v7 - a3;
          if ( v8 == -1 )
            v8 = a4;
          v33 = v8;
        }
      }
      v9 = v33 - v37;
      v36 = (char *)v29;
      v10 = v4 >> 3;
      v11 = &a3[v37];
      while ( 1 )
      {
        while ( 1 )
        {
          v12 = v10 >> 1;
          v13 = &v36[8 * (v10 >> 1)];
          s2 = (char *)(*(_QWORD *)v13 + v37);
          if ( strncmp(s2, v11, v9) >= 0 )
            break;
          v36 = v13 + 8;
          v10 = v10 - v12 - 1;
          if ( v10 <= 0 )
            goto LABEL_15;
        }
        if ( strncmp(v11, s2, v9) >= 0 )
          break;
        v10 >>= 1;
        if ( v12 <= 0 )
          goto LABEL_15;
      }
      v28 = &v36[8 * (v10 >> 1)];
      v34 = v10;
      v19 = (8 * (v10 >> 1)) >> 3;
      s2a = v36;
      while ( v19 > 0 )
      {
        while ( 1 )
        {
          v20 = &s2a[8 * (v19 >> 1)];
          if ( strncmp((const char *)(*(_QWORD *)v20 + v37), v11, v9) >= 0 )
            break;
          v19 = v19 - (v19 >> 1) - 1;
          s2a = v20 + 8;
          if ( v19 <= 0 )
            goto LABEL_25;
        }
        v19 >>= 1;
      }
LABEL_25:
      v21 = &v36[8 * v34];
      v35 = v28 + 8;
      for ( i = (v21 - (v28 + 8)) >> 3; i > 0; i >>= 1 )
      {
        while ( 1 )
        {
          v23 = &v35[8 * (i >> 1)];
          if ( strncmp(&a3[v37], (const char *)(*(_QWORD *)v23 + v37), v9) < 0 )
            break;
          i = i - (i >> 1) - 1;
          v35 = v23 + 8;
          if ( i <= 0 )
            goto LABEL_29;
        }
      }
LABEL_29:
      v4 = v35 - s2a;
      if ( a4 <= v33 )
        break;
      if ( v4 <= 0 )
        goto LABEL_15;
      v29 = (const char **)s2a;
      v37 = v33;
    }
    v24 = v29;
    v30 = s2a;
  }
  if ( v4 > 0 )
    v24 = (const char **)v30;
  v29 = v24;
LABEL_15:
  if ( v29 == v27 )
    return (unsigned int)-1;
  v14 = *v29;
  if ( !*v29 )
  {
    if ( !a4 )
      return (unsigned int)(v29 - a1);
LABEL_38:
    if ( *a3 == 46 )
      return (unsigned int)(v29 - a1);
    return (unsigned int)-1;
  }
  v15 = strlen(*v29);
  v16 = v15;
  if ( v15 == a4 )
  {
    if ( !a4 || !memcmp(a3, v14, a4) )
      return (unsigned int)(v29 - a1);
    return (unsigned int)-1;
  }
  v17 = -1;
  if ( v15 <= a4 )
  {
    if ( v15 )
    {
      if ( memcmp(a3, v14, v15) )
        return v17;
      if ( a3[v16] != 46 )
        return (unsigned int)-1;
      return (unsigned int)(v29 - a1);
    }
    goto LABEL_38;
  }
  return v17;
}
