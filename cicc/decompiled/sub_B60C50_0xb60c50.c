// Function: sub_B60C50
// Address: 0xb60c50
//
__int64 __fastcall sub_B60C50(_BYTE *a1, size_t a2)
{
  char ***v2; // rax
  char ***v3; // rbx
  void (**v4)(); // rax
  void (**v5)(); // r14
  unsigned int v6; // eax
  size_t v7; // rdx
  __int64 v8; // rax
  size_t v9; // rax
  char *v10; // rsi
  size_t v11; // r14
  char (**v12)[5]; // rcx
  __int64 v13; // r12
  size_t v14; // rdx
  char (**v15)[5]; // rbx
  size_t v16; // r13
  int v17; // eax
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  _BYTE *v20; // rax
  size_t v21; // rax
  size_t v22; // r15
  __int64 v23; // r14
  const char *v24; // rbx
  __int64 v25; // r13
  __int64 *v26; // r12
  const char *v27; // r12
  size_t v28; // rax
  size_t v29; // rbx
  unsigned int v30; // r12d
  __int64 v32; // r12
  __int64 *v33; // r14
  __int64 *v34; // rbx
  __int64 j; // rbx
  char *v36; // r13
  __int64 v37; // rax
  __int64 *v38; // rax
  _BYTE *v39; // [rsp+0h] [rbp-B0h]
  __int64 *v40; // [rsp+8h] [rbp-A8h]
  __int64 *v41; // [rsp+10h] [rbp-A0h]
  __int64 *v42; // [rsp+18h] [rbp-98h]
  __int64 *v43; // [rsp+20h] [rbp-90h]
  size_t v45; // [rsp+30h] [rbp-80h]
  __int64 v47; // [rsp+40h] [rbp-70h]
  char *v48; // [rsp+40h] [rbp-70h]
  __int64 *v49; // [rsp+48h] [rbp-68h]
  char (**v50)[5]; // [rsp+50h] [rbp-60h]
  unsigned __int64 i; // [rsp+50h] [rbp-60h]
  const char *s2; // [rsp+58h] [rbp-58h]
  char (**s2a)[5]; // [rsp+58h] [rbp-58h]
  __int64 *s2b; // [rsp+58h] [rbp-58h]
  char v55; // [rsp+6Fh] [rbp-41h] BYREF
  char *v56; // [rsp+70h] [rbp-40h] BYREF
  size_t n; // [rsp+78h] [rbp-38h]

  v56 = &v55;
  v2 = (char ***)(__readfsqword(0) - 24);
  *v2 = &v56;
  v3 = v2;
  v4 = (void (**)())(__readfsqword(0) - 32);
  *v4 = sub_B5B9E0;
  if ( !&_pthread_key_create )
  {
    v6 = -1;
    goto LABEL_82;
  }
  v5 = v4;
  v6 = pthread_once(&dword_4F818F8, init_routine);
  if ( v6
    || (v56 = &v55, *v3 = &v56, v39 = a1, *v5 = sub_B5B9E0, (v6 = pthread_once(&dword_4F818F8, init_routine)) != 0) )
  {
LABEL_82:
    sub_4264C5(v6);
  }
  if ( a2 <= 4 )
  {
    v8 = a2;
    v7 = 0;
  }
  else
  {
    v7 = a2 - 5;
    v8 = 5;
  }
  n = v7;
  v56 = &a1[v8];
  v55 = 46;
  v9 = sub_C931B0(&v56, &v55, 1, 0);
  v10 = v56;
  v11 = v9;
  if ( v9 == -1 )
  {
    v11 = n;
  }
  else if ( n <= v9 )
  {
    v11 = n;
  }
  v12 = &off_4979780;
  v13 = 19;
  do
  {
    while ( 1 )
    {
      v14 = v11;
      v15 = &v12[4 * (v13 >> 1)];
      v16 = (size_t)v15[1];
      if ( v16 <= v11 )
        v14 = (size_t)v15[1];
      if ( v14 )
      {
        v50 = v12;
        v17 = memcmp(*v15, v10, v14);
        v12 = v50;
        if ( v17 )
          break;
      }
      if ( v16 == v11 || v16 >= v11 )
      {
        v13 >>= 1;
        goto LABEL_18;
      }
LABEL_10:
      v12 = v15 + 4;
      v13 = v13 - (v13 >> 1) - 1;
      if ( v13 <= 0 )
        goto LABEL_19;
    }
    if ( v17 < 0 )
      goto LABEL_10;
    v13 >>= 1;
LABEL_18:
    ;
  }
  while ( v13 > 0 );
LABEL_19:
  if ( v12 != (char (**)[5])off_49799E0 && v12[1] == (char (*)[5])v11 )
  {
    if ( v11 )
    {
      s2a = v12;
      if ( memcmp(*v12, v10, v11) )
        goto LABEL_21;
      v18 = 8LL * (_QWORD)s2a[3];
      v43 = &qword_4B91188[(_QWORD)s2a[2]];
      i = v11 + 5;
    }
    else
    {
      i = 4;
      v18 = 8LL * (_QWORD)v12[3];
      v43 = &qword_4B91188[(_QWORD)v12[2]];
    }
    v41 = (__int64 *)((char *)v43 + v18);
    if ( i >= a2 )
      goto LABEL_80;
    v40 = v43;
    if ( v18 > 0 )
    {
      v19 = i + 1;
      if ( i + 1 >= a2 )
        goto LABEL_47;
      goto LABEL_24;
    }
  }
  else
  {
LABEL_21:
    if ( a2 > 4 )
    {
      v40 = qword_4B91188;
      v18 = 3968;
      v41 = &qword_4B92108;
      v43 = qword_4B91188;
      for ( i = 4; ; i = v45 )
      {
        v19 = i + 1;
        if ( i + 1 < a2 )
        {
LABEL_24:
          v20 = memchr(&a1[v19], 46, a2 - v19);
          if ( v20 )
          {
            v21 = v20 - a1;
            if ( v21 == -1 )
              v21 = a2;
            v45 = v21;
            goto LABEL_28;
          }
        }
LABEL_47:
        v45 = a2;
LABEL_28:
        v22 = v45 - i;
        v49 = v43;
        v23 = v18 >> 3;
        v24 = &a1[i];
        while ( 1 )
        {
          while ( 1 )
          {
            v25 = v23 >> 1;
            v26 = &v49[v23 >> 1];
            s2 = (const char *)(*v26 + i);
            if ( strncmp(s2, v24, v22) >= 0 )
              break;
            v49 = v26 + 1;
            v23 = v23 - v25 - 1;
            if ( v23 <= 0 )
              goto LABEL_33;
          }
          if ( strncmp(v24, s2, v22) >= 0 )
            break;
          v23 >>= 1;
          if ( v25 <= 0 )
            goto LABEL_33;
        }
        v42 = &v49[v23 >> 1];
        v47 = v23;
        v32 = (8 * (v23 >> 1)) >> 3;
        s2b = v49;
        while ( v32 > 0 )
        {
          while ( 1 )
          {
            v33 = &s2b[v32 >> 1];
            if ( strncmp((const char *)(*v33 + i), v24, v22) >= 0 )
              break;
            v32 = v32 - (v32 >> 1) - 1;
            s2b = v33 + 1;
            if ( v32 <= 0 )
              goto LABEL_52;
          }
          v32 >>= 1;
        }
LABEL_52:
        v34 = &v49[v47];
        v48 = (char *)(v42 + 1);
        for ( j = v34 - (v42 + 1); j > 0; j >>= 1 )
        {
          while ( 1 )
          {
            v36 = &v48[8 * (j >> 1)];
            if ( strncmp(&a1[i], (const char *)(*(_QWORD *)v36 + i), v22) < 0 )
              break;
            j = j - (j >> 1) - 1;
            v48 = v36 + 8;
            if ( j <= 0 )
              goto LABEL_56;
          }
        }
LABEL_56:
        v18 = v48 - (char *)s2b;
        if ( v45 >= a2 )
        {
          v38 = v43;
          v43 = s2b;
          goto LABEL_76;
        }
        if ( v18 <= 0 )
          goto LABEL_33;
        v43 = s2b;
      }
    }
    v43 = qword_4B91188;
    v41 = &qword_4B92108;
LABEL_80:
    v38 = v43;
    v40 = v43;
    v18 = (char *)v41 - (char *)v43;
LABEL_76:
    if ( v18 > 0 )
      v38 = v43;
    v43 = v38;
  }
LABEL_33:
  if ( v43 == v41 )
    return 0;
  v27 = (const char *)*v43;
  if ( !*v43 )
  {
    if ( a2 )
      goto LABEL_70;
    goto LABEL_63;
  }
  v28 = strlen((const char *)*v43);
  v29 = v28;
  if ( v28 == a2 )
  {
    if ( a2 && memcmp(a1, v27, a2) )
      goto LABEL_73;
    goto LABEL_63;
  }
  if ( v28 > a2 )
    return 0;
  if ( !v28 )
  {
LABEL_70:
    if ( *v39 != 46 )
      return 0;
LABEL_63:
    v37 = v43 - v40;
    if ( (_DWORD)v37 == -1 )
      return 0;
    v30 = v37 + (((char *)v40 - (char *)&off_4B91180) >> 3);
    if ( strlen((const char *)v40[(int)v37]) != a2 && !(unsigned __int8)sub_B60C20(v30) )
      return 0;
    return v30;
  }
LABEL_73:
  if ( !memcmp(a1, v27, v29) )
  {
    v39 = &a1[v29];
    goto LABEL_70;
  }
  return 0;
}
