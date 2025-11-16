// Function: sub_33F8310
// Address: 0x33f8310
//
__int64 __fastcall sub_33F8310(_QWORD *a1, __int64 a2, __int64 a3)
{
  size_t v4; // r13
  size_t v5; // r14
  const void *v6; // r15
  void *v7; // r9
  size_t v8; // rdx
  int v9; // eax
  signed __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rax
  void *v13; // rcx
  const void *v14; // r9
  __int64 v15; // r14
  size_t v16; // rdx
  int v17; // eax
  signed __int64 v18; // rax
  __int64 result; // rax
  signed __int64 v20; // rax
  int v21; // eax
  signed __int64 v22; // rax
  signed __int64 v23; // rax
  __int64 v24; // rax
  void *v25; // rcx
  const void *v26; // r9
  __int64 v27; // r14
  size_t v28; // rdx
  int v29; // eax
  signed __int64 v30; // rax
  signed __int64 v31; // rax
  __int64 v32; // rbx
  size_t v33; // r14
  const void *v34; // r10
  size_t v35; // r13
  size_t v36; // r15
  const void *v37; // r9
  int v38; // eax
  int v39; // eax
  signed __int64 v40; // rax
  __int64 v41; // r14
  signed __int64 v42; // rax
  void *v43; // [rsp+0h] [rbp-50h]
  void *v44; // [rsp+0h] [rbp-50h]
  size_t n; // [rsp+8h] [rbp-48h]
  size_t na; // [rsp+8h] [rbp-48h]
  size_t nb; // [rsp+8h] [rbp-48h]
  size_t nc; // [rsp+8h] [rbp-48h]
  size_t nd; // [rsp+8h] [rbp-48h]
  size_t ne; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+10h] [rbp-40h]
  void *s1a; // [rsp+10h] [rbp-40h]
  void *s1b; // [rsp+10h] [rbp-40h]
  void *s1c; // [rsp+10h] [rbp-40h]
  void *s1d; // [rsp+10h] [rbp-40h]
  void *s1e; // [rsp+10h] [rbp-40h]
  void *s1f; // [rsp+10h] [rbp-40h]
  void *s1g; // [rsp+10h] [rbp-40h]

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_33F8090((__int64)a1, a3);
    v32 = a1[4];
    v33 = *(_QWORD *)(a3 + 8);
    v34 = *(const void **)a3;
    v35 = *(_QWORD *)(v32 + 40);
    v36 = v33;
    v37 = *(const void **)(v32 + 32);
    if ( v35 <= v33 )
      v36 = *(_QWORD *)(v32 + 40);
    if ( v36 )
    {
      ne = *(_QWORD *)a3;
      s1g = *(void **)(v32 + 32);
      v38 = memcmp(s1g, *(const void **)a3, v36);
      v37 = s1g;
      v34 = (const void *)ne;
      if ( v38 )
      {
        if ( v38 < 0 )
          return 0;
LABEL_67:
        v39 = memcmp(v34, v37, v36);
        if ( v39 )
          goto LABEL_68;
        goto LABEL_87;
      }
      v40 = v35 - v33;
      if ( (__int64)(v35 - v33) > 0x7FFFFFFF )
        goto LABEL_67;
    }
    else
    {
      v40 = v35 - v33;
      if ( (__int64)(v35 - v33) > 0x7FFFFFFF )
        goto LABEL_87;
    }
    if ( v40 < (__int64)0xFFFFFFFF80000000LL || (int)v40 < 0 )
      return 0;
    if ( v36 )
      goto LABEL_67;
LABEL_87:
    v41 = v33 - v35;
    if ( v41 > 0x7FFFFFFF )
    {
LABEL_69:
      if ( *(_DWORD *)(v32 + 64) >= *(_DWORD *)(a3 + 32) )
        return sub_33F8090((__int64)a1, a3);
      return 0;
    }
    if ( v41 < (__int64)0xFFFFFFFF80000000LL )
      return sub_33F8090((__int64)a1, a3);
    v39 = v41;
LABEL_68:
    if ( v39 < 0 )
      return sub_33F8090((__int64)a1, a3);
    goto LABEL_69;
  }
  v4 = *(_QWORD *)(a3 + 8);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(const void **)a3;
  v7 = *(void **)(a2 + 32);
  v8 = v5;
  if ( v4 <= v5 )
    v8 = v4;
  if ( v8 )
  {
    n = v8;
    s1 = *(void **)(a2 + 32);
    v9 = memcmp(v6, s1, v8);
    v7 = s1;
    v8 = n;
    if ( v9 )
    {
      if ( v9 < 0 )
        goto LABEL_44;
      goto LABEL_10;
    }
    v10 = v4 - v5;
    if ( (__int64)(v4 - v5) > 0x7FFFFFFF )
      goto LABEL_10;
  }
  else
  {
    v10 = v4 - v5;
    if ( (__int64)(v4 - v5) > 0x7FFFFFFF )
      goto LABEL_27;
  }
  if ( v10 < (__int64)0xFFFFFFFF80000000LL || (int)v10 < 0 )
    goto LABEL_44;
  if ( !v8 )
    goto LABEL_27;
LABEL_10:
  na = v8;
  s1a = v7;
  v11 = memcmp(v7, v6, v8);
  v7 = s1a;
  v8 = na;
  if ( !v11 )
  {
LABEL_27:
    v20 = v5 - v4;
    if ( (__int64)(v5 - v4) <= 0x7FFFFFFF && (v20 < (__int64)0xFFFFFFFF80000000LL || (int)v20 < 0) )
    {
      if ( !v8 )
      {
        v22 = v5 - v4;
LABEL_34:
        if ( v22 < (__int64)0xFFFFFFFF80000000LL || (int)v22 < 0 )
          goto LABEL_12;
        if ( v8 )
        {
LABEL_37:
          LODWORD(v23) = memcmp(v6, v7, v8);
          if ( (_DWORD)v23 )
            goto LABEL_40;
        }
LABEL_38:
        v23 = v4 - v5;
        if ( (__int64)(v4 - v5) > 0x7FFFFFFF )
          goto LABEL_41;
        if ( v23 < (__int64)0xFFFFFFFF80000000LL )
          return a2;
LABEL_40:
        if ( (int)v23 >= 0 )
        {
LABEL_41:
          if ( *(_DWORD *)(a2 + 64) >= *(_DWORD *)(a3 + 32) )
            return a2;
LABEL_12:
          if ( a1[4] != a2 )
          {
            v12 = sub_220EEE0(a2);
            v13 = *(void **)(v12 + 40);
            v14 = *(const void **)(v12 + 32);
            v15 = v12;
            v16 = (size_t)v13;
            if ( v4 <= (unsigned __int64)v13 )
              v16 = v4;
            if ( v16 )
            {
              v43 = *(void **)(v12 + 40);
              nb = v16;
              s1b = *(void **)(v12 + 32);
              v17 = memcmp(v6, s1b, v16);
              v14 = s1b;
              v16 = nb;
              v13 = v43;
              if ( v17 )
              {
                if ( v17 < 0 )
                  goto LABEL_23;
LABEL_18:
                s1c = v13;
                LODWORD(v18) = memcmp(v14, v6, v16);
                v13 = s1c;
                if ( (_DWORD)v18 )
                  goto LABEL_21;
                goto LABEL_19;
              }
              v42 = v4 - (_QWORD)v43;
              if ( (__int64)(v4 - (_QWORD)v43) > 0x7FFFFFFF )
                goto LABEL_18;
            }
            else
            {
              v42 = v4 - (_QWORD)v13;
              if ( (__int64)(v4 - (_QWORD)v13) > 0x7FFFFFFF )
                goto LABEL_19;
            }
            if ( v42 < (__int64)0xFFFFFFFF80000000LL || (int)v42 < 0 )
              goto LABEL_23;
            if ( v16 )
              goto LABEL_18;
LABEL_19:
            v18 = (signed __int64)v13 - v4;
            if ( (__int64)((__int64)v13 - v4) > 0x7FFFFFFF )
            {
LABEL_22:
              if ( *(_DWORD *)(a3 + 32) < *(_DWORD *)(v15 + 64) )
              {
LABEL_23:
                result = 0;
                if ( *(_QWORD *)(a2 + 24) )
                  return v15;
                return result;
              }
              return sub_33F8090((__int64)a1, a3);
            }
            if ( v18 < (__int64)0xFFFFFFFF80000000LL )
              return sub_33F8090((__int64)a1, a3);
LABEL_21:
            if ( (int)v18 < 0 )
              return sub_33F8090((__int64)a1, a3);
            goto LABEL_22;
          }
          return 0;
        }
        return a2;
      }
LABEL_32:
      nc = v8;
      s1d = v7;
      v21 = memcmp(v7, v6, v8);
      v7 = s1d;
      v8 = nc;
      if ( v21 )
      {
        if ( v21 < 0 )
          goto LABEL_12;
        goto LABEL_37;
      }
      v22 = v5 - v4;
      if ( (__int64)(v5 - v4) > 0x7FFFFFFF )
        goto LABEL_37;
      goto LABEL_34;
    }
    goto LABEL_30;
  }
  if ( v11 < 0 )
    goto LABEL_12;
LABEL_30:
  if ( *(_DWORD *)(a3 + 32) >= *(_DWORD *)(a2 + 64) )
  {
    if ( !v8 )
    {
      v22 = v5 - v4;
      if ( (__int64)(v5 - v4) > 0x7FFFFFFF )
        goto LABEL_38;
      goto LABEL_34;
    }
    goto LABEL_32;
  }
LABEL_44:
  if ( a1[3] == a2 )
    return a2;
  v24 = sub_220EF80(a2);
  v25 = *(void **)(v24 + 40);
  v26 = *(const void **)(v24 + 32);
  v27 = v24;
  v28 = (size_t)v25;
  if ( v4 <= (unsigned __int64)v25 )
    v28 = v4;
  if ( !v28 )
  {
    v30 = (signed __int64)v25 - v4;
    if ( (__int64)((__int64)v25 - v4) > 0x7FFFFFFF )
      goto LABEL_54;
LABEL_50:
    if ( v30 < (__int64)0xFFFFFFFF80000000LL || (int)v30 < 0 )
      goto LABEL_58;
    if ( !v28 )
      goto LABEL_54;
    goto LABEL_53;
  }
  v44 = *(void **)(v24 + 40);
  nd = v28;
  s1e = *(void **)(v24 + 32);
  v29 = memcmp(s1e, v6, v28);
  v26 = s1e;
  v28 = nd;
  v25 = v44;
  if ( v29 )
  {
    if ( v29 < 0 )
      goto LABEL_58;
  }
  else
  {
    v30 = (signed __int64)v44 - v4;
    if ( (__int64)((__int64)v44 - v4) <= 0x7FFFFFFF )
      goto LABEL_50;
  }
LABEL_53:
  s1f = v25;
  LODWORD(v31) = memcmp(v6, v26, v28);
  v25 = s1f;
  if ( (_DWORD)v31 )
  {
LABEL_56:
    if ( (int)v31 < 0 )
      return sub_33F8090((__int64)a1, a3);
    goto LABEL_57;
  }
LABEL_54:
  v31 = v4 - (_QWORD)v25;
  if ( (__int64)(v4 - (_QWORD)v25) <= 0x7FFFFFFF )
  {
    if ( v31 < (__int64)0xFFFFFFFF80000000LL )
      return sub_33F8090((__int64)a1, a3);
    goto LABEL_56;
  }
LABEL_57:
  if ( *(_DWORD *)(v27 + 64) >= *(_DWORD *)(a3 + 32) )
    return sub_33F8090((__int64)a1, a3);
LABEL_58:
  result = 0;
  if ( *(_QWORD *)(v27 + 24) )
    return a2;
  return result;
}
