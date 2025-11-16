// Function: sub_183FEC0
// Address: 0x183fec0
//
__int64 __fastcall sub_183FEC0(size_t *a1, __int64 a2)
{
  const void *v4; // rdi
  unsigned __int64 v5; // r12
  char **v6; // rdi
  __int64 v7; // r15
  __int64 v8; // rax
  _QWORD *v9; // r14
  unsigned __int64 v10; // r13
  _QWORD *v11; // rax
  char v12; // si
  unsigned __int64 v13; // rcx
  _QWORD *v14; // rsi
  _BOOL4 v15; // r10d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // r8d
  int v20; // r9d
  __int64 i; // r13
  _QWORD *v22; // rbx
  _QWORD *v23; // r12
  __int64 v24; // rdi
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // r14
  _QWORD *v29; // r15
  size_t v30; // rdx
  _BYTE *v31; // rax
  _BYTE *v32; // rsi
  int v33; // r12d
  size_t v34; // r9
  _BYTE *v35; // r13
  __int64 v36; // rax
  void *v37; // r10
  void *v38; // rax
  unsigned __int64 v39; // r12
  char *v40; // rdi
  const void *v41; // rsi
  __int64 v42; // r12
  int v43; // eax
  _QWORD *v44; // [rsp+10h] [rbp-90h]
  char v45; // [rsp+1Ch] [rbp-84h]
  _BOOL4 v46; // [rsp+1Ch] [rbp-84h]
  size_t n; // [rsp+28h] [rbp-78h]
  size_t nb; // [rsp+28h] [rbp-78h]
  size_t na; // [rsp+28h] [rbp-78h]
  size_t nc; // [rsp+28h] [rbp-78h]
  __int64 v52; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v53; // [rsp+38h] [rbp-68h]
  __int64 v54; // [rsp+40h] [rbp-60h]
  unsigned int v55; // [rsp+48h] [rbp-58h]
  char **v56; // [rsp+50h] [rbp-50h] BYREF
  __int64 v57; // [rsp+58h] [rbp-48h]
  char *v58; // [rsp+60h] [rbp-40h] BYREF
  char *v59; // [rsp+68h] [rbp-38h]

  v4 = (const void *)*a1;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  sub_183FAF0((size_t)v4, a2, (__int64)&v52, a1);
  if ( (_DWORD)v54 )
  {
    v27 = v53;
    v28 = &v53[5 * v55];
    if ( v53 != v28 )
    {
      while ( 1 )
      {
        v29 = v27;
        if ( *v27 != -2 && *v27 != -16 )
          break;
        v27 += 5;
        if ( v28 == v27 )
          goto LABEL_2;
      }
      while ( v29 != v28 )
      {
        v30 = *a1;
        v31 = *(_BYTE **)(*a1 + 88);
        v32 = *(_BYTE **)(*a1 + 80);
        v33 = *(_DWORD *)(*a1 + 72);
        v34 = v31 - v32;
        v35 = (_BYTE *)(v31 - v32);
        if ( v31 == v32 )
        {
          v37 = 0;
        }
        else
        {
          if ( v34 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_86;
          v4 = (const void *)(*(_QWORD *)(*a1 + 88) - (_QWORD)v32);
          nb = *a1;
          v36 = sub_22077B0(v34);
          v30 = nb;
          v37 = (void *)v36;
          v31 = *(_BYTE **)(nb + 88);
          v32 = *(_BYTE **)(nb + 80);
          v34 = v31 - v32;
        }
        if ( v32 == v31 )
        {
          if ( v33 != *((_DWORD *)v29 + 2) || (v4 = (const void *)v29[2], v29[3] - (_QWORD)v4 != v34) )
          {
            if ( v37 )
              goto LABEL_57;
            goto LABEL_58;
          }
          if ( v34 )
          {
LABEL_76:
            nc = (size_t)v37;
            v43 = memcmp(v4, v37, v34);
            v37 = (void *)nc;
            if ( !v43 )
              goto LABEL_77;
LABEL_57:
            v32 = v35;
            v4 = v37;
            j_j___libc_free_0(v37, v35);
LABEL_58:
            LODWORD(v56) = *((_DWORD *)v29 + 2);
            v39 = v29[3] - v29[2];
            v57 = 0;
            v58 = 0;
            v59 = 0;
            if ( v39 )
            {
              if ( v39 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_86:
                sub_4261EA(v4, v32, v30);
              v40 = (char *)sub_22077B0(v39);
            }
            else
            {
              v39 = 0;
              v40 = 0;
            }
            v57 = (__int64)v40;
            v59 = &v40[v39];
            v58 = v40;
            v41 = (const void *)v29[2];
            v42 = v29[3] - (_QWORD)v41;
            if ( (const void *)v29[3] != v41 )
              v40 = (char *)memmove(v40, v41, v29[3] - (_QWORD)v41);
            v58 = &v40[v42];
            sub_183BFC0((__int64)a1, *v29, (__int64)&v56);
            v4 = (const void *)v57;
            if ( v57 )
              j_j___libc_free_0(v57, &v59[-v57]);
            goto LABEL_65;
          }
        }
        else
        {
          na = v34;
          v38 = memmove(v37, v32, v34);
          v34 = na;
          v37 = v38;
          if ( v33 != *((_DWORD *)v29 + 2) )
            goto LABEL_57;
          v4 = (const void *)v29[2];
          if ( na != v29[3] - (_QWORD)v4 )
            goto LABEL_57;
          if ( na )
            goto LABEL_76;
        }
        if ( v37 )
        {
LABEL_77:
          v4 = v37;
          j_j___libc_free_0(v37, v35);
        }
LABEL_65:
        v29 += 5;
        if ( v29 == v28 )
          break;
        while ( *v29 == -2 || *v29 == -16 )
        {
          v29 += 5;
          if ( v28 == v29 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 16) - 25 > 9 )
    goto LABEL_27;
  v56 = &v58;
  v57 = 0x1000000000LL;
  sub_183D8F0((unsigned __int64)a1, a2, &v56, 1);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = v56;
  if ( !(_DWORD)v57 )
    goto LABEL_25;
  v7 = 0;
  n = (unsigned int)v57;
  v44 = a1 + 159;
  do
  {
    while ( !*((_BYTE *)v6 + v7) )
    {
      if ( n == ++v7 )
        goto LABEL_25;
    }
    v45 = *((_BYTE *)v6 + v7);
    v8 = sub_15F4DF0(a2, v7);
    v9 = (_QWORD *)a1[160];
    v10 = v8;
    if ( !v9 )
    {
      v9 = a1 + 159;
      if ( v44 != (_QWORD *)a1[161] )
        goto LABEL_36;
      v9 = a1 + 159;
      v15 = 1;
      goto LABEL_18;
    }
    while ( 1 )
    {
      v13 = v9[4];
      if ( v5 < v13 || v5 == v13 && v10 < v9[5] )
        break;
      v11 = (_QWORD *)v9[3];
      v12 = 0;
      if ( !v11 )
        goto LABEL_15;
LABEL_12:
      v9 = v11;
    }
    v11 = (_QWORD *)v9[2];
    v12 = v45;
    if ( v11 )
      goto LABEL_12;
LABEL_15:
    if ( !v12 )
    {
      v14 = v9;
      if ( v5 > v13 )
        goto LABEL_17;
LABEL_38:
      if ( v5 == v13 && v10 > v9[5] )
      {
        v9 = v14;
        goto LABEL_41;
      }
LABEL_23:
      v6 = v56;
      goto LABEL_24;
    }
    if ( v9 != (_QWORD *)a1[161] )
    {
LABEL_36:
      v26 = sub_220EF80(v9);
      v13 = *(_QWORD *)(v26 + 32);
      if ( v13 >= v5 )
      {
        v14 = v9;
        v9 = (_QWORD *)v26;
        goto LABEL_38;
      }
LABEL_41:
      if ( !v9 )
      {
        v6 = v56;
        goto LABEL_24;
      }
    }
LABEL_17:
    v15 = 1;
    if ( v44 != v9 && v5 >= v9[4] )
    {
      v15 = 0;
      if ( v5 == v9[4] )
        v15 = v10 < v9[5];
    }
LABEL_18:
    v46 = v15;
    v16 = sub_22077B0(48);
    *(_QWORD *)(v16 + 32) = v5;
    *(_QWORD *)(v16 + 40) = v10;
    sub_220F040(v46, v16, v9, v44);
    ++a1[163];
    if ( sub_183E920((__int64)(a1 + 5), v10) )
    {
      for ( i = *(_QWORD *)(v10 + 48); ; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        if ( *(_BYTE *)(i - 8) != 77 )
          break;
        sub_183D9D0(a1, i - 24);
      }
      goto LABEL_23;
    }
    sub_183B530((__int64)a1, v10, v17, v18, v19, v20);
    v6 = v56;
LABEL_24:
    ++v7;
  }
  while ( n != v7 );
LABEL_25:
  if ( v6 != &v58 )
    _libc_free((unsigned __int64)v6);
LABEL_27:
  if ( v55 )
  {
    v22 = v53;
    v23 = &v53[5 * v55];
    do
    {
      if ( *v22 != -16 && *v22 != -2 )
      {
        v24 = v22[2];
        if ( v24 )
          j_j___libc_free_0(v24, v22[4] - v24);
      }
      v22 += 5;
    }
    while ( v23 != v22 );
  }
  return j___libc_free_0(v53);
}
