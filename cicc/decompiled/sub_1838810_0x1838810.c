// Function: sub_1838810
// Address: 0x1838810
//
int __fastcall sub_1838810(char **a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  _QWORD *v5; // rsi
  _QWORD *v6; // r14
  char *v7; // r9
  char *v8; // rdi
  _QWORD *v9; // r11
  _QWORD *v10; // rdx
  signed __int64 v11; // rax
  char *v12; // rcx
  char *v13; // rax
  const void *v14; // rdi
  size_t v15; // rdx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // r8
  _QWORD *v20; // r14
  bool v21; // r15
  __int64 v22; // rax
  char *v23; // rsi
  _QWORD *v24; // r8
  char *v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  signed __int64 v28; // rdx
  char *v29; // rcx
  size_t v30; // r9
  char *v31; // rax
  unsigned __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r14
  const void *v37; // rsi
  size_t v38; // rdx
  char *v39; // rsi
  char *v40; // rcx
  char *v41; // rax
  char *v42; // rdx
  _QWORD *v44; // [rsp+0h] [rbp-40h]
  _QWORD *v45; // [rsp+0h] [rbp-40h]
  _QWORD *v46; // [rsp+8h] [rbp-38h]
  signed __int64 v47; // [rsp+8h] [rbp-38h]
  size_t v48; // [rsp+8h] [rbp-38h]

  v2 = a2 + 1;
  v5 = (_QWORD *)a2[2];
  v6 = (_QWORD *)a2[3];
  if ( !v5 )
  {
    v9 = v2;
    if ( v2 != v6 )
    {
LABEL_13:
      v6 = (_QWORD *)sub_220EFE0(v9);
      goto LABEL_14;
    }
    goto LABEL_19;
  }
  v7 = a1[1];
  v8 = *a1;
  v9 = v2;
  do
  {
    v10 = (_QWORD *)v5[4];
    v11 = v5[5] - (_QWORD)v10;
    v12 = &v8[v11];
    if ( v7 - v8 <= v11 )
      v12 = v7;
    if ( v8 != v12 )
    {
      v13 = v8;
      while ( *(_QWORD *)v13 >= *v10 )
      {
        if ( *(_QWORD *)v13 > *v10 )
          goto LABEL_38;
        v13 += 8;
        ++v10;
        if ( v12 == v13 )
          goto LABEL_37;
      }
LABEL_10:
      v9 = v5;
      v5 = (_QWORD *)v5[2];
      continue;
    }
LABEL_37:
    if ( (_QWORD *)v5[5] != v10 )
      goto LABEL_10;
LABEL_38:
    v5 = (_QWORD *)v5[3];
  }
  while ( v5 );
  if ( v9 != v6 )
    goto LABEL_13;
LABEL_14:
  if ( v6 == v2 )
  {
LABEL_19:
    v17 = sub_1838550(a2, v6, a1);
    v19 = v18;
    v20 = v17;
    if ( v18 )
    {
      v21 = v17 != 0 || v2 == v18;
      if ( !v21 )
      {
        v39 = (char *)v18[5];
        v40 = a1[1];
        v41 = *a1;
        v42 = (char *)v18[4];
        if ( v40 - *a1 > v39 - v42 )
          v40 = &(*a1)[v39 - v42];
        if ( v41 == v40 )
        {
LABEL_50:
          v21 = v39 != v42;
        }
        else
        {
          while ( *(_QWORD *)v41 >= *(_QWORD *)v42 )
          {
            if ( *(_QWORD *)v41 > *(_QWORD *)v42 )
              goto LABEL_21;
            v41 += 8;
            v42 += 8;
            if ( v40 == v41 )
              goto LABEL_50;
          }
          v21 = 1;
        }
      }
LABEL_21:
      v46 = v19;
      v22 = sub_22077B0(56);
      v23 = *a1;
      v24 = v46;
      v20 = (_QWORD *)v22;
      v25 = a1[1];
      v20[4] = 0;
      v20[5] = 0;
      v26 = v25 - v23;
      v20[6] = 0;
      if ( v25 == v23 )
      {
        v30 = 0;
        v28 = 0;
        v29 = 0;
      }
      else
      {
        if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(56, v23, v26);
        v44 = v46;
        v47 = v25 - v23;
        v27 = sub_22077B0(v26);
        v23 = *a1;
        v28 = v47;
        v29 = (char *)v27;
        v25 = a1[1];
        v24 = v44;
        v30 = v25 - *a1;
      }
      v20[4] = v29;
      v20[5] = v29;
      v20[6] = &v29[v28];
      if ( v25 != v23 )
      {
        v45 = v24;
        v48 = v30;
        v31 = (char *)memmove(v29, v23, v30);
        v24 = v45;
        v30 = v48;
        v29 = v31;
      }
      v20[5] = &v29[v30];
      sub_220F040(v21, v20, v24, v2);
      ++a2[5];
    }
    v16 = sub_220EF30(v20);
    v32 = v16;
    if ( (_QWORD *)v16 != v2 )
    {
      while ( 1 )
      {
        v37 = *(const void **)(v32 + 32);
        v38 = a1[1] - *a1;
        v16 = *(_QWORD *)(v32 + 40) - (_QWORD)v37;
        if ( v16 < v38 )
          break;
        if ( v38 )
        {
          LODWORD(v16) = memcmp(*a1, v37, v38);
          if ( (_DWORD)v16 )
            break;
        }
        v33 = sub_220EF30(v32);
        v34 = sub_220F330(v32, v2);
        v35 = *(_QWORD *)(v34 + 32);
        v36 = v34;
        if ( v35 )
          j_j___libc_free_0(v35, *(_QWORD *)(v34 + 48) - v35);
        LODWORD(v16) = j_j___libc_free_0(v36, 56);
        --a2[5];
        if ( (_QWORD *)v33 == v2 )
          break;
        v32 = v33;
      }
    }
    return v16;
  }
  v14 = (const void *)v6[4];
  v15 = v6[5] - (_QWORD)v14;
  v16 = a1[1] - *a1;
  if ( v16 < v15 || v15 && (LODWORD(v16) = memcmp(v14, *a1, v15), (_DWORD)v16) )
  {
    v6 = (_QWORD *)sub_220EF30(v6);
    goto LABEL_19;
  }
  return v16;
}
