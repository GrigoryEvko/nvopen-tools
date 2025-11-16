// Function: sub_38DB430
// Address: 0x38db430
//
_BYTE *__fastcall sub_38DB430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v8; // rdx
  size_t v9; // rax
  _BYTE *v10; // rax
  char *v11; // rbx
  _WORD *v12; // rax
  char *v13; // r14
  char v14; // si
  unsigned __int64 v15; // rdx
  char *v16; // r8
  __int64 v17; // rdi
  char v18; // si
  char *v19; // rax
  _WORD *v20; // rdi
  size_t v21; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  _BYTE *v24; // rax
  unsigned __int64 v25; // rcx
  _QWORD *v26; // rdx
  __int64 v27; // rdi
  _BYTE *result; // rax
  void *v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // rax
  __int64 v32; // r14
  void *v33; // rdi
  char *v34; // rsi
  size_t v35; // r15
  _BYTE *v36; // rax
  size_t v37; // r14
  __int64 v38; // rax
  char *v39; // [rsp+0h] [rbp-50h]
  void *src; // [rsp+10h] [rbp-40h] BYREF
  size_t n; // [rsp+18h] [rbp-38h]

  if ( (unsigned __int8)sub_38DB420(a1, *(_QWORD *)(a1 + 152), *(_QWORD *)(a1 + 160), a2) )
  {
    v31 = *(_BYTE **)(a4 + 24);
    if ( (unsigned __int64)v31 >= *(_QWORD *)(a4 + 16) )
    {
      v32 = sub_16E7DE0(a4, 9);
    }
    else
    {
      v32 = a4;
      *(_QWORD *)(a4 + 24) = v31 + 1;
      *v31 = 9;
    }
    v33 = *(void **)(v32 + 24);
    v34 = *(char **)(a1 + 152);
    v35 = *(_QWORD *)(a1 + 160);
    if ( v35 > *(_QWORD *)(v32 + 16) - (_QWORD)v33 )
    {
      sub_16E7EE0(v32, v34, v35);
      if ( a5 )
        goto LABEL_60;
    }
    else
    {
      if ( v35 )
      {
        memcpy(v33, v34, v35);
        *(_QWORD *)(v32 + 24) += v35;
      }
      if ( a5 )
      {
LABEL_60:
        v36 = *(_BYTE **)(a4 + 24);
        if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v36 )
        {
          sub_16E7DE0(a4, 9);
        }
        else
        {
          *(_QWORD *)(a4 + 24) = v36 + 1;
          *v36 = 9;
        }
LABEL_42:
        sub_38CDBE0(a5, a4, a2);
      }
    }
    result = *(_BYTE **)(a4 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 16) )
      return (_BYTE *)sub_16E7DE0(a4, 10);
    *(_QWORD *)(a4 + 24) = result + 1;
    *result = 10;
    return result;
  }
  v8 = *(void **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v8 <= 9u )
  {
    sub_16E7EE0(a4, "\t.section\t", 0xAu);
  }
  else
  {
    qmemcpy(v8, "\t.section\t", 10);
    *(_QWORD *)(a4 + 24) += 10LL;
  }
  v9 = *(_QWORD *)(a1 + 160);
  src = *(void **)(a1 + 152);
  n = v9;
  if ( sub_16D24E0(&src, "0123456789_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 64, 0) == -1 )
  {
    v20 = *(_WORD **)(a4 + 24);
    v37 = n;
    v21 = *(_QWORD *)(a4 + 16) - (_QWORD)v20;
    if ( n > v21 )
    {
      sub_16E7EE0(a4, (char *)src, n);
      v20 = *(_WORD **)(a4 + 24);
      v21 = *(_QWORD *)(a4 + 16) - (_QWORD)v20;
    }
    else if ( n )
    {
      memcpy(v20, src, n);
      v38 = *(_QWORD *)(a4 + 16);
      v20 = (_WORD *)(v37 + *(_QWORD *)(a4 + 24));
      *(_QWORD *)(a4 + 24) = v20;
      v21 = v38 - (_QWORD)v20;
    }
LABEL_26:
    if ( v21 <= 1 )
      goto LABEL_49;
LABEL_27:
    *v20 = 8748;
    v22 = (_BYTE *)(*(_QWORD *)(a4 + 24) + 2LL);
    *(_QWORD *)(a4 + 24) = v22;
    if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v22 )
      goto LABEL_50;
LABEL_28:
    *(_QWORD *)(a4 + 24) = v22 + 1;
    *v22 = 34;
    v23 = *(_BYTE **)(a4 + 24);
    if ( (unsigned __int64)v23 >= *(_QWORD *)(a4 + 16) )
      goto LABEL_51;
    goto LABEL_29;
  }
  v10 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(a4 + 16) )
  {
    sub_16E7DE0(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v10 + 1;
    *v10 = 34;
  }
  v11 = (char *)src;
  v12 = *(_WORD **)(a4 + 24);
  v13 = (char *)src + n;
  if ( src >= (char *)src + n )
    goto LABEL_24;
  while ( 1 )
  {
    v14 = *v11;
    v15 = *(_QWORD *)(a4 + 16);
    if ( *v11 == 34 )
    {
      if ( v15 - (unsigned __int64)v12 <= 1 )
      {
        sub_16E7EE0(a4, "\\\"", 2u);
        v12 = *(_WORD **)(a4 + 24);
      }
      else
      {
        *v12 = 8796;
        v12 = (_WORD *)(*(_QWORD *)(a4 + 24) + 2LL);
        *(_QWORD *)(a4 + 24) = v12;
      }
      goto LABEL_13;
    }
    if ( v14 != 92 )
    {
      if ( (unsigned __int64)v12 >= v15 )
      {
        sub_16E7DE0(a4, v14);
      }
      else
      {
        *(_QWORD *)(a4 + 24) = (char *)v12 + 1;
        *(_BYTE *)v12 = v14;
      }
      v12 = *(_WORD **)(a4 + 24);
      goto LABEL_13;
    }
    v16 = v11 + 1;
    if ( v13 == v11 + 1 )
      break;
    if ( (unsigned __int64)v12 >= v15 )
    {
      v30 = sub_16E7DE0(a4, 92);
      v16 = v11 + 1;
      v18 = v11[1];
      v17 = v30;
      v19 = *(char **)(v30 + 24);
      if ( (unsigned __int64)v19 >= *(_QWORD *)(v17 + 16) )
      {
LABEL_53:
        v39 = v16;
        sub_16E7DE0(v17, v18);
        v12 = *(_WORD **)(a4 + 24);
        v11 = v39;
        goto LABEL_13;
      }
    }
    else
    {
      v17 = a4;
      *(_QWORD *)(a4 + 24) = (char *)v12 + 1;
      *(_BYTE *)v12 = 92;
      v18 = v11[1];
      v19 = *(char **)(a4 + 24);
      if ( (unsigned __int64)v19 >= *(_QWORD *)(a4 + 16) )
        goto LABEL_53;
    }
    v11 = v16;
    *(_QWORD *)(v17 + 24) = v19 + 1;
    *v19 = v18;
    v12 = *(_WORD **)(a4 + 24);
LABEL_13:
    if ( v13 <= ++v11 )
      goto LABEL_24;
  }
  if ( v15 - (unsigned __int64)v12 <= 1 )
  {
    sub_16E7EE0(a4, "\\\\", 2u);
    v12 = *(_WORD **)(a4 + 24);
LABEL_24:
    if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v12 )
      goto LABEL_48;
    goto LABEL_25;
  }
  *v12 = 23644;
  v12 = (_WORD *)(*(_QWORD *)(a4 + 24) + 2LL);
  *(_QWORD *)(a4 + 24) = v12;
  if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v12 )
  {
LABEL_25:
    *(_QWORD *)(a4 + 24) = (char *)v12 + 1;
    *(_BYTE *)v12 = 34;
    v20 = *(_WORD **)(a4 + 24);
    v21 = *(_QWORD *)(a4 + 16) - (_QWORD)v20;
    goto LABEL_26;
  }
LABEL_48:
  sub_16E7DE0(a4, 34);
  v20 = *(_WORD **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v20 > 1u )
    goto LABEL_27;
LABEL_49:
  sub_16E7EE0(a4, ",\"", 2u);
  v22 = *(_BYTE **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v22 )
    goto LABEL_28;
LABEL_50:
  sub_16E7DE0(a4, 34);
  v23 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)v23 < *(_QWORD *)(a4 + 16) )
  {
LABEL_29:
    *(_QWORD *)(a4 + 24) = v23 + 1;
    *v23 = 44;
    goto LABEL_30;
  }
LABEL_51:
  sub_16E7DE0(a4, 44);
LABEL_30:
  v24 = *(_BYTE **)(a4 + 24);
  v25 = *(_QWORD *)(a4 + 16);
  if ( **(_BYTE **)(a2 + 48) == 64 )
  {
    if ( (unsigned __int64)v24 >= v25 )
    {
      sub_16E7DE0(a4, 37);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v24 + 1;
      *v24 = 37;
    }
  }
  else if ( (unsigned __int64)v24 >= v25 )
  {
    sub_16E7DE0(a4, 64);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v24 + 1;
    *v24 = 64;
  }
  if ( *(_DWORD *)(a1 + 168) != -1 )
  {
    v26 = *(_QWORD **)(a4 + 24);
    if ( *(_QWORD *)(a4 + 16) - (_QWORD)v26 <= 7u )
    {
      v27 = sub_16E7EE0(a4, ",unique,", 8u);
    }
    else
    {
      v27 = a4;
      *v26 = 0x2C657571696E752CLL;
      *(_QWORD *)(a4 + 24) += 8LL;
    }
    sub_16E7A90(v27, *(unsigned int *)(a1 + 168));
  }
  result = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 16) )
  {
    result = (_BYTE *)sub_16E7DE0(a4, 10);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = result + 1;
    *result = 10;
  }
  if ( a5 )
  {
    v29 = *(void **)(a4 + 24);
    if ( *(_QWORD *)(a4 + 16) - (_QWORD)v29 <= 0xCu )
    {
      sub_16E7EE0(a4, "\t.subsection\t", 0xDu);
    }
    else
    {
      qmemcpy(v29, "\t.subsection\t", 13);
      *(_QWORD *)(a4 + 24) += 13LL;
    }
    goto LABEL_42;
  }
  return result;
}
