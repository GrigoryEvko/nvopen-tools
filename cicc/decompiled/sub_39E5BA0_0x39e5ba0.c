// Function: sub_39E5BA0
// Address: 0x39e5ba0
//
_BYTE *__fastcall sub_39E5BA0(__int64 a1, _BYTE *a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  __int64 v9; // rdi
  void *v10; // rdx
  __int64 v11; // r13
  void *v12; // rsi
  char *v13; // rdi
  size_t v14; // rdx
  char *v15; // rsi
  size_t v16; // rax
  void *v17; // rdi
  size_t v18; // rdx
  size_t v19; // rax
  char *v20; // rax
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // rdi
  _BYTE *v24; // rax
  unsigned __int64 v25; // r13
  _BYTE *result; // rax
  char *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // r14
  char *v30; // rsi
  size_t v31; // rdx
  void *v32; // rdi
  __int64 v33; // rdi
  _BYTE *v34; // rax
  unsigned int v35; // ebx
  size_t src; // [rsp+0h] [rbp-40h]
  char *srca; // [rsp+0h] [rbp-40h]

  if ( a3 )
    sub_38E1870(a1, a3, (__int64)(a2 + 48));
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(void **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 9u )
  {
    sub_16E7EE0(v9, ".zerofill ", 0xAu);
  }
  else
  {
    qmemcpy(v10, ".zerofill ", 10);
    *(_QWORD *)(v9 + 24) += 10LL;
  }
  v11 = *(_QWORD *)(a1 + 272);
  v12 = a2 + 152;
  if ( a2[167] )
  {
    v13 = *(char **)(v11 + 24);
    v14 = 16;
    if ( *(_QWORD *)(v11 + 16) - (_QWORD)v13 <= 0xFu )
      goto LABEL_7;
LABEL_27:
    srca = (char *)v14;
    memcpy(v13, v12, v14);
    v27 = &srca[*(_QWORD *)(v11 + 24)];
    *(_QWORD *)(v11 + 24) = v27;
    v20 = *(char **)(v11 + 16);
    v13 = v27;
    goto LABEL_14;
  }
  v19 = strlen(a2 + 152);
  v13 = *(char **)(v11 + 24);
  v12 = a2 + 152;
  v14 = v19;
  v20 = *(char **)(v11 + 16);
  if ( v20 - v13 < v14 )
  {
LABEL_7:
    v11 = sub_16E7EE0(v11, (char *)v12, v14);
    v13 = *(char **)(v11 + 24);
    if ( v13 == *(char **)(v11 + 16) )
      goto LABEL_8;
    goto LABEL_15;
  }
  if ( v14 )
    goto LABEL_27;
LABEL_14:
  if ( v13 == v20 )
  {
LABEL_8:
    v15 = a2 + 168;
    v11 = sub_16E7EE0(v11, ",", 1u);
    if ( !a2[183] )
      goto LABEL_9;
LABEL_16:
    v17 = *(void **)(v11 + 24);
    v18 = 16;
    if ( *(_QWORD *)(v11 + 16) - (_QWORD)v17 > 0xFu )
      goto LABEL_11;
LABEL_17:
    sub_16E7EE0(v11, v15, v18);
    goto LABEL_18;
  }
LABEL_15:
  *v13 = 44;
  v15 = a2 + 168;
  ++*(_QWORD *)(v11 + 24);
  if ( a2[183] )
    goto LABEL_16;
LABEL_9:
  v16 = strlen(v15);
  v17 = *(void **)(v11 + 24);
  v18 = v16;
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v17 < v16 )
    goto LABEL_17;
  if ( v16 )
  {
LABEL_11:
    src = v18;
    memcpy(v17, v15, v18);
    *(_QWORD *)(v11 + 24) += src;
  }
LABEL_18:
  if ( a3 )
  {
    v21 = *(_QWORD *)(a1 + 272);
    v22 = *(_BYTE **)(v21 + 24);
    if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 16) )
    {
      sub_16E7DE0(v21, 44);
    }
    else
    {
      *(_QWORD *)(v21 + 24) = v22 + 1;
      *v22 = 44;
    }
    sub_38E2490(a3, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
    v23 = *(_QWORD *)(a1 + 272);
    v24 = *(_BYTE **)(v23 + 24);
    if ( (unsigned __int64)v24 >= *(_QWORD *)(v23 + 16) )
    {
      v23 = sub_16E7DE0(v23, 44);
    }
    else
    {
      *(_QWORD *)(v23 + 24) = v24 + 1;
      *v24 = 44;
    }
    sub_16E7A90(v23, a4);
    if ( a5 )
    {
      v33 = *(_QWORD *)(a1 + 272);
      v34 = *(_BYTE **)(v33 + 24);
      if ( (unsigned __int64)v34 >= *(_QWORD *)(v33 + 16) )
      {
        v33 = sub_16E7DE0(v33, 44);
      }
      else
      {
        *(_QWORD *)(v33 + 24) = v34 + 1;
        *v34 = 44;
      }
      _BitScanReverse(&v35, a5);
      sub_16E7A90(v33, 31 - (v35 ^ 0x1F));
    }
  }
  v25 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v29 = *(_QWORD *)(a1 + 272);
    v30 = *(char **)(a1 + 304);
    v31 = *(unsigned int *)(a1 + 312);
    v32 = *(void **)(v29 + 24);
    if ( v25 > *(_QWORD *)(v29 + 16) - (_QWORD)v32 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v30, v31);
    }
    else
    {
      memcpy(v32, v30, v31);
      *(_QWORD *)(v29 + 24) += v25;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v28 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v28 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v28 + 16) )
    return (_BYTE *)sub_16E7DE0(v28, 10);
  *(_QWORD *)(v28 + 24) = result + 1;
  *result = 10;
  return result;
}
