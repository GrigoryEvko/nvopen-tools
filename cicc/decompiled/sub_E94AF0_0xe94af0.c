// Function: sub_E94AF0
// Address: 0xe94af0
//
_BYTE *__fastcall sub_E94AF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v6; // rdx
  __int64 v7; // r13
  _BYTE *v8; // rdi
  size_t v9; // r14
  __int64 v10; // rax
  size_t v11; // rax
  unsigned __int64 v12; // rax
  void *v13; // rdi
  unsigned __int64 v14; // r14
  unsigned __int8 *v15; // rsi
  int v16; // r15d
  _BYTE *result; // rax
  unsigned __int64 v18; // rcx
  size_t v19; // r13
  void *v20; // rdi
  unsigned __int8 *v21; // rsi
  size_t v22; // rdx
  unsigned int v23; // r15d
  char **v24; // r14
  char v25; // si
  unsigned int v26; // eax
  char *v27; // r13
  void *v28; // rdi
  size_t v29; // rdx
  unsigned __int8 *v30; // rsi
  _WORD *v31; // rdx
  __int64 v32; // r13
  char *v33; // rdi
  size_t v34; // rdx
  unsigned __int8 *v35; // rsi
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  char *v41; // [rsp+8h] [rbp-38h]

  v6 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v6 <= 9u )
  {
    v7 = sub_CB6200(a4, "\t.section\t", 0xAu);
  }
  else
  {
    v7 = a4;
    qmemcpy(v6, "\t.section\t", 10);
    *(_QWORD *)(a4 + 32) += 10LL;
  }
  if ( *(_BYTE *)(a1 + 163) )
  {
    v8 = *(_BYTE **)(v7 + 32);
    v9 = 16;
    if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 0xFu )
      goto LABEL_5;
LABEL_34:
    memcpy(v8, (const void *)(a1 + 148), v9);
    v12 = *(_QWORD *)(v7 + 24);
    v8 = (_BYTE *)(v9 + *(_QWORD *)(v7 + 32));
    *(_QWORD *)(v7 + 32) = v8;
LABEL_9:
    if ( v12 <= (unsigned __int64)v8 )
      goto LABEL_6;
    goto LABEL_10;
  }
  v11 = strlen((const char *)(a1 + 148));
  v8 = *(_BYTE **)(v7 + 32);
  v9 = v11;
  v12 = *(_QWORD *)(v7 + 24);
  if ( v12 - (unsigned __int64)v8 >= v9 )
  {
    if ( !v9 )
      goto LABEL_9;
    goto LABEL_34;
  }
LABEL_5:
  v10 = sub_CB6200(v7, (unsigned __int8 *)(a1 + 148), v9);
  v8 = *(_BYTE **)(v10 + 32);
  v7 = v10;
  if ( *(_QWORD *)(v10 + 24) <= (unsigned __int64)v8 )
  {
LABEL_6:
    v7 = sub_CB5D20(v7, 44);
    goto LABEL_11;
  }
LABEL_10:
  *(_QWORD *)(v7 + 32) = v8 + 1;
  *v8 = 44;
LABEL_11:
  v13 = *(void **)(v7 + 32);
  v14 = *(_QWORD *)(a1 + 136);
  v15 = *(unsigned __int8 **)(a1 + 128);
  if ( v14 > *(_QWORD *)(v7 + 24) - (_QWORD)v13 )
  {
    sub_CB6200(v7, v15, *(_QWORD *)(a1 + 136));
  }
  else if ( v14 )
  {
    memcpy(v13, v15, *(_QWORD *)(a1 + 136));
    *(_QWORD *)(v7 + 32) += v14;
  }
  v16 = *(_DWORD *)(a1 + 164);
  result = *(_BYTE **)(a4 + 32);
  v18 = *(_QWORD *)(a4 + 24);
  if ( v16 && (v19 = (size_t)*(&off_497A9A0 + 4 * (unsigned __int8)v16 + 1)) != 0 )
  {
    if ( (unsigned __int64)result >= v18 )
    {
      sub_CB5D20(a4, 44);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = result + 1;
      *result = 44;
    }
    v20 = *(void **)(a4 + 32);
    v21 = (unsigned __int8 *)*(&off_497A9A0 + 4 * (unsigned __int8)v16);
    if ( v19 > *(_QWORD *)(a4 + 24) - (_QWORD)v20 )
    {
      sub_CB6200(a4, v21, v19);
      v22 = *(_QWORD *)(a4 + 32);
    }
    else
    {
      memcpy(v20, v21, v19);
      v22 = v19 + *(_QWORD *)(a4 + 32);
      *(_QWORD *)(a4 + 32) = v22;
    }
    v23 = v16 & 0xFFFFFF00;
    if ( v23 )
    {
      v24 = &off_497A7E8;
      v25 = 44;
      v26 = 0x80000000;
      while ( (v26 & v23) == 0 )
      {
        v24 += 5;
LABEL_30:
        v26 = *((_DWORD *)v24 - 2);
        if ( !v26 )
        {
LABEL_31:
          if ( !*(_DWORD *)(a1 + 168) )
            goto LABEL_32;
          if ( v22 >= *(_QWORD *)(a4 + 24) )
          {
            v39 = sub_CB5D20(a4, 44);
          }
          else
          {
            v39 = a4;
            *(_QWORD *)(a4 + 32) = v22 + 1;
            *(_BYTE *)v22 = 44;
          }
LABEL_60:
          sub_CB59D0(v39, *(unsigned int *)(a1 + 168));
          v22 = *(_QWORD *)(a4 + 32);
          goto LABEL_32;
        }
      }
      v23 &= ~v26;
      if ( *(_QWORD *)(a4 + 24) <= v22 )
      {
        sub_CB5D20(a4, v25);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v22 + 1;
        *(_BYTE *)v22 = v25;
      }
      v27 = v24[1];
      if ( v27 )
      {
        v28 = *(void **)(a4 + 32);
        v29 = (size_t)v24[1];
        v30 = (unsigned __int8 *)*v24;
        if ( (unsigned __int64)v27 > *(_QWORD *)(a4 + 24) - (_QWORD)v28 )
        {
          sub_CB6200(a4, v30, v29);
          v22 = *(_QWORD *)(a4 + 32);
        }
        else
        {
          memcpy(v28, v30, v29);
          v22 = (size_t)&v27[*(_QWORD *)(a4 + 32)];
          *(_QWORD *)(a4 + 32) = v22;
        }
        goto LABEL_28;
      }
      v31 = *(_WORD **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v31 <= 1u )
      {
        v38 = sub_CB6200(a4, (unsigned __int8 *)"<<", 2u);
        v33 = *(char **)(v38 + 32);
        v32 = v38;
      }
      else
      {
        v32 = a4;
        *v31 = 15420;
        v33 = (char *)(*(_QWORD *)(a4 + 32) + 2LL);
        *(_QWORD *)(a4 + 32) = v33;
      }
      v34 = (size_t)v24[3];
      v35 = (unsigned __int8 *)v24[2];
      v36 = *(_QWORD *)(v32 + 24) - (_QWORD)v33;
      if ( v36 < v34 )
      {
        v37 = sub_CB6200(v32, v35, v34);
        v33 = *(char **)(v37 + 32);
        v32 = v37;
        if ( *(_QWORD *)(v37 + 24) - (_QWORD)v33 > 1u )
        {
LABEL_42:
          *(_WORD *)v33 = 15934;
          *(_QWORD *)(v32 + 32) += 2LL;
          v22 = *(_QWORD *)(a4 + 32);
LABEL_28:
          v24 += 5;
          if ( !v23 )
            goto LABEL_31;
          v25 = 43;
          goto LABEL_30;
        }
      }
      else
      {
        if ( v34 )
        {
          v41 = v24[3];
          memcpy(v33, v35, v34);
          v40 = *(_QWORD *)(v32 + 24);
          v33 = &v41[*(_QWORD *)(v32 + 32)];
          *(_QWORD *)(v32 + 32) = v33;
          v36 = v40 - (_QWORD)v33;
        }
        if ( v36 > 1 )
          goto LABEL_42;
      }
      sub_CB6200(v32, (unsigned __int8 *)">>", 2u);
      v22 = *(_QWORD *)(a4 + 32);
      goto LABEL_28;
    }
    if ( *(_DWORD *)(a1 + 168) )
    {
      if ( *(_QWORD *)(a4 + 24) - v22 <= 5 )
      {
        v39 = sub_CB6200(a4, ",none,", 6u);
      }
      else
      {
        *(_DWORD *)v22 = 1852796460;
        v39 = a4;
        *(_WORD *)(v22 + 4) = 11365;
        *(_QWORD *)(a4 + 32) += 6LL;
      }
      goto LABEL_60;
    }
LABEL_32:
    if ( v22 < *(_QWORD *)(a4 + 24) )
    {
      *(_QWORD *)(a4 + 32) = v22 + 1;
      *(_BYTE *)v22 = 10;
      return (_BYTE *)(v22 + 1);
    }
  }
  else if ( (unsigned __int64)result < v18 )
  {
    *(_QWORD *)(a4 + 32) = result + 1;
    *result = 10;
    return result;
  }
  return (_BYTE *)sub_CB5D20(a4, 10);
}
