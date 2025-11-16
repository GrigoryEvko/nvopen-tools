// Function: sub_38D9A50
// Address: 0x38d9a50
//
_BYTE *__fastcall sub_38D9A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v6; // rdx
  __int64 v7; // r13
  _BYTE *v8; // rdi
  size_t v9; // r14
  __int64 v10; // rax
  char *v11; // r15
  size_t v12; // rax
  void *v13; // rdi
  size_t v14; // r14
  size_t v15; // rax
  unsigned __int64 v16; // rax
  int v17; // r15d
  _BYTE *result; // rax
  unsigned __int64 v19; // rcx
  size_t v20; // r13
  void *v21; // rdi
  char *v22; // rsi
  size_t v23; // rdx
  unsigned int v24; // r15d
  char **v25; // r14
  unsigned int v26; // eax
  char v27; // si
  char *v28; // r13
  void *v29; // rdi
  size_t v30; // rdx
  char *v31; // rsi
  _WORD *v32; // rdx
  __int64 v33; // r13
  char *v34; // rdi
  size_t v35; // rdx
  char *v36; // rsi
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rax
  char *v42; // [rsp+8h] [rbp-38h]

  v6 = *(void **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v6 <= 9u )
  {
    v7 = sub_16E7EE0(a4, "\t.section\t", 0xAu);
  }
  else
  {
    v7 = a4;
    qmemcpy(v6, "\t.section\t", 10);
    *(_QWORD *)(a4 + 24) += 10LL;
  }
  if ( *(_BYTE *)(a1 + 167) )
  {
    v8 = *(_BYTE **)(v7 + 24);
    v9 = 16;
    if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0xFu )
      goto LABEL_5;
LABEL_36:
    memcpy(v8, (const void *)(a1 + 152), v9);
    v16 = *(_QWORD *)(v7 + 16);
    v8 = (_BYTE *)(v9 + *(_QWORD *)(v7 + 24));
    *(_QWORD *)(v7 + 24) = v8;
    goto LABEL_12;
  }
  v15 = strlen((const char *)(a1 + 152));
  v8 = *(_BYTE **)(v7 + 24);
  v9 = v15;
  v16 = *(_QWORD *)(v7 + 16);
  if ( v16 - (unsigned __int64)v8 < v9 )
  {
LABEL_5:
    v10 = sub_16E7EE0(v7, (char *)(a1 + 152), v9);
    v8 = *(_BYTE **)(v10 + 24);
    v7 = v10;
    if ( *(_QWORD *)(v10 + 16) <= (unsigned __int64)v8 )
      goto LABEL_6;
    goto LABEL_13;
  }
  if ( v9 )
    goto LABEL_36;
LABEL_12:
  if ( v16 <= (unsigned __int64)v8 )
  {
LABEL_6:
    v11 = (char *)(a1 + 168);
    v7 = sub_16E7DE0(v7, 44);
    if ( !*(_BYTE *)(a1 + 183) )
      goto LABEL_7;
LABEL_14:
    v13 = *(void **)(v7 + 24);
    v14 = 16;
    if ( *(_QWORD *)(v7 + 16) - (_QWORD)v13 > 0xFu )
      goto LABEL_9;
LABEL_15:
    sub_16E7EE0(v7, v11, v14);
    goto LABEL_16;
  }
LABEL_13:
  v11 = (char *)(a1 + 168);
  *(_QWORD *)(v7 + 24) = v8 + 1;
  *v8 = 44;
  if ( *(_BYTE *)(a1 + 183) )
    goto LABEL_14;
LABEL_7:
  v12 = strlen(v11);
  v13 = *(void **)(v7 + 24);
  v14 = v12;
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v13 < v12 )
    goto LABEL_15;
  if ( v12 )
  {
LABEL_9:
    memcpy(v13, v11, v14);
    *(_QWORD *)(v7 + 24) += v14;
  }
LABEL_16:
  v17 = *(_DWORD *)(a1 + 184);
  result = *(_BYTE **)(a4 + 24);
  v19 = *(_QWORD *)(a4 + 16);
  if ( v17 && (v20 = (size_t)*(&off_49D9040 + 4 * (unsigned __int8)v17 + 1)) != 0 )
  {
    if ( v19 <= (unsigned __int64)result )
    {
      sub_16E7DE0(a4, 44);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = result + 1;
      *result = 44;
    }
    v21 = *(void **)(a4 + 24);
    v22 = (char *)*(&off_49D9040 + 4 * (unsigned __int8)v17);
    if ( v20 > *(_QWORD *)(a4 + 16) - (_QWORD)v21 )
    {
      sub_16E7EE0(a4, v22, v20);
      v23 = *(_QWORD *)(a4 + 24);
    }
    else
    {
      memcpy(v21, v22, v20);
      v23 = v20 + *(_QWORD *)(a4 + 24);
      *(_QWORD *)(a4 + 24) = v23;
    }
    v24 = v17 & 0xFFFFFF00;
    if ( v24 )
    {
      v25 = &off_49D8E88;
      v26 = 0x80000000;
      v27 = 44;
      while ( 1 )
      {
        if ( (v26 & v24) != 0 )
        {
          v24 &= ~v26;
          if ( v23 >= *(_QWORD *)(a4 + 16) )
          {
            sub_16E7DE0(a4, v27);
          }
          else
          {
            *(_QWORD *)(a4 + 24) = v23 + 1;
            *(_BYTE *)v23 = v27;
          }
          v28 = v25[1];
          if ( v28 )
          {
            v29 = *(void **)(a4 + 24);
            v30 = (size_t)v25[1];
            v31 = *v25;
            if ( (unsigned __int64)v28 > *(_QWORD *)(a4 + 16) - (_QWORD)v29 )
            {
              sub_16E7EE0(a4, v31, v30);
              v23 = *(_QWORD *)(a4 + 24);
            }
            else
            {
              memcpy(v29, v31, v30);
              v23 = (size_t)&v28[*(_QWORD *)(a4 + 24)];
              *(_QWORD *)(a4 + 24) = v23;
            }
          }
          else
          {
            v32 = *(_WORD **)(a4 + 24);
            if ( *(_QWORD *)(a4 + 16) - (_QWORD)v32 <= 1u )
            {
              v39 = sub_16E7EE0(a4, "<<", 2u);
              v34 = *(char **)(v39 + 24);
              v33 = v39;
            }
            else
            {
              v33 = a4;
              *v32 = 15420;
              v34 = (char *)(*(_QWORD *)(a4 + 24) + 2LL);
              *(_QWORD *)(a4 + 24) = v34;
            }
            v35 = (size_t)v25[3];
            v36 = v25[2];
            v37 = *(_QWORD *)(v33 + 16) - (_QWORD)v34;
            if ( v37 < v35 )
            {
              v38 = sub_16E7EE0(v33, v36, v35);
              v34 = *(char **)(v38 + 24);
              v33 = v38;
              v37 = *(_QWORD *)(v38 + 16) - (_QWORD)v34;
            }
            else if ( v35 )
            {
              v42 = v25[3];
              memcpy(v34, v36, v35);
              v41 = *(_QWORD *)(v33 + 16);
              v34 = &v42[*(_QWORD *)(v33 + 24)];
              *(_QWORD *)(v33 + 24) = v34;
              v37 = v41 - (_QWORD)v34;
            }
            if ( v37 <= 1 )
            {
              sub_16E7EE0(v33, ">>", 2u);
            }
            else
            {
              *(_WORD *)v34 = 15934;
              *(_QWORD *)(v33 + 24) += 2LL;
            }
            v23 = *(_QWORD *)(a4 + 24);
          }
          v25 += 5;
          if ( !v24 )
          {
LABEL_33:
            if ( !*(_DWORD *)(a1 + 188) )
              goto LABEL_34;
            if ( *(_QWORD *)(a4 + 16) <= v23 )
            {
              v40 = sub_16E7DE0(a4, 44);
            }
            else
            {
              v40 = a4;
              *(_QWORD *)(a4 + 24) = v23 + 1;
              *(_BYTE *)v23 = 44;
            }
LABEL_62:
            sub_16E7A90(v40, *(unsigned int *)(a1 + 188));
            v23 = *(_QWORD *)(a4 + 24);
            goto LABEL_34;
          }
          v27 = 43;
        }
        else
        {
          v25 += 5;
        }
        v26 = *((_DWORD *)v25 - 2);
        if ( !v26 )
          goto LABEL_33;
      }
    }
    if ( *(_DWORD *)(a1 + 188) )
    {
      if ( *(_QWORD *)(a4 + 16) - v23 <= 5 )
      {
        v40 = sub_16E7EE0(a4, ",none,", 6u);
      }
      else
      {
        *(_DWORD *)v23 = 1852796460;
        v40 = a4;
        *(_WORD *)(v23 + 4) = 11365;
        *(_QWORD *)(a4 + 24) += 6LL;
      }
      goto LABEL_62;
    }
LABEL_34:
    if ( *(_QWORD *)(a4 + 16) > v23 )
    {
      *(_QWORD *)(a4 + 24) = v23 + 1;
      *(_BYTE *)v23 = 10;
      return (_BYTE *)(v23 + 1);
    }
  }
  else if ( v19 > (unsigned __int64)result )
  {
    *(_QWORD *)(a4 + 24) = result + 1;
    *result = 10;
    return result;
  }
  return (_BYTE *)sub_16E7DE0(a4, 10);
}
