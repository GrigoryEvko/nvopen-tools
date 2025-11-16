// Function: sub_38D85A0
// Address: 0x38d85a0
//
_WORD *__fastcall sub_38D85A0(__int64 a1, void *a2, size_t a3)
{
  _BYTE *v4; // rax
  char *v5; // rbx
  _WORD *result; // rax
  char *v7; // r13
  char v8; // si
  unsigned __int64 v9; // rdx
  __int64 v10; // rdi
  char v11; // si
  char *v12; // rax
  void *v13; // rdi
  size_t v14; // r13
  void *src; // [rsp+0h] [rbp-30h] BYREF
  size_t n; // [rsp+8h] [rbp-28h]

  src = a2;
  n = a3;
  if ( sub_16D24E0(&src, "0123456789_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 64, 0) == -1 )
  {
    v13 = *(void **)(a1 + 24);
    v14 = n;
    result = (_WORD *)(*(_QWORD *)(a1 + 16) - (_QWORD)v13);
    if ( n > (unsigned __int64)result )
      return (_WORD *)sub_16E7EE0(a1, (char *)src, n);
    if ( n )
    {
      result = memcpy(v13, src, n);
      *(_QWORD *)(a1 + 24) += v14;
    }
    return result;
  }
  v4 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(a1 + 16) )
  {
    sub_16E7DE0(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = v4 + 1;
    *v4 = 34;
  }
  v5 = (char *)src;
  result = *(_WORD **)(a1 + 24);
  v7 = (char *)src + n;
  if ( (char *)src + n <= src )
  {
LABEL_22:
    if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 16) )
      return (_WORD *)sub_16E7DE0(a1, 34);
    goto LABEL_23;
  }
  while ( 1 )
  {
    v8 = *v5;
    v9 = *(_QWORD *)(a1 + 16);
    if ( *v5 != 34 )
      break;
    if ( v9 - (unsigned __int64)result <= 1 )
    {
      sub_16E7EE0(a1, "\\\"", 2u);
      result = *(_WORD **)(a1 + 24);
    }
    else
    {
      *result = 8796;
      result = (_WORD *)(*(_QWORD *)(a1 + 24) + 2LL);
      *(_QWORD *)(a1 + 24) = result;
    }
LABEL_9:
    if ( ++v5 >= v7 )
      goto LABEL_22;
  }
  if ( v8 != 92 )
  {
    if ( v9 <= (unsigned __int64)result )
    {
      sub_16E7DE0(a1, v8);
    }
    else
    {
      *(_QWORD *)(a1 + 24) = (char *)result + 1;
      *(_BYTE *)result = v8;
    }
    result = *(_WORD **)(a1 + 24);
    goto LABEL_9;
  }
  if ( v7 != v5 + 1 )
  {
    if ( v9 <= (unsigned __int64)result )
    {
      v10 = sub_16E7DE0(a1, 92);
    }
    else
    {
      v10 = a1;
      *(_QWORD *)(a1 + 24) = (char *)result + 1;
      *(_BYTE *)result = 92;
    }
    v11 = v5[1];
    v12 = *(char **)(v10 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v10 + 16) )
    {
      sub_16E7DE0(v10, v11);
      ++v5;
    }
    else
    {
      ++v5;
      *(_QWORD *)(v10 + 24) = v12 + 1;
      *v12 = v11;
    }
    result = *(_WORD **)(a1 + 24);
    goto LABEL_9;
  }
  if ( v9 - (unsigned __int64)result <= 1 )
  {
    sub_16E7EE0(a1, "\\\\", 2u);
    result = *(_WORD **)(a1 + 24);
    goto LABEL_22;
  }
  *result = 23644;
  result = (_WORD *)(*(_QWORD *)(a1 + 24) + 2LL);
  *(_QWORD *)(a1 + 24) = result;
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 16) )
    return (_WORD *)sub_16E7DE0(a1, 34);
LABEL_23:
  *(_QWORD *)(a1 + 24) = (char *)result + 1;
  *(_BYTE *)result = 34;
  return result;
}
