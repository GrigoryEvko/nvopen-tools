// Function: sub_39E2260
// Address: 0x39e2260
//
_BYTE *__fastcall sub_39E2260(__int64 a1, char *a2, size_t a3)
{
  __int64 v4; // r14
  size_t v5; // r13
  size_t v6; // rdx
  void *v7; // rdi
  size_t v8; // rax
  size_t v9; // r13
  _BYTE *result; // rax
  __int64 v11; // r14
  char *v12; // rsi
  void *v13; // rdi
  __int64 v14; // rdi

  if ( !a3 )
    goto LABEL_5;
  v4 = *(_QWORD *)(a1 + 272);
  v5 = a3;
  v6 = a3 - 1;
  v7 = *(void **)(v4 + 24);
  v8 = *(_QWORD *)(v4 + 16) - (_QWORD)v7;
  if ( a2[v5 - 1] == 10 )
  {
    if ( v8 >= v6 )
    {
      if ( v6 )
      {
        v5 = v6;
        goto LABEL_12;
      }
    }
    else
    {
      sub_16E7EE0(v4, a2, v6);
    }
LABEL_5:
    v9 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_6;
LABEL_13:
    v11 = *(_QWORD *)(a1 + 272);
    v12 = *(char **)(a1 + 304);
    v13 = *(void **)(v11 + 24);
    if ( v9 > *(_QWORD *)(v11 + 16) - (_QWORD)v13 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v12, v9);
    }
    else
    {
      memcpy(v13, v12, v9);
      *(_QWORD *)(v11 + 24) += v9;
    }
    goto LABEL_6;
  }
  if ( v8 < v5 )
  {
    sub_16E7EE0(v4, a2, v5);
    goto LABEL_5;
  }
LABEL_12:
  memcpy(v7, a2, v5);
  *(_QWORD *)(v4 + 24) += v5;
  v9 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_13;
LABEL_6:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v14 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v14 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v14 + 16) )
    return (_BYTE *)sub_16E7DE0(v14, 10);
  *(_QWORD *)(v14 + 24) = result + 1;
  *result = 10;
  return result;
}
