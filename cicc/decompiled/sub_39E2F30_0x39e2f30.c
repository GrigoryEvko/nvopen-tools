// Function: sub_39E2F30
// Address: 0x39e2f30
//
_BYTE *__fastcall sub_39E2F30(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  char *v7; // rsi
  size_t v8; // r15
  void *v9; // rdi
  unsigned __int64 v10; // r13
  _BYTE *result; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  char *v14; // rsi
  size_t v15; // rdx
  void *v16; // rdi
  __int64 v17; // rdi
  _BYTE *v18; // rax

  if ( a3 )
  {
    v17 = *(_QWORD *)(a1 + 272);
    v18 = *(_BYTE **)(v17 + 24);
    if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 16) )
    {
      sub_16E7DE0(v17, 9);
    }
    else
    {
      *(_QWORD *)(v17 + 24) = v18 + 1;
      *v18 = 9;
    }
  }
  v5 = *(_QWORD *)(a1 + 280);
  v6 = *(_QWORD *)(a1 + 272);
  v7 = *(char **)(v5 + 48);
  v8 = *(_QWORD *)(v5 + 56);
  v9 = *(void **)(v6 + 24);
  if ( v8 > *(_QWORD *)(v6 + 16) - (_QWORD)v9 )
  {
    v6 = sub_16E7EE0(*(_QWORD *)(a1 + 272), v7, v8);
  }
  else if ( v8 )
  {
    memcpy(v9, v7, v8);
    *(_QWORD *)(v6 + 24) += v8;
  }
  sub_16E2CE0(a2, v6);
  v10 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v13 = *(_QWORD *)(a1 + 272);
    v14 = *(char **)(a1 + 304);
    v15 = *(unsigned int *)(a1 + 312);
    v16 = *(void **)(v13 + 24);
    if ( v10 > *(_QWORD *)(v13 + 16) - (_QWORD)v16 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v14, v15);
    }
    else
    {
      memcpy(v16, v14, v15);
      *(_QWORD *)(v13 + 24) += v10;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v12 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v12 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v12 + 16) )
    return (_BYTE *)sub_16E7DE0(v12, 10);
  *(_QWORD *)(v12 + 24) = result + 1;
  *result = 10;
  return result;
}
