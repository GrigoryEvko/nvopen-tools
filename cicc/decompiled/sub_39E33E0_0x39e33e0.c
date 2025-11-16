// Function: sub_39E33E0
// Address: 0x39e33e0
//
_BYTE *__fastcall sub_39E33E0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  unsigned __int64 v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  char *v11; // rsi
  size_t v12; // rdx
  void *v13; // rdi
  __int64 v14; // rdi
  _BYTE *v15; // rax

  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(void **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0xAu )
  {
    sub_16E7EE0(v5, "\t.secrel32\t", 0xBu);
  }
  else
  {
    qmemcpy(v6, "\t.secrel32\t", 11);
    *(_QWORD *)(v5 + 24) += 11LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  if ( a3 )
  {
    v14 = *(_QWORD *)(a1 + 272);
    v15 = *(_BYTE **)(v14 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
    {
      v14 = sub_16E7DE0(v14, 43);
    }
    else
    {
      *(_QWORD *)(v14 + 24) = v15 + 1;
      *v15 = 43;
    }
    sub_16E7A90(v14, a3);
  }
  v7 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v10 = *(_QWORD *)(a1 + 272);
    v11 = *(char **)(a1 + 304);
    v12 = *(unsigned int *)(a1 + 312);
    v13 = *(void **)(v10 + 24);
    if ( v7 > *(_QWORD *)(v10 + 16) - (_QWORD)v13 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v11, v12);
    }
    else
    {
      memcpy(v13, v11, v12);
      *(_QWORD *)(v10 + 24) += v7;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v9 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v9 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v9 + 16) )
    return (_BYTE *)sub_16E7DE0(v9, 10);
  *(_QWORD *)(v9 + 24) = result + 1;
  *result = 10;
  return result;
}
