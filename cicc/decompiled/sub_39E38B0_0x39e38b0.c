// Function: sub_39E38B0
// Address: 0x39e38b0
//
_BYTE *__fastcall sub_39E38B0(__int64 a1, _BYTE *a2, unsigned int a3)
{
  __int64 v4; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // rax
  __int64 v8; // rdi
  _BYTE *v9; // rax
  unsigned __int64 v10; // r13
  _BYTE *result; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  char *v14; // rsi
  size_t v15; // rdx
  void *v16; // rdi

  v4 = *(_QWORD *)(a1 + 272);
  v6 = *(_QWORD *)(v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v6) <= 4 )
  {
    v4 = sub_16E7EE0(v4, ".desc", 5u);
    v7 = *(_BYTE **)(v4 + 24);
  }
  else
  {
    *(_DWORD *)v6 = 1936024622;
    *(_BYTE *)(v6 + 4) = 99;
    v7 = (_BYTE *)(*(_QWORD *)(v4 + 24) + 5LL);
    *(_QWORD *)(v4 + 24) = v7;
  }
  if ( *(_QWORD *)(v4 + 16) <= (unsigned __int64)v7 )
  {
    sub_16E7DE0(v4, 32);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v7 + 1;
    *v7 = 32;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v8 = *(_QWORD *)(a1 + 272);
  v9 = *(_BYTE **)(v8 + 24);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
  {
    v8 = sub_16E7DE0(v8, 44);
  }
  else
  {
    *(_QWORD *)(v8 + 24) = v9 + 1;
    *v9 = 44;
  }
  sub_16E7A90(v8, a3);
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
