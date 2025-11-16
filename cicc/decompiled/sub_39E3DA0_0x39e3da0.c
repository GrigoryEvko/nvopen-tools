// Function: sub_39E3DA0
// Address: 0x39e3da0
//
_BYTE *__fastcall sub_39E3DA0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rdi
  _WORD *v8; // rdx
  unsigned __int64 v9; // r13
  _BYTE *result; // rax
  __int64 v11; // rdi
  __int64 v12; // r14
  char *v13; // rsi
  size_t v14; // rdx
  void *v15; // rdi

  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(_QWORD *)(v5 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 6 )
  {
    sub_16E7EE0(v5, "\t.size\t", 7u);
  }
  else
  {
    *(_DWORD *)v6 = 1769156105;
    *(_WORD *)(v6 + 4) = 25978;
    *(_BYTE *)(v6 + 6) = 9;
    *(_QWORD *)(v5 + 24) += 7LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v7 = *(_QWORD *)(a1 + 272);
  v8 = *(_WORD **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 1u )
  {
    sub_16E7EE0(v7, ", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 24) += 2LL;
  }
  sub_38CDBE0(a3, *(_QWORD *)(a1 + 272), *(_QWORD *)(a1 + 280));
  v9 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v12 = *(_QWORD *)(a1 + 272);
    v13 = *(char **)(a1 + 304);
    v14 = *(unsigned int *)(a1 + 312);
    v15 = *(void **)(v12 + 24);
    if ( v9 > *(_QWORD *)(v12 + 16) - (_QWORD)v15 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v13, v14);
    }
    else
    {
      memcpy(v15, v13, v14);
      *(_QWORD *)(v12 + 24) += v9;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v11 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v11 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v11 + 16) )
    return (_BYTE *)sub_16E7DE0(v11, 10);
  *(_QWORD *)(v11 + 24) = result + 1;
  *result = 10;
  return result;
}
