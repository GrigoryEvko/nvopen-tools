// Function: sub_39E4A70
// Address: 0x39e4a70
//
_BYTE *__fastcall sub_39E4A70(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _WORD *v10; // rdx
  __int64 v11; // rax
  _DWORD *v12; // rdx
  unsigned __int64 v13; // r13
  _BYTE *result; // rax
  __int64 v15; // rdi
  __int64 v16; // r14
  char *v17; // rsi
  size_t v18; // rdx
  void *v19; // rdi

  v6 = *(_QWORD *)(a1 + 272);
  v8 = *(_QWORD *)(v6 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v8) <= 6 )
  {
    sub_16E7EE0(v6, "\t.fill\t", 7u);
  }
  else
  {
    *(_DWORD *)v8 = 1768304137;
    *(_WORD *)(v8 + 4) = 27756;
    *(_BYTE *)(v8 + 6) = 9;
    *(_QWORD *)(v6 + 24) += 7LL;
  }
  sub_38CDBE0(a2, *(_QWORD *)(a1 + 272), *(_QWORD *)(a1 + 280));
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_WORD **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
  {
    v9 = sub_16E7EE0(v9, ", ", 2u);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  v11 = sub_16E7AB0(v9, a3);
  v12 = *(_DWORD **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 3u )
  {
    sub_16E7EE0(v11, ", 0x", 4u);
  }
  else
  {
    *v12 = 2016419884;
    *(_QWORD *)(v11 + 24) += 4LL;
  }
  sub_16E7B10(*(_QWORD *)(a1 + 272), a4);
  v13 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v16 = *(_QWORD *)(a1 + 272);
    v17 = *(char **)(a1 + 304);
    v18 = *(unsigned int *)(a1 + 312);
    v19 = *(void **)(v16 + 24);
    if ( v13 > *(_QWORD *)(v16 + 16) - (_QWORD)v19 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v17, v18);
    }
    else
    {
      memcpy(v19, v17, v18);
      *(_QWORD *)(v16 + 24) += v13;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v15 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v15 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v15 + 16) )
    return (_BYTE *)sub_16E7DE0(v15, 10);
  *(_QWORD *)(v15 + 24) = result + 1;
  *result = 10;
  return result;
}
