// Function: sub_39E4C60
// Address: 0x39e4c60
//
_BYTE *__fastcall sub_39E4C60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  void *v8; // rdx
  __int64 v9; // rdi
  _WORD *v10; // rdx
  __int64 v11; // rdi
  _WORD *v12; // rdx
  unsigned __int64 v13; // r13
  _BYTE *result; // rax
  __int64 v15; // rdi
  __int64 v16; // r14
  char *v17; // rsi
  size_t v18; // rdx
  void *v19; // rdi

  v6 = *(_QWORD *)(a1 + 272);
  v8 = *(void **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v8 <= 0xCu )
  {
    sub_16E7EE0(v6, "\t.cg_profile ", 0xDu);
  }
  else
  {
    qmemcpy(v8, "\t.cg_profile ", 13);
    *(_QWORD *)(v6 + 24) += 13LL;
  }
  sub_38E2490(*(_BYTE **)(a2 + 24), *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_WORD **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
  {
    sub_16E7EE0(v9, ", ", 2u);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  sub_38E2490(*(_BYTE **)(a3 + 24), *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(_WORD **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 1u )
  {
    v11 = sub_16E7EE0(v11, ", ", 2u);
  }
  else
  {
    *v12 = 8236;
    *(_QWORD *)(v11 + 24) += 2LL;
  }
  sub_16E7A90(v11, a4);
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
