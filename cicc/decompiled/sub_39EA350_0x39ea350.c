// Function: sub_39EA350
// Address: 0x39ea350
//
_BYTE *__fastcall sub_39EA350(__int64 a1, unsigned __int8 a2, unsigned __int64 a3)
{
  __int64 v4; // rdi
  void *v5; // rdx
  unsigned __int64 v6; // r13
  _BYTE *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  char *v10; // rsi
  size_t v11; // rdx
  void *v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdx

  sub_38E0EF0((_QWORD *)a1, a2, a3);
  v4 = *(_QWORD *)(a1 + 272);
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0xEu )
  {
    sub_16E7EE0(v4, "\t.seh_pushframe", 0xFu);
  }
  else
  {
    qmemcpy(v5, "\t.seh_pushframe", 15);
    *(_QWORD *)(v4 + 24) += 15LL;
  }
  if ( a2 )
  {
    v13 = *(_QWORD *)(a1 + 272);
    v14 = *(_QWORD *)(v13 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v13 + 16) - v14) <= 5 )
    {
      sub_16E7EE0(v13, " @code", 6u);
    }
    else
    {
      *(_DWORD *)v14 = 1868775456;
      *(_WORD *)(v14 + 4) = 25956;
      *(_QWORD *)(v13 + 24) += 6LL;
    }
  }
  v6 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v9 = *(_QWORD *)(a1 + 272);
    v10 = *(char **)(a1 + 304);
    v11 = *(unsigned int *)(a1 + 312);
    v12 = *(void **)(v9 + 24);
    if ( v6 > *(_QWORD *)(v9 + 16) - (_QWORD)v12 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v10, v11);
    }
    else
    {
      memcpy(v12, v10, v11);
      *(_QWORD *)(v9 + 24) += v6;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v8 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v8 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 16) )
    return (_BYTE *)sub_16E7DE0(v8, 10);
  *(_QWORD *)(v8 + 24) = result + 1;
  *result = 10;
  return result;
}
