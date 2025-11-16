// Function: sub_39EA010
// Address: 0x39ea010
//
_BYTE *__fastcall sub_39EA010(__int64 a1, _BYTE *a2, char a3, char a4, unsigned __int64 a5)
{
  __int64 v8; // rdi
  void *v9; // rdx
  unsigned __int64 v10; // r13
  _BYTE *result; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  char *v14; // rsi
  size_t v15; // rdx
  void *v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rdx

  sub_38DD410(a1, (__int64)a2, a3, a4, a5);
  v8 = *(_QWORD *)(a1 + 272);
  v9 = *(void **)(v8 + 24);
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 0xDu )
  {
    sub_16E7EE0(v8, "\t.seh_handler ", 0xEu);
  }
  else
  {
    qmemcpy(v9, "\t.seh_handler ", 14);
    *(_QWORD *)(v8 + 24) += 14LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  if ( a3 )
  {
    v19 = *(_QWORD *)(a1 + 272);
    v20 = *(_QWORD *)(v19 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v19 + 16) - v20) <= 8 )
    {
      sub_16E7EE0(v19, ", @unwind", 9u);
    }
    else
    {
      *(_BYTE *)(v20 + 8) = 100;
      *(_QWORD *)v20 = 0x6E69776E7540202CLL;
      *(_QWORD *)(v19 + 24) += 9LL;
    }
  }
  if ( a4 )
  {
    v17 = *(_QWORD *)(a1 + 272);
    v18 = *(_QWORD *)(v17 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v18) <= 8 )
    {
      sub_16E7EE0(v17, ", @except", 9u);
    }
    else
    {
      *(_BYTE *)(v18 + 8) = 116;
      *(_QWORD *)v18 = 0x706563786540202CLL;
      *(_QWORD *)(v17 + 24) += 9LL;
    }
  }
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
