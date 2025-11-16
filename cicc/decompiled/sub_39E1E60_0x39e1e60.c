// Function: sub_39E1E60
// Address: 0x39e1e60
//
_BYTE *__fastcall sub_39E1E60(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  unsigned __int64 v5; // r13
  _BYTE *result; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  char *v9; // rsi
  size_t v10; // rdx
  void *v11; // rdi

  v3 = *(_QWORD *)(a1 + 272);
  v4 = *(void **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 9u )
  {
    sub_16E7EE0(v3, "\t.safeseh\t", 0xAu);
  }
  else
  {
    qmemcpy(v4, "\t.safeseh\t", 10);
    *(_QWORD *)(v3 + 24) += 10LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v5 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v8 = *(_QWORD *)(a1 + 272);
    v9 = *(char **)(a1 + 304);
    v10 = *(unsigned int *)(a1 + 312);
    v11 = *(void **)(v8 + 24);
    if ( v5 > *(_QWORD *)(v8 + 16) - (_QWORD)v11 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v9, v10);
    }
    else
    {
      memcpy(v11, v9, v10);
      *(_QWORD *)(v8 + 24) += v5;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v7 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v7 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v7 + 16) )
    return (_BYTE *)sub_16E7DE0(v7, 10);
  *(_QWORD *)(v7 + 24) = result + 1;
  *result = 10;
  return result;
}
