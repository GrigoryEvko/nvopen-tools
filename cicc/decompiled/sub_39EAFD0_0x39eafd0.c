// Function: sub_39EAFD0
// Address: 0x39eafd0
//
_BYTE *__fastcall sub_39EAFD0(__int64 a1, _BYTE *a2, unsigned __int64 a3)
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

  sub_38E0570(a1, (__int64)a2, a3);
  v4 = *(_QWORD *)(a1 + 272);
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 9u )
  {
    sub_16E7EE0(v4, ".seh_proc ", 0xAu);
  }
  else
  {
    qmemcpy(v5, ".seh_proc ", 10);
    *(_QWORD *)(v4 + 24) += 10LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
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
