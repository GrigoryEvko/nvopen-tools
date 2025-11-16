// Function: sub_39E1A70
// Address: 0x39e1a70
//
_BYTE *__fastcall sub_39E1A70(__int64 a1, unsigned __int8 *a2, int a3)
{
  __int64 v5; // rdi
  _QWORD *v6; // rdx
  unsigned __int64 v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  char *v11; // rsi
  size_t v12; // rdx
  void *v13; // rdi

  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 7u )
  {
    sub_16E7EE0(v5, "\t.ident\t", 8u);
  }
  else
  {
    *v6 = 0x9746E6564692E09LL;
    *(_QWORD *)(v5 + 24) += 8LL;
  }
  sub_39E0070(a2, a3, *(_QWORD *)(a1 + 272));
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
