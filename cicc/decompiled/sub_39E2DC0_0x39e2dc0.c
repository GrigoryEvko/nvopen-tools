// Function: sub_39E2DC0
// Address: 0x39e2dc0
//
_BYTE *__fastcall sub_39E2DC0(__int64 a1, int a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rdi
  _BYTE *v6; // rax
  unsigned __int64 v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  char *v11; // rsi
  size_t v12; // rdx
  void *v13; // rdi

  v3 = *(_QWORD *)(a1 + 272);
  v4 = *(_QWORD *)(v3 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 16) - v4) <= 6 )
  {
    v3 = sub_16E7EE0(v3, "\t.type\t", 7u);
  }
  else
  {
    *(_DWORD *)v4 = 2037657097;
    *(_WORD *)(v4 + 4) = 25968;
    *(_BYTE *)(v4 + 6) = 9;
    *(_QWORD *)(v3 + 24) += 7LL;
  }
  v5 = sub_16E7AB0(v3, a2);
  v6 = *(_BYTE **)(v5 + 24);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 16) )
  {
    sub_16E7DE0(v5, 59);
  }
  else
  {
    *(_QWORD *)(v5 + 24) = v6 + 1;
    *v6 = 59;
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
