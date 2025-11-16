// Function: sub_39EB900
// Address: 0x39eb900
//
_BYTE *__fastcall sub_39EB900(__int64 a1, void *a2, signed __int64 a3)
{
  __int64 v5; // rcx
  int v6; // r8d
  int v7; // r9d
  unsigned __int64 v8; // r13
  _BYTE *result; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  char *v12; // rsi
  size_t v13; // rdx
  void *v14; // rdi

  sub_38DE060(a1, a2, a3);
  sub_39DFF00(*(_QWORD *)(a1 + 272), (char *)a2, a3, v5, v6, v7);
  v8 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v11 = *(_QWORD *)(a1 + 272);
    v12 = *(char **)(a1 + 304);
    v13 = *(unsigned int *)(a1 + 312);
    v14 = *(void **)(v11 + 24);
    if ( v8 > *(_QWORD *)(v11 + 16) - (_QWORD)v14 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v12, v13);
    }
    else
    {
      memcpy(v14, v12, v13);
      *(_QWORD *)(v11 + 24) += v8;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v10 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 16) )
    return (_BYTE *)sub_16E7DE0(v10, 10);
  *(_QWORD *)(v10 + 24) = result + 1;
  *result = 10;
  return result;
}
