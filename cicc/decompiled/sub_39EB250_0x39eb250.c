// Function: sub_39EB250
// Address: 0x39eb250
//
_BYTE *__fastcall sub_39EB250(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi
  unsigned __int64 v10; // r13
  _BYTE *result; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  char *v14; // rsi
  size_t v15; // rdx
  void *v16; // rdi

  sub_38DE270(a1, a2, a3);
  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(void **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0xEu )
  {
    v5 = sub_16E7EE0(v5, "\t.cfi_register ", 0xFu);
  }
  else
  {
    qmemcpy(v6, "\t.cfi_register ", 15);
    *(_QWORD *)(v5 + 24) += 15LL;
  }
  v7 = sub_16E7AB0(v5, a2);
  v8 = *(_WORD **)(v7 + 24);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 1u )
  {
    v9 = sub_16E7EE0(v7, ", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 24) += 2LL;
  }
  sub_16E7AB0(v9, a3);
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
