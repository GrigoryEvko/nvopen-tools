// Function: sub_39EA4F0
// Address: 0x39ea4f0
//
_BYTE *__fastcall sub_39EA4F0(__int64 a1, unsigned int a2, unsigned int a3, unsigned __int64 a4)
{
  __int64 v6; // rdi
  void *v7; // rdx
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // rdi
  unsigned __int64 v11; // r13
  _BYTE *result; // rax
  __int64 v13; // rdi
  __int64 v14; // r14
  char *v15; // rsi
  size_t v16; // rdx
  void *v17; // rdi

  sub_38E0CB0((_QWORD *)a1, a2, a3, a4);
  v6 = *(_QWORD *)(a1 + 272);
  v7 = *(void **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0xDu )
  {
    v6 = sub_16E7EE0(v6, "\t.seh_savexmm ", 0xEu);
  }
  else
  {
    qmemcpy(v7, "\t.seh_savexmm ", 14);
    *(_QWORD *)(v6 + 24) += 14LL;
  }
  v8 = sub_16E7A90(v6, a2);
  v9 = *(_WORD **)(v8 + 24);
  v10 = v8;
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 1u )
  {
    v10 = sub_16E7EE0(v8, ", ", 2u);
  }
  else
  {
    *v9 = 8236;
    *(_QWORD *)(v8 + 24) += 2LL;
  }
  sub_16E7A90(v10, a3);
  v11 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v14 = *(_QWORD *)(a1 + 272);
    v15 = *(char **)(a1 + 304);
    v16 = *(unsigned int *)(a1 + 312);
    v17 = *(void **)(v14 + 24);
    if ( v11 > *(_QWORD *)(v14 + 16) - (_QWORD)v17 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v15, v16);
    }
    else
    {
      memcpy(v17, v15, v16);
      *(_QWORD *)(v14 + 24) += v11;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v13 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v13 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v13 + 16) )
    return (_BYTE *)sub_16E7DE0(v13, 10);
  *(_QWORD *)(v13 + 24) = result + 1;
  *result = 10;
  return result;
}
