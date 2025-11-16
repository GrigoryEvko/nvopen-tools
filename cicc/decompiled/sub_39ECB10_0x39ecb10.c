// Function: sub_39ECB10
// Address: 0x39ecb10
//
_BYTE *__fastcall sub_39ECB10(__int64 a1, char a2, char a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  unsigned __int64 v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdi
  void *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // r14
  char *v13; // rsi
  size_t v14; // rdx
  void *v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdx

  nullsub_1944();
  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(void **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0xEu )
  {
    sub_16E7EE0(v5, "\t.cfi_sections ", 0xFu);
  }
  else
  {
    qmemcpy(v6, "\t.cfi_sections ", 15);
    *(_QWORD *)(v5 + 24) += 15LL;
  }
  if ( a2 )
  {
    v16 = *(_QWORD *)(a1 + 272);
    v17 = *(_QWORD *)(v16 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v17) <= 8 )
    {
      sub_16E7EE0(v16, ".eh_frame", 9u);
      if ( !a3 )
        goto LABEL_5;
    }
    else
    {
      *(_BYTE *)(v17 + 8) = 101;
      *(_QWORD *)v17 = 0x6D6172665F68652ELL;
      *(_QWORD *)(v16 + 24) += 9LL;
      if ( !a3 )
        goto LABEL_5;
    }
    sub_1263B40(*(_QWORD *)(a1 + 272), ", .debug_frame");
    goto LABEL_5;
  }
  if ( a3 )
  {
    v9 = *(_QWORD *)(a1 + 272);
    v10 = *(void **)(v9 + 24);
    if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0xBu )
    {
      sub_16E7EE0(v9, ".debug_frame", 0xCu);
    }
    else
    {
      qmemcpy(v10, ".debug_frame", 12);
      *(_QWORD *)(v9 + 24) += 12LL;
    }
  }
LABEL_5:
  v7 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v12 = *(_QWORD *)(a1 + 272);
    v13 = *(char **)(a1 + 304);
    v14 = *(unsigned int *)(a1 + 312);
    v15 = *(void **)(v12 + 24);
    if ( v7 > *(_QWORD *)(v12 + 16) - (_QWORD)v15 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v13, v14);
    }
    else
    {
      memcpy(v15, v13, v14);
      *(_QWORD *)(v12 + 24) += v7;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v11 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v11 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v11 + 16) )
    return (_BYTE *)sub_16E7DE0(v11, 10);
  *(_QWORD *)(v11 + 24) = result + 1;
  *result = 10;
  return result;
}
