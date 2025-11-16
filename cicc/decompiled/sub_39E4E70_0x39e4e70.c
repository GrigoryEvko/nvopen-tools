// Function: sub_39E4E70
// Address: 0x39e4e70
//
_BYTE *__fastcall sub_39E4E70(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rdi
  _WORD *v12; // rdx
  size_t v13; // r13
  _BYTE *result; // rax
  __int64 v15; // rdi
  _WORD *v16; // rdx
  __int64 v17; // r14
  char *v18; // rsi
  void *v19; // rdi
  __int64 v20; // rdi

  sub_38E1870(a1, a3, a2 + 48);
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_QWORD *)(v9 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v9 + 16) - v10) <= 5 )
  {
    sub_16E7EE0(v9, ".tbss ", 6u);
  }
  else
  {
    *(_DWORD *)v10 = 1935832110;
    *(_WORD *)(v10 + 4) = 8307;
    *(_QWORD *)(v9 + 24) += 6LL;
  }
  sub_38E2490(a3, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(_WORD **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 1u )
  {
    v11 = sub_16E7EE0(v11, ", ", 2u);
  }
  else
  {
    *v12 = 8236;
    *(_QWORD *)(v11 + 24) += 2LL;
  }
  sub_16E7A90(v11, a4);
  if ( a5 <= 1 )
  {
    v13 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_7;
LABEL_12:
    v17 = *(_QWORD *)(a1 + 272);
    v18 = *(char **)(a1 + 304);
    v19 = *(void **)(v17 + 24);
    if ( v13 > *(_QWORD *)(v17 + 16) - (_QWORD)v19 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v18, v13);
    }
    else
    {
      memcpy(v19, v18, v13);
      *(_QWORD *)(v17 + 24) += v13;
    }
    goto LABEL_7;
  }
  v15 = *(_QWORD *)(a1 + 272);
  v16 = *(_WORD **)(v15 + 24);
  if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 1u )
  {
    v15 = sub_16E7EE0(v15, ", ", 2u);
  }
  else
  {
    *v16 = 8236;
    *(_QWORD *)(v15 + 24) += 2LL;
  }
  _BitScanReverse(&a5, a5);
  sub_16E7A90(v15, 31 - (a5 ^ 0x1F));
  v13 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_12;
LABEL_7:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v20 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v20 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v20 + 16) )
    return (_BYTE *)sub_16E7DE0(v20, 10);
  *(_QWORD *)(v20 + 24) = result + 1;
  *result = 10;
  return result;
}
