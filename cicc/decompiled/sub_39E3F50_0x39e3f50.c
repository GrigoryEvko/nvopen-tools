// Function: sub_39E3F50
// Address: 0x39e3f50
//
_BYTE *__fastcall sub_39E3F50(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  size_t v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // r14
  char *v12; // rsi
  void *v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // rax

  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(_QWORD *)(v5 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 5 )
  {
    sub_16E7EE0(v5, "\t.rva\t", 6u);
  }
  else
  {
    *(_DWORD *)v6 = 1987194377;
    *(_WORD *)(v6 + 4) = 2401;
    *(_QWORD *)(v5 + 24) += 6LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  if ( a3 > 0 )
  {
    v9 = *(_QWORD *)(a1 + 272);
    v10 = *(_BYTE **)(v9 + 24);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
    {
      v9 = sub_16E7DE0(v9, 43);
    }
    else
    {
      *(_QWORD *)(v9 + 24) = v10 + 1;
      *v10 = 43;
    }
    sub_16E7AB0(v9, a3);
    v7 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_6;
LABEL_11:
    v11 = *(_QWORD *)(a1 + 272);
    v12 = *(char **)(a1 + 304);
    v13 = *(void **)(v11 + 24);
    if ( v7 > *(_QWORD *)(v11 + 16) - (_QWORD)v13 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v12, v7);
    }
    else
    {
      memcpy(v13, v12, v7);
      *(_QWORD *)(v11 + 24) += v7;
    }
    goto LABEL_6;
  }
  if ( a3 )
  {
    v15 = *(_QWORD *)(a1 + 272);
    v16 = *(_BYTE **)(v15 + 24);
    if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
    {
      v15 = sub_16E7DE0(v15, 45);
    }
    else
    {
      *(_QWORD *)(v15 + 24) = v16 + 1;
      *v16 = 45;
    }
    sub_16E7AB0(v15, -a3);
  }
  v7 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_11;
LABEL_6:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v14 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v14 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v14 + 16) )
    return (_BYTE *)sub_16E7DE0(v14, 10);
  *(_QWORD *)(v14 + 24) = result + 1;
  *result = 10;
  return result;
}
