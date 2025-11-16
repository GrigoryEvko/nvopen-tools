// Function: sub_39E4320
// Address: 0x39e4320
//
_BYTE *__fastcall sub_39E4320(__int64 a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rax
  unsigned __int64 v13; // rdx
  size_t v14; // r13
  _BYTE *result; // rax
  __int64 v16; // r14
  char *v17; // rsi
  void *v18; // rdi
  __int64 v19; // rdi

  v6 = *(_QWORD *)(a1 + 272);
  v8 = *(_QWORD *)(v6 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v8) <= 6 )
  {
    sub_16E7EE0(v6, "\t.comm\t", 7u);
  }
  else
  {
    *(_DWORD *)v8 = 1868770825;
    *(_WORD *)(v8 + 4) = 28013;
    *(_BYTE *)(v8 + 6) = 9;
    *(_QWORD *)(v6 + 24) += 7LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_BYTE **)(v9 + 24);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
  {
    v9 = sub_16E7DE0(v9, 44);
  }
  else
  {
    *(_QWORD *)(v9 + 24) = v10 + 1;
    *v10 = 44;
  }
  sub_16E7A90(v9, a3);
  if ( !a4 )
    goto LABEL_10;
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(_BYTE **)(v11 + 24);
  v13 = *(_QWORD *)(v11 + 16);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 280) + 298LL) )
  {
    if ( v13 <= (unsigned __int64)v12 )
    {
      v11 = sub_16E7DE0(v11, 44);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 44;
    }
    sub_16E7A90(v11, a4);
LABEL_10:
    v14 = *(unsigned int *)(a1 + 312);
    if ( !*(_DWORD *)(a1 + 312) )
      goto LABEL_11;
LABEL_16:
    v16 = *(_QWORD *)(a1 + 272);
    v17 = *(char **)(a1 + 304);
    v18 = *(void **)(v16 + 24);
    if ( v14 > *(_QWORD *)(v16 + 16) - (_QWORD)v18 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v17, v14);
    }
    else
    {
      memcpy(v18, v17, v14);
      *(_QWORD *)(v16 + 24) += v14;
    }
    goto LABEL_11;
  }
  if ( v13 <= (unsigned __int64)v12 )
  {
    v11 = sub_16E7DE0(v11, 44);
  }
  else
  {
    *(_QWORD *)(v11 + 24) = v12 + 1;
    *v12 = 44;
  }
  _BitScanReverse(&a4, a4);
  sub_16E7A90(v11, 31 - (a4 ^ 0x1F));
  v14 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
    goto LABEL_16;
LABEL_11:
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v19 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v19 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v19 + 16) )
    return (_BYTE *)sub_16E7DE0(v19, 10);
  *(_QWORD *)(v19 + 24) = result + 1;
  *result = 10;
  return result;
}
