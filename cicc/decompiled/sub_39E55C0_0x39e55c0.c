// Function: sub_39E55C0
// Address: 0x39e55c0
//
__int64 __fastcall sub_39E55C0(__int64 a1, __int64 a2, char *a3, size_t a4, __int64 a5)
{
  __int64 v9; // rdi
  _QWORD *v10; // rdx
  __int64 v11; // r13
  _WORD *v12; // rdx
  void *v13; // rdi
  __int64 v14; // rdi
  _WORD *v15; // rdx
  unsigned __int64 v16; // r13
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // r14
  char *v21; // rsi
  size_t v22; // rdx
  void *v23; // rdi
  __int64 v24; // rax

  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 7u )
  {
    sub_16E7EE0(v9, "\t.reloc ", 8u);
  }
  else
  {
    *v10 = 0x20636F6C65722E09LL;
    *(_QWORD *)(v9 + 24) += 8LL;
  }
  sub_38CDBE0(a2, *(_QWORD *)(a1 + 272), *(_QWORD *)(a1 + 280));
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(_WORD **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 1u )
  {
    v24 = sub_16E7EE0(*(_QWORD *)(a1 + 272), ", ", 2u);
    v13 = *(void **)(v24 + 24);
    v11 = v24;
  }
  else
  {
    *v12 = 8236;
    v13 = (void *)(*(_QWORD *)(v11 + 24) + 2LL);
    *(_QWORD *)(v11 + 24) = v13;
  }
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v13 < a4 )
  {
    sub_16E7EE0(v11, a3, a4);
  }
  else if ( a4 )
  {
    memcpy(v13, a3, a4);
    *(_QWORD *)(v11 + 24) += a4;
  }
  if ( a5 )
  {
    v14 = *(_QWORD *)(a1 + 272);
    v15 = *(_WORD **)(v14 + 24);
    if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
    {
      sub_16E7EE0(v14, ", ", 2u);
    }
    else
    {
      *v15 = 8236;
      *(_QWORD *)(v14 + 24) += 2LL;
    }
    sub_38CDBE0(a5, *(_QWORD *)(a1 + 272), *(_QWORD *)(a1 + 280));
  }
  v16 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v20 = *(_QWORD *)(a1 + 272);
    v21 = *(char **)(a1 + 304);
    v22 = *(unsigned int *)(a1 + 312);
    v23 = *(void **)(v20 + 24);
    if ( v16 > *(_QWORD *)(v20 + 16) - (_QWORD)v23 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v21, v22);
    }
    else
    {
      memcpy(v23, v21, v22);
      *(_QWORD *)(v20 + 24) += v16;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 272);
    v19 = *(_BYTE **)(v18 + 24);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 16) )
    {
      sub_16E7DE0(v18, 10);
    }
    else
    {
      *(_QWORD *)(v18 + 24) = v19 + 1;
      *v19 = 10;
    }
  }
  return 0;
}
