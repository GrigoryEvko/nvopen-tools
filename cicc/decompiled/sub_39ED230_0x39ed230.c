// Function: sub_39ED230
// Address: 0x39ed230
//
void __fastcall sub_39ED230(__int64 a1, unsigned int a2, _BYTE *a3, _BYTE *a4)
{
  __int64 v7; // rdi
  void *v8; // rdx
  __int64 v9; // rax
  _WORD *v10; // rdx
  __int64 v11; // rdi
  _WORD *v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // r8
  char *v17; // rsi
  void *v18; // rdi
  __int64 v19; // [rsp+8h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 272);
  v8 = *(void **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0xEu )
  {
    v7 = sub_16E7EE0(v7, "\t.cv_linetable\t", 0xFu);
  }
  else
  {
    qmemcpy(v8, "\t.cv_linetable\t", 15);
    *(_QWORD *)(v7 + 24) += 15LL;
  }
  v9 = sub_16E7A90(v7, a2);
  v10 = *(_WORD **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
  {
    sub_16E7EE0(v9, ", ", 2u);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  sub_38E2490(a3, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(_WORD **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 1u )
  {
    sub_16E7EE0(v11, ", ", 2u);
  }
  else
  {
    *v12 = 8236;
    *(_QWORD *)(v11 + 24) += 2LL;
  }
  sub_38E2490(a4, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v13 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v16 = *(_QWORD *)(a1 + 272);
    v17 = *(char **)(a1 + 304);
    v18 = *(void **)(v16 + 24);
    if ( v13 > *(_QWORD *)(v16 + 16) - (_QWORD)v18 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v17, *(unsigned int *)(a1 + 312));
    }
    else
    {
      v19 = *(_QWORD *)(a1 + 272);
      memcpy(v18, v17, *(unsigned int *)(a1 + 312));
      *(_QWORD *)(v19 + 24) += v13;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v14 = *(_QWORD *)(a1 + 272);
    v15 = *(_BYTE **)(v14 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
    {
      sub_16E7DE0(v14, 10);
    }
    else
    {
      *(_QWORD *)(v14 + 24) = v15 + 1;
      *v15 = 10;
    }
  }
  nullsub_1941();
}
