// Function: sub_39ECD60
// Address: 0x39ecd60
//
void __fastcall sub_39ECD60(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, int a5)
{
  __int64 v6; // rdi
  void *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  _BYTE *v13; // r14
  _BYTE *v14; // r12
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _WORD *v17; // rdx
  unsigned __int64 v18; // r12
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // r13
  char *v22; // rsi
  size_t v23; // rdx
  void *v24; // rdi

  v6 = *(_QWORD *)(a1 + 272);
  v7 = *(void **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0xEu )
  {
    sub_16E7EE0(v6, "\t.cv_def_range\t", 0xFu);
  }
  else
  {
    qmemcpy(v7, "\t.cv_def_range\t", 15);
    *(_QWORD *)(v6 + 24) += 15LL;
  }
  v8 = a2 + 16 * a3;
  if ( a2 != v8 )
  {
    v9 = a2;
    do
    {
      v12 = *(_QWORD *)(a1 + 272);
      v13 = *(_BYTE **)v9;
      v14 = *(_BYTE **)(v9 + 8);
      v15 = *(_BYTE **)(v12 + 24);
      if ( (unsigned __int64)v15 < *(_QWORD *)(v12 + 16) )
      {
        *(_QWORD *)(v12 + 24) = v15 + 1;
        *v15 = 32;
      }
      else
      {
        sub_16E7DE0(v12, 32);
      }
      sub_38E2490(v13, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
      v10 = *(_QWORD *)(a1 + 272);
      v11 = *(_BYTE **)(v10 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
      {
        sub_16E7DE0(v10, 32);
      }
      else
      {
        *(_QWORD *)(v10 + 24) = v11 + 1;
        *v11 = 32;
      }
      v9 += 16;
      sub_38E2490(v14, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
    }
    while ( v8 != v9 );
  }
  v16 = *(_QWORD *)(a1 + 272);
  v17 = *(_WORD **)(v16 + 24);
  if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 1u )
  {
    sub_16E7EE0(v16, ", ", 2u);
  }
  else
  {
    *v17 = 8236;
    *(_QWORD *)(v16 + 24) += 2LL;
  }
  sub_39E0070(a4, a5, *(_QWORD *)(a1 + 272));
  v18 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v21 = *(_QWORD *)(a1 + 272);
    v22 = *(char **)(a1 + 304);
    v23 = *(unsigned int *)(a1 + 312);
    v24 = *(void **)(v21 + 24);
    if ( v18 > *(_QWORD *)(v21 + 16) - (_QWORD)v24 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v22, v23);
    }
    else
    {
      memcpy(v24, v22, v23);
      *(_QWORD *)(v21 + 24) += v18;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 272);
    v20 = *(_BYTE **)(v19 + 24);
    if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 16) )
    {
      sub_16E7DE0(v19, 10);
    }
    else
    {
      *(_QWORD *)(v19 + 24) = v20 + 1;
      *v20 = 10;
    }
  }
  nullsub_1943();
}
