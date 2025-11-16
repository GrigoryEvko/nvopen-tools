// Function: sub_396D290
// Address: 0x396d290
//
void __fastcall sub_396D290(__int64 a1, __int64 a2, unsigned int a3)
{
  _QWORD *v5; // rax
  int v6; // esi
  unsigned int v7; // esi
  __int64 v8; // rax
  void *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  _BYTE *v19; // rax

  if ( a2 )
  {
    sub_396D290(a1, *(_QWORD *)a2);
    v5 = *(_QWORD **)a2;
    if ( *(_QWORD *)a2 )
    {
      v6 = 1;
      do
      {
        v5 = (_QWORD *)*v5;
        ++v6;
      }
      while ( v5 );
      v7 = 2 * v6;
    }
    else
    {
      v7 = 2;
    }
    v8 = sub_16E8750(a1, v7);
    v9 = *(void **)(v8 + 24);
    v10 = v8;
    if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 0xDu )
    {
      v10 = sub_16E7EE0(v8, "Parent Loop BB", 0xEu);
    }
    else
    {
      qmemcpy(v9, "Parent Loop BB", 14);
      *(_QWORD *)(v8 + 24) += 14LL;
    }
    v11 = sub_16E7A90(v10, a3);
    v12 = *(_BYTE **)(v11 + 24);
    if ( *(_BYTE **)(v11 + 16) == v12 )
    {
      v11 = sub_16E7EE0(v11, "_", 1u);
    }
    else
    {
      *v12 = 95;
      ++*(_QWORD *)(v11 + 24);
    }
    v13 = sub_16E7AB0(v11, *(int *)(**(_QWORD **)(a2 + 32) + 48LL));
    v14 = *(_QWORD *)(v13 + 24);
    v15 = v13;
    if ( (unsigned __int64)(*(_QWORD *)(v13 + 16) - v14) <= 6 )
    {
      v15 = sub_16E7EE0(v13, " Depth=", 7u);
    }
    else
    {
      *(_DWORD *)v14 = 1885684768;
      *(_WORD *)(v14 + 4) = 26740;
      *(_BYTE *)(v14 + 6) = 61;
      *(_QWORD *)(v13 + 24) += 7LL;
    }
    v16 = *(_QWORD **)a2;
    if ( *(_QWORD *)a2 )
    {
      LODWORD(v17) = 1;
      do
      {
        v16 = (_QWORD *)*v16;
        v17 = (unsigned int)(v17 + 1);
      }
      while ( v16 );
    }
    else
    {
      v17 = 1;
    }
    v18 = sub_16E7A90(v15, v17);
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
}
