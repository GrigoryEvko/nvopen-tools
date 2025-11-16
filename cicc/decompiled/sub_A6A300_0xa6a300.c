// Function: sub_A6A300
// Address: 0xa6a300
//
__int64 __fastcall sub_A6A300(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r14
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // r13
  _WORD *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // r15
  void *v21; // rdx
  int v22; // eax
  __int64 v23; // rax
  void *v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rdi
  _BYTE *v27; // rax

  v3 = *(_QWORD *)(a2 + 56);
  if ( v3 )
  {
    v4 = *(_QWORD **)v3;
    v5 = (__int64)(*(_QWORD *)(v3 + 8) - *(_QWORD *)v3) >> 4;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = sub_904010(*a1, ", varFlags: (readonly: ");
  v7 = sub_CB59D0(v6, *(_BYTE *)(a2 + 64) & 1);
  v8 = sub_904010(v7, ", ");
  v9 = sub_904010(v8, "writeonly: ");
  v10 = sub_CB59D0(v9, (*(_BYTE *)(a2 + 64) & 2) != 0);
  v11 = sub_904010(v10, ", ");
  v12 = sub_904010(v11, "constant: ");
  sub_CB59D0(v12, (*(_BYTE *)(a2 + 64) & 4) != 0);
  if ( v5 )
  {
    v14 = &v4[2 * v5];
    v15 = sub_904010(*a1, ", ");
    v16 = sub_904010(v15, "vcall_visibility: ");
    v17 = 1;
    sub_CB59D0(v16, (*(_BYTE *)(a2 + 64) >> 3) & 3);
    sub_904010(*a1, ")");
    sub_904010(*a1, ", vTableFuncs: (");
    for ( ; v14 != v4; v4 += 2 )
    {
      v19 = *a1;
      v20 = *a1;
      if ( v17 )
      {
        v17 = 0;
      }
      else
      {
        v18 = *(_WORD **)(v19 + 32);
        if ( *(_QWORD *)(v19 + 24) - (_QWORD)v18 <= 1u )
        {
          sub_CB6200(v19, ", ", 2);
        }
        else
        {
          *v18 = 8236;
          *(_QWORD *)(v19 + 32) += 2LL;
        }
        v19 = *a1;
        v20 = *a1;
      }
      v21 = *(void **)(v19 + 32);
      if ( *(_QWORD *)(v19 + 24) - (_QWORD)v21 <= 0xBu )
      {
        v20 = sub_CB6200(v19, "(virtFunc: ^", 12);
      }
      else
      {
        qmemcpy(v21, "(virtFunc: ^", 12);
        *(_QWORD *)(v19 + 32) += 12LL;
      }
      v22 = sub_A6A250(a1[4], *(_QWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL));
      v23 = sub_CB59F0(v20, v22);
      v24 = *(void **)(v23 + 32);
      v25 = v23;
      if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 9u )
      {
        v25 = sub_CB6200(v23, ", offset: ", 10);
      }
      else
      {
        qmemcpy(v24, ", offset: ", 10);
        *(_QWORD *)(v23 + 32) += 10LL;
      }
      sub_CB59D0(v25, v4[1]);
      v26 = *a1;
      v27 = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) == v27 )
      {
        sub_CB6200(v26, ")", 1);
      }
      else
      {
        *v27 = 41;
        ++*(_QWORD *)(v26 + 32);
      }
    }
  }
  return sub_904010(*a1, ")");
}
