// Function: sub_23B1FA0
// Address: 0x23b1fa0
//
__int64 __fastcall sub_23B1FA0(__int64 a1, void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  void *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax

  if ( (_BYTE)qword_4FDF208 )
  {
    v21 = sub_904010(*(_QWORD *)(a1 + 40), "*** IR Dump Before ");
    v22 = sub_A51340(v21, a2, a3);
    v23 = sub_904010(v22, " on ");
    v24 = sub_CB6200(v23, *(unsigned __int8 **)a4, *(_QWORD *)(a4 + 8));
    v25 = sub_904010(v24, " ***\n");
    sub_CB6200(v25, *(unsigned __int8 **)a5, *(_QWORD *)(a5 + 8));
  }
  v9 = *(_QWORD *)(a1 + 40);
  if ( *(_QWORD *)(a6 + 8) )
  {
    v10 = sub_904010(v9, "*** IR Dump After ");
    v11 = *(void **)(v10 + 32);
    v12 = v10;
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 < a3 )
    {
      v12 = sub_CB6200(v10, (unsigned __int8 *)a2, a3);
    }
    else if ( a3 )
    {
      memcpy(v11, a2, a3);
      *(_QWORD *)(v12 + 32) += a3;
    }
    v13 = sub_904010(v12, " on ");
    v14 = sub_CB6200(v13, *(unsigned __int8 **)a4, *(_QWORD *)(a4 + 8));
    v15 = sub_904010(v14, " ***\n");
    return sub_CB6200(v15, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
  }
  else
  {
    v17 = sub_904010(v9, "*** IR Deleted After ");
    v18 = sub_A51340(v17, a2, a3);
    v19 = sub_904010(v18, " on ");
    v20 = sub_CB6200(v19, *(unsigned __int8 **)a4, *(_QWORD *)(a4 + 8));
    return sub_904010(v20, " ***\n");
  }
}
