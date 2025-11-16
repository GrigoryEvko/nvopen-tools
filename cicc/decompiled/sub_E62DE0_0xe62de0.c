// Function: sub_E62DE0
// Address: 0xe62de0
//
__int64 __fastcall sub_E62DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  void *v6; // rdx
  _WORD *v7; // rdi
  unsigned __int64 v8; // r13
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax

  v4 = a4;
  v6 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v6 <= 0xAu )
  {
    v14 = sub_CB6200(a4, "\t.section\t\"", 0xBu);
    v7 = *(_WORD **)(v14 + 32);
    v4 = v14;
  }
  else
  {
    qmemcpy(v6, "\t.section\t\"", 11);
    v7 = (_WORD *)(*(_QWORD *)(a4 + 32) + 11LL);
    *(_QWORD *)(a4 + 32) = v7;
  }
  v8 = *(_QWORD *)(a1 + 136);
  v9 = *(unsigned __int8 **)(a1 + 128);
  v10 = *(_QWORD *)(v4 + 24) - (_QWORD)v7;
  if ( v8 > v10 )
  {
    v13 = sub_CB6200(v4, v9, *(_QWORD *)(a1 + 136));
    v7 = *(_WORD **)(v13 + 32);
    v4 = v13;
    v10 = *(_QWORD *)(v13 + 24) - (_QWORD)v7;
  }
  else if ( v8 )
  {
    memcpy(v7, v9, *(_QWORD *)(a1 + 136));
    v7 = (_WORD *)(v8 + *(_QWORD *)(v4 + 32));
    v12 = *(_QWORD *)(v4 + 24) - (_QWORD)v7;
    *(_QWORD *)(v4 + 32) = v7;
    if ( v12 > 1 )
      goto LABEL_6;
    return sub_CB6200(v4, (unsigned __int8 *)"\"\n", 2u);
  }
  if ( v10 > 1 )
  {
LABEL_6:
    *v7 = 2594;
    *(_QWORD *)(v4 + 32) += 2LL;
    return 2594;
  }
  return sub_CB6200(v4, (unsigned __int8 *)"\"\n", 2u);
}
