// Function: sub_E51D00
// Address: 0xe51d00
//
__int64 __fastcall sub_E51D00(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, size_t a5, __int64 a6)
{
  __int64 v11; // rdi
  _QWORD *v12; // rdx
  __int64 v13; // r8
  _WORD *v14; // rdx
  void *v15; // rdi
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 304);
  v12 = *(_QWORD **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 7u )
  {
    v19 = a3;
    sub_CB6200(v11, "\t.reloc ", 8u);
    a3 = v19;
  }
  else
  {
    *v12 = 0x20636F6C65722E09LL;
    *(_QWORD *)(v11 + 32) += 8LL;
  }
  sub_E7FAD0(a3, *(_QWORD *)(a2 + 304), *(_QWORD *)(a2 + 312), 0);
  v13 = *(_QWORD *)(a2 + 304);
  v14 = *(_WORD **)(v13 + 32);
  if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 1u )
  {
    v17 = sub_CB6200(*(_QWORD *)(a2 + 304), (unsigned __int8 *)", ", 2u);
    v15 = *(void **)(v17 + 32);
    v13 = v17;
  }
  else
  {
    *v14 = 8236;
    v15 = (void *)(*(_QWORD *)(v13 + 32) + 2LL);
    *(_QWORD *)(v13 + 32) = v15;
  }
  if ( *(_QWORD *)(v13 + 24) - (_QWORD)v15 < a5 )
  {
    sub_CB6200(v13, a4, a5);
  }
  else if ( a5 )
  {
    v18 = v13;
    memcpy(v15, a4, a5);
    *(_QWORD *)(v18 + 32) += a5;
  }
  if ( a6 )
  {
    sub_904010(*(_QWORD *)(a2 + 304), ", ");
    sub_E7FAD0(a6, *(_QWORD *)(a2 + 304), *(_QWORD *)(a2 + 312), 0);
  }
  sub_E4D880(a2);
  *(_BYTE *)(a1 + 40) = 0;
  return a1;
}
