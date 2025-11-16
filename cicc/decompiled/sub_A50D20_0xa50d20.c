// Function: sub_A50D20
// Address: 0xa50d20
//
__int64 __fastcall sub_A50D20(_QWORD *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  _WORD *v9; // rdx

  if ( !*((_DWORD *)a1 + 128) )
    sub_B6F820(a2, a1 + 63);
  v5 = *a1;
  v6 = *(void **)(*a1 + 32LL);
  if ( *(_QWORD *)(*a1 + 24LL) - (_QWORD)v6 <= 0xBu )
  {
    sub_CB6200(v5, " syncscope(\"", 12);
  }
  else
  {
    qmemcpy(v6, " syncscope(\"", 12);
    *(_QWORD *)(v5 + 32) += 12LL;
  }
  v7 = (_QWORD *)(a1[63] + 16LL * a3);
  sub_C92400(*v7, v7[1], *a1);
  v8 = *a1;
  v9 = *(_WORD **)(*a1 + 32LL);
  if ( *(_QWORD *)(*a1 + 24LL) - (_QWORD)v9 <= 1u )
    return sub_CB6200(v8, "\")", 2);
  *v9 = 10530;
  *(_QWORD *)(v8 + 32) += 2LL;
  return 10530;
}
