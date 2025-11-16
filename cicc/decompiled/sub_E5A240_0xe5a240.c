// Function: sub_E5A240
// Address: 0xe5a240
//
__int64 __fastcall sub_E5A240(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  __int64 v5; // rdi
  _BYTE *v6; // rax

  v3 = *(_QWORD *)(a1 + 304);
  v4 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0xCu )
  {
    v3 = sub_CB6200(v3, "\t.cv_func_id ", 0xDu);
  }
  else
  {
    qmemcpy(v4, "\t.cv_func_id ", 13);
    *(_QWORD *)(v3 + 32) += 13LL;
  }
  v5 = sub_CB59D0(v3, a2);
  v6 = *(_BYTE **)(v5 + 32);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 24) )
  {
    sub_CB5D20(v5, 10);
  }
  else
  {
    *(_QWORD *)(v5 + 32) = v6 + 1;
    *v6 = 10;
  }
  return sub_E978D0(a1, a2);
}
